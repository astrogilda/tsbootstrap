"""Run the tsbootstrap mutation ratchet via the subprocess-per-mutant runner (Layer 3 driver).

In-process mutmut deadlocks our numba parallel kernels (see pyproject [tool.mutmut]); this driver
keeps mutmut ONLY for AST mutant generation and executes every mutant in an isolated subprocess via
mutation_ratchet_core.subprocess_runner. It is the reliable, repeatable mutation gate.

Pipeline:
  1. Ensure the mutants/ tree exists (regenerate with `mutmut run` if missing or --regen; the stats
     phase will crash in-process, which is expected and harmless -- generation completes first).
  2. Read every mutant name from `mutmut results --all`.
  3. Map each mutant to its covering test files by source module (MODULE_TESTS below).
  4. Execute all mutants concurrently, one fresh subprocess each.
  5. Report killed / survived / timeout and list survivors for triage against
     tests/mutation_equivalents.md.

Run (idle box recommended):
  PYTHONPATH=<path-to>/mutation-ratchet-core uv run python tools/run_mutation_gate.py [--regen] [--workers N] [--only MODULE]
"""

from __future__ import annotations

# This driver legitimately shells out to `uv`/`mutmut` (on PATH locally and on the remote box);
# the subprocess-with-partial-path / untrusted-input lints do not apply.
# ruff: noqa: S603, S607
import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from mutation_ratchet_core.differ import diff_survivors, gate
from mutation_ratchet_core.mutmut_adapter import collect_survivors
from mutation_ratchet_core.subprocess_runner import run_mutants

REPO = Path(__file__).resolve().parent.parent
MUTANTS_SRC = REPO / "mutants" / "src"
CACHE = REPO / ".mutmut-numba-cache"  # shared persistent numba cache across mutant subprocesses

# Covering test files per mutated source module. Generous on purpose (includes the property layer)
# so a kill is never missed for lack of the right test; refine with a coverage-context map later.
_PROP = ["tests/property/test_invariants.py", "tests/property/test_properties.py"]
MODULE_TESTS: dict[str, list[str]] = {
    "tsbootstrap.model.fit": [
        "tests/unit/test_recursive_ar.py",
        "tests/unit/test_recursive_arima.py",
        "tests/unit/test_recursive_var.py",
        "tests/unit/test_exog.py",
        *_PROP,
    ],
    "tsbootstrap.model.recursive": [
        "tests/unit/test_recursive_ar.py",
        "tests/unit/test_recursive_arima.py",
        "tests/unit/test_recursive_var.py",
        "tests/unit/test_exog.py",
        "tests/unit/test_batched_engine.py",
        *_PROP,
    ],
    "tsbootstrap.model.arima": [
        "tests/unit/test_recursive_arima.py",
        "tests/unit/test_exog.py",
        *_PROP,
    ],
    "tsbootstrap.model.stability": [
        "tests/unit/test_recursive_ar.py",
        "tests/unit/test_recursive_var.py",
        *_PROP,
    ],
    "tsbootstrap.engines.var": [
        "tests/unit/test_recursive_var.py",
        "tests/unit/test_batched_engine.py",
        "tests/unit/test_compiled.py",
        *_PROP,
    ],
    "tsbootstrap.engines.arma_scipy": [
        "tests/unit/test_recursive_ar.py",
        "tests/unit/test_recursive_arima.py",
        "tests/unit/test_batched_engine.py",
        *_PROP,
    ],
}

_MUTMUT_ENV = {"PYTHONPATH": "tools/mutmut_sitecustomize", "HYPOTHESIS_PROFILE": "mutmut"}


def _module_of(mutant_name: str) -> str:
    """tsbootstrap.model.fit.x_select_ar_order__mutmut_3 -> tsbootstrap.model.fit."""
    return mutant_name.rpartition(".")[0]


def _ensure_mutants(regen: bool) -> None:
    if MUTANTS_SRC.exists() and not regen:
        return
    print(
        "[gen] generating mutants/ (mutmut run; the in-process stats crash after generation is expected)"
    )
    subprocess.run(
        ["uv", "run", "mutmut", "run"],  # noqa: S603, S607 - uv is on PATH (local dev + remote box)
        cwd=REPO,
        env={**os.environ, **_MUTMUT_ENV},
        capture_output=True,
        text=True,
        timeout=600,
    )
    if not MUTANTS_SRC.exists():
        sys.exit("mutants/ was not generated; inspect mutmut output")


_KEY_RE = re.compile(r"\['(x_.+?__mutmut_\d+)'\]")


def _mutant_names() -> list[str]:
    """Enumerate every generated mutant name directly from the trampolined source.

    This is independent of mutmut's result status: `mutmut results` lists nothing on a freshly
    generated (untested) store, so the gate must read the names from the `mutants/src` tree itself.
    Each mutated module populates its mutants dict with assignments like
    ``mutants_x_func__mutmut['x_func__mutmut_3'] = x_func__mutmut_3``; the bracketed key is the
    mutant function name and the file path gives the dotted module.
    """
    names: list[str] = []
    for py in (MUTANTS_SRC / "tsbootstrap").rglob("*.py"):
        module = ".".join(py.relative_to(MUTANTS_SRC).with_suffix("").parts)
        for line in py.read_text(encoding="utf-8").splitlines():
            if not line.startswith("mutants_"):
                continue
            m = _KEY_RE.search(line)
            if m:
                names.append(f"{module}.{m.group(1)}")
    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true", help="regenerate the mutants/ tree first")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--only", default=None, help="restrict to one source module (substring match)")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--out", default="mutation_outcomes.json", help="per-mutant outcomes JSON path")
    ap.add_argument(
        "--allowlist",
        default="tools/mutation_allowlist.txt",
        help="committed file of accepted (equivalent) survivor stable identities",
    )
    ap.add_argument(
        "--update-allowlist",
        action="store_true",
        help="write the current survivors' identities to --allowlist and do not gate",
    )
    args = ap.parse_args()

    _ensure_mutants(args.regen)
    CACHE.mkdir(exist_ok=True)
    names = _mutant_names()
    if not names:
        sys.exit(
            "FATAL: 0 mutants enumerated from mutants/src. Generation produced no mutants (a 0-mutant "
            "run is never a real pass). Check that `mutmut run` generated the trampolined source and "
            "that [tool.mutmut] mutate_only_covered_lines is false (the coverage pass is fragile)."
        )
    if args.only:
        names = [n for n in names if args.only in n]

    tests_for: dict[str, list[str]] = {}
    skipped = []
    for n in names:
        mod = _module_of(n)
        tests = MODULE_TESTS.get(mod)
        if tests is None:
            skipped.append(n)
            continue
        tests_for[n] = tests
    if skipped:
        print(
            f"[warn] {len(skipped)} mutants in unmapped modules skipped (add to MODULE_TESTS): "
            f"{sorted({_module_of(n) for n in skipped})}"
        )

    print(f"[run] {len(tests_for)} mutants, {args.workers} workers")
    outcomes = run_mutants(
        tests_for,
        repo_root=REPO,
        mutated_src=MUTANTS_SRC,
        cache_dir=CACHE,
        timeout=args.timeout,
        max_workers=args.workers,
    )
    hist = Counter(o.status for o in outcomes)
    survivors = sorted(o.name for o in outcomes if o.status == "survived")
    print(f"[done] {dict(hist)}")

    # (2) Full per-mutant outcomes JSON so survivors AND timeouts are itemized and diffable.
    Path(args.out).write_text(
        json.dumps({o.name: o.status for o in sorted(outcomes, key=lambda x: x.name)}, indent=1),
        encoding="utf-8",
    )
    print(f"[outcomes] wrote {args.out} ({len(outcomes)} mutants)")

    print(f"[survivors] {len(survivors)} (triage vs tests/mutation_equivalents.md):")
    for s in survivors:
        print(f"    {s}")

    # (3) New-survivor stable-identity gate. Map each survivor to a refactor-stable AST identity
    # (Layer-2) and diff against the committed allowlist of accepted (equivalent) identities. The
    # gate FAILS only on NEW survivors; catalogued equivalents do not trip it. Identity hashing uses
    # the ORIGINAL source (REPO/src), not the mutated tree.
    cur_ids = _survivor_identities(survivors)
    allowlist_path = Path(args.allowlist)
    if args.update_allowlist:
        allowlist_path.parent.mkdir(parents=True, exist_ok=True)
        allowlist_path.write_text(
            "# Accepted (equivalent) survivor stable identities -- the mutation new-survivor gate\n"
            "# fails only on identities NOT listed here. Regenerate with --update-allowlist after\n"
            "# triaging every survivor as a genuine equivalent (see tests/mutation_equivalents.md).\n"
            + "".join(f"{i}\n" for i in sorted(cur_ids)),
            encoding="utf-8",
        )
        print(f"[allowlist] wrote {len(cur_ids)} identities to {allowlist_path}")
        return 0

    allow = set()
    if allowlist_path.exists():
        allow = {
            ln.strip()
            for ln in allowlist_path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        }
    result = diff_survivors(cur_ids, allow)
    if result["new"]:
        print(
            f"[GATE FAIL] {len(result['new'])} NEW survivor identities (not in {allowlist_path}):"
        )
        for i in result["new"]:
            print(f"    {i}")
        return gate(result)
    print(f"[GATE PASS] all {len(cur_ids)} survivor identities are accepted equivalents")
    return 0


def _survivor_identities(survivors: list[str]) -> set[str]:
    """Map surviving mutant names to refactor-stable AST identities via Layer-2.

    Drives Layer-2 collect_survivors with our own survivor list (the subprocess runner is the source
    of truth, not `mutmut results`): synthesize the results text, capture each mutant's `mutmut show`
    diff, and let collect_survivors infer the node types + hash the enclosing statement of the
    ORIGINAL source. Survivors whose identity cannot be inferred (multi-line diffs) are returned by
    their mutant name so they still trip the gate (fail-safe, never silently dropped).
    """
    results_text = "".join(f"    {s}: survived\n" for s in survivors)
    show_texts: dict[str, str] = {}
    for s in survivors:
        show_texts[s] = subprocess.run(
            ["uv", "run", "mutmut", "show", s],  # noqa: S603, S607 - uv on PATH (local + remote box)
            cwd=REPO,
            env={**os.environ, **_MUTMUT_ENV},
            capture_output=True,
            text=True,
        ).stdout
    found = collect_survivors(REPO / "src", results_text=results_text, show_texts=show_texts)
    ids: set[str] = set()
    for sv in found:
        ids.add(sv.identity if sv.identity else f"UNINFERRED:{sv.mutant_name}")
    return ids


if __name__ == "__main__":
    raise SystemExit(main())
