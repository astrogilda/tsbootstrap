"""Generate a function-level test-impact map for the mutation gate (coverage contexts).

Run the test suite ONCE under coverage.py with per-test line contexts, then map each source function
(by AST line range) to the set of tests that execute any of its lines. The mutation gate uses this to
run each mutant against ONLY its covering tests -- fast, deterministic, and no timeout-masking from
the broad property suite. This is a SINGLE clean coverage run of the normal suite (one process), not a
mutation run, so it is fine to run locally.

Writes tools/mutation_test_impact.json: {"module.function": ["tests/...::Test::test", ...], ...}.

Run:  HYPOTHESIS_PROFILE=mutmut uv run python tools/gen_test_impact.py
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
OUT = REPO / "tools" / "mutation_test_impact.json"
# only_mutate scope (must match pyproject [tool.mutmut].only_mutate).
SCOPE = ("tsbootstrap/engines", "tsbootstrap/model")


def _run_coverage() -> None:
    data = REPO / ".coverage_impact"
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "pytest",
        "tests/unit/",
        "-m",
        "not slow",
        "--cov=src/tsbootstrap",
        "--cov-context=test",
        "-q",
        "-p",
        "no:cacheprovider",
        "-o",
        "addopts=",
        "--cov-report=",
        "--no-header",
    ]
    env_data = {"COVERAGE_FILE": str(data)}
    import os

    print("[impact] running suite under coverage (one clean pass)...", flush=True)
    subprocess.run(cmd, cwd=REPO, env={**os.environ, **env_data}, check=False)  # noqa: S603, S607


def _function_ranges(py: Path) -> dict[str, tuple[int, int]]:
    """Map each top-level and nested function name to its (start, end) line range."""
    tree = ast.parse(py.read_text(encoding="utf-8"))
    ranges: dict[str, tuple[int, int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            ranges[node.name] = (node.lineno, getattr(node, "end_lineno", node.lineno))
    return ranges


def main() -> int:
    import os

    os.environ.setdefault("COVERAGE_FILE", str(REPO / ".coverage_impact"))
    _run_coverage()

    import coverage

    cov = coverage.Coverage(data_file=str(REPO / ".coverage_impact"))
    cov.load()
    data = cov.get_data()

    impact: dict[str, list[str]] = {}
    for measured in data.measured_files():
        mp = Path(measured)
        rel = mp.relative_to(SRC) if SRC in mp.parents else None
        if rel is None or not str(rel).startswith(SCOPE):
            continue
        module = ".".join(rel.with_suffix("").parts)
        ranges = _function_ranges(mp)
        ctx_by_line = data.contexts_by_lineno(measured)  # {lineno: [context, ...]}
        for func, (start, end) in ranges.items():
            tests: set[str] = set()
            for line in range(start, end + 1):
                for ctx in ctx_by_line.get(line, ()):
                    nodeid = ctx.split("|", 1)[0]  # strip the "|run" phase suffix
                    if nodeid and "::" in nodeid:
                        tests.add(nodeid)
            if tests:
                impact[f"{module}.{func}"] = sorted(tests)

    OUT.write_text(json.dumps(impact, indent=1), encoding="utf-8")
    covered = sum(1 for v in impact.values() if v)
    print(f"[impact] wrote {OUT} : {covered} functions mapped", flush=True)
    # quick visibility on the scope's functions and their covering-test counts
    for k, v in sorted(impact.items()):
        if any(s in k for s in ("stability", "fit", "arima", "recursive", "var", "arma_scipy")):
            print(f"  {len(v):2d} tests -> {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
