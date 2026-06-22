#!/usr/bin/env python3
"""Per-file coverage threshold gate.

Reads a ``coverage.json`` file produced by::

    uv run pytest tests/ --cov=src/tsbootstrap --cov-report=xml
    uv run coverage json -o coverage.json

and fails (exit 1) if any non-excluded file's statement coverage is below the
threshold. Unlike pytest-cov's ``--fail-under`` (which only checks the aggregate
across the whole package), this script enforces the requirement independently
for every source file, so a single well-covered module cannot mask an
undertested one.

Usage::

    python scripts/coverage_per_file.py coverage.json
    python scripts/coverage_per_file.py coverage.json --threshold 75
    python scripts/coverage_per_file.py coverage.json --exclude src/tsbootstrap/api.py

Exit codes:
    0  All non-excluded files meet the threshold.
    1  One or more files are below the threshold.
    2  Input file missing or malformed.

Justified exclusions
--------------------
Files passed to ``--exclude`` in CI must appear in the table below with a
one-line rationale. Silent exclusions (not listed here) are not permitted:
every excluded file must be documented here AND named in the CI step.

First-principles policy: only genuinely-hard-to-unit-test files belong here
(entry points, platform-specific branches). A file is never excluded merely to
dodge writing tests; the default remedy for a low number is a test, not an
exclusion entry.

    src/tsbootstrap/__init__.py
        Package entry point: re-export surface only (no logic). The module is
        already omitted from the interrogate gate for the same reason. Its
        statements are import-and-assign lines whose coverage tracks whatever
        importer happens to run first; gating it adds noise, not signal.

Add new entries in the format::

    src/<path>: <one-line rationale>
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path
from typing import Any


def _load_coverage(path: Path) -> dict[str, Any]:
    """Load and minimally validate coverage.json.

    Parameters
    ----------
    path : Path
        Path to the coverage JSON file.

    Returns
    -------
    dict[str, Any]
        Parsed JSON dictionary.

    Raises
    ------
    SystemExit
        On missing file or JSON parse error (exit code 2).
    """
    if not path.exists():
        print(f"ERROR: coverage file not found: {path}", file=sys.stderr)
        sys.exit(2)
    try:
        with path.open() as fh:
            data: dict[str, Any] = json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"ERROR: malformed JSON in {path}: {exc}", file=sys.stderr)
        sys.exit(2)
    if "files" not in data:
        print(f"ERROR: 'files' key missing in {path}", file=sys.stderr)
        sys.exit(2)
    return data


def _is_excluded(filepath: str, patterns: list[str]) -> bool:
    """Return True if filepath matches any exclusion pattern.

    Patterns are matched against the bare filepath as it appears in
    coverage.json (typically a path relative to the project root). Both
    fnmatch glob patterns and a plain suffix match are supported so an
    exclusion entry can be written as either ``src/tsbootstrap/x.py`` or a
    glob like ``src/tsbootstrap/**/x.py``.

    Parameters
    ----------
    filepath : str
        The file path string from coverage.json.
    patterns : list[str]
        List of exclusion patterns supplied via ``--exclude``.

    Returns
    -------
    bool
        True if any pattern matches.
    """
    return any(fnmatch.fnmatch(filepath, pat) or filepath.endswith(pat) for pat in patterns)


def _run(
    data: dict[str, Any],
    threshold: float,
    excludes: list[str],
) -> int:
    """Evaluate per-file coverage and print a results table.

    Parameters
    ----------
    data : dict[str, Any]
        Parsed coverage.json dictionary.
    threshold : float
        Minimum acceptable coverage percentage (0-100).
    excludes : list[str]
        File path patterns to skip.

    Returns
    -------
    int
        Exit code: 0 if all files pass, 1 if any fail.
    """
    files: dict[str, Any] = data["files"]

    rows: list[tuple[str, float, int, bool, bool]] = []
    # (filepath, pct, uncovered_count, excluded, passing)

    for filepath, file_data in sorted(files.items()):
        summary = file_data.get("summary", {})
        pct: float = summary.get("percent_covered", 0.0)
        missing: list[int] = file_data.get("missing_lines", [])
        excluded = _is_excluded(filepath, excludes)
        passing = excluded or pct >= threshold
        rows.append((filepath, pct, len(missing), excluded, passing))

    max_path = max((len(r[0]) for r in rows), default=4)
    col_path = max(max_path, 4)

    header = f"{'STATUS':<6}  {'COVERAGE':>8}  {'UNCOV':>5}  {'FILE':<{col_path}}"
    separator = "-" * len(header)

    print(f"\nPer-file coverage gate  (threshold: {threshold:.1f}%)\n")
    print(header)
    print(separator)

    failing: list[str] = []
    for filepath, pct, uncov, excluded, passing in rows:
        if excluded:
            status = "SKIP"
        elif passing:
            status = "PASS"
        else:
            status = "FAIL"
            failing.append(filepath)
        excl_tag = "  [excluded]" if excluded else ""
        print(f"{status:<6}  {pct:>7.1f}%  {uncov:>5}  {filepath:<{col_path}}{excl_tag}")

    print(separator)

    total = len(rows)
    skipped = sum(1 for _, _, _, exc, _ in rows if exc)
    checked = total - skipped
    passed = sum(1 for _, _, _, exc, ok in rows if not exc and ok)
    n_failing = len(failing)

    print(f"\n{checked} files checked ({skipped} skipped): {passed} PASS, {n_failing} FAIL\n")

    if failing:
        print(f"Files below {threshold:.1f}%:")
        for fp in failing:
            print(f"  {fp}")
        print()
        return 1

    return 0


def main() -> int:
    """CLI entry point.

    Returns
    -------
    int
        Exit code (0 = all pass, 1 = failures, 2 = input error).
    """
    parser = argparse.ArgumentParser(
        description="Per-file coverage threshold gate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "coverage_json",
        metavar="COVERAGE_JSON",
        type=Path,
        help="Path to coverage.json produced by `coverage json -o coverage.json`.",
    )
    parser.add_argument(
        "--threshold",
        metavar="N",
        type=float,
        default=80.0,
        help="Minimum coverage percentage per file (default: 80).",
    )
    parser.add_argument(
        "--exclude",
        metavar="PATTERN",
        action="append",
        default=[],
        dest="excludes",
        help=(
            "Exclude files matching PATTERN from the gate. Repeatable. Every "
            "exclusion must be justified in the docstring table inside this script."
        ),
    )

    args = parser.parse_args()

    coverage_path = args.coverage_json.resolve()
    if not coverage_path.is_relative_to(Path.cwd().resolve()):
        print(
            f"ERROR: refusing to read a coverage file outside the project root: {coverage_path}",
            file=sys.stderr,
        )
        return 2
    data = _load_coverage(coverage_path)
    return _run(data, args.threshold, args.excludes)


if __name__ == "__main__":
    sys.exit(main())
