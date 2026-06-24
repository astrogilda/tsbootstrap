#!/usr/bin/env python3
"""Keep uv.lock's self-version in step with pyproject.

release-please bumps pyproject.toml (and the conf.py / CITATION.cff / server.json
files listed in release-please-config.json) on a release, but it does not touch
uv.lock. The lockfile carries the project's own version in its editable
self-package entry, so without this guard that entry drifts one release behind and
the repo ends up with two different version numbers.

Usage:
  scripts/check_lock_version.py          # fix uv.lock to match pyproject (exit 0)
  scripts/check_lock_version.py --check  # exit 1 if uv.lock diverges

--check runs in .pre-commit-config.yaml so drift cannot land on main.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import tomllib

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
UV_LOCK = ROOT / "uv.lock"


def _pyproject() -> dict[str, Any]:
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)


def _self_version_match(text: str, name: str) -> re.Match[str] | None:
    """Locate the editable self-package's version line in uv.lock.

    Matches the project's own ``[[package]]`` entry (``name`` then ``version``),
    not the ``{ name = ... }`` dependency references elsewhere in the lock.
    """
    return re.search(rf'(?m)^name = "{re.escape(name)}"\nversion = "([^"]+)"', text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check", action="store_true", help="Exit 1 if uv.lock diverges from pyproject."
    )
    args = ap.parse_args()

    if not UV_LOCK.exists():
        return 0

    project = _pyproject()["project"]
    version = str(project["version"])
    name = str(project["name"])

    text = UV_LOCK.read_text(encoding="utf-8")
    match = _self_version_match(text, name)
    if match is None:
        sys.stderr.write(f"could not find the {name} self-package entry in uv.lock\n")
        return 1
    current = match.group(1)
    if current == version:
        return 0

    if args.check:
        sys.stderr.write(
            f"uv.lock self-version {current!r} != pyproject {version!r}.\n"
            "Run `scripts/check_lock_version.py` (or `uv lock`) to fix.\n"
        )
        return 1

    UV_LOCK.write_text(text[: match.start(1)] + version + text[match.end(1) :], encoding="utf-8")
    print(f"Synced uv.lock self-version to {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
