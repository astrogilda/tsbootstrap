"""The uv version contract between pyproject.toml, the CI workflows and Dependabot.

``required-version`` in ``[tool.uv]`` is a floor (``>=X.Y.Z``), and every setup-uv step
installs exactly that floor through ``resolution-strategy: lowest``. Both halves matter:

* An exact ``==`` pin makes Dependabot refuse every uv update with
  ``tool_version_not_supported`` as soon as the uv it bundles moves past the pin, which is
  what failed the weekly uv updates from 2026-09-15 on.
* A floor without ``lowest`` lets setup-uv install the newest uv release on every run,
  which is the supply-chain exposure the pin exists to close.

A setup-uv step that names its own ``version`` or ``version-file`` would split the pin
into two sources that drift apart, so those are refused too.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import yaml

if sys.version_info >= (3, 11):
    import tomllib
else:  # pytest itself depends on tomli below 3.11
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"


def _setup_uv_steps() -> list[tuple[str, dict[str, Any]]]:
    steps: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job_name, job in (doc.get("jobs") or {}).items():
            for step in job.get("steps") or []:
                if str(step.get("uses", "")).startswith("astral-sh/setup-uv@"):
                    steps.append((f"{path.name}:{job_name}", step.get("with") or {}))
    return steps


def test_required_version_is_a_single_floor() -> None:
    with (ROOT / "pyproject.toml").open("rb") as f:
        required = tomllib.load(f)["tool"]["uv"]["required-version"]
    assert re.fullmatch(r">=\d+\.\d+\.\d+", required), (
        f"required-version must be a floor like '>=0.12.15', got {required!r}"
    )


def test_every_setup_uv_step_installs_exactly_the_floor() -> None:
    steps = _setup_uv_steps()
    assert steps, "found no setup-uv step under .github/workflows; the scan is broken"
    for where, inputs in steps:
        assert inputs.get("resolution-strategy") == "lowest", (
            f"{where}: setup-uv must set resolution-strategy: lowest"
        )
        assert "version" not in inputs, f"{where}: setup-uv must not name its own version"
        assert "version-file" not in inputs, f"{where}: setup-uv must not name a version-file"
