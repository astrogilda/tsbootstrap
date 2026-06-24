"""Make ``mutmut`` 3.x safe for C-extension-heavy test suites (numpy / numba / scipy / ...).

Put this directory on ``PYTHONPATH`` when invoking the mutation ratchet:

    PYTHONPATH=tools/mutmut_sitecustomize HYPOTHESIS_PROFILE=mutmut uv run mutmut run

Python's ``site`` machinery imports ``sitecustomize`` automatically at interpreter startup, which
runs this BEFORE mutmut's ``run`` command, so the patch below is installed in time.

Why this is needed
------------------
mutmut 3.x runs pytest in-process (``pytest.main()``), repeatedly, in one process. To measure
which lines are covered (``mutate_only_covered_lines = true``) it calls
``mutmut.code_coverage.gather_coverage``, which:

  1. snapshots ``sys.modules`` BEFORE a coverage test pass,
  2. runs the pass (the tests import numpy, numba, scipy, statsmodels, ...),
  3. ``_unload_modules_not_in`` pops EVERY module imported during the pass so the source under
     test reloads fresh on the next pass.

Step 3 is correct for the project package (it must reload to pick up each mutant) but FATAL for
libraries that cannot be re-imported in the same process:

  * numpy: ``_multiarray_umath`` is a single-phase-init C extension ->
    ``ImportError: cannot load module more than once per process`` on the next pass.
  * numba: re-registers global type singletons on import ->
    ``KeyError: duplicate registration for ... PolynomialType`` on the next pass.

These libraries are never mutated (``only_mutate`` is scoped to the project source), so they must
stay resident across passes. We wrap ``_unload_modules_not_in`` to skip any submodule whose
top-level package is in the protected set, while still unloading the project package and the test
modules so mutation detection is unaffected.

This is an upstream mutmut-3.x interaction (2.x did not copy + reload this way), not specific to
this repository. The canonical, fleet-reusable version of this guard lives in the
``mutation-ratchet-core`` package (Layer 2); this file is the self-contained Layer-3 copy so the
repo's ratchet runs standalone (e.g. on an ephemeral CI box) without that package installed.
"""

from __future__ import annotations

import contextlib
import os

# Top-level packages that must never be unloaded mid-run: single-phase-init C extensions and
# libraries with global registration state. Extend via TSB_MUTMUT_PROTECT (comma-separated).
_PROTECTED = {
    "numpy",
    "scipy",
    "pandas",
    "numba",
    "llvmlite",
    "statsmodels",
    "patsy",
    "cffi",
    "_cffi_backend",
    "sklearn",
    "pyarrow",
}


def _install_guard() -> None:
    extra = {p.strip() for p in os.environ.get("TSB_MUTMUT_PROTECT", "").split(",") if p.strip()}

    # Prefer the canonical fleet implementation when the Layer-2 package is installed, so the
    # logic stays in one place; fall back to the inline copy below for standalone runs (e.g. an
    # ephemeral CI box that only `uv sync`s this repo).
    with contextlib.suppress(Exception):
        from mutation_ratchet_core.ceext_guard import install

        if install(extra_protected=extra):
            return

    import importlib
    import sys

    import mutmut.code_coverage as cc

    marker = "_mutation_ratchet_ceext_guard_installed"
    if getattr(cc._unload_modules_not_in, marker, False):
        return

    protected = _PROTECTED | extra

    def _safe_unload(modules: dict) -> None:
        for name in list(sys.modules):
            if name == "mutmut.code_coverage":
                continue
            if name.split(".")[0] in protected:
                continue
            if name not in modules:
                sys.modules.pop(name, None)
        importlib.invalidate_caches()

    setattr(_safe_unload, marker, True)
    cc._unload_modules_not_in = _safe_unload


# Only active under the mutation-ratchet profile so normal interpreter use is untouched. Never let
# the guard break the interpreter; a missing/renamed mutmut internal simply means the run proceeds
# unpatched (and fails loudly the same way it did before).
if os.environ.get("HYPOTHESIS_PROFILE") == "mutmut":
    with contextlib.suppress(Exception):
        _install_guard()
