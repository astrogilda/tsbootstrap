"""Pytest and Hypothesis configuration for the test suite."""

from __future__ import annotations

import contextlib
import importlib.util
import os
import tempfile
import warnings

# Silence pkg_resources deprecation noise from an upstream dependency chain (fs).
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
warnings.filterwarnings(
    "ignore", message="pkg_resources is deprecated", category=DeprecationWarning
)
with contextlib.suppress(ImportError):
    import fs  # noqa: F401  (trigger + filter the warning before downstream imports re-raise it)

import pytest
from hypothesis import HealthCheck, settings

# Hypothesis profiles, select with the HYPOTHESIS_PROFILE env var (default "dev"):
#   dev       fast local feedback (few examples)
#   ci        more examples, slow-health-check relaxed
#   thorough  nightly deep search (1000 examples)
#   mutmut    deterministic (derandomized) profile for mutation testing
#   symbolic  CrossHair concolic backend (symbolic execution; best on pure-Python logic)
settings.register_profile("dev", max_examples=25, deadline=None)
settings.register_profile(
    "ci", max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.register_profile(
    "thorough", max_examples=1000, deadline=None, suppress_health_check=list(HealthCheck)
)
# Deterministic profile for mutation testing: derandomize so the property tests run a FIXED
# example set on every mutant, giving a reproducible clean baseline and reproducible mutant
# detection (a stochastic property failure would otherwise break mutmut's clean-baseline gate).
settings.register_profile(
    "mutmut",
    max_examples=40,
    deadline=None,
    derandomize=True,
    suppress_health_check=list(HealthCheck),
)
# The symbolic (CrossHair concolic) profile is optional: it needs the
# hypothesis-crosshair backend, a dev-only extra. Registering it (or selecting it)
# without the backend installed only fails when a test actually *runs* under it, so
# we register it solely when the backend is importable; selecting symbolic without it
# then falls back to dev (below) rather than erroring mid-run. (test_symbolic.py guards
# itself with importorskip for the same reason.)
_PROFILES = ["dev", "ci", "thorough", "mutmut"]
if importlib.util.find_spec("hypothesis_crosshair") is not None:
    settings.register_profile(
        "symbolic",
        backend="crosshair",
        max_examples=50,
        deadline=None,
        suppress_health_check=list(HealthCheck),
    )
    _PROFILES.append("symbolic")

_profile = os.environ.get("HYPOTHESIS_PROFILE", "dev")
if _profile not in _PROFILES:
    warnings.warn(
        f"unknown HYPOTHESIS_PROFILE={_profile!r}; valid profiles are {_PROFILES}. "
        "Falling back to 'dev'.",
        stacklevel=1,
    )
    _profile = "dev"
settings.load_profile(_profile)


# --- Mutation-testing-only fixtures (active solely under HYPOTHESIS_PROFILE=mutmut) ----------
#
# These two fixtures exist purely to make `mutmut run` (the mutation-score ratchet) correct and
# fast. They are gated on HYPOTHESIS_PROFILE=mutmut, which is exactly how the ratchet is invoked
# (see [tool.mutmut] in pyproject.toml), so they are completely inert during normal test runs.

_UNDER_MUTMUT = os.environ.get("HYPOTHESIS_PROFILE") == "mutmut"


@pytest.fixture(scope="session", autouse=True)
def _mutmut_ephemeral_numba_cache():
    """Give each mutation-testing session a fresh, throwaway numba on-disk cache.

    The VAR recurrence kernel is compiled with ``@numba.njit(..., cache=True)`` (see
    ``engines/var.py``), so its machine code is cached to disk keyed by the *unmutated* source.
    Under mutmut that cache masks source mutations to the kernel: the mutant imports, numba finds a
    matching cache entry, and the original (unmutated) compiled code runs, so the mutation never
    executes and the mutant survives spuriously. Pointing ``NUMBA_CACHE_DIR`` at a per-session
    tempdir forces a fresh compile of the mutated source on every mutant, at native JIT speed.
    This replaces the old ``NUMBA_DISABLE_JIT=1`` workaround (interpreted execution, ~28 min/run).
    """
    if not _UNDER_MUTMUT:
        yield
        return
    with tempfile.TemporaryDirectory(prefix="tsbootstrap-mutmut-numba-") as cache_dir:
        prev = os.environ.get("NUMBA_CACHE_DIR")
        os.environ["NUMBA_CACHE_DIR"] = cache_dir
        try:
            yield
        finally:
            if prev is None:
                os.environ.pop("NUMBA_CACHE_DIR", None)
            else:
                os.environ["NUMBA_CACHE_DIR"] = prev


@pytest.fixture(autouse=True)
def _mutmut_clamp_solver_maxiter(monkeypatch):
    """Clamp the statsmodels MLE solver to few iterations during mutation testing.

    ARIMA/ARMA are the only iteratively-fit models here (AR/VAR/sieve are direct OLS); they go
    through ``statsmodels`` ``MLEModel.fit`` (default ``maxiter=50``). A mutant that degrades the
    objective or the recursion can make that optimizer crawl, turning a would-be KILL into a long
    spin. Capping ``maxiter`` low makes such mutants fail fast (or hit the pytest ``--timeout``)
    instead of stalling the run, without changing which mutants are detectable for the OLS models.
    """
    if not _UNDER_MUTMUT:
        return
    try:
        from statsmodels.tsa.statespace.mlemodel import MLEModel
    except ImportError:
        return

    _orig_fit = MLEModel.fit

    def _clamped_fit(self, *args, **kwargs):
        kwargs.setdefault("maxiter", 5)
        return _orig_fit(self, *args, **kwargs)

    monkeypatch.setattr(MLEModel, "fit", _clamped_fit)
