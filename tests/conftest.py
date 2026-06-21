"""Pytest and Hypothesis configuration for the test suite."""

from __future__ import annotations

import contextlib
import os
import warnings

# Silence pkg_resources deprecation noise from an upstream dependency chain (fs).
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=DeprecationWarning)
with contextlib.suppress(ImportError):
    import fs  # noqa: F401  (trigger + filter the warning before downstream imports re-raise it)

from hypothesis import HealthCheck, settings

# Hypothesis profiles — select with the HYPOTHESIS_PROFILE env var (default "dev"):
#   dev       fast local feedback (few examples)
#   ci        more examples, slow-health-check relaxed
#   thorough  nightly deep search (1000 examples)
#   symbolic  CrossHair concolic backend (symbolic execution; best on pure-Python logic)
settings.register_profile("dev", max_examples=25, deadline=None)
settings.register_profile(
    "ci", max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.register_profile(
    "thorough", max_examples=1000, deadline=None, suppress_health_check=list(HealthCheck)
)
settings.register_profile(
    "symbolic",
    backend="crosshair",
    max_examples=50,
    deadline=None,
    suppress_health_check=list(HealthCheck),
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
