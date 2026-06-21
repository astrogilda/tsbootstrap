"""Model fitting and recursive regeneration for model-based bootstraps.

Fitting uses statsmodels as the correctness reference (imported lazily, so the
core package does not hard-require it). The recursive engines regenerate series
from fitted coefficients and resampled, centered innovations.
"""

# Importing this submodule registers the recursive executors and preparers.
from tsbootstrap.model import recursive  # noqa: E402,F401
