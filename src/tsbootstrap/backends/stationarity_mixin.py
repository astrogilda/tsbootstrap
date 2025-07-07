"""
Stationarity testing: The statistical detective that validates our assumptions.

When we build time series models, we make critical assumptions about the data's
statistical properties. Chief among these is stationarity—the assumption that
the statistical properties don't change over time. This mixin represents our
systematic approach to validating that assumption across all backends.

We've designed this as a mixin to avoid code duplication between backends while
maintaining flexibility. Each backend generates residuals differently, but they
all need the same stationarity tests. By extracting this functionality into a
mixin, we ensure consistent testing logic while allowing backends to focus on
their core responsibilities.

The implementation supports both major stationarity tests:
- ADF (Augmented Dickey-Fuller): Tests for unit roots (non-stationarity)
- KPSS: Tests the null hypothesis of stationarity

These complementary tests help us avoid false conclusions. When ADF says
"stationary" and KPSS agrees, we have strong evidence. When they disagree,
we know to investigate further. This defensive approach has caught many
subtle modeling issues in production.
"""

from typing import Any, Dict

import numpy as np


class StationarityMixin:
    """Mixin class providing stationarity testing functionality.

    This mixin provides check_stationarity method implementation that can be
    shared between different backend implementations. It requires the backend
    to have a 'residuals' property.
    """

    def check_stationarity(
        self,
        test: str = "adf",
        significance: float = 0.05,
    ) -> Dict[str, Any]:
        """Check stationarity of residuals.

        Parameters
        ----------
        test : str, default="adf"
            Test to use ('adf' for Augmented Dickey-Fuller, 'kpss' for KPSS)
        significance : float, default=0.05
            Significance level for the test

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'statistic': float test statistic
            - 'p_value': float p-value from the statistical test
            - 'is_stationary': bool indicating whether residuals are stationary
            - 'critical_values': dict of critical values (if available)
        """
        # Lazy import to handle optional dependency
        from statsmodels.tsa.stattools import adfuller, kpss

        # Get residuals for testing - backend must have residuals property
        residuals = self.residuals  # type: ignore

        # Handle multiple series or VAR by testing the first series
        if residuals.ndim > 1:
            residuals = residuals[0]

        # Remove NaN values
        residuals = residuals[~np.isnan(residuals)]

        if len(residuals) < 10:
            # Not enough data for reliable test
            return {
                "statistic": np.nan,
                "p_value": 1.0,
                "is_stationary": False,
                "critical_values": {},
            }

        if test.lower() == "adf":
            # Augmented Dickey-Fuller test
            # Null hypothesis: unit root exists (non-stationary)
            result = adfuller(residuals, autolag="AIC")
            statistic = result[0]
            p_value = result[1]
            critical_values = result[4]
            is_stationary = p_value < significance
        elif test.lower() == "kpss":
            # KPSS test
            # Null hypothesis: series is stationary
            result = kpss(residuals, regression="c", nlags="auto")
            statistic = result[0]
            p_value = result[1]
            critical_values = result[3]
            is_stationary = p_value > significance
        else:
            raise ValueError(f"Unknown test type: {test}. Use 'adf' or 'kpss'.")

        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "is_stationary": bool(is_stationary),
            "critical_values": critical_values,
        }
