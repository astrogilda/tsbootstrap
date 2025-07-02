"""Mixin for stationarity testing in backends.

This module provides a reusable mixin for stationarity testing that can be
shared across different backend implementations.
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
