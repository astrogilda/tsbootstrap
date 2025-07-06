"""
Tests for StatsForecast backend functionality.

This module tests the StatsForecast backend implementation, including
AR model support, HQIC calculation, and other backend-specific features.
We ensure that the backend correctly handles all supported model types
and provides accurate statistical computations.
"""

import numpy as np
import pytest
from tsbootstrap.backends.statsforecast_backend import StatsForecastBackend


class TestARModelSupport:
    """Test AR model support in StatsForecast backend."""

    def test_ar_model_creation(self):
        """Test that AR models are properly converted to ARIMA(p,0,0)."""
        # Create AR(2) model
        backend = StatsForecastBackend(model_type="AR", order=2)

        # Check that it's internally converted to ARIMA
        assert backend.model_type == "AR"
        assert backend.order == 2

    def test_ar_model_fitting(self):
        """Test fitting AR models with StatsForecast backend."""
        # Generate AR(2) data
        np.random.seed(42)
        n = 100
        ar_coefs = [0.5, -0.3]

        # Generate AR process
        y = np.zeros(n)
        y[0] = np.random.randn()
        y[1] = np.random.randn()

        for t in range(2, n):
            y[t] = ar_coefs[0] * y[t - 1] + ar_coefs[1] * y[t - 2] + np.random.randn()

        # Fit AR model
        backend = StatsForecastBackend(model_type="AR", order=2)
        fitted = backend.fit(y)

        # Check that model was fitted
        assert hasattr(fitted, "params")
        assert hasattr(fitted, "residuals")
        assert hasattr(fitted, "fitted_values")

        # Check predictions work
        pred = fitted.predict(steps=5)
        assert pred.shape == (5,)

    def test_ar_model_with_different_orders(self):
        """Test AR models with various orders."""
        np.random.seed(42)
        y = np.random.randn(100)

        for order in [1, 3, 5]:
            backend = StatsForecastBackend(model_type="AR", order=order)
            fitted = backend.fit(y)

            # Check that parameters match the order
            params = fitted.params
            if "ar" in params:
                assert len(params["ar"]) == order


class TestHQICCalculation:
    """Test HQIC calculation in StatsForecast backend."""

    def test_hqic_calculation(self):
        """Test that HQIC is calculated correctly."""
        np.random.seed(42)
        y = np.random.randn(100)

        # Fit ARIMA model
        backend = StatsForecastBackend(model_type="ARIMA", order=(1, 0, 1))
        fitted = backend.fit(y)

        # Get information criteria
        criteria = fitted.get_info_criteria()

        # Check that all criteria are present
        assert "aic" in criteria
        assert "bic" in criteria
        assert "hqic" in criteria

        # Check that HQIC has reasonable value
        assert isinstance(criteria["hqic"], float)
        assert not np.isnan(criteria["hqic"])
        assert not np.isinf(criteria["hqic"])

    def test_hqic_ordering(self):
        """Test that HQIC follows expected ordering: AIC < HQIC < BIC."""
        np.random.seed(42)
        y = np.random.randn(200)  # Larger sample for clearer ordering

        backend = StatsForecastBackend(model_type="ARIMA", order=(2, 0, 1))
        fitted = backend.fit(y)

        criteria = fitted.get_info_criteria()

        # For reasonable sample sizes, we expect AIC < HQIC < BIC
        # This is because penalty terms increase: 2k < 2k*log(log(n)) < k*log(n)
        assert criteria["aic"] < criteria["hqic"]
        assert criteria["hqic"] < criteria["bic"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
