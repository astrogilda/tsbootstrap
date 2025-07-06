"""
Tests for RescalingService functionality.

This module tests the RescalingService implementation, ensuring proper
detection of when rescaling is needed, correct scaling and unscaling
of data, and integration with backend systems. We verify numerical
stability improvements through comprehensive test cases.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal
from tsbootstrap.backends.statsforecast_backend import StatsForecastBackend
from tsbootstrap.backends.statsmodels_backend import StatsModelsBackend
from tsbootstrap.services.rescaling_service import RescalingService


class TestRescalingService:
    """Test the RescalingService for numerical stability."""

    def test_rescaling_detection(self):
        """Test detection of when rescaling is needed."""
        service = RescalingService()

        # Normal data - no rescaling needed
        normal_data = np.random.randn(100)
        needs_rescaling, factors = service.check_if_rescale_needed(normal_data)
        assert not needs_rescaling
        assert factors == {}

        # Large range data - rescaling needed
        large_range = np.linspace(0, 2000, 100)
        needs_rescaling, factors = service.check_if_rescale_needed(large_range)
        assert needs_rescaling
        assert "shift" in factors
        assert "scale" in factors

        # Very small values - rescaling needed
        tiny_values = np.random.randn(100) * 1e-7
        needs_rescaling, factors = service.check_if_rescale_needed(tiny_values)
        assert needs_rescaling

        # Very large values - rescaling needed
        huge_values = np.random.randn(100) * 1e7
        needs_rescaling, factors = service.check_if_rescale_needed(huge_values)
        assert needs_rescaling

    def test_rescaling_reversibility(self):
        """Test that rescaling is perfectly reversible."""
        service = RescalingService()

        # Test various data patterns
        test_data = [
            np.random.randn(100) * 1000 + 5000,  # Large scale and shift
            np.random.randn(100) * 0.001,  # Small scale
            np.linspace(-1000, 1000, 100),  # Large range
            np.ones(100) * 42,  # Constant (edge case)
        ]

        for original in test_data:
            _, factors = service.check_if_rescale_needed(original)

            if factors:
                # Forward transform
                rescaled = service.rescale_data(original, factors)

                # Reverse transform
                recovered = service.rescale_back_data(rescaled, factors)

                # Check recovery within numerical precision
                assert_allclose(original, recovered, rtol=1e-10)

    def test_residual_rescaling(self):
        """Test that residuals are rescaled correctly (scale only, no shift)."""
        service = RescalingService()

        # Create residuals with zero mean
        residuals = np.random.randn(100)
        residuals = residuals - np.mean(residuals)  # Ensure zero mean

        factors = {"shift": 100.0, "scale": 10.0}

        # Rescale residuals
        rescaled = service.rescale_residuals(residuals, factors)

        # Check that mean is still approximately zero
        assert np.abs(np.mean(rescaled)) < 1e-10

        # Check that scale was applied
        assert_allclose(rescaled, residuals * factors["scale"], rtol=1e-10)

    def test_parameter_rescaling(self):
        """Test parameter adjustment for rescaling."""
        service = RescalingService()

        params = {"ar": np.array([0.5, -0.3]), "ma": np.array([0.2]), "sigma2": 1.0, "d": 0}

        factors = {"shift": 10.0, "scale": 2.0}

        adjusted = service.rescale_parameters(params, factors)

        # AR and MA coefficients should not change
        assert_array_almost_equal(adjusted["ar"], params["ar"])
        assert_array_almost_equal(adjusted["ma"], params["ma"])

        # Variance should be scaled by scale^2
        assert adjusted["sigma2"] == params["sigma2"] * (factors["scale"] ** 2)

    def test_rescaling_in_backends(self):
        """Test that rescaling works correctly in both backends."""
        np.random.seed(42)

        # Create data that needs rescaling
        y = np.random.randn(100) * 1000 + 5000

        # Test StatsForecast backend
        sf_backend = StatsForecastBackend(model_type="ARIMA", order=(1, 0, 1))
        sf_fitted = sf_backend.fit(y)

        # Predictions should be in original scale
        sf_pred = sf_fitted.predict(steps=5)
        assert np.mean(sf_pred) > 4000  # Should be near 5000

        # Test StatsModels backend
        sm_backend = StatsModelsBackend(model_type="ARIMA", order=(1, 0, 1))
        sm_fitted = sm_backend.fit(y)

        # Predictions should be in original scale
        sm_pred = sm_fitted.predict(steps=5)
        assert np.mean(sm_pred) > 4000  # Should be near 5000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
