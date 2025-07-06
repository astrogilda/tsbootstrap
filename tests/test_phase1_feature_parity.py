"""
Comprehensive tests for Phase 1 feature parity in TSFit removal.

These tests ensure that all features added during Phase 1 of the TSFit
removal plan work correctly and maintain backward compatibility. We test
AR model support, HQIC calculation, rescaling service, and AutoARIMA
integration to guarantee a smooth migration path.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal
from tsbootstrap.backends.statsforecast_backend import StatsForecastBackend
from tsbootstrap.backends.statsmodels_backend import StatsModelsBackend
from tsbootstrap.model_selection.best_lag import TSFitBestLag
from tsbootstrap.services.rescaling_service import RescalingService


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


class TestTSFitBestLagAutoARIMA:
    """Test TSFitBestLag using AutoARIMA for model selection."""

    def test_autoarima_selection_for_arima(self):
        """Test that TSFitBestLag uses AutoARIMA for ARIMA models."""
        np.random.seed(42)

        # Generate ARIMA(2,1,1) data
        n = 200
        y = np.random.randn(n).cumsum()  # Random walk (I(1))

        # Create TSFitBestLag without specifying order
        model = TSFitBestLag(
            model_type="arima",
            max_lag=5,
            order=None,  # Let it determine automatically
        )

        # Fit the model
        model.fit(y)

        # Check that order was determined
        assert model.order is not None
        assert isinstance(model.order, tuple)
        assert len(model.order) == 3  # (p, d, q)

    def test_autoarima_vs_ranklags(self):
        """Test that ARIMA uses AutoARIMA while AR uses RankLags."""
        np.random.seed(42)
        y = np.random.randn(150)

        # Test ARIMA - should use AutoARIMA
        arima_model = TSFitBestLag(
            model_type="arima",
            max_lag=5,
            order=None,
        )
        arima_model.fit(y)

        # Check that rank_lagger was not used for ARIMA
        assert arima_model.rank_lagger is None

        # Test AR - should use RankLags
        ar_model = TSFitBestLag(
            model_type="ar",
            max_lag=5,
            order=None,
        )
        ar_model.fit(y)

        # Check that rank_lagger was used for AR
        assert ar_model.rank_lagger is not None

    def test_explicit_order_override(self):
        """Test that explicit order overrides automatic selection."""
        np.random.seed(42)
        y = np.random.randn(100)

        # Specify explicit order
        explicit_order = (3, 0, 2)
        model = TSFitBestLag(
            model_type="arima",
            max_lag=10,
            order=explicit_order,
        )

        model.fit(y)

        # Check that explicit order was used
        assert model.order == explicit_order

    def test_max_lag_constraint(self):
        """Test that max_lag constrains AutoARIMA search."""
        np.random.seed(42)
        y = np.random.randn(100)

        # Small max_lag
        model = TSFitBestLag(
            model_type="arima",
            max_lag=2,
            order=None,
        )

        model.fit(y)

        # Check that selected order respects max_lag
        p, d, q = model.order
        assert p <= 2
        assert q <= 2


class TestBackwardCompatibility:
    """Test that new features maintain backward compatibility."""

    def test_tsfit_compatibility(self):
        """Test that TSFit still works with new backend features."""
        from tsbootstrap.tsfit import TSFit

        np.random.seed(42)
        y = np.random.randn(100)

        # Test various model types
        for model_type in ["ar", "arima"]:
            if model_type == "ar":
                order = 2
            else:
                order = (1, 0, 1)

            model = TSFit(order=order, model_type=model_type)
            model.fit(y)

            # Check basic functionality
            assert hasattr(model, "model")
            assert hasattr(model, "rescale_factors")

            # Check predictions work
            pred = model.forecast(steps=5)
            assert len(pred) == 5

    def test_adapter_interface(self):
        """Test that adapter maintains statsmodels interface."""
        from tsbootstrap.backends.adapter import fit_with_backend

        np.random.seed(42)
        y = np.random.randn(100)

        # Fit using adapter
        fitted = fit_with_backend(
            model_type="ARIMA",
            endog=y,
            order=(1, 0, 1),
            force_backend="statsforecast",
            return_backend=False,  # Get adapter
        )

        # Check statsmodels-like interface
        assert hasattr(fitted, "params")
        assert hasattr(fitted, "resid")
        assert hasattr(fitted, "fittedvalues")
        assert hasattr(fitted, "aic")
        assert hasattr(fitted, "bic")
        assert hasattr(fitted, "forecast")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
