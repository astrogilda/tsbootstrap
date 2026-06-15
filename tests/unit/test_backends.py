"""
Backend implementation tests: Validating time series model backends.

This module tests the backend implementations that power our bootstrap methods.
We validate statsmodels and statsforecast backends, ensuring they provide
consistent interfaces and behavior while leveraging their respective strengths.

The tests cover model fitting, prediction, parameter extraction, and adapter
functionality to ensure seamless backend switching and feature compatibility.
"""

import numpy as np
import pytest

from tsbootstrap.backends.adapter import BackendToStatsmodelsAdapter, fit_with_backend
from tsbootstrap.backends.factory import create_backend
from tsbootstrap.backends.statsforecast_backend import StatsForecastBackend
from tsbootstrap.backends.statsmodels_backend import StatsModelsBackend


class TestStatsModelsBackend:
    """Test StatsModels backend implementation."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data."""
        np.random.seed(42)
        return np.cumsum(np.random.randn(100))

    def test_ar_model_fitting(self, sample_data):
        """Test AR model fitting."""
        backend = StatsModelsBackend(model_type="AR", order=2)
        fitted = backend.fit(sample_data)

        assert hasattr(fitted, "params")
        assert hasattr(fitted, "fitted_values")
        assert hasattr(fitted, "residuals")

    def test_arima_model_fitting(self, sample_data):
        """Test ARIMA model fitting."""
        backend = StatsModelsBackend(model_type="ARIMA", order=(1, 1, 1))
        fitted = backend.fit(sample_data)

        # Test predictions
        predictions = fitted.predict(steps=5)
        assert len(predictions) == 5

    def test_var_model_fitting(self):
        """Test VAR model fitting with multivariate data."""
        np.random.seed(42)
        data = np.random.randn(100, 3)

        backend = StatsModelsBackend(model_type="VAR", order=2)
        fitted = backend.fit(data)

        assert fitted.params is not None
        # VAR models need last observations for prediction
        last_obs = data[-2:]  # Last 2 observations for order=2
        predictions = fitted.predict(steps=5, X=last_obs)
        assert predictions.shape == (5, 3)

    def test_invalid_model_type(self):
        """Test error handling for invalid model type."""
        with pytest.raises(ValueError, match="Model type 'INVALID' is not supported"):
            StatsModelsBackend(model_type="INVALID", order=1)


class TestStatsForecastBackend:
    """Test StatsForecast backend implementation."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data."""
        np.random.seed(42)
        return np.cumsum(np.random.randn(100))

    def test_ar_model_support(self, sample_data):
        """Test AR model support in StatsForecast."""
        backend = StatsForecastBackend(model_type="AR", order=2)
        fitted = backend.fit(sample_data)

        assert hasattr(fitted, "params")
        assert "ar" in fitted.params

    def test_arima_model_fitting(self, sample_data):
        """Test ARIMA model fitting."""
        backend = StatsForecastBackend(model_type="ARIMA", order=(2, 1, 1))
        fitted = backend.fit(sample_data)

        predictions = fitted.predict(steps=10)
        assert len(predictions) == 10

    def test_auto_arima(self, sample_data):
        """Test AutoARIMA functionality."""
        backend = StatsForecastBackend(model_type="AutoARIMA")
        fitted = backend.fit(sample_data)

        # Should have selected order automatically
        assert hasattr(fitted, "params")
        assert "order" in fitted.params

    def test_information_criteria(self, sample_data):
        """Test information criteria calculation."""
        backend = StatsForecastBackend(model_type="ARIMA", order=(1, 0, 1))
        fitted = backend.fit(sample_data)

        criteria = fitted.get_info_criteria()
        assert "aic" in criteria
        assert "bic" in criteria
        assert "hqic" in criteria

        # Test ordering: AIC < HQIC < BIC
        assert criteria["aic"] < criteria["hqic"]
        assert criteria["hqic"] < criteria["bic"]

    def test_rescaling_integration(self):
        """Test rescaling service integration."""
        # Data that needs rescaling
        data = np.random.randn(100) * 1000 + 5000

        backend = StatsForecastBackend(model_type="ARIMA", order=(1, 0, 1))
        fitted = backend.fit(data)

        # Predictions should be in original scale
        predictions = fitted.predict(steps=5)
        assert np.mean(predictions) > 4000  # Near 5000


class TestBackendAdapter:
    """Test backend adapter functionality."""

    def test_adapter_interface(self):
        """Test that adapter provides statsmodels-like interface."""
        np.random.seed(42)
        data = np.random.randn(100)

        # Create backend and adapter
        backend = StatsForecastBackend(model_type="ARIMA", order=(1, 0, 1))
        fitted_backend = backend.fit(data)
        adapter = BackendToStatsmodelsAdapter(fitted_backend, model_type="ARIMA")

        # Check statsmodels interface
        assert hasattr(adapter, "params")
        assert hasattr(adapter, "resid")
        assert hasattr(adapter, "fittedvalues")
        assert hasattr(adapter, "aic")
        assert hasattr(adapter, "bic")
        assert hasattr(adapter, "forecast")

    def test_forecast_method(self):
        """Test forecast method compatibility."""
        np.random.seed(42)
        data = np.random.randn(100)

        backend = StatsForecastBackend(model_type="ARIMA", order=(1, 0, 1))
        fitted_backend = backend.fit(data)
        adapter = BackendToStatsmodelsAdapter(fitted_backend, model_type="ARIMA")

        # Test forecast method
        forecast = adapter.forecast(steps=5)
        assert len(forecast) == 5


class TestBackendFactory:
    """Test backend factory pattern."""

    def test_backend_selection(self):
        """Test automatic backend selection."""
        # Should select statsmodels for VAR
        backend = create_backend(model_type="VAR", order=2)
        assert isinstance(backend, StatsModelsBackend)

        # Can force statsforecast for ARIMA
        backend = create_backend(model_type="ARIMA", order=(1, 0, 1), force_backend="statsforecast")
        assert isinstance(backend, StatsForecastBackend)

    def test_fit_with_backend(self):
        """Test fit_with_backend convenience function."""
        np.random.seed(42)
        data = np.random.randn(100)

        # Fit with automatic backend selection
        fitted = fit_with_backend(
            model_type="ARIMA", endog=data, order=(1, 0, 1), return_backend=False  # Get adapter
        )

        assert isinstance(fitted, BackendToStatsmodelsAdapter)
        assert hasattr(fitted, "forecast")


class TestBackendCompatibility:
    """Test compatibility between backends."""

    @pytest.mark.parametrize(
        "model_type,order",
        [
            ("AR", 2),
            ("ARIMA", (1, 0, 1)),
            ("ARIMA", (2, 1, 1)),
        ],
    )
    def test_consistent_predictions(self, model_type, order):
        """Test that backends produce similar predictions."""
        np.random.seed(42)
        data = np.cumsum(np.random.randn(100))

        # Fit with both backends
        sm_backend = StatsModelsBackend(model_type=model_type, order=order)
        sf_backend = StatsForecastBackend(model_type=model_type, order=order)

        sm_fitted = sm_backend.fit(data)
        sf_fitted = sf_backend.fit(data)

        # Compare fitted values (allowing for numerical differences)
        sm_fitted_values = sm_fitted.fitted_values
        sf_fitted_values = sf_fitted.fitted_values

        # Ensure same length
        min_len = min(len(sm_fitted_values), len(sf_fitted_values))
        sm_fitted_values = sm_fitted_values[-min_len:]
        sf_fitted_values = sf_fitted_values[-min_len:]

        # Check correlation is high (not exact match due to implementation differences)
        correlation = np.corrcoef(sm_fitted_values, sf_fitted_values)[0, 1]
        assert correlation > 0.95

    def test_parameter_consistency(self):
        """Test that parameters are consistently represented."""
        np.random.seed(42)
        data = np.random.randn(100)

        # Simple AR model
        sm_backend = StatsModelsBackend(model_type="AR", order=2)
        sf_backend = StatsForecastBackend(model_type="AR", order=2)

        sm_fitted = sm_backend.fit(data)
        sf_fitted = sf_backend.fit(data)

        # Both should have AR parameters
        assert "ar" in sm_fitted.params or "ar_coef" in sm_fitted.params
        assert "ar" in sf_fitted.params
