"""
Comprehensive feature coverage tests for backend implementations.

This module tests all features supported by the backend system to ensure
complete functionality without relying on TSFit comparisons.
"""

from typing import Any, Dict

import numpy as np
import pytest
from tsbootstrap.backends.adapter import fit_with_backend
from tsbootstrap.backends.statsforecast_backend import StatsForecastBackend
from tsbootstrap.backends.statsmodels_backend import StatsModelsBackend


class TestBackendFeatureCoverage:
    """Test all features supported by backend implementations."""

    @pytest.fixture
    def sample_data(self) -> Dict[str, np.ndarray]:
        """Generate sample time series data for testing."""
        np.random.seed(42)
        n = 200
        return {
            "univariate": np.random.randn(n).cumsum(),
            "multivariate": np.random.randn(n, 3).cumsum(axis=0),
            "returns": np.random.randn(n) * 0.01,  # For ARCH models
            "seasonal": np.sin(np.arange(n) * 2 * np.pi / 12) + np.random.randn(n) * 0.1,
        }

    @pytest.mark.parametrize(
        "backend_cls,model_type,order,data_key",
        [
            (StatsModelsBackend, "AR", 2, "univariate"),
            (StatsModelsBackend, "ARIMA", (1, 1, 1), "univariate"),
            (StatsModelsBackend, "ARIMA", (2, 0, 1), "univariate"),
            (StatsModelsBackend, "VAR", 2, "multivariate"),
            (StatsModelsBackend, "ARCH", 1, "returns"),
            (StatsForecastBackend, "ARIMA", (1, 1, 1), "univariate"),
            (StatsForecastBackend, "AutoARIMA", None, "univariate"),
        ],
    )
    def test_model_fitting_and_prediction(
        self,
        sample_data: Dict[str, np.ndarray],
        backend_cls: type,
        model_type: str,
        order: Any,
        data_key: str,
    ) -> None:
        """Test model fitting and prediction for various model types."""
        data = sample_data[data_key]

        # Create backend instance
        backend = backend_cls(model_type=model_type, order=order)

        # Fit the model
        # All models including VAR now expect data in standard format
        fitted = backend.fit(data)

        assert fitted is not None

        # Test prediction
        if hasattr(fitted, "predict"):
            if model_type == "VAR":
                # VAR needs last observations for prediction
                last_obs = data[-order:]  # Get last 'order' observations
                predictions = fitted.predict(steps=5, X=last_obs)
            else:
                predictions = fitted.predict(steps=5)
            assert predictions is not None
            assert len(predictions) > 0

    def test_seasonal_models(self, sample_data: Dict[str, np.ndarray]) -> None:
        """Test seasonal ARIMA models."""
        data = sample_data["seasonal"]

        # Test StatsModels SARIMA
        backend = StatsModelsBackend(
            model_type="SARIMA", order=(1, 0, 1), seasonal_order=(1, 0, 1, 12)
        )
        fitted = backend.fit(data)

        assert fitted is not None
        assert hasattr(fitted, "aic")
        assert hasattr(fitted, "bic")

        # Test predictions
        forecast = fitted.predict(steps=12)
        assert len(forecast) == 12

    def test_information_criteria(self, sample_data: Dict[str, np.ndarray]) -> None:
        """Test information criteria calculation."""
        data = sample_data["univariate"]

        # Test with both backends
        for backend_cls in [StatsModelsBackend, StatsForecastBackend]:
            backend = backend_cls(model_type="ARIMA", order=(1, 0, 1))
            fitted = backend.fit(data)

            # Check information criteria
            assert hasattr(fitted, "aic")
            assert hasattr(fitted, "bic")
            assert hasattr(fitted, "hqic")

            # Values should be finite
            assert np.isfinite(fitted.aic)
            assert np.isfinite(fitted.bic)
            assert np.isfinite(fitted.hqic)

    def test_residuals_and_fitted_values(self, sample_data: Dict[str, np.ndarray]) -> None:
        """Test residuals and fitted values."""
        data = sample_data["univariate"]

        for backend_cls in [StatsModelsBackend, StatsForecastBackend]:
            backend = backend_cls(model_type="ARIMA", order=(1, 0, 1))
            fitted = backend.fit(data)

            # Check residuals
            assert hasattr(fitted, "resid")
            residuals = fitted.resid
            assert residuals is not None
            assert len(residuals) > 0

            # Check fitted values
            assert hasattr(fitted, "fitted_values")
            fitted_vals = fitted.fitted_values
            assert fitted_vals is not None
            assert len(fitted_vals) > 0

    def test_forecast_with_exogenous(self, sample_data: Dict[str, np.ndarray]) -> None:
        """Test forecasting with exogenous variables."""
        data = sample_data["univariate"]
        exog = np.random.randn(len(data), 2)

        # Test StatsModels with exogenous
        backend = StatsModelsBackend(model_type="ARIMA", order=(1, 0, 1))
        fitted = backend.fit(data, X=exog)  # Use X instead of exog

        # Forecast with future exogenous
        future_exog = np.random.randn(5, 2)
        forecast = fitted.predict(steps=5, X=future_exog)  # Use X instead of exog
        assert len(forecast) == 5

    def test_adapter_interface(self, sample_data: Dict[str, np.ndarray]) -> None:
        """Test the adapter interface for statsmodels compatibility."""
        data = sample_data["univariate"]

        # Use adapter
        fitted = fit_with_backend(
            model_type="ARIMA",
            endog=data,
            order=(1, 0, 1),
            force_backend="statsforecast",
            return_backend=False,  # Get adapter
        )

        # Check statsmodels-like interface on fitted model
        assert hasattr(fitted, "predict")
        assert hasattr(fitted, "forecast")
        assert hasattr(fitted, "params")
        assert hasattr(fitted, "resid")
        assert hasattr(fitted, "fittedvalues")
        assert hasattr(fitted, "aic")
        assert hasattr(fitted, "bic")

        # Test that methods work
        forecast = fitted.forecast(steps=5)
        assert len(forecast) == 5

        # Test params property
        params = fitted.params
        assert isinstance(params, (dict, np.ndarray))

        # Test residuals
        residuals = fitted.resid
        assert isinstance(residuals, np.ndarray)
        assert len(residuals) == len(data)

    def test_var_multivariate_functionality(self, sample_data: Dict[str, np.ndarray]) -> None:
        """Test VAR model specific functionality."""
        data = sample_data["multivariate"]

        backend = StatsModelsBackend(model_type="VAR", order=2)
        fitted = backend.fit(data)  # VAR expects (n_obs, n_vars)

        # Test VAR-specific functionality
        assert fitted is not None

        # Check IRF if available
        if hasattr(fitted, "irf"):
            irf = fitted.irf(10)
            assert irf is not None

        # Check forecast
        last_obs = data[-2:]  # Get last 2 observations for order=2
        forecast = fitted.predict(steps=5, X=last_obs)
        assert forecast.shape[0] == 5
        assert forecast.shape[1] == data.shape[1]

    def test_arch_volatility_modeling(self, sample_data: Dict[str, np.ndarray]) -> None:
        """Test ARCH model functionality."""
        returns = sample_data["returns"]

        backend = StatsModelsBackend(model_type="ARCH", order=1)
        fitted = backend.fit(returns)

        assert fitted is not None
        assert hasattr(fitted, "conditional_volatility")

        # Check conditional volatility
        vol = fitted.conditional_volatility
        assert vol is not None
        assert len(vol) > 0
        assert np.all(vol >= 0)  # Volatility should be non-negative

    def test_batch_operations(self, sample_data: Dict[str, np.ndarray]) -> None:
        """Test batch operations on multiple series."""
        # Generate multiple series
        n_series = 5
        n_obs = 100
        series_list = [np.random.randn(n_obs).cumsum() for _ in range(n_series)]

        # Test StatsForecast batch operations
        backend = StatsForecastBackend(model_type="ARIMA", order=(1, 0, 1))

        # Fit multiple series
        results = []
        for series in series_list:
            fitted = backend.fit(series)
            results.append(fitted)

        # All should succeed
        assert all(r is not None for r in results)
        assert all(hasattr(r, "aic") for r in results)

    def test_edge_cases(self) -> None:
        """Test edge cases and error handling."""
        # Very short series
        short_data = np.array([1, 2, 3, 4, 5])

        # Should handle gracefully
        backend = StatsModelsBackend(model_type="AR", order=1)
        fitted = backend.fit(short_data)
        assert fitted is not None

        # Empty data should raise error
        with pytest.raises((ValueError, IndexError)):
            backend.fit(np.array([]))

        # Wrong dimensions for VAR
        backend_var = StatsModelsBackend(model_type="VAR", order=1)
        with pytest.raises((ValueError, IndexError)):
            backend_var.fit(short_data)  # VAR needs multivariate data

    def test_model_summary_and_diagnostics(self, sample_data: Dict[str, np.ndarray]) -> None:
        """Test model summary and diagnostic information."""
        data = sample_data["univariate"]

        backend = StatsModelsBackend(model_type="ARIMA", order=(1, 0, 1))
        fitted = backend.fit(data)

        # Check if summary is available
        if hasattr(fitted, "summary"):
            summary = fitted.summary()
            assert summary is not None

        # Check parameters
        assert hasattr(fitted, "params")
        params = fitted.params
        assert params is not None
        assert len(params) > 0

    @pytest.mark.parametrize("sample_size", [50, 100, 500, 1000])
    def test_different_sample_sizes(self, sample_size: int) -> None:
        """Test backends with different sample sizes."""
        np.random.seed(42)
        data = np.random.randn(sample_size).cumsum()

        # Test both backends
        for backend_cls in [StatsModelsBackend, StatsForecastBackend]:
            backend = backend_cls(model_type="ARIMA", order=(1, 0, 1))
            fitted = backend.fit(data)

            assert fitted is not None
            assert hasattr(fitted, "aic")

            # Larger samples should generally have better fits
            if sample_size > 100:
                assert fitted.resid is not None
                assert len(fitted.resid) > 0

    def test_statsforecast_auto_models(self, sample_data: Dict[str, np.ndarray]) -> None:
        """Test StatsForecast AutoARIMA functionality."""
        data = sample_data["univariate"]

        # Test AutoARIMA
        backend = StatsForecastBackend(model_type="AutoARIMA")
        fitted = backend.fit(data)

        assert fitted is not None
        assert hasattr(fitted, "aic")
        assert hasattr(fitted, "bic")

        # Should select order automatically
        assert hasattr(fitted, "model")

        # Test predictions
        forecast = fitted.predict(steps=10)
        assert len(forecast) == 10

    def test_rescaling_service_integration(self) -> None:
        """Test that rescaling service works with backends."""
        # Create data that needs rescaling
        large_scale_data = np.random.randn(100) * 1000 + 5000

        # Both backends should handle this gracefully
        for backend_cls in [StatsModelsBackend, StatsForecastBackend]:
            backend = backend_cls(model_type="ARIMA", order=(1, 0, 1))
            fitted = backend.fit(large_scale_data)

            assert fitted is not None

            # Predictions should be in original scale
            forecast = fitted.predict(steps=5)
            assert np.mean(forecast) > 4000  # Should be near 5000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
