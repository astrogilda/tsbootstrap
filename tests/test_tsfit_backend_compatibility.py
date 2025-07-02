"""Tests for TSFitBackendWrapper compatibility with TSFit."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from tsbootstrap.backends.tsfit_wrapper import TSFitBackendWrapper
from tsbootstrap.tsfit.base import TSFit


class TestTSFitBackendCompatibility:
    """Test that TSFitBackendWrapper provides full TSFit compatibility."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data."""
        np.random.seed(42)
        return {
            "X": np.random.randn(100),
            "y": np.random.randn(100, 2),
            "X_test": np.random.randn(20),
            "y_test": np.random.randn(20, 2),
        }

    def test_initialization_compatibility(self):
        """Test that TSFitBackendWrapper accepts same parameters as TSFit."""
        # Test AR model
        wrapper = TSFitBackendWrapper(order=2, model_type="ar")
        tsfit = TSFit(order=2, model_type="ar")

        assert wrapper.order == tsfit.order
        assert wrapper.model_type == tsfit.model_type
        assert wrapper.seasonal_order == tsfit.seasonal_order

        # Test ARIMA model
        wrapper = TSFitBackendWrapper(order=(1, 1, 1), model_type="arima")
        tsfit = TSFit(order=(1, 1, 1), model_type="arima")

        assert wrapper.order == tsfit.order
        assert wrapper.model_type == tsfit.model_type

        # Test SARIMA model
        wrapper = TSFitBackendWrapper(
            order=(1, 1, 1), model_type="sarima", seasonal_order=(1, 1, 1, 12)
        )
        tsfit = TSFit(order=(1, 1, 1), model_type="sarima", seasonal_order=(1, 1, 1, 12))

        assert wrapper.seasonal_order == tsfit.seasonal_order

    def test_fit_method_compatibility(self, sample_data):
        """Test that fit method works the same way."""
        wrapper = TSFitBackendWrapper(order=2, model_type="ar")

        # Test fit returns self
        result = wrapper.fit(sample_data["X"], sample_data["y"])
        assert result is wrapper

        # Test that model is fitted
        assert wrapper.model is not None

        # Test that data is stored
        assert wrapper._X is not None
        assert wrapper._y is not None
        np.testing.assert_array_equal(wrapper._X, sample_data["X"])
        np.testing.assert_array_equal(wrapper._y, sample_data["y"])

    def test_predict_method_compatibility(self, sample_data):
        """Test that predict method works the same way."""
        wrapper = TSFitBackendWrapper(order=2, model_type="ar")
        wrapper.fit(sample_data["X"], sample_data["y"])

        # Test prediction without exog
        predictions = wrapper.predict()
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) > 0

        # Test prediction with start/end
        predictions = wrapper.predict(start=10, end=20)
        assert isinstance(predictions, np.ndarray)

    def test_forecast_method_compatibility(self, sample_data):
        """Test that forecast method works the same way."""
        wrapper = TSFitBackendWrapper(order=2, model_type="ar")
        wrapper.fit(sample_data["X"])

        # Test forecast
        forecasts = wrapper.forecast(steps=5)
        assert isinstance(forecasts, np.ndarray)
        assert len(forecasts) == 5

    def test_score_method_compatibility(self, sample_data):
        """Test that score method works the same way."""
        wrapper = TSFitBackendWrapper(order=2, model_type="ar")
        wrapper.fit(sample_data["X"], sample_data["y"])

        # Test scoring with default metric
        score = wrapper.score(sample_data["X"], sample_data["y"])
        assert isinstance(score, float)

        # Test scoring with different metrics
        for metric in ["mse", "mae", "mape"]:
            score = wrapper.score(sample_data["X"], sample_data["y"], metric=metric)
            assert isinstance(score, float)

    def test_get_residuals_compatibility(self, sample_data):
        """Test that get_residuals works the same way."""
        wrapper = TSFitBackendWrapper(order=2, model_type="ar")
        wrapper.fit(sample_data["X"])

        residuals = wrapper.get_residuals()
        assert isinstance(residuals, np.ndarray)
        assert len(residuals) > 0

    def test_get_fitted_values_compatibility(self, sample_data):
        """Test that get_fitted_values works the same way."""
        wrapper = TSFitBackendWrapper(order=2, model_type="ar")
        wrapper.fit(sample_data["X"])

        fitted_values = wrapper.get_fitted_values()
        assert isinstance(fitted_values, np.ndarray)
        assert len(fitted_values) > 0

    def test_information_criteria_compatibility(self, sample_data):
        """Test that get_information_criterion works the same way."""
        wrapper = TSFitBackendWrapper(order=2, model_type="ar")
        wrapper.fit(sample_data["X"])

        # Test different criteria
        for criterion in ["aic", "bic", "hqic"]:
            ic_value = wrapper.get_information_criterion(criterion)
            assert isinstance(ic_value, float)

    def test_stationarity_check_compatibility(self, sample_data):
        """Test that check_residual_stationarity works the same way."""
        wrapper = TSFitBackendWrapper(order=2, model_type="ar")
        wrapper.fit(sample_data["X"])

        result = wrapper.check_residual_stationarity()
        assert isinstance(result, dict)
        assert "statistic" in result
        assert "pvalue" in result
        assert "is_stationary" in result

    def test_summary_compatibility(self, sample_data):
        """Test that summary method works."""
        wrapper = TSFitBackendWrapper(order=2, model_type="ar")
        wrapper.fit(sample_data["X"])

        summary = wrapper.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_repr_compatibility(self):
        """Test that string representation works."""
        wrapper = TSFitBackendWrapper(order=2, model_type="ar")
        repr_str = repr(wrapper)
        assert "TSFitBackendWrapper" in repr_str
        assert "model_type=ar" in repr_str
        assert "order=2" in repr_str

    def test_backend_fallback(self, sample_data):
        """Test that wrapper can fall back to statsmodels when needed."""
        # Test with use_backend=False
        wrapper = TSFitBackendWrapper(order=2, model_type="ar", use_backend=False)
        wrapper.fit(sample_data["X"])

        assert wrapper.model is not None

        # Test unsupported model fallback
        with patch("tsbootstrap.backends.adapter.fit_with_backend") as mock_fit:
            # First call raises exception, second succeeds
            mock_fit.side_effect = [
                Exception("Backend not supported"),
                Mock(resid=np.zeros(10), fittedvalues=np.zeros(10)),
            ]

            wrapper = TSFitBackendWrapper(order=2, model_type="ar", use_backend=True)
            wrapper.fit(sample_data["X"])

            # Should have been called twice (once failed, once with statsmodels)
            assert mock_fit.call_count == 2
            assert mock_fit.call_args_list[1][1]["force_backend"] == "statsmodels"

    def test_service_integration(self):
        """Test that wrapper properly uses TSFit services."""
        wrapper = TSFitBackendWrapper(order=2, model_type="ar")

        # Check services are initialized
        assert hasattr(wrapper, "_validation_service")
        assert hasattr(wrapper, "_prediction_service")
        assert hasattr(wrapper, "_scoring_service")
        assert hasattr(wrapper, "_helper_service")

    def test_additional_parameters(self):
        """Test that additional parameters are passed through."""
        wrapper = TSFitBackendWrapper(order=2, model_type="ar", trend="c", method="mle")

        assert wrapper.model_params == {"trend": "c", "method": "mle"}

    def test_scikit_base_tags(self):
        """Test that scikit-base tags are preserved."""
        wrapper = TSFitBackendWrapper(order=2, model_type="ar")
        tsfit = TSFit(order=2, model_type="ar")

        # Check that tags match
        assert wrapper._tags == tsfit._tags

    @pytest.mark.parametrize(
        "model_type,order",
        [
            ("ar", 2),
            ("arima", (1, 0, 1)),
            ("arima", (2, 1, 2)),
        ],
    )
    def test_different_models(self, model_type, order, sample_data):
        """Test wrapper with different model types."""
        wrapper = TSFitBackendWrapper(order=order, model_type=model_type)
        wrapper.fit(sample_data["X"])

        # Test basic functionality
        assert wrapper.model is not None
        residuals = wrapper.get_residuals()
        assert len(residuals) > 0

        predictions = wrapper.predict()
        assert len(predictions) > 0

    def test_error_handling(self):
        """Test proper error handling."""
        wrapper = TSFitBackendWrapper(order=2, model_type="ar")

        # Test methods before fitting
        with pytest.raises(ValueError, match="Model must be fitted"):
            wrapper.predict()

        with pytest.raises(ValueError, match="Model must be fitted"):
            wrapper.forecast()

        with pytest.raises(ValueError, match="Model must be fitted"):
            wrapper.get_residuals()

        with pytest.raises(ValueError, match="Model must be fitted"):
            wrapper.get_fitted_values()

        with pytest.raises(ValueError, match="Model must be fitted"):
            wrapper.score(np.zeros(10))

    def test_calculate_trend_terms_compatibility(self, sample_data):
        """Test _calculate_trend_terms method for compatibility."""
        wrapper = TSFitBackendWrapper(order=2, model_type="ar")
        wrapper.fit(sample_data["X"])

        # Test the method exists and returns appropriate shape
        trend_terms = wrapper._calculate_trend_terms(sample_data["X"])
        assert isinstance(trend_terms, np.ndarray)
        assert trend_terms.shape == sample_data["X"].shape
