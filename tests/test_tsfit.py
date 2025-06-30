"""Tests for TSFit class."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from tsbootstrap.tsfit import TSFit


class TestTSFit:
    """Test suite for TSFit in the main test directory."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data."""
        np.random.seed(42)
        n = 100
        return {
            "univariate": np.random.randn(n).cumsum(),
            "multivariate": np.random.randn(n, 3).cumsum(axis=0),
        }

    def test_inheritance(self):
        """Test that TSFit implements sklearn interfaces."""
        assert issubclass(TSFit, BaseEstimator)
        assert issubclass(TSFit, RegressorMixin)

    def test_services_composition(self):
        """Test that TSFit uses service composition."""
        tsfit = TSFit(order=2, model_type="ar")

        # Check that services are initialized
        assert hasattr(tsfit, "_validation_service")
        assert hasattr(tsfit, "_prediction_service")
        assert hasattr(tsfit, "_scoring_service")
        assert hasattr(tsfit, "_helper_service")

        # Check that services are not None
        assert tsfit._validation_service is not None
        assert tsfit._prediction_service is not None
        assert tsfit._scoring_service is not None
        assert tsfit._helper_service is not None

    @pytest.mark.parametrize(
        "model_type,order",
        [
            ("ar", 2),
            ("arima", (1, 1, 1)),
            ("sarima", (1, 1, 1)),
            ("var", 2),
            ("arch", 1),
        ],
    )
    def test_model_types(self, sample_data, model_type, order):
        """Test different model types."""
        kwargs = {}
        if model_type == "sarima":
            kwargs["seasonal_order"] = (1, 0, 1, 12)

        tsfit = TSFit(order=order, model_type=model_type, **kwargs)

        # Use appropriate data
        data = sample_data["multivariate"] if model_type == "var" else sample_data["univariate"]

        # Fit and predict
        tsfit.fit(data)

        # VAR models need X for prediction
        predictions = tsfit.predict(X=data[-2:]) if model_type == "var" else tsfit.predict()

        assert predictions is not None
        assert len(predictions) > 0

    def test_forecast_functionality(self, sample_data):
        """Test that forecast method works."""
        tsfit = TSFit(order=(1, 1, 1), model_type="arima")
        tsfit.fit(sample_data["univariate"])

        # Test forecast
        forecast = tsfit.forecast(steps=10)
        assert len(forecast) == 10

    def test_information_criteria(self, sample_data):
        """Test information criteria methods."""
        tsfit = TSFit(order=2, model_type="ar")
        tsfit.fit(sample_data["univariate"])

        # Test all criteria
        for criterion in ["aic", "bic", "hqic"]:
            ic = tsfit.get_information_criterion(criterion)
            assert isinstance(ic, float)
            assert not np.isnan(ic)

    def test_residual_methods(self, sample_data):
        """Test residual extraction methods."""
        tsfit = TSFit(order=(1, 0, 1), model_type="arima")
        tsfit.fit(sample_data["univariate"])

        # Test basic residuals
        residuals = tsfit.get_residuals()
        assert residuals.shape[0] > 0

        # Test standardized residuals
        residuals_std = tsfit.get_residuals(standardize=True)
        assert residuals_std.shape == residuals.shape
        # Check that standardization worked
        assert abs(np.std(residuals_std) - 1.0) < 0.1

    def test_stationarity_check(self, sample_data):
        """Test stationarity checking functionality."""
        tsfit = TSFit(order=(1, 1, 1), model_type="arima")
        tsfit.fit(sample_data["univariate"])

        # Test ADF test
        is_stationary, p_value = tsfit.check_residual_stationarity(test="adf")
        assert isinstance(is_stationary, (bool, np.bool_))
        assert isinstance(p_value, float)

        # Test KPSS test
        is_stationary, p_value = tsfit.check_residual_stationarity(test="kpss")
        assert isinstance(is_stationary, (bool, np.bool_))
        assert isinstance(p_value, float)

    def test_summary_method(self, sample_data):
        """Test summary functionality."""
        tsfit = TSFit(order=2, model_type="ar")
        tsfit.fit(sample_data["univariate"])

        summary = tsfit.summary()
        assert summary is not None

    def test_sklearn_interface(self, sample_data):
        """Test sklearn-compatible interface."""
        tsfit = TSFit(order=2, model_type="ar")
        data = sample_data["univariate"]

        # Test fit
        fitted = tsfit.fit(data)
        assert fitted is tsfit  # Should return self

        # Test score (R²)
        score = tsfit.score(data)
        assert isinstance(score, float)
        assert -1 <= score <= 1

        # Test get_params / set_params
        params = tsfit.get_params()
        assert "order" in params
        assert "model_type" in params

        tsfit.set_params(order=3)
        assert tsfit.order == 3

    def test_error_handling(self):
        """Test error handling."""
        # Invalid model type
        with pytest.raises(ValueError):
            TSFit(order=1, model_type="invalid")

        # Invalid order for VAR
        with pytest.raises(TypeError):
            TSFit(order=(1, 2), model_type="var")

        # Seasonal order for non-SARIMA
        with pytest.raises(ValueError):
            TSFit(order=2, model_type="ar", seasonal_order=(1, 0, 1, 12))

    def test_var_model_specifics(self, sample_data):
        """Test VAR model specific functionality."""
        tsfit = TSFit(order=2, model_type="var")
        data = sample_data["multivariate"]

        tsfit.fit(data)

        # VAR needs last observations for prediction
        last_obs = data[-2:]
        predictions = tsfit.predict(X=last_obs)
        assert predictions.shape[1] == data.shape[1]

        # Test forecast with required X
        forecast = tsfit.forecast(steps=5, X=last_obs)
        assert forecast.shape[0] == 5
        assert forecast.shape[1] == data.shape[1]

    def test_arch_model_specifics(self, sample_data):
        """Test ARCH model specific functionality."""
        # Generate returns data suitable for ARCH
        np.random.seed(42)
        returns = np.random.randn(200) * 0.01

        tsfit = TSFit(order=1, model_type="arch")
        tsfit.fit(returns)

        # Test volatility forecast
        forecast = tsfit.forecast(steps=5)
        assert len(forecast) > 0
