"""
Tests for time series fitting services.

This module provides comprehensive test coverage for the TSFit service
components that handle model validation, prediction, scoring, and various
helper utilities for time series analysis.
"""

import numpy as np
import pytest
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from tsbootstrap.services.tsfit_services import (
    TSFitHelperService,
    TSFitPredictionService,
    TSFitScoringService,
    TSFitValidationService,
)


class TestTSFitValidationService:
    """Test the validation service for time series models.

    The validation service ensures that model parameters and configurations
    are valid before they're used in fitting operations.
    """

    def test_validate_model_type_valid(self):
        """Test valid model type validation."""
        service = TSFitValidationService()

        # Validate each supported model type
        for model_type in ["ar", "arima", "sarima", "var", "arch"]:
            result = service.validate_model_type(model_type)
            assert result == model_type

    def test_validate_model_type_invalid(self):
        """Test invalid model type validation."""
        service = TSFitValidationService()

        # Ensure invalid model types are rejected
        with pytest.raises(ValueError) as exc_info:
            service.validate_model_type("invalid_model")
        assert "Expected one of" in str(exc_info.value)

    def test_validate_order_ar_integer(self):
        """Test AR order validation with integer."""
        service = TSFitValidationService()
        result = service.validate_order(2, "ar")
        assert result == 2

    def test_validate_order_ar_list_fails(self):
        """Test that AR models don't accept list-based orders."""
        service = TSFitValidationService()
        with pytest.raises(TypeError) as exc_info:
            service.validate_order([1, 3, 5], "ar")
        assert "must not be a tuple/list" in str(exc_info.value)

    def test_validate_order_arima_tuple(self):
        """Test ARIMA order validation."""
        service = TSFitValidationService()
        result = service.validate_order((1, 1, 1), "arima")
        assert result == (1, 1, 1)

    def test_validate_order_var_integer(self):
        """Test VAR order validation."""
        service = TSFitValidationService()
        result = service.validate_order(2, "var")
        assert result == 2

    def test_validate_order_invalid_var_tuple(self):
        """Test VAR with tuple should fail."""
        service = TSFitValidationService()
        with pytest.raises(TypeError) as exc_info:
            service.validate_order((1, 2), "var")
        assert "must be an integer" in str(exc_info.value)

    def test_validate_seasonal_order_sarima(self):
        """Test seasonal order validation for SARIMA."""
        service = TSFitValidationService()
        result = service.validate_seasonal_order((1, 0, 1, 12), "sarima")
        assert result == (1, 0, 1, 12)

    def test_validate_seasonal_order_non_sarima(self):
        """Test seasonal order for non-SARIMA models."""
        service = TSFitValidationService()
        with pytest.raises(ValueError) as exc_info:
            service.validate_seasonal_order((1, 0, 1, 12), "arima")
        assert "only valid for SARIMA" in str(exc_info.value)

    def test_validate_seasonal_order_invalid_period(self):
        """Test seasonal order with invalid period."""
        service = TSFitValidationService()
        with pytest.raises(ValueError) as exc_info:
            service.validate_seasonal_order((1, 0, 1, 1), "sarima")
        assert "must be at least 2" in str(exc_info.value)


class TestTSFitPredictionService:
    """Test prediction service functionality."""

    @pytest.fixture
    def sample_models(self):
        """Create sample models for testing."""
        np.random.seed(42)
        data = np.random.randn(100).cumsum()

        models = {}

        # AR model
        ar_model = AutoReg(data, lags=2, trend="c")
        models["ar"] = ar_model.fit()

        # ARIMA model
        arima_model = ARIMA(data, order=(1, 0, 1))
        models["arima"] = arima_model.fit()

        # VAR model (multivariate)
        data_mv = np.random.randn(100, 2).cumsum(axis=0)
        var_model = VAR(data_mv)
        models["var"] = var_model.fit(2)

        return models

    def test_predict_ar(self, sample_models):
        """Test AR model predictions."""
        service = TSFitPredictionService()

        predictions = service.predict(model=sample_models["ar"], model_type="ar", start=10, end=20)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape[1] == 1  # Should be 2D
        assert len(predictions) == 11  # end - start + 1

    def test_predict_var_requires_x(self, sample_models):
        """Test VAR model requires X for prediction."""
        service = TSFitPredictionService()

        with pytest.raises(ValueError) as exc_info:
            service.predict(model=sample_models["var"], model_type="var")
        assert "X is required for VAR" in str(exc_info.value)

    def test_predict_fallback(self, sample_models):
        """Test prediction fallback for unknown types uses model.predict."""
        service = TSFitPredictionService()

        # This should use the else clause and call model.predict()
        predictions = service.predict(
            model=sample_models["ar"], model_type="unknown", start=0, end=10
        )

        assert isinstance(predictions, np.ndarray)
        assert predictions.ndim == 2

    def test_forecast_ar(self, sample_models):
        """Test AR model forecasting."""
        service = TSFitPredictionService()

        forecast = service.forecast(model=sample_models["ar"], model_type="ar", steps=5)

        assert isinstance(forecast, np.ndarray)
        assert len(forecast) == 5

    def test_forecast_var_requires_x(self, sample_models):
        """Test VAR forecast requires X."""
        service = TSFitPredictionService()

        with pytest.raises(ValueError) as exc_info:
            service.forecast(model=sample_models["var"], model_type="var", steps=5)
        assert "X is required for VAR" in str(exc_info.value)


class TestTSFitScoringService:
    """Test scoring service functionality."""

    def test_score_mse(self):
        """Test MSE scoring."""
        service = TSFitScoringService()

        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])

        score = service.score(y_true, y_pred, metric="mse")
        expected = np.mean((y_true - y_pred) ** 2)
        assert np.isclose(score, expected)

    def test_score_mae(self):
        """Test MAE scoring."""
        service = TSFitScoringService()

        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])

        score = service.score(y_true, y_pred, metric="mae")
        expected = np.mean(np.abs(y_true - y_pred))
        assert np.isclose(score, expected)

    def test_score_rmse(self):
        """Test RMSE scoring."""
        service = TSFitScoringService()

        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])

        score = service.score(y_true, y_pred, metric="rmse")
        expected = np.sqrt(np.mean((y_true - y_pred) ** 2))
        assert np.isclose(score, expected)

    def test_score_mape(self):
        """Test MAPE scoring."""
        service = TSFitScoringService()

        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])

        score = service.score(y_true, y_pred, metric="mape")
        assert isinstance(score, float)
        assert score > 0

    def test_score_shape_mismatch(self):
        """Test shape mismatch error."""
        service = TSFitScoringService()

        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2])

        with pytest.raises(ValueError) as exc_info:
            service.score(y_true, y_pred)
        assert "Shape mismatch" in str(exc_info.value)

    def test_score_unknown_metric(self):
        """Test unknown metric error."""
        service = TSFitScoringService()

        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])

        with pytest.raises(ValueError) as exc_info:
            service.score(y_true, y_pred, metric="invalid")
        assert "Unknown metric" in str(exc_info.value)

    def test_get_information_criteria_aic(self):
        """Test AIC retrieval."""
        service = TSFitScoringService()

        # Mock model with AIC
        class MockModel:
            aic = 100.0

        result = service.get_information_criteria(MockModel(), "aic")
        assert result == 100.0

    def test_get_information_criteria_no_attribute(self):
        """Test information criteria when model lacks attribute."""
        service = TSFitScoringService()

        class MockModel:
            pass

        result = service.get_information_criteria(MockModel(), "aic")
        assert np.isinf(result)


class TestTSFitHelperService:
    """Test helper service functionality."""

    @pytest.fixture
    def sample_ar_model(self):
        """Create a sample AR model for testing."""
        np.random.seed(42)
        data = np.random.randn(100).cumsum()
        model = AutoReg(data, lags=2, trend="c")
        return model.fit()

    def test_get_residuals(self, sample_ar_model):
        """Test residual extraction."""
        service = TSFitHelperService()

        residuals = service.get_residuals(sample_ar_model)
        assert isinstance(residuals, np.ndarray)
        assert residuals.ndim == 2  # Should be 2D

    def test_get_residuals_standardized(self, sample_ar_model):
        """Test standardized residual extraction."""
        service = TSFitHelperService()

        residuals = service.get_residuals(sample_ar_model, standardize=True)
        assert isinstance(residuals, np.ndarray)
        # Check standardization (approximately)
        assert abs(np.std(residuals) - 1.0) < 0.1

    def test_get_fitted_values(self, sample_ar_model):
        """Test fitted value extraction."""
        service = TSFitHelperService()

        fitted = service.get_fitted_values(sample_ar_model)
        assert isinstance(fitted, np.ndarray)
        assert fitted.ndim == 2  # Should be 2D

    def test_calculate_trend_terms_ar(self, sample_ar_model):
        """Test trend term calculation for AR models."""
        service = TSFitHelperService()

        trend_terms = service.calculate_trend_terms("ar", sample_ar_model)
        assert isinstance(trend_terms, int)
        assert trend_terms >= 0

    def test_calculate_trend_terms_non_ar(self):
        """Test trend terms for non-AR models."""
        service = TSFitHelperService()

        # Models without trend terms return 0
        for model_type in ["var", "arch", "unknown"]:
            trend_terms = service.calculate_trend_terms(model_type, None)
            assert trend_terms == 0

    def test_check_stationarity_adf(self):
        """Test ADF stationarity test."""
        service = TSFitHelperService()

        # Generate stationary data
        np.random.seed(42)
        residuals = np.random.randn(100)

        is_stationary, p_value = service.check_stationarity(residuals, test="adf")
        # Check the stationarity result
        assert isinstance(is_stationary, (bool, np.bool_))
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1

    def test_check_stationarity_kpss(self):
        """Test KPSS stationarity test."""
        service = TSFitHelperService()

        # Generate data
        np.random.seed(42)
        residuals = np.random.randn(100)

        is_stationary, p_value = service.check_stationarity(residuals, test="kpss")
        assert isinstance(is_stationary, (bool, np.bool_))
        assert isinstance(p_value, float)

    def test_check_stationarity_invalid_test(self):
        """Test invalid stationarity test."""
        service = TSFitHelperService()

        with pytest.raises(ValueError) as exc_info:
            service.check_stationarity(np.random.randn(100), test="invalid")
        assert "Unknown test" in str(exc_info.value)


class TestIntegration:
    """Integration tests for TSFit services."""

    def test_model_fitting_prediction_scoring_workflow(self):
        """Test complete workflow with all services."""
        # Generate test data
        np.random.seed(42)
        data = np.random.randn(100).cumsum()

        # Initialize services
        validation_service = TSFitValidationService()
        prediction_service = TSFitPredictionService()
        scoring_service = TSFitScoringService()
        helper_service = TSFitHelperService()

        # Validate model type and order
        model_type = validation_service.validate_model_type("ar")
        order = validation_service.validate_order(2, model_type)

        # Fit model
        model = AutoReg(data, lags=order, trend="c")
        fitted_model = model.fit()

        # Get predictions
        predictions = prediction_service.predict(
            model=fitted_model, model_type=model_type, start=50, end=80
        )

        # Score predictions
        y_true = data[50:81].reshape(-1, 1)
        score = scoring_service.score(y_true, predictions, metric="rmse")

        # Check residuals
        residuals = helper_service.get_residuals(fitted_model)

        # All operations should succeed
        assert isinstance(predictions, np.ndarray)
        assert isinstance(score, float)
        assert isinstance(residuals, np.ndarray)
