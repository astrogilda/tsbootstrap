"""Tests for backend-compatible services."""

from typing import Any, Dict, Optional, Tuple
from unittest.mock import Mock

import numpy as np
import pytest
from tsbootstrap.backends.protocol import FittedModelBackend, ModelBackend
from tsbootstrap.services.backend_services import (
    BackendCompositeService,
    BackendHelperService,
    BackendPredictionService,
    BackendScoringService,
    BackendValidationService,
)


class MockFittedBackend:
    """Mock fitted backend for testing."""

    def __init__(
        self,
        residuals: Optional[np.ndarray] = None,
        fitted_values: Optional[np.ndarray] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        self._residuals = residuals if residuals is not None else np.random.randn(100)
        self._fitted_values = fitted_values if fitted_values is not None else np.random.randn(100)
        self._params = params if params is not None else {"ar": [0.5], "sigma2": 1.0}

    @property
    def residuals(self) -> np.ndarray:
        return self._residuals

    @property
    def fitted_values(self) -> np.ndarray:
        return self._fitted_values

    @property
    def params(self) -> Dict[str, Any]:
        return self._params

    def predict(self, steps: int, X: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        return np.random.randn(steps)

    def simulate(
        self,
        steps: int,
        n_paths: int = 1,
        X: Optional[np.ndarray] = None,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        if random_state is not None:
            np.random.seed(random_state)
        return np.random.randn(n_paths, steps)

    def get_info_criteria(self) -> Dict[str, float]:
        return {"aic": 100.0, "bic": 110.0, "hqic": 105.0}

    def check_stationarity(
        self, test: str = "adf", significance: float = 0.05
    ) -> Tuple[bool, float]:
        return True, 0.01

    def score(
        self,
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
        metric: str = "r2",
    ) -> float:
        if metric == "r2":
            return 0.85
        return 0.1


class MockBackend:
    """Mock backend for testing."""

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None, **kwargs) -> MockFittedBackend:
        return MockFittedBackend()


class TestBackendValidationService:
    """Test backend validation service."""

    def test_validate_model_config_basic(self):
        """Test basic model configuration validation."""
        backend = MockBackend()
        service = BackendValidationService()

        config = service.validate_model_config(
            backend=backend,
            model_type="ARIMA",
            order=(1, 0, 1),
        )

        assert config["model_type"] == "ARIMA"
        assert config["order"] == (1, 0, 1)

    def test_validate_order_integer(self):
        """Test integer order validation."""
        service = BackendValidationService()

        # Valid integer
        assert service._validate_order(1) == 1
        assert service._validate_order(0) == 0

        # Invalid negative
        with pytest.raises(ValueError, match="must be non-negative"):
            service._validate_order(-1)

    def test_validate_order_tuple(self):
        """Test tuple order validation."""
        service = BackendValidationService()

        # Valid tuples
        assert service._validate_order((1, 0, 1)) == (1, 0, 1)
        assert service._validate_order([2, 1, 2]) == (2, 1, 2)
        assert service._validate_order((1, 0, 1, 0)) == (1, 0, 1, 0)

        # Invalid element
        with pytest.raises(ValueError, match="non-negative integers"):
            service._validate_order((1, -1, 1))

        # Invalid length
        with pytest.raises(ValueError, match="2, 3, or 4 elements"):
            service._validate_order((1,))

    def test_validate_order_none(self):
        """Test None order validation."""
        service = BackendValidationService()
        assert service._validate_order(None) is None

    def test_validate_order_invalid_type(self):
        """Test invalid order type."""
        service = BackendValidationService()
        with pytest.raises(TypeError, match="Invalid order type"):
            service._validate_order("invalid")

    def test_validate_seasonal_order(self):
        """Test seasonal order validation."""
        service = BackendValidationService()

        # Valid seasonal order
        assert service._validate_seasonal_order((1, 0, 1, 12)) == (1, 0, 1, 12)

        # None is valid
        assert service._validate_seasonal_order(None) is None

        # Invalid length
        with pytest.raises(ValueError, match="4 elements"):
            service._validate_seasonal_order((1, 0, 1))

        # Invalid seasonal period
        with pytest.raises(ValueError, match="at least 2"):
            service._validate_seasonal_order((1, 0, 1, 1))

        # Invalid type
        with pytest.raises(TypeError, match="tuple or list"):
            service._validate_seasonal_order("invalid")


class TestBackendPredictionService:
    """Test backend prediction service."""

    def test_predict_basic(self):
        """Test basic prediction."""
        fitted = MockFittedBackend()
        service = BackendPredictionService()

        predictions = service.predict(fitted, steps=5)
        assert len(predictions) == 5

    def test_predict_with_start_end(self):
        """Test prediction with start and end indices."""
        fitted = MockFittedBackend()
        service = BackendPredictionService()

        predictions = service.predict(fitted, start=0, end=4)
        assert len(predictions) == 5

    def test_predict_in_sample(self):
        """Test in-sample prediction."""
        fitted_vals = np.arange(100)
        fitted = MockFittedBackend(fitted_values=fitted_vals)
        service = BackendPredictionService()

        # Get in-sample predictions
        predictions = service.predict(fitted, start=10, end=14)
        assert len(predictions) == 5
        # Should return fitted values for in-sample range
        np.testing.assert_array_equal(predictions, fitted_vals[10:15])

    def test_forecast(self):
        """Test forecasting."""
        fitted = MockFittedBackend()
        service = BackendPredictionService()

        forecasts = service.forecast(fitted, steps=10)
        assert len(forecasts) == 10


class TestBackendScoringService:
    """Test backend scoring service."""

    def test_score_mse(self):
        """Test MSE scoring."""
        service = BackendScoringService()
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 3.9, 5.1])

        score = service.score(y_true, y_pred, metric="mse")
        expected = np.mean((y_true - y_pred) ** 2)
        assert np.isclose(score, expected)

    def test_score_mae(self):
        """Test MAE scoring."""
        service = BackendScoringService()
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 3.9, 5.1])

        score = service.score(y_true, y_pred, metric="mae")
        expected = np.mean(np.abs(y_true - y_pred))
        assert np.isclose(score, expected)

    def test_score_rmse(self):
        """Test RMSE scoring."""
        service = BackendScoringService()
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 3.9, 5.1])

        score = service.score(y_true, y_pred, metric="rmse")
        expected = np.sqrt(np.mean((y_true - y_pred) ** 2))
        assert np.isclose(score, expected)

    def test_score_mape(self):
        """Test MAPE scoring."""
        service = BackendScoringService()
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 3.9, 5.1])

        score = service.score(y_true, y_pred, metric="mape")
        expected = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        assert np.isclose(score, expected)

    def test_score_mape_with_zeros(self):
        """Test MAPE with zeros in y_true."""
        service = BackendScoringService()
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])

        score = service.score(y_true, y_pred, metric="mape")
        assert score == np.inf

    def test_score_r2(self):
        """Test R-squared scoring."""
        service = BackendScoringService()
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 3.9, 5.1])

        score = service.score(y_true, y_pred, metric="r2")
        # Should be close to 1 for good predictions
        assert 0.9 < score < 1.0

    def test_score_shape_mismatch(self):
        """Test error on shape mismatch."""
        service = BackendScoringService()
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2])

        with pytest.raises(ValueError, match="Shape mismatch"):
            service.score(y_true, y_pred)

    def test_score_unknown_metric(self):
        """Test error on unknown metric."""
        service = BackendScoringService()
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="Unknown metric"):
            service.score(y_true, y_pred, metric="unknown")

    def test_get_information_criteria(self):
        """Test getting information criteria."""
        fitted = MockFittedBackend()
        service = BackendScoringService()

        aic = service.get_information_criteria(fitted, "aic")
        assert aic == 100.0

        bic = service.get_information_criteria(fitted, "bic")
        assert bic == 110.0


class TestBackendHelperService:
    """Test backend helper service."""

    def test_get_residuals(self):
        """Test getting residuals."""
        residuals = np.array([1, -1, 2, -2, 0])
        fitted = MockFittedBackend(residuals=residuals)
        service = BackendHelperService()

        result = service.get_residuals(fitted)
        np.testing.assert_array_equal(result, residuals)

    def test_get_residuals_standardized(self):
        """Test getting standardized residuals."""
        residuals = np.array([1, -1, 2, -2, 0])
        fitted = MockFittedBackend(residuals=residuals)
        service = BackendHelperService()

        result = service.get_residuals(fitted, standardize=True)
        std = np.std(residuals)
        expected = residuals / std
        np.testing.assert_array_almost_equal(result, expected)

    def test_get_fitted_values(self):
        """Test getting fitted values."""
        fitted_values = np.array([1, 2, 3, 4, 5])
        fitted = MockFittedBackend(fitted_values=fitted_values)
        service = BackendHelperService()

        result = service.get_fitted_values(fitted)
        np.testing.assert_array_equal(result, fitted_values)

    def test_calculate_trend_terms(self):
        """Test calculating trend terms."""
        service = BackendHelperService()

        # No trend
        fitted = MockFittedBackend(params={"trend": "n"})
        assert service.calculate_trend_terms(fitted) == 0

        # Constant trend
        fitted = MockFittedBackend(params={"trend": "c"})
        assert service.calculate_trend_terms(fitted) == 1

        # Time trend
        fitted = MockFittedBackend(params={"trend": "t"})
        assert service.calculate_trend_terms(fitted) == 1

        # Constant + time trend
        fitted = MockFittedBackend(params={"trend": "ct"})
        assert service.calculate_trend_terms(fitted) == 2

        # Intercept/const in params
        fitted = MockFittedBackend(params={"const": 1.0})
        assert service.calculate_trend_terms(fitted) == 1

        # No trend info
        fitted = MockFittedBackend(params={})
        assert service.calculate_trend_terms(fitted) == 0

    def test_check_stationarity(self):
        """Test stationarity check."""
        fitted = MockFittedBackend()
        service = BackendHelperService()

        is_stationary, p_value = service.check_stationarity(fitted)
        assert is_stationary is True
        assert p_value == 0.01

    def test_validate_predictions_shape(self):
        """Test prediction shape validation."""
        service = BackendHelperService()

        # Basic validation
        predictions = np.array([1, 2, 3])
        result = service.validate_predictions_shape(predictions)
        np.testing.assert_array_equal(result, predictions)

        # Ensure 2D
        result = service.validate_predictions_shape(predictions, ensure_2d=True)
        assert result.shape == (3, 1)

        # Expected shape matching
        predictions = np.array([1, 2, 3, 4, 5, 6])
        result = service.validate_predictions_shape(predictions, expected_shape=(2, 3))
        assert result.shape == (2, 3)

        # Shape mismatch error
        with pytest.raises(ValueError, match="Cannot reshape"):
            service.validate_predictions_shape(predictions, expected_shape=(2, 4))


class TestBackendCompositeService:
    """Test composite backend service."""

    def test_validate_and_fit(self):
        """Test validate and fit workflow."""
        backend = MockBackend()
        service = BackendCompositeService()

        y = np.random.randn(100)
        fitted = service.validate_and_fit(
            backend=backend,
            y=y,
            model_type="ARIMA",
            order=(1, 0, 1),
        )

        assert isinstance(fitted, MockFittedBackend)

    def test_evaluate_model_in_sample(self):
        """Test model evaluation with in-sample metrics."""
        residuals = np.random.randn(100) * 0.1
        fitted_values = np.sin(np.linspace(0, 4 * np.pi, 100))
        fitted = MockFittedBackend(
            residuals=residuals,
            fitted_values=fitted_values,
        )
        service = BackendCompositeService()

        results = service.evaluate_model(fitted)

        # Check in-sample metrics exist
        assert "in_sample_mse" in results
        assert "in_sample_mae" in results
        assert "in_sample_rmse" in results
        assert "in_sample_r2" in results

        # Check information criteria
        assert "aic" in results
        assert "bic" in results
        assert "hqic" in results

        # Check stationarity
        assert "residuals_stationary" in results
        assert "residuals_stationarity_pvalue" in results

    def test_evaluate_model_out_sample(self):
        """Test model evaluation with out-of-sample metrics."""
        fitted = MockFittedBackend()
        service = BackendCompositeService()

        y_test = np.random.randn(20)
        results = service.evaluate_model(fitted, y_test=y_test, n_ahead=20)

        # Check out-of-sample metrics exist
        assert "out_sample_mse" in results
        assert "out_sample_mae" in results
        assert "out_sample_rmse" in results
        assert "out_sample_r2" in results

    def test_evaluate_model_custom_metrics(self):
        """Test model evaluation with custom metrics."""
        fitted = MockFittedBackend()
        service = BackendCompositeService()

        results = service.evaluate_model(fitted, metrics=["mse", "mae"])

        # Only requested metrics should be computed
        assert "in_sample_mse" in results
        assert "in_sample_mae" in results
        assert "in_sample_rmse" not in results
        assert "in_sample_r2" not in results


class TestBackendProtocolCompliance:
    """Test that services work with any protocol-compliant backend."""

    def test_with_mock_protocol_backend(self):
        """Test services with a mock that implements the protocol."""
        # Create protocol-compliant mocks
        backend = Mock(spec=ModelBackend)
        fitted_backend = Mock(spec=FittedModelBackend)

        # Set up mock behavior
        backend.fit.return_value = fitted_backend
        fitted_backend.residuals = np.random.randn(100)
        fitted_backend.fitted_values = np.random.randn(100)
        fitted_backend.params = {"ar": [0.5], "sigma2": 1.0}
        fitted_backend.predict.return_value = np.random.randn(10)
        fitted_backend.get_info_criteria.return_value = {
            "aic": 100.0,
            "bic": 110.0,
        }
        fitted_backend.check_stationarity.return_value = (True, 0.01)

        # Test composite service
        service = BackendCompositeService()
        y = np.random.randn(100)

        # Validate and fit
        result = service.validate_and_fit(backend, y, order=(1, 0, 1))
        assert result == fitted_backend
        backend.fit.assert_called_once()

        # Test prediction
        predictions = service.prediction.predict(fitted_backend, steps=10)
        assert len(predictions) == 10

        # Test scoring
        aic = service.scoring.get_information_criteria(fitted_backend, "aic")
        assert aic == 100.0

        # Test helper
        residuals = service.helper.get_residuals(fitted_backend)
        assert len(residuals) == 100
