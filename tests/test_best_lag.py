"""
Comprehensive tests for best_lag.py to achieve 80%+ coverage.

Tests TSFitBestLag class for automatic lag selection.
"""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from tsbootstrap.model_selection.best_lag import TSFitBestLag


class TestTSFitBestLag:
    """Test TSFitBestLag class."""

    def test_init_default(self):
        """Test default initialization."""
        model = TSFitBestLag(model_type="ar")
        assert model.model_type == "ar"
        assert model.max_lag == 10
        assert model.order is None
        assert model.seasonal_order is None
        assert model.save_models is False
        assert model.model_params == {}

    def test_init_with_params(self):
        """Test initialization with parameters."""
        model = TSFitBestLag(
            model_type="arima",
            max_lag=20,
            order=(2, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            save_models=True,
            trend="c",
            enforce_stationarity=False,
        )
        assert model.model_type == "arima"
        assert model.max_lag == 20
        assert model.order == (2, 1, 1)
        assert model.seasonal_order == (1, 1, 1, 12)
        assert model.save_models is True
        assert model.model_params["trend"] == "c"
        assert model.model_params["enforce_stationarity"] is False

    def test_compute_best_order_ar(self):
        """Test automatic order computation for AR model."""
        np.random.seed(42)
        X = np.random.randn(100).cumsum()

        model = TSFitBestLag(model_type="ar", max_lag=5)
        order = model._compute_best_order(X)

        assert isinstance(order, (int, np.integer))
        assert 1 <= order <= 5

    def test_compute_best_order_arima(self):
        """Test automatic order computation for ARIMA model."""
        np.random.seed(42)
        X = np.random.randn(100).cumsum()

        model = TSFitBestLag(model_type="arima", max_lag=5)
        order = model._compute_best_order(X)

        assert isinstance(order, tuple)
        assert len(order) == 3
        assert order[1] == 0  # d=0
        assert order[2] == 0  # q=0

    def test_compute_best_order_sarima(self):
        """Test automatic order computation for SARIMA model."""
        np.random.seed(42)
        X = np.random.randn(100).cumsum()

        model = TSFitBestLag(model_type="sarima", max_lag=5)
        order = model._compute_best_order(X)

        assert isinstance(order, tuple)
        assert len(order) == 3
        # For SARIMA, returns non-seasonal order

    def test_fit_ar_auto_order(self):
        """Test fitting AR model with automatic order selection."""
        np.random.seed(42)
        X = np.random.randn(100).cumsum().reshape(-1, 1)

        model = TSFitBestLag(model_type="ar", max_lag=5)
        model.fit(X)

        assert model.order is not None
        assert model.fitted_adapter is not None
        assert model.model is not None
        assert hasattr(model, "X_fitted_")
        assert hasattr(model, "resids_")

    def test_fit_ar_manual_order(self):
        """Test fitting AR model with manual order."""
        np.random.seed(42)
        X = np.random.randn(100).cumsum().reshape(-1, 1)

        model = TSFitBestLag(model_type="ar", order=2)
        model.fit(X)

        assert model.order == 2
        assert model.fitted_adapter is not None
        assert model.model is not None

    def test_fit_arima(self):
        """Test fitting ARIMA model."""
        np.random.seed(42)
        X = np.random.randn(100).cumsum().reshape(-1, 1)

        model = TSFitBestLag(model_type="arima", order=(1, 1, 1))
        model.fit(X)

        assert model.order == (1, 1, 1)
        assert model.fitted_adapter is not None
        assert model.model is not None

    def test_fit_sarima(self):
        """Test fitting SARIMA model."""
        np.random.seed(42)
        X = np.random.randn(120).cumsum().reshape(-1, 1)

        model = TSFitBestLag(model_type="sarima", order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model.fit(X)

        assert model.order == (1, 1, 1)
        assert model.seasonal_order == (1, 1, 1, 12)
        assert model.fitted_adapter is not None
        assert model.model is not None

    def test_fit_var(self):
        """Test fitting VAR model."""
        np.random.seed(42)
        X = np.random.randn(100, 2)  # Multivariate

        model = TSFitBestLag(model_type="var", max_lag=3)
        model.fit(X)

        assert model.order is not None
        assert model.fitted_adapter is not None
        assert model.model is not None

    def test_fit_with_exogenous(self):
        """Test fitting with exogenous variables."""
        np.random.seed(42)
        X = np.random.randn(100).cumsum().reshape(-1, 1)
        y = np.random.randn(100, 2)

        model = TSFitBestLag(model_type="ar", order=2)
        model.fit(X, y=y)

        assert model.fitted_adapter is not None
        assert model.model is not None

    def test_get_coefs(self):
        """Test getting coefficients."""
        np.random.seed(42)
        X = np.random.randn(100).cumsum().reshape(-1, 1)

        model = TSFitBestLag(model_type="ar", order=2)
        model.fit(X)

        coefs = model.get_coefs()
        assert isinstance(coefs, np.ndarray)
        assert len(coefs) > 0

    def test_get_coefs_not_fitted(self):
        """Test getting coefficients before fitting."""
        model = TSFitBestLag(model_type="ar")

        with pytest.raises(NotFittedError):
            model.get_coefs()

    def test_get_intercepts(self):
        """Test getting intercepts."""
        np.random.seed(42)
        X = np.random.randn(100).cumsum().reshape(-1, 1)

        model = TSFitBestLag(model_type="ar", order=2)
        model.fit(X)

        intercepts = model.get_intercepts()
        assert isinstance(intercepts, np.ndarray)

    def test_get_intercepts_not_fitted(self):
        """Test getting intercepts before fitting."""
        model = TSFitBestLag(model_type="ar")

        with pytest.raises(NotFittedError):
            model.get_intercepts()

    def test_get_residuals(self):
        """Test getting residuals."""
        np.random.seed(42)
        X = np.random.randn(100).cumsum().reshape(-1, 1)

        model = TSFitBestLag(model_type="ar", order=2)
        model.fit(X)

        residuals = model.get_residuals()
        assert isinstance(residuals, np.ndarray)
        # AR models lose observations equal to the order
        assert residuals.shape[0] == X.shape[0] - model.order

    def test_get_residuals_not_fitted(self):
        """Test getting residuals before fitting."""
        model = TSFitBestLag(model_type="ar")

        with pytest.raises(NotFittedError):
            model.get_residuals()

    def test_get_fitted_X(self):
        """Test getting fitted values."""
        np.random.seed(42)
        X = np.random.randn(100).cumsum().reshape(-1, 1)

        model = TSFitBestLag(model_type="ar", order=2)
        model.fit(X)

        fitted = model.get_fitted_X()
        assert isinstance(fitted, np.ndarray)
        # AR models lose observations equal to the order
        assert fitted.shape[0] == X.shape[0] - model.order
        assert fitted.shape[1] == X.shape[1]

    def test_get_fitted_X_not_fitted(self):
        """Test getting fitted values before fitting."""
        model = TSFitBestLag(model_type="ar")

        with pytest.raises(NotFittedError):
            model.get_fitted_X()

    def test_get_order(self):
        """Test getting order."""
        np.random.seed(42)
        X = np.random.randn(100).cumsum().reshape(-1, 1)

        model = TSFitBestLag(model_type="ar", order=3)
        model.fit(X)

        order = model.get_order()
        assert order == 3

    def test_get_order_not_fitted(self):
        """Test getting order before fitting."""
        model = TSFitBestLag(model_type="ar")

        with pytest.raises(NotFittedError):
            model.get_order()

    def test_get_model(self):
        """Test getting the underlying model."""
        np.random.seed(42)
        X = np.random.randn(100).cumsum().reshape(-1, 1)

        model = TSFitBestLag(model_type="ar", order=2)
        model.fit(X)

        underlying_model = model.get_model()
        assert underlying_model is not None

    def test_get_model_not_fitted(self):
        """Test getting model before fitting."""
        model = TSFitBestLag(model_type="ar")

        with pytest.raises(NotFittedError):
            model.get_model()

    def test_predict(self):
        """Test prediction."""
        np.random.seed(42)
        X = np.random.randn(100).cumsum().reshape(-1, 1)

        model = TSFitBestLag(model_type="ar", order=2)
        model.fit(X)

        # Predict using the fitted values - TSFit predict just returns fitted values
        predictions = model.predict(X)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) > 0

    def test_predict_not_fitted(self):
        """Test prediction before fitting."""
        model = TSFitBestLag(model_type="ar")
        X = np.random.randn(10).reshape(-1, 1)

        with pytest.raises(NotFittedError):
            model.predict(X)

    def test_score(self):
        """Test scoring."""
        np.random.seed(42)
        X_train = np.random.randn(80).cumsum().reshape(-1, 1)
        X_test = np.random.randn(20).cumsum().reshape(-1, 1)

        model = TSFitBestLag(model_type="ar", order=2)
        model.fit(X_train)

        # Score on test data
        score = model.score(X_train, X_test)
        assert isinstance(score, float)

    def test_score_not_fitted(self):
        """Test scoring before fitting."""
        model = TSFitBestLag(model_type="ar")
        X = np.random.randn(20).reshape(-1, 1)
        y = np.random.randn(20).reshape(-1, 1)

        with pytest.raises(NotFittedError):
            model.score(X, y)

    def test_repr(self):
        """Test string representation."""
        model = TSFitBestLag(model_type="arima", order=(2, 1, 1), max_lag=15, trend="ct")
        repr_str = repr(model)

        assert "TSFitBestLag" in repr_str
        assert "model_type='arima'" in repr_str
        assert "order=(2, 1, 1)" in repr_str
        assert "max_lag=15" in repr_str
        assert "trend" in repr_str and "ct" in repr_str

    def test_str(self):
        """Test string conversion."""
        model = TSFitBestLag(model_type="ar", order=2)
        str_repr = str(model)

        assert "TSFitBestLag" in str_repr
        assert "model_type='ar'" in str_repr
        assert "order=2" in str_repr

    def test_equality(self):
        """Test equality comparison."""
        model1 = TSFitBestLag(model_type="ar", order=2, max_lag=10)
        model2 = TSFitBestLag(model_type="ar", order=2, max_lag=10)
        model3 = TSFitBestLag(model_type="ar", order=3, max_lag=10)

        assert model1 == model2
        assert model1 != model3
        assert model1 != "not_a_model"

    def test_equality_with_fitted_models(self):
        """Test equality with fitted models."""
        np.random.seed(42)
        X = np.random.randn(100).cumsum().reshape(-1, 1)

        model1 = TSFitBestLag(model_type="ar", order=2)
        model2 = TSFitBestLag(model_type="ar", order=2)

        model1.fit(X)
        model2.fit(X)

        # Models should be equal in configuration
        # But exact model comparison is tricky, so we check type
        assert isinstance(model1.model, type(model2.model))

    @pytest.mark.skipif(
        True,  # Skip ARCH tests - TSFitBestLag doesn't fully support ARCH models
        reason="ARCH models don't have fitted values in the same way as other models",
    )
    def test_fit_arch(self):
        """Test fitting ARCH model."""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.01

        model = TSFitBestLag(model_type="arch", order=1)
        model.fit(returns.reshape(-1, 1))

        assert model.order == 1
        assert model.fitted_adapter is not None
        assert model.model is not None

    def test_error_no_order_determinable(self):
        """Test error when order cannot be determined."""
        # This is a bit artificial, but tests the error path
        model = TSFitBestLag(model_type="ar")
        model.order = None

        # Mock _compute_best_order to return None
        original_compute = model._compute_best_order
        model._compute_best_order = lambda X: None

        X = np.random.randn(100).reshape(-1, 1)

        with pytest.raises(ValueError, match="Failed to determine model order automatically"):
            model.fit(X)

        # Restore
        model._compute_best_order = original_compute

    def test_save_models_flag(self):
        """Test save_models flag is passed to RankLags."""
        np.random.seed(42)
        X = np.random.randn(100).cumsum().reshape(-1, 1)

        model = TSFitBestLag(model_type="ar", save_models=True)
        model.fit(X)

        # Check that RankLags was created with save_models=True
        assert model.rank_lagger is not None
        # Note: Can't directly check save_models on rank_lagger without accessing private attributes


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    def test_small_sample_size(self):
        """Test with small sample size."""
        X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)

        model = TSFitBestLag(model_type="ar", max_lag=2)

        # Should handle small samples gracefully
        model.fit(X)
        assert model.order is not None

    def test_multivariate_for_univariate_model(self):
        """Test multivariate data with univariate model."""
        X = np.random.randn(100, 3)

        model = TSFitBestLag(model_type="ar", order=2)

        # AR models require univariate data, so we should get an error
        with pytest.raises(ValueError, match="Univariate models.*require single time series data"):
            model.fit(X)

    def test_predict_with_exogenous(self):
        """Test prediction with exogenous variables."""
        np.random.seed(42)
        X = np.random.randn(100).cumsum().reshape(-1, 1)
        y = np.random.randn(100, 2)

        model = TSFitBestLag(model_type="ar", order=2)
        model.fit(X, y=y)

        # Predict - TSFit doesn't use exogenous for predict
        predictions = model.predict(X)
        assert len(predictions) > 0
