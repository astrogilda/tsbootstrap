"""Tests for TimeSeriesModelSklearn - sklearn-compatible interface."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tsbootstrap.time_series_model_sklearn import TimeSeriesModelSklearn


@pytest.fixture
def sample_data():
    """Generate sample time series data."""
    np.random.seed(42)
    n_samples = 100
    X = np.cumsum(np.random.randn(n_samples)) + 50
    y = np.random.randn(n_samples, 2)  # Exogenous variables
    return X, y


@pytest.fixture
def multivariate_data():
    """Generate multivariate time series data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 3
    X = np.cumsum(np.random.randn(n_samples, n_features), axis=0) + 50
    return X


class TestTimeSeriesModelSklearn:
    """Test TimeSeriesModelSklearn class."""

    def test_initialization(self):
        """Test model initialization with various parameters."""
        # Test default initialization
        model = TimeSeriesModelSklearn()
        assert model.model_type == "ar"
        assert model.verbose == True
        assert model.use_backend == False
        assert model.order is None
        assert model.seasonal_order is None

        # Test with custom parameters
        model = TimeSeriesModelSklearn(
            model_type="arima", verbose=False, use_backend=True, order=(2, 1, 1), trend="c"
        )
        assert model.model_type == "arima"
        assert model.verbose == False
        assert model.use_backend == True
        assert model.order == (2, 1, 1)
        assert model.model_params["trend"] == "c"

    def test_fit_predict_ar(self, sample_data):
        """Test fit and predict for AR model."""
        X, y = sample_data

        model = TimeSeriesModelSklearn(model_type="ar", order=2)
        model.fit(X)

        # Check fitted attributes
        assert hasattr(model, "fitted_model_")
        assert hasattr(model, "X_")
        assert model.X_ is X

        # Test predictions
        predictions = model.predict()
        assert isinstance(predictions, np.ndarray)
        assert predictions.ndim == 2
        assert predictions.shape[1] == 1

    def test_fit_predict_arima(self, sample_data):
        """Test fit and predict for ARIMA model."""
        X, y = sample_data

        model = TimeSeriesModelSklearn(model_type="arima", order=(2, 1, 1))
        model.fit(X)

        predictions = model.predict()
        assert isinstance(predictions, np.ndarray)
        assert predictions.ndim == 2

    def test_fit_predict_sarima(self, sample_data):
        """Test fit and predict for SARIMA model."""
        X, y = sample_data

        model = TimeSeriesModelSklearn(
            model_type="sarima", order=(1, 0, 1), seasonal_order=(1, 0, 1, 12)
        )
        model.fit(X)

        predictions = model.predict()
        assert isinstance(predictions, np.ndarray)
        assert predictions.ndim == 2

    def test_fit_predict_var(self, multivariate_data):
        """Test fit and predict for VAR model."""
        X = multivariate_data

        model = TimeSeriesModelSklearn(model_type="var", order=2)
        model.fit(X)

        # VAR requires data for prediction
        predictions = model.predict(X=X[:10])
        assert isinstance(predictions, np.ndarray)
        assert predictions.ndim == 2
        assert predictions.shape[1] == X.shape[1]

    def test_fit_predict_arch(self, sample_data):
        """Test fit and predict for ARCH model."""
        X, y = sample_data

        model = TimeSeriesModelSklearn(
            model_type="arch", order=1, p=1, q=1, arch_model_type="GARCH"
        )
        model.fit(X)

        predictions = model.predict()
        assert isinstance(predictions, np.ndarray)
        assert predictions.ndim == 2

    def test_forecast(self, sample_data):
        """Test forecasting functionality."""
        X, y = sample_data

        model = TimeSeriesModelSklearn(model_type="ar", order=2)
        model.fit(X)

        # Test single step forecast
        forecast = model.forecast(steps=1)
        assert forecast.shape == (1, 1)

        # Test multi-step forecast
        forecast = model.forecast(steps=5)
        assert forecast.shape == (5, 1)

    def test_score_metrics(self, sample_data):
        """Test various scoring metrics."""
        X, y = sample_data

        model = TimeSeriesModelSklearn(model_type="ar", order=2)
        model.fit(X)

        # Test R² score (default)
        score = model.score()
        assert isinstance(score, float)
        assert -1 <= score <= 1 or np.isnan(score)

        # Test MSE
        mse = model.score(metric="mse")
        assert isinstance(mse, float)
        assert mse >= 0 or np.isnan(mse)

        # Test MAE
        mae = model.score(metric="mae")
        assert isinstance(mae, float)
        assert mae >= 0 or np.isnan(mae)

        # Test RMSE
        rmse = model.score(metric="rmse")
        assert isinstance(rmse, float)
        assert rmse >= 0 or np.isnan(rmse)

        # Test MAPE
        mape = model.score(metric="mape")
        assert isinstance(mape, float)

        # Test with explicit X
        score_with_x = model.score(X=X)
        assert isinstance(score_with_x, float)

        # Test invalid metric
        with pytest.raises(ValueError, match="Unknown metric"):
            model.score(metric="invalid")

    def test_get_residuals(self, sample_data):
        """Test residuals extraction."""
        X, y = sample_data

        model = TimeSeriesModelSklearn(model_type="ar", order=2)
        model.fit(X)

        # Test raw residuals
        residuals = model.get_residuals()
        assert isinstance(residuals, np.ndarray)

        # Test standardized residuals
        std_residuals = model.get_residuals(standardize=True)
        assert isinstance(std_residuals, np.ndarray)
        # Check that standardization worked (should have unit variance)
        assert np.allclose(np.std(std_residuals), 1.0, rtol=0.1)

    def test_get_fitted_values(self, sample_data):
        """Test fitted values extraction."""
        X, y = sample_data

        model = TimeSeriesModelSklearn(model_type="ar", order=2)
        model.fit(X)

        fitted = model.get_fitted_values()
        assert isinstance(fitted, np.ndarray)
        assert fitted.ndim == 2

    def test_information_criteria(self, sample_data):
        """Test information criteria methods."""
        X, y = sample_data

        model = TimeSeriesModelSklearn(model_type="ar", order=2)
        model.fit(X)

        # Test AIC
        aic = model.get_information_criterion("aic")
        assert isinstance(aic, float)

        # Test BIC
        bic = model.get_information_criterion("bic")
        assert isinstance(bic, float)

        # Test HQIC
        hqic = model.get_information_criterion("hqic")
        assert isinstance(hqic, float)

        # Test invalid criterion
        with pytest.raises(ValueError, match="Unknown criterion"):
            model.get_information_criterion("invalid")

    def test_summary(self, sample_data):
        """Test model summary."""
        X, y = sample_data

        model = TimeSeriesModelSklearn(model_type="ar", order=2)
        model.fit(X)

        summary = model.summary()
        assert summary is not None

    def test_sklearn_clone(self, sample_data):
        """Test sklearn clone functionality."""
        X, y = sample_data

        model = TimeSeriesModelSklearn(model_type="ar", order=2)

        # Clone before fitting
        cloned = clone(model)
        assert cloned.model_type == model.model_type
        assert cloned.order == model.order

        # Fit original
        model.fit(X)

        # Cloned should not be fitted
        with pytest.raises(Exception):
            cloned.predict()

    def test_sklearn_pipeline(self, sample_data):
        """Test usage in sklearn pipeline."""
        X, y = sample_data

        # Create pipeline with preprocessing
        # Note: StandardScaler expects 2D input, so reshape
        X_2d = X.reshape(-1, 1)

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", TimeSeriesModelSklearn(model_type="ar", order=2)),
            ]
        )

        # Fit pipeline
        pipeline.fit(X_2d)

        # Predict
        predictions = pipeline.predict()
        assert isinstance(predictions, np.ndarray)

    def test_sklearn_gridsearch(self, sample_data):
        """Test usage with GridSearchCV."""
        X, y = sample_data

        model = TimeSeriesModelSklearn(model_type="ar")

        # Define parameter grid
        param_grid = {"order": [1, 2, 3]}

        # Create GridSearchCV
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,  # Time series split would be better in practice
            scoring="r2",
        )

        # Fit grid search
        grid.fit(X)

        # Check best parameters
        assert hasattr(grid, "best_params_")
        assert "order" in grid.best_params_
        assert grid.best_params_["order"] in [1, 2, 3]

        # Check predictions work
        predictions = grid.predict()
        assert isinstance(predictions, np.ndarray)

    def test_get_params_set_params(self):
        """Test get_params and set_params for sklearn compatibility."""
        model = TimeSeriesModelSklearn(
            model_type="arima", order=(2, 1, 1), verbose=False, trend="c"
        )

        # Test get_params
        params = model.get_params()
        assert isinstance(params, dict)
        assert params["model_type"] == "arima"
        assert params["order"] == (2, 1, 1)
        assert params["verbose"] == False
        assert "trend" in params
        assert params["trend"] == "c"

        # Test set_params
        model.set_params(order=(1, 0, 1), verbose=True)
        assert model.order == (1, 0, 1)
        assert model.verbose == True

        # Test set_params returns self
        result = model.set_params(model_type="ar")
        assert result is model
        assert model.model_type == "ar"

    def test_repr(self):
        """Test string representation."""
        model = TimeSeriesModelSklearn(
            model_type="sarima",
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 12),
            verbose=False,
            trend="ct",
        )

        repr_str = repr(model)
        assert "TimeSeriesModelSklearn" in repr_str
        assert "model_type='sarima'" in repr_str
        assert "order=(1, 1, 1)" in repr_str
        assert "seasonal_order=(1, 0, 1, 12)" in repr_str
        assert "verbose=False" in repr_str
        assert "trend='ct'" in repr_str

    def test_edge_cases(self, sample_data):
        """Test edge cases and error handling."""
        X, y = sample_data

        model = TimeSeriesModelSklearn(model_type="ar", order=2)

        # Test predict before fit
        with pytest.raises(Exception):  # Should raise NotFittedError
            model.predict()

        # Test score before fit
        with pytest.raises(Exception):
            model.score()

        # Fit model
        model.fit(X)

        # Test VAR without required X
        var_model = TimeSeriesModelSklearn(model_type="var")
        var_model.fit(multivariate_data())
        with pytest.raises(ValueError, match="X is required"):
            var_model.predict()

    def test_exogenous_variables(self, sample_data):
        """Test models with exogenous variables."""
        X, y = sample_data

        # Test AR with exogenous
        model = TimeSeriesModelSklearn(model_type="ar", order=2)
        model.fit(X, y)

        assert model.y_ is y
        predictions = model.predict()
        assert isinstance(predictions, np.ndarray)

    def test_backend_system(self, sample_data):
        """Test backend system usage."""
        X, y = sample_data

        # Test with backend enabled
        model = TimeSeriesModelSklearn(model_type="ar", order=2, use_backend=True)

        # This might fail if backend not properly configured,
        # but should at least not crash during initialization
        try:
            model.fit(X)
            predictions = model.predict()
            assert isinstance(predictions, np.ndarray)
        except ImportError:
            # Backend might not be available
            pytest.skip("Backend system not available")

    def test_nan_handling(self):
        """Test handling of NaN values in scoring."""
        # Create data with NaNs
        X = np.array([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10])

        model = TimeSeriesModelSklearn(model_type="ar", order=1)

        # Most models should fail with NaN in input
        with pytest.raises(Exception):
            model.fit(X)

    @pytest.mark.parametrize("model_type", ["ar", "arima", "sarima"])
    def test_model_types(self, sample_data, model_type):
        """Test different model types."""
        X, y = sample_data

        if model_type == "sarima":
            model = TimeSeriesModelSklearn(
                model_type=model_type, order=(1, 0, 1), seasonal_order=(1, 0, 1, 12)
            )
        else:
            model = TimeSeriesModelSklearn(
                model_type=model_type, order=2 if model_type == "ar" else (1, 0, 1)
            )

        model.fit(X)
        predictions = model.predict()

        assert isinstance(predictions, np.ndarray)
        assert predictions.ndim == 2
