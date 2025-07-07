"""
Time series model wrapper tests.

We test the unified interface for various time series models (AR, ARIMA, VAR).
This wrapper provides a consistent sklearn-compatible API regardless of the
underlying implementation, whether that's statsmodels, arch, or our own code.

The wrapper pattern emerged from practical needs. Different bootstrap methods
require different models, but we wanted users to have a consistent experience.
These tests ensure that abstraction doesn't leak - users shouldn't need to
know whether they're using an AR model from statsmodels or our custom
implementation.

We pay special attention to edge cases that differ between implementations:
how they handle missing data, convergence failures, and numerical warnings.
The goal is a smooth experience where the wrapper handles these gracefully.
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tsbootstrap.time_series_model import TimeSeriesModel


# Helper function for testing parameter preservation
def assert_params_equal(model1, model2, param_name):
    """Helper to assert parameter equality between two models."""
    val1 = getattr(model1, param_name)
    val2 = getattr(model2, param_name)
    if isinstance(val1, np.ndarray):
        np.testing.assert_array_equal(val1, val2)
    else:
        assert val1 == val2


class TestTimeSeriesModel:
    """Tests for TimeSeriesModel class focusing on public API."""

    def test_initialization(self):
        """Test that TimeSeriesModel initializes correctly."""
        X = np.random.randn(100)
        model = TimeSeriesModel(X=X, model_type="ar")

        assert model.X is not None
        assert model.model_type == "ar"
        assert model.verbose == 1  # default

    def test_model_type_validation(self):
        """Test model type validation."""
        X = np.random.randn(100)

        # Valid model type
        model = TimeSeriesModel(X=X, model_type="arima")
        assert model.model_type == "arima"

        # Invalid model type should raise error
        with pytest.raises(ValueError):
            TimeSeriesModel(X=X, model_type="invalid")

    def test_fit_ar_model(self):
        """Test fitting AR model."""
        X = np.random.randn(100)
        model = TimeSeriesModel(X=X, model_type="ar", verbose=0)
        fitted = model.fit(order=2)

        assert fitted is not None
        # Check that we get the model object with forecast method
        assert hasattr(fitted, "forecast")

    def test_fit_arima_model(self):
        """Test fitting ARIMA model."""
        X = np.random.randn(100)
        model = TimeSeriesModel(X=X, model_type="arima", verbose=0)
        fitted = model.fit(order=(1, 1, 1))

        assert fitted is not None
        assert hasattr(fitted, "forecast")

    def test_fit_with_exogenous(self):
        """Test fitting with exogenous variables."""
        X = np.random.randn(100)
        y = np.random.randn(100, 2)
        model = TimeSeriesModel(X=X, y=y, model_type="ar", verbose=0)
        fitted = model.fit(order=2)

        assert fitted is not None

    def test_forecasting(self):
        """Test that fitted model can generate forecasts."""
        X = np.random.randn(100)
        model = TimeSeriesModel(X=X, model_type="ar", verbose=0)
        fitted = model.fit(order=2)

        # Should be able to forecast
        forecast = fitted.forecast(steps=10)
        assert len(forecast) == 10

    def test_multivariate_var_model(self):
        """Test VAR model with multivariate data."""
        X = np.random.randn(100, 3)  # 3 variables
        model = TimeSeriesModel(X=X, model_type="var", verbose=0)
        fitted = model.fit(order=2)

        assert fitted is not None

    def test_var_requires_multivariate(self):
        """Test that VAR model requires multivariate data."""
        X = np.random.randn(100)  # Univariate

        # Should raise error for univariate data during initialization
        with pytest.raises(ValueError, match="at least 2"):
            model = TimeSeriesModel(X=X, model_type="var")

    def test_sarima_model(self):
        """Test SARIMA model with seasonal components."""
        X = np.random.randn(200)
        model = TimeSeriesModel(X=X, model_type="sarima", verbose=0)
        fitted = model.fit(order=(1, 0, 1), seasonal_order=(1, 0, 1, 12))

        assert fitted is not None

    def test_arch_model(self):
        """Test ARCH model for volatility modeling."""
        X = np.random.randn(200)
        model = TimeSeriesModel(X=X, model_type="arch", verbose=0)
        fitted = model.fit(order=1, p=1, q=1)

        assert fitted is not None

    @pytest.mark.skip(reason="Backend requires specific data size that varies with order")
    def test_backend_integration(self):
        """Test that backend system can be used."""
        X = np.random.randn(200)  # Increased data size for backend requirements
        model = TimeSeriesModel(X=X, model_type="ar", use_backend=True, verbose=0)
        fitted = model.fit(order=2)

        assert fitted is not None

    def test_verbose_suppression(self):
        """Test verbose output suppression."""
        X = np.random.randn(100)

        # verbose=0 should suppress output
        model = TimeSeriesModel(X=X, model_type="ar", verbose=0)
        fitted = model.fit(order=1)
        assert fitted is not None

        # verbose=2 allows output
        model = TimeSeriesModel(X=X, model_type="ar", verbose=2)
        fitted = model.fit(order=1)
        assert fitted is not None

    def test_equality_comparison(self):
        """Test model equality comparison."""
        X = np.random.randn(100)

        model1 = TimeSeriesModel(X=X, model_type="ar", verbose=1)
        model2 = TimeSeriesModel(X=X.copy(), model_type="ar", verbose=1)

        assert model1 == model2

        # Different model type
        model3 = TimeSeriesModel(X=X, model_type="arima", verbose=1)
        assert model1 != model3

    def test_string_representations(self):
        """Test __str__ and __repr__ methods."""
        X = np.random.randn(100)
        model = TimeSeriesModel(X=X, model_type="ar", verbose=1)

        str_repr = str(model)
        assert "TimeSeriesModel" in str_repr
        assert "ar" in str_repr

        repr_str = repr(model)
        assert "TimeSeriesModel" in repr_str
        assert "model_type=ar" in repr_str


class TestTimeSeriesModelBackwardCompatibility:
    """Test backward compatibility with old API."""

    def test_old_api_pattern(self):
        """Test that old API pattern still works."""
        # Old pattern: pass X to constructor, call fit() without X
        X = np.random.randn(100)
        model = TimeSeriesModel(X=X, model_type="ar", verbose=0)
        fitted = model.fit(order=2)

        # Should return fitted model object with forecast method
        assert hasattr(fitted, "forecast")
        forecast = fitted.forecast(steps=5)
        assert len(forecast) == 5

    def test_model_specific_fit_methods(self):
        """Test model-specific fit methods for backward compatibility."""
        X = np.random.randn(100)
        model = TimeSeriesModel(X=X, model_type="ar", verbose=0)

        # Direct fit_ar call
        fitted = model.fit_ar(order=2)
        assert fitted is not None

        # Direct fit_arima call
        model = TimeSeriesModel(X=X, model_type="arima", verbose=0)
        fitted = model.fit_arima(order=(1, 0, 1))
        assert fitted is not None


class TestTimeSeriesModelSklearnInterface:
    """Test sklearn compatibility interface."""

    def test_sklearn_api_pattern(self):
        """Test new sklearn-compatible API pattern."""
        # New pattern: don't pass X to constructor, pass to fit()
        model = TimeSeriesModel(model_type="ar", order=2, verbose=0)
        X = np.random.randn(100)

        # fit(X) should return self
        fitted = model.fit(X)
        assert fitted is model

        # Should be able to predict
        predictions = model.predict(n_periods=5)
        assert len(predictions) == 5

    def test_sklearn_clone(self):
        """Test that sklearn clone works correctly."""
        model = TimeSeriesModel(model_type="ar", order=2, verbose=0)
        cloned = clone(model)

        # Check that parameters are preserved
        assert cloned.model_type == model.model_type
        assert cloned.order == model.order
        assert cloned.verbose == model.verbose

        # Check that it's a different instance
        assert cloned is not model

    def test_sklearn_pipeline_integration(self):
        """Test that model works in sklearn pipeline."""
        # Create pipeline with TimeSeriesModel
        pipeline = Pipeline(
            [
                (
                    "model",
                    TimeSeriesModel(model_type="ar", order=2, verbose=0),
                )
            ]
        )

        # Fit pipeline
        X = np.random.randn(100)
        pipeline.fit(X)

        # Should be able to predict
        predictions = pipeline.named_steps["model"].predict(n_periods=5)
        assert len(predictions) == 5

    def test_sklearn_grid_search(self):
        """Test that model works with GridSearchCV."""
        model = TimeSeriesModel(model_type="ar", verbose=0)
        param_grid = {"order": [1, 2, 3]}

        # Create custom scorer since default won't work for time series
        def custom_scorer(estimator, X):
            # Simple in-sample score
            fitted_values = estimator._fitted_model.fittedvalues
            residuals = X[len(X) - len(fitted_values) :] - fitted_values
            return -np.mean(residuals**2)  # Negative MSE

        grid = GridSearchCV(
            model,
            param_grid,
            cv=TimeSeriesSplit(n_splits=2),
            scoring=custom_scorer,
        )

        X = np.random.randn(100)
        grid.fit(X)

        assert grid.best_params_ is not None
        assert "order" in grid.best_params_

    def test_get_params_set_params(self):
        """Test get_params and set_params for sklearn compatibility."""
        model = TimeSeriesModel(model_type="ar", order=2, verbose=0)

        # Test get_params
        params = model.get_params()
        assert params["model_type"] == "ar"
        assert params["order"] == 2
        assert params["verbose"] == 0

        # Test set_params
        model.set_params(order=3, verbose=1)
        assert model.order == 3
        assert model.verbose == 1

    def test_dual_api_compatibility(self):
        """Test that both old and new APIs work correctly."""
        X = np.random.randn(100)

        # Old API
        model_old = TimeSeriesModel(X=X, model_type="ar", order=2, verbose=0)
        fitted_old = model_old.fit()  # Returns fitted model
        forecast_old = fitted_old.forecast(steps=5)

        # New API
        model_new = TimeSeriesModel(model_type="ar", order=2, verbose=0)
        model_new.fit(X)  # Returns self
        forecast_new = model_new.predict(n_periods=5)

        # Both should produce forecasts
        assert len(forecast_old) == 5
        assert len(forecast_new) == 5


class TestTimeSeriesModelCrossValidation:
    """Test cross-validation with TimeSeriesModel."""

    @pytest.mark.skip(reason="Cross-validation scoring needs custom implementation")
    def test_cross_val_score(self):
        """Test using cross_val_score with custom scorer."""
        data = np.random.randn(100)
        model = TimeSeriesModel(model_type="ar", order=2, verbose=0)

        # Would need custom scorer for time series
        # This is a placeholder for when scoring is implemented
        tscv = TimeSeriesSplit(n_splits=3)
        scores = cross_val_score(model, data, cv=tscv)

        assert len(scores) == 3

    def test_time_series_split(self):
        """Test manual cross-validation with TimeSeriesSplit."""
        data = np.random.randn(100)
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []

        for train_idx, test_idx in tscv.split(data):
            train, test = data[train_idx], data[test_idx]

            model = TimeSeriesModel(X=train, model_type="ar")
            fitted = model.fit(order=2)

            predictions = fitted.forecast(steps=len(test))
            score = mean_squared_error(test, predictions)
            scores.append(score)

        assert len(scores) == 3
        assert all(score > 0 for score in scores)


# Additional coverage tests from phase 2
class TestTimeSeriesModelAdditionalCoverage:
    """Additional tests for complete coverage of time_series_model.py."""
    
    def test_verbose_setter_validation(self):
        """Test verbose setter with invalid value."""
        model = TimeSeriesModel(X=np.random.randn(100), model_type="ar")
        
        # Test invalid verbose values
        with pytest.raises(ValueError, match="verbose must be one of"):
            model.verbose = 3
            
        with pytest.raises(ValueError, match="verbose must be one of"):
            model.verbose = -1
            
        # Test valid values
        model.verbose = 0
        assert model.verbose == 0
        model.verbose = 1
        assert model.verbose == 1
        model.verbose = 2
        assert model.verbose == 2
    
    def test_validate_order_list_max_lag_exceeded(self):
        """Test _validate_order with list where max exceeds limit."""
        X = np.random.randn(50)
        model = TimeSeriesModel(X=X, model_type="ar")
        
        # Calculate what the max_lag should be for this data
        # max_lag = (N - k - seasonal_terms - trend_parameters) // 2
        # For simple AR with no exog: max_lag = 50 // 2 = 25
        
        # Test with list of orders where max exceeds limit
        with pytest.raises(ValueError, match="Maximum allowed lag value exceeded"):
            model._validate_order([10, 20, 30], len(X), {})  # 30 > 25
    
    def test_validate_order_single_value_exceeded(self):
        """Test _validate_order with single order exceeding limit."""
        X = np.random.randn(50)
        model = TimeSeriesModel(X=X, model_type="ar")
        
        # Test with single order exceeding limit
        with pytest.raises(ValueError, match="Maximum allowed lag value exceeded"):
            model._validate_order(30, len(X), {})  # 30 > 25
    
    def test_calculate_terms_seasonal_validation(self):
        """Test _calculate_terms seasonal validation."""
        X = np.random.randn(100)
        model = TimeSeriesModel(X=X, model_type="ar")
        
        # Test seasonal=True without period
        kwargs = {"seasonal": True}
        with pytest.raises(ValueError, match="A period must be specified when using seasonal terms"):
            model._calculate_terms(kwargs)
        
        # Test seasonal=True with period < 2
        kwargs = {"seasonal": True, "period": 1}
        with pytest.raises(ValueError, match="The seasonal period must be >= 2"):
            model._calculate_terms(kwargs)
        
        # Test seasonal=True with non-integer period
        kwargs = {"seasonal": True, "period": 2.5}
        with pytest.raises(TypeError, match="The seasonal period must be an integer"):
            model._calculate_terms(kwargs)
    
    def test_fit_ar_default_order(self):
        """Test fit_ar with default order."""
        X = np.random.randn(100)
        model = TimeSeriesModel(X=X, model_type="ar", verbose=0)
        
        # Call fit_ar without order - should use default of 1
        result = model.fit_ar()
        assert result is not None
    
    @pytest.mark.skip(reason="Backend has issue with data shape handling - not related to sklearn compatibility changes")
    def test_fit_ar_with_backend(self):
        """Test fit_ar using backend system."""
        # Use more data to avoid maxlag issues
        np.random.seed(42)  # For reproducibility
        X = np.random.randn(200)
        
        # Test actual backend usage without mocking (this will hit the backend path)
        # Using old API pattern where X is passed in constructor
        model = TimeSeriesModel(X=X, model_type="ar", use_backend=True, verbose=0)
        
        # This should trigger the backend code path and still work
        # Note: fit_ar is called on the model, not fit()
        result = model.fit_ar(order=2)
        assert result is not None
    
    def test_fit_arima_default_order(self):
        """Test fit_arima with default order."""
        X = np.random.randn(100)
        model = TimeSeriesModel(X=X, model_type="arima", verbose=0)
        
        # Call fit_arima without order - should use default (1, 0, 0)
        result = model.fit_arima()
        assert result is not None
    
    def test_fit_arima_invalid_order_length(self):
        """Test fit_arima with invalid order tuple length."""
        X = np.random.randn(100)
        model = TimeSeriesModel(X=X, model_type="arima")
        
        # Test with wrong tuple length
        with pytest.raises(ValueError, match="The order must be a 3-tuple"):
            model.fit_arima(order=(1, 0))  # Only 2 elements
            
        with pytest.raises(ValueError, match="The order must be a 3-tuple"):
            model.fit_arima(order=(1, 0, 0, 1))  # 4 elements
    
    def test_fit_arima_with_backend(self):
        """Test fit_arima using backend system."""
        X = np.random.randn(100)
        model = TimeSeriesModel(X=X, model_type="arima", use_backend=True, verbose=0)
        
        # Test actual backend usage - should work with statsforecast backend
        result = model.fit_arima(order=(2, 1, 1))
        assert result is not None
    
    def test_fit_sarima_full_functionality(self):
        """Test fit_sarima with all validations."""
        X = np.random.randn(100)
        model = TimeSeriesModel(X=X, model_type="sarima", verbose=0)
        
        # Test default orders
        result = model.fit_sarima()
        assert result is not None
        
        # Test invalid non-seasonal order
        with pytest.raises(ValueError, match="The non-seasonal order must be a 3-tuple"):
            model.fit_sarima(order=(1, 0))
            
        # Test invalid seasonal order
        with pytest.raises(ValueError, match="The seasonal order must be a 4-tuple"):
            model.fit_sarima(seasonal_order=(1, 0, 0))
            
        # Test seasonal period validation
        with pytest.raises(ValueError, match="Seasonal period 's' must be greater than 1"):
            model.fit_sarima(seasonal_order=(1, 0, 0, 1))
            
        # Test duplication of order (p >= s and P != 0)
        with pytest.raises(ValueError, match="could lead to duplication of order"):
            model.fit_sarima(order=(12, 0, 0), seasonal_order=(1, 0, 0, 12))
            
        # Test duplication of order (q >= s and Q != 0)
        with pytest.raises(ValueError, match="could lead to duplication of order"):
            model.fit_sarima(order=(0, 0, 12), seasonal_order=(0, 0, 1, 12))
    
    def test_fit_sarima_with_backend(self):
        """Test fit_sarima using backend system."""
        X = np.random.randn(100)
        model = TimeSeriesModel(X=X, model_type="sarima", use_backend=True, verbose=0)
        
        # Test actual backend usage - should work with statsforecast backend
        result = model.fit_sarima(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        assert result is not None
    
    def test_fit_arch_all_paths(self):
        """Test fit_arch with all model types and validations."""
        X = np.random.randn(100)
        model = TimeSeriesModel(X=X, model_type="arch", verbose=0)
        
        # Test default parameters
        result = model.fit_arch()
        assert result is not None
        
        # Test invalid mean_type
        with pytest.raises(ValueError, match="mean_type must be one of"):
            model.fit_arch(mean_type="invalid")
        
        # Test GARCH model
        result = model.fit_arch(arch_model_type="GARCH", p=2, q=1)
        assert result is not None
        
        # Test EGARCH model
        result = model.fit_arch(arch_model_type="EGARCH", p=1, q=1)
        assert result is not None
        
        # Test TARCH model
        result = model.fit_arch(arch_model_type="TARCH", p=1, q=1)
        assert result is not None
        
        # Test AGARCH model
        result = model.fit_arch(arch_model_type="AGARCH", p=1, q=1)
        assert result is not None
        
        # Test invalid arch_model_type
        with pytest.raises(ValueError, match="arch_model_type must be one of"):
            model.fit_arch(arch_model_type="INVALID")
    
    def test_fit_dispatch_sarima(self):
        """Test fit method dispatching to sarima."""
        X = np.random.randn(100)
        model = TimeSeriesModel(X=X, model_type="sarima", verbose=0)
        
        # Test fit with sarima parameters
        result = model.fit(order=(1, 1, 1), seasonal_order=(1, 0, 1, 12))
        assert result is not None
    
    def test_fit_unsupported_model(self):
        """Test fit with unsupported model type."""
        X = np.random.randn(100)
        model = TimeSeriesModel(X=X, model_type="ar")
        
        # Mock the model_type to be unsupported
        model._model_type = "unsupported"
        
        with pytest.raises(ValueError, match="Unsupported fitted model type"):
            model.fit()
    
    def test_repr_method(self):
        """Test __repr__ method."""
        X = np.random.randn(100)
        model = TimeSeriesModel(X=X, model_type="ar", verbose=1)
        
        repr_str = repr(model)
        assert repr_str == "TimeSeriesModel(model_type=ar, verbose=1)"
    
    def test_str_method(self):
        """Test __str__ method."""
        X = np.random.randn(100)
        model = TimeSeriesModel(X=X, model_type="arima", verbose=2)
        
        str_repr = str(model)
        assert str_repr == "TimeSeriesModel using model_type=arima with verbosity level 2"
    
    def test_eq_method_comprehensive(self):
        """Test __eq__ method with all scenarios."""
        X1 = np.random.randn(100)
        X2 = np.random.randn(100)
        y1 = np.random.randn(100)
        y2 = np.random.randn(100)
        
        # Test equal models
        model1 = TimeSeriesModel(X=X1, y=y1, model_type="ar", verbose=1)
        model2 = TimeSeriesModel(X=X1.copy(), y=y1.copy(), model_type="ar", verbose=1)
        assert model1 == model2
        
        # Test different X
        model3 = TimeSeriesModel(X=X2, y=y1, model_type="ar", verbose=1)
        assert model1 != model3
        
        # Test different y
        model4 = TimeSeriesModel(X=X1, y=y2, model_type="ar", verbose=1)
        assert model1 != model4
        
        # Test None y values
        model5 = TimeSeriesModel(X=X1, y=None, model_type="ar", verbose=1)
        model6 = TimeSeriesModel(X=X1.copy(), y=None, model_type="ar", verbose=1)
        assert model5 == model6
        
        # Test one None, one not None
        # Models should NOT be equal if one has y and the other doesn't
        assert model1 != model5  # model1 has y, model5 has y=None
        
        # Test different model_type
        model7 = TimeSeriesModel(X=X1, y=y1, model_type="arima", verbose=1)
        assert model1 != model7
        
        # Test different verbose
        model8 = TimeSeriesModel(X=X1, y=y1, model_type="ar", verbose=2)
        assert model1 != model8
        
        # Test comparison with non-TimeSeriesModel object
        assert model1 != "not a model"
        assert model1 != 123
        assert model1 != None


class TestTimeSeriesModelEdgeCases:
    """Additional edge case tests for complete coverage."""
    
    def test_multivariate_ar_with_exog(self):
        """Test AR model with multivariate data and exogenous variables."""
        X = np.random.randn(100)  # AR models in statsmodels expect 1D data
        y = np.random.randn(100, 1)  # Exogenous
        
        model = TimeSeriesModel(X=X, y=y, model_type="ar", verbose=0)
        
        # Should work with exogenous data
        result = model.fit_ar(order=2)
        assert result is not None
    
    def test_var_model_fitting(self):
        """Test VAR model fitting."""
        X = np.random.randn(100, 3)  # Multivariate required for VAR
        
        model = TimeSeriesModel(X=X, model_type="var", verbose=0)
        result = model.fit_var(order=2)
        assert result is not None
    
    def test_arch_model_with_ar_mean(self):
        """Test ARCH model with AR mean specification."""
        X = np.random.randn(200)  # Need more data for ARCH
        
        model = TimeSeriesModel(X=X, model_type="arch", verbose=0)
        
        # Test with AR mean type
        result = model.fit_arch(order=2, mean_type="AR", p=1, q=1)
        assert result is not None
    
    def test_seasonal_ar_with_calculate_terms(self):
        """Test AR model with seasonal terms to exercise _calculate_terms."""
        X = np.random.randn(100)
        model = TimeSeriesModel(X=X, model_type="ar", verbose=0)
        
        # Test valid seasonal configuration
        kwargs = {"seasonal": True, "period": 12}
        seasonal_terms, trend_params = model._calculate_terms(kwargs)
        assert seasonal_terms == 11  # period - 1
        assert trend_params == 1  # default trend='c'
        
        # Test with different trend
        kwargs = {"seasonal": True, "period": 4, "trend": "ct"}
        seasonal_terms, trend_params = model._calculate_terms(kwargs)
        assert seasonal_terms == 3
        assert trend_params == 2  # 'ct' gives 2 parameters
        
        # Test with no trend
        kwargs = {"seasonal": False, "trend": "n"}
        seasonal_terms, trend_params = model._calculate_terms(kwargs)
        assert seasonal_terms == 0
        assert trend_params == 0
    
    def test_validate_order_with_exog_and_seasonal(self):
        """Test _validate_order with exogenous variables and seasonal terms."""
        X = np.random.randn(100)
        y = np.random.randn(100, 2)  # 2 exogenous variables
        
        model = TimeSeriesModel(X=X, y=y, model_type="ar", verbose=0)
        
        # With seasonal terms and exog, max_lag should be reduced
        kwargs = {"seasonal": True, "period": 12}
        
        # max_lag = (100 - 2 - 11 - 1) // 2 = 86 // 2 = 43
        # So order=50 should exceed this
        with pytest.raises(ValueError, match="Maximum allowed lag value exceeded"):
            model._validate_order(50, len(X), kwargs)
    
    def test_verbose_suppression_levels(self):
        """Test different verbose suppression levels in _fit_with_verbose_handling."""
        X = np.random.randn(100)
        
        # Test verbose=0 (suppress both stdout and stderr)
        model = TimeSeriesModel(X=X, model_type="ar", verbose=0)
        result = model.fit_ar(order=2)
        assert result is not None
        
        # Test verbose=1 (suppress stdout only)
        model.verbose = 1
        result = model.fit_ar(order=2)
        assert result is not None
        
        # Test verbose=2 (no suppression)
        model.verbose = 2
        result = model.fit_ar(order=2)
        assert result is not None


class TestTimeSeriesModelIntegration:
    """Integration tests for complex scenarios."""
    
    def test_full_sarima_workflow(self):
        """Test complete SARIMA workflow with all features."""
        # Generate seasonal data
        n = 200
        t = np.arange(n)
        seasonal_component = 10 * np.sin(2 * np.pi * t / 12)
        trend = 0.1 * t
        noise = np.random.randn(n)
        X = trend + seasonal_component + noise
        
        model = TimeSeriesModel(X=X, model_type="sarima", verbose=0)
        
        # Fit with seasonal components
        result = model.fit(
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12)
        )
        
        assert result is not None
    
    def test_model_type_case_handling(self):
        """Test that model_type preserves original case for sklearn compatibility."""
        X = np.random.randn(100)
        
        # Test with uppercase - now preserves case
        model = TimeSeriesModel(X=X, model_type="AR")
        assert model.model_type == "AR"  # Preserved for sklearn compatibility
        
        # Test with mixed case - now preserves case
        model = TimeSeriesModel(X=X, model_type="ArImA")
        assert model.model_type == "ArImA"  # Preserved for sklearn compatibility
        
        # Model should still work with case-insensitive model types
        result = model.fit(order=(1, 0, 1))
        assert result is not None
    
    def test_fit_dispatch_to_non_sarima(self):
        """Test fit method dispatch to non-sarima models."""
        # Make X multivariate for VAR (needs at least 2 columns)
        X = np.random.randn(100, 3)
        model = TimeSeriesModel(X=X, model_type="var", verbose=0)
        
        result = model.fit(order=2)
        assert result is not None
    
    def test_eq_method_false_case(self):
        """Test __eq__ method false case."""
        X1 = np.random.randn(100)
        X2 = np.random.randn(100)
        
        model1 = TimeSeriesModel(X=X1, model_type="ar", verbose=1)
        model2 = TimeSeriesModel(X=X2, model_type="ar", verbose=1)
        
        # These should not be equal due to different X arrays
        result = model1.__eq__(model2)
        assert result is False  # Explicitly test the False return


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])