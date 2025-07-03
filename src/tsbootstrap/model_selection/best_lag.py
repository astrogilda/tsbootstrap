"""
Automatic lag selection: Data-driven model order determination for time series.

This module implements sophisticated algorithms for automatically determining
optimal lag orders in time series models. The challenge of lag selection
represents a fundamental bias-variance tradeoff: too few lags miss important
dynamics, while too many lags lead to overfitting and poor out-of-sample
performance.

We've designed this module around the RankLags algorithm, which evaluates
multiple lag configurations using information criteria and cross-validation.
This data-driven approach removes the guesswork from model specification,
automatically identifying the lag structure that best captures the temporal
dependencies in your data.

The implementation seamlessly integrates with our backend system, supporting
automatic order selection across various model families including AR, ARIMA,
VAR, and ARCH models. This unified interface simplifies the model selection
workflow while maintaining the flexibility to override automatic choices when
domain knowledge suggests specific lag structures.
"""

from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

# Import model result types
from statsmodels.tsa.ar_model import AutoRegResultsWrapper
from statsmodels.tsa.arima.model import ARIMAResultsWrapper
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

from tsbootstrap.backends.adapter import fit_with_backend
from tsbootstrap.ranklags import RankLags
from tsbootstrap.utils.types import (
    ModelTypes,
    OrderTypes,
    OrderTypesWithoutNone,
)

try:
    from arch.univariate.base import ARCHModelResult
except ImportError:
    ARCHModelResult = None  # type: ignore


class TSFitBestLag(BaseEstimator, RegressorMixin):
    """
    Intelligent lag order selection with integrated model fitting.

    This class implements an automated workflow for time series modeling that
    removes the burden of manual lag specification. We combine sophisticated
    lag ranking algorithms with seamless model fitting, providing a single
    interface that handles the complete model selection and estimation process.

    The core innovation is the integration of the RankLags algorithm, which
    systematically evaluates different lag configurations using multiple
    criteria. This data-driven approach ensures that the selected model
    complexity matches the inherent structure of your time series, avoiding
    both underfitting and overfitting.

    Our implementation supports the full spectrum of time series models, from
    simple autoregressive models to complex seasonal specifications. The class
    automatically adapts its selection strategy based on the model type,
    applying appropriate constraints and search spaces for each model family.

    Parameters
    ----------
    model_type : ModelTypes
        The family of time series models to consider. Options include 'ar'
        for pure autoregressive, 'arima' for integrated models, 'sarima'
        for seasonal patterns, 'var' for multivariate dynamics, and 'arch'
        for volatility modeling.

    max_lag : int, default=10
        Upper bound for lag order search. This parameter controls the
        computational complexity and maximum model flexibility. Larger values
        allow capturing longer dependencies but increase estimation time.

    order : OrderTypes, optional
        Explicit model order specification. When provided, bypasses automatic
        selection. Use this when domain knowledge suggests specific lag
        structures or to reproduce previous analyses.

    seasonal_order : tuple, optional
        Seasonal specification for SARIMA models in format (P, D, Q, s).
        Required for seasonal models where s is the seasonal period.

    save_models : bool, default=False
        Whether to retain all candidate models evaluated during selection.
        Useful for model comparison and diagnostic analysis but increases
        memory usage.

    **kwargs
        Additional parameters passed to the underlying model estimators.
        Additional parameters passed to the model
    """

    def __init__(
        self,
        model_type: ModelTypes,
        max_lag: int = 10,
        order: OrderTypes = None,  # Can be None initially
        seasonal_order: Optional[tuple] = None,
        save_models=False,
        **kwargs,
    ):
        self.model_type = model_type
        self.max_lag = max_lag
        self.order: Union[
            OrderTypesWithoutNone, None
        ] = order  # Allow None initially, will be set in fit
        self.seasonal_order: Optional[tuple] = seasonal_order
        self.save_models = save_models
        self.model_params = kwargs
        self.rank_lagger: Optional[RankLags] = None
        self.fitted_adapter = None
        self.model: Union[
            AutoRegResultsWrapper,
            ARIMAResultsWrapper,
            SARIMAXResultsWrapper,
            VARResultsWrapper,
            ARCHModelResult,
            None,
        ] = None
        self.rescale_factors: dict = {}

    def _compute_best_order(self, X: np.ndarray) -> Union[OrderTypesWithoutNone, tuple]:
        # Ensure X is 2D for RankLags
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.rank_lagger = RankLags(
            X=X,
            max_lag=self.max_lag,
            model_type=self.model_type,
            save_models=self.save_models,  # Pass save_models to RankLags
        )
        # estimate_conservative_lag returns int, but TSFit order can be more complex
        # For now, assume RankLags gives an appropriate int order for non-ARIMA/SARIMA
        # or that this will be handled/overridden if self.order is explicitly set.
        best_lag_int = self.rank_lagger.estimate_conservative_lag()

        # Convert integer lag to appropriate tuple for ARIMA/SARIMA if needed by TSFit
        if self.model_type == "arima":
            return (best_lag_int, 0, 0)
        elif self.model_type == "sarima":
            # For SARIMA, _compute_best_order only determines the non-seasonal AR order (p)
            # The seasonal order (P, D, Q, s) should be passed separately or default.
            # Here, we return the non-seasonal order, and seasonal_order will be handled by TSFit.
            return (best_lag_int, 0, 0)  # Return non-seasonal order
        return best_lag_int

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        # Store original data shape for later use
        self._original_X_shape = X.shape

        if self.order is None:
            self.order = self._compute_best_order(X)

        if self.order is None:  # Should be set by _compute_best_order
            raise ValueError(
                "Failed to determine model order automatically. This can occur when the lag selection "
                "algorithm cannot find a suitable order within the specified max_lag range. Consider "
                "increasing max_lag or providing an explicit order parameter."
            )

        # Prepare data for backend
        if self.model_type == "var":
            # VAR needs multivariate data
            if X.ndim == 1:
                raise ValueError(
                    "VAR (Vector Autoregression) models require multivariate time series data with "
                    "at least 2 variables to capture cross-series dynamics. Received univariate data. "
                    "For single time series analysis, use AR, ARIMA, or SARIMA models instead."
                )
            endog = X.T  # Backend expects (n_vars, n_obs) for VAR
        else:
            # For univariate models
            if X.ndim == 2:
                if X.shape[1] == 1:
                    endog = X.flatten()
                else:
                    # For univariate models, reject multivariate data
                    raise ValueError(
                        f"Univariate models (AR, ARIMA, SARIMA) require single time series data. "
                        f"Received multivariate data with {X.shape[1]} columns. "
                        f"Either select a single column or use VAR models for multivariate analysis."
                    )
            else:
                endog = X

        # Fit using backend
        fitted_adapter = fit_with_backend(
            model_type=self.model_type,
            endog=endog,
            exog=y,
            order=self.order,
            seasonal_order=self.seasonal_order,
            force_backend="statsmodels",  # Use statsmodels for stability
            return_backend=False,  # Get adapter for compatibility
            **self.model_params,
        )

        # Store the fitted model and adapter
        self.fitted_adapter = fitted_adapter
        # Get the underlying statsmodels model from the backend
        if hasattr(fitted_adapter, "_backend") and hasattr(
            fitted_adapter._backend, "_fitted_models"
        ):
            # For adapter, get the first fitted model
            self.model = fitted_adapter._backend._fitted_models[0]
        else:
            # Fallback to the adapter itself
            self.model = fitted_adapter

        # Get fitted values and residuals
        fitted_values = fitted_adapter.fitted_values
        residuals = fitted_adapter.residuals

        # Ensure 2D shape for compatibility
        if fitted_values.ndim == 1:
            fitted_values = fitted_values.reshape(-1, 1)
        if residuals.ndim == 1:
            residuals = residuals.reshape(-1, 1)

        self.X_fitted_ = fitted_values
        self.resids_ = residuals

        # Store rescale factors if available
        if hasattr(fitted_adapter, "rescale_factors"):
            self.rescale_factors = fitted_adapter.rescale_factors
        else:
            self.rescale_factors = None

        return self

    def get_coefs(self) -> np.ndarray:
        check_is_fitted(self, "model")
        if self.model is None:
            raise NotFittedError(
                "Model has not been fitted yet. The get_coefs() method requires a fitted model "
                "to extract coefficient values. Call fit() with your time series data first."
            )
        # Get coefficients from the underlying model
        if hasattr(self.model, "params"):
            params = self.model.params
            # If params is a dict (from BackendToStatsmodelsAdapter), extract AR coefficients
            if isinstance(params, dict):
                # Extract AR coefficients
                ar_coeffs = []
                for key in sorted(params.keys()):
                    if key.startswith("ar.L"):
                        ar_coeffs.append(params[key])
                return np.array(ar_coeffs) if ar_coeffs else np.array([])
            return params
        elif hasattr(self.model, "coef_"):
            return self.model.coef_
        else:
            raise AttributeError("Model does not have coefficients.")

    def get_intercepts(self) -> np.ndarray:
        check_is_fitted(self, "model")
        if self.model is None:
            raise NotFittedError(
                "Model has not been fitted yet. The get_intercepts() method requires a fitted model "
                "to extract intercept values. Call fit() with your time series data first."
            )
        # Get intercept from the underlying model
        if hasattr(self.model, "const"):
            return np.array([self.model.const])
        elif hasattr(self.model, "intercept_"):
            return np.array([self.model.intercept_])
        else:
            return np.array([0.0])  # Default if no intercept

    def get_residuals(self) -> np.ndarray:
        check_is_fitted(self, "fitted_adapter")
        if self.fitted_adapter is None:
            raise NotFittedError(
                "Model has not been fitted yet. The get_residuals() method requires a fitted model "
                "to extract residual values. Call fit() with your time series data first."
            )
        return self.resids_

    def get_fitted_X(self) -> np.ndarray:
        check_is_fitted(self, "fitted_adapter")
        if self.fitted_adapter is None:
            raise NotFittedError(
                "Model has not been fitted yet. The get_fitted_X() method requires a fitted model "
                "to return the fitted values. Call fit() with your time series data first."
            )
        return self.X_fitted_

    def get_order(self) -> OrderTypesWithoutNone:
        check_is_fitted(self, "order")
        if self.order is None:
            raise NotFittedError(
                "Model order has not been determined yet. The get_order() method requires either "
                "a fitted model (which determines optimal order) or an explicitly specified order. "
                "Call fit() with your time series data first."
            )
        return self.order

    def get_model(self):  # Returns the fitted model instance
        check_is_fitted(self, "model")
        if self.model is None:
            raise NotFittedError(
                "Model has not been fitted yet. The get_model() method requires a fitted model "
                "instance to return. Call fit() with your time series data first."
            )
        return self.model

    def predict(self, X: np.ndarray, y: Optional[np.ndarray] = None, n_steps: int = 1):
        check_is_fitted(self, "fitted_adapter")
        if self.fitted_adapter is None:
            raise NotFittedError(
                "Model has not been fitted yet. The predict() method requires a fitted model "
                "to generate forecasts. Call fit() with your time series data first."
            )
        # Use the fitted adapter's predict method
        # Note: Most backends expect steps parameter, not X for predict
        return self.fitted_adapter.predict(steps=n_steps, X=X if self.model_type == "var" else None)

    def score(
        self,
        X: NDArray,  # Changed np.ndarray to NDArray
        y: NDArray,  # Changed np.ndarray to NDArray
        sample_weight: Optional[NDArray] = None,  # Changed np.ndarray to NDArray
    ) -> float:
        check_is_fitted(self, "fitted_adapter")
        if self.fitted_adapter is None:
            raise NotFittedError(
                "Model has not been fitted yet. The score() method requires a fitted model "
                "to evaluate performance metrics. Call fit() with your time series data first."
            )
        # Use the fitted adapter's score method
        return self.fitted_adapter.score(X, y)

    def __repr__(self, N_CHAR_MAX=700) -> str:
        params_str = ", ".join(f"{k!r}={v!r}" for k, v in self.model_params.items())
        return f"{self.__class__.__name__}(model_type={self.model_type!r}, order={self.order!r}, seasonal_order={self.seasonal_order!r}, max_lag={self.max_lag!r}, save_models={self.save_models!r}, model_params={{{params_str}}})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__} using model_type='{self.model_type}' with order={self.order}, seasonal_order={self.seasonal_order}, max_lag={self.max_lag}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TSFitBestLag):
            return False
        return (
            self.model_type == other.model_type
            and self.order == other.order
            and self.seasonal_order == other.seasonal_order  # Added seasonal_order
            and self.max_lag == other.max_lag
            and self.save_models == other.save_models
            and self.rescale_factors == other.rescale_factors
            and self.model_params == other.model_params
            # Model comparison can be tricky. For now, check if both are None or same type.
            # A deeper comparison might involve checking model parameters if available and consistent.
            and (
                (self.model is None and other.model is None)
                or (
                    self.model is not None
                    and other.model is not None
                    and isinstance(other.model, type(self.model))
                )
            )
        )
