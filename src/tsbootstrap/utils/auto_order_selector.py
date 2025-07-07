"""
Automatic lag selection: Data-driven model order determination for time series.

This module implements sophisticated algorithms for automatically determining
optimal lag orders in time series models. The challenge of lag selection
represents a fundamental bias-variance tradeoff: too few lags miss important
dynamics, while too many lags lead to overfitting and poor out-of-sample
performance.

We've designed this module around the AutoOrderSelector class, which evaluates
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

__all__ = ["AutoOrderSelector"]


class AutoOrderSelector(BaseEstimator, RegressorMixin):
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

    For advanced automatic model selection, we support StatsForecast's Auto
    models including AutoARIMA, AutoETS, AutoTheta, and AutoCES. These models
    use sophisticated algorithms to automatically determine the best model
    specification without requiring explicit order parameters.

    Parameters
    ----------
    model_type : ModelTypes | str
        The family of time series models to consider. Options include:
        - Traditional models: 'ar', 'arima', 'sarima', 'var', 'arch'
        - Auto models: 'autoarima' (or 'arima' with use_auto=True),
          'autoets', 'autotheta', 'autoces'

    max_lag : int, default=10
        Upper bound for lag order search. This parameter controls the
        computational complexity and maximum model flexibility. Larger values
        allow capturing longer dependencies but increase estimation time.
        For Auto models, this sets the maximum p and q parameters.

    order : OrderTypes, optional
        Explicit model order specification. When provided, bypasses automatic
        selection. Use this when domain knowledge suggests specific lag
        structures or to reproduce previous analyses. Not applicable for
        Auto models like AutoETS, AutoTheta, AutoCES.

    seasonal_order : tuple, optional
        Seasonal specification for SARIMA models in format (P, D, Q, s).
        Required for seasonal models where s is the seasonal period.

    information_criterion : str, default="aic"
        Information criterion for model selection. Options include 'aic', 'bic', 'hqic'.
        Used by automatic order selection algorithms to evaluate model quality.

    save_models : bool, default=False
        Whether to retain all candidate models evaluated during selection.
        Useful for model comparison and diagnostic analysis but increases
        memory usage.

    use_auto : bool, default=True
        For ARIMA/SARIMA models, whether to use AutoARIMA for automatic
        order selection. If False, uses traditional RankLags approach.

    **kwargs
        Additional parameters passed to the underlying model estimators.
        For Auto models, this can include model-specific parameters like
        'season_length' for AutoETS/AutoTheta.
    """

    def __init__(
        self,
        model_type: Union[ModelTypes, str],
        max_lag: int = 10,
        order: OrderTypes = None,  # Can be None initially
        seasonal_order: Optional[tuple] = None,
        information_criterion: str = "aic",
        save_models=False,
        use_auto: bool = True,
        **kwargs,
    ):
        # Store original parameter for sklearn compatibility
        self.model_type = model_type
        
        # Normalize model type to handle Auto models internally
        if isinstance(model_type, str):
            model_type_lower = model_type.lower()
            # Map Auto model names to their base types
            if model_type_lower in ["autoarima", "auto_arima"]:
                self._internal_model_type = "arima"
                self.auto_model = "AutoARIMA"
            elif model_type_lower in ["autoets", "auto_ets"]:
                self._internal_model_type = "ets"  # Not in ModelTypes, but we'll handle specially
                self.auto_model = "AutoETS"
            elif model_type_lower in ["autotheta", "auto_theta"]:
                self._internal_model_type = "theta"  # Not in ModelTypes, but we'll handle specially
                self.auto_model = "AutoTheta"
            elif model_type_lower in ["autoces", "auto_ces"]:
                self._internal_model_type = "ces"  # Not in ModelTypes, but we'll handle specially
                self.auto_model = "AutoCES"
            elif model_type_lower in ModelTypes.__args__:  # type: ignore
                self._internal_model_type = model_type_lower  # type: ignore
                self.auto_model = None
            else:
                raise ValueError(
                    f"Unknown model type '{model_type}'. Supported types are: "
                    f"{list(ModelTypes.__args__)}, 'autoarima', 'autoets', 'autotheta', 'autoces'"  # type: ignore
                )
        else:
            self._internal_model_type = model_type
            self.auto_model = None

        self.max_lag = max_lag
        self.order: Union[
            OrderTypesWithoutNone, None
        ] = order  # Allow None initially, will be set in fit
        self.seasonal_order: Optional[tuple] = seasonal_order
        self.information_criterion = information_criterion
        self.save_models = save_models
        self.use_auto = use_auto
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

    def _compute_best_order(self, X: np.ndarray) -> Union[OrderTypesWithoutNone, tuple, None]:
        # For Auto models (AutoETS, AutoTheta, AutoCES), order is not applicable
        if self.auto_model in ["AutoETS", "AutoTheta", "AutoCES"]:
            # These models don't have traditional order parameters
            return None

        # For ARIMA/SARIMA models, use AutoARIMA if enabled
        if self._internal_model_type in ["arima", "sarima"] and (
            self.use_auto or self.auto_model == "AutoARIMA"
        ):
            # Use AutoARIMA from statsforecast backend for efficient order selection
            from tsbootstrap.backends.adapter import fit_with_backend

            # Flatten data if needed
            endog = X.flatten() if X.ndim > 1 else X

            # Fit AutoARIMA model
            fitted_adapter = fit_with_backend(
                model_type="AutoARIMA",
                endog=endog,
                exog=None,
                order=None,  # Let AutoARIMA determine order
                seasonal_order=self.seasonal_order if self._internal_model_type == "sarima" else None,
                force_backend="statsforecast",  # Use efficient statsforecast backend
                return_backend=False,
                max_p=self.max_lag,  # Use max_lag as upper bound for p
                max_q=self.max_lag,  # Use max_lag as upper bound for q
                **self.model_params,
            )

            # Extract the selected order from AutoARIMA
            if hasattr(fitted_adapter, "_backend"):
                backend = fitted_adapter._backend
                # Try to extract order from parameters
                if hasattr(backend, "params"):
                    params = backend.params
                    if isinstance(params, dict) and "order" in params:
                        return params["order"]
                # Try to extract from _order attribute
                if hasattr(backend, "_order"):
                    return backend._order

            # Fallback to default if order extraction fails
            return (self.max_lag // 2, 0, 0)

        # For traditional models without auto, use RankLags
        if self._internal_model_type in ModelTypes.__args__:  # type: ignore
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            self.rank_lagger = RankLags(
                X=X,
                max_lag=self.max_lag,
                model_type=self._internal_model_type,  # type: ignore
                save_models=self.save_models,
            )
            best_lag_int = self.rank_lagger.estimate_conservative_lag()

            return best_lag_int

        # For other model types (e.g., ets, theta, ces without auto), return None
        return None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        # Store original data shape for later use
        self._original_X_shape = X.shape

        # For Auto models that don't need order, skip order computation
        if self.order is None and self.auto_model not in ["AutoETS", "AutoTheta", "AutoCES"]:
            self.order = self._compute_best_order(X)

        # For traditional models, order must be determined
        if self.order is None and self._internal_model_type in ModelTypes.__args__:  # type: ignore
            raise ValueError(
                "Failed to determine model order automatically. This can occur when the lag selection "
                "algorithm cannot find a suitable order within the specified max_lag range. Consider "
                "increasing max_lag or providing an explicit order parameter."
            )

        # Prepare data for backend
        if self._internal_model_type == "var":
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
                        f"Univariate models require single time series data. "
                        f"Received multivariate data with {X.shape[1]} columns. "
                        f"Either select a single column or use VAR models for multivariate analysis."
                    )
            else:
                endog = X

        # Determine which model to use for fitting
        if self.auto_model:
            # Use the Auto model directly
            model_to_fit = self.auto_model
            # For Auto models, we generally use statsforecast backend
            backend_choice = "statsforecast"
            # Add seasonality parameters if applicable
            if (
                self.auto_model in ["AutoETS", "AutoTheta"]
                and "season_length" not in self.model_params
            ):
                if self.seasonal_order and len(self.seasonal_order) >= 4:
                    self.model_params["season_length"] = self.seasonal_order[3]
                else:
                    self.model_params["season_length"] = 1  # Default to non-seasonal
        else:
            # Use traditional model
            model_to_fit = self._internal_model_type
            backend_choice = "statsmodels"  # Traditional models use statsmodels

        # Fit using backend
        fitted_adapter = fit_with_backend(
            model_type=model_to_fit,
            endog=endog,
            exog=y,
            order=self.order,
            seasonal_order=self.seasonal_order,
            force_backend=backend_choice,
            return_backend=False,  # Get adapter for compatibility
            **self.model_params,
        )

        # Store the fitted model and adapter
        self.fitted_adapter = fitted_adapter
        # Get the underlying model from the adapter
        # The adapter wraps the backend, so we access through the adapter
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
            self.rescale_factors = {}

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

    def get_order(self) -> Union[OrderTypesWithoutNone, None]:
        check_is_fitted(self, "fitted_adapter")

        # For Auto models that don't have traditional order
        if self.auto_model in ["AutoETS", "AutoTheta", "AutoCES"]:
            return None  # These models don't have order parameters

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
        return self.fitted_adapter.predict(steps=n_steps, X=X if self._internal_model_type == "var" else None)

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
        if not isinstance(other, AutoOrderSelector):
            return False
        return (
            self.model_type == other.model_type
            and self._internal_model_type == other._internal_model_type
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
