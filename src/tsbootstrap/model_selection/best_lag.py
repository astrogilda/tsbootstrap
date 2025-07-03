"""TSFitBestLag class for automatic lag selection in time series models."""

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
    A class used to fit time series data and find the best lag for forecasting.

    This class automatically determines the optimal lag order for time series
    models using the RankLags algorithm, then fits the model using TSFit.

    Parameters
    ----------
    model_type : ModelTypes
        Type of time series model ('ar', 'arima', 'sarima', 'var', 'arch')
    max_lag : int, default=10
        Maximum lag to consider for order selection
    order : OrderTypes, optional
        Model order. If None, will be determined automatically
    seasonal_order : tuple, optional
        Seasonal order for SARIMA models
    save_models : bool, default=False
        Whether to save fitted models during lag selection
    **kwargs
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
            raise ValueError("Order could not be determined.")

        # Prepare data for backend
        if self.model_type == "var":
            # VAR needs multivariate data
            if X.ndim == 1:
                raise ValueError("VAR models require multivariate data")
            endog = X.T  # Backend expects (n_vars, n_obs) for VAR
        else:
            # For univariate models
            if X.ndim == 2:
                if X.shape[1] == 1:
                    endog = X.flatten()
                else:
                    # For univariate models, reject multivariate data
                    raise ValueError(
                        "X must be 1-dimensional or 2-dimensional with a single column for univariate models"
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
            raise NotFittedError("Model not fitted.")
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
            raise NotFittedError("Model not fitted.")
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
            raise NotFittedError("Model not fitted yet.")
        return self.resids_

    def get_fitted_X(self) -> np.ndarray:
        check_is_fitted(self, "fitted_adapter")
        if self.fitted_adapter is None:
            raise NotFittedError("Model not fitted yet.")
        return self.X_fitted_

    def get_order(self) -> OrderTypesWithoutNone:
        check_is_fitted(self, "order")
        if self.order is None:
            raise NotFittedError("Order not available.")
        return self.order

    def get_model(self):  # Returns the fitted model instance
        check_is_fitted(self, "model")
        if self.model is None:
            raise NotFittedError("Model not fitted.")
        return self.model

    def predict(self, X: np.ndarray, y: Optional[np.ndarray] = None, n_steps: int = 1):
        check_is_fitted(self, "fitted_adapter")
        if self.fitted_adapter is None:
            raise NotFittedError("Model not fitted yet.")
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
            raise NotFittedError("Model not fitted yet.")
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
