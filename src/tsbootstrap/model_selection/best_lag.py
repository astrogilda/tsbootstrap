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

from tsbootstrap.ranklags import RankLags
from tsbootstrap.tsfit_compat import TSFit
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
        self.ts_fit: Optional[TSFit] = None
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
        if self.order is None:
            self.order = self._compute_best_order(X)

        if self.order is None:  # Should be set by _compute_best_order
            raise ValueError("Order could not be determined.")

        self.ts_fit = TSFit(
            order=self.order,  # Now OrderTypesWithoutNone
            model_type=self.model_type,
            seasonal_order=self.seasonal_order,  # Pass seasonal_order
            **self.model_params,
        )
        self.ts_fit.fit(X, y=y)  # Fit the TSFit instance
        self.model = self.ts_fit.model  # Get the underlying statsmodels model
        self.rescale_factors = self.ts_fit.rescale_factors

        # Store fitted values and residuals on TSFitBestLag instance,
        # using the getter methods from TSFit which ensure 2D.
        if self.ts_fit is not None:  # Should be fitted now
            self.X_fitted_ = self.ts_fit.get_fitted_values()
            self.resids_ = self.ts_fit.get_residuals()
            # Also store order and n_lags if they are determined by TSFit
            # and needed by BaseResidualBootstrap (self.order_ was used)
            # self.order_ = self.ts_fit.get_order() # TSFitBestLag already has self.order
            # self.n_lags_ might not be directly on TSFit, but self.order reflects it.
        else:  # Should not happen if fit was successful
            raise NotFittedError("TSFit instance was not properly fitted within TSFitBestLag.")

        return self

    def get_coefs(self) -> np.ndarray:
        check_is_fitted(self, "model")
        if self.model is None:
            raise NotFittedError("Model not fitted.")
        # Get coefficients from the underlying model
        if hasattr(self.model, "params"):
            return self.model.params
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
        check_is_fitted(self, "ts_fit")
        if self.ts_fit is None:
            raise NotFittedError("ts_fit not available.")
        return self.ts_fit.get_residuals()

    def get_fitted_X(self) -> np.ndarray:
        check_is_fitted(self, "ts_fit")
        if self.ts_fit is None:
            raise NotFittedError("ts_fit not available.")
        return self.ts_fit.get_fitted_values()

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
        check_is_fitted(self, "ts_fit")
        if self.ts_fit is None:
            raise NotFittedError("ts_fit not available.")
        # TSFit.predict doesn't have y or n_steps parameters
        # For now, just use the basic predict method
        return self.ts_fit.predict(X)

    def score(
        self,
        X: NDArray,  # Changed np.ndarray to NDArray
        y: NDArray,  # Changed np.ndarray to NDArray
        sample_weight: Optional[NDArray] = None,  # Changed np.ndarray to NDArray
    ) -> float:
        check_is_fitted(self, "ts_fit")
        if self.ts_fit is None:
            raise NotFittedError("ts_fit not available.")
        # TSFit.score doesn't have sample_weight parameter
        return self.ts_fit.score(X, y)

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
