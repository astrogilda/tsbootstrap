"""
Core TSFit class for time series model fitting.
"""

from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
from arch.univariate.base import ARCHModelResult
from numpy import ndarray
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import NotFittedError, check_is_fitted
from statsmodels.tsa.ar_model import AutoRegResultsWrapper
from statsmodels.tsa.arima.model import ARIMAResultsWrapper
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

from tsbootstrap.time_series_model import TimeSeriesModel
from tsbootstrap.tsfit.helpers import TSFitHelpers
from tsbootstrap.tsfit.predictors import TSFitPredictors
from tsbootstrap.tsfit.scorers import TSFitScorers
from tsbootstrap.tsfit.validators import TSFitValidators
from tsbootstrap.utils.types import (
    ModelTypes,
    OrderTypes,
    OrderTypesWithoutNone,
)


class TSFit(
    BaseEstimator,
    RegressorMixin,
    TSFitValidators,
    TSFitPredictors,
    TSFitScorers,
    TSFitHelpers,
):
    """
    A class for fitting time series models.

    This class provides a unified interface for fitting various time series
    models including AR, MA, ARMA, ARIMA, SARIMA, VAR, and ARCH models.

    Parameters
    ----------
    order : OrderTypesWithoutNone
        Order of the model.
    model_type : ModelTypes
        Type of the model.
    seasonal_order : Optional[tuple], default=None
        Seasonal order of the model for SARIMA.
    **kwargs
        Additional parameters to be passed to the model.

    Attributes
    ----------
    model : Optional[Union[AutoRegResultsWrapper, ARIMAResultsWrapper,
                          SARIMAXResultsWrapper, VARResultsWrapper,
                          ARCHModelResult]]
        The fitted model object.
    rescale_factors : dict
        Dictionary containing rescaling factors used during fitting.
    model_params : dict
        Additional model parameters.
    """

    _tags = {
        "X_types": ["pd_DataFrame_Table", "np_ndarray"],
        "y_types": ["pd_DataFrame_Table", "np_ndarray", "None"],
        "allow_nan": False,
        "allow_inf": False,
        "allow_multivariate": True,
        "allow_multioutput": True,
        "enforce_index": False,
        "enforce_index_type": None,
        "y_required": False,
        "X_required": True,
    }

    def __init__(
        self,
        order: OrderTypesWithoutNone,
        model_type: ModelTypes,
        seasonal_order: Optional[tuple] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the TSFit object.

        Parameters
        ----------
        order : OrderTypesWithoutNone
            Order of the model.
        model_type : ModelTypes
            Type of the model.
        seasonal_order : Optional[tuple], default=None
            Seasonal order of the model for SARIMA.
        **kwargs
            Additional parameters to be passed to the model.
        """
        # Set model_type first, before order (order validation depends on model_type)
        self._model_type = self.validate_model_type(model_type)
        self._order = self.validate_order(order, self._model_type)
        self._seasonal_order = self.validate_seasonal_order(seasonal_order, self._model_type)
        self.rescale_factors: dict = {}
        self.model_params = kwargs
        self.model: Optional[
            Union[
                AutoRegResultsWrapper,
                ARIMAResultsWrapper,
                SARIMAXResultsWrapper,
                VARResultsWrapper,
                ARCHModelResult,
            ]
        ] = None

    @property
    def model_type(self) -> ModelTypes:
        """Get the model type."""
        return self._model_type

    @model_type.setter
    def model_type(self, value: ModelTypes) -> None:
        """Set the model type with validation."""
        self._model_type = self.validate_model_type(value)

    @property
    def order(self) -> OrderTypes:
        """Get the model order."""
        return self._order

    @order.setter
    def order(self, value: OrderTypes) -> None:
        """Set the model order with validation."""
        self._order = self.validate_order(value, self.model_type)

    @property
    def seasonal_order(self) -> Optional[tuple]:
        """Get the seasonal order."""
        return self._seasonal_order

    @seasonal_order.setter
    def seasonal_order(self, value: Optional[tuple]) -> None:
        """Set the seasonal order with validation."""
        self._seasonal_order = self.validate_seasonal_order(value, self.model_type)

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator."""
        return {
            "order": self.order,
            "model_type": self.model_type,
            "seasonal_order": self.seasonal_order,
            **self.model_params,
        }

    def set_params(self, **params) -> TSFit:
        """Set the parameters of this estimator."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def fit(
        self,
        X: Union[np.ndarray, NDArray[np.float64]],
        y: Optional[Union[np.ndarray, NDArray[np.float64]]] = None,
    ) -> TSFit:
        """
        Fit the time series model.

        Parameters
        ----------
        X : Union[np.ndarray, NDArray[np.float64]]
            Time series data.
        y : Optional[Union[np.ndarray, NDArray[np.float64]]], default=None
            Exogenous variables (not used in this implementation).

        Returns
        -------
        self : TSFit
            The fitted estimator.
        """
        # Validate input
        if not isinstance(X, np.ndarray):
            if isinstance(X, list) and len(X) == 0:
                raise TypeError("Cannot fit model with empty data")
            try:
                X = np.asarray(X)
            except Exception as e:
                raise TypeError(f"X must be array-like, got {type(X).__name__}") from e

        # Check for scalar input
        if X.ndim == 0:
            raise TypeError("X must be array-like with at least 1 dimension")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Create the time series model
        ts_model = TimeSeriesModel(
            X,
            y=y,
            model_type=self.model_type,
        )

        # Fit the model with order and seasonal_order
        self.model = ts_model.fit(
            order=self.order,
            seasonal_order=self.seasonal_order,
            **self.model_params,
        )

        # Store the rescale factors
        self.rescale_factors = getattr(ts_model, "rescale_factors", {})

        return self

    def get_coefs(self) -> ndarray:
        """Get model coefficients."""
        check_is_fitted(self, "model")
        if self.model is None:
            raise AttributeError("Model not fitted")
        # Determine number of features
        if hasattr(self.model, "k_endog"):
            n_features = self.model.k_endog
        elif hasattr(self.model, "neqs"):
            n_features = self.model.neqs
        else:
            n_features = 1
        return self._get_coefs_helper(self.model, n_features)

    def get_intercepts(self) -> ndarray:
        """Get model intercepts."""
        check_is_fitted(self, "model")
        if self.model is None:
            raise AttributeError("Model not fitted")
        # Determine number of features
        if hasattr(self.model, "k_endog"):
            n_features = self.model.k_endog
        elif hasattr(self.model, "neqs"):
            n_features = self.model.neqs
        else:
            n_features = 1
        return self._get_intercepts_helper(self.model, n_features)

    def get_residuals(self) -> ndarray:
        """Get model residuals."""
        check_is_fitted(self, "model")
        if self.model is None:
            raise AttributeError("Model not fitted")
        return self._get_residuals_helper()

    def get_fitted_X(self) -> ndarray:
        """Get fitted values."""
        check_is_fitted(self, "model")
        if self.model is None:
            raise AttributeError("Model not fitted")
        return self._get_fitted_X_helper()

    def get_order(self) -> OrderTypes:
        """Get the order of the fitted model."""
        check_is_fitted(self, "model")
        if self.model is None:
            raise AttributeError("Model not fitted")
        return self._get_order_helper()
