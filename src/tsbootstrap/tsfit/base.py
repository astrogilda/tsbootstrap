"""
TSFit implementation using composition over inheritance.

This module provides the TSFit class that uses service composition
for time series model fitting and prediction.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
from arch.univariate.base import ARCHModelResult
from sklearn.base import (  # sklearn's RegressorMixin provides score() method
    BaseEstimator,
    RegressorMixin,
)
from sklearn.utils.validation import check_is_fitted
from statsmodels.tsa.ar_model import AutoRegResultsWrapper
from statsmodels.tsa.arima.model import ARIMAResultsWrapper
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

from tsbootstrap.services.tsfit_services import (
    TSFitHelperService,
    TSFitPredictionService,
    TSFitScoringService,
    TSFitValidationService,
)
from tsbootstrap.time_series_model import TimeSeriesModel
from tsbootstrap.utils.types import ModelTypes, OrderTypesWithoutNone


class TSFit(BaseEstimator, RegressorMixin):
    """
    TSFit class using composition over inheritance.

    This class provides a unified interface for fitting various time series
    models including AR, MA, ARMA, ARIMA, SARIMA, VAR, and ARCH models.

    It uses service composition for better maintainability and testability.

    Parameters
    ----------
    order : OrderTypesWithoutNone
        Order of the model
    model_type : ModelTypes
        Type of the model
    seasonal_order : Optional[tuple], default=None
        Seasonal order of the model for SARIMA
    use_backend : bool, default False
        Whether to use the new backend system. If True, uses statsforecast
        for supported models based on feature flags.
    **kwargs
        Additional parameters to be passed to the model

    Attributes
    ----------
    model : Optional[Union[AutoRegResultsWrapper, ...]]
        The fitted model object
    rescale_factors : dict
        Dictionary containing rescaling factors used during fitting
    model_params : dict
        Additional model parameters
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
        use_backend: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize TSFit with service composition.

        Parameters
        ----------
        order : OrderTypesWithoutNone
            Order of the model
        model_type : ModelTypes
            Type of the model
        seasonal_order : Optional[tuple], default=None
            Seasonal order of the model for SARIMA
        use_backend : bool, default False
            Whether to use the new backend system. If True, uses statsforecast
            for supported models based on feature flags.
        **kwargs
            Additional parameters to be passed to the model
        """
        # Initialize services
        self._validation_service = TSFitValidationService()
        self._prediction_service = TSFitPredictionService()
        self._scoring_service = TSFitScoringService()
        self._helper_service = TSFitHelperService()

        # Validate inputs using service
        self.model_type = self._validation_service.validate_model_type(model_type)
        self.order = self._validation_service.validate_order(order, model_type)
        self.seasonal_order = self._validation_service.validate_seasonal_order(
            seasonal_order, model_type
        )

        # Store additional parameters
        self.model_params = kwargs
        self.use_backend = use_backend

        # Initialize attributes
        self.model: Optional[
            Union[
                AutoRegResultsWrapper,
                ARIMAResultsWrapper,
                SARIMAXResultsWrapper,
                VARResultsWrapper,
                ARCHModelResult,
            ]
        ] = None
        self.rescale_factors: Dict[str, Any] = {}
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> TSFit:
        """
        Fit the time series model.

        Parameters
        ----------
        X : np.ndarray
            Time series data
        y : Optional[np.ndarray]
            Target values (for supervised models)

        Returns
        -------
        self : TSFit
            Fitted estimator
        """
        # Store data
        self._X = X
        self._y = y

        # Create and fit the appropriate model
        ts_model = TimeSeriesModel(
            X=X,
            y=y,
            model_type=self.model_type,
            use_backend=self.use_backend,
        )

        # Fit model with order and seasonal_order
        self.model = ts_model.fit(
            order=self.order,
            seasonal_order=self.seasonal_order,
            **self.model_params,
        )

        # Store any rescaling factors
        if hasattr(ts_model, "rescale_factors"):
            self.rescale_factors = ts_model.rescale_factors

        return self

    def predict(
        self,
        X: Optional[np.ndarray] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate in-sample predictions.

        Parameters
        ----------
        X : Optional[np.ndarray]
            Data for prediction (required for VAR models)
        start : Optional[int]
            Start index for prediction
        end : Optional[int]
            End index for prediction

        Returns
        -------
        np.ndarray
            Predictions
        """
        check_is_fitted(self, "model")

        return self._prediction_service.predict(
            model=self.model,
            model_type=self.model_type,
            start=start,
            end=end,
            X=X,
        )

    def forecast(self, steps: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate out-of-sample forecasts.

        Parameters
        ----------
        steps : int
            Number of steps to forecast
        X : Optional[np.ndarray]
            Data for VAR model forecast

        Returns
        -------
        np.ndarray
            Forecasts
        """
        check_is_fitted(self, "model")

        return self._prediction_service.forecast(
            model=self.model,
            model_type=self.model_type,
            steps=steps,
            X=X,
        )

    def score(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        metric: str = "r2",
    ) -> float:
        """
        Score the model.

        This method supports both sklearn interface (default R² score)
        and custom metrics.

        Parameters
        ----------
        X : Optional[np.ndarray]
            Input data (ground truth)
        y : Optional[np.ndarray]
            Not used for time series, kept for sklearn compatibility
        metric : str
            Scoring metric ('r2', 'mse', 'mae', 'rmse')

        Returns
        -------
        float
            Score value
        """
        check_is_fitted(self, "model")

        # Use stored data if not provided
        if X is None and self._X is not None:
            X = self._X

        # Get predictions
        y_pred = self.predict()

        # For sklearn compatibility, use X as ground truth
        y_true = X

        # Handle shape mismatch for scoring
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        # Ensure same length (predictions might be shorter due to lag)
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[-min_len:]
        y_pred = y_pred[-min_len:]

        # Remove NaN values that might be in AR predictions
        mask = ~(np.isnan(y_true).any(axis=1) | np.isnan(y_pred).any(axis=1))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            return np.nan

        # Use R² for sklearn compatibility when called without metric
        if metric == "r2":
            from sklearn.metrics import r2_score

            return r2_score(y_true, y_pred)

        return self._scoring_service.score(
            y_true=y_true,
            y_pred=y_pred,
            metric=metric,
        )

    def get_residuals(self, standardize: bool = False) -> np.ndarray:
        """
        Get model residuals.

        Parameters
        ----------
        standardize : bool
            Whether to standardize residuals

        Returns
        -------
        np.ndarray
            Residuals
        """
        check_is_fitted(self, "model")

        return self._helper_service.get_residuals(
            model=self.model,
            standardize=standardize,
        )

    def get_fitted_values(self) -> np.ndarray:
        """
        Get fitted values.

        Returns
        -------
        np.ndarray
            Fitted values
        """
        check_is_fitted(self, "model")

        return self._helper_service.get_fitted_values(model=self.model)

    @classmethod
    def _calculate_trend_terms(cls, model_type: str, model: Any) -> int:
        """
        Calculate the number of trend terms in a model.

        Legacy method for backward compatibility.
        Delegates to TSFitHelperService.

        Parameters
        ----------
        model_type : str
            Type of model (e.g., 'ar', 'arima')
        model : Any
            The fitted model object

        Returns
        -------
        int
            Number of trend terms
        """
        from tsbootstrap.services.tsfit_services import TSFitHelperService

        return TSFitHelperService.calculate_trend_terms(model_type, model)

    def get_information_criterion(self, criterion: str = "aic") -> float:
        """
        Get information criterion.

        Parameters
        ----------
        criterion : str
            Criterion type ('aic', 'bic', 'hqic')

        Returns
        -------
        float
            Criterion value
        """
        check_is_fitted(self, "model")

        return self._scoring_service.get_information_criteria(
            model=self.model,
            criterion=criterion,
        )

    def check_residual_stationarity(
        self, test: str = "adf", significance: float = 0.05
    ) -> tuple[bool, float]:
        """
        Check if residuals are stationary.

        Parameters
        ----------
        test : str
            Test to use ('adf', 'kpss')
        significance : float
            Significance level

        Returns
        -------
        tuple[bool, float]
            (is_stationary, p_value)
        """
        residuals = self.get_residuals()

        # Flatten residuals for stationarity test
        if residuals.ndim > 1:
            residuals = residuals.ravel()

        return self._helper_service.check_stationarity(
            residuals=residuals,
            test=test,
            significance=significance,
        )

    def summary(self) -> Any:
        """
        Get model summary.

        Returns
        -------
        Model summary object
        """
        check_is_fitted(self, "model")

        if hasattr(self.model, "summary"):
            return self.model.summary()
        else:
            # Return basic info if summary not available
            return {
                "model_type": self.model_type,
                "order": self.order,
                "seasonal_order": self.seasonal_order,
                "aic": self.get_information_criterion("aic"),
                "bic": self.get_information_criterion("bic"),
            }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TSFit(model_type='{self.model_type}', "
            f"order={self.order}, seasonal_order={self.seasonal_order})"
        )
