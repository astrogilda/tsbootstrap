"""TSFit-compatible wrapper for backends to ensure smooth migration."""

from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from tsbootstrap.backends.adapter import BackendToStatsmodelsAdapter, fit_with_backend
from tsbootstrap.services.tsfit_services import (
    TSFitHelperService,
    TSFitPredictionService,
    TSFitScoringService,
    TSFitValidationService,
)
from tsbootstrap.utils.types import ModelTypes, OrderTypesWithoutNone


class TSFitBackendWrapper(BaseEstimator, RegressorMixin):
    """
    TSFit-compatible wrapper that delegates to backend implementations.

    This wrapper provides 100% TSFit API compatibility while leveraging
    the backend system for improved performance and flexibility.

    Parameters
    ----------
    order : OrderTypesWithoutNone
        Order of the model
    model_type : ModelTypes
        Type of the model
    seasonal_order : Optional[tuple], default=None
        Seasonal order of the model for SARIMA
    use_backend : bool, default True
        Whether to use the new backend system. If True, uses appropriate
        backend based on feature flags. If False, falls back to statsmodels.
    **kwargs
        Additional parameters to be passed to the model

    Attributes
    ----------
    model : BackendToStatsmodelsAdapter or None
        The fitted model wrapped in a statsmodels-compatible adapter
    rescale_factors : dict
        Scaling factors used for data transformation
    _X : np.ndarray or None
        Stored exogenous variables from fitting
    _y : np.ndarray or None
        Stored endogenous variables from fitting
    """

    # Tags for scikit-base compatibility
    _tags = {
        "scitype:y": "univariate",
        "capability:multivariate": False,
        "capability:missing_values": False,
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "requires_y": True,
        "requires_X": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "handles-own-nan-values": False,
    }

    def __init__(
        self,
        order: OrderTypesWithoutNone,
        model_type: ModelTypes,
        seasonal_order: Optional[tuple] = None,
        use_backend: bool = True,
        **kwargs,
    ) -> None:
        """Initialize TSFitBackendWrapper with service composition."""
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
        self.model: Optional[BackendToStatsmodelsAdapter] = None
        self.rescale_factors: Dict[str, Any] = {}
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "TSFitBackendWrapper":
        """
        Fit the time series model using the backend system.

        Parameters
        ----------
        X : np.ndarray
            Time series data (endog)
        y : np.ndarray, optional
            Exogenous variables (exog)

        Returns
        -------
        TSFitBackendWrapper
            Self for method chaining
        """
        # Store original data for scoring
        self._X = X
        self._y = y

        # Handle data rescaling if needed
        endog = X
        exog = y

        # Check if we need to rescale
        if hasattr(self._helper_service, "check_if_rescale_needed"):
            rescale_needed, self.rescale_factors = self._helper_service.check_if_rescale_needed(
                endog, self.model_type
            )
            if rescale_needed:
                endog = self._helper_service.rescale_data(endog, self.rescale_factors)

        # Determine backend usage
        if self.use_backend:
            force_backend = None
        else:
            force_backend = "statsmodels"

        # Fit using backend system
        try:
            self.model = fit_with_backend(
                model_type=self.model_type,
                endog=endog,
                exog=exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                force_backend=force_backend,
                return_backend=False,  # Get adapter
                **self.model_params,
            )
        except Exception as e:
            # If backend fails and we were trying to use it, fall back to statsmodels
            if self.use_backend and force_backend is None:
                self.model = fit_with_backend(
                    model_type=self.model_type,
                    endog=endog,
                    exog=exog,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    force_backend="statsmodels",
                    return_backend=False,
                    **self.model_params,
                )
            else:
                raise e

        return self

    def predict(
        self,
        exog: Optional[np.ndarray] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate in-sample predictions.

        Parameters
        ----------
        exog : np.ndarray, optional
            Exogenous variables for prediction
        start : int, optional
            Starting index for prediction
        end : int, optional
            Ending index for prediction

        Returns
        -------
        np.ndarray
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        # Use prediction service for complex logic
        predictions = self._prediction_service.predict(
            self.model, self.model_type, start, end, exog
        )

        # Rescale if needed
        if self.rescale_factors:
            predictions = self._helper_service.rescale_back_data(predictions, self.rescale_factors)

        return predictions

    def forecast(self, steps: int = 1, exog: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate out-of-sample forecasts.

        Parameters
        ----------
        steps : int, default 1
            Number of steps to forecast
        exog : np.ndarray, optional
            Exogenous variables for forecasting

        Returns
        -------
        np.ndarray
            Forecasted values
        """
        if self.model is None:
            raise ValueError("Model must be fitted before forecasting")

        # Use the adapter's forecast method
        forecasts = self.model.forecast(steps, exog)

        # Rescale if needed
        if self.rescale_factors:
            forecasts = self._helper_service.rescale_back_data(forecasts, self.rescale_factors)

        return forecasts

    def score(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        metric: str = "mse",
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """
        Score the model using various metrics.

        Parameters
        ----------
        X : np.ndarray
            Time series data (endog)
        y : np.ndarray, optional
            Exogenous variables (exog)
        metric : str, default 'mse'
            Scoring metric to use
        sample_weight : np.ndarray, optional
            Sample weights

        Returns
        -------
        float
            Score value
        """
        if self.model is None:
            raise ValueError("Model must be fitted before scoring")

        # Generate predictions
        predictions = self.predict(exog=y)

        # Flatten predictions if needed
        if predictions.ndim == 2 and predictions.shape[1] == 1:
            predictions = predictions.ravel()

        # Align shapes - for AR models, predictions may be shorter due to lags
        if len(predictions) < len(X):
            # Trim X to match prediction length from the end
            X_aligned = X[-len(predictions) :]
        else:
            X_aligned = X

        # Use scoring service with correct parameters
        return self._scoring_service.score(
            y_true=X_aligned,
            y_pred=predictions,
            metric=metric,
        )

    def get_residuals(self) -> np.ndarray:
        """
        Get model residuals.

        Returns
        -------
        np.ndarray
            Model residuals
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting residuals")

        return self.model.resid

    def get_fitted_values(self) -> np.ndarray:
        """
        Get fitted values from the model.

        Returns
        -------
        np.ndarray
            Fitted values
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting fitted values")

        fitted_values = self.model.fittedvalues

        # Rescale if needed
        if self.rescale_factors:
            fitted_values = self._helper_service.rescale_back_data(
                fitted_values, self.rescale_factors
            )

        return fitted_values

    def get_information_criterion(self, criterion: str = "aic") -> float:
        """
        Get information criterion value.

        Parameters
        ----------
        criterion : str, default 'aic'
            Type of criterion ('aic', 'bic', 'hqic')

        Returns
        -------
        float
            Information criterion value
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting information criteria")

        return self._scoring_service.get_information_criteria(self.model, criterion)

    def check_residual_stationarity(self, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Check if residuals are stationary using statistical tests.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level for tests

        Returns
        -------
        dict
            Test results including statistic, p-value, and stationarity status
        """
        if self.model is None:
            raise ValueError("Model must be fitted before checking stationarity")

        residuals = self.get_residuals()

        # Use helper service for stationarity tests
        if hasattr(self._helper_service, "check_stationarity"):
            is_stationary, p_value = self._helper_service.check_stationarity(
                residuals, test="adf", significance=alpha
            )
            # Return in the expected format
            from statsmodels.tsa.stattools import adfuller

            result = adfuller(residuals)
            return {
                "statistic": result[0],
                "pvalue": p_value,
                "is_stationary": is_stationary,
                "critical_values": result[4],
            }
        else:
            # Fallback implementation
            from statsmodels.tsa.stattools import adfuller

            result = adfuller(residuals)
            return {
                "statistic": result[0],
                "pvalue": result[1],
                "is_stationary": result[1] < alpha,
                "critical_values": result[4],
            }

    def summary(self) -> str:
        """
        Get model summary.

        Returns
        -------
        str
            Model summary
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting summary")

        return self.model.summary()

    def __repr__(self) -> str:
        """String representation of the wrapper."""
        backend_info = "Backend" if self.use_backend else "Statsmodels"
        return (
            f"TSFitBackendWrapper(model_type={self.model_type}, "
            f"order={self.order}, seasonal_order={self.seasonal_order}, "
            f"backend={backend_info})"
        )

    def _calculate_trend_terms(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate trend terms for the model.

        This is a compatibility method for TSFit interface.

        Parameters
        ----------
        X : np.ndarray
            Input data

        Returns
        -------
        np.ndarray
            Trend terms
        """
        # This method exists for compatibility but may not be needed
        # for all backend implementations
        if hasattr(self.model, "_calculate_trend_terms"):
            return self.model._calculate_trend_terms(X)
        else:
            # Return zeros as default
            return np.zeros_like(X)
