"""TSFit Compatibility Adapter - Provides TSFit interface using backend system.

This module should be placed at src/tsbootstrap/tsfit.py to maintain import compatibility.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError

from tsbootstrap.backends.adapter import BackendToStatsmodelsAdapter, fit_with_backend
from tsbootstrap.services.tsfit_services import (
    TSFitHelperService,
    TSFitPredictionService,
    TSFitScoringService,
    TSFitValidationService,
)
from tsbootstrap.utils.types import ModelTypes, OrderTypes


class TSFit(BaseEstimator, RegressorMixin):
    """
    TSFit Compatibility Adapter - Maintains backward compatibility while using backends.

    This class provides the exact TSFit interface expected by existing code while
    internally delegating to the new backend system. This ensures zero breaking
    changes during the migration period.

    Parameters
    ----------
    order : OrderTypes
        The order of the model. Can be:
        - int: for AR, MA, ARCH models
        - tuple: for ARIMA (p,d,q), SARIMA models
        - None: will be determined automatically (not recommended)
    model_type : ModelTypes
        Type of time series model ('ar', 'ma', 'arma', 'arima', 'sarima', 'var', 'arch')
    seasonal_order : Optional[tuple], default=None
        Seasonal order for SARIMA models (P,D,Q,s)
    **kwargs
        Additional parameters passed to the underlying model

    Attributes
    ----------
    model : BackendToStatsmodelsAdapter
        The fitted model wrapped in a statsmodels-compatible adapter
    rescale_factors : Dict[str, Any]
        Scaling factors used for data transformation
    _X : np.ndarray
        Stored data from fitting (for scoring)
    _y : Optional[np.ndarray]
        Stored exogenous variables from fitting
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
        order: OrderTypes,
        model_type: ModelTypes,
        seasonal_order: Optional[tuple] = None,
        **kwargs,
    ) -> None:
        """Initialize TSFit with service composition."""
        # Initialize services
        self._validation_service = TSFitValidationService()
        self._prediction_service = TSFitPredictionService()
        self._scoring_service = TSFitScoringService()
        self._helper_service = TSFitHelperService()

        # Validate and store parameters
        self.model_type = self._validation_service.validate_model_type(model_type)
        self.order = order  # Store as-is, validate during fit if None
        self.seasonal_order = self._validation_service.validate_seasonal_order(
            seasonal_order, model_type
        )
        self.model_params = kwargs

        # Initialize attributes
        self.model: Optional[BackendToStatsmodelsAdapter] = None
        self.rescale_factors: Dict[str, Any] = {}
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "TSFit":
        """
        Fit the time series model.

        Parameters
        ----------
        X : np.ndarray
            Time series data (endogenous variable)
        y : Optional[np.ndarray], default=None
            Exogenous variables

        Returns
        -------
        TSFit
            Self for method chaining (sklearn compatibility)
        """
        # Validate order if it was None
        if self.order is None:
            # Default orders based on model type
            if self.model_type == "var":
                self.order = 1
            elif self.model_type in ["arima", "sarima"]:
                self.order = (1, 1, 1)
            else:  # ar, ma, arma, arch
                self.order = 1

        # Validate order with the actual value
        self.order = self._validation_service.validate_order(self.order, self.model_type)

        # Store original data for scoring
        self._X = X
        self._y = y

        # Prepare data
        endog = X
        exog = y

        # Check if rescaling needed
        if hasattr(self._helper_service, "check_if_rescale_needed"):
            rescale_needed, self.rescale_factors = self._helper_service.check_if_rescale_needed(
                endog, self.model_type
            )
            if rescale_needed:
                endog = self._helper_service.rescale_data(endog, self.rescale_factors)

        # Fit using backend system
        try:
            # Try with backend first
            self.model = fit_with_backend(
                model_type=self.model_type,
                endog=endog,
                exog=exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                force_backend=None,  # Use appropriate backend
                return_backend=False,  # Get adapter for statsmodels compatibility
                **self.model_params,
            )
        except Exception as e:
            # Fallback to statsmodels if backend fails
            try:
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
            except Exception:
                # Re-raise original exception if fallback also fails
                raise e from None

        return self

    def predict(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate predictions.

        Parameters
        ----------
        X : Optional[np.ndarray], default=None
            If provided, generate predictions for this data (out-of-sample).
            If None, return in-sample predictions.

        Returns
        -------
        np.ndarray
            Predicted values
        """
        if self.model is None:
            raise NotFittedError("Model must be fitted before prediction")

        if X is None:
            # In-sample predictions
            predictions = self._prediction_service.predict(
                self.model, self.model_type, exog=self._y, start=None, end=None
            )
        else:
            # Out-of-sample predictions (for VAR models)
            if self.model_type == "var":
                # VAR needs special handling for out-of-sample
                predictions = self.model.forecast(X, steps=len(X))
            else:
                # For other models, use standard predict
                predictions = self._prediction_service.predict(
                    self.model, self.model_type, exog=X, start=0, end=len(X) - 1
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
        steps : int, default=1
            Number of steps to forecast
        exog : Optional[np.ndarray], default=None
            Exogenous variables for forecasting

        Returns
        -------
        np.ndarray
            Forecasted values
        """
        if self.model is None:
            raise NotFittedError("Model must be fitted before forecasting")

        # Use adapter's forecast method
        forecasts = self.model.forecast(steps, exog)

        # Rescale if needed
        if self.rescale_factors:
            forecasts = self._helper_service.rescale_back_data(forecasts, self.rescale_factors)

        return forecasts

    def score(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """
        Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : np.ndarray
            Test samples
        y : Optional[np.ndarray], default=None
            Exogenous variables for test samples
        sample_weight : Optional[np.ndarray], default=None
            Sample weights

        Returns
        -------
        float
            R^2 score
        """
        if self.model is None:
            raise NotFittedError("Model must be fitted before scoring")

        # For time series, we compare against the input X
        return self._scoring_service.score(
            model=self,
            fitted_model=self.model,
            X=X,
            y=y,
            metric="r2",
            sample_weight=sample_weight,
        )

    def get_residuals(self, standardize: bool = False) -> np.ndarray:
        """
        Get model residuals.

        Parameters
        ----------
        standardize : bool, default=False
            Whether to standardize residuals

        Returns
        -------
        np.ndarray
            Model residuals
        """
        if self.model is None:
            raise NotFittedError("Model must be fitted before getting residuals")

        residuals = self.model.resid

        if standardize:
            # Standardize residuals
            residuals = (residuals - np.mean(residuals)) / np.std(residuals)

        return residuals

    def get_fitted_values(self) -> np.ndarray:
        """
        Get fitted values from the model.

        Returns
        -------
        np.ndarray
            Fitted values
        """
        if self.model is None:
            raise NotFittedError("Model must be fitted before getting fitted values")

        fitted_values = self.model.fittedvalues

        # Rescale if needed
        if self.rescale_factors:
            fitted_values = self._helper_service.rescale_back_data(
                fitted_values, self.rescale_factors
            )

        return fitted_values

    def check_residual_stationarity(
        self, test: str = "adf", alpha: float = 0.05
    ) -> Tuple[bool, float]:
        """
        Check if residuals are stationary.

        Parameters
        ----------
        test : str, default="adf"
            Test to use ('adf' or 'kpss')
        alpha : float, default=0.05
            Significance level

        Returns
        -------
        Tuple[bool, float]
            (is_stationary, p_value)
        """
        if self.model is None:
            raise NotFittedError("Model must be fitted before checking stationarity")

        residuals = self.get_residuals()

        if test == "adf":
            from statsmodels.tsa.stattools import adfuller

            result = adfuller(residuals)
            p_value = result[1]
            is_stationary = p_value < alpha
        elif test == "kpss":
            from statsmodels.tsa.stattools import kpss

            result = kpss(residuals, regression="c")
            p_value = result[1]
            is_stationary = p_value >= alpha  # KPSS null is stationarity
        else:
            raise ValueError(f"Unknown test: {test}. Use 'adf' or 'kpss'.")

        return is_stationary, p_value

    def get_information_criterion(self, criterion: str = "aic") -> float:
        """
        Get information criterion value.

        Parameters
        ----------
        criterion : str, default="aic"
            Type of criterion ('aic', 'bic', 'hqic')

        Returns
        -------
        float
            Information criterion value
        """
        if self.model is None:
            raise NotFittedError("Model must be fitted before getting information criteria")

        return self._scoring_service.get_information_criteria(
            self.model, self.model_type, criterion
        )

    def summary(self) -> Any:
        """
        Get model summary.

        Returns
        -------
        Any
            Model summary (usually statsmodels Summary object)
        """
        if self.model is None:
            raise NotFittedError("Model must be fitted before getting summary")

        return self.model.summary()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TSFit(order={self.order}, model_type={self.model_type}, "
            f"seasonal_order={self.seasonal_order})"
        )

    def _more_tags(self):
        """Additional tags for sklearn compatibility."""
        return {
            "poor_score": True,
            "non_deterministic": True,
            "binary_only": False,
            "requires_positive_X": False,
            "requires_positive_y": False,
            "_skip_test": True,  # Skip sklearn estimator tests
        }


# Maintain backward compatibility for direct imports
TSFitCompatibilityAdapter = TSFit


__all__ = ["TSFit", "TSFitCompatibilityAdapter"]
