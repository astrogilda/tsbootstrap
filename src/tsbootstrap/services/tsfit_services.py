"""
Services for TSFit functionality.

This module provides services to replace the complex multiple inheritance
in the TSFit implementation.
"""

from typing import Any, Optional, Tuple, Union

import numpy as np
from arch.univariate.base import ARCHModelResult
from statsmodels.tsa.ar_model import AutoRegResultsWrapper
from statsmodels.tsa.arima.model import ARIMAResultsWrapper
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

from tsbootstrap.utils.types import ModelTypes, OrderTypes
from tsbootstrap.utils.validate import validate_literal_type


class TSFitValidationService:
    """Service for TSFit validation operations."""

    @staticmethod
    def validate_model_type(value: ModelTypes) -> ModelTypes:
        """Validate and return the model type."""
        validate_literal_type(value, ModelTypes)
        return value

    @staticmethod
    def validate_order(value: OrderTypes, model_type: ModelTypes) -> OrderTypes:
        """
        Validate the order parameter based on model type.

        Parameters
        ----------
        value : OrderTypes
            The order value to validate
        model_type : ModelTypes
            The type of model being used

        Returns
        -------
        OrderTypes
            The validated order

        Raises
        ------
        TypeError
            If the order type is invalid for the given model type
        ValueError
            If the order value is invalid
        """
        from numbers import Integral

        # VAR models require integer order
        if model_type == "var":
            if not isinstance(value, Integral):
                raise TypeError(
                    f"Order must be an integer for VAR model. Got {type(value).__name__}."
                )
            if value < 1:
                raise ValueError(f"Order must be positive for VAR model. Got {value}.")
            return value

        # ARCH models require integer order
        if model_type == "arch":
            if not isinstance(value, Integral):
                raise TypeError(
                    f"Order must be an integer for ARCH model. Got {type(value).__name__}."
                )
            if value < 1:
                raise ValueError(f"Order must be positive for ARCH model. Got {value}.")
            return value

        # AR/MA models can have None order
        if value is None:
            if model_type in ["ar", "ma"]:
                return value
            else:
                raise ValueError(f"Order cannot be None for {model_type} model.")

        # Validate tuple orders for ARMA/ARIMA/SARIMA
        if isinstance(value, (list, tuple)):
            if model_type not in ["arma", "arima", "sarima"]:
                raise TypeError(f"Order must not be a tuple/list for {model_type} model.")

            # Convert to tuple and validate length
            value = tuple(value)
            expected_lengths = {"arma": 2, "arima": 3, "sarima": 3}
            expected_length = expected_lengths.get(model_type)

            if expected_length and len(value) != expected_length:
                raise ValueError(
                    f"Order must have {expected_length} elements for {model_type} model. "
                    f"Got {len(value)}."
                )

            # Validate all elements are non-negative integers
            for i, v in enumerate(value):
                if not isinstance(v, Integral) or v < 0:
                    raise ValueError(
                        f"All order elements must be non-negative integers. Element {i} is {v}."
                    )

            return value

        # Single integer order
        if isinstance(value, Integral):
            if model_type in ["arma", "arima", "sarima"]:
                raise TypeError(f"Order must be a tuple/list for {model_type} model, not integer.")
            if value < 0:
                raise ValueError(f"Order must be non-negative. Got {value}.")
            return value

        raise TypeError(f"Invalid order type: {type(value).__name__}")

    @staticmethod
    def validate_seasonal_order(value: Optional[tuple], model_type: ModelTypes) -> Optional[tuple]:
        """
        Validate seasonal order for SARIMA models.

        Parameters
        ----------
        value : Optional[tuple]
            The seasonal order (P, D, Q, s)
        model_type : ModelTypes
            The type of model

        Returns
        -------
        Optional[tuple]
            The validated seasonal order

        Raises
        ------
        ValueError
            If seasonal order is invalid
        """
        if value is None:
            return None

        if model_type != "sarima":
            if value is not None:
                raise ValueError(
                    f"seasonal_order is only valid for SARIMA models, not {model_type}."
                )
            return None

        if not isinstance(value, (list, tuple)):
            raise TypeError("seasonal_order must be a tuple or list.")

        value = tuple(value)

        if len(value) != 4:
            raise ValueError(f"seasonal_order must have 4 elements (P, D, Q, s). Got {len(value)}.")

        # Validate all elements
        from numbers import Integral

        for i, v in enumerate(value):
            if not isinstance(v, Integral) or v < 0:
                raise ValueError(
                    f"All seasonal_order elements must be non-negative integers. "
                    f"Element {i} is {v}."
                )

        # The seasonal period (s) must be at least 2
        if value[3] < 2:
            raise ValueError(f"Seasonal period (s) must be at least 2. Got {value[3]}.")

        return value


class TSFitPredictionService:
    """Service for TSFit prediction operations."""

    def predict(
        self,
        model: Union[
            AutoRegResultsWrapper,
            ARIMAResultsWrapper,
            SARIMAXResultsWrapper,
            VARResultsWrapper,
            ARCHModelResult,
        ],
        model_type: ModelTypes,
        start: Optional[int] = None,
        end: Optional[int] = None,
        X: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate predictions from fitted model.

        Parameters
        ----------
        model : Model result object
            The fitted model
        model_type : ModelTypes
            Type of the model
        start : Optional[int]
            Start index for prediction
        end : Optional[int]
            End index for prediction
        X : Optional[np.ndarray]
            Data for prediction (used for VAR models)

        Returns
        -------
        np.ndarray
            Predictions
        """
        if model is None:
            raise ValueError("Model must be fitted before prediction.")

        # Set default values for start and end if not provided
        if start is None or end is None:
            if hasattr(model, "nobs"):
                n_obs = model.nobs
            elif hasattr(model, "_nobs"):
                n_obs = model._nobs
            else:
                # For ARCH models
                n_obs = len(model.resid)

            if start is None:
                start = 0
            if end is None:
                end = n_obs - 1

        # Handle different model types
        if model_type == "var":
            if X is None:
                raise ValueError("X is required for VAR model prediction.")
            steps = len(X) if end is None else end - (start or 0)
            predictions = model.forecast(X, steps=steps)

        elif model_type == "arch":
            # ARCH models have different prediction interface
            predictions = model.forecast(horizon=end - (start or 0) if end else 1).mean.values

        else:
            # AR, MA, ARMA, ARIMA, SARIMA models
            predictions = model.predict(start=start, end=end)

        # Ensure numpy array and consistent shape
        if hasattr(predictions, "values"):
            predictions = predictions.values

        predictions = np.asarray(predictions)

        # Ensure consistent output shape - match original behavior
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        elif predictions.ndim > 2:
            predictions = predictions.reshape(predictions.shape[0], -1)

        return predictions

    def forecast(
        self,
        model: Union[
            AutoRegResultsWrapper,
            ARIMAResultsWrapper,
            SARIMAXResultsWrapper,
            VARResultsWrapper,
            ARCHModelResult,
        ],
        model_type: ModelTypes,
        steps: int = 1,
        X: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate out-of-sample forecasts.

        Parameters
        ----------
        model : Model result object
            The fitted model
        model_type : ModelTypes
            Type of the model
        steps : int
            Number of steps to forecast
        X : Optional[np.ndarray]
            Data for VAR model forecast

        Returns
        -------
        np.ndarray
            Forecasts
        """
        if model is None:
            raise ValueError("Model must be fitted before forecasting.")

        if model_type == "var":
            if X is None:
                raise ValueError("X is required for VAR model forecast.")
            predictions = model.forecast(X, steps=steps)

        elif model_type == "arch":
            predictions = model.forecast(horizon=steps).mean.values

        else:
            predictions = model.forecast(steps=steps)

        # Ensure numpy array and consistent shape
        if hasattr(predictions, "values"):
            predictions = predictions.values

        predictions = np.asarray(predictions)

        # For univariate forecasts, keep 1D shape
        # Only reshape to 2D if multivariate
        if predictions.ndim == 2 and predictions.shape[1] == 1:
            predictions = predictions.ravel()

        return predictions


class TSFitScoringService:
    """Service for TSFit scoring operations."""

    def score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric: str = "mse",
    ) -> float:
        """
        Score predictions against true values.

        Parameters
        ----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
        metric : str
            Scoring metric ('mse', 'mae', 'rmse', 'mape')

        Returns
        -------
        float
            Score value
        """
        # Ensure same shape
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

        if metric == "mse":
            return np.mean((y_true - y_pred) ** 2)
        elif metric == "mae":
            return np.mean(np.abs(y_true - y_pred))
        elif metric == "rmse":
            return np.sqrt(np.mean((y_true - y_pred) ** 2))
        elif metric == "mape":
            # Avoid division by zero
            mask = y_true != 0
            if not np.any(mask):
                return np.inf
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def get_information_criteria(
        self,
        model: Union[
            AutoRegResultsWrapper,
            ARIMAResultsWrapper,
            SARIMAXResultsWrapper,
            VARResultsWrapper,
            ARCHModelResult,
        ],
        criterion: str = "aic",
    ) -> float:
        """
        Get information criterion from fitted model.

        Parameters
        ----------
        model : Model result object
            The fitted model
        criterion : str
            Information criterion ('aic', 'bic', 'hqic')

        Returns
        -------
        float
            Criterion value
        """
        if model is None:
            raise ValueError("Model must be fitted first.")

        if criterion == "aic":
            return model.aic if hasattr(model, "aic") else np.inf
        elif criterion == "bic":
            return model.bic if hasattr(model, "bic") else np.inf
        elif criterion == "hqic":
            return model.hqic if hasattr(model, "hqic") else np.inf
        else:
            raise ValueError(f"Unknown criterion: {criterion}")


class TSFitHelperService:
    """Service for TSFit helper operations."""

    @staticmethod
    def get_residuals(
        model: Union[
            AutoRegResultsWrapper,
            ARIMAResultsWrapper,
            SARIMAXResultsWrapper,
            VARResultsWrapper,
            ARCHModelResult,
        ],
        standardize: bool = False,
    ) -> np.ndarray:
        """
        Extract residuals from fitted model.

        Parameters
        ----------
        model : Model result object
            The fitted model
        standardize : bool
            Whether to standardize residuals

        Returns
        -------
        np.ndarray
            Residuals
        """
        if model is None:
            raise ValueError("Model must be fitted first.")

        if hasattr(model, "resid"):
            residuals = model.resid
        elif hasattr(model, "residuals"):
            residuals = model.residuals
        else:
            raise AttributeError("Model has no residuals attribute.")

        # Ensure numpy array
        residuals = np.asarray(residuals)

        if standardize:
            std = np.std(residuals)
            if std > 0:
                residuals = residuals / std

        # Ensure 2D shape for consistency with original
        if residuals.ndim == 1:
            residuals = residuals.reshape(-1, 1)

        return residuals

    @staticmethod
    def get_fitted_values(
        model: Union[
            AutoRegResultsWrapper,
            ARIMAResultsWrapper,
            SARIMAXResultsWrapper,
            VARResultsWrapper,
            ARCHModelResult,
        ],
    ) -> np.ndarray:
        """
        Extract fitted values from model.

        Parameters
        ----------
        model : Model result object
            The fitted model

        Returns
        -------
        np.ndarray
            Fitted values
        """
        if model is None:
            raise ValueError("Model must be fitted first.")

        # Special handling for ARCH models
        if isinstance(model, ARCHModelResult):
            # ARCH models are volatility models, not mean models
            # For ARCH, fitted values = original data - residuals
            # The model object should have the original data
            if hasattr(model.model, "_y"):
                original_data = np.asarray(model.model._y)
                residuals = np.asarray(model.resid)
                fitted = original_data - residuals
            else:
                # Fallback: return zeros with same shape as residuals
                # This maintains the interface even if we can't compute true fitted values
                fitted = np.zeros_like(model.resid)
        elif hasattr(model, "fittedvalues"):
            fitted = np.asarray(model.fittedvalues)
        elif hasattr(model, "fitted_values"):
            fitted = np.asarray(model.fitted_values)
        else:
            raise AttributeError("Model has no fitted values attribute.")

        # Ensure 2D shape for consistency with original
        if fitted.ndim == 1:
            fitted = fitted.reshape(-1, 1)

        return fitted

    @staticmethod
    def calculate_trend_terms(model_type: str, model: Any) -> int:
        """
        Calculate the number of trend terms in a model.

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
        if model_type not in ["ar", "arima", "arma"]:
            return 0

        if hasattr(model, "model") and hasattr(model.model, "trend"):
            trend = model.model.trend
            if trend == "n":  # no trend
                return 0
            elif trend in ["c", "t"]:  # constant or time trend
                return 1
            elif trend == "ct":  # constant + time trend
                return 2

        return 0

    @staticmethod
    def check_stationarity(
        residuals: np.ndarray,
        test: str = "adf",
        significance: float = 0.05,
    ) -> Tuple[bool, float]:
        """
        Check stationarity of residuals.

        Parameters
        ----------
        residuals : np.ndarray
            Residuals to test
        test : str
            Test to use ('adf', 'kpss')
        significance : float
            Significance level

        Returns
        -------
        Tuple[bool, float]
            (is_stationary, p_value)
        """
        from statsmodels.tsa.stattools import adfuller, kpss

        if test == "adf":
            result = adfuller(residuals)
            p_value = result[1]
            # For ADF, reject null (non-stationary) if p < significance
            is_stationary = p_value < significance
        elif test == "kpss":
            result = kpss(residuals)
            p_value = result[1]
            # For KPSS, reject null (stationary) if p < significance
            is_stationary = p_value >= significance
        else:
            raise ValueError(f"Unknown test: {test}")

        return is_stationary, p_value

    def check_if_rescale_needed(self, endog: np.ndarray, model_type: str) -> Tuple[bool, dict]:
        """Check if data needs rescaling based on model type and data range.

        Parameters
        ----------
        endog : np.ndarray
            Time series data
        model_type : str
            Type of model being used

        Returns
        -------
        Tuple[bool, dict]
            (needs_rescaling, rescale_factors)
        """
        # Simple implementation: rescale if range > 1000 or very small values
        data_range = np.ptp(endog)
        data_mean = np.mean(np.abs(endog))

        needs_rescaling = data_range > 1000 or data_mean < 0.001

        rescale_factors = {}
        if needs_rescaling:
            rescale_factors["scale"] = np.std(endog)
            rescale_factors["shift"] = np.mean(endog)

        return needs_rescaling, rescale_factors

    def rescale_data(self, endog: np.ndarray, rescale_factors: dict) -> np.ndarray:
        """Rescale data to reasonable range for model fitting.

        Parameters
        ----------
        endog : np.ndarray
            Data to rescale
        rescale_factors : dict
            Dictionary with 'scale' and 'shift' factors

        Returns
        -------
        np.ndarray
            Rescaled data
        """
        if not rescale_factors:
            return endog

        scale = rescale_factors.get("scale", 1.0)
        shift = rescale_factors.get("shift", 0.0)

        # Avoid division by zero
        if scale == 0:
            scale = 1.0

        return (endog - shift) / scale

    def rescale_back_data(self, data: np.ndarray, rescale_factors: dict) -> np.ndarray:
        """Rescale predictions back to original scale.

        Parameters
        ----------
        data : np.ndarray
            Data to rescale back
        rescale_factors : dict
            Dictionary with 'scale' and 'shift' factors

        Returns
        -------
        np.ndarray
            Data in original scale
        """
        if not rescale_factors:
            return data

        scale = rescale_factors.get("scale", 1.0)
        shift = rescale_factors.get("shift", 0.0)

        return data * scale + shift
