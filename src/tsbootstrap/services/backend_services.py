"""Backend-compatible services for time series operations.

This module provides services that work with any backend implementing the
ModelBackend protocol, offering enhanced functionality beyond the base protocol.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tsbootstrap.backends.protocol import FittedModelBackend, ModelBackend
from tsbootstrap.utils.types import OrderTypes


class BackendValidationService:
    """Service for backend-agnostic validation operations."""

    @staticmethod
    def validate_model_config(
        backend: ModelBackend,
        model_type: Optional[str] = None,
        order: Optional[OrderTypes] = None,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Validate model configuration for a backend.

        Parameters
        ----------
        backend : ModelBackend
            The backend to validate configuration for
        model_type : Optional[str]
            Type of model (backend-specific)
        order : Optional[OrderTypes]
            Model order configuration
        seasonal_order : Optional[Tuple[int, int, int, int]]
            Seasonal order for seasonal models
        **kwargs : Any
            Additional backend-specific parameters

        Returns
        -------
        Dict[str, Any]
            Validated configuration dict

        Raises
        ------
        TypeError
            If configuration types are invalid
        ValueError
            If configuration values are invalid
        """
        config = {}

        # Validate model type if provided
        if model_type is not None:
            if not isinstance(model_type, str):
                raise TypeError(f"Model type must be string, got {type(model_type).__name__}")
            config["model_type"] = model_type

        # Validate order if provided
        if order is not None:
            validated_order = BackendValidationService._validate_order(order, model_type)
            config["order"] = validated_order

        # Validate seasonal order if provided
        if seasonal_order is not None:
            validated_seasonal = BackendValidationService._validate_seasonal_order(
                seasonal_order, model_type
            )
            config["seasonal_order"] = validated_seasonal

        # Add any additional kwargs
        config.update(kwargs)

        return config

    @staticmethod
    def _validate_order(value: OrderTypes, model_type: Optional[str] = None) -> OrderTypes:
        """
        Validate order parameter.

        Parameters
        ----------
        value : OrderTypes
            The order value to validate
        model_type : Optional[str]
            The type of model being used

        Returns
        -------
        OrderTypes
            The validated order

        Raises
        ------
        TypeError
            If the order type is invalid
        ValueError
            If the order value is invalid
        """
        from numbers import Integral

        # None is valid for some models
        if value is None:
            return value

        # Single integer order
        if isinstance(value, Integral):
            if value < 0:
                raise ValueError(f"Order must be non-negative. Got {value}.")
            return value

        # List or tuple order
        if isinstance(value, (list, tuple)):
            # Convert to tuple
            value = tuple(value)

            # Validate all elements are non-negative integers
            for i, v in enumerate(value):
                if not isinstance(v, Integral) or v < 0:
                    raise ValueError(
                        f"All order elements must be non-negative integers. Element {i} is {v}."
                    )

            # Validate length (3 for ARIMA, 4 for seasonal)
            if len(value) not in [2, 3, 4]:
                raise ValueError(f"Order tuple must have 2, 3, or 4 elements. Got {len(value)}.")

            return value

        raise TypeError(f"Invalid order type: {type(value).__name__}")

    @staticmethod
    def _validate_seasonal_order(
        value: Optional[Tuple[int, int, int, int]], model_type: Optional[str] = None
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Validate seasonal order.

        Parameters
        ----------
        value : Optional[Tuple[int, int, int, int]]
            The seasonal order (P, D, Q, s)
        model_type : Optional[str]
            The type of model

        Returns
        -------
        Optional[Tuple[int, int, int, int]]
            The validated seasonal order

        Raises
        ------
        ValueError
            If seasonal order is invalid
        """
        if value is None:
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


class BackendPredictionService:
    """Service for backend-agnostic prediction operations."""

    def predict(
        self,
        fitted_backend: FittedModelBackend,
        start: Optional[int] = None,
        end: Optional[int] = None,
        steps: Optional[int] = None,
        X: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate predictions from fitted backend.

        Parameters
        ----------
        fitted_backend : FittedModelBackend
            The fitted backend
        start : Optional[int]
            Start index for prediction
        end : Optional[int]
            End index for prediction
        steps : Optional[int]
            Number of steps to predict (alternative to end)
        X : Optional[np.ndarray]
            Exogenous variables for prediction

        Returns
        -------
        np.ndarray
            Predictions
        """
        # Calculate steps from start/end if needed
        if steps is None:
            if end is not None and start is not None:
                steps = end - start + 1
            elif end is not None:
                steps = end + 1
            else:
                steps = 1

        # Use backend's predict method
        predictions = fitted_backend.predict(steps=steps, X=X)

        # Handle start offset if needed
        if start is not None and start > 0:
            # For in-sample prediction, we might need to return fitted values
            fitted_vals = fitted_backend.fitted_values
            if start < len(fitted_vals):
                # Mix fitted values and predictions
                n_fitted = min(len(fitted_vals) - start, steps)
                result = np.empty(steps)
                result[:n_fitted] = fitted_vals[start : start + n_fitted]
                if n_fitted < steps:
                    result[n_fitted:] = predictions[: steps - n_fitted]
                return result

        return predictions

    def forecast(
        self,
        fitted_backend: FittedModelBackend,
        steps: int = 1,
        X: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate out-of-sample forecasts.

        Parameters
        ----------
        fitted_backend : FittedModelBackend
            The fitted backend
        steps : int
            Number of steps to forecast
        X : Optional[np.ndarray]
            Exogenous variables for forecast

        Returns
        -------
        np.ndarray
            Forecasts
        """
        # Direct delegation to backend's predict
        return fitted_backend.predict(steps=steps, X=X)


class BackendScoringService:
    """Service for backend-agnostic scoring operations."""

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
            Scoring metric ('mse', 'mae', 'rmse', 'mape', 'r2')

        Returns
        -------
        float
            Score value
        """
        # Ensure same shape
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

        # Handle different metrics
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
        elif metric == "r2":
            # R-squared calculation
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            if ss_tot == 0:
                return 1.0 if ss_res == 0 else -np.inf
            return 1 - (ss_res / ss_tot)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def get_information_criteria(
        self,
        fitted_backend: FittedModelBackend,
        criterion: str = "aic",
    ) -> float:
        """
        Get information criterion from fitted backend.

        Parameters
        ----------
        fitted_backend : FittedModelBackend
            The fitted backend
        criterion : str
            Information criterion ('aic', 'bic', 'hqic')

        Returns
        -------
        float
            Criterion value
        """
        # Use backend's method
        criteria = fitted_backend.get_info_criteria()

        if criterion not in criteria:
            raise ValueError(f"Criterion '{criterion}' not available from backend")

        return criteria[criterion]


class BackendHelperService:
    """Service for backend-agnostic helper operations."""

    @staticmethod
    def get_residuals(
        fitted_backend: FittedModelBackend,
        standardize: bool = False,
    ) -> np.ndarray:
        """
        Extract residuals from fitted backend.

        Parameters
        ----------
        fitted_backend : FittedModelBackend
            The fitted backend
        standardize : bool
            Whether to standardize residuals

        Returns
        -------
        np.ndarray
            Residuals
        """
        residuals = fitted_backend.residuals

        if standardize:
            std = np.std(residuals)
            if std > 0:
                residuals = residuals / std

        return residuals

    @staticmethod
    def get_fitted_values(fitted_backend: FittedModelBackend) -> np.ndarray:
        """
        Extract fitted values from backend.

        Parameters
        ----------
        fitted_backend : FittedModelBackend
            The fitted backend

        Returns
        -------
        np.ndarray
            Fitted values
        """
        return fitted_backend.fitted_values

    @staticmethod
    def calculate_trend_terms(fitted_backend: FittedModelBackend) -> int:
        """
        Calculate the number of trend terms in a model.

        Parameters
        ----------
        fitted_backend : FittedModelBackend
            The fitted backend

        Returns
        -------
        int
            Number of trend terms
        """
        # Check if backend has trend information in params
        params = fitted_backend.params

        # Look for trend indicators in params
        if "trend" in params:
            trend = params["trend"]
            if trend == "n":  # no trend
                return 0
            elif trend in ["c", "t"]:  # constant or time trend
                return 1
            elif trend == "ct":  # constant + time trend
                return 2

        # Check for intercept/const in params
        if "const" in params or "intercept" in params:
            return 1

        return 0

    @staticmethod
    def check_stationarity(
        fitted_backend: FittedModelBackend,
        test: str = "adf",
        significance: float = 0.05,
    ) -> Tuple[bool, float]:
        """
        Check stationarity of residuals.

        Parameters
        ----------
        fitted_backend : FittedModelBackend
            The fitted backend
        test : str
            Test to use ('adf', 'kpss')
        significance : float
            Significance level

        Returns
        -------
        Tuple[bool, float]
            (is_stationary, p_value)
        """
        # Use backend's method directly
        return fitted_backend.check_stationarity(test=test, significance=significance)

    @staticmethod
    def validate_predictions_shape(
        predictions: np.ndarray,
        expected_shape: Optional[Tuple[int, ...]] = None,
        ensure_2d: bool = False,
    ) -> np.ndarray:
        """
        Validate and reshape predictions.

        Parameters
        ----------
        predictions : np.ndarray
            Predictions to validate
        expected_shape : Optional[Tuple[int, ...]]
            Expected shape
        ensure_2d : bool
            Whether to ensure 2D output

        Returns
        -------
        np.ndarray
            Validated predictions
        """
        # Ensure numpy array
        predictions = np.asarray(predictions)

        # Check expected shape
        if expected_shape is not None and predictions.shape != expected_shape:
            # Try to reshape if possible
            if np.prod(predictions.shape) == np.prod(expected_shape):
                predictions = predictions.reshape(expected_shape)
            else:
                raise ValueError(
                    f"Cannot reshape predictions from {predictions.shape} to {expected_shape}"
                )

        # Ensure 2D if requested
        if ensure_2d and predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        return predictions


class BackendCompositeService:
    """Composite service that combines all backend services."""

    def __init__(self):
        """Initialize composite service with all sub-services."""
        self.validation = BackendValidationService()
        self.prediction = BackendPredictionService()
        self.scoring = BackendScoringService()
        self.helper = BackendHelperService()

    def validate_and_fit(
        self,
        backend: ModelBackend,
        y: np.ndarray,
        X: Optional[np.ndarray] = None,
        model_type: Optional[str] = None,
        order: Optional[OrderTypes] = None,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        **kwargs: Any,
    ) -> FittedModelBackend:
        """
        Validate configuration and fit model.

        Parameters
        ----------
        backend : ModelBackend
            The backend to use
        y : np.ndarray
            Time series data
        X : Optional[np.ndarray]
            Exogenous variables
        model_type : Optional[str]
            Model type
        order : Optional[OrderTypes]
            Model order
        seasonal_order : Optional[Tuple[int, int, int, int]]
            Seasonal order
        **kwargs : Any
            Additional parameters

        Returns
        -------
        FittedModelBackend
            Fitted model
        """
        # Validate configuration
        config = self.validation.validate_model_config(
            backend=backend,
            model_type=model_type,
            order=order,
            seasonal_order=seasonal_order,
            **kwargs,
        )

        # Fit model with validated config
        return backend.fit(y=y, X=X, **config)

    def evaluate_model(
        self,
        fitted_backend: FittedModelBackend,
        y_test: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        metrics: Optional[List[str]] = None,
        n_ahead: int = 1,
    ) -> Dict[str, float]:
        """
        Comprehensive model evaluation.

        Parameters
        ----------
        fitted_backend : FittedModelBackend
            Fitted model to evaluate
        y_test : Optional[np.ndarray]
            Test data for out-of-sample evaluation
        X_test : Optional[np.ndarray]
            Test exogenous variables
        metrics : Optional[List[str]]
            List of metrics to compute
        n_ahead : int
            Steps ahead for forecast evaluation

        Returns
        -------
        Dict[str, float]
            Dictionary of metric values
        """
        if metrics is None:
            metrics = ["mse", "mae", "rmse", "r2"]

        results = {}

        # In-sample metrics using fitted values
        y_fitted = fitted_backend.fitted_values
        y_train = y_fitted  # Assuming we have access to training data through fitted values

        # Get residuals for in-sample evaluation
        residuals = fitted_backend.residuals
        n_obs = len(residuals)

        # Reconstruct training data from fitted values and residuals
        # This assumes additive model: y = fitted + residual
        y_train_reconstructed = y_fitted + residuals

        for metric in metrics:
            try:
                in_sample_score = self.scoring.score(
                    y_true=y_train_reconstructed,
                    y_pred=y_fitted,
                    metric=metric,
                )
                results[f"in_sample_{metric}"] = in_sample_score
            except Exception:
                # Skip if metric calculation fails
                pass

        # Out-of-sample metrics if test data provided
        if y_test is not None:
            y_pred = self.prediction.forecast(fitted_backend, steps=len(y_test), X=X_test)

            # Ensure shapes match
            if y_pred.shape != y_test.shape:
                y_pred = self.helper.validate_predictions_shape(y_pred, expected_shape=y_test.shape)

            for metric in metrics:
                try:
                    out_sample_score = self.scoring.score(
                        y_true=y_test, y_pred=y_pred, metric=metric
                    )
                    results[f"out_sample_{metric}"] = out_sample_score
                except Exception:
                    # Skip if metric calculation fails
                    pass

        # Information criteria
        try:
            info_criteria = fitted_backend.get_info_criteria()
            results.update(info_criteria)
        except Exception:
            # Skip if not available
            pass

        # Stationarity test
        try:
            is_stationary, p_value = fitted_backend.check_stationarity()
            results["residuals_stationary"] = is_stationary
            results["residuals_stationarity_pvalue"] = p_value
        except Exception:
            # Skip if not available
            pass

        return results
