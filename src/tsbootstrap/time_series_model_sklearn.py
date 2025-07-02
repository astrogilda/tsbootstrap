"""Sklearn-compatible interface for TimeSeriesModel."""

from typing import Any, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from tsbootstrap.time_series_model import TimeSeriesModel
from tsbootstrap.utils.types import ModelTypes, OrderTypes


class TimeSeriesModelSklearn(BaseEstimator, RegressorMixin):
    """
    Sklearn-compatible wrapper for TimeSeriesModel.

    This class provides a unified sklearn interface for fitting various time series
    models including AR, ARIMA, SARIMA, VAR, and ARCH models while maintaining
    compatibility with sklearn pipelines and tools.

    Parameters
    ----------
    model_type : ModelTypes, default "ar"
        The type of model to fit. Supported types are "ar", "arima", "sarima", "var", "arch".
    verbose : bool, default True
        Verbosity level controlling suppression.
    use_backend : bool, default False
        Whether to use the new backend system. If True, uses statsforecast
        for supported models based on feature flags.
    order : Optional[OrderTypes], default None
        Order of the model. If None, default order is used based on model type.
    seasonal_order : Optional[tuple], default None
        Seasonal order for SARIMA models.
    **kwargs
        Additional parameters passed to the underlying model.

    Attributes
    ----------
    fitted_model_ : Model result object
        The fitted time series model
    X_ : np.ndarray
        Stored training data
    y_ : Optional[np.ndarray]
        Stored exogenous variables

    Examples
    --------
    >>> from tsbootstrap.time_series_model_sklearn import TimeSeriesModelSklearn
    >>> model = TimeSeriesModelSklearn(model_type="ar", order=2)
    >>> model.fit(X_train)
    >>> predictions = model.predict()
    >>> score = model.score(X_test)
    """

    def __init__(
        self,
        model_type: ModelTypes = "ar",
        verbose: bool = True,
        use_backend: bool = False,
        order: Optional[OrderTypes] = None,
        seasonal_order: Optional[tuple] = None,
        **kwargs,
    ):
        """Initialize TimeSeriesModelSklearn."""
        self.model_type = model_type
        self.verbose = verbose
        self.use_backend = use_backend
        self.order = order
        self.seasonal_order = seasonal_order

        # Store additional model parameters
        self.model_params = kwargs

        # Set parameter names for sklearn compatibility
        self._parameter_names = ["model_type", "verbose", "use_backend", "order", "seasonal_order"]
        # Add all kwargs keys to parameter names
        self._parameter_names.extend(kwargs.keys())

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "TimeSeriesModelSklearn":
        """
        Fit the time series model.

        Parameters
        ----------
        X : np.ndarray
            Time series data (n_samples, n_features) or (n_samples,)
        y : Optional[np.ndarray]
            Exogenous variables for the model

        Returns
        -------
        self : TimeSeriesModelSklearn
            Fitted estimator
        """
        # Store training data
        self.X_ = X
        self.y_ = y

        # Create TimeSeriesModel instance
        self._ts_model = TimeSeriesModel(
            X=X,
            y=y,
            model_type=self.model_type,
            verbose=self.verbose,
            use_backend=self.use_backend,
        )

        # Fit the model
        if self.model_type == "sarima":
            self.fitted_model_ = self._ts_model.fit(
                order=self.order, seasonal_order=self.seasonal_order, **self.model_params
            )
        else:
            self.fitted_model_ = self._ts_model.fit(order=self.order, **self.model_params)

        return self

    def predict(
        self, X: Optional[np.ndarray] = None, start: Optional[int] = None, end: Optional[int] = None
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
            Predictions with shape (n_samples, n_features)
        """
        check_is_fitted(self, "fitted_model_")

        # Set defaults if not provided
        if start is None or end is None:
            if hasattr(self.fitted_model_, "nobs"):
                n_obs = self.fitted_model_.nobs
            elif hasattr(self.fitted_model_, "_nobs"):
                n_obs = self.fitted_model_._nobs
            else:
                # For ARCH models
                n_obs = len(self.fitted_model_.resid)

            if start is None:
                start = 0
            if end is None:
                end = n_obs - 1

        # Handle different model types
        if self.model_type == "var":
            if X is None:
                raise ValueError("X is required for VAR model prediction.")
            steps = len(X) if end is None else end - (start or 0)
            predictions = self.fitted_model_.forecast(X, steps=steps)

        elif self.model_type == "arch":
            # ARCH models have different prediction interface
            predictions = self.fitted_model_.forecast(
                horizon=end - (start or 0) if end else 1
            ).mean.values

        else:
            # AR, ARIMA, SARIMA models
            predictions = self.fitted_model_.predict(start=start, end=end)

        # Ensure numpy array and consistent shape
        if hasattr(predictions, "values"):
            predictions = predictions.values

        predictions = np.asarray(predictions)

        # Ensure consistent output shape
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        elif predictions.ndim > 2:
            predictions = predictions.reshape(predictions.shape[0], -1)

        return predictions

    def forecast(self, steps: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate out-of-sample forecasts.

        Parameters
        ----------
        steps : int, default 1
            Number of steps to forecast
        X : Optional[np.ndarray]
            Data for VAR model forecast

        Returns
        -------
        np.ndarray
            Forecasts with shape (steps, n_features)
        """
        check_is_fitted(self, "fitted_model_")

        if self.model_type == "var":
            if X is None:
                raise ValueError("X is required for VAR model forecast.")
            forecasts = self.fitted_model_.forecast(X, steps=steps)

        elif self.model_type == "arch":
            forecasts = self.fitted_model_.forecast(horizon=steps).mean.values

        else:
            # AR, ARIMA, SARIMA models
            forecasts = self.fitted_model_.forecast(steps=steps)

        # Ensure numpy array and consistent shape
        if hasattr(forecasts, "values"):
            forecasts = forecasts.values

        forecasts = np.asarray(forecasts)

        # Ensure consistent output shape
        if forecasts.ndim == 1:
            forecasts = forecasts.reshape(-1, 1)
        elif forecasts.ndim > 2:
            forecasts = forecasts.reshape(forecasts.shape[0], -1)

        return forecasts

    def score(
        self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None, metric: str = "r2"
    ) -> float:
        """
        Score the model using various metrics.

        This method supports both sklearn interface (default R² score)
        and custom time series metrics.

        Parameters
        ----------
        X : Optional[np.ndarray]
            Ground truth data. If None, uses stored training data.
        y : Optional[np.ndarray]
            Not used, kept for sklearn compatibility
        metric : str, default "r2"
            Scoring metric. Options: 'r2', 'mse', 'mae', 'rmse', 'mape'

        Returns
        -------
        float
            Score value
        """
        check_is_fitted(self, "fitted_model_")

        # Use stored data if not provided
        if X is None:
            X = self.X_

        # Get predictions
        y_pred = self.predict()

        # Use X as ground truth
        y_true = X

        # Handle shape mismatch for scoring
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        # Ensure same length (predictions might be shorter due to lag)
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[-min_len:]
        y_pred = y_pred[-min_len:]

        # Remove NaN values that might be in predictions
        mask = ~(np.isnan(y_true).any(axis=1) | np.isnan(y_pred).any(axis=1))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            return np.nan

        # Calculate score based on metric
        if metric == "r2":
            from sklearn.metrics import r2_score

            return r2_score(y_true, y_pred)
        elif metric == "mse":
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
            raise ValueError(
                f"Unknown metric: {metric}. "
                f"Supported metrics: 'r2', 'mse', 'mae', 'rmse', 'mape'"
            )

    def get_residuals(self, standardize: bool = False) -> np.ndarray:
        """
        Get model residuals.

        Parameters
        ----------
        standardize : bool, default False
            Whether to standardize residuals

        Returns
        -------
        np.ndarray
            Residuals
        """
        check_is_fitted(self, "fitted_model_")

        if hasattr(self.fitted_model_, "resid"):
            residuals = self.fitted_model_.resid
        elif hasattr(self.fitted_model_, "residuals"):
            residuals = self.fitted_model_.residuals
        else:
            raise AttributeError("Model does not have residuals attribute")

        # Ensure numpy array
        if hasattr(residuals, "values"):
            residuals = residuals.values
        residuals = np.asarray(residuals)

        if standardize:
            std = np.std(residuals, axis=0)
            if np.any(std == 0):
                raise ValueError("Cannot standardize residuals with zero variance")
            residuals = residuals / std

        return residuals

    def get_fitted_values(self) -> np.ndarray:
        """
        Get fitted values from the model.

        Returns
        -------
        np.ndarray
            Fitted values
        """
        check_is_fitted(self, "fitted_model_")

        if hasattr(self.fitted_model_, "fittedvalues"):
            fitted = self.fitted_model_.fittedvalues
        elif hasattr(self.fitted_model_, "fitted_values"):
            fitted = self.fitted_model_.fitted_values
        else:
            # Calculate fitted values as original - residuals
            residuals = self.get_residuals()
            fitted = self.X_[-len(residuals) :] - residuals

        # Ensure numpy array
        if hasattr(fitted, "values"):
            fitted = fitted.values
        fitted = np.asarray(fitted)

        # Ensure consistent shape
        if fitted.ndim == 1:
            fitted = fitted.reshape(-1, 1)

        return fitted

    def get_information_criterion(self, criterion: str = "aic") -> float:
        """
        Get information criterion value.

        Parameters
        ----------
        criterion : str, default "aic"
            Criterion type ('aic', 'bic', 'hqic')

        Returns
        -------
        float
            Criterion value
        """
        check_is_fitted(self, "fitted_model_")

        criterion = criterion.lower()

        if criterion == "aic":
            if hasattr(self.fitted_model_, "aic"):
                return self.fitted_model_.aic
        elif criterion == "bic":
            if hasattr(self.fitted_model_, "bic"):
                return self.fitted_model_.bic
        elif criterion == "hqic":
            if hasattr(self.fitted_model_, "hqic"):
                return self.fitted_model_.hqic
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        # If attribute not found
        raise AttributeError(f"Model does not have {criterion} attribute")

    def check_residual_stationarity(
        self, test: str = "adf", significance: float = 0.05
    ) -> Tuple[bool, float]:
        """
        Check stationarity of model residuals.

        Parameters
        ----------
        test : str, default "adf"
            Statistical test to use. Options:
            - "adf": Augmented Dickey-Fuller test
            - "kpss": Kwiatkowski-Phillips-Schmidt-Shin test
        significance : float, default 0.05
            Significance level for the test

        Returns
        -------
        Tuple[bool, float]
            Tuple containing:
            - is_stationary: bool indicating whether residuals are stationary
            - p_value: float p-value from the statistical test

        Raises
        ------
        ValueError
            If test type is not recognized
        RuntimeError
            If model is not fitted

        Examples
        --------
        >>> model = TimeSeriesModelSklearn(model_type="ar", order=2)
        >>> model.fit(X_train)
        >>> is_stationary, p_value = model.check_residual_stationarity()
        >>> print(f"Stationary: {is_stationary}, p-value: {p_value:.4f}")
        """
        check_is_fitted(self, "fitted_model_")

        # Try to use backend's check_stationarity if available
        if hasattr(self.fitted_model_, "check_stationarity"):
            return self.fitted_model_.check_stationarity(test=test, significance=significance)

        # Otherwise, implement directly using residuals
        # Lazy import to handle optional dependency
        from statsmodels.tsa.stattools import adfuller, kpss

        # Get residuals
        residuals = self.get_residuals(standardize=False)

        # Handle multiple series or VAR by testing the first series
        if residuals.ndim > 1:
            residuals = residuals[:, 0]

        # Remove NaN values
        residuals = residuals[~np.isnan(residuals)]

        if len(residuals) < 10:
            # Not enough data for reliable test
            return False, 1.0

        if test.lower() == "adf":
            # Augmented Dickey-Fuller test
            # Null hypothesis: unit root exists (non-stationary)
            result = adfuller(residuals, autolag="AIC")
            p_value = result[1]
            is_stationary = p_value < significance
        elif test.lower() == "kpss":
            # KPSS test
            # Null hypothesis: series is stationary
            result = kpss(residuals, regression="c", nlags="auto")
            p_value = result[1]
            is_stationary = p_value > significance
        else:
            raise ValueError(f"Unknown test type: {test}. Use 'adf' or 'kpss'.")

        return bool(is_stationary), float(p_value)

    def _calculate_trend_terms(self) -> int:
        """
        Calculate the number of trend terms in the fitted model.

        This is a helper method that examines the model parameters to determine
        how many trend components (constant, time trend) are included.

        Returns
        -------
        int
            Number of trend terms:
            - 0: No trend
            - 1: Constant or time trend
            - 2: Both constant and time trend

        Raises
        ------
        RuntimeError
            If model is not fitted

        Examples
        --------
        >>> model = TimeSeriesModelSklearn(model_type="arima", order=(2, 1, 1))
        >>> model.fit(X_train)
        >>> n_trend = model._calculate_trend_terms()
        >>> print(f"Number of trend terms: {n_trend}")
        """
        check_is_fitted(self, "fitted_model_")

        # If fitted model has _calculate_trend_terms method, use it
        if hasattr(self.fitted_model_, "_calculate_trend_terms"):
            return self.fitted_model_._calculate_trend_terms()

        # Otherwise, check model parameters
        if hasattr(self.fitted_model_, "trend"):
            trend = self.fitted_model_.trend
            if trend == "n":  # no trend
                return 0
            elif trend in ["c", "t"]:  # constant or time trend
                return 1
            elif trend == "ct":  # constant + time trend
                return 2

        # Check for ARIMA/SARIMA models
        if self.model_type in ["arima", "sarima"]:
            # These models typically have a constant term if not explicitly disabled
            if hasattr(self.fitted_model_, "k_trend"):
                return self.fitted_model_.k_trend
            # Default to 1 if trend wasn't explicitly disabled
            return 1 if self.model_params.get("trend", "c") != "n" else 0

        # For AR models
        if self.model_type == "ar":
            # AR models from statsmodels have trend parameter
            if hasattr(self.fitted_model_, "k_trend"):
                return self.fitted_model_.k_trend
            return 1  # Default AR has constant

        # For VAR models
        if self.model_type == "var":
            if hasattr(self.fitted_model_, "k_trend"):
                return self.fitted_model_.k_trend
            return 1  # Default VAR has constant

        # For ARCH models
        if self.model_type == "arch":
            # ARCH models typically don't have trend terms in the variance equation
            # but may have them in the mean model
            if hasattr(self.fitted_model_, "model") and hasattr(self.fitted_model_.model, "mean"):
                mean_model = self.fitted_model_.model.mean
                if hasattr(mean_model, "constant"):
                    return 1 if mean_model.constant else 0
            return 0

        # Default: assume no trend
        return 0

    def summary(self) -> Any:
        """
        Get model summary.

        Returns
        -------
        Model summary object or dict
        """
        check_is_fitted(self, "fitted_model_")

        if hasattr(self.fitted_model_, "summary"):
            return self.fitted_model_.summary()
        else:
            # Return basic info if summary not available
            info = {
                "model_type": self.model_type,
                "order": self.order,
                "seasonal_order": self.seasonal_order,
            }

            # Try to add information criteria
            try:
                info["aic"] = self.get_information_criterion("aic")
            except (AttributeError, ValueError):
                pass

            try:
                info["bic"] = self.get_information_criterion("bic")
            except (AttributeError, ValueError):
                pass

            return info

    def __repr__(self) -> str:
        """String representation."""
        class_name = self.__class__.__name__
        params = []

        # Add main parameters
        params.append(f"model_type='{self.model_type}'")

        if self.verbose != True:
            params.append(f"verbose={self.verbose}")

        if self.use_backend:
            params.append(f"use_backend={self.use_backend}")

        if self.order is not None:
            params.append(f"order={self.order}")

        if self.seasonal_order is not None:
            params.append(f"seasonal_order={self.seasonal_order}")

        # Add any additional parameters
        for key, value in self.model_params.items():
            params.append(f"{key}={repr(value)}")

        return f"{class_name}({', '.join(params)})"
