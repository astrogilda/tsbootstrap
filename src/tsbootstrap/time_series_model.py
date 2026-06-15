"""
Time series model fitting: A unified interface for temporal data analysis.

This module provides a comprehensive framework for fitting various time series
models, from simple autoregressive processes to complex multivariate systems.
We've abstracted the complexities of different modeling libraries behind a
consistent interface, enabling seamless model comparison and selection.
"""

from numbers import Integral
from typing import Any, Literal, Optional  # Added Union

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from tsbootstrap.utils.odds_and_ends import suppress_output
from tsbootstrap.utils.types import ModelTypes, OrderTypes
from tsbootstrap.utils.validate import (
    validate_integers,
    validate_literal_type,
    validate_X_and_y,
)


class TimeSeriesModel(BaseEstimator, RegressorMixin):
    """
    Unified interface for time series model estimation.

    This class provides a consistent API for fitting diverse time series models,
    abstracting the underlying implementation details of various statistical
    libraries. Whether you're working with simple AR models or complex SARIMAX
    specifications, the interface remains intuitive and predictable.

    We designed this abstraction layer after experiencing the friction of
    switching between different modeling libraries, each with its own conventions
    and quirks. By standardizing the interface, we enable rapid experimentation
    and model comparison without the cognitive overhead of learning multiple APIs.
    """

    _tags = {"python_dependencies": ["arch", "statsmodels"]}

    def __init__(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        model_type: ModelTypes = "ar",
        order: Optional[int] = None,
        verbose: bool = True,
        use_backend: bool = False,
    ):
        """Initializes a TimeSeriesModel object.

        Parameters
        ----------
        X : Optional[np.ndarray], default None
            The input data. If provided, maintains backward compatibility with old API.
            For sklearn compatibility, pass data to fit() instead.
        y : Optional[np.ndarray], default None
            Optional array of exogenous variables.
        model_type : ModelTypes, default "ar"
            The type of model to fit. Supported types are "ar", "arma", "arima", "sarimax", "var", "arch".
        order : Optional[int], default None
            The order of the model. If None, will use default order for model type.
        verbose : bool, default True
            Verbosity level controlling suppression.
        use_backend : bool, default False
            Whether to use the new backend system. If True, uses statsforecast
            for supported models based on feature flags.

        Example
        -------
        >>> # Old API (backward compatibility)
        >>> time_series_model = TimeSeriesModel(X=data, model_type="ar")
        >>> results = time_series_model.fit()

        >>> # New sklearn-compatible API
        >>> time_series_model = TimeSeriesModel(model_type="ar", order=2)
        >>> results = time_series_model.fit(X)
        """
        self.model_type = model_type
        self.order = order
        self.verbose = verbose
        self.use_backend = use_backend

        # Handle both old and new API
        if X is not None:
            # Old API - data provided in constructor
            self._set_X_y(X, y)
        else:
            # New API - data will be provided in fit()
            self._X = None
            self._y = None

        self._fitted_model = None

    @property
    def model_type(self) -> ModelTypes:
        """The type of model to fit."""
        return self._model_type  # type: ignore

    @model_type.setter
    def model_type(self, value: ModelTypes) -> None:
        """Sets the type of model to fit."""
        validate_literal_type(value, ModelTypes)
        # Store the original value for sklearn compatibility
        # Only convert to lowercase internally when needed
        self._model_type = value

    @property
    def X(self) -> Optional[np.ndarray]:
        """The input data."""
        return self._X

    def _set_X_y(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Internal method to set and validate X and y."""
        self._X, self._y = validate_X_and_y(
            X,
            y,
            model_is_var=self.model_type.lower() == "var",
            model_is_arch=self.model_type.lower() == "arch",
        )

    @property
    def y(self) -> Optional[np.ndarray]:
        """Optional array of exogenous variables."""
        return self._y

    @property
    def verbose(self) -> int:
        """The verbosity level controlling suppression.

        Verbosity levels:
        - 2: No suppression (default)
        - 1: Suppress stdout only
        - 0: Suppress both stdout and stderr

        Returns
        -------
        int
            The verbosity level.
        """
        return self._verbose

    @verbose.setter
    def verbose(self, value: int) -> None:
        """Sets the verbosity level controlling suppression.

        Parameters
        ----------
        value : int
            The verbosity level. Must be one of {0, 1, 2}.

        Raises
        ------
        ValueError
            If the value is not one of {0, 1, 2}.
        """
        if value not in {0, 1, 2}:
            raise ValueError("verbose must be one of {0, 1, 2}")
        self._verbose = value

    def _fit_with_verbose_handling(self, fit_function) -> Any:
        """
        Executes the given fit function with or without suppressing standard output and error, based on the verbose attribute.

        Parameters
        ----------
        fit_function : Callable[[], Any]
            A function that represents the fitting logic.

        Returns
        -------
        Any
            The result of the function.
        """
        with suppress_output(self.verbose):
            return fit_function()

    def _validate_order(self, order, N: int, kwargs: dict) -> None:
        """
        Validates the order parameter and checks against the maximum allowed lag value.

        Parameters
        ----------
        order : Optional[Union[int, List[int]]]
            The order of the AR model or a list of order to include.
        N : int
            The length of the input data.
        kwargs : dict
            Additional keyword arguments for the AR model.

        Raises
        ------
        ValueError
            If the specified order value exceeds the allowed range.
        """
        k = self._y.shape[1] if self._y is not None else 0
        seasonal_terms, trend_parameters = self._calculate_terms(kwargs)
        max_lag = (N - k - seasonal_terms - trend_parameters) // 2  # type: ignore  # - 1

        if order is not None:
            if isinstance(order, list):
                if max(order) > max_lag:
                    raise ValueError(
                        f"Maximum allowed lag value exceeded. The allowed maximum is {max_lag}."
                    )
            else:
                if order > max_lag:
                    raise ValueError(
                        f"Maximum allowed lag value exceeded. The allowed maximum is {max_lag}."
                    )

    def _calculate_terms(self, kwargs: dict):
        """
        Calculates the number of exogenous variables, seasonal terms, and trend parameters.

        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments for the AR model.

        Returns
        -------
        Tuple[int, int]
            The number of seasonal terms and trend parameters.

        Raises
        ------
        ValueError
            If seasonal is set to True and period is not specified or if period is less than 2.
        TypeError
            If period is not an integer when seasonal is set to True.
        """
        seasonal = kwargs.get("seasonal", False)
        period = kwargs.get("period")
        if seasonal:
            if period is None:
                raise ValueError("A period must be specified when using seasonal terms.")
            elif isinstance(period, Integral):
                if period < 2:
                    raise ValueError("The seasonal period must be >= 2.")
            else:
                raise TypeError("The seasonal period must be an integer.")

        seasonal_terms = (period - 1) if seasonal and period is not None else 0
        trend_parameters = (
            1 if kwargs.get("trend", "c") == "c" else 2 if kwargs.get("trend") == "ct" else 0
        )

        return seasonal_terms, trend_parameters

    def fit_ar(self, order=None, **kwargs):
        """
        Fits an AR model to the input data.

        Parameters
        ----------
        order : Union[int, List[int]], optional
            The order of the AR model or a list of order to include.
        **kwargs
            Additional keyword arguments for the AutoReg model, including:
                - seasonal (bool): Whether to include seasonal terms in the model.
                - period (int): The seasonal period, if using seasonal terms.
                - trend (str): The trend component to include in the model.

        Returns
        -------
        AutoRegResultsWrapper
            The fitted AR model.

        Raises
        ------
        ValueError
            If an invalid period is specified for seasonal terms or if the maximum allowed lag value is exceeded.
        """
        if order is None:
            order = 1
        N = len(self._X)
        self._validate_order(order, N, kwargs)

        # Use backend system if enabled
        if self.use_backend:
            from tsbootstrap.backends.adapter import fit_with_backend

            def fit_logic():
                """Logic for fitting AR model with backend."""
                return fit_with_backend(
                    model_type="AR", endog=self._X, exog=self._y, order=order, **kwargs
                )

            return self._fit_with_verbose_handling(fit_logic)

        # Original implementation
        from statsmodels.tsa.ar_model import AutoReg

        def fit_logic():
            """Logic for fitting ARIMA model."""
            model = AutoReg(endog=self._X, lags=order, exog=self._y, **kwargs)
            model_fit = model.fit()
            return model_fit

        return self._fit_with_verbose_handling(fit_logic)

    def fit_arima(self, order=None, **kwargs):
        """Fits an ARIMA model to the input data.

        Parameters
        ----------
        order : Tuple[int, int, int], optional
            The order of the ARIMA model (p, d, q).
        **kwargs
            Additional keyword arguments for the ARIMA model, including:
                - seasonal (bool): Whether to include seasonal terms in the model.
                - period (int): The seasonal period, if using seasonal terms.
                - trend (str): The trend component to include in the model.

        Returns
        -------
        ARIMAResultsWrapper
            The fitted ARIMA model.

        Raises
        ------
        ValueError
            If an invalid period is specified for seasonal terms or if the maximum allowed lag value is exceeded.

        Notes
        -----
        The ARIMA model is fit using the statsmodels implementation. The default solver is 'lbfgs' and the default
        optimization method is 'css'. The default maximum number of iterations is 50. These values can be changed by
        passing the appropriate keyword arguments to the fit method.
        """
        if order is None:
            order = (1, 0, 0)
        if len(order) != 3:
            raise ValueError("The order must be a 3-tuple")

        # Use backend system if enabled
        if self.use_backend:
            from tsbootstrap.backends.adapter import fit_with_backend

            def fit_logic():
                """Logic for fitting ARIMA model with backend."""
                return fit_with_backend(
                    model_type="ARIMA", endog=self._X, exog=self._y, order=order, **kwargs
                )

            return self._fit_with_verbose_handling(fit_logic)

        # Original implementation
        from statsmodels.tsa.arima.model import ARIMA

        def fit_logic():
            """Logic for fitting ARIMA model."""
            model = ARIMA(endog=self._X, order=order, exog=self._y, **kwargs)
            model_fit = model.fit()
            return model_fit

        return self._fit_with_verbose_handling(fit_logic)

    def fit_sarima(self, order=None, seasonal_order=None, **kwargs):
        """Fits a SARIMA model to the input data.

        Parameters
        ----------
        order : Tuple[int, int, int], optional
            The non-seasonal order of the SARIMA model (p, d, q).
        seasonal_order : Tuple[int, int, int, int], optional
            The seasonal order of the SARIMA model (P, D, Q, s).
        **kwargs
            Additional keyword arguments for the SARIMA model, including:
                - trend (str): The trend component to include in the model.

        Returns
        -------
        SARIMAXResultsWrapper
            The fitted SARIMA model.

        Raises
        ------
        ValueError
            If an invalid order is specified.

        Notes
        -----
        The SARIMA model is fit using the statsmodels implementation. The default solver is 'lbfgs' and the default
        optimization method is 'css'. The default maximum number of iterations is 50. These values can be changed by
        passing the appropriate keyword arguments to the fit method.
        """
        if order is None:
            order = (1, 0, 0)
        if seasonal_order is None:
            seasonal_order = (0, 0, 0, 0)  # Default to no seasonality

        if len(order) != 3:
            raise ValueError("The non-seasonal order must be a 3-tuple (p, d, q).")
        if len(seasonal_order) != 4:
            raise ValueError("The seasonal order must be a 4-tuple (P, D, Q, s).")

        # Validate orders
        validate_integers(*order, min_value=0)
        validate_integers(*seasonal_order, min_value=0)

        # Check seasonal period
        if seasonal_order[3] <= 1 and any(s > 0 for s in seasonal_order[:3]):
            raise ValueError(
                "Seasonal period 's' must be greater than 1 if seasonal components (P, D, Q) are non-zero."
            )

        # Check for duplication of order (p >= s and P != 0)
        if order[0] >= seasonal_order[3] and seasonal_order[0] != 0 and seasonal_order[3] > 0:
            raise ValueError(
                f"The non-seasonal autoregressive term 'p' ({order[0]}) is greater than or equal to the seasonal period 's' ({seasonal_order[3]}) while the seasonal autoregressive term 'P' is not zero ({seasonal_order[0]}). This could lead to duplication of order."
            )

        # Check for duplication of order (q >= s and Q != 0)
        if order[2] >= seasonal_order[3] and seasonal_order[2] != 0 and seasonal_order[3] > 0:
            raise ValueError(
                f"The non-seasonal moving average term 'q' ({order[2]}) is greater than or equal to the seasonal period 's' ({seasonal_order[3]}) while the seasonal moving average term 'Q' is not zero ({seasonal_order[2]}). This could lead to duplication of order."
            )

        # Use backend system if enabled
        if self.use_backend:
            from tsbootstrap.backends.adapter import fit_with_backend

            def fit_logic():
                """Logic for fitting SARIMA model with backend."""
                return fit_with_backend(
                    model_type="SARIMA",
                    endog=self._X,
                    exog=self._y,
                    order=order,
                    seasonal_order=seasonal_order,
                    **kwargs,
                )

            return self._fit_with_verbose_handling(fit_logic)

        # Original implementation
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        def fit_logic():
            model = SARIMAX(
                endog=self._X,
                order=order,
                seasonal_order=seasonal_order,
                exog=self._y,
                **kwargs,
            )
            model_fit = model.fit(disp=-1)
            return model_fit

        return self._fit_with_verbose_handling(fit_logic)

    def fit_var(self, order: Optional[int] = None, **kwargs):
        """Fits a Vector Autoregression (VAR) model to the input data.

        Parameters
        ----------
        order : int, optional
            The number of order to include in the VAR model.
        **kwargs
            Additional keyword arguments for the VAR model.

        Raises
        ------
        ValueError
            If the maximum allowed lag value is exceeded.

        Returns
        -------
        VARResultsWrapper
            The fitted VAR model.

        Notes
        -----
        The VAR model is fit using the statsmodels implementation. The default solver is 'bfgs' and the default
        optimization method is 'css'. The default maximum number of iterations is 50. These values can be changed by
        passing the appropriate keyword arguments to the fit method.
        """

        def fit_logic():
            """Logic for fitting ARIMA model."""
            from statsmodels.tsa.vector_ar.var_model import VAR

            model = VAR(endog=self._X, exog=self._y)
            model_fit = model.fit(**kwargs)
            return model_fit

        return self._fit_with_verbose_handling(fit_logic)

    def fit_arch(
        self,
        order: Optional[int] = None,
        p: int = 1,
        q: int = 1,
        arch_model_type: Literal["GARCH", "EGARCH", "TARCH", "AGARCH"] = "GARCH",
        mean_type: Literal["zero", "AR"] = "zero",
        **kwargs,
    ):
        """
        Fits a GARCH, GARCH-M, EGARCH, TARCH, or AGARCH model to the input data.

        Parameters
        ----------
        order : int, optional
            The number of order to include in the AR part of the model.
        p : int, default 1
            The number of order in the GARCH part of the model.
        q : int, default 1
            The number of order in the ARCH part of the model.
        arch_model_type : Literal["GARCH", "EGARCH", "TARCH", "AGARCH"], default "GARCH"
            The type of GARCH model to fit.
        mean_type : Literal["zero", "AR"], default "zero"
            The type of mean model to use.
        **kwargs
            Additional keyword arguments for the ARCH model.

        Returns
        -------
            The fitted GARCH model.

        Raises
        ------
        ValueError
            If the maximum allowed lag value is exceeded or if an invalid arch_model_type is specified.
        """
        from arch import arch_model

        if order is None:
            order = 1

        # Assuming a validate_X_and_y function exists for data validation
        validate_integers(p, q, order, min_value=1)  # type: ignore

        if mean_type not in ["zero", "AR"]:
            raise ValueError("mean_type must be one of 'zero' or 'AR'")

        if arch_model_type in ["GARCH", "EGARCH"]:
            model = arch_model(
                y=self._X,
                x=self._y,
                mean=mean_type,
                lags=order,
                vol=arch_model_type,  # type: ignore
                p=p,
                q=q,
                **kwargs,
            )
        elif arch_model_type == "TARCH":
            model = arch_model(
                y=self._X,
                x=self._y,
                mean=mean_type,
                lags=order,
                vol="GARCH",
                p=p,
                o=1,
                q=q,
                power=1,
                **kwargs,
            )
        elif arch_model_type == "AGARCH":
            model = arch_model(
                y=self._X,
                x=self._y,
                mean=mean_type,
                lags=order,
                vol="GARCH",
                p=p,
                o=1,
                q=q,
                **kwargs,
            )
        else:
            raise ValueError(
                "arch_model_type must be one of 'GARCH', 'EGARCH', 'TARCH', or 'AGARCH'"
            )

        options = {"maxiter": 200}

        def fit_logic(model=model, options=options):
            """Logic for fitting ARIMA model."""
            model_fit = model.fit(disp="off", options=options)
            return model_fit

        return self._fit_with_verbose_handling(fit_logic)

    def _fit_model(self, order: OrderTypes = None, seasonal_order: Optional[tuple] = None, **kwargs):  # type: ignore
        """Fits a time series model to the input data.

        Parameters
        ----------
        order : OrderTypes, optional
            The order of the model. If not specified, the default order for the selected model type is used.
        seasonal_order : Optional[tuple], optional
            The seasonal order of the model for SARIMA.
        **kwargs
            Additional keyword arguments for the model.

        Returns
        -------
            The fitted time series model.

        Raises
        ------
        ValueError
            If an invalid order is specified for the model type.
        """
        fitted_models = {
            "ar": self.fit_ar,
            "arima": self.fit_arima,
            "sarima": self.fit_sarima,
            "var": self.fit_var,
            "arch": self.fit_arch,
        }
        model_type_lower = self.model_type.lower()
        if model_type_lower in fitted_models:
            if model_type_lower == "sarima":
                return fitted_models[model_type_lower](
                    order=order, seasonal_order=seasonal_order, **kwargs
                )
            else:
                return fitted_models[model_type_lower](order=order, **kwargs)
        raise ValueError(f"Unsupported fitted model type {self.model_type}.")

    def fit(self, X=None, y=None, order: OrderTypes = None, seasonal_order: Optional[tuple] = None, **kwargs):  # type: ignore
        """Fit method supporting both old and new API.

        This method maintains backward compatibility by accepting order parameters
        like the old API, while also supporting the sklearn pattern when called
        with X parameter.

        Parameters
        ----------
        X : np.ndarray, optional
            For sklearn compatibility. If provided, this is the input time series data.
        y : np.ndarray, optional
            For sklearn compatibility. Optional exogenous variables.
        order : OrderTypes, optional
            The order of the model. If not specified, uses the order from constructor
            or the default order for the selected model type.
        seasonal_order : Optional[tuple], optional
            The seasonal order of the model for SARIMA.
        **kwargs
            Additional keyword arguments for the model.

        Returns
        -------
            For backward compatibility: The fitted time series model if called without X.
            For sklearn compatibility: self if called with X.
        """
        # Detect which API is being used
        if X is not None:
            # Sklearn API - X passed as parameter
            self._set_X_y(X, y)
            # Use provided order or the one from constructor
            if order is None:
                order = self.order
            # Fit the model
            self._fitted_model = self._fit_model(
                order=order, seasonal_order=seasonal_order, **kwargs
            )
            # Return self for sklearn compatibility
            return self
        else:
            # Old API - X should already be set in constructor
            if self._X is None:
                raise ValueError("No data provided. Pass X to constructor or use sklearn pattern.")

            # Use provided order or the one from constructor
            if order is None:
                order = self.order

            # Fit the model using the existing fit method
            self._fitted_model = self._fit_model(
                order=order, seasonal_order=seasonal_order, **kwargs
            )

            # For backward compatibility, return the fitted model object that has forecast() method
            return self._fitted_model

    def predict(self, n_periods: int = 1) -> np.ndarray:
        """Generate predictions from the fitted model.

        Parameters
        ----------
        n_periods : int, default 1
            Number of periods to forecast.

        Returns
        -------
        np.ndarray
            Forecasted values.
        """
        check_is_fitted(self, "_fitted_model")

        if self._fitted_model is None:
            raise NotFittedError("Model must be fitted before making predictions.")

        # Use the predict method of the fitted model
        if hasattr(self._fitted_model, "predict"):
            # For statsmodels ARIMA/SARIMAX models
            return self._fitted_model.predict(start=len(self._X), end=len(self._X) + n_periods - 1)
        elif hasattr(self._fitted_model, "forecast"):
            # For statsmodels AR, VAR, and other models
            if self.model_type.lower() == "var":
                # VAR models need the last observations
                last_obs = self._X[-self.order :] if isinstance(self.order, int) else self._X[-2:]
                return self._fitted_model.forecast(y=last_obs, steps=n_periods)
            else:
                return self._fitted_model.forecast(steps=n_periods)
        else:
            raise AttributeError("Fitted model does not have a predict or forecast method.")

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Sklearn-compatible score method.

        Parameters
        ----------
        X : np.ndarray
            Test data.
        y : np.ndarray
            Target values.

        Returns
        -------
        float
            R² score.
        """
        from sklearn.metrics import r2_score

        # Generate predictions for the length of y
        predictions = self.predict(n_periods=len(y))

        # Ensure same shape
        if predictions.shape != y.shape:
            predictions = predictions[: len(y)]

        return r2_score(y, predictions)

    def __repr__(self) -> str:
        return f"TimeSeriesModel(model_type={self.model_type}, verbose={self.verbose})"

    def __str__(self) -> str:
        return f"TimeSeriesModel using model_type={self.model_type} with verbosity level {self.verbose}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TimeSeriesModel):
            # Check X equality
            x_equal = (
                np.array_equal(self._X, other._X)
                if (self._X is not None and other._X is not None)
                else (self._X is None and other._X is None)
            )

            # Check y equality
            y_equal = (
                np.array_equal(self._y, other._y)
                if (self._y is not None and other._y is not None)
                else (self._y is None and other._y is None)
            )

            return (
                x_equal
                and y_equal
                and self.model_type == other.model_type
                and self.verbose == other.verbose
                and self.order == other.order
            )
        return False
