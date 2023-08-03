from __future__ import annotations

import warnings
from numbers import Integral

import numpy as np
from arch.univariate.base import ARCHModelResult
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_is_fitted
from statsmodels.tsa.ar_model import AutoRegResultsWrapper
from statsmodels.tsa.arima.model import ARIMAResultsWrapper
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper
from utils.types import (
    FittedModelType,
    ModelTypes,
    OrderTypes,
    OrderTypesWithoutNone,
)
from utils.validate import validate_literal_type, validate_X_and_exog

from src.time_series_model import TimeSeriesModel


class TSFit(BaseEstimator, RegressorMixin):
    """
    Performs fitting for various time series models including 'ar', 'arima', 'sarima', 'var', and 'arch'.

    Attributes
    ----------
    model_type : str
        Type of the model.
    order : int, List[int], Tuple[int, int, int], Tuple[int, int, int, int]
        Order of the model.
    rescale_factors : dict
        Rescaling factors for the input data and exogenous variables.
    model : Union[AutoRegResultsWrapper, ARIMAResultsWrapper, SARIMAXResultsWrapper, VARResultsWrapper, ARCHModelResult]
        The fitted model.
    model_params : dict
        Additional parameters to be passed to the model.

    Methods
    -------
    fit(X, exog=None)
        Fit the chosen model to the data.
    get_coefs()
        Return the coefficients of the fitted model.
    get_intercepts()
        Return the intercepts of the fitted model.
    get_residuals()
        Return the residuals of the fitted model.
    get_fitted_X()
        Return the fitted values of the model.
    get_order()
        Return the order of the fitted model.
    predict(X, n_steps=1)
        Predict future values using the fitted model.
    score(X, y_true)
        Compute the R-squared score for the fitted model.

    Raises
    ------
    ValueError
        If the model type or the model order is invalid.

    Notes
    -----
    The following table shows the valid model types and their corresponding orders.

    +--------+-------------------+-------------------+
    | Model  | Valid orders      | Invalid orders    |
    +========+===================+===================+
    | 'ar'   | int               | list, tuple       |
    +--------+-------------------+-------------------+
    | 'arima'| tuple of length 3 | int, list, tuple  |
    +--------+-------------------+-------------------+
    | 'sarima'| tuple of length 4| int, list, tuple  |
    +--------+-------------------+-------------------+
    | 'var'  | int               | list, tuple       |
    +--------+-------------------+-------------------+
    | 'arch' | int               | list, tuple       |
    +--------+-------------------+-------------------+

    Examples
    --------
    >>> from src.tsfit import TSFit
    >>> import numpy as np
    >>> X = np.random.normal(size=(100, 1))
    >>> fit_obj = TSFit(order=2, model_type='ar')
    >>> fit_obj.fit(X)
    TSFit(order=2, model_type='ar')
    >>> fit_obj.get_coefs()
    array([[ 0.003, -0.002]])
    >>> fit_obj.get_intercepts()
    array([0.001])
    >>> fit_obj.get_residuals()
    array([[ 0.001],
              [-0.002],
                [-0.002],
                    [-0.002],
                        [-0.002], ...
    >>> fit_obj.get_fitted_X()
    array([[ 0.001],
                [-0.002],
                    [-0.002],
                        [-0.002],
                            [-0.002], ...
    >>> fit_obj.get_order()
    2
    >>> fit_obj.predict(X, n_steps=5)
    array([[ 0.001],
                [-0.002],
                    [-0.002],
                        [-0.002],
                            [-0.002], ...
    >>> fit_obj.score(X, X)
    0.999
    """

    def __init__(
        self, order: OrderTypesWithoutNone, model_type: ModelTypes, **kwargs
    ) -> None:
        self.model_type = model_type
        self.order = order
        self.rescale_factors = {}
        self.model = None
        self.model_params = kwargs

    @property
    def model_type(self) -> str:
        """The type of the model."""
        return self._model_type

    @model_type.setter
    def model_type(self, value: ModelTypes) -> None:
        """Set the model type."""
        value = value.lower()
        validate_literal_type(value, ModelTypes)
        self._model_type = value

    @property
    def order(self) -> OrderTypesWithoutNone:
        """The order of the model."""
        return self._order

    @order.setter
    def order(self, value) -> None:
        """Set the order of the model."""
        if isinstance(value, tuple) and self.model_type not in [
            "arima",
            "sarima",
        ]:
            raise ValueError(
                f"Invalid order '{value}', should be an integer for model type '{self.model_type}'"
            )

        if isinstance(value, Integral) and self.model_type in {
            "sarima",
            "arima",
        }:
            if self.model_type == "sarima":
                value = (value, 0, 0, 2)
                warning_msg = f"{self.model_type.upper()} model requires a tuple of order (p, d, q, s), where d is the order of differencing and s is the seasonal period. Setting d=0, q=0, and s=2."
            else:
                value = (value, 0, 0)
                warning_msg = f"{self.model_type.upper()} model requires a tuple of order (p, d, q), where d is the order of differencing. Setting d=0, q=0."
            warnings.warn(warning_msg, stacklevel=2)

        self._order = value

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Args:
            deep: When set to True, will return the parameters for this estimator and contained subobjects that are estimators.

        Returns
        -------
            params: Dictionary of parameter names mapped to their values.
        """
        return {
            "order": self.order,
            "model_type": self.model_type,
            **self.model_params,
        }

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Args:
            **params: Dictionary of parameter names mapped to their values.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.model_params[key] = value
        return self

    def __repr__(self):
        """
        Official string representation of a TSFit object.
        """
        return f"TSFit(order={self.order}, model_type='{self.model_type}')"

    def fit(
        self, X: np.ndarray, exog: np.ndarray | None = None
    ) -> FittedModelType:
        """
        Fit the chosen model to the data.

        Args:
            X: The input data.
            exog: Exogenous variables, optional.

        Raises
        ------
            ValueError: If the model type or the model order is invalid.
        """
        # Check if the input shapes are valid
        validate_X_and_exog(
            X,
            exog,
            model_is_var=self.model_type == "var",
            model_is_arch=self.model_type == "arch",
        )

        def _rescale_inputs(
            X: np.ndarray, exog: np.ndarray | None = None
        ) -> tuple[
            np.ndarray, np.ndarray | None, tuple[float, list[float] | None]
        ]:
            """
            Rescale the inputs to ensure that the variance of the input data is within the interval [1, 1000].

            Parameters
            ----------
            X : np.ndarray
                The input data.
            exog : np.ndarray, optional
                The exogenous variables, by default None.

            Returns
            -------
            Tuple[np.ndarray, np.ndarray, Tuple[float, List[float] | None]]
                A tuple containing the rescaled input data, the rescaled exogenous variables, and the rescaling factors used.

            Raises
            ------
            RuntimeError
                If the maximum number of iterations is reached before the variance is within the desired range.
            """

            def rescale_array(
                arr: np.ndarray, max_iter: int = 100
            ) -> tuple[np.ndarray, float]:
                """
                Iteratively rescales an array to ensure its variance is within the interval [1, 1000].

                Parameters
                ----------
                arr : np.ndarray
                    The input array to be rescaled.
                max_iter : int, optional
                    The maximum number of iterations for rescaling, by default 100.

                Returns
                -------
                Tuple[np.ndarray, float]
                    A tuple containing the rescaled array and the total rescaling factor used.

                Raises
                ------
                RuntimeError
                    If the maximum number of iterations is reached before the variance is within the desired range.
                """
                variance = np.var(arr)
                total_rescale_factor = 1
                iterations = 0

                while not 1 <= variance <= 1000:
                    if iterations >= max_iter:
                        raise RuntimeError(
                            f"Maximum iterations ({max_iter}) reached. Variance is still not in the range [1, 1000]."
                        )

                    rescale_factor = np.sqrt(100 / variance)
                    arr = arr * rescale_factor
                    total_rescale_factor *= rescale_factor
                    variance = np.var(arr)
                    iterations += 1

                return arr, total_rescale_factor

            X, x_rescale_factor = rescale_array(X)

            if exog is not None:
                exog_rescale_factors = []
                for i in range(exog.shape[1]):
                    exog[:, i], factor = rescale_array(exog[:, i])
                    exog_rescale_factors.append(factor)
            else:
                exog_rescale_factors = None

            return X, exog, (x_rescale_factor, exog_rescale_factors)

        fit_func = TimeSeriesModel(X=X, exog=exog, model_type=self.model_type)
        self.model = fit_func.fit(order=self.order, **self.model_params)
        if self.model_type == "arch":
            (
                X,
                exog,
                (x_rescale_factor, exog_rescale_factors),
            ) = _rescale_inputs(X, exog)
            self.rescale_factors["x"] = x_rescale_factor
            self.rescale_factors["exog"] = exog_rescale_factors

        return self

    def get_coefs(self) -> np.ndarray:
        """
        Return the coefficients of the fitted model.

        Returns
        -------
        np.ndarray
            The coefficients of the fitted model.

        Raises
        ------
        NotFittedError
            If the model is not fitted.

        Notes
        -----
        The shape of the coefficients depends on the model type.

        +--------+-------------------+
        | Model  | Coefficient shape |
        +========+===================+
        | 'ar'   | (1, order)        |
        +--------+-------------------+
        | 'arima'| (1, order)        |
        +--------+-------------------+
        | 'sarima'| (1, order)       |
        +--------+-------------------+
        | 'var'  | (n_features, n_features, order) |
        +--------+-------------------+
        | 'arch' | (1, order)        |
        +--------+-------------------+
        """
        n_features = (
            self.model.model.endog.shape[1]
            if len(self.model.model.endog.shape) > 1
            else 1
        )
        return self._get_coefs_helper(self.model, n_features)

    def get_intercepts(self) -> np.ndarray:
        """
        Return the intercepts of the fitted model.

        Returns
        -------
        np.ndarray
            The intercepts of the fitted model.

        Raises
        ------
        NotFittedError
            If the model is not fitted.

        Notes
        -----
        The shape of the intercepts depends on the model type.

        +--------+-------------------+
        | Model  | Intercept shape   |
        +========+===================+
        | 'ar'   | (1, trend_terms)  |
        +--------+-------------------+
        | 'arima'| (1, trend_terms)  |
        +--------+-------------------+
        | 'sarima'| (1, trend_terms) |
        +--------+-------------------+
        | 'var'  | (n_features, trend_terms) |
        +--------+-------------------+
        | 'arch' | (0,)              |
        +--------+-------------------+
        """
        n_features = (
            self.model.model.endog.shape[1]
            if len(self.model.model.endog.shape) > 1
            else 1
        )
        return self._get_intercepts_helper(self.model, n_features)

    def get_residuals(self) -> np.ndarray:
        """
        Return the residuals of the fitted model.

        Returns
        -------
        np.ndarray
            The residuals of the fitted model.

        Raises
        ------
        NotFittedError
            If the model is not fitted.

        Notes
        -----
        The shape of the residuals depends on the model type.

        +--------+-------------------+
        | Model  | Residual shape    |
        +========+===================+
        | 'ar'   | (n, 1)            |
        +--------+-------------------+
        | 'arima'| (n, 1)            |
        +--------+-------------------+
        | 'sarima'| (n, 1)           |
        +--------+-------------------+
        | 'var'  | (n, k)            |
        +--------+-------------------+
        | 'arch' | (n, 1)            |
        +--------+-------------------+
        """
        return self._get_residuals_helper(self.model)

    def get_fitted_X(self) -> np.ndarray:
        """
        Return the fitted values of the model.

        Returns
        -------
        np.ndarray
            The fitted values of the model.

        Raises
        ------
        NotFittedError
            If the model is not fitted.

        Notes
        -----
        The shape of the fitted values depends on the model type.

        +--------+-------------------+
        | Model  | Fitted values shape|
        +========+===================+
        | 'ar'   | (n, 1)            |
        +--------+-------------------+
        | 'arima'| (n, 1)            |
        +--------+-------------------+
        | 'sarima'| (n, 1)           |
        +--------+-------------------+
        | 'var'  | (n, k)            |
        +--------+-------------------+
        | 'arch' | (n, 1)            |
        +--------+-------------------+
        """
        return self._get_fitted_X_helper(self.model)

    def get_order(self) -> OrderTypesWithoutNone:
        """
        Return the order of the fitted model.

        Returns
        -------
        OrderTypesWithoutNone
            The order of the fitted model.

        Raises
        ------
        NotFittedError
            If the model is not fitted.

        Notes
        -----
        The shape of the order depends on the model type.

        +--------+-------------------+
        | Model  | Order shape       |
        +========+===================+
        | 'ar'   | int               |
        +--------+-------------------+
        | 'arima'| tuple of length 3 |
        +--------+-------------------+
        | 'sarima'| tuple of length 4|
        +--------+-------------------+
        | 'var'  | int               |
        +--------+-------------------+
        | 'arch' | int               |
        +--------+-------------------+
        """
        return self._get_order_helper(self.model)

    def predict(self, X: np.ndarray, n_steps: int = 1) -> np.ndarray:
        """
        Predict future values using the fitted model.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        n_steps : int, optional
            The number of steps to forecast, by default 1.

        Returns
        -------
        np.ndarray
            The predicted values.

        Raises
        ------
        NotFittedError
            If the model is not fitted.
        ValueError
            If the number of lags is greater than the length of the input data.
        """
        # Check if the model is already fitted
        check_is_fitted(self, ["model"])
        if self.model_type == "var":
            return self.model.forecast(X, n_steps)
        else:
            n_features = X.shape[1] if len(X.shape) > 1 else 1
            coefs = self.get_coefs().T.reshape(n_features, -1)
            X_lagged = self._lag(X, coefs.shape[1])
            return np.dot(X_lagged, coefs)

    def score(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute the R-squared score for the fitted model.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        y_true : np.ndarray
            The true values.

        Returns
        -------
        float
            The R-squared score.

        Raises
        ------
        NotFittedError
            If the model is not fitted.
        ValueError
            If the number of lags is greater than the length of the input data.
        """
        y_pred = self.predict(X)
        # Use r2 as the score
        return r2_score(y_true, y_pred)

    # These helper methods are internal and still take the model as a parameter.
    # They can be used by the public methods above which do not take the model parameter.

    def _get_coefs_helper(self, model, n_features) -> np.ndarray:
        trend_terms = TSFit._calculate_trend_terms(self.model_type, model)
        if self.model_type == "var":
            # Exclude the trend terms and reshape the remaining coefficients
            return (
                model.params[trend_terms:]
                .reshape(n_features, self.get_order(), n_features)
                .transpose(0, 2, 1)
            )
            # shape = (n_features, n_features, order)

        elif self.model_type == "ar":
            # Exclude the trend terms
            if isinstance(self.order, list):
                # Autoreg does not sort the passed lags, but the output from model.params is sorted
                coefs = np.zeros((1, len(self.order)))
                for i, _ in enumerate(self.order):
                    # Exclude the trend terms
                    coefs[0, i] = model.params[trend_terms + i]
            else:
                # Exclude the trend terms
                coefs = model.params[trend_terms:].reshape(1, -1)
            # shape = (1, order)
            return coefs

        elif self.model_type in ["arima", "sarima"]:
            # Exclude the trend terms
            # shape = (1, order)
            return model.params[trend_terms:].reshape(1, -1)

        elif self.model_type == "arch":
            # ARCH models don't include trend terms by default, so just return the params as is
            return model.params

    def _get_intercepts_helper(self, model, n_features) -> np.ndarray:
        trend_terms = TSFit._calculate_trend_terms(self.model_type, model)
        if self.model_type == "var":
            # Include just the trend terms and reshape
            return model.params[:trend_terms].reshape(n_features, trend_terms)
            # shape = (n_features, trend_terms)
        elif self.model_type in ["ar", "arima", "sarima"]:
            # Include just the trend terms
            return model.params[:trend_terms].reshape(1, trend_terms)
            # shape = (1, trend_terms)
        elif self.model_type == "arch":
            # ARCH models don't include trend terms by default, so just return the params as is
            return np.array([])

    @staticmethod
    def _calculate_trend_terms(model_type: str, model) -> int:
        """
        Determine the number of trend terms based on the 'trend' attribute of the model.
        """
        if model_type in ["ar", "arima", "sarima"]:
            trend = model.model.trend
            if trend == "n" or trend is None:
                trend_terms = 0
            elif trend in ["c", "t"]:
                trend_terms = 1
            elif trend == "ct":
                trend_terms = 2
            else:
                raise ValueError(f"Unknown trend term: {trend}")
            return trend_terms

        elif model_type == "var":
            trend = model.trend
            if trend == "nc":
                trend_terms_per_variable = 0
            elif trend == "c":
                trend_terms_per_variable = 1
            elif trend == "ct":
                trend_terms_per_variable = 2
            elif trend == "ctt":
                trend_terms_per_variable = 3
            else:
                raise ValueError(f"Unknown trend term: {trend}")
            return trend_terms_per_variable

    def _get_residuals_helper(self, model) -> np.ndarray:
        model_resid = model.resid

        # Ensure model_resid has the correct shape, (n, 1) or (n, k)
        if model_resid.ndim == 1:
            model_resid = model_resid.reshape(-1, 1)

        if self.model_type in ["ar", "var"]:
            max_lag = np.max(self.get_order())
            values_to_add_back = self.model.model.endog[:max_lag]

            # Ensure values_to_add_back has the same shape as model_resid
            if values_to_add_back.ndim != model_resid.ndim:
                values_to_add_back = values_to_add_back.reshape(-1, 1)

            model_resid = np.vstack((values_to_add_back, model_resid))

        if self.model_type == "arch":
            model_resid = model_resid / self.rescale_factors["x"]

        return model_resid

    def _get_fitted_X_helper(self, model) -> np.ndarray:
        model_fittedvalues = model.fittedvalues

        # Ensure model_fittedvalues has the correct shape, (n, 1) or (n, k)
        if model_fittedvalues.ndim == 1:
            model_fittedvalues = model_fittedvalues.reshape(-1, 1)

        if self.model_type in ["ar", "var"]:
            max_lag = np.max(self.get_order())
            values_to_add_back = self.model.model.endog[:max_lag]

            # Ensure values_to_add_back has the same shape as model_fittedvalues
            if values_to_add_back.ndim != model_fittedvalues.ndim:
                values_to_add_back = values_to_add_back.reshape(-1, 1)

            model_fittedvalues = np.vstack(
                (values_to_add_back, model_fittedvalues)
            )

        if self.model_type == "arch":
            return (
                model.resid + model.conditional_volatility
            ) / self.rescale_factors["x"]
        else:
            return model_fittedvalues

    def _get_order_helper(self, model) -> OrderTypesWithoutNone:
        """
        Return the order of the fitted model.
        """
        if self.model_type == "arch":
            return model.p
        elif self.model == "var":
            return model.k_ar
        elif self.model_type == "ar" and isinstance(self.order, list):
            return sorted(self.order)
        else:
            return self.order

    def _lag(self, X: np.ndarray, n_lags: int) -> np.ndarray:
        """
        Lag the input data.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        n_lags : int
            The number of lags.

        Returns
        -------
        np.ndarray
            The lagged data.

        Raises
        ------
        ValueError
            If the number of lags is greater than the length of the input data.
        """
        if len(X) < n_lags:
            raise ValueError(
                "Number of lags is greater than the length of the input data."
            )
        return np.column_stack(
            [X[i : -(n_lags - i), :] for i in range(n_lags)]
        )


class RankLags:
    """
    A class that uses several metrics to rank lags for time series models.

    Attributes
    ----------
    X : np.ndarray
        The input data.
    model_type : str
        Type of the model.
    max_lag : int, optional, default=10
        Maximum lag to consider.
    exog : np.ndarray, optional, default=None
        Exogenous variables to include in the model.
    save_models : bool, optional, default=False
        Whether to save the models.

    Methods
    -------
    rank_lags_by_aic_bic()
        Rank lags based on Akaike information criterion (AIC) and Bayesian information criterion (BIC).
    rank_lags_by_pacf()
        Rank lags based on Partial Autocorrelation Function (PACF) values.
    estimate_conservative_lag()
        Estimate a conservative lag value by considering various metrics.
    get_model(order)
        Retrieve a previously fitted model given an order.

    Examples
    --------
    >>> from src.tsfit import RankLags
    >>> import numpy as np
    >>> X = np.random.normal(size=(100, 1))
    >>> rank_obj = RankLags(X, model_type='ar')
    >>> rank_obj.estimate_conservative_lag()
    2
    >>> rank_obj.rank_lags_by_aic_bic()
    (array([2, 1]), array([2, 1]))
    >>> rank_obj.rank_lags_by_pacf()
    array([1, 2])
    """

    def __init__(
        self,
        X: np.ndarray,
        model_type: str,
        max_lag: int = 10,
        exog: np.ndarray | None = None,
        save_models: bool = False,
    ) -> None:
        self.X = X
        self.max_lag = max_lag
        self.model_type = model_type.lower()
        self.exog = exog
        self.save_models = save_models
        self.models = []

    def rank_lags_by_aic_bic(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Rank lags based on Akaike information criterion (AIC) and Bayesian information criterion (BIC).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            aic_ranked_lags: Lags ranked by AIC.
            bic_ranked_lags: Lags ranked by BIC.
        """
        aic_values = []
        bic_values = []
        for lag in range(1, self.max_lag + 1):
            try:
                fit_obj = TSFit(order=lag, model_type=self.model_type)
                model = fit_obj.fit(X=self.X, exog=self.exog).model
            except Exception as e:
                # raise RuntimeError(f"An error occurred during fitting: {e}")
                print(f"{e}")
                break
            if self.save_models:
                self.models.append(model)
            aic_values.append(model.aic)
            bic_values.append(model.bic)

        aic_ranked_lags = np.argsort(aic_values) + 1
        bic_ranked_lags = np.argsort(bic_values) + 1

        return aic_ranked_lags, bic_ranked_lags

    def rank_lags_by_pacf(self) -> np.ndarray:
        """
        Rank lags based on Partial Autocorrelation Function (PACF) values.

        Returns
        -------
        np.ndarray
            Lags ranked by PACF values.
        """
        # Can only compute partial correlations for lags up to 50% of the sample size. We use the minimum of max_lag and third of the sample size, to allow for other parameters and trends to be included in the model.
        pacf_values = pacf(
            self.X, nlags=max(min(self.max_lag, self.X.shape[0] // 3 - 1), 1)
        )[1:]
        ci = 1.96 / np.sqrt(len(self.X))
        significant_lags = np.where(np.abs(pacf_values) > ci)[0] + 1
        return significant_lags

    def estimate_conservative_lag(self) -> int:
        """
        Estimate a conservative lag value by considering various metrics.

        Returns
        -------
        int
            A conservative lag value.
        """
        aic_ranked_lags, bic_ranked_lags = self.rank_lags_by_aic_bic()
        # PACF is only available for univariate data
        if self.X.shape[1] == 1:
            pacf_ranked_lags = self.rank_lags_by_pacf()
            highest_ranked_lags = set(aic_ranked_lags).intersection(
                bic_ranked_lags, pacf_ranked_lags
            )
        else:
            highest_ranked_lags = set(aic_ranked_lags).intersection(
                bic_ranked_lags
            )

        if not highest_ranked_lags:
            return aic_ranked_lags[-1]
        else:
            return min(highest_ranked_lags)

    def get_model(
        self, order: int
    ) -> (
        AutoRegResultsWrapper
        | ARIMAResultsWrapper
        | SARIMAXResultsWrapper
        | VARResultsWrapper
        | ARCHModelResult
        | None
    ):
        """
        Retrieve a previously fitted model given an order.

        Parameters
        ----------
        order : int
            Order of the model to retrieve.

        Returns
        -------
        Union[AutoRegResultsWrapper, ARIMAResultsWrapper, SARIMAXResultsWrapper, VARResultsWrapper, ARCHModelResult]
            The fitted model.
        """
        return self.models[order - 1] if self.save_models else None


class TSFitBestLag(BaseEstimator, RegressorMixin):
    """
    A class used to fit time series data and find the best lag for forecasting.

    Attributes
    ----------
    model_type : str
        The type of time series model to use.
    max_lag : int, optional, default=10
        The maximum lag to consider during model fitting.
    order : int, List[int], Tuple[int, int, int], Tuple[int, int, int, int], optional, default=None
        The order of the time series model.
    save_models : bool, optional, default=False
        Whether to save the fitted models for each lag.

    Methods
    -------
    fit(X, exog=None)
        Fit the time series model to the data.
    get_coefs()
        Return the coefficients of the fitted model.
    get_intercepts()
        Return the intercepts of the fitted model.
    get_residuals()
        Return the residuals of the fitted model.
    get_fitted_X()
        Return the fitted values of the model.
    get_order()
        Return the order of the fitted model.
    get_model()
        Return the fitted time series model.
    predict(X, n_steps=1)
        Predict future values using the fitted model.
    score(X, y_true)
        Compute the R-squared score for the fitted model.
    """

    def __init__(
        self,
        model_type: str,
        max_lag: int = 10,
        order: OrderTypes = None,
        save_models=False,
        **kwargs,
    ):
        self.model_type = model_type
        self.max_lag = max_lag
        self.order = order
        self.save_models = save_models
        self.rank_lagger = None
        self.ts_fit = None
        self.model = None
        self.model_params = kwargs

    def _compute_best_order(self, X) -> int:
        """
        Internal method to compute the best order for the given data.

        Parameters
        ----------
        X : np.ndarray
            The input data.

        Returns
        -------
        int
            The best order for the given data.
        """
        self.rank_lagger = RankLags(
            X=X,
            max_lag=self.max_lag,
            model_type=self.model_type,
            save_models=self.save_models,
        )
        best_order = self.rank_lagger.estimate_conservative_lag()
        return best_order

    def fit(
        self, X: np.ndarray, exog: np.ndarray | None = None
    ) -> (
        AutoRegResultsWrapper
        | ARIMAResultsWrapper
        | SARIMAXResultsWrapper
        | VARResultsWrapper
        | ARCHModelResult
    ):
        """
        Fit the time series model to the data.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        exog : np.ndarray, optional, default=None
            Exogenous variables to include in the model.

        Returns
        -------
        self
            The fitted model.
        """
        if self.order is None:
            self.order = self._compute_best_order(X)
            if self.save_models:
                self.model = self.rank_lagger.get_model(self.order)
        self.ts_fit = TSFit(
            order=self.order, model_type=self.model_type, **self.model_params
        )
        self.model = self.ts_fit.fit(X, exog=exog).model
        return self

    def get_coefs(self) -> np.ndarray:
        """
        Return the coefficients of the fitted model.

        Returns
        -------
        np.ndarray
            The coefficients of the fitted model.
        """
        return self.ts_fit.get_coefs()

    def get_residuals(self) -> np.ndarray:
        """
        Return the residuals of the fitted model.

        Returns
        -------
        np.ndarray
            The residuals of the fitted model.
        """
        return self.ts_fit.get_residuals()

    def get_fitted_X(self) -> np.ndarray:
        """
        Return the fitted values of the model.

        Returns
        -------
        np.ndarray
            The fitted values of the model.
        """
        return self.ts_fit.get_fitted_X()

    def get_order(self) -> OrderTypesWithoutNone:
        """
        Return the order of the fitted model.

        Returns
        -------
        int, List[int], Tuple[int, int, int], Tuple[int, int, int, int]
            The order of the fitted model.
        """
        return self.ts_fit.get_order()

    def get_model(self) -> FittedModelType:
        """
        Return the fitted time series model.

        Returns
        -------
        Union[AutoRegResultsWrapper, ARIMAResultsWrapper, SARIMAXResultsWrapper, VARResultsWrapper, ARCHModelResult]
            The fitted time series model.

        Raises
        ------
        ValueError
            If models were not saved during initialization.
        """
        if self.save_models:
            return self.rank_lagger.get_model(self.order)
        else:
            raise ValueError(
                "Models were not saved. Please set save_models=True during initialization."
            )

    def predict(self, X: np.ndarray, n_steps: int = 1):
        """
        Predict future values using the fitted model.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        n_steps : int, optional, default=1
            The number of steps to predict.

        Returns
        -------
        np.ndarray
            The predicted values.
        """
        return self.ts_fit.predict(X, n_steps)

    def score(self, X: np.ndarray, y_true: np.ndarray):
        """
        Compute the R-squared score for the fitted model.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        y_true : np.ndarray
            The true values of the target variable.

        Returns
        -------
        float
            The R-squared score.
        """
        return self.ts_fit.score(X, y_true)
