from __future__ import annotations
from statsmodels.tsa.stattools import pacf
from typing import Optional, Tuple, Union, List

import warnings
from statsmodels.tsa.ar_model import AutoRegResultsWrapper
from arch.univariate.base import ARCHModelResult
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from statsmodels.tsa.arima.model import ARIMAResultsWrapper
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, RegressorMixin

from functools import lru_cache
import numpy as np

from utils.tsmodels import fit_ar, fit_arima, fit_sarima, fit_var, fit_arch


class TSFit(BaseEstimator, RegressorMixin):
    """
    This class performs fitting for various time series models including 'ar', 'arima', 'sarima', 'var', and 'arch'.
    It inherits the BaseEstimator and RegressorMixin classes from scikit-learn library.
    """

    def __init__(self, order: Union[int, List[int], Tuple[int, int, int], Tuple[int, int, int, int]], model_type: str, **kwargs) -> None:
        """
        Constructor for the TSFit class.

        Args:
            order: Specifies the order of the model. For 'ar', 'var' and 'arch' models, it's an integer. 
                For 'arima' and 'sarima', it's a tuple of integers (p, d, q, s).
            model_type: Type of the model. It can be one of 'ar', 'arima', 'sarima', 'var', 'arch'.
            **kwargs: Additional keyword arguments which will be passed to the model.
        """
        if model_type not in ['ar', 'arima', 'sarima', 'var', 'arch']:
            raise ValueError(
                f"Invalid model type '{model_type}', should be one of ['ar', 'arima', 'sarima', 'var', 'arch']")

        if type(order) == tuple and model_type not in ['arima', 'sarima']:
            raise ValueError(
                f"Invalid order '{order}', should be an integer for model type '{model_type}'")

        if type(order) == int and model_type in ['arima', 'sarima']:
            order = (order, 0, 0, 0)
            warnings.warn(
                f"{model_type.upper()} model requires a tuple of order (p, d, q, s), where d is the order of differencing and s is the seasonal period. Setting d=0, q=0 and s=0.")

        self.order = order
        self.model_type = model_type.lower()
        self.rescale_factors = {}
        self.model = None
        self.model_params = kwargs

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Args:
            deep: When set to True, will return the parameters for this estimator and contained subobjects that are estimators.

        Returns:
            params: Dictionary of parameter names mapped to their values.
        """
        return {"order": self.order, "model_type": self.model_type, **self.model_params}

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

    def fit_func(self, model_type):
        """
        Returns the appropriate fitting function based on the model type.

        Args:
            model_type: Type of the model.

        Returns:
            The function used for fitting the model.
        """
        if model_type == 'arima':
            return fit_arima
        elif model_type == 'ar':
            return fit_ar
        elif model_type == 'var':
            return fit_var
        elif model_type == 'sarima':
            return fit_sarima
        elif model_type == 'arch':
            return fit_arch
        else:
            raise ValueError(f"Invalid model type {model_type}")

    # @lru_cache(maxsize=None)
    def fit(self, X: np.ndarray, exog: Optional[np.ndarray] = None) -> Union[AutoRegResultsWrapper, ARIMAResultsWrapper, SARIMAXResultsWrapper, VARResultsWrapper, ARCHModelResult]:
        """
        Fit the chosen model to the data.

        Args:
            X: The input data.
            exog: Exogenous variables, optional.

        Raises:
            ValueError: If the model type or the model order is invalid.
        """
        # Check if the input shapes are valid
        if len(X.shape) != 2 or X.shape[1] < 1:
            raise ValueError(
                "X should be 2-D with the second dimension greater than or equal to 1.")
        if exog is not None:
            # checking whether X and exog have compatible shapes
            check_X_y(X, exog)
            if len(exog.shape) != 2 or exog.shape[1] < 1:
                raise ValueError(
                    "exog should be 2-D with the second dimension greater than or equal to 1.")

        def _rescale_inputs(X: np.ndarray, exog: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[float, Optional[List[float]]]]:
            def rescale_array(arr: np.ndarray) -> Tuple[np.ndarray, float]:
                variance = np.var(arr)
                rescale_factor = 1
                if variance < 1 or variance > 1000:
                    rescale_factor = np.sqrt(100 / variance)
                    arr_rescaled = arr * rescale_factor
                return arr_rescaled, rescale_factor

            X, x_rescale_factor = rescale_array(X)

            if exog is not None:
                exog_rescale_factors = []
                for i in range(exog.shape[1]):
                    exog[:, i], factor = rescale_array(exog[:, i])
                    exog_rescale_factors.append(factor)
            else:
                exog_rescale_factors = None

            return X, exog, (x_rescale_factor, exog_rescale_factors)

        fit_func = self.fit_func(self.model_type)

        if self.model_type == 'arch':
            X, exog, (x_rescale_factor,
                      exog_rescale_factors) = _rescale_inputs(X, exog)
            self.model = fit_func(
                X, order=self.order, exog=exog, **self.model_params)
            self.rescale_factors['x'] = x_rescale_factor
            self.rescale_factors['exog'] = exog_rescale_factors
        else:
            self.model = fit_func(
                X, order=self.order, exog=exog, **self.model_params)

        return self

    def get_coefs(self) -> np.ndarray:
        n_features = self.model.model.endog.shape[1] if len(
            self.model.model.endog.shape) > 1 else 1
        return self._get_coefs_helper(self.model, n_features)

    def get_residuals(self) -> np.ndarray:
        return self._get_residuals_helper(self.model)

    def get_fitted_X(self) -> np.ndarray:
        return self._get_fitted_X_helper(self.model)

    def get_order(self) -> Union[int, List[int], Tuple[int, int, int], Tuple[int, int, int, int]]:
        return self._get_order_helper(self.model)

    def predict(self, X: np.ndarray, n_steps: int = 1):
        # Check if the model is already fitted
        check_is_fitted(self, ['model'])
        if self.model_type == 'var':
            return self.model.forecast(X, n_steps)
        else:
            n_features = X.shape[1] if len(X.shape) > 1 else 1
            coefs = self.get_coefs().T.reshape(n_features, -1)
            X_lagged = self._lag(X, coefs.shape[1])
            return np.dot(X_lagged, coefs)

    def score(self, X: np.ndarray, y_true: np.ndarray):
        y_pred = self.predict(X)
        # Use r2 as the score
        return r2_score(y_true, y_pred)

    # These helper methods are internal and still take the model as a parameter.
    # They can be used by the public methods above which do not take the model parameter.

    def _get_coefs_helper(self, model, n_features) -> np.ndarray:
        if self.model_type == 'var':
            return model.params[1:].reshape(self.get_order(), n_features, n_features).transpose(1, 0, 2)
        elif self.model_type == 'ar':
            if isinstance(self.order, list):
                coefs = np.zeros((n_features, len(self.order)))
                for i, lag in enumerate(self.order):
                    coefs[:, i] = model.params[1 + i::len(self.order)]
            else:
                coefs = model.params[1:].reshape(n_features, self.order)
            return coefs
        elif self.model_type in ['arima', 'sarima', 'arch']:
            return model.params

    def _get_residuals_helper(self, model) -> np.ndarray:
        model_resid = model.resid

        # Ensure model_resid has the correct shape, (n, 1) or (n, k)
        if model_resid.ndim == 1:
            model_resid = model_resid.reshape(-1, 1)

        if self.model_type in ['ar', 'var']:
            max_lag = np.max(self.get_order())
            values_to_add_back = self.model.model.endog[:max_lag]

            # Ensure values_to_add_back has the same shape as model_resid
            if values_to_add_back.ndim != model_resid.ndim:
                values_to_add_back = values_to_add_back.reshape(-1, 1)

            model_resid = np.vstack((values_to_add_back, model_resid))

        if self.model_type == 'arch':
            model_resid = model_resid / self.rescale_factors['x']

        return model_resid

    def _get_fitted_X_helper(self, model) -> np.ndarray:
        model_fittedvalues = model.fittedvalues

        # Ensure model_fittedvalues has the correct shape, (n, 1) or (n, k)
        if model_fittedvalues.ndim == 1:
            model_fittedvalues = model_fittedvalues.reshape(-1, 1)

        if self.model_type in ['ar', 'var']:
            max_lag = np.max(self.get_order())
            values_to_add_back = self.model.model.endog[:max_lag]

            # Ensure values_to_add_back has the same shape as model_fittedvalues
            if values_to_add_back.ndim != model_fittedvalues.ndim:
                values_to_add_back = values_to_add_back.reshape(-1, 1)

            model_fittedvalues = np.vstack(
                (values_to_add_back, model_fittedvalues))

        if self.model_type == 'arch':
            return (model.resid + model.conditional_volatility) / self.rescale_factors['x']
        else:
            return model_fittedvalues

    def _get_order_helper(self, model) -> Union[int, List[int], Tuple[int, int, int], Tuple[int, int, int, int]]:
        return model.k_ar if self.model == 'var' else self.order

    def _lag(self, X: np.ndarray, n_lags: int):
        if len(X) < n_lags:
            raise ValueError(
                "Number of lags is greater than the length of the input data.")
        return np.column_stack([X[i:-(n_lags - i), :] for i in range(n_lags)])


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
    """

    def __init__(self, X: np.ndarray, model_type: str, max_lag: int = 10, exog: Optional[np.ndarray] = None, save_models: bool = False) -> None:
        self.X = X
        self.max_lag = max_lag
        self.model_type = model_type.lower()
        self.exog = exog
        self.save_models = save_models
        self.models = []

    def rank_lags_by_aic_bic(self) -> Tuple[np.ndarray, np.ndarray]:
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
            fit_obj = TSFit(order=lag, model_type=self.model_type)
            model = fit_obj.fit(X=self.X, exog=self.exog)
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
        pacf_values = pacf(self.X, nlags=self.max_lag)[1:]
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
                bic_ranked_lags, pacf_ranked_lags)
        else:
            highest_ranked_lags = set(aic_ranked_lags).intersection(
                bic_ranked_lags)

        if not highest_ranked_lags:
            return aic_ranked_lags[-1]
        else:
            return min(highest_ranked_lags)

    def get_model(self, order: int) -> Union[AutoRegResultsWrapper, ARIMAResultsWrapper, SARIMAXResultsWrapper, VARResultsWrapper, ARCHModelResult]:
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

    def __init__(self, model_type: str, max_lag: int = 10, order: Optional[Union[int, List[int], Tuple[int, int, int], Tuple[int, int, int, int]]] = None, save_models=False):
        self.model_type = model_type
        self.max_lag = max_lag
        self.order = order
        self.save_models = save_models
        self.rank_lagger = None
        self.ts_fit = None
        self.model = None

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
            X=X, max_lag=self.max_lag, model_type=self.model_type, save_models=self.save_models)
        best_order = self.rank_lagger.estimate_conservative_lag()
        return best_order

    def fit(self, X: np.ndarray, exog: Optional[np.ndarray] = None) -> Union[AutoRegResultsWrapper, ARIMAResultsWrapper, SARIMAXResultsWrapper, VARResultsWrapper, ARCHModelResult]:
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
        self.ts_fit = TSFit(order=self.order, model_type=self.model_type)
        self.model = self.ts_fit.fit(X, exog=exog)
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

    def get_order(self) -> Union[int, List[int], Tuple[int, int, int], Tuple[int, int, int, int]]:
        """
        Return the order of the fitted model.

        Returns
        -------
        int, List[int], Tuple[int, int, int], Tuple[int, int, int, int]
            The order of the fitted model.
        """
        return self.ts_fit.get_order()

    def get_model(self) -> Union[AutoRegResultsWrapper, ARIMAResultsWrapper, SARIMAXResultsWrapper, VARResultsWrapper, ARCHModelResult]:
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
                'Models were not saved. Please set save_models=True during initialization.')

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
