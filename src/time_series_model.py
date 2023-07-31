from typing import Union, List, Optional, Tuple, Literal
from arch.univariate.base import ARCHModelResult

import numpy as np

from statsmodels.tsa.ar_model import AutoReg, AutoRegResultsWrapper
from statsmodels.tsa.arima.model import ARIMA, ARIMAResultsWrapper
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper
from statsmodels.tsa.vector_ar.var_model import VAR, VARResultsWrapper
from arch import arch_model

from utils.validate import validate_X_and_exog, validate_integers, validate_X, validate_literal_type
from utils.types import ModelTypes


class TimeSeriesModel:
    """A class for fitting time series models to data."""

    def __init__(self, X: np.ndarray, exog: Optional[np.ndarray] = None, model_type: ModelTypes = "ar"):
        """Initializes a TimeSeriesModel object.

        Args:
            X (ndarray): The input data.
            exog (ndarray, optional): Optional array of exogenous variables.
        """
        self.model_type = model_type
        self.X = X
        self.exog = exog

    @property
    def model_type(self) -> ModelTypes:
        return self._model_type

    @model_type.setter
    def model_type(self, value: ModelTypes) -> None:
        value = value.lower()
        validate_literal_type(value, ModelTypes)
        self._model_type = value

    @property
    def X(self) -> np.ndarray:
        return self._X

    @X.setter
    def X(self, value: np.ndarray) -> None:
        validate_X(value)
        self._X = value

    @property
    def exog(self) -> Optional[np.ndarray]:
        return self._exog

    @exog.setter
    def exog(self, value: Optional[np.ndarray]) -> None:
        validate_X_and_exog(self.X, value, model_is_var=self.model_type ==
                            "var", model_is_arch=self.model_type == "arch")
        self._exog = value

    def fit_ar(self, order: Union[int, List[int]] = 1, **kwargs) -> AutoRegResultsWrapper:
        """Fits an AR model to the input data.

        Args:
            order (Union[int, List[int]]): The order of the AR model or a list of order to include.
            **kwargs: Additional keyword arguments for the AutoReg model, including:
                - seasonal (bool): Whether to include seasonal terms in the model.
                - period (int): The seasonal period, if using seasonal terms.
                - trend (str): The trend component to include in the model.

        Returns:
            AutoRegResultsWrapper: The fitted AR model.
        """

        N = len(self.X)

        # Check if period is specified when using seasonal terms, and that it is >= 2
        if kwargs.get('seasonal', False):
            if kwargs.get('period') is None:
                raise ValueError(
                    "A period must be specified when using seasonal terms.")
            if kwargs.get('period') < 2:
                raise ValueError("The seasonal period must be >= 2.")

        # Calculate the number of exogenous variables, seasonal terms, and trend parameters
        k = self.exog.shape[1] if self.exog is not None else 0
        seasonal_terms = kwargs.get('period', 0) - 1 if kwargs.get(
            'seasonal', False) and kwargs.get('period') is not None else 0
        trend_parameters = 1 if kwargs.get(
            'trend', "c") == "c" else 2 if kwargs.get('trend') == "ct" else 0

        # Calculate the maximum allowed lag value
        max_lag = (N - k - seasonal_terms - trend_parameters) // 2

        # Check if the specified order value is within the allowed range
        if isinstance(order, list):
            if max(order) > max_lag:
                raise ValueError(
                    f"Maximum allowed lag value exceeded. The allowed maximum is {max_lag}.")
        else:
            if order > max_lag:
                raise ValueError(
                    f"Maximum allowed lag value exceeded. The allowed maximum is {max_lag}.")

        model = AutoReg(endog=self.X, lags=order, exog=self.exog, **kwargs)
        model_fit = model.fit()
        return model_fit

    def fit_arima(self, order: Tuple[int, int, int] = (1, 0, 0), **kwargs) -> ARIMAResultsWrapper:
        """Fits an ARIMA model to the input data.

        Args:
            order (Tuple[int, int, int]): The order of the ARIMA model (p, d, q).

        Returns:
            ARIMAResultsWrapper: The fitted ARIMA model.
        """
        if len(order) != 3:
            raise ValueError("The order must be a 3-tuple")

        model = ARIMA(endog=self.X, order=order, exog=self.exog, **kwargs)
        model_fit = model.fit()
        return model_fit

    def fit_sarima(self, order: Tuple[int, int, int, int] = (0, 0, 0, 2),
                   arima_order: Optional[Tuple[int, int, int]] = (1, 0, 0), **kwargs) -> SARIMAXResultsWrapper:
        """Fits a SARIMA model to the input data.

        Args:
            X (ndarray): The input data.
            order (Tuple[int, int, int, int]): The order of the SARIMA model.
            arima_order (Tuple[int, int, int], optional): The order of the ARIMA model.
            exog (ndarray, optional): Optional array of exogenous variables.

        Returns:
            SARIMAXResultsWrapper: The fitted SARIMA model.
        """
        if order[-1] < 2:
            raise ValueError("The seasonal periodicity must be greater than 1")

        if len(order) != 4:
            raise ValueError("The seasonal_order must be a 4-tuple")

        if arima_order is None:
            # If 'q' is the same as 's' in order, set 'q' to 0 to prevent overlap
            if order[2] == order[-1]:
                arima_order = (order[0], order[1], 0)
            else:
                arima_order = order[:3]

        if len(arima_order) != 3:
            raise ValueError("The order must be a 3-tuple")

        # Check to ensure that the AR terms (p and P) don't duplicate order
        if arima_order[0] >= order[-1] and order[0] != 0:
            raise ValueError(
                f"The autoregressive term 'p' ({arima_order[0]}) is greater than or equal to the seasonal period 's' ({order[-1]}) while the seasonal autoregressive term 'P' is not zero ({order[0]}). This could lead to duplication of order.")

        # Check to ensure that the MA terms (q and Q) don't duplicate order
        if arima_order[2] >= order[-1] and order[2] != 0:
            raise ValueError(
                f"The moving average term 'q' ({arima_order[2]}) is greater than or equal to the seasonal period 's' ({order[-1]}) while the seasonal moving average term 'Q' is not zero ({order[2]}). This could lead to duplication of order.")

        model = SARIMAX(endog=self.X, order=arima_order,
                        seasonal_order=order, exog=self.exog, **kwargs)
        model_fit = model.fit(disp=-1)
        return model_fit

    def fit_var(self, order: Optional[int] = None, **kwargs) -> VARResultsWrapper:
        """Fits a Vector Autoregression (VAR) model to the input data.

        Args:
            order (int, optional): This argument is not used in the current implementation, 
                                it is only included for consistency with other similar methods.

        Returns:
            VARResultsWrapper: The fitted VAR model.
        """
        model = VAR(endog=self.X, exog=self.exog)
        model_fit = model.fit(**kwargs)
        return model_fit

    def fit_arch(self, p: int = 1, q: int = 1, arch_model_type: Literal["GARCH", "EGARCH", "TARCH", "AGARCH"] = 'GARCH', mean_type: Literal["zero", "AR"] = "zero", order: int = 1, **kwargs) -> ARCHModelResult:
        """
        Fits a GARCH, GARCH-M, EGARCH, TARCH, or AGARCH model to the input data.

        Args:
            p (int): The number of order in the GARCH part of the model.
            q (int): The number of order in the ARCH part of the model.
            arch_model_type (str): The type of GARCH model to fit. Options are 'GARCH', 'EGARCH', 'TARCH', and 'AGARCH'.
            order (int): The number of order to include in the AR part of the model.
            mean_type (str): The type of mean model to use. Options are 'zero' and 'AR'.
        Returns:
            The fitted GARCH model.
        """

        # Assuming a validate_X_and_exog function exists for data validation
        validate_integers(p, q, order, positive=True)

        if mean_type not in ['zero', 'AR']:
            raise ValueError("mean_type must be one of 'zero' or 'AR'")

        if arch_model_type == 'GARCH':
            model = arch_model(y=self.X, x=self.exog, mean=mean_type, lags=order,
                               vol=arch_model_type, p=p, q=q, **kwargs)
        elif arch_model_type == 'EGARCH':
            model = arch_model(y=self.X, x=self.exog, mean=mean_type, lags=order,
                               vol=arch_model_type, p=p, q=q, **kwargs)
        elif arch_model_type == 'TARCH':
            model = arch_model(y=self.X, x=self.exog, mean=mean_type, lags=order,
                               vol="GARCH", p=p, o=1, q=q, power=1, **kwargs)
        elif arch_model_type == 'AGARCH':
            model = arch_model(y=self.X, x=self.exog, mean=mean_type, lags=order,
                               vol="GARCH", p=p, o=1, q=q, **kwargs)
        else:
            raise ValueError(
                "arch_model_type must be one of 'GARCH', 'EGARCH', 'TARCH', or 'AGARCH'")

        options = {"maxiter": 200}
        model_fit = model.fit(disp='off', options=options)

        return model_fit

    def fit(self, order: Optional[Union[int, List[int], Tuple[int, int, int], Tuple[int, int, int, int]]], **kwargs):
        """Fits a time series model to the input data.

        Args:
            **kwargs: Additional keyword arguments for the model.

        Returns:
            The fitted time series model.
        """

        fitted_models = {
            "ar": self.fit_ar(order, **kwargs),
            "arima": self.fit_arima(order, **kwargs),
            "sarima": self.fit_sarima(order, **kwargs),
            "var": self.fit_var(order, **kwargs),
            "arch": self.fit_arch(order, **kwargs),
        }
        for model_type, fitted_model in fitted_models.items():
            if isinstance(self.model_type, model_type):
                return fitted_model
        raise ValueError("Unsupported fitted model type.")
