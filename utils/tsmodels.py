from typing import Union, List, Optional, Tuple, Literal
from arch.univariate.base import ARCHModelResult

import numpy as np
from numpy import ndarray
from statsmodels.tsa.ar_model import AutoReg, AutoRegResultsWrapper
from statsmodels.tsa.arima.model import ARIMA, ARIMAResultsWrapper
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper
from statsmodels.tsa.vector_ar.var_model import VAR, VARResultsWrapper
from arch import arch_model

from utils.validate import validate_X_and_exog, validate_integers


def fit_ar(X: ndarray, order: Union[int, List[int]] = 1, exog: Optional[np.ndarray] = None, **kwargs) -> AutoRegResultsWrapper:
    """Fits an AR model to the input data.

    Args:
        X (ndarray): The input data.
        order (Union[int, List[int]]): The order of the AR model or a list of order to include.
        exog (ndarray): Optional exogenous variables.
        **kwargs: Additional keyword arguments for the AutoReg model, including:
            - seasonal (bool): Whether to include seasonal terms in the model.
            - period (int): The seasonal period, if using seasonal terms.
            - trend (str): The trend component to include in the model.

    Returns:
        AutoRegResultsWrapper: The fitted AR model.
    """
    X, exog = validate_X_and_exog(X, exog)
    N = len(X)

    # Check if period is specified when using seasonal terms, and that it is >= 2
    if kwargs.get('seasonal', False):
        if kwargs.get('period') is None:
            raise ValueError(
                "A period must be specified when using seasonal terms.")
        if kwargs.get('period') < 2:
            raise ValueError("The seasonal period must be >= 2.")

    # Calculate the number of exogenous variables, seasonal terms, and trend parameters
    k = exog.shape[1] if exog is not None else 0
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

    model = AutoReg(endog=X, lags=order, exog=exog, **kwargs)
    model_fit = model.fit()
    return model_fit


def fit_arima(X: ndarray, order: Tuple[int, int, int] = (1, 0, 0), exog: Optional[np.ndarray] = None, **kwargs) -> ARIMAResultsWrapper:
    """Fits an ARIMA model to the input data.

    Args:
        X (ndarray): The input data.
        order (Tuple[int, int, int]): The order of the ARIMA model (p, d, q).
        exog (ndarray, optional): Optional array of exogenous variables.

    Returns:
        ARIMAResultsWrapper: The fitted ARIMA model.
    """
    X, exog = validate_X_and_exog(X, exog)

    if len(order) != 3:
        raise ValueError("The order must be a 3-tuple")

    model = ARIMA(endog=X, order=order, exog=exog, **kwargs)
    model_fit = model.fit()
    return model_fit


def fit_sarima(X: ndarray, order: Tuple[int, int, int, int] = (0, 0, 0, 2),
               arima_order: Optional[Tuple[int, int, int]] = (1, 0, 0),
               exog: Optional[np.ndarray] = None, **kwargs) -> SARIMAXResultsWrapper:
    """Fits a SARIMA model to the input data.

    Args:
        X (ndarray): The input data.
        order (Tuple[int, int, int, int]): The order of the SARIMA model.
        arima_order (Tuple[int, int, int], optional): The order of the ARIMA model.
        exog (ndarray, optional): Optional array of exogenous variables.

    Returns:
        SARIMAXResultsWrapper: The fitted SARIMA model.
    """
    X, exog = validate_X_and_exog(
        X, exog)  # assuming this function exists in your code

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

    model = SARIMAX(endog=X, order=arima_order,
                    seasonal_order=order, exog=exog, **kwargs)
    model_fit = model.fit()
    return model_fit


def fit_var(X: np.ndarray, order: Optional[int] = None, exog: Optional[np.ndarray] = None, **kwargs) -> VARResultsWrapper:
    """Fits a Vector Autoregression (VAR) model to the input data.

    Args:
        X (ndarray): The input data.
        order (int, optional): This argument is not used in the current implementation, 
                              it is only included for consistency with other similar methods.
        exog (ndarray, optional): Optional array of exogenous variables.

    Returns:
        VARResultsWrapper: The fitted VAR model.
    """
    X, exog = validate_X_and_exog(X, exog, model_is_var=True)
    model = VAR(endog=X, exog=exog)
    model_fit = model.fit(**kwargs)
    return model_fit


def fit_arch(X: np.ndarray, p: int = 1, q: int = 1, model_type: Literal["GARCH", "EGARCH", "TARCH", "AGARCH"] = 'GARCH', mean_type: Literal["zero", "AR"] = "zero", order: int = 1, exog: Optional[np.ndarray] = None, **kwargs) -> ARCHModelResult:
    """
    Fits a GARCH, GARCH-M, EGARCH, TARCH, or AGARCH model to the input data.

    Args:
        X (ndarray): The input data.
        p (int): The number of order in the GARCH part of the model.
        q (int): The number of order in the ARCH part of the model.
        model_type (str): The type of GARCH model to fit. Options are 'GARCH', 'EGARCH', 'TARCH', and 'AGARCH'.
        order (int): The number of order to include in the AR part of the model.
        exog (ndarray, optional): Optional array of exogenous variables.

    Returns:
        The fitted GARCH model.
    """

    # Assuming a validate_X_and_exog function exists for data validation
    X, exog = validate_X_and_exog(X, exog, model_is_arch=True)
    validate_integers(p, q)
    validate_integers(order, positive=True)

    if mean_type not in ['zero', 'AR']:
        raise ValueError("mean_type must be one of 'zero' or 'AR'")

    if model_type == 'GARCH':
        model = arch_model(y=X, x=exog, mean=mean_type, lags=order,
                           vol=model_type, p=p, q=q, **kwargs)
    elif model_type == 'EGARCH':
        model = arch_model(y=X, x=exog, mean=mean_type, lags=order,
                           vol=model_type, p=p, q=q, **kwargs)
    elif model_type == 'TARCH':
        model = arch_model(y=X, x=exog, mean=mean_type, lags=order,
                           vol="GARCH", p=p, o=1, q=q, power=1, **kwargs)
    elif model_type == 'AGARCH':
        model = arch_model(y=X, x=exog, mean=mean_type, lags=order,
                           vol="GARCH", p=p, o=1, q=q, **kwargs)
    else:
        raise ValueError(
            "model_type must be one of 'GARCH', 'EGARCH', 'TARCH', or 'AGARCH'")

    options = {"maxiter": 200}
    model_fit = model.fit(disp='off', options=options)

    return model_fit
