from typing import Union, List, Optional, Tuple

import numpy as np
from numpy import ndarray
from sklearn.utils import check_array, check_X_y
from statsmodels.tsa.ar_model import AutoReg, AutoRegResultsWrapper
from statsmodels.tsa.arima.model import ARIMA, ARIMAResultsWrapper
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper
from statsmodels.tsa.vector_ar.var_model import VAR, VARResultsWrapper
from arch import arch_model
from arch.univariate.base import ARCHModelResult
from arch.univariate.mean import ARX
from arch.univariate import GARCH


def validate_X_and_exog(X: ndarray, exog: Optional[np.ndarray], model_is_var: bool = False, model_is_arch: bool = False) -> Tuple[ndarray, Optional[np.ndarray]]:
    """
    Validate and reshape input data and exogenous variables.

    Args:
        X (ndarray): The input data.
        exog (Optional[np.ndarray]): Optional exogenous variables.
        model_is_var (bool): Indicates if the model is a VAR model.
        model_is_arch (bool): Indicates if the model is an ARCH model.

    Returns:
        Tuple[ndarray, Optional[np.ndarray]]: The validated and reshaped X and exog arrays.
    """
    # Validate and reshape X
    if not model_is_var:
        X = check_array(X, ensure_2d=False, force_all_finite=True)
        X = np.squeeze(X)
        if X.ndim != 1:
            raise ValueError("X must be 1-dimensional")
    else:
        X = check_array(X, ensure_2d=True, force_all_finite=True)
        if X.shape[1] < 2:
            raise ValueError("X must be 2-dimensional with at least 2 columns")

    # Validate and reshape exog if necessary
    if exog is not None:
        if exog.ndim == 1:
            exog = exog[:, np.newaxis]
        exog = check_array(exog, ensure_2d=True, force_all_finite=True)
        X, exog = check_X_y(X, exog, force_all_finite=True, multi_output=True)

    # Ensure contiguous arrays for ARCH models
    if model_is_arch:
        X = np.ascontiguousarray(X)
        if exog is not None:
            exog = np.ascontiguousarray(exog)

    return X, exog


def fit_ar(X: ndarray, lags: Union[int, List[int]] = 1, exog: Optional[np.ndarray] = None, **kwargs) -> AutoRegResultsWrapper:
    """Fits an AR model to the input data.

    Args:
        X (ndarray): The input data.
        lags (Union[int, List[int]]): The order of the AR model or a list of lags to include.
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

    # Calculate the number of exogenous variables, seasonal terms, and trend parameters
    k = exog.shape[1] if exog is not None else 0
    seasonal_terms = kwargs.get('period', 0) - 1 if kwargs.get(
        'seasonal', False) and kwargs.get('period') is not None else 0
    trend_parameters = 1 if kwargs.get(
        'trend', "c") == "c" else 2 if kwargs.get('trend') == "ct" else 0

    # Calculate the maximum allowed lag value
    max_lag = (N - k - seasonal_terms - trend_parameters) // 2

    # Check if the specified lags value is within the allowed range
    if isinstance(lags, list):
        if max(lags) > max_lag:
            raise ValueError(
                f"Maximum allowed lag value exceeded. The allowed maximum is {max_lag}.")
    else:
        if lags > max_lag:
            raise ValueError(
                f"Maximum allowed lag value exceeded. The allowed maximum is {max_lag}.")

    model = AutoReg(endog=X, lags=lags, exog=exog, **kwargs)
    model_fit = model.fit()
    return model_fit


def fit_arima(X: ndarray, arima_order: Tuple[int, int, int] = (1, 0, 0), exog: Optional[np.ndarray] = None, **kwargs) -> ARIMAResultsWrapper:
    """Fits an ARIMA model to the input data.

    Args:
        X (ndarray): The input data.
        arima_order (Tuple[int, int, int]): The order of the ARIMA model (p, d, q).
        exog (ndarray, optional): Optional array of exogenous variables.

    Returns:
        ARIMAResultsWrapper: The fitted ARIMA model.
    """
    X, exog = validate_X_and_exog(X, exog)
    model = ARIMA(endog=X, order=arima_order, exog=exog, **kwargs)
    model_fit = model.fit()
    return model_fit


def fit_sarima(X: ndarray, sarima_order: Tuple[int, int, int, int] = (0, 0, 0, 2),
               arima_order: Optional[Tuple[int, int, int]] = (1, 0, 0),
               exog: Optional[np.ndarray] = None, **kwargs) -> SARIMAXResultsWrapper:
    """Fits a SARIMA model to the input data.

    Args:
        X (ndarray): The input data.
        sarima_order (Tuple[int, int, int, int]): The order of the SARIMA model.
        arima_order (Tuple[int, int, int], optional): The order of the ARIMA model.
        exog (ndarray, optional): Optional array of exogenous variables.

    Returns:
        SARIMAXResultsWrapper: The fitted SARIMA model.
    """
    X, exog = validate_X_and_exog(X, exog)

    if sarima_order[-1] < 2:
        raise ValueError("The seasonal periodicity must be greater than 1")

    if len(sarima_order) != 4:
        raise ValueError("The seasonal_order must be a 4-tuple")

    arima_order = arima_order if arima_order is not None else sarima_order[:3]

    model = SARIMAX(endog=X, order=arima_order,
                    seasonal_order=sarima_order, exog=exog, **kwargs)
    model_fit = model.fit()
    return model_fit


def fit_var(X: ndarray, lags: Optional[int] = None, exog: Optional[np.ndarray] = None, **kwargs) -> VARResultsWrapper:
    """Fits a Vector Autoregression (VAR) model to the input data.

    Args:
        X (ndarray): The input data.
        lags (int, optional): The number of lags to include in the VAR model.
        exog (ndarray, optional): Optional array of exogenous variables.

    Returns:
        VARResultsWrapper: The fitted VAR model.
    """
    X, exog = validate_X_and_exog(X, exog, model_is_var=True)
    model = VAR(endog=X, exog=exog, **kwargs)
    model_fit = model.fit()
    return model_fit


def fit_arch(X: np.ndarray, p: int = 1, q: int = 1, model_type: str = 'GARCH',
             lags: Union[int, List[int]] = 0, exog: Optional[np.ndarray] = None, **kwargs) -> ARCHModelResult:
    """Fits a GARCH, GARCH-M, EGARCH, TARCH, or AGARCH model to the input data.

    Args:
        X (ndarray): The input data.
        p (int): The number of lags in the GARCH part of the model.
        q (int): The number of lags in the ARCH part of the model.
        model_type (str): The type of GARCH model to fit. Options are 'GARCH', 'GARCH-M', 'EGARCH', 'TARCH', and 'AGARCH'.
        lags (Union[int, List[int]]): The number of lags or a list of lag indices to include in the AR part of the model.
        exog (ndarray, optional): Optional array of exogenous variables.

    Returns:
        ARCHModelResult: The fitted GARCH model.
    """
    X, exog = validate_X_and_exog(X, exog, model_is_arch=True)

    if model_type == 'GARCH':
        model = ARX(y=X, x=exog, lags=lags)
        model.volatility = GARCH(p=p, q=q)
    elif model_type in ['GARCH-M', 'EGARCH', 'TARCH']:
        model = arch_model(y=X, x=exog, mean='Zero' if model_type != 'GARCH-M' else 'AR', lags=lags,
                           vol=model_type, p=p, q=q, **kwargs)
    else:
        raise ValueError(
            "model_type must be one of 'GARCH', 'GARCH-M', 'EGARCH', 'TARCH', or 'AGARCH'")

    options = {"maxiter": 200}
    model_fit = model.fit(disp='off', options=options)
    return model_fit
