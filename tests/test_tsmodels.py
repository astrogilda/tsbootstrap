import pytest
import numpy as np
from utils.tsmodels import fit_ar, fit_arima, fit_sarima, fit_var, fit_arch, validate_X_and_exog
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from statsmodels.tsa.ar_model import AutoRegResultsWrapper
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper
from statsmodels.tsa.arima.model import ARIMAResultsWrapper
from arch.univariate.base import ARCHModelResult


@pytest.fixture(scope="module")
def input_1d():
    return np.random.rand(100)


@pytest.fixture(scope="module")
def input_2d():
    return np.random.rand(100, 2)


@pytest.fixture(scope="module")
def exog_1d():
    return np.random.rand(100)


@pytest.fixture(scope="module")
def exog_2d():
    return np.random.rand(100, 2)


def test_validate_X_and_exog(input_1d, exog_1d, exog_2d):
    # Test with None exog
    assert validate_X_and_exog(input_1d, None) == (input_1d, None)
    # Test with 1D exog
    assert np.array_equal(validate_X_and_exog(
        input_1d, exog_1d)[1], exog_1d[:, np.newaxis])
    # Test with 2D exog
    assert np.array_equal(validate_X_and_exog(input_1d, exog_2d)[1], exog_2d)
    # Test with invalid input dimensions
    with pytest.raises(ValueError):
        validate_X_and_exog(np.random.rand(100, 2, 2), exog_1d)
    # Test with invalid exog dimensions
    with pytest.raises(ValueError):
        validate_X_and_exog(input_1d, np.random.rand(100, 2, 2))


@pytest.mark.parametrize('lags', [1, 2, 10, 50, 99, [1, 3], [2, 5, 10], [1, 10, 50]])
def test_fit_ar(input_1d, exog_1d, lags):

    # Test with no exog, seasonal lags, and set trend to 'c' (constant, default)
    max_lag = (input_1d.shape[0] - 1) // 2

    if np.max(lags) <= max_lag:
        model_fit = fit_ar(input_1d, lags, exog=None)
        assert isinstance(model_fit, AutoRegResultsWrapper)

        # Test with exog
        model_fit_exog = fit_ar(input_1d, lags, exog=exog_1d)
        assert isinstance(model_fit_exog, AutoRegResultsWrapper)

        if isinstance(lags, list):
            assert model_fit.params.size == len(lags) + 1
            assert model_fit_exog.params.size == len(lags) + 2
        else:
            assert model_fit.params.size == lags + 1
            assert model_fit_exog.params.size == lags + 2

    else:
        with pytest.raises(ValueError, match=f"Maximum allowed lag value exceeded. The allowed maximum is {max_lag}"):
            fit_ar(input_1d, lags, exog=None)
        with pytest.raises(ValueError, match=f"Maximum allowed lag value exceeded. The allowed maximum is {max_lag}"):
            fit_ar(input_1d, lags, exog=exog_1d)

    # Test with seasonal and period kwargs
    model_fit_seasonal = fit_ar(
        input_1d, lags=1, exog=None, seasonal=True, period=2)
    assert isinstance(model_fit_seasonal, AutoRegResultsWrapper)
    assert model_fit_seasonal.params.size == 3

    # Test with trend kwargs
    model_fit_trend = fit_ar(input_1d, lags=1, exog=None, trend='ct')
    assert isinstance(model_fit_trend, AutoRegResultsWrapper)
    assert model_fit_trend.params.size == 3

    # Test with all kwargs and exog
    model_fit_all = fit_ar(input_1d, lags=1, exog=exog_1d,
                           seasonal=True, period=2, trend='ct')
    assert isinstance(model_fit_all, AutoRegResultsWrapper)
    assert model_fit_all.params.size == 5


def test_fit_ar_errors(input_1d, input_2d):
    # Test lags value out of bound
    with pytest.raises(ValueError):
        fit_ar(input_1d, len(input_1d) + 1)

    # Test invalid input dimension
    with pytest.raises(ValueError):
        fit_ar(input_2d, lags=3)

    # Test invalid lags input types
    with pytest.raises(TypeError):
        fit_ar(input_1d, 1.5)
    with pytest.raises(TypeError):
        fit_ar(input_1d, [1, 2.5, 3])
    with pytest.raises(ValueError):
        fit_ar(input_1d, [-1, 2, 3])

    # Test invalid kwargs
    with pytest.raises(ValueError):
        fit_ar(input_1d, lags=1, exog=None, seasonal='True')
    with pytest.raises(ValueError):
        fit_ar(input_1d, lags=1, exog=None, trend='invalid')
    with pytest.raises(ValueError):
        fit_ar(input_1d, lags=1, exog=None, seasonal=True, period=0)
    with pytest.raises(TypeError):
        fit_ar(input_1d, lags=1, exog=None, trend=True)


@pytest.mark.parametrize('arima_order', [(1, 0, 0), (2, 1, 2), (0, 0, 1), (3, 2, 0)])
def test_fit_arima(input_1d, exog_1d, exog_2d, arima_order):
    """
    Testing ARIMA model fitting with different orders and with or without exogenous variables.
    """
    # Test with no exog
    model_fit = fit_arima(input_1d, arima_order, exog=None)
    assert isinstance(model_fit, ARIMAResultsWrapper)

    # Test with 1D exog
    model_fit_exog_1d = fit_arima(input_1d, arima_order, exog=exog_1d)
    assert isinstance(model_fit_exog_1d, ARIMAResultsWrapper)

    # Test with 2D exog
    model_fit_exog_2d = fit_arima(input_1d, arima_order, exog=exog_2d)
    assert isinstance(model_fit_exog_2d, ARIMAResultsWrapper)


def test_fit_arima_errors(input_1d, exog_1d, exog_2d):
    """
    Testing ARIMA model fitting with invalid orders and exogenous variables.
    """
    # Test invalid arima_order input types
    with pytest.raises(ValueError):
        fit_arima(input_1d, (1, 0))  # less than 3 elements
    with pytest.raises(ValueError):
        fit_arima(input_1d, (1, 0, 0, 1))  # more than 3 elements

    # Test invalid exog dimensions
    with pytest.raises(ValueError):
        fit_arima(input_1d, (1, 0, 0), np.random.rand(100, 2, 2))

    # Test with incompatible exog size
    with pytest.raises(ValueError):
        fit_arima(input_1d, (1, 0, 0), np.random.rand(101, 1))


# pairs of valid (arima_order, sarima_order)
valid_orders = [
    ((1, 0, 0), (1, 0, 0, 2)),
    ((1, 0, 0), (0, 1, 2, 2)),
    ((2, 1, 2), (1, 0, 0, 3)),  # high order ARIMA with simple seasonal ARIMA
    ((2, 1, 2), (2, 0, 1, 4)),  # high order ARIMA with high order seasonal ARIMA
    ((1, 0, 0), (2, 0, 1, 4)),  # simple ARIMA with high order seasonal ARIMA
    ((0, 0, 1), (1, 0, 0, 2)),  # simple MA ARIMA with simple seasonal ARIMA
    ((0, 0, 1), (0, 0, 0, 2)),  # simple MA ARIMA with no seasonal ARIMA
    ((3, 2, 0), (0, 0, 0, 2))  # high order AR ARIMA with no seasonal ARIMA
]


@pytest.mark.parametrize('orders', valid_orders)
def test_fit_sarima(input_1d, exog_1d, exog_2d, orders):
    """
    Testing SARIMA model fitting with different orders and with or without exogenous variables.
    """
    arima_order, sarima_order = orders

    # Test with no exog and arima_order
    model_fit = fit_sarima(input_1d, sarima_order, None, exog=None)
    assert isinstance(model_fit, SARIMAXResultsWrapper)

    # Test with arima_order and 1D exog
    model_fit_exog_1d = fit_sarima(
        input_1d, sarima_order, arima_order, exog=exog_1d)
    assert isinstance(model_fit_exog_1d, SARIMAXResultsWrapper)

    # Test with arima_order and 2D exog
    model_fit_exog_2d = fit_sarima(
        input_1d, sarima_order, arima_order, exog=exog_2d)
    assert isinstance(model_fit_exog_2d, SARIMAXResultsWrapper)


def test_fit_sarima_errors(input_1d):
    """
    Testing SARIMA model fitting with invalid orders and exogenous variables.
    """
    # Test invalid arima_order input types
    with pytest.raises(ValueError):
        # sarima_order has less than 4 elements
        fit_sarima(input_1d, (1, 0, 0, 1), (1, 0, 0))
    with pytest.raises(ValueError):
        # sarima_order has more than 4 elements
        fit_sarima(input_1d, (1, 0, 0, 2, 1), (1, 0, 0))
    with pytest.raises(ValueError):
        # arima_order has less than 3 elements
        fit_sarima(input_1d, (1, 0, 0, 2), (1, 0))
    with pytest.raises(ValueError):
        # arima_order has more than 3 elements
        fit_sarima(input_1d, (1, 0, 0, 2), (1, 0, 0, 1))
    with pytest.raises(ValueError):
        # sarima_order's seasonality < 2
        fit_sarima(input_1d, (1, 0, 0, 1), (1, 0, 0))

    # Test invalid exog dimensions
    with pytest.raises(ValueError):
        fit_sarima(input_1d, (1, 0, 0, 2), (1, 0, 0),
                   np.random.rand(100, 2, 2))

    # Test with incompatible exog size
    with pytest.raises(ValueError):
        fit_sarima(input_1d, (1, 0, 0, 2), (1, 0, 0), np.random.rand(101, 1))

    # Test duplication of lags
    with pytest.raises(ValueError):
        # 'p' >= 's' and 'P' != 0
        fit_sarima(input_1d, (1, 0, 0, 2), (3, 0, 0))
    with pytest.raises(ValueError):
        # 'q' >= 's' and 'Q' != 0
        fit_sarima(input_1d, (0, 0, 1, 2), (0, 0, 3))
