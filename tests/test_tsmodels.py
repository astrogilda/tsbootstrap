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

    with pytest.raises(ValueError):
        fit_ar(input_1d, lags=1, exog=None, seasonal='True')
    with pytest.raises(ValueError):
        fit_ar(input_1d, lags=1, exog=None, trend='invalid')
    with pytest.raises(ValueError):
        fit_ar(input_1d, lags=1, exog=None, seasonal=True, period=0)
    with pytest.raises(TypeError):
        fit_ar(input_1d, lags=1, exog=None, trend=True)


'''

@pytest.mark.parametrize('order', [(1, 0, 1), (2, 1, 2), (3, 1, 3)])
def test_fit_arima(input_1d, order):
    model_fit = fit_arima(input_1d, order)
    assert isinstance(model_fit, ARIMAResultsWrapper)
    ar_order, diff_deg, ma_order = order
    assert model_fit.params.size == ar_order + ma_order + diff_deg + 1

    # Test invalid order
    with pytest.raises(ValueError):
        fit_arima(input_1d, (4, 2, 4))

    # Test invalid input dimension
    with pytest.raises(ValueError):
        fit_arima(np.random.rand(100, 2), order)

    # Test mismatched exog shape
    with pytest.raises(ValueError):
        fit_arima(input_1d, order, exog=np.random.rand(50, 1))


@pytest.mark.parametrize('order', [(1, 0, 1, 2), (2, 1, 2, 3), (0, 1, 0, 2)])
def test_fit_sarima(input_1d, order):
    model_fit = fit_sarima(input_1d, order)
    assert isinstance(model_fit, SARIMAXResultsWrapper)
    ar_order, diff_deg, ma_order, s_order = order
    assert model_fit.params.size == ar_order + ma_order + diff_deg + s_order + 1

    # Test invalid seasonal periodicity
    with pytest.raises(ValueError):
        fit_sarima(input_1d, (1, 1, 1, 1))

    # Test invalid input dimension
    with pytest.raises(ValueError):
        fit_sarima(np.random.rand(100, 2), order)

    # Test mismatched exog shape
    with pytest.raises(ValueError):
        fit_sarima(input_1d, order, exog=np.random.rand(50, 1))


@pytest.mark.parametrize('lags', [1, 2, 10, 50, 100])
def test_fit_var(input_2d, lags):
    model_fit = fit_var(input_2d, lags)
    assert isinstance(model_fit, VARResultsWrapper)
    _, n_vars = input_2d.shape
    assert model_fit.params.shape == (lags * n_vars, n_vars)

    # Test lags value out of bound
    with pytest.raises(ValueError):
        fit_var(input_2d, input_2d.shape[0] + 1)

    # Test invalid input dimensions
    with pytest.raises(ValueError):
        fit_var(np.random.rand(100), lags)
    with pytest.raises(ValueError):
        fit_var(np.random.rand(100, 1), lags)

    # Test mismatched exog shape
    with pytest.raises(ValueError):
        fit_var(input_2d, lags, exog=np.random.rand(50, 1))


@pytest.mark.parametrize('lags', [1, 2, 10, 50, 100])
def test_fit_arch(input_1d, lags):
    model_fit = fit_arch(input_1d, lags)
    assert isinstance(model_fit, ARCHModelResult)
    assert model_fit.params.size == lags + 1

    # Test lags value out of bound
    with pytest.raises(ValueError):
        fit_arch(input_1d, len(input_1d) + 1)

    # Test invalid input dimension
    with pytest.raises(ValueError):
        fit_arch(np.random.rand(100, 2), lags)

'''
