import pytest
import numpy as np
from utils.tsmodels import fit_ar, fit_arima, fit_sarima, fit_var, fit_arch, validate_exog
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


def test_validate_exog():
    exog_1d = np.random.rand(100)
    exog_2d = exog_1d.reshape(-1, 1)
    exog_3d = np.random.rand(100, 1, 2)

    assert validate_exog(None) is None
    assert np.array_equal(validate_exog(exog_1d), exog_2d)
    with pytest.raises(ValueError):
        validate_exog(exog_3d)


@pytest.mark.parametrize('lags', [1, 2, 10, 50, 99, [1, 3], [2, 5, 10], [1, 10, 50]])
def test_fit_ar(input_1d, lags):
    model_fit = fit_ar(input_1d, lags)
    assert isinstance(model_fit, AutoRegResultsWrapper)

    if isinstance(lags, list):
        assert model_fit.params.size == len(lags) + 1
    else:
        assert model_fit.params.size == lags + 1

    # Test lags value out of bound
    with pytest.raises(ValueError):
        fit_ar(input_1d, len(input_1d) + 1)

    # Test invalid input dimension
    with pytest.raises(ValueError):
        fit_ar(np.random.rand(100, 2), lags)

    # Test invalid lags input types
    with pytest.raises(TypeError):
        fit_ar(input_1d, 1.5)
    with pytest.raises(TypeError):
        fit_ar(input_1d, [1, 2.5, 3])
    with pytest.raises(ValueError):
        fit_ar(input_1d, [-1, 2, 3])


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
