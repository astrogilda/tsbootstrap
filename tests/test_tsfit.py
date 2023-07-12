import pandas as pd
import pytest
import numpy as np
from hypothesis import given
from hypothesis.extra import numpy as npy
from hypothesis.strategies import integers, floats, sampled_from, tuples, just, lists
from hypothesis import settings
from statsmodels.tsa.ar_model import AutoRegResultsWrapper
from statsmodels.tsa.arima.model import ARIMAResultsWrapper
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper
from arch.univariate.base import ARCHModelResult
from sklearn.exceptions import NotFittedError
from utils.tsfit import TSFit


# Test data strategy
test_data = lists(floats(min_value=-100, max_value=100), min_size=100,
                  max_size=100)


# Test order strategy
ar_order_strategy = lists(
    integers(min_value=1, max_value=10), min_size=1, max_size=10)
var_arch_order_strategy = integers(min_value=1, max_value=10)
arima_order_strategy = tuples(integers(min_value=1, max_value=10),
                              integers(min_value=0, max_value=2),
                              integers(min_value=0, max_value=2))
sarima_order_strategy = tuples(integers(min_value=1, max_value=10),
                               integers(min_value=0, max_value=2),
                               integers(min_value=0, max_value=2),
                               integers(min_value=0, max_value=2))

# Test model type strategy
model_type_strategy = sampled_from(['ar', 'arima', 'sarima', 'var', 'arch'])


# Test optional exog strategy
exog_strategy = lists(lists(floats(
    min_value=-100, max_value=100, allow_nan=False, allow_infinity=False), min_size=2, max_size=2), min_size=100, max_size=100)

# Test invalid order strategy
invalid_order_strategy = tuples(
    integers(min_value=1, max_value=10), integers(min_value=1, max_value=10))

# Test invalid model type strategy
invalid_model_type_strategy = sampled_from(['invalid_model', 'not_supported'])

# Test invalid data strategy
invalid_data_strategy = npy.arrays(dtype=float, shape=(
    100, 0), elements=floats(min_value=-100, max_value=100), unique=True)


# Test TSFit initialization
@given(order=ar_order_strategy, model_type=just('ar'))
def test_init_ar(order, model_type):
    tsfit = TSFit(order, model_type)
    assert tsfit.order == order
    assert tsfit.model_type == model_type.lower()


@given(order=arima_order_strategy, model_type=just('arima'))
def test_init_arima(order, model_type):
    tsfit = TSFit(order, model_type)
    assert tsfit.order == order
    assert tsfit.model_type == model_type.lower()


@given(order=arima_order_strategy, model_type=just('sarima'))
def test_init_sarima(order, model_type):
    print(order, model_type)
    tsfit = TSFit(order, model_type)
    assert tsfit.order == order
    assert tsfit.model_type == model_type.lower()


@given(order=var_arch_order_strategy, model_type=sampled_from(['var', 'arch']))
def test_init_var_arch(order, model_type):
    tsfit = TSFit(order, model_type)
    assert tsfit.order == order
    assert tsfit.model_type == model_type.lower()


# Test fit method with valid inputs
@given(data=test_data, order=ar_order_strategy, model_type=just('ar'))
def test_fit_valid_ar(data, order, model_type):
    data = np.array(data).reshape(-1, 1)
    tsfit = TSFit(order, model_type)
    fitted_model = tsfit.fit(data)
    assert isinstance(fitted_model, AutoRegResultsWrapper)


@given(data=test_data, order=arima_order_strategy, model_type=just('arima'))
def test_fit_valid_arima(data, order, model_type):
    data = np.array(data).reshape(-1, 1)
    tsfit = TSFit(order, model_type)
    fitted_model = tsfit.fit(data)
    assert isinstance(fitted_model, ARIMAResultsWrapper)


@given(data=test_data, order=sarima_order_strategy, model_type=just('sarima'))
def test_fit_valid_sarima(data, order, model_type):
    data = np.array(data).reshape(-1, 1)
    tsfit = TSFit(order, model_type)
    fitted_model = tsfit.fit(data)
    assert isinstance(fitted_model, SARIMAXResultsWrapper)


@given(data=test_data, order=var_arch_order_strategy, model_type=sampled_from(['var', 'arch']))
def test_fit_valid_var_arch(data, order, model_type):
    tsfit = TSFit(order, model_type)
    if model_type == 'var':
        data = np.array(data)
        data = pd.DataFrame(data=np.hstack((data, data)), columns=['y1', 'y2'])
    else:
        data = np.array(data).reshape(-1, 1)
    fitted_model = tsfit.fit(data)
    if model_type == 'var':
        assert isinstance(fitted_model, VARResultsWrapper)
    else:
        assert isinstance(fitted_model, ARCHModelResult)
