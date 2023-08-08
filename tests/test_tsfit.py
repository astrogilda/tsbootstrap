import math

import numpy as np
import pytest
from arch.univariate.base import ARCHModelResult
from hypothesis import given, settings
from hypothesis.extra import numpy as npy
from hypothesis.strategies import (
    floats,
    integers,
    just,
    lists,
    sampled_from,
    tuples,
)
from numpy.linalg import LinAlgError
from statsmodels.tsa.ar_model import AutoRegResultsWrapper
from statsmodels.tsa.arima.model import ARIMAResultsWrapper
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper
from ts_bs.tsfit import TSFit


# Test data strategy
def high_variance_floats():
    return lists(
        floats(
            min_value=1, max_value=100, allow_infinity=False, allow_nan=False
        ),
        min_size=100,
        max_size=100,
    ).filter(lambda generated_list: np.var(generated_list) > 0.01)


test_data = high_variance_floats()


# Test order strategy
ar_order_strategy = lists(
    integers(min_value=1, max_value=10), min_size=1, max_size=10
)
var_arch_order_strategy = integers(min_value=1, max_value=10)
arima_order_strategy = tuples(
    integers(min_value=1, max_value=10),
    integers(min_value=0, max_value=2),
    integers(min_value=0, max_value=2),
)


def valid_sarima_order():
    p = integers(min_value=1, max_value=10)
    d = integers(min_value=0, max_value=2)
    q = integers(min_value=0, max_value=2)
    s = integers(min_value=2, max_value=10)

    return tuples(p, d, q, s).filter(
        lambda order: (order[-1] >= 2)
        and (order[0] < order[-1] or order[0] == 0)
        and (order[2] < order[-1] or order[2] == 0)
    )


sarima_order_strategy = valid_sarima_order()

# Test model type strategy
model_type_strategy = sampled_from(["ar", "arima", "sarima", "var", "arch"])


# Test optional exog strategy
exog_strategy = lists(
    lists(
        floats(
            min_value=-100,
            max_value=100,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=2,
        max_size=2,
    ),
    min_size=100,
    max_size=100,
)

# Test invalid order strategy
invalid_order_strategy = tuples(
    integers(min_value=1, max_value=10), integers(min_value=1, max_value=10)
)

# Test invalid model type strategy
invalid_model_type_strategy = sampled_from(["invalid_model", "not_supported"])

# Test invalid data strategy
invalid_data_strategy = npy.arrays(
    dtype=float,
    shape=(100, 0),
    elements=floats(min_value=-100, max_value=100),
    unique=True,
)


# Test TSFit initialization
@given(order=ar_order_strategy, model_type=just("ar"))
def test_init_ar(order, model_type):
    tsfit = TSFit(order, model_type)
    assert tsfit.order == order
    assert tsfit.model_type == model_type.lower()


@given(order=arima_order_strategy, model_type=just("arima"))
def test_init_arima(order, model_type):
    tsfit = TSFit(order, model_type)
    assert tsfit.order == order
    assert tsfit.model_type == model_type.lower()


@given(order=arima_order_strategy, model_type=just("sarima"))
def test_init_sarima(order, model_type):
    print(order, model_type)
    tsfit = TSFit(order, model_type)
    assert tsfit.order == order
    assert tsfit.model_type == model_type.lower()


@given(order=var_arch_order_strategy, model_type=sampled_from(["var", "arch"]))
def test_init_var_arch(order, model_type):
    tsfit = TSFit(order, model_type)
    assert tsfit.order == order
    assert tsfit.model_type == model_type.lower()


# Test fit method with valid inputs
@settings(deadline=None)
@given(data=test_data, order=ar_order_strategy, model_type=just("ar"))
def test_fit_valid_ar(data, order, model_type):
    order = list(np.unique(np.array(order)))
    data = np.array(data).reshape(-1, 1)
    tsfit = TSFit(order, model_type)
    fitted_model = tsfit.fit(data).model
    assert isinstance(fitted_model, AutoRegResultsWrapper)


@settings(deadline=None)
@given(data=test_data, order=arima_order_strategy, model_type=just("arima"))
def test_fit_valid_arima(data, order, model_type):
    data = np.array(data).reshape(-1, 1)
    tsfit = TSFit(order, model_type)
    var = np.var(data)
    if not math.isclose(var, 0, abs_tol=0.01):
        fitted_model = tsfit.fit(data).model
        assert isinstance(fitted_model, ARIMAResultsWrapper)


@settings(deadline=None)
@given(data=test_data, order=sarima_order_strategy, model_type=just("sarima"))
def test_fit_valid_sarima(data, order, model_type):
    data = np.array(data).reshape(-1, 1)
    tsfit = TSFit(order, model_type)
    var = np.var(data)
    if not math.isclose(var, 0, abs_tol=0.01):
        try:
            fitted_model = tsfit.fit(data).model
            assert isinstance(fitted_model, SARIMAXResultsWrapper)
        except LinAlgError:
            pass  # Ignore LinAlgError, as it's expected in some cases


@settings(deadline=None)
@given(
    data=test_data,
    order=var_arch_order_strategy,
    model_type=sampled_from(["var", "arch"]),
)
def test_fit_valid_var_arch(data, order, model_type):
    tsfit = TSFit(order, model_type)
    data = np.array(data).reshape(-1, 1)
    var = np.var(data)
    if model_type == "var":
        data = np.hstack((data, data))
    if model_type == "var":
        if not math.isclose(var, 0, abs_tol=0.01):
            try:
                fitted_model = tsfit.fit(data).model
                assert isinstance(fitted_model, VARResultsWrapper)
            except ValueError as e:
                if "x contains one or more constant columns" in str(e):
                    pass  # Ignore ValueError, as it's expected when the input contains one or more constant columns and trend == 'c'
                else:
                    raise  # If it's a different ValueError, raise it again
    else:
        if not math.isclose(var, 0, abs_tol=0.01):
            fitted_model = tsfit.fit(data).model
            assert isinstance(fitted_model, ARCHModelResult)
