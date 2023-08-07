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
from statsmodels.tsa.ar_model import AutoRegResultsWrapper
from statsmodels.tsa.arima.model import ARIMAResultsWrapper
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper
from ts_bs.tsfit import TSFit

# Test data strategy
test_data = lists(
    floats(
        min_value=-100, max_value=100, allow_infinity=False, allow_nan=False
    ),
    min_size=100,
    max_size=100,
)


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
sarima_order_strategy = tuples(
    integers(min_value=1, max_value=10),
    integers(min_value=0, max_value=2),
    integers(min_value=0, max_value=2),
    integers(min_value=2, max_value=10),
)

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
    if (
        (order[-1] < 2)
        or (order[0] >= order[-1] and order[0] != 0)
        or (order[2] >= order[-1] and order[2] != 0)
    ):
        with pytest.raises(ValueError):
            _ = tsfit.fit(data).model
    elif not math.isclose(var, 0, abs_tol=0.01):
        fitted_model = tsfit.fit(data).model
        assert isinstance(fitted_model, SARIMAXResultsWrapper)
    """
    try:
        fitted_model = tsfit.fit(data).model
        assert isinstance(fitted_model, SARIMAXResultsWrapper)
    except Exception as e:
        print(e)
        pass
    """


@settings(deadline=None)
@given(
    data=test_data,
    order=var_arch_order_strategy,
    model_type=sampled_from(["var", "arch"]),
)
def test_fit_valid_var_arch(data, order, model_type):
    tsfit = TSFit(order, model_type)
    data = np.array(data).reshape(-1, 1)
    if model_type == "var":
        data = np.hstack((data, data))

    var = np.var(data)
    fitted_model = tsfit.fit(data).model
    if model_type == "var":
        if not math.isclose(var, 0, abs_tol=0.01):
            fitted_model = tsfit.fit(data).model
            assert isinstance(fitted_model, VARResultsWrapper)
    else:
        if math.isclose(var, 0, abs_tol=0.01):
            with pytest.raises(RuntimeError):
                _ = tsfit.fit(data).model
        else:
            assert isinstance(fitted_model, ARCHModelResult)
