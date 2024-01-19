import sys

import numpy as np
import pytest
from numpy.linalg import LinAlgError
from numpy.testing import assert_allclose
from skbase.utils.dependencies import _check_soft_dependencies
from tsbootstrap import TimeSeriesModel


@pytest.fixture(scope="module")
def input_1d():
    return np.random.rand(100)


@pytest.fixture(scope="module")
def input_2d():
    return np.random.rand(100, 2)


@pytest.fixture
def input_2d_short():
    return np.random.rand(10, 2)


@pytest.fixture
def exog_2d_short():
    return np.random.rand(10, 2)


@pytest.fixture(scope="module")
def exog_1d():
    return np.random.rand(100)


@pytest.fixture(scope="module")
def exog_2d():
    return np.random.rand(100, 2)


@pytest.mark.skipif(
    not _check_soft_dependencies(["arch", "statsmodels"], severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize(
    "order", [1, 2, 10, 50, 99, [1, 3], [2, 5, 10], [1, 10, 50]]
)
def test_fit_ar(input_1d, exog_1d, order):
    # Test with no exog, seasonal order, and set trend to 'c' (constant, default)
    from statsmodels.tsa.ar_model import AutoRegResultsWrapper

    max_lag = (input_1d.shape[0] - 1) // 2
    print(f"max_lag: {max_lag}")
    tsm = TimeSeriesModel(X=input_1d, y=None, model_type="ar")
    tsm_exog = TimeSeriesModel(X=input_1d, y=exog_1d, model_type="ar")
    if np.max(order) <= max_lag:
        model_fit = tsm.fit(order=order)
        assert isinstance(model_fit, AutoRegResultsWrapper)

        # Test with exog
        model_fit_exog = tsm_exog.fit(order=order)
        assert isinstance(model_fit_exog, AutoRegResultsWrapper)

        # Test with seasonal and period kwargs
        model_fit_seasonal = tsm.fit(order=order, seasonal=True, period=2)
        # fit_ar(input_1d, order=1, exog=None, seasonal=True, period=2)
        assert isinstance(model_fit_seasonal, AutoRegResultsWrapper)

        # Test with trend kwargs
        model_fit_trend = tsm.fit(order=order, trend="ct")
        assert isinstance(model_fit_trend, AutoRegResultsWrapper)

        # Test with all kwargs and exog
        model_fit_all = tsm_exog.fit(
            order=order, seasonal=True, period=2, trend="ct"
        )
        assert isinstance(model_fit_all, AutoRegResultsWrapper)

        if isinstance(order, list):
            assert model_fit.params.size == len(order) + 1
            assert model_fit_exog.params.size == len(order) + 2
            assert model_fit_seasonal.params.size == len(order) + 2
            assert model_fit_trend.params.size == len(order) + 2
            assert model_fit_all.params.size == len(order) + 4

        else:
            assert model_fit.params.size == order + 1
            assert model_fit_exog.params.size == order + 2
            assert model_fit_seasonal.params.size == order + 2
            assert model_fit_trend.params.size == order + 2
            assert model_fit_all.params.size == order + 4

    else:
        with pytest.raises(
            ValueError,
            match=f"Maximum allowed lag value exceeded. The allowed maximum is {max_lag}",
        ):
            tsm.fit(order=order)
        with pytest.raises(
            ValueError,
            match=f"Maximum allowed lag value exceeded. The allowed maximum is {max_lag}",
        ):
            tsm_exog.fit(order=order)
        with pytest.raises(
            ValueError, match="Maximum allowed lag value exceeded."
        ):
            tsm.fit(order=order, seasonal=True, period=2)
        with pytest.raises(
            ValueError, match="Maximum allowed lag value exceeded."
        ):
            tsm.fit(order=order, trend="ct")
        with pytest.raises(
            ValueError, match="Maximum allowed lag value exceeded."
        ):
            tsm_exog.fit(order=order, seasonal=True, period=2, trend="ct")


@pytest.mark.skipif(
    not _check_soft_dependencies(["arch", "statsmodels"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_fit_ar_errors(input_1d, input_2d):
    # Test order value out of bound
    with pytest.raises(ValueError):
        tsm = TimeSeriesModel(X=input_1d, y=None, model_type="ar")
        tsm.fit(len(input_1d) + 1)

    # Test invalid input dimension
    with pytest.raises(ValueError):
        tsm = TimeSeriesModel(X=input_2d, y=None, model_type="ar")
        tsm.fit(3)

    # Test invalid order input types
    tsm = TimeSeriesModel(X=input_1d, y=None, model_type="ar")
    with pytest.raises(TypeError):
        tsm.fit(1.5)
    with pytest.raises(TypeError):
        tsm.fit([1, 2.5, 3])
    with pytest.raises(ValueError):
        tsm.fit([-1, 2, 3])

    # Test invalid kwargs
    tsm = TimeSeriesModel(X=input_1d, y=None, model_type="ar")
    with pytest.raises(ValueError):
        tsm.fit(order=1, seasonal="True")
    with pytest.raises(ValueError):
        tsm.fit(order=1, trend="invalid")
    with pytest.raises(ValueError):
        tsm.fit(order=1, seasonal=True, period=0)
    with pytest.raises(TypeError):
        tsm.fit(order=1, rend=True)


@pytest.mark.skipif(
    not _check_soft_dependencies(["arch", "statsmodels"], severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize(
    "arima_order", [(1, 0, 0), (2, 1, 2), (0, 0, 1), (3, 2, 0)]
)
def test_fit_arima(input_1d, exog_1d, exog_2d, arima_order):
    """
    Testing ARIMA model fitting with different orders and with or without exogenous variables.
    """
    from statsmodels.tsa.arima.model import ARIMAResultsWrapper

    # Test with no exog
    tsm = TimeSeriesModel(X=input_1d, y=None, model_type="arima")
    try:
        model_fit = tsm.fit(arima_order)
        assert isinstance(model_fit, ARIMAResultsWrapper)
    except LinAlgError:
        pass

    # Test with 1D exog
    tsm = TimeSeriesModel(X=input_1d, y=exog_1d, model_type="arima")
    try:
        model_fit_exog_1d = tsm.fit(arima_order)
        assert isinstance(model_fit_exog_1d, ARIMAResultsWrapper)
    except LinAlgError:
        pass

    # Test with 2D exog
    tsm = TimeSeriesModel(X=input_1d, y=exog_2d, model_type="arima")
    try:
        model_fit_exog_2d = tsm.fit(arima_order)
        assert isinstance(model_fit_exog_2d, ARIMAResultsWrapper)
    except LinAlgError:
        pass


@pytest.mark.skipif(
    not _check_soft_dependencies(["arch", "statsmodels"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_fit_arima_errors(input_1d, exog_1d, exog_2d):
    """
    Testing ARIMA model fitting with invalid orders and exogenous variables.
    """
    # Test invalid arima_order input types
    with pytest.raises(ValueError):
        tsm = TimeSeriesModel(X=input_1d, y=None, model_type="arima")
        tsm.fit((1, 0))  # less than 3 elements
    with pytest.raises(ValueError):
        tsm = TimeSeriesModel(X=input_1d, y=None, model_type="arima")
        tsm.fit((1, 0, 0, 1))  # more than 3 elements

    # Test invalid exog dimensions
    with pytest.raises(ValueError):
        tsm = TimeSeriesModel(
            X=input_1d, y=np.random.rand(100, 2, 2), model_type="arima"
        )

    # Test with incompatible exog size
    with pytest.raises(ValueError):
        tsm = TimeSeriesModel(
            X=input_1d, y=np.random.rand(101, 1), model_type="arima"
        )


# pairs of valid (arima_order, sarima_order)
valid_orders = [
    ((1, 0, 0), (1, 0, 0, 2)),
    ((1, 0, 0), (0, 1, 2, 2)),
    ((2, 1, 2), (1, 0, 0, 3)),  # high order ARIMA with simple seasonal ARIMA
    pytest.param(
        (
            (2, 1, 2),
            (2, 0, 1, 4),
        ),  # high order ARIMA with high order seasonal ARIMA
        marks=pytest.mark.skipif(
            sys.platform.startswith("linux"),
            reason="Skipping for Python on Ubuntu",
        ),
    ),
    pytest.param(
        (
            (1, 0, 0),
            (2, 0, 1, 4),
        ),  # simple ARIMA with high order seasonal ARIMA
        marks=pytest.mark.skipif(
            sys.platform.startswith("linux"),
            reason="Skipping for Python on Ubuntu",
        ),
    ),
    ((0, 0, 1), (1, 0, 0, 2)),  # simple MA ARIMA with simple seasonal ARIMA
    ((0, 0, 1), (0, 0, 0, 2)),  # simple MA ARIMA with no seasonal ARIMA
    ((3, 2, 0), (0, 0, 0, 2)),  # high order AR ARIMA with no seasonal ARIMA
]

# sys.version_info >= (3, 10) and


@pytest.mark.skipif(
    not _check_soft_dependencies(["arch", "statsmodels"], severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("orders", valid_orders)
def test_fit_sarima(input_1d, exog_1d, exog_2d, orders):
    """
    Testing SARIMA model fitting with different orders and with or without exogenous variables.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper

    arima_order, sarima_order = orders

    # Test with no exog and arima_order
    tsm = TimeSeriesModel(X=input_1d, y=None, model_type="sarima")
    try:
        model_fit = tsm.fit(order=sarima_order, arima_order=None)
        assert isinstance(model_fit, SARIMAXResultsWrapper)
    except LinAlgError:
        pass

    # Test with arima_order and 1D exog
    tsm = TimeSeriesModel(X=input_1d, y=exog_1d, model_type="sarima")
    try:
        model_fit_exog_1d = tsm.fit(
            order=sarima_order, arima_order=arima_order
        )
        assert isinstance(model_fit_exog_1d, SARIMAXResultsWrapper)
    except LinAlgError:
        pass

    # Test with arima_order and 2D exog
    tsm = TimeSeriesModel(X=input_1d, y=exog_2d, model_type="sarima")
    try:
        model_fit_exog_2d = tsm.fit(
            order=sarima_order, arima_order=arima_order
        )
        assert isinstance(model_fit_exog_2d, SARIMAXResultsWrapper)
    except LinAlgError:
        pass


@pytest.mark.skipif(
    not _check_soft_dependencies(["arch", "statsmodels"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_fit_sarima_errors(input_1d):
    """
    Testing SARIMA model fitting with invalid orders and exogenous variables.
    """
    # Test invalid arima_order input types
    tsm = TimeSeriesModel(X=input_1d, y=None, model_type="sarima")
    with pytest.raises(ValueError):
        # sarima_order has less than 4 elements
        tsm.fit(order=(1, 0, 0), arima_order=(1, 0, 0))
    with pytest.raises(ValueError):
        # sarima_order has more than 4 elements
        tsm.fit(order=(1, 0, 0, 2, 1), arima_order=(1, 0, 0))
    with pytest.raises(ValueError):
        # arima_order has less than 3 elements
        tsm.fit(order=(1, 0, 0, 2), arima_order=(1, 0))
    with pytest.raises(ValueError):
        # arima_order has more than 3 elements
        tsm.fit(order=(1, 0, 0, 2), arima_order=(1, 0, 0, 1))
    with pytest.raises(ValueError):
        # sarima_order's seasonality < 2
        tsm.fit(order=(1, 0, 0, 1), arima_order=(1, 0, 0))

    # Test invalid exog dimensions
    with pytest.raises(ValueError):
        tsm = TimeSeriesModel(
            X=input_1d, y=np.random.rand(100, 2, 2), model_type="sarima"
        )

    # Test with incompatible exog size
    with pytest.raises(ValueError):
        tsm = TimeSeriesModel(
            X=input_1d, y=np.random.rand(101, 1), model_type="sarima"
        )

    # Test duplication of order
    tsm = TimeSeriesModel(X=input_1d, y=None, model_type="sarima")
    with pytest.raises(ValueError):
        # 'p' >= 's' and 'P' != 0
        tsm.fit(order=(1, 0, 0, 2), arima_order=(3, 0, 0))
    with pytest.raises(ValueError):
        # 'q' >= 's' and 'Q' != 0
        tsm.fit(order=(0, 0, 1, 2), arima_order=(0, 0, 3))


# Tests for fit_var
@pytest.mark.skipif(
    not _check_soft_dependencies(["arch", "statsmodels"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_fit_var(input_2d, input_2d_short, exog_1d, exog_2d, exog_2d_short):
    """Testing VAR model fitting, with orders and with/without exogenous variables."""
    from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

    # Test with no exog
    tsm = TimeSeriesModel(X=input_2d, y=None, model_type="var")
    try:
        model_fit = tsm.fit()
        assert isinstance(model_fit, VARResultsWrapper)
    except LinAlgError:
        pass

    # Test with exog
    tsm = TimeSeriesModel(X=input_2d, y=exog_1d, model_type="var")
    try:
        model_fit_exog = tsm.fit()
        assert isinstance(model_fit_exog, VARResultsWrapper)
    except LinAlgError:
        pass

    # Test with different kwargs
    tsm = TimeSeriesModel(X=input_2d, y=exog_2d, model_type="var")
    try:
        model_fit_no_trend = tsm.fit(trend="n")
        assert isinstance(model_fit_no_trend, VARResultsWrapper)
        assert model_fit_no_trend.k_trend == 0
    except LinAlgError:
        pass

    tsm = TimeSeriesModel(X=input_2d, y=exog_2d, model_type="var")
    try:
        model_fit_trend = tsm.fit(trend="c")
        assert isinstance(model_fit_trend, VARResultsWrapper)
        assert model_fit_trend.k_trend == 1
    except LinAlgError:
        pass

    tsm = TimeSeriesModel(X=input_2d, y=exog_2d, model_type="var")
    try:
        model_fit_trend = tsm.fit(trend="ct")
        assert isinstance(model_fit_trend, VARResultsWrapper)
        assert model_fit_trend.k_trend == 2
    except LinAlgError:
        pass

    tsm = TimeSeriesModel(X=input_2d, y=exog_2d, model_type="var")
    try:
        model_fit_trend = tsm.fit(trend="ctt")
        assert isinstance(model_fit_trend, VARResultsWrapper)
        assert model_fit_trend.k_trend == 3
    except LinAlgError:
        pass

    # Test with 1D exog
    tsm = TimeSeriesModel(X=input_2d, y=exog_1d, model_type="var")
    model_fit_exog_1d = tsm.fit()
    assert isinstance(model_fit_exog_1d, VARResultsWrapper)

    # Test with 2D exog of different width
    exog_2d_wide = np.random.rand(input_2d.shape[0], input_2d.shape[1] + 1)
    tsm = TimeSeriesModel(X=input_2d, y=exog_2d_wide, model_type="var")
    try:
        model_fit_exog_2d_wide = tsm.fit()
        assert isinstance(model_fit_exog_2d_wide, VARResultsWrapper)
    except LinAlgError:
        pass

    # Test with short input arrays
    tsm = TimeSeriesModel(X=input_2d_short, y=exog_2d_short, model_type="var")
    try:
        model_fit_short = tsm.fit()
        assert isinstance(model_fit_short, VARResultsWrapper)
    except LinAlgError:
        pass

    # Test deterministic input
    deterministic_2d = np.ones_like(input_2d)
    tsm = TimeSeriesModel(X=deterministic_2d, y=exog_2d, model_type="var")
    try:
        model_fit_deterministic = tsm.fit(trend="n")
        assert isinstance(model_fit_deterministic, VARResultsWrapper)
        assert_allclose(model_fit_deterministic.endog, deterministic_2d)
    except LinAlgError:
        pass


@pytest.mark.skipif(
    not _check_soft_dependencies(["arch", "statsmodels"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_fit_var_errors(input_1d, input_2d, exog_2d):
    # Test invalid input dimension
    with pytest.raises(ValueError):
        TimeSeriesModel(X=input_1d, y=None, model_type="var")

    # Test exog of different length
    with pytest.raises(ValueError):
        TimeSeriesModel(
            X=input_2d,
            y=np.random.rand(input_2d.shape[0] + 1),
            model_type="var",
        )

    # Test exog of different number of dimensions
    with pytest.raises(ValueError):
        TimeSeriesModel(
            X=input_2d,
            y=np.random.rand(input_2d.shape[0], input_2d.shape[1], 2),
            model_type="var",
        )

    # Test invalid trend option
    tsm = TimeSeriesModel(X=input_2d, y=exog_2d, model_type="var")
    with pytest.raises(ValueError):
        tsm.fit(trend="invalid")

    # Test 3D input array
    with pytest.raises(ValueError):
        TimeSeriesModel(
            X=np.random.rand(input_2d.shape[0], input_2d.shape[1], 2),
            y=None,
            model_type="var",
        )

    # Test invalid dtype
    with pytest.raises(TypeError):
        TimeSeriesModel(X=input_2d.astype(str), y=None, model_type="var")

    # Test with empty arrays
    with pytest.raises(ValueError):
        TimeSeriesModel(X=np.empty(shape=(0, 0)), y=None, model_type="var")
    with pytest.raises(ValueError):
        TimeSeriesModel(
            X=np.empty(shape=(0, 0)),
            y=np.empty(shape=(0, 0)),
            model_type="var",
        )


@pytest.mark.skipif(
    not _check_soft_dependencies(["arch", "statsmodels"], severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("q", [1, 2])
@pytest.mark.parametrize(
    "arch_model_type", ["GARCH", "EGARCH", "TARCH", "AGARCH"]
)
@pytest.mark.parametrize("order", [1, 2, [1, 2], 49])
@pytest.mark.parametrize("mean_type", ["zero", "AR"])
def test_fit_arch(input_1d, exog_1d, p, q, arch_model_type, order, mean_type):
    """Testing ARCH model fitting, with orders and with/without exogenous variables."""
    from arch.univariate.base import ARCHModelResult

    # TODO: figure out max_lag for arch_models; currently using 49 copied from fit_ar
    max_lag = (input_1d.shape[0] - 1) // 2

    if np.max(order) <= max_lag:
        # Test with no exog
        tsm = TimeSeriesModel(X=input_1d, y=None, model_type="arch")
        model_fit = tsm.fit(
            p=p,
            q=q,
            arch_model_type=arch_model_type,
            order=order,
            mean_type=mean_type,
        )
        assert isinstance(model_fit, ARCHModelResult)

        # Test with exog
        tsm = TimeSeriesModel(X=input_1d, y=exog_1d, model_type="arch")
        model_fit_exog = tsm.fit(
            p=p,
            q=q,
            arch_model_type=arch_model_type,
            order=order,
            mean_type=mean_type,
        )
        assert isinstance(model_fit_exog, ARCHModelResult)

    else:
        with pytest.raises(ValueError):
            tsm = TimeSeriesModel(X=input_1d, y=None, model_type="arch")
            tsm.fit(
                p=p,
                q=q,
                arch_model_type=arch_model_type,
                order=order,
                mean_type=mean_type,
            )
        with pytest.raises(ValueError):
            tsm = TimeSeriesModel(X=input_1d, y=exog_1d, model_type="arch")
            tsm.fit(
                p=p,
                q=q,
                arch_model_type=arch_model_type,
                order=order,
                mean_type=mean_type,
            )


@pytest.mark.skipif(
    not _check_soft_dependencies(["arch", "statsmodels"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_fit_arch_errors(input_1d, input_2d):
    # Test invalid input dimension
    with pytest.raises(ValueError):
        TimeSeriesModel(X=input_2d, y=None, model_type="arch")

    # Test invalid order input types
    tsm = TimeSeriesModel(X=input_1d, y=None, model_type="arch")
    with pytest.raises(TypeError):
        tsm.fit(p=1, q=1, arch_model_type="GARCH", order=1.5)
    with pytest.raises(TypeError):
        tsm.fit(p=1, q=1, arch_model_type="GARCH", order=[1, 2.5, 3])
    with pytest.raises(ValueError):
        tsm.fit(p=1, q=1, arch_model_type="GARCH", order=[-1, 2, 3])

    # Test invalid model_type
    tsm = TimeSeriesModel(X=input_1d, y=None, model_type="arch")
    with pytest.raises(ValueError):
        tsm.fit(p=1, q=1, arch_model_type="INVALID", order=1)

    # Test model_type set to 'ARCH'
    tsm = TimeSeriesModel(X=input_1d, y=None, model_type="arch")
    with pytest.raises(ValueError):
        tsm.fit(p=1, q=1, order=1, arch_model_type=None)

    # Test input with NaN values
    with pytest.raises(ValueError, match="Input contains NaN."):
        TimeSeriesModel(
            X=np.array([1.0, 2.0, np.nan]), y=None, model_type="arch"
        )

    # Test exog with NaN values
    with pytest.raises(ValueError, match="Input contains NaN."):
        TimeSeriesModel(
            X=input_1d, y=np.array([1.0, 2.0, np.nan]), model_type="arch"
        )

    # Test with zero-length input
    with pytest.raises(ValueError):
        TimeSeriesModel(X=np.array([]), y=None, model_type="arch")

    # Test with single value input
    with pytest.raises(ValueError):
        TimeSeriesModel(X=np.array([1.0]), y=None, model_type="arch")
