import logging

import arch
import numpy as np
import pytest
from arch.univariate.base import ARCHModelResult
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.random import Generator, default_rng
from tsbootstrap import TimeSeriesSimulator
from tsbootstrap.utils.odds_and_ends import assert_arrays_compare
from tsbootstrap.utils.skbase_compat import safe_check_soft_dependencies as _check_soft_dependencies

# TODO: test for generate_samples_sieve
# TODO: test samples are same/different with same/different random seeds

MIN_INT = 0
MAX_INT = 2**32 - 1

# Condition to skip ARCH reproducibility test if arch environment is problematic
# This checks if arch is v7.x AND ARCHModelResult.simulate is missing.
SKIP_ARCH_REPRODUCIBILITY_TEST = False
try:
    if arch.__version__.startswith("7.") and not hasattr(ARCHModelResult, "simulate"):
        SKIP_ARCH_REPRODUCIBILITY_TEST = True
except Exception:
    # If arch or ARCHModelResult is not available for some reason, don't skip by default
    logging.exception(
        "Exception occurred while checking arch version and ARCHModelResult.simulate availability"
    )


# Define some common strategies for generating test data
integer_array = st.lists(st.integers(min_value=1, max_value=10), min_size=10, max_size=10).map(
    np.array
)
float_array = st.lists(
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
    min_size=10,
    max_size=10,
).map(lambda x: np.array(x).reshape(-1, 1))
float_array_unique = st.just(np.random.rand(10, 1))


def get_model(str, data):
    from statsmodels.tsa.ar_model import AutoReg
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.vector_ar.var_model import VAR

    if str == "ar":
        return AutoReg(data, lags=1)
    elif str == "arima":
        return ARIMA(data, order=(1, 0, 0))
    elif str == "sarima":
        return SARIMAX(data, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0))
    elif str == "var":
        return VAR(data)
    else:
        raise ValueError(f"Unknown model string: {str}")


def ar_model_strategy():
    return st.builds(lambda data: get_model("ar", data).fit(), float_array_unique)


def arima_model_strategy():
    return st.builds(lambda data: get_model("arima", data).fit(), float_array_unique)


def sarima_model_strategy():
    return st.builds(
        lambda data: get_model("sarima", data).fit(),
        float_array_unique,
    )


def _fit_var_model_for_strategy(data_for_var):
    """Helper function to fit VAR model, aiding type inference."""
    var_model_instance = get_model("var", data_for_var)
    # VAR.fit() can take maxlags. Trend is 'c' by default in VAR constructor.
    return var_model_instance.fit(maxlags=1)  # type: ignore


def var_model_strategy():
    return st.builds(
        _fit_var_model_for_strategy,
        float_array_unique.map(lambda x: np.column_stack([x, x])),
    )


def scale_and_fit_arch(data):
    from arch import arch_model

    scaled_data = data * np.sqrt(100 / np.var(data))
    return arch_model(scaled_data).fit()


def arch_model_strategy():
    return st.builds(scale_and_fit_arch, float_array_unique)


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
)
class TestARModel:
    class TestPassingCases:
        @given(
            fitted_model=ar_model_strategy(),
            X_fitted=float_array,
            rng=st.none()
            | st.integers(min_value=MIN_INT, max_value=MAX_INT)
            | st.just(default_rng()),
        )
        @settings(deadline=1000)
        def test_init_valid(self, fitted_model, X_fitted, rng):
            """Test that AR model initialization works with valid inputs."""
            TimeSeriesSimulator(fitted_model, X_fitted, rng)

        @given(
            fitted_model=ar_model_strategy(),
            X_fitted=float_array,
            rng=st.none()
            | st.integers(min_value=MIN_INT, max_value=MAX_INT)
            | st.just(default_rng()),
        )
        def test_fitted_model_valid(self, fitted_model, X_fitted, rng):
            """Test that AR model fitted_model property getter and setter work correctly."""
            simulator = TimeSeriesSimulator(fitted_model, X_fitted, rng)
            assert simulator.fitted_model == fitted_model

        @given(
            fitted_model=ar_model_strategy(),
            X_fitted=float_array,
            rng=st.none()
            | st.integers(min_value=MIN_INT, max_value=MAX_INT)
            | st.just(default_rng()),
        )
        def test_X_fitted_valid(self, fitted_model, X_fitted, rng):
            """Test that AR model X_fitted property getter and setter work correctly."""
            simulator = TimeSeriesSimulator(fitted_model, X_fitted, rng)
            assert np.allclose(simulator.X_fitted, X_fitted)

        @given(
            fitted_model=ar_model_strategy(),
            X_fitted=float_array,
            rng=st.none()
            | st.integers(min_value=MIN_INT, max_value=MAX_INT)
            | st.just(default_rng()),
        )
        def test_rng_valid(self, fitted_model, X_fitted, rng):
            """Test that AR model rng property getter and setter work correctly."""
            simulator = TimeSeriesSimulator(fitted_model, X_fitted, rng)
            assert isinstance(simulator.rng, Generator)

        @given(
            fitted_model=ar_model_strategy(),
            X_fitted=float_array,
            rng=st.none()
            | st.integers(min_value=MIN_INT, max_value=MAX_INT)
            | st.just(default_rng()),
            resids_lags=integer_array,
            resids_coefs=float_array,
            resids=float_array,
        )
        def test_simulate_ar_process_valid(
            self,
            fitted_model,
            X_fitted,
            rng,
            resids_lags,
            resids_coefs,
            resids,
        ):
            """Test that AR model simulation works with valid inputs."""
            simulator = TimeSeriesSimulator(fitted_model, X_fitted, rng)
            simulator.simulate_ar_process(
                resids_lags.tolist(),
                resids_coefs.reshape(1, -1),
                resids,
                n_samples=simulator.n_samples,
            )

        @given(
            fitted_model=ar_model_strategy(),
            X_fitted=float_array_unique,
            rng=st.none()
            | st.integers(min_value=MIN_INT, max_value=MAX_INT)
            | st.just(default_rng()),
            resids_lags=integer_array,
            resids_coefs=float_array_unique,
            resids=float_array_unique,
        )
        def test_simulate_ar_process_valid_large_input(
            self,
            fitted_model,
            X_fitted,
            rng,
            resids_lags,
            resids_coefs,
            resids,
        ):
            """Test that AR model simulation works with larger input arrays."""
            large_X_fitted = np.repeat(X_fitted, 10, axis=0)
            large_resids_lags = np.repeat(resids_lags, 10)
            large_resids_coefs = np.repeat(resids_coefs, 10).reshape(1, -1)
            large_resids = np.repeat(resids, 10)
            simulator = TimeSeriesSimulator(fitted_model, large_X_fitted, rng)
            simulator.simulate_ar_process(
                large_resids_lags.tolist(),
                large_resids_coefs,
                large_resids,
                n_samples=simulator.n_samples,
            )

        @settings(
            suppress_health_check=(HealthCheck.too_slow,),
            max_examples=10,
            deadline=None,
        )
        @given(
            fitted_model=ar_model_strategy(),
            X_fitted=float_array,
            resids_lags=integer_array,
            resids_coefs=float_array,
            resids=float_array,
        )
        def test_simulate_ar_same_rng(
            self, fitted_model, X_fitted, resids_lags, resids_coefs, resids
        ):
            """Test that AR model simulation gives same results with same rng."""
            rng_seed = 12345

            rng1 = np.random.default_rng(rng_seed)
            simulator1 = TimeSeriesSimulator(fitted_model, X_fitted, rng1)
            simulated_series1 = simulator1.simulate_ar_process(
                resids_lags.tolist(),
                resids_coefs.reshape(1, -1),
                resids,
                n_samples=simulator1.n_samples,
            )

            rng2 = np.random.default_rng(rng_seed)
            simulator2 = TimeSeriesSimulator(fitted_model, X_fitted, rng2)
            simulated_series2 = simulator2.simulate_ar_process(
                resids_lags.tolist(),
                resids_coefs.reshape(1, -1),
                resids,
                n_samples=simulator2.n_samples,
            )
            assert_arrays_compare(simulated_series1, simulated_series2)

        @given(
            fitted_model=ar_model_strategy(),
            X_fitted=float_array,
            resids_lags=integer_array,
            resids_coefs=float_array,
            resids=float_array,
        )
        def test_simulate_ar_different_rng(
            self, fitted_model, X_fitted, resids_lags, resids_coefs, resids
        ):
            """Test that AR model simulation gives different results with different rng."""
            rng = np.random.default_rng()
            simulator1 = TimeSeriesSimulator(fitted_model, X_fitted, rng)
            simulated_series1 = simulator1.simulate_ar_process(
                resids_lags.tolist(),
                resids_coefs.reshape(1, -1),
                resids,
                n_samples=simulator1.n_samples,
            )

            simulator2 = TimeSeriesSimulator(fitted_model, X_fitted, rng)
            simulated_series2 = simulator2.simulate_ar_process(
                resids_lags.tolist(),
                resids_coefs.reshape(1, -1),
                resids,
                n_samples=simulator2.n_samples,
            )

            assert_arrays_compare(simulated_series1, simulated_series2, check_same=False)

            simulated_series3 = simulator2.simulate_ar_process(
                resids_lags.tolist(),
                resids_coefs.reshape(1, -1),
                resids,
                n_samples=simulator2.n_samples,
            )

            assert_arrays_compare(simulated_series2, simulated_series3, check_same=False)

    class TestFailingCases:
        @given(
            fitted_model=st.none() | st.integers() | st.floats() | st.text(),
            X_fitted=float_array,
            rng=st.none()
            | st.integers(min_value=MIN_INT, max_value=MAX_INT)
            | st.just(default_rng()),
        )
        def test_init_invalid_fitted_model(self, fitted_model, X_fitted, rng):
            """Test that AR model initialization fails with invalid fitted_model."""
            with pytest.raises(TypeError):
                TimeSeriesSimulator(fitted_model, X_fitted, rng)

        @given(
            fitted_model=ar_model_strategy(),
            X_fitted=float_array,
            rng=st.none()
            | st.integers(min_value=MIN_INT, max_value=MAX_INT)
            | st.just(default_rng()),
            resids_lags=st.none() | float_array | st.text(),
            resids_coefs=float_array,
            resids=float_array,
        )
        def test_simulate_ar_process_invalid_resids_lags(
            self,
            fitted_model,
            X_fitted,
            rng,
            resids_lags,
            resids_coefs,
            resids,
        ):
            """Test that AR model simulation fails with invalid resids_lags."""
            simulator = TimeSeriesSimulator(fitted_model, X_fitted, rng)
            with pytest.raises((ValueError, TypeError)):
                simulator.simulate_ar_process(
                    resids_lags,
                    resids_coefs.reshape(1, -1),
                    resids,
                    n_samples=simulator.n_samples,
                )

        @given(
            fitted_model=ar_model_strategy(),
            X_fitted=st.none() | integer_array | st.text(),
            rng=st.none()
            | st.integers(min_value=MIN_INT, max_value=MAX_INT)
            | st.just(default_rng()),
        )
        def test_init_invalid_X_fitted(self, fitted_model, X_fitted, rng):
            """Test that AR model initialization fails with invalid X_fitted."""
            with pytest.raises(TypeError):
                TimeSeriesSimulator(fitted_model, X_fitted, rng)


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
)
class TestARIMAModel:
    class TestPassingCases:
        @settings(suppress_health_check=(HealthCheck.too_slow,))
        @given(
            fitted_model=arima_model_strategy(),
            X_fitted=float_array,
            rng=st.none()
            | st.integers(min_value=MIN_INT, max_value=MAX_INT)
            | st.just(default_rng()),
        )
        def test_init_valid(self, fitted_model, X_fitted, rng):
            """Test that ARIMA model initialization works with valid inputs."""
            TimeSeriesSimulator(fitted_model, X_fitted, rng)

        @settings(suppress_health_check=(HealthCheck.too_slow,))
        @given(
            fitted_model=arima_model_strategy(),
            X_fitted=float_array,
            rng=st.none()
            | st.integers(min_value=MIN_INT, max_value=MAX_INT)
            | st.just(default_rng()),
        )
        def test_fitted_model_valid(self, fitted_model, X_fitted, rng):
            """Test that ARIMA model fitted_model property getter and setter work correctly."""
            simulator = TimeSeriesSimulator(fitted_model, X_fitted, rng)
            assert simulator.fitted_model == fitted_model

        @settings(suppress_health_check=(HealthCheck.too_slow,))
        @given(
            fitted_model=arima_model_strategy(),
            X_fitted=float_array,
            rng=st.none()
            | st.integers(min_value=MIN_INT, max_value=MAX_INT)
            | st.just(default_rng()),
        )
        def test_X_fitted_valid(self, fitted_model, X_fitted, rng):
            """Test that ARIMA model X_fitted property getter and setter work correctly."""
            simulator = TimeSeriesSimulator(fitted_model, X_fitted, rng)
            assert np.allclose(simulator.X_fitted, X_fitted)

        @settings(suppress_health_check=(HealthCheck.too_slow,))
        @given(
            fitted_model=arima_model_strategy(),
            X_fitted=float_array,
            rng=st.none()
            | st.integers(min_value=MIN_INT, max_value=MAX_INT)
            | st.just(default_rng()),
        )
        def test_rng_valid(self, fitted_model, X_fitted, rng):
            """Test that ARIMA model rng property getter and setter work correctly."""
            simulator = TimeSeriesSimulator(fitted_model, X_fitted, rng)
            assert isinstance(simulator.rng, Generator)

        @settings(
            suppress_health_check=(HealthCheck.too_slow,),
            max_examples=10,
            deadline=None,
        )
        @given(
            fitted_model=arima_model_strategy(),
            X_fitted=float_array,
            rng=st.none()
            | st.integers(min_value=MIN_INT, max_value=MAX_INT)
            | st.just(default_rng()),
        )
        def test_simulate_non_ar_process_valid(self, fitted_model, X_fitted, rng):
            """Test that ARIMA model simulation works with valid inputs."""
            simulator = TimeSeriesSimulator(fitted_model, X_fitted, rng)
            simulator.simulate_non_ar_process(n_samples=simulator.n_samples)

        @settings(
            suppress_health_check=(HealthCheck.too_slow,),
            max_examples=10,
            deadline=None,
        )
        @given(fitted_model=arima_model_strategy(), X_fitted=float_array)
        def test_simulate_non_ar_same_rng(self, fitted_model, X_fitted):
            """Test that ARIMA model simulation gives same results with same rng."""
            rng_seed = 12345

            rng1 = np.random.default_rng(rng_seed)
            simulator1 = TimeSeriesSimulator(fitted_model, X_fitted, rng1)
            simulated_series1 = simulator1.simulate_non_ar_process(n_samples=simulator1.n_samples)

            rng2 = np.random.default_rng(rng_seed)
            simulator2 = TimeSeriesSimulator(fitted_model, X_fitted, rng2)
            simulated_series2 = simulator2.simulate_non_ar_process(n_samples=simulator2.n_samples)

            assert_arrays_compare(simulated_series1, simulated_series2)

    class TestFailingCases:
        @given(
            fitted_model=st.none() | st.integers() | st.floats() | st.text(),
            X_fitted=float_array,
            rng=st.none()
            | st.integers(min_value=MIN_INT, max_value=MAX_INT)
            | st.just(default_rng()),
        )
        def test_init_invalid_fitted_model(self, fitted_model, X_fitted, rng):
            """Test that ARIMA model initialization fails with invalid fitted_model."""
            with pytest.raises(TypeError):
                TimeSeriesSimulator(fitted_model, X_fitted, rng)

        @given(
            fitted_model=arima_model_strategy(),
            X_fitted=st.none() | integer_array | st.text(),
            rng=st.none()
            | st.integers(min_value=MIN_INT, max_value=MAX_INT)
            | st.just(default_rng()),
        )
        def test_init_invalid_X_fitted(self, fitted_model, X_fitted, rng):
            """Test that ARIMA model initialization fails with invalid X_fitted."""
            if not isinstance(X_fitted, np.ndarray):
                with pytest.raises(TypeError):
                    TimeSeriesSimulator(fitted_model, X_fitted, rng)


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
)
class TestSARIMAModel:
    class TestPassingCases:
        @settings(suppress_health_check=(HealthCheck.too_slow,))
        @given(
            fitted_model=sarima_model_strategy(),
            X_fitted=float_array,
            rng=st.none()
            | st.integers(min_value=MIN_INT, max_value=MAX_INT)
            | st.just(default_rng()),
        )
        def test_init_valid(self, fitted_model, X_fitted, rng):
            """Test that SARIMA model initialization works with valid inputs."""
            TimeSeriesSimulator(fitted_model, X_fitted, rng)

        @settings(suppress_health_check=(HealthCheck.too_slow,))
        @given(
            fitted_model=sarima_model_strategy(),
            X_fitted=float_array,
            rng=st.none()
            | st.integers(min_value=MIN_INT, max_value=MAX_INT)
            | st.just(default_rng()),
        )
        def test_fitted_model_valid(self, fitted_model, X_fitted, rng):
            """Test that SARIMA model fitted_model property getter and setter work correctly."""
            simulator = TimeSeriesSimulator(fitted_model, X_fitted, rng)
            assert simulator.fitted_model == fitted_model

        @settings(suppress_health_check=(HealthCheck.too_slow,))
        @given(
            fitted_model=sarima_model_strategy(),
            X_fitted=float_array,
            rng=st.none()
            | st.integers(min_value=MIN_INT, max_value=MAX_INT)
            | st.just(default_rng()),
        )
        def test_X_fitted_valid(self, fitted_model, X_fitted, rng):
            """Test that SARIMA model X_fitted property getter and setter work correctly."""
            simulator = TimeSeriesSimulator(fitted_model, X_fitted, rng)
            assert np.allclose(simulator.X_fitted, X_fitted)

        @settings(suppress_health_check=(HealthCheck.too_slow,))
        @given(
            fitted_model=sarima_model_strategy(),
            X_fitted=float_array,
            rng=st.none()
            | st.integers(min_value=MIN_INT, max_value=MAX_INT)
            | st.just(default_rng()),
        )
        def test_rng_valid(self, fitted_model, X_fitted, rng):
            """Test that SARIMA model rng property getter and setter work correctly."""
            simulator = TimeSeriesSimulator(fitted_model, X_fitted, rng)
            assert isinstance(simulator.rng, Generator)

        @settings(
            suppress_health_check=(HealthCheck.too_slow,),
            max_examples=10,
            deadline=None,
        )
        @given(
            fitted_model=sarima_model_strategy(),
            X_fitted=float_array,
            rng=st.none()
            | st.integers(min_value=MIN_INT, max_value=MAX_INT)
            | st.just(default_rng()),
        )
        def test_simulate_non_ar_process_valid(self, fitted_model, X_fitted, rng):
            """Test that SARIMA model simulation works with valid inputs."""
            simulator = TimeSeriesSimulator(fitted_model, X_fitted, rng)
            simulator.simulate_non_ar_process(n_samples=simulator.n_samples)

        @settings(
            suppress_health_check=(HealthCheck.too_slow,),
            max_examples=10,
            deadline=None,
        )
        @given(fitted_model=sarima_model_strategy(), X_fitted=float_array)
        def test_simulate_non_ar_same_rng(self, fitted_model, X_fitted):
            """Test that SARIMA model simulation gives same results with same rng."""
            rng_seed = 12345

            rng1 = np.random.default_rng(rng_seed)
            simulator1 = TimeSeriesSimulator(fitted_model, X_fitted, rng1)
            simulated_series1 = simulator1.simulate_non_ar_process(n_samples=simulator1.n_samples)

            rng2 = np.random.default_rng(rng_seed)
            simulator2 = TimeSeriesSimulator(fitted_model, X_fitted, rng2)
            simulated_series2 = simulator2.simulate_non_ar_process(n_samples=simulator2.n_samples)

            assert_arrays_compare(simulated_series1, simulated_series2)

    class TestFailingCases:
        @given(
            fitted_model=st.none() | st.integers() | st.floats() | st.text(),
            X_fitted=float_array,
            rng=st.none()
            | st.integers(min_value=MIN_INT, max_value=MAX_INT)
            | st.just(default_rng()),
        )
        def test_init_invalid_fitted_model(self, fitted_model, X_fitted, rng):
            """Test that SARIMA model initialization fails with invalid fitted_model."""
            with pytest.raises(TypeError):
                TimeSeriesSimulator(fitted_model, X_fitted, rng)


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
)
class TestVARModel:
    class TestPassingCases:
        @given(
            fitted_model=var_model_strategy(),
            X_fitted=float_array.map(lambda x: np.column_stack([x, x])),
            rng=st.none()
            | st.integers(min_value=MIN_INT, max_value=MAX_INT)
            | st.just(default_rng()),
        )
        def test_init_valid(self, fitted_model, X_fitted, rng):
            """Test that VAR model initialization works with valid inputs."""
            TimeSeriesSimulator(fitted_model, X_fitted, rng)

        @given(
            fitted_model=var_model_strategy(),
            X_fitted=float_array.map(lambda x: np.column_stack([x, x])),
            rng=st.none()
            | st.integers(min_value=MIN_INT, max_value=MAX_INT)
            | st.just(default_rng()),
        )
        def test_fitted_model_valid(self, fitted_model, X_fitted, rng):
            """Test that VAR model fitted_model property getter and setter work correctly."""
            simulator = TimeSeriesSimulator(fitted_model, X_fitted, rng)
            assert simulator.fitted_model == fitted_model

        @given(
            fitted_model=var_model_strategy(),
            X_fitted=float_array.map(lambda x: np.column_stack([x, x])),
            rng=st.none()
            | st.integers(min_value=MIN_INT, max_value=MAX_INT)
            | st.just(default_rng()),
        )
        def test_X_fitted_valid(self, fitted_model, X_fitted, rng):
            """Test that VAR model X_fitted property getter and setter work correctly."""
            simulator = TimeSeriesSimulator(fitted_model, X_fitted, rng)
            assert np.allclose(simulator.X_fitted, X_fitted)

        @given(
            fitted_model=var_model_strategy(),
            X_fitted=float_array.map(lambda x: np.column_stack([x, x])),
            rng=st.none()
            | st.integers(min_value=MIN_INT, max_value=MAX_INT)
            | st.just(default_rng()),
        )
        def test_rng_valid(self, fitted_model, X_fitted, rng):
            """Test that VAR model rng property getter and setter work correctly."""
            simulator = TimeSeriesSimulator(fitted_model, X_fitted, rng)
            assert isinstance(simulator.rng, Generator)

        @settings(
            suppress_health_check=(HealthCheck.too_slow,),
            max_examples=10,
            deadline=None,
        )
        @given(
            fitted_model=var_model_strategy(),
            X_fitted=float_array.map(lambda x: np.column_stack([x, x])),
            rng=st.none()
            | st.integers(min_value=MIN_INT, max_value=MAX_INT)
            | st.just(default_rng()),
        )
        def test_simulate_non_ar_process_valid(self, fitted_model, X_fitted, rng):
            """Test that VAR model simulation works with valid inputs."""
            simulator = TimeSeriesSimulator(fitted_model, X_fitted, rng)
            simulator.simulate_non_ar_process(n_samples=simulator.n_samples)

        @given(
            fitted_model=var_model_strategy(),
            X_fitted=float_array.map(lambda x: np.column_stack([x, x])),
        )
        def test_simulate_non_ar_same_rng(self, fitted_model, X_fitted):
            """Test that SARIMA model simulation gives same results with same rng."""
            rng_seed = 12345

            rng1 = np.random.default_rng(rng_seed)
            simulator1 = TimeSeriesSimulator(fitted_model, X_fitted, rng1)
            simulated_series1 = simulator1.simulate_non_ar_process(n_samples=simulator1.n_samples)

            rng2 = np.random.default_rng(rng_seed)
            simulator2 = TimeSeriesSimulator(fitted_model, X_fitted, rng2)
            simulated_series2 = simulator2.simulate_non_ar_process(n_samples=simulator2.n_samples)

            assert_arrays_compare(simulated_series1, simulated_series2)

        @given(
            fitted_model=var_model_strategy(),
            X_fitted=float_array.map(lambda x: np.column_stack([x, x])),
        )
        def test_simulate_non_ar_different_rng(self, fitted_model, X_fitted):
            """Test that SARIMA model simulation gives same results with same rng."""
            rng = np.random.default_rng()
            simulator1 = TimeSeriesSimulator(fitted_model, X_fitted, rng)
            simulated_series1 = simulator1.simulate_non_ar_process(n_samples=simulator1.n_samples)

            simulator2 = TimeSeriesSimulator(fitted_model, X_fitted, rng)
            simulated_series2 = simulator2.simulate_non_ar_process(n_samples=simulator2.n_samples)

            assert_arrays_compare(simulated_series1, simulated_series2, check_same=False)

            simulated_series3 = simulator2.simulate_non_ar_process(n_samples=simulator2.n_samples)

            assert_arrays_compare(simulated_series2, simulated_series3, check_same=False)

    class TestFailingCases:
        @given(
            fitted_model=st.none() | st.integers() | st.floats() | st.text(),
            X_fitted=float_array.map(lambda x: np.column_stack([x, x])),
            rng=st.none()
            | st.integers(min_value=MIN_INT, max_value=MAX_INT)
            | st.just(default_rng()),
        )
        def test_init_invalid_fitted_model(self, fitted_model, X_fitted, rng):
            """Test that VAR model initialization fails with invalid fitted_model."""
            with pytest.raises(TypeError):
                TimeSeriesSimulator(fitted_model, X_fitted, rng)


@pytest.mark.skipif(
    not _check_soft_dependencies("arch", severity="none"),
    reason="skip test if required soft dependency not available",
)
class TestARCHModel:
    class TestPassingCases:
        @settings(suppress_health_check=(HealthCheck.too_slow,))
        @given(
            fitted_model=arch_model_strategy(),
            X_fitted=float_array,
            rng=st.none()
            | st.integers(min_value=MIN_INT, max_value=MAX_INT)
            | st.just(default_rng()),
        )
        def test_init_valid(self, fitted_model, X_fitted, rng):
            """Test that ARCH model initialization works with valid inputs."""
            TimeSeriesSimulator(fitted_model, X_fitted, rng)

        @settings(suppress_health_check=(HealthCheck.too_slow,))
        @given(
            fitted_model=arch_model_strategy(),
            X_fitted=float_array,
            rng=st.none()
            | st.integers(min_value=MIN_INT, max_value=MAX_INT)
            | st.just(default_rng()),
        )
        def test_fitted_model_valid(self, fitted_model, X_fitted, rng):
            """Test that ARCH model fitted_model property getter and setter work correctly."""
            simulator = TimeSeriesSimulator(fitted_model, X_fitted, rng)
            assert simulator.fitted_model == fitted_model

        @settings(suppress_health_check=(HealthCheck.too_slow,))
        @given(
            fitted_model=arch_model_strategy(),
            X_fitted=float_array,
            rng=st.none()
            | st.integers(min_value=MIN_INT, max_value=MAX_INT)
            | st.just(default_rng()),
        )
        def test_X_fitted_valid(self, fitted_model, X_fitted, rng):
            """Test that ARCH model X_fitted property getter and setter work correctly."""
            simulator = TimeSeriesSimulator(fitted_model, X_fitted, rng)
            assert np.allclose(simulator.X_fitted, X_fitted)

        @settings(suppress_health_check=(HealthCheck.too_slow,))
        @given(
            fitted_model=arch_model_strategy(),
            X_fitted=float_array,
            rng=st.none()
            | st.integers(min_value=MIN_INT, max_value=MAX_INT)
            | st.just(default_rng()),
        )
        def test_rng_valid(self, fitted_model, X_fitted, rng):
            """Test that ARCH model rng property getter and setter work correctly."""
            simulator = TimeSeriesSimulator(fitted_model, X_fitted, rng)
            assert isinstance(simulator.rng, Generator)

        @settings(
            suppress_health_check=(HealthCheck.too_slow,),
            max_examples=10,
            deadline=None,
        )
        @given(
            fitted_model=arch_model_strategy(),
            X_fitted=float_array,
            rng=st.none()
            | st.integers(min_value=MIN_INT, max_value=MAX_INT)
            | st.just(default_rng()),
        )
        def test_simulate_non_ar_process_valid(self, fitted_model, X_fitted, rng):
            """Test that ARCH model simulation works with valid inputs."""
            simulator = TimeSeriesSimulator(fitted_model, X_fitted, rng)
            simulator.simulate_non_ar_process(n_samples=simulator.n_samples)

        @settings(
            suppress_health_check=(HealthCheck.too_slow,),
            max_examples=10,
            deadline=None,
        )
        @pytest.mark.skipif(
            SKIP_ARCH_REPRODUCIBILITY_TEST,
            reason="Installed arch v7.x has API inconsistencies affecting reproducibility of simulate().",
        )
        @given(fitted_model=arch_model_strategy(), X_fitted=float_array)
        def test_simulate_non_ar_same_rng(self, fitted_model, X_fitted):
            """Test that ARCH model simulation gives same results with same rng."""
            rng_seed = 12345

            rng1 = np.random.default_rng(rng_seed)
            simulator1 = TimeSeriesSimulator(fitted_model, X_fitted, rng1)
            simulated_series1 = simulator1.simulate_non_ar_process(n_samples=simulator1.n_samples)

            rng2 = np.random.default_rng(rng_seed)
            simulator2 = TimeSeriesSimulator(fitted_model, X_fitted, rng2)
            simulated_series2 = simulator2.simulate_non_ar_process(n_samples=simulator2.n_samples)

            assert_arrays_compare(simulated_series1, simulated_series2)

    class TestFailingCases:
        @settings(suppress_health_check=(HealthCheck.too_slow,))
        @given(
            fitted_model=st.none() | st.integers() | st.floats() | st.text(),
            X_fitted=float_array,
            rng=st.none()
            | st.integers(min_value=MIN_INT, max_value=MAX_INT)
            | st.just(default_rng()),
        )
        def test_init_invalid_fitted_model(self, fitted_model, X_fitted, rng):
            """Test that ARCH model initialization fails with invalid fitted_model."""
            with pytest.raises(TypeError):
                TimeSeriesSimulator(fitted_model, X_fitted, rng)

        @given(
            fitted_model=arch_model_strategy(),
            X_fitted=st.none() | integer_array | st.text(),
            rng=st.none()
            | st.integers(min_value=MIN_INT, max_value=MAX_INT)
            | st.just(default_rng()),
        )
        # Adjusting settings to allow more time and limit the number of examples
        @settings(
            deadline=2000,
            max_examples=10,
            suppress_health_check=[HealthCheck.too_slow],
        )
        def test_init_invalid_X_fitted(self, fitted_model, X_fitted, rng):
            """Test that ARCH model initialization fails with invalid X_fitted."""
            with pytest.raises(TypeError):
                TimeSeriesSimulator(fitted_model, X_fitted, rng)

        @settings(suppress_health_check=(HealthCheck.too_slow,))
        @given(
            fitted_model=arch_model_strategy(),
            X_fitted=float_array,
            rng=st.floats() | st.text(),
        )
        def test_init_invalid_rng(self, fitted_model, X_fitted, rng):
            """Test that ARCH model initialization fails with invalid rng."""
            with pytest.raises(TypeError):
                TimeSeriesSimulator(fitted_model, X_fitted, rng)
