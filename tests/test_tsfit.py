import math

import numpy as np
import pytest
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
from skbase.utils.dependencies import _check_soft_dependencies
from tsbootstrap import TSFit


# Test data strategy
def high_variance_floats():
    return lists(
        floats(
            min_value=1, max_value=50, allow_infinity=False, allow_nan=False
        ),
        min_size=50,
        max_size=50,
    ).filter(lambda generated_list: np.var(generated_list) > 0.01)


test_data = high_variance_floats()


# Test order strategy
ar_order_strategy = lists(
    integers(min_value=1, max_value=5), min_size=1, max_size=5
)
var_arch_order_strategy = integers(min_value=1, max_value=5)
arima_order_strategy = tuples(
    integers(min_value=1, max_value=5),
    integers(min_value=0, max_value=2),
    integers(min_value=0, max_value=2),
)


def valid_sarima_order():
    p = integers(min_value=1, max_value=5)
    d = integers(min_value=0, max_value=2)
    q = integers(min_value=0, max_value=2)
    s = integers(min_value=2, max_value=5)

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
            min_value=-50,
            max_value=50,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=2,
        max_size=2,
    ),
    min_size=50,
    max_size=50,
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


@pytest.mark.skipif(
    not _check_soft_dependencies(["arch", "statsmodels"], severity="none"),
    reason="skip test if required soft dependency not available",
)
class TestTSFit:
    class TestPassingCases:
        @given(order=ar_order_strategy, model_type=just("ar"))
        def test_init_ar(self, order, model_type):
            """Test TSFit initialization with valid inputs and model_type = 'ar'."""
            tsfit = TSFit(order, model_type)
            assert tsfit.order == sorted(order)
            assert tsfit.model_type == model_type.lower()

        @given(order=arima_order_strategy, model_type=just("arima"))
        def test_init_arima(self, order, model_type):
            """Test TSFit initialization with valid inputs and model_type = 'arima'."""
            tsfit = TSFit(order, model_type)
            assert tsfit.order == order
            assert tsfit.model_type == model_type.lower()

        @given(order=arima_order_strategy, model_type=just("sarima"))
        def test_init_sarima(self, order, model_type):
            """Test TSFit initialization with valid inputs and model_type = 'sarima'."""
            tsfit = TSFit(order, model_type)
            assert tsfit.order == order
            assert tsfit.model_type == model_type.lower()

        @given(
            order=var_arch_order_strategy,
            model_type=sampled_from(["var", "arch"]),
        )
        def test_init_var_arch(self, order, model_type):
            """Test TSFit initialization with valid inputs and model_type = 'var' or 'arch'."""
            tsfit = TSFit(order, model_type)
            assert tsfit.order == order
            assert tsfit.model_type == model_type.lower()

        @settings(deadline=None)
        @given(data=test_data, order=ar_order_strategy, model_type=just("ar"))
        def test_fit_valid_ar(self, data, order, model_type):
            """Test TSFit fit method with valid inputs and model_type = 'ar'."""
            from statsmodels.tsa.ar_model import AutoRegResultsWrapper

            order = list(np.unique(np.array(order)))
            data = np.array(data).reshape(-1, 1)
            tsfit = TSFit(order, model_type)
            fitted_model = tsfit.fit(data).model
            assert isinstance(fitted_model, AutoRegResultsWrapper)

        @pytest.mark.skip(reason="known LU decomposition issue, see #41")
        @settings(deadline=None)
        @given(
            data=test_data,
            order=arima_order_strategy,
            model_type=just("arima"),
        )
        def test_fit_valid_arima(self, data, order, model_type):
            """Test TSFit fit method with valid inputs and model_type = 'arima'."""
            from statsmodels.tsa.arima.model import ARIMAResultsWrapper

            data = np.array(data).reshape(-1, 1)
            tsfit = TSFit(order, model_type)
            var = np.var(data)
            if not math.isclose(var, 0, abs_tol=0.01):
                fitted_model = tsfit.fit(data).model
                assert isinstance(fitted_model, ARIMAResultsWrapper)

        @pytest.mark.skip(reason="known LU decomposition issue, see #41")
        @settings(deadline=None)
        @given(
            data=test_data,
            order=sarima_order_strategy,
            model_type=just("sarima"),
        )
        def test_fit_valid_sarima(self, data, order, model_type):
            """Test TSFit fit method with valid inputs and model_type = 'sarima'."""
            from statsmodels.tsa.statespace.sarimax import (
                SARIMAXResultsWrapper,
            )

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
        def test_fit_valid_var_arch(self, data, order, model_type):
            """Test TSFit fit method with valid inputs and model_type = 'var' or 'arch'."""
            from arch.univariate.base import ARCHModelResult
            from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

            tsfit = TSFit(order, model_type)
            data = np.array(data).reshape(-1, 1)
            var = np.var(data)
            is_data_var_zero = math.isclose(var, 0, abs_tol=0.01)
            if model_type == "var":
                data = np.hstack((data, data))
            if model_type == "var":
                if not is_data_var_zero:
                    try:
                        fitted_model = tsfit.fit(data).model
                        assert isinstance(fitted_model, VARResultsWrapper)
                    except ValueError as e:
                        if "x contains one or more constant columns" in str(e):
                            pass  # Ignore ValueError, as it's expected when the input contains one or more constant columns and trend == 'c'
                        else:
                            raise  # If it's a different ValueError, raise it again
            else:
                if not is_data_var_zero:
                    fitted_model = tsfit.fit(data).model
                    assert isinstance(fitted_model, ARCHModelResult)

        @settings(deadline=None)
        @given(
            data=test_data,
            order=ar_order_strategy,
            model_type=just("ar"),
            exog=exog_strategy,
        )
        def test_fit_valid_ar_with_exog(self, data, order, model_type, exog):
            """Test TSFit fit method with valid inputs and model_type = 'ar' and exog."""
            from statsmodels.tsa.ar_model import AutoRegResultsWrapper

            order = list(np.unique(np.array(order)))
            data = np.array(data).reshape(-1, 1)
            exog = np.array(exog)
            tsfit = TSFit(order, model_type)
            fitted_model = tsfit.fit(data, y=exog).model
            assert isinstance(fitted_model, AutoRegResultsWrapper)

        @pytest.mark.skip(reason="known LU decomposition issue, see #41")
        @settings(deadline=None)
        @given(
            data=test_data,
            order=arima_order_strategy,
            model_type=just("arima"),
            exog=exog_strategy,
        )
        def test_fit_valid_arima_with_exog(
            self, data, order, model_type, exog
        ):
            """Test TSFit fit method with valid inputs and model_type = 'arima' and exog."""
            from statsmodels.tsa.arima.model import ARIMAResultsWrapper

            data = np.array(data).reshape(-1, 1)
            exog = np.array(exog)
            tsfit = TSFit(order, model_type)
            var = np.var(data)
            if not math.isclose(var, 0, abs_tol=0.01):
                fitted_model = tsfit.fit(data, y=exog).model
                assert isinstance(fitted_model, ARIMAResultsWrapper)

        @pytest.mark.skip(reason="known LU decomposition issue, see #41")
        @settings(deadline=None)
        @given(
            data=test_data,
            order=sarima_order_strategy,
            model_type=just("sarima"),
            exog=exog_strategy,
        )
        def test_fit_valid_sarima_with_exog(
            self, data, order, model_type, exog
        ):
            """Test TSFit fit method with valid inputs and model_type = 'sarima' and exog."""
            from statsmodels.tsa.statespace.sarimax import (
                SARIMAXResultsWrapper,
            )

            data = np.array(data).reshape(-1, 1)
            exog = np.array(exog)
            tsfit = TSFit(order, model_type)
            var = np.var(data)
            if not math.isclose(var, 0, abs_tol=0.01):
                try:
                    fitted_model = tsfit.fit(data, y=exog).model
                    assert isinstance(fitted_model, SARIMAXResultsWrapper)
                except LinAlgError:
                    pass

        @settings(deadline=None)
        @given(
            data=test_data,
            order=var_arch_order_strategy,
            model_type=sampled_from(["var", "arch"]),
            exog=exog_strategy,
        )
        def test_fit_valid_var_arch_with_exog(
            self, data, order, model_type, exog
        ):
            """Test TSFit fit method with valid inputs and model_type = 'var' or 'arch' and exog."""
            from arch.univariate.base import ARCHModelResult
            from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

            tsfit = TSFit(order, model_type)
            data = np.array(data).reshape(-1, 1)
            exog = np.array(exog)
            var = np.var(data)
            var_exog = np.var(exog, axis=0)
            is_data_var_zero = math.isclose(var, 0, abs_tol=0.01)
            is_exog_var_zero = any(
                math.isclose(var_exog_i, 0, abs_tol=0.01)
                for var_exog_i in var_exog
            )
            if model_type == "var":
                data = np.hstack((data, data))
            if model_type == "var":
                if not is_data_var_zero and not is_exog_var_zero:
                    try:
                        fitted_model = tsfit.fit(data, y=exog).model
                        assert isinstance(fitted_model, VARResultsWrapper)
                    except ValueError as e:
                        if "x contains one or more constant columns" in str(e):
                            pass
                        else:
                            raise
            else:
                if not is_data_var_zero and not is_exog_var_zero:
                    fitted_model = tsfit.fit(data, y=exog).model
                    assert isinstance(fitted_model, ARCHModelResult)

        @settings(deadline=None)
        @given(
            data=test_data,
            order=ar_order_strategy,
            model_type=just("ar"),
            exog=exog_strategy,
        )
        def test_predict_valid_ar(self, data, order, model_type, exog):
            """Test TSFit predict method with valid inputs and model_type = 'ar'."""
            order = list(np.unique(np.array(order)))
            data = np.array(data).reshape(-1, 1)
            exog = np.array(exog)
            tsfit = TSFit(order, model_type)
            fitted_model = tsfit.fit(data, y=exog)
            predicted = fitted_model.predict(data, n_steps=5, y=exog[:5, :])
            assert isinstance(predicted, np.ndarray)
            assert predicted.shape == (5,)

        @pytest.mark.skip(reason="known LU decomposition issue, see #41")
        @settings(deadline=None)
        @given(
            data=test_data,
            order=arima_order_strategy,
            model_type=just("arima"),
            exog=exog_strategy,
        )
        def test_predict_valid_arima(self, data, order, model_type, exog):
            """Test TSFit predict method with valid inputs and model_type = 'arima'."""
            data = np.array(data).reshape(-1, 1)
            exog = np.array(exog)
            tsfit = TSFit(order, model_type)
            fitted_model = tsfit.fit(data, y=exog)
            predicted = fitted_model.predict(data, n_steps=5, y=exog[:5, :])
            assert isinstance(predicted, np.ndarray)
            assert predicted.shape == (5,)

        @pytest.mark.skip(reason="known LU decomposition issue, see #41")
        @settings(deadline=None)
        @given(
            data=test_data,
            order=sarima_order_strategy,
            model_type=just("sarima"),
            exog=exog_strategy,
        )
        def test_predict_valid_sarima(self, data, order, model_type, exog):
            """Test TSFit predict method with valid inputs and model_type = 'sarima'."""
            data = np.array(data).reshape(-1, 1)
            exog = np.array(exog)
            tsfit = TSFit(order, model_type)
            try:
                fitted_model = tsfit.fit(data, y=exog)
                predicted = fitted_model.predict(
                    data, n_steps=5, y=exog[:5, :]
                )
                assert isinstance(predicted, np.ndarray)
                assert predicted.shape == (5,)
            except LinAlgError:
                pass  # Ignore LinAlgError, as it's expected in some cases

        @settings(deadline=None)
        @given(
            data=test_data,
            order=var_arch_order_strategy,
            model_type=sampled_from(["var", "arch"]),
            exog=exog_strategy,
        )
        def test_predict_valid_var_arch(self, data, order, model_type, exog):
            """Test TSFit predict method with valid inputs and model_type = 'var' or 'arch'."""
            tsfit = TSFit(order, model_type)
            data = np.array(data).reshape(-1, 1)
            exog = np.array(exog)
            exog = exog[: len(data), :]
            var = np.var(data)
            var_exog = np.var(exog, axis=0)
            is_data_var_zero = math.isclose(var, 0, abs_tol=0.01)
            is_exog_var_zero = any(
                math.isclose(var_exog_i, 0, abs_tol=0.01)
                for var_exog_i in var_exog
            )
            if model_type == "var":
                data = np.hstack((data, data))
                if not is_data_var_zero and not is_exog_var_zero:
                    try:
                        fitted_model = tsfit.fit(data, y=exog)
                        predicted = fitted_model.predict(
                            data, n_steps=5, y=exog[:5, :]
                        )
                        assert isinstance(predicted, np.ndarray)
                        assert predicted.shape == (5, 2)
                    except ValueError as e:
                        if "x contains one or more constant columns" in str(e):
                            pass
                        else:
                            raise
            else:
                if not is_data_var_zero and not is_exog_var_zero:
                    fitted_model = tsfit.fit(data, y=exog)
                    predicted = fitted_model.predict(data, n_steps=5)
                    print(f"predicted.type: {type(predicted)}")
                    print(f"predicted: {predicted}")
                    assert isinstance(predicted, np.ndarray)
                    assert predicted.shape == (5,)

        @settings(deadline=None)
        @given(
            data=test_data,
            order=ar_order_strategy,
            model_type=just("ar"),
            exog=exog_strategy,
        )
        def test_get_residuals_valid_ar(self, data, order, model_type, exog):
            """Test TSFit get_residuals method with valid inputs and model_type = 'ar'."""
            order = list(np.unique(np.array(order)))
            data = np.array(data).reshape(-1, 1)
            exog = np.array(exog)
            tsfit = TSFit(order, model_type)
            fitted_model = tsfit.fit(data, y=exog)
            residuals = fitted_model.get_residuals()
            assert isinstance(residuals, np.ndarray)
            assert residuals.shape == (len(data), 1)

        @pytest.mark.skip(reason="known LU decomposition issue, see #41")
        @settings(deadline=None)
        @given(
            data=test_data,
            order=arima_order_strategy,
            model_type=just("arima"),
            exog=exog_strategy,
        )
        def test_get_residuals_valid_arima(
            self, data, order, model_type, exog
        ):
            """Test TSFit get_residuals method with valid inputs and model_type = 'arima'."""
            data = np.array(data).reshape(-1, 1)
            exog = np.array(exog)
            tsfit = TSFit(order, model_type)
            fitted_model = tsfit.fit(data, y=exog)
            residuals = fitted_model.get_residuals()
            assert isinstance(residuals, np.ndarray)
            assert residuals.shape == (len(data), 1)

        @pytest.mark.skip(reason="known LU decomposition issue, see #41")
        @settings(deadline=None)
        @given(
            data=test_data,
            order=sarima_order_strategy,
            model_type=just("sarima"),
            exog=exog_strategy,
        )
        def test_get_residuals_valid_sarima(
            self, data, order, model_type, exog
        ):
            """Test TSFit get_residuals method with valid inputs and model_type = 'sarima'."""
            data = np.array(data).reshape(-1, 1)
            exog = np.array(exog)
            tsfit = TSFit(order, model_type)
            fitted_model = tsfit.fit(data, y=exog)
            residuals = fitted_model.get_residuals()
            assert isinstance(residuals, np.ndarray)
            assert residuals.shape == (len(data), 1)

        @settings(deadline=None)
        @given(
            data=test_data,
            order=var_arch_order_strategy,
            model_type=sampled_from(["var", "arch"]),
            exog=exog_strategy,
        )
        def test_get_residuals_valid_var_arch(
            self, data, order, model_type, exog
        ):
            """Test TSFit get_residuals method with valid inputs and model_type = 'var' or 'arch'."""
            print(f"input order: {order}")
            tsfit = TSFit(order, model_type)
            data = np.array(data).reshape(-1, 1)
            exog = np.array(exog)
            var = np.var(data)
            var_exog = np.var(exog, axis=0)
            is_data_var_zero = math.isclose(var, 0, abs_tol=0.01)
            is_exog_var_zero = any(
                math.isclose(var_exog_i, 0, abs_tol=0.01)
                for var_exog_i in var_exog
            )
            if model_type == "var":
                data = np.hstack((data, data))
                if not is_data_var_zero and not is_exog_var_zero:
                    try:
                        fitted_model = tsfit.fit(data, y=exog)
                        residuals = fitted_model.get_residuals()
                        assert isinstance(residuals, np.ndarray)
                        assert residuals.shape == (len(data), 2)
                    except ValueError as e:
                        if "x contains one or more constant columns" in str(e):
                            pass
                        else:
                            raise
            else:
                if not is_data_var_zero and not is_exog_var_zero:
                    fitted_model = tsfit.fit(data, y=exog)
                    residuals = fitted_model.get_residuals()
                    assert isinstance(residuals, np.ndarray)
                    assert residuals.shape == (len(data), 1)

        @settings(deadline=None)
        @given(
            data=test_data,
            order=ar_order_strategy,
            model_type=just("ar"),
            exog=exog_strategy,
        )
        def test_get_fitted_X_valid_ar(self, data, order, model_type, exog):
            """Test TSFit get_fitted_X method with valid inputs and model_type = 'ar'."""
            order = list(np.unique(np.array(order)))
            data = np.array(data).reshape(-1, 1)
            exog = np.array(exog)
            tsfit = TSFit(order, model_type)
            fitted_model = tsfit.fit(data, y=exog)
            fitted_X = fitted_model.get_fitted_X()
            assert isinstance(fitted_X, np.ndarray)
            assert fitted_X.shape == (len(data), 1)

        @pytest.mark.skip(reason="known LU decomposition issue, see #41")
        @settings(deadline=None)
        @given(
            data=test_data,
            order=arima_order_strategy,
            model_type=just("arima"),
            exog=exog_strategy,
        )
        def test_get_fitted_X_valid_arima(self, data, order, model_type, exog):
            """Test TSFit get_fitted_X method with valid inputs and model_type = 'arima'."""
            data = np.array(data).reshape(-1, 1)
            exog = np.array(exog)
            tsfit = TSFit(order, model_type)
            fitted_model = tsfit.fit(data, y=exog)
            fitted_X = fitted_model.get_fitted_X()
            assert isinstance(fitted_X, np.ndarray)
            assert fitted_X.shape == (len(data), 1)

        @pytest.mark.skip(reason="known LU decomposition issue, see #41")
        @settings(deadline=None)
        @given(
            data=test_data,
            order=sarima_order_strategy,
            model_type=just("sarima"),
            exog=exog_strategy,
        )
        def test_get_fitted_X_valid_sarima(
            self, data, order, model_type, exog
        ):
            """Test TSFit get_fitted_X method with valid inputs and model_type = 'sarima'."""
            data = np.array(data).reshape(-1, 1)
            exog = np.array(exog)
            tsfit = TSFit(order, model_type)
            fitted_model = tsfit.fit(data, y=exog)
            fitted_X = fitted_model.get_fitted_X()
            assert isinstance(fitted_X, np.ndarray)
            assert fitted_X.shape == (len(data), 1)

        @settings(deadline=None)
        @given(
            data=test_data,
            order=var_arch_order_strategy,
            model_type=sampled_from(["var", "arch"]),
            exog=exog_strategy,
        )
        def test_get_fitted_X_valid_var_arch(
            self, data, order, model_type, exog
        ):
            """Test TSFit get_fitted_X method with valid inputs and model_type = 'var' or 'arch'."""
            tsfit = TSFit(order, model_type)
            data = np.array(data).reshape(-1, 1)
            exog = np.array(exog)
            var = np.var(data)
            var_exog = np.var(exog, axis=0)
            is_data_var_zero = math.isclose(var, 0, abs_tol=0.01)
            is_exog_var_zero = any(
                math.isclose(var_exog_i, 0, abs_tol=0.01)
                for var_exog_i in var_exog
            )
            if model_type == "var":
                data = np.hstack((data, data))
            if model_type == "var":
                if not is_data_var_zero and not is_exog_var_zero:
                    try:
                        fitted_model = tsfit.fit(data, y=exog)
                        fitted_X = fitted_model.get_fitted_X()
                        assert isinstance(fitted_X, np.ndarray)
                        assert fitted_X.shape == (len(data), 2)
                    except ValueError as e:
                        if "x contains one or more constant columns" in str(e):
                            pass
                        else:
                            raise
            else:
                if not is_data_var_zero and not is_exog_var_zero:
                    fitted_model = tsfit.fit(data, y=exog)
                    fitted_X = fitted_model.get_fitted_X()
                    assert isinstance(fitted_X, np.ndarray)
                    assert fitted_X.shape == (len(data),)

    class TestFailingCases:
        def test_tsfit_fit_invalid_data(self):
            """Test TSFit fit method with invalid data."""
            model = TSFit(model_type="ar", order=1)
            with pytest.raises(TypeError):
                model.fit([])  # Empty data

        def test_tsfit_predict_without_fit(self):
            """Test TSFit predict method without fitting."""
            model = TSFit(model_type="ar", order=1)
            with pytest.raises(ValueError):
                model.predict(np.array([1, 2, 3]), n_steps=5)

        def test_tsfit_accessor_methods_without_fit(self):
            """Test TSFit accessor methods without fitting."""
            model = TSFit(model_type="ar", order=1)
            with pytest.raises(AttributeError):
                model.get_residuals()
            with pytest.raises(AttributeError):
                model.get_fitted_X()

        def test_tsfit_fit_invalid_order(self):
            """Test TSFit fit method with invalid order."""
            with pytest.raises(ValueError):
                TSFit(model_type="ar", order=[])

        def test_tsfit_fit_invalid_model_type(self):
            """Test TSFit fit method with invalid model_type."""
            with pytest.raises(ValueError):
                TSFit(model_type="invalid_model", order=1)

        def test_tsfit_predict_invalid_n_steps(self):
            """Test TSFit predict method with invalid n_steps."""
            model = TSFit(model_type="var", order=1)
            model.fit(np.arange(20).reshape(10, 2))
            with pytest.raises(ValueError):
                model.predict(np.arange(6).reshape(3, 2), n_steps=-1)

        def test_tsfit_predict_invalid_exog(self):
            """Test TSFit predict method with invalid exog."""
            model = TSFit(model_type="ar", order=1)
            model.fit(np.arange(10))
            with pytest.raises(ValueError):
                model.predict(np.array([1, 2, 3]), y=[])

        def test_tsfit_fit_invalid_exog(self):
            """Test TSFit fit method with invalid exog."""
            model = TSFit(model_type="ar", order=1)
            with pytest.raises(TypeError):
                model.fit(np.array([1, 2, 3]), y=[])

        def test_tsfit_fit_invalid_data_type(self):
            """Test TSFit fit method with invalid data type."""
            model = TSFit(model_type="ar", order=1)
            with pytest.raises(TypeError):
                model.fit(1)

        def test_tsfit_fit_invalid_data_shape(self):
            """Test TSFit fit method with invalid data shape."""
            model = TSFit(model_type="ar", order=1)
            with pytest.raises(ValueError):
                model.fit(np.arange(10).reshape(-1, 1, 1))

        def test_tsfit_fit_invalid_data_shape_with_exog(self):
            """Test TSFit fit method with invalid data shape and exog."""
            model = TSFit(model_type="ar", order=1)
            with pytest.raises(ValueError):
                model.fit(
                    np.array([1, 2, 3]).reshape(-1, 1),
                    y=np.array([1, 2, 3]).reshape(-1, 1),
                )
