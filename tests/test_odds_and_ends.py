from hypothesis import given, settings, strategies as st
import pytest
import numpy as np
from numba import njit
from numba.core.errors import TypingError

from utils.odds_and_ends import choice_with_p, time_series_split, is_callable, is_numba_compiled, normalize_array


# create a callable function for testing
@njit
def identity(x):
    return x


class TestIsCallable:
    class TestPassingCases:
        def test_callable(self):
            """
            Test if a callable function is identified correctly.
            """
            assert is_callable(identity)

    class TestFailingCases:
        def test_non_callable(self):
            """
            Test if a non-callable object is identified correctly.
            """
            assert not is_callable(123)


class TestIsNumbaCompiled:
    class TestPassingCases:
        def test_numba_compiled(self):
            """
            Test if a numba compiled function is identified correctly.
            """
            assert is_numba_compiled(identity)

    class TestFailingCases:
        def test_not_numba_compiled(self):
            """
            Test if a non-numba compiled function is identified correctly.
            """
            assert not is_numba_compiled(is_callable)


class TestNormalizeArray:
    class TestPassingCases:
        @settings(deadline=None)
        @given(st.lists(st.floats(min_value=1, max_value=100), min_size=1))
        def test_positive_values(self, array):
            """
            Test normalization of an array with positive values.
            """
            array = np.array(array)
            result = normalize_array(array)
            assert np.isclose(np.sum(result), 1.0)

        @settings(deadline=None)
        @given(st.floats(min_value=1, max_value=100))
        def test_single_value(self, value):
            """
            Test normalization of an array with a single positive value.
            """
            array = np.array([value])
            result = normalize_array(array)
            assert np.isclose(np.sum(result), 1.0)

        @settings(deadline=None)
        @given(st.lists(st.floats(min_value=-100, max_value=100), min_size=1))
        def test_zero_sum(self, array):
            """
            Test normalization of an array where the sum of values is zero.
            This should return an array with equal values.
            """
            array = np.array(array)
            array_sum = np.sum(array)
            if array_sum == 0:
                result = normalize_array(array)
                assert np.allclose(result, 1.0/len(array))

        @settings(deadline=None)
        @given(st.lists(st.floats(min_value=-100, max_value=100), min_size=1))
        def test_negative_values(self, array):
            """
            Test normalization of an array with negative values.
            The sum of the normalized array should be 1.0.
            """
            array = np.array(array)
            result = normalize_array(array)
            assert np.isclose(np.sum(result), 1.0)

    class TestFailingCases:
        def test_non_array_input(self):
            """
            Test normalization with a non-array input (a single number), which should raise a TypingError.
            """
            with pytest.raises(TypingError):
                normalize_array(5)

        def test_none_input(self):
            """
            Test normalization with a None input, which should raise a TypingError.
            """
            with pytest.raises(TypingError):
                normalize_array(None)


class TestChoiceWithP:
    class TestPassingCases:
        @settings(deadline=None)
        @given(st.lists(st.floats(min_value=0, max_value=1), min_size=2, max_size=100))
        def test_valid_weights(self, weights):
            """
            Test that the function works for valid weights and returns an array of indices
            with the same length as the input weights.
            """
            weights = np.array(weights)
            indices = choice_with_p(weights)
            assert len(indices) == len(weights)
            assert np.all(indices >= 0)
            assert np.all(indices < len(weights))

        def test_equal_weights(self):
            """
            Test that the function works for equal weights and returns an array of indices
            with the same length as the input weights.
            """
            weights = np.array([0.25, 0.25, 0.25, 0.25])
            indices = choice_with_p(weights)
            assert len(indices) == len(weights)
            assert np.all(indices >= 0)
            assert np.all(indices < 4)

    class TestFailingCases:
        def test_negative_weights(self):
            """
            Test that the function raises a ValueError if any of the input weights are negative.
            """
            weights = np.array([-0.25, -0.25, -0.25, -0.25])
            with pytest.raises(ValueError):
                choice_with_p(weights)

        def test_weights_not_1d(self):
            """
            Test that the function raises a ValueError if the input weights are not a 1-dimensional array.
            """
            weights = np.array([[0.25, 0.25], [0.25, 0.25]])
            with pytest.raises(ValueError):
                choice_with_p(weights)


class TestTimeSeriesSplit:
    class TestPassingCases:
        @given(st.lists(st.floats(allow_infinity=False, allow_nan=False), min_size=2, max_size=100), st.floats(min_value=0.1, max_value=0.9))
        def test_valid_input(self, X, test_ratio):
            X = np.array(X)
            X_train, X_test = time_series_split(X, test_ratio)
            assert len(X_train) == int(len(X) * (1 - test_ratio))
            assert len(X_test) == len(X) - len(X_train)
            assert np.all(X_train == X[:len(X_train)])
            assert np.all(X_test == X[len(X_train):])

        def test_zero_ratio(self):
            X = np.array([1, 2, 3, 4, 5])
            X_train, X_test = time_series_split(X, 0)
            assert len(X_train) == 5
            assert len(X_test) == 0

        def test_full_ratio(self):
            X = np.array([1, 2, 3, 4, 5])
            X_train, X_test = time_series_split(X, 1)
            assert len(X_train) == 0
            assert len(X_test) == 5

    class TestFailingCases:
        def test_negative_ratio(self):
            X = np.array([1, 2, 3, 4, 5])
            with pytest.raises(ValueError):
                time_series_split(X, -0.5)

        def test_large_ratio(self):
            X = np.array([1, 2, 3, 4, 5])
            with pytest.raises(ValueError):
                time_series_split(X, 1.5)
