from hypothesis import given, assume, strategies as st, settings
import numpy as np
import pytest
from numba import njit
from src.bootstrap_numba import generate_random_indices, _prepare_block_weights, _prepare_tapered_weights

MIN_FLOAT_VALUE = -1e100  # np.finfo(np.float64).min
MAX_FLOAT_VALUE = 1e100  # np.finfo(np.float64).max


# create a callable function for testing
@njit
def identity(x):
    return x


class TestGenerateIndicesRandom:
    class TestPassingCases:
        def test_positive_samples(self):
            """
            Test if indices are generated correctly for a positive number of samples.
            """
            result = generate_random_indices(10, 42)
            assert result.size == 10
            assert result.dtype == np.int

        def test_single_sample(self):
            """
            Test if indices are generated correctly for a single sample.
            """
            result = generate_random_indices(1, 42)
            assert result.size == 1
            assert result.dtype == np.int

    class TestFailingCases:
        def test_zero_samples(self):
            """
            Test with zero samples, should raise a ValueError.
            """
            with pytest.raises(ValueError):
                generate_random_indices(0, 42)

        def test_negative_samples(self):
            """
            Test with negative number of samples, should raise a ValueError.
            """
            with pytest.raises(ValueError):
                generate_random_indices(-1, 42)


class TestPrepareBlockWeights:
    class TestPassingCases:

        @settings(deadline=None)
        @given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=MIN_FLOAT_VALUE, max_value=MAX_FLOAT_VALUE), min_size=1))
        def test_callable_weights(self, X):
            """
            Test with a callable function as block_weights. The function should be correctly identified, 
            executed and the result should be a normalized array.
            """
            def weights_callable(X):
                X_sum = np.sum(X)
                if X_sum == 0.0:
                    return np.full_like(X, 1.0/len(X))
                else:
                    return abs(np.array(X) / (X_sum + 1e-3))

            result = _prepare_block_weights(weights_callable, np.array(X))
            assert np.isclose(np.sum(result), 1.0, atol=1e-3)

        @settings(deadline=None)
        @given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1))
        def test_empty_weights(self, X):
            """
            Test with an empty block_weights array. The function should generate a new weights array with equal
            weights for all elements.
            """
            result = _prepare_block_weights(np.array([]), np.array(X))
            assert np.allclose(result, np.full(len(X), 1.0 / len(X)))

        @settings(deadline=None)
        @given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=0), min_size=1))
        def test_weights_same_as_input(self, X):
            """
            Test with a block_weights array with the same elements as the input array. The function should return
            a normalized weights array.
            """
            assume(np.sum(X) != 0)  # Exclude zero-sum cases
            result = _prepare_block_weights(np.array(X), np.array(X))
            assert np.allclose(
                result, (np.array(X) / np.sum(X)).reshape(-1, 1))

        @settings(deadline=None)
        @given(st.floats(allow_nan=False, allow_infinity=False, min_value=1e-1))
        def test_single_element_input(self, X_element):
            """
            Test with a single-element input array and a single-element weights array. The weights array should be
            normalized to 1.
            """
            result = _prepare_block_weights(
                np.array([X_element]), np.array([1]))
            assert result == 1.0

        @settings(deadline=None)
        @given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=1e-1, max_value=MAX_FLOAT_VALUE), min_size=100, max_size=100))
        def test_large_input(self, X):
            """
            Test with a large input array and weights array. The function should correctly normalize the weights array.
            """
            result = _prepare_block_weights(np.array(X), np.array(X))
            assert np.isclose(np.sum(result), 1.0)

        @settings(deadline=None)
        @given(st.lists(st.floats(min_value=0, max_value=MAX_FLOAT_VALUE), min_size=10, max_size=10))
        def test_edge_weights(self, X):
            """
            Test with maximum and minimum possible weights. Should not raise any error and should return normalized weights.
            """
            assume(np.sum(X) != 0)  # Exclude zero-sum cases
            result = _prepare_block_weights(np.array(X), np.ones(10))
            assert np.isclose(np.sum(result), 1.0)

    class TestFailingCases:

        def test_callable_weights_invalid_output(self):
            """
            Test with a callable function as block_weights that returns invalid output.
            The function should raise a ValueError.
            """
            invalid_outputs = [
                lambda _: np.array([np.inf]),
                lambda _: np.array([-np.inf]),
                lambda _: np.array([np.nan]),
            ]

            for weights_callable in invalid_outputs:
                with pytest.raises(ValueError):
                    _prepare_block_weights(
                        weights_callable, np.array([1.0, 2.0, 3.0]))

        @settings(deadline=None)
        @given(st.lists(st.integers(), min_size=1), st.lists(st.integers(min_value=1), min_size=1))
        def test_incompatible_weights(self, X, block_weights):
            """
            Test with a block_weights array of a different size than the input array X. 
            Should raise a ValueError.
            """
            assume(len(X) != len(block_weights))  # Exclude equal length cases
            with pytest.raises(ValueError):
                _prepare_block_weights(np.array(block_weights), np.array(X))

        @settings(deadline=None)
        @given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1))
        def test_negative_weights(self, X):
            """
            Test with a block_weights array containing negative weights. The function should raise a ValueError
            as weights cannot be negative.
            """
            assume(any(x < 0 for x in X))  # Exclude non-negative cases
            with pytest.raises(ValueError):
                _prepare_block_weights(np.array(X), np.array(X))

        @settings(deadline=None)
        @given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1))
        def test_non_callable_non_array_weights(self, X):
            """
            Test with a block_weights value that is neither a callable function nor an array. The function should
            raise a TypeError.
            """
            with pytest.raises(TypeError):
                _prepare_block_weights(42, np.array(X))

        @settings(deadline=None)
        @given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1))
        def test_callable_weights_with_side_effects(self, X):
            """
            Test with a callable function as block_weights that has side effects (changes the input array). The function
            should raise a ValueError as the callable function should not have side effects.
            """
            def side_effect_fn(X):
                X *= 2
                return X

            with pytest.raises(ValueError):
                _prepare_block_weights(side_effect_fn, np.array(X))
