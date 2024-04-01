import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tsbootstrap.utils.validate import (
    validate_block_indices,
    validate_blocks,
    validate_integers,
    validate_weights,
    validate_X_and_y,
)

MIN_INT_VALUE = np.iinfo(np.int64).min
MAX_INT_VALUE = np.iinfo(np.int64).max


class TestValidateIntegers:
    """Test the validate_integers function."""

    class TestPassingCases:
        """Test cases where validate_integers should work correctly."""

        @given(st.integers(min_value=1, max_value=MAX_INT_VALUE))
        def test_single_positive_integer(self, x: int):
            """Test that the function accepts a single positive integer."""
            validate_integers(x, min_value=1)

        @given(
            st.lists(
                st.integers(min_value=1, max_value=MAX_INT_VALUE), min_size=1
            )
        )
        def test_list_of_positive_integers(self, xs: list):
            """Test that the function accepts a list of positive integers."""
            validate_integers(xs, min_value=1)

        @given(
            st.lists(
                st.integers(min_value=1, max_value=MAX_INT_VALUE), min_size=1
            ).map(np.array)
        )
        def test_numpy_array_of_positive_integers(self, arr: np.ndarray):
            """Test that the function accepts a 1D NumPy array of positive integers."""
            validate_integers(arr, min_value=1)

        @given(
            st.integers(min_value=1, max_value=MAX_INT_VALUE),
            st.lists(
                st.integers(min_value=1, max_value=MAX_INT_VALUE), min_size=1
            ),
            st.lists(
                st.integers(min_value=1, max_value=MAX_INT_VALUE), min_size=1
            ).map(np.array),
        )
        def test_mixed_valid_positive_inputs(
            self, x: int, xs: list, arr: np.ndarray
        ):
            """Test that the functionaccepts a mix of valid positive input types."""
            validate_integers(x, xs, arr, min_value=1)

        def test_maximum_integer(
            self,
        ):
            """Test that the function accepts the maximum integer value."""
            max_int = MAX_INT_VALUE
            validate_integers(max_int)

        def test_minimum_integer(
            self,
        ):
            """Test that the function accepts the minimum integer value."""
            min_int = MIN_INT_VALUE
            validate_integers(min_int)

        @given(st.integers(min_value=MIN_INT_VALUE, max_value=0))
        def test_single_non_positive_integer(self, x: int):
            """Test that the function accepts a single non-positive integer when positive=False."""
            validate_integers(x, min_value=MIN_INT_VALUE)

        @given(
            st.lists(
                st.integers(min_value=MIN_INT_VALUE, max_value=0), min_size=1
            )
        )
        def test_list_of_non_positive_integers(self, xs: list):
            """Test that the function accepts a list of non-positive integers when positive=False."""
            validate_integers(xs)

        @given(
            st.lists(
                st.integers(min_value=MIN_INT_VALUE, max_value=0), min_size=1
            ).map(np.array)
        )
        def test_numpy_array_of_non_positive_integers(self, arr: np.ndarray):
            """Test that the function accepts a 1D NumPy array of non-positive integers when positive=False."""
            validate_integers(arr)

        @given(
            st.integers(min_value=MIN_INT_VALUE, max_value=MAX_INT_VALUE),
            st.lists(
                st.integers(min_value=MIN_INT_VALUE, max_value=MAX_INT_VALUE),
                min_size=1,
            ),
            st.lists(
                st.integers(min_value=MIN_INT_VALUE, max_value=MAX_INT_VALUE),
                min_size=1,
            ).map(np.array),
        )
        def test_mixed_valid_inputs(self, x: int, xs: list, arr: np.ndarray):
            """Test that the function accepts a mix of valid input types, including non-positive integers."""
            validate_integers(x, xs, arr)

    class TestFailingCases:
        """Test cases where validate_integers should fail."""

        @given(st.integers(min_value=MIN_INT_VALUE, max_value=0))
        def test_single_non_positive_integer(self, x: int):
            """Test that the function raises a TypeError when given a non-positive integer and positive=True."""
            with pytest.raises(ValueError, match="Integer must be at least 1"):
                validate_integers(x, min_value=1)

        @given(
            st.lists(
                st.integers(min_value=MIN_INT_VALUE, max_value=0), min_size=1
            )
        )
        def test_list_of_non_positive_integers(self, xs: list):
            """Test that the function raises a TypeError when given a list of non-positive integers and positive=True."""
            with pytest.raises(ValueError):
                validate_integers(xs, min_value=1)

        @given(
            st.lists(
                st.integers(min_value=MIN_INT_VALUE, max_value=0), min_size=1
            ).map(np.array)
        )
        def test_numpy_array_of_non_positive_integers(self, arr: np.ndarray):
            """Test that the function raises a TypeError when given a 1D NumPy array of non-positive integers and positive=True."""
            with pytest.raises(
                ValueError,
                match="All integers in the array must be at least 1.",
            ):
                validate_integers(arr, min_value=1)

        @given(
            st.lists(st.integers(), min_size=1).map(lambda x: np.array([x, x]))
        )
        def test_numpy_2d_array(self, arr: np.ndarray):
            """Test that the function raises a TypeError when given a 2D NumPy array."""
            with pytest.raises(
                TypeError, match="Array must be 1D and contain only integers."
            ):
                validate_integers(arr)

        @given(
            st.lists(
                st.floats(allow_nan=False, allow_infinity=False), min_size=1
            ).map(np.array)
        )
        def test_numpy_array_of_floats(self, arr: np.ndarray):
            """Test that the function raises a TypeError when given a 1D NumPy array of floats."""
            with pytest.raises(
                TypeError, match="Array must be 1D and contain only integers."
            ):
                validate_integers(arr)

        @given(st.floats(allow_nan=False, allow_infinity=False))
        def test_invalid_input_type(self, x: float):
            """Test that the function raises a TypeError when given an invalid input type."""
            with pytest.raises(
                TypeError,
                match="Input must be an integer, a list of integers, or a 1D array of integers.",
            ):
                validate_integers(x)

        @given(
            st.lists(
                st.floats(allow_nan=False, allow_infinity=False), min_size=1
            )
        )
        def test_list_with_invalid_element_type(self, xs: list):
            """Test that the function raises a TypeError when given a list containing an invalid element type."""
            with pytest.raises(
                TypeError, match="All elements in the list must be integers."
            ):
                validate_integers(xs)

        @given(
            st.lists(
                st.floats(allow_nan=False, allow_infinity=False), min_size=1
            ).map(np.array)
        )
        def test_numpy_array_with_invalid_element_type(self, arr: np.ndarray):
            """Test that the function raises a TypeError when given a list containing an invalid element type."""
            with pytest.raises(
                TypeError, match="Array must be 1D and contain only integers."
            ):
                validate_integers(arr)


# Hypothesis strategy for generating 1D NumPy arrays
array_1d = st.lists(
    st.floats(allow_nan=False, allow_infinity=False), min_size=2
).map(np.array)


# Hypothesis strategy for generating 2D NumPy arrays
array_2d = st.integers(min_value=2, max_value=10).flatmap(
    lambda n: st.builds(
        np.array,
        st.lists(
            st.lists(
                st.floats(allow_nan=False, allow_infinity=False),
                min_size=n,
                max_size=n,
            ),
            min_size=2,
        ),
    )
)


class TestValidateXAndY:
    """
    Test the validate_X_and_y function.
    """

    class TestPassingCases:
        """
        Test cases where validate_X_and_y should work correctly.
        """

        @given(array_1d)
        def test_1d_X_no_y(self, X: np.ndarray):
            """Test that the function accepts a 1D X array and no y array."""
            validate_X_and_y(X, None)

        @given(array_1d)
        def test_1d_X_1d_y(self, X: np.ndarray):
            """Test that the function accepts a 1D X array and a 1D y array."""
            validate_X_and_y(X, X)

        @given(array_1d)
        def test_1d_X_2d_y(self, X: np.ndarray):
            """Test that the function accepts a 1D X array and a 2D y array."""
            validate_X_and_y(X, X[:, np.newaxis])

        @given(array_2d)
        def test_2d_X_no_y_var_model(self, X: np.ndarray):
            """Test that the function accepts a 2D X array and no y array when model_is_var=True."""
            validate_X_and_y(X, None, model_is_var=True)

        @given(array_2d)
        def test_2d_X_1d_y_var_model(self, y: np.ndarray):
            """Test that the function accepts a 2D X array and a 1D y array when model_is_var=True."""
            validate_X_and_y(y, y[:, 0], model_is_var=True)

        @given(array_2d)
        def test_2d_X_2d_y_var_model(self, y: np.ndarray):
            """Test that the function accepts a 2D X array and a 2D y array when model_is_var=True."""
            validate_X_and_y(y, y, model_is_var=True)

        @given(array_1d)
        def test_1d_X_no_y_arch_model(self, X: np.ndarray):
            """Test that the function accepts a 1D X array and no y array when model_is_arch=True."""
            validate_X_and_y(X, None, model_is_arch=True)

        @given(array_1d)
        def test_1d_X_1d_y_arch_model(self, y: np.ndarray):
            """Test that the function accepts a 1D X array and a 1D y array when model_is_arch=True."""
            validate_X_and_y(y, y, model_is_arch=True)

        @given(array_1d)
        def test_1d_X_2d_y_arch_model(self, X: np.ndarray):
            """Test that the function accepts a 1D X array and a 2D y array when model_is_arch=True."""
            validate_X_and_y(X, X[:, np.newaxis], model_is_arch=True)

    class TestFailingCases:
        """
        Test cases where validate_X_and_y should fail.
        """

        @given(array_2d)
        def test_error_X_not_1d(self, X: np.ndarray):
            """Test that a ValueError is raised if X is not 1D when model_is_var=False."""
            with pytest.raises(ValueError):
                validate_X_and_y(X, None, model_is_var=False)

        @given(array_1d)
        def test_error_X_not_2d_or_less_than_2_columns(self, X: np.ndarray):
            """Test that a ValueError is raised if X is not 2D or has less than 2 columns when model_is_var=True."""
            with pytest.raises(ValueError):
                validate_X_and_y(X, None, model_is_var=True)

        @given(array_2d)
        def test_error_X_2d_with_only_1_column(self, X: np.ndarray):
            """Test that a ValueError is raised if X is 2D with only 1 column when model_is_var=True."""
            with pytest.raises(ValueError):
                validate_X_and_y(X[:, 0], None, model_is_var=True)


# Hypothesis strategy for generating valid block indices and corresponding input length
valid_block_indices_and_length = st.integers(
    min_value=2, max_value=100
).flatmap(
    lambda n: st.tuples(
        st.builds(
            list,
            st.lists(
                st.builds(
                    np.array,
                    st.lists(
                        st.integers(min_value=0, max_value=n - 1),
                        min_size=2,
                        max_size=n,
                    ),
                ),
                min_size=1,
                max_size=n,
            ),
        ),
        st.just(n),
    )
)

# Hypothesis strategy for generating invalid block indices
invalid_block_indices = st.lists(
    st.floats(allow_nan=False, allow_infinity=False), min_size=2
).map(np.array)


class TestValidateBlockIndices:
    """
    Test the validate_block_indices function.
    """

    class TestPassingCases:
        """
        Test cases where validate_block_indices should work correctly.
        """

        @given(valid_block_indices_and_length)
        def test_valid_block_indices(self, block_indices_and_length):
            """Test that the function accepts a valid block indices list."""
            block_indices, input_length = block_indices_and_length
            validate_block_indices(block_indices, input_length)

    class TestFailingCases:
        """
        Test cases where validate_block_indices should fail.
        """

        @given(invalid_block_indices, st.integers(min_value=2, max_value=100))
        def test_invalid_block_indices(self, block_indices, input_length: int):
            """Test that the function raises a TypeError for an invalid block indices list."""
            with pytest.raises(TypeError):
                validate_block_indices(block_indices, input_length)

        @given(st.integers(min_value=1, max_value=100))
        def test_empty_block_indices(self, input_length: int):
            """Test that the function raises a ValueError for an empty block indices list."""
            with pytest.raises(ValueError):
                validate_block_indices([], input_length)

        @given(valid_block_indices_and_length)
        def test_indices_beyond_input_length(self, block_indices_and_length):
            """Test that the function raises a ValueError for block indices beyond the range of X."""
            block_indices, input_length = block_indices_and_length
            # Make the first index out-of-range
            block_indices[0][0] = input_length
            with pytest.raises(ValueError):
                validate_block_indices(block_indices, input_length)

        @given(valid_block_indices_and_length)
        def test_2d_or_higher_ndarray(self, block_indices_and_length):
            """Test that the function raises a ValueError for 2D or higher ndarray in the block indices list."""
            block_indices, input_length = block_indices_and_length
            # Make the first ndarray 2D
            block_indices[0] = np.array([block_indices[0], block_indices[0]])
            with pytest.raises(ValueError):
                validate_block_indices(block_indices, input_length)

        @given(valid_block_indices_and_length)
        def test_noninteger_ndarray(self, block_indices_and_length):
            """Test that the function raises a ValueError for non-integer ndarray in the block indices list."""
            block_indices, input_length = block_indices_and_length
            # Make the first ndarray non-integer
            block_indices[0] = block_indices[0].astype(float)
            with pytest.raises(ValueError):
                validate_block_indices(block_indices, input_length)

        @given(valid_block_indices_and_length)
        def test_empty_ndarray(self, block_indices_and_length):
            """Test that the function raises a ValueError for an empty ndarray in the block indices list."""
            block_indices, input_length = block_indices_and_length
            # Make the first ndarray empty
            block_indices[0] = np.array([])
            with pytest.raises(ValueError):
                validate_block_indices(block_indices, input_length)


# Hypothesis strategy for generating valid blocks
valid_blocks = st.integers(min_value=1, max_value=10).flatmap(
    lambda n: st.lists(
        st.builds(
            np.array,
            st.lists(
                st.lists(
                    st.floats(allow_nan=False, allow_infinity=False),
                    min_size=n,
                    max_size=n,
                ),
                min_size=1,
            ),
        ),
        min_size=1,
    )
)


# Hypothesis strategy for generating blocks with different number of features
blocks_diff_features = st.tuples(
    st.builds(
        np.array,
        st.lists(
            st.lists(
                st.floats(allow_nan=False, allow_infinity=False),
                min_size=1,
                max_size=1,
            ),
            min_size=1,
        ),
    ),
    st.builds(
        np.array,
        st.lists(
            st.lists(
                st.floats(allow_nan=False, allow_infinity=False),
                min_size=2,
                max_size=2,
            ),
            min_size=1,
        ),
    ),
).map(list)


no_samples_blocks = st.lists(
    st.builds(np.array, st.just([])),  # This will always produce an empty list
    min_size=1,
)


one_dim_blocks = st.lists(
    st.builds(
        np.array,
        st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1),
    ),
    min_size=1,
)


class TestValidateBlocks:
    """
    Test the validate_blocks function.
    """

    class TestPassingCases:
        """
        Test cases where validate_blocks should work correctly.
        """

        @given(valid_blocks)
        def test_valid_blocks(self, blocks):
            """Test that the function accepts a valid blocks list."""
            validate_blocks(blocks)

    class TestFailingCases:
        """
        Test cases where validate_blocks should fail.
        """

        @given(st.integers())
        def test_nonlist_input(self, blocks: int):
            """Test that the function raises a TypeError for non-list input."""
            with pytest.raises(TypeError):
                validate_blocks(blocks)

        def test_empty_blocks(self):
            """Test that the function raises a ValueError for an empty blocks list."""
            with pytest.raises(ValueError):
                validate_blocks([])

        @given(st.lists(st.integers(), min_size=1))
        def test_nonndarray_blocks(self, blocks: list):
            """Test that the function raises a TypeError for list of non-ndarray blocks."""
            with pytest.raises(TypeError):
                validate_blocks(blocks)

        @given(
            st.lists(
                st.builds(np.array, st.lists(st.floats(), min_size=1)),
                min_size=1,
            )
        )
        def test_non2d_ndarray_blocks(self, blocks):
            """Test that the function raises a ValueError for list of non-2D ndarray blocks."""
            with pytest.raises(ValueError):
                validate_blocks(blocks)

        @given(no_samples_blocks)
        def test_no_timestamp_blocks(self, blocks):
            """Test that the function raises a ValueError for blocks with no timestamp."""
            with pytest.raises(ValueError):
                validate_blocks(blocks)

        @given(one_dim_blocks)
        def test_no_feature_blocks(self, blocks):
            """Test that the function raises a ValueError for blocks with no feature."""
            with pytest.raises(ValueError):
                validate_blocks(blocks)

        @given(blocks_diff_features)
        def test_diff_feature_blocks(self, blocks):
            """Test that the function raises a ValueError for blocks with different number of features."""
            with pytest.raises(ValueError):
                validate_blocks(blocks)

        def test_nan_blocks(self):
            """Test that the function raises a ValueError for blocks with NaN values."""
            # Manually create a block with a NaN value
            block_with_nan = [np.array([[1.0, 2.0], [np.nan, 4.0]])]
            with pytest.raises(ValueError):
                validate_blocks(block_with_nan)

        def test_infinite_blocks(self):
            """Test that the function raises a ValueError for blocks with infinite values."""
            block_with_inf = [np.array([[1.0, 2.0], [np.inf, 4.0]])]
            with pytest.raises(ValueError):
                validate_blocks(block_with_inf)
            block_with_neginf = [np.array([[1.0, 2.0], [-np.inf, 4.0]])]
            with pytest.raises(ValueError):
                validate_blocks(block_with_neginf)


# Hypothesis strategy for creating valid weights
valid_weights = st.lists(
    st.floats(
        min_value=1e-10, max_value=10, allow_nan=False, allow_infinity=False
    ),
    min_size=1,
).map(np.array)

# Hypothesis strategy for creating infinitesimally small but non-zero weights
small_weights = st.lists(
    st.floats(
        min_value=1e-10, max_value=1e-9, allow_nan=False, allow_infinity=False
    ),
    min_size=1,
).map(np.array)

# Hypothesis strategy for creating large but finite weights
large_weights = st.lists(
    st.floats(
        min_value=1e10, max_value=1e20, allow_nan=False, allow_infinity=False
    ),
    min_size=1,
).map(np.array)

# Hypothesis strategy for creating invalid weights
negative_weights = st.lists(st.floats(max_value=-0.1), min_size=1).map(
    np.array
)

negative_small_weights = st.lists(
    st.floats(
        min_value=-1e-6, max_value=0, allow_nan=False, allow_infinity=False
    ),
    min_size=1,
).map(np.array)

complex_weights = st.lists(
    st.complex_numbers(allow_nan=False, allow_infinity=False), min_size=1
).map(np.array)

zero_weights = st.just(np.array([0.0]))

one_dim_zero_weights = st.just(np.array([[0.0]]))


multi_dimensional_weights = st.lists(
    st.lists(
        st.floats(
            min_value=1e-10,
            max_value=10,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=2,
        max_size=2,  # two elements in the second axis
    ),
    min_size=1,  # At least one array in the first axis
).map(np.array)


class TestValidateWeights:
    """
    Test the validate_weights function.
    """

    class TestPassingCases:
        """
        Test cases where validate_weights should work correctly.
        """

        @given(valid_weights)
        def test_valid_weights(self, weights: np.ndarray):
            """Test that the function does not raise an error for valid weights."""
            validate_weights(weights)

        @given(small_weights)
        def test_small_weights(self, weights: np.ndarray):
            """Test that the function does not raise an error for small but non-zero weights."""
            validate_weights(weights)

        @given(large_weights)
        def test_large_weights(self, weights: np.ndarray):
            """Test that the function does not raise an error for large but finite weights."""
            validate_weights(weights)

    class TestFailingCases:
        """
        Test cases where validate_weights should fail.
        """

        def test_non_finite_weights(self):
            """Test that the function raises an error for weights containing non-finite values."""
            non_finite_weights = np.array([np.nan, np.inf, -np.inf])
            with pytest.raises(ValueError):
                validate_weights(non_finite_weights)

        @given(negative_weights)
        def test_negative_weights(self, weights: np.ndarray):
            """Test that the function raises an error for weights containing negative values."""
            with pytest.raises(ValueError):
                validate_weights(weights)

        @given(negative_small_weights)
        def test_negative_small_weights(self, weights: np.ndarray):
            """Test that the function raises an error for weights containing small negative values."""
            with pytest.raises(ValueError):
                validate_weights(weights)

        @given(complex_weights)
        def test_complex_weights(self, weights: np.ndarray):
            """Test that the function raises an error for weights containing complex numbers."""
            if any(np.iscomplex(weights)):
                with pytest.raises(ValueError):
                    validate_weights(weights)

        @given(zero_weights)
        def test_zero_weights(self, weights: np.ndarray):
            """Test that the function raises an error for weights that are all zero."""
            with pytest.raises(ValueError):
                validate_weights(weights)

        @given(one_dim_zero_weights)
        def test_one_dim_zero_weights(self, weights: np.ndarray):
            """Test that the function raises an error for weights that are a 2D array with a single column of zeros."""
            with pytest.raises(ValueError):
                validate_weights(weights)

        @given(multi_dimensional_weights)
        def test_multi_dimensional_weights(self, weights: np.ndarray):
            """Test that the function raises an error for weights that are a 2D array with more than one column."""
            with pytest.raises(ValueError):
                validate_weights(weights)
