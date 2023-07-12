import numpy as np
import pytest
from hypothesis import given, strategies as st, assume
from typing import List
from utils.validate import validate_integers, validate_X_and_exog


MIN_INT_VALUE = np.iinfo(np.int64).min
MAX_INT_VALUE = np.iinfo(np.int64).max


# Test with a single positive integer
@given(st.integers(min_value=1, max_value=MAX_INT_VALUE))
def test_single_positive_integer(x: int):
    """Test that the function accepts a single positive integer."""
    validate_integers(x, positive=True)


# Test with a single non-positive integer and positive flag, expecting a TypeError
@given(st.integers(min_value=MIN_INT_VALUE, max_value=0))
def test_single_non_positive_integer(x: int):
    """Test that the function raises a TypeError when given a non-positive integer and positive=True."""
    with pytest.raises(TypeError, match="All integers must be positive."):
        validate_integers(x, positive=True)


# Test with a list of positive integers
@given(st.lists(st.integers(min_value=1, max_value=MAX_INT_VALUE), min_size=1))
def test_list_of_positive_integers(xs: List[int]):
    """Test that the function accepts a list of positive integers."""
    validate_integers(xs, positive=True)


# Test with a list of non-positive integers and positive flag, expecting a TypeError
@given(st.lists(st.integers(min_value=MIN_INT_VALUE, max_value=0)))
def test_list_of_non_positive_integers(xs: List[int]):
    """Test that the function raises a TypeError when given a list of non-positive integers and positive=True."""
    with pytest.raises(TypeError):
        validate_integers(xs, positive=True)


# Test with a 1D NumPy array of positive integers
@given(st.lists(st.integers(min_value=1, max_value=MAX_INT_VALUE), min_size=1).map(np.array))
def test_numpy_array_of_positive_integers(arr: np.ndarray):
    """Test that the function accepts a 1D NumPy array of positive integers."""
    validate_integers(arr, positive=True)


# Test with a 1D NumPy array of non-positive integers and positive flag, expecting a TypeError
@given(st.lists(st.integers(min_value=MIN_INT_VALUE, max_value=0), min_size=1).map(np.array))
def test_numpy_array_of_non_positive_integers(arr: np.ndarray):
    """Test that the function raises a TypeError when given a 1D NumPy array of non-positive integers and positive=True."""
    with pytest.raises(TypeError, match="All integers in the array must be positive."):
        validate_integers(arr, positive=True)


# Test with a 2D NumPy array of integers, expecting a TypeError
@given(st.lists(st.integers(), min_size=1).map(lambda x: np.array([x, x])))
def test_numpy_2d_array(arr: np.ndarray):
    """Test that the function raises a TypeError when given a 2D NumPy array."""
    with pytest.raises(TypeError, match="Array must be 1D and contain only integers."):
        validate_integers(arr)


# Test with a 1D NumPy array of floats, expecting a TypeError
@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1).map(np.array))
def test_numpy_array_of_floats(arr: np.ndarray):
    """Test that the function raises a TypeError when given a 1D NumPy array of floats."""
    with pytest.raises(TypeError, match="Array must be 1D and contain only integers."):
        validate_integers(arr)


# Test with a mix of valid positive input types
@given(st.integers(min_value=1, max_value=MAX_INT_VALUE), st.lists(st.integers(min_value=1, max_value=MAX_INT_VALUE), min_size=1), st.lists(st.integers(min_value=1, max_value=MAX_INT_VALUE), min_size=1).map(np.array))
def test_mixed_valid_positive_inputs(x: int, xs: List[int], arr: np.ndarray):
    """Test that the functionaccepts a mix of valid positive input types."""
    validate_integers(x, xs, arr, positive=True)


# Test with an invalid input type (float), expecting a TypeError
@given(st.floats(allow_nan=False, allow_infinity=False))
def test_invalid_input_type(x: float):
    """Test that the function raises a TypeError when given an invalid input type."""
    with pytest.raises(TypeError, match="Input must be an integer, a list of integers, or a 1D array of integers."):
        validate_integers(x)


# Test with a list containing an invalid element type (float), expecting a TypeError
@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1))
def test_list_with_invalid_element_type(xs: List[float]):
    """Test that the function raises a TypeError when given a list containing an invalid element type."""
    with pytest.raises(TypeError, match="All elements in the list must be integers."):
        validate_integers(xs)


# Test with an edge case: maximum integer value
def test_maximum_integer():
    """Test that the function accepts the maximum integer value."""
    max_int = np.iinfo(np.int64).max
    validate_integers(max_int)


# Test with an edge case: minimum integer value
def test_minimum_integer():
    """Test that the function accepts the minimum integer value."""
    min_int = np.iinfo(np.int64).min
    validate_integers(min_int)


# Hypothesis strategy for generating 1D NumPy arrays
array_1d = st.lists(
    st.floats(allow_nan=False, allow_infinity=False), min_size=2).map(np.array)


# Hypothesis strategy for generating 2D NumPy arrays
array_2d = st.integers(min_value=2, max_value=10).flatmap(
    lambda n: st.builds(
        np.array,
        st.lists(
            st.lists(
                st.floats(allow_nan=False, allow_infinity=False),
                min_size=n, max_size=n
            ),
            min_size=2
        )
    )
)

# Test with a 1D X array and no exog array


@given(array_1d)
def test_1d_X_no_exog(X: np.ndarray):
    """Test that the function accepts a 1D X array and no exog array."""
    validate_X_and_exog(X, None)


# Test with a 1D X array and a 1D exog array
@given(array_1d)
def test_1d_X_1d_exog(X: np.ndarray):
    """Test that the function accepts a 1D X array and a 1D exog array."""
    validate_X_and_exog(X, X)


# Test with a 1D X array and a 2D exog array
@given(array_1d)
def test_1d_X_2d_exog(X: np.ndarray):
    """Test that the function accepts a 1D X array and a 2D exog array."""
    validate_X_and_exog(X, X[:, np.newaxis])


# Test with a 2D X array and no exog array, when model_is_var=True
@given(array_2d)
def test_2d_X_no_exog_var_model(X: np.ndarray):
    """Test that the function accepts a 2D X array and no exog array when model_is_var=True."""
    validate_X_and_exog(X, None, model_is_var=True)


# Test with a 2D X array and a 1D exog array, when model_is_var=True
@given(array_2d)
def test_2d_X_1d_exog_var_model(exog: np.ndarray):
    """Test that the function accepts a 2D X array and a 1D exog array when model_is_var=True."""
    validate_X_and_exog(exog, exog[:, 0], model_is_var=True)


# Test with a 2D X array and a 2D exog array, when model_is_var=True
@given(array_2d)
def test_2d_X_2d_exog_var_model(exog: np.ndarray):
    """Test that the function accepts a 2D X array and a 2D exog array when model_is_var=True."""
    validate_X_and_exog(exog, exog, model_is_var=True)


# Test with a 1D X array and no exog array, when model_is_arch=True
@given(array_1d)
def test_1d_X_no_exog_arch_model(X: np.ndarray):
    """Test that the function accepts a 1D X array and no exog array when model_is_arch=True."""
    validate_X_and_exog(X, None, model_is_arch=True)


# Test with a 1D X array and a 1D exog array, when model_is_arch=True
@given(array_1d)
def test_1d_X_1d_exog_arch_model(exog: np.ndarray):
    """Test that the function accepts a 1D X array and a 1D exog array when model_is_arch=True."""
    validate_X_and_exog(exog, exog, model_is_arch=True)


# Test with a 1D X array and a 2D exog array, when model_is_arch=True
@given(array_1d)
def test_1d_X_2d_exog_arch_model(X: np.ndarray):
    """Test that the function accepts a 1D X array and a 2D exog array when model_is_arch=True."""
    validate_X_and_exog(X, X[:, np.newaxis], model_is_arch=True)


# Test that an error is raised if X is not 1D and model_is_var=False
@given(array_2d)
def test_error_X_not_1d(X: np.ndarray):
    """Test that a ValueError is raised if X is not 1D when model_is_var=False."""
    with pytest.raises(ValueError):
        validate_X_and_exog(X, None, model_is_var=False)


# Test that an error is raised if X is not 2D or has less than 2 columns when model_is_var=True
@given(array_1d)
def test_error_X_not_2d_or_less_than_2_columns(X: np.ndarray):
    """Test that a ValueError is raised if X is not 2D or has less than 2 columns when model_is_var=True."""
    with pytest.raises(ValueError):
        validate_X_and_exog(X, None, model_is_var=True)


# Test with a 2D X array with only 1 column when model_is_var=True
@given(array_2d)
def test_error_X_2d_with_only_1_column(X: np.ndarray):
    """Test that a ValueError is raised if X is 2D with only 1 column when model_is_var=True."""
    with pytest.raises(ValueError):
        validate_X_and_exog(X[:, 0], None, model_is_var=True)
