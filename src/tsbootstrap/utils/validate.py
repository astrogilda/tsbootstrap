"""Validate module."""

from collections.abc import Mapping
from numbers import Integral
from typing import Any, Optional, Tuple, get_args

import numpy as np
from numpy.random import Generator
from sklearn.utils import check_array

from tsbootstrap.utils.types import FittedModelTypes, RngTypes


def check_is_finite(input_array: np.ndarray, input_name: str) -> np.ndarray:
    """
    Check if all elements in the input NumPy array are finite.
    """
    if not np.isfinite(input_array).all():
        raise ValueError(
            f"The provided callable function or array '{input_name}' resulted in non-finite values. Please check your inputs."
        )
    return input_array


def check_are_nonnegative(input_array: np.ndarray, input_name: str) -> np.ndarray:
    """
    Check if all elements in the input NumPy array are nonnegative.
    """
    if np.any(input_array < 0):
        raise ValueError(
            f"The provided callable function '{input_name}' resulted in negative values. Please check your function."
        )
    return input_array


def check_are_real(input_array: np.ndarray, input_name: str) -> np.ndarray:
    """
    Check if all elements in the input NumPy array are real.
    """
    if np.any(np.iscomplex(input_array)):
        raise ValueError(
            f"The provided callable function '{input_name}' resulted in complex values. Please check your function."
        )
    return input_array


def check_is_not_all_zero(input_array: np.ndarray, input_name: str) -> np.ndarray:
    """
    Check if the input NumPy array is not all zeros.

    An empty array is not considered "all zero" for this check.
    """
    if input_array.size > 0 and np.all(input_array == 0):
        raise ValueError(
            f"The provided callable function '{input_name}' resulted in all zero values. Please check your function."
        )
    return input_array


def check_is_1d_or_2d_single_column(input_array: np.ndarray, input_name: str) -> np.ndarray:
    """
    Check if the input NumPy array is a 1D array or a 2D array with a single column.
    """
    if (input_array.ndim == 2 and input_array.shape[1] != 1) or input_array.ndim > 2:
        raise ValueError(
            f"The provided callable function '{input_name}' resulted in a 2D array with more than one column. Please check your function."
        )
    return input_array


def check_is_np_array(input_array: np.ndarray, input_name: str) -> np.ndarray:
    """
    Check if the input is a NumPy array.
    """
    if not isinstance(input_array, np.ndarray):
        raise TypeError(f"Input '{input_name}' must be a NumPy array.")
    return input_array


def check_are_2d_arrays(input_list, input_name: str):
    """
    Check if all NumPy arrays in the input list are 2D.
    """
    if not all(element.ndim == 2 for element in input_list):
        raise ValueError(f"Input '{input_name}' must be a list of 2D NumPy arrays.")
    return input_list


def check_have_at_least_one_element(input_list, input_name: str):
    """
    Check if all NumPy arrays in the input list have at least one element.
    """
    if not all(element.shape[0] > 0 for element in input_list):
        raise ValueError(
            f"Input '{input_name}' must be a list of 2D NumPy arrays with at least one element."
        )
    return input_list


def check_have_at_least_one_feature(input_list, input_name: str):
    """
    Check if all NumPy arrays in the input list have at least one feature.
    """
    if not all(element.shape[1] > 0 for element in input_list):
        raise ValueError(
            f"Input '{input_name}' must be a list of 2D NumPy arrays with at least one feature."
        )
    return input_list


def check_have_same_num_of_features(input_list, input_name: str):
    """
    Check if all NumPy arrays in the input list have the same number of features.
    """
    if not all(element.shape[1] == input_list[0].shape[1] for element in input_list):
        raise ValueError(
            f"Input '{input_name}' must be a list of 2D NumPy arrays with the same number of features."
        )
    return input_list


def check_are_finite(input_list, input_name: str):
    """
    Check if all elements in the NumPy arrays in the input list are finite.
    """
    if not all(np.all(np.isfinite(element)) for element in input_list):
        raise ValueError(
            f"Input '{input_name}' must be a list of 2D NumPy arrays with finite values."
        )
    return input_list


def check_is_list(input_list: list, input_name: str) -> list:
    """
    Check if the input is a list.
    """
    if not isinstance(input_list, list):
        raise TypeError(f"Input '{input_name}' must be a list.")
    return input_list


def check_is_nonempty(input_list: list, input_name: str) -> list:
    """
    Check if the input list is nonempty.
    """
    if len(input_list) == 0:
        raise ValueError(f"Input '{input_name}' must not be empty.")
    return input_list


def check_are_np_arrays(input_list, input_name: str):
    """
    Check if all elements in the input list are NumPy arrays.
    """
    if not all(isinstance(element, np.ndarray) for element in input_list):
        raise TypeError(f"Input '{input_name}' must be a list of NumPy arrays.")
    return input_list


def check_are_1d_integer_arrays(input_list, input_name: str):
    """
    Check if all NumPy arrays in the input list are 1D and contain integer values.
    """
    if not all(
        element.ndim == 1 and np.issubdtype(element.dtype, np.integer) for element in input_list
    ):
        raise ValueError(
            f"Input '{input_name}' must be a list of 1D NumPy arrays with integer values."
        )
    return input_list


def check_have_at_least_one_index(input_list, input_name: str):
    """
    Check if all NumPy arrays in the input list have at least one index.
    """
    if not all(element.size > 0 for element in input_list):
        raise ValueError(
            f"Input '{input_name}' must be a list of 1D NumPy arrays with at least one index."
        )
    return input_list


def check_indices_within_range(input_list, input_length: int, input_name: str):
    """
    Check if all indices in the NumPy arrays in the input list are within the range of the input length.
    """
    if not all(np.all(element < input_length) for element in input_list):
        raise ValueError(
            f"Input '{input_name}' must be a list of 1D NumPy arrays with indices within the range of X."
        )
    return input_list


def check_array_type(X: np.ndarray) -> np.ndarray:
    """
    Check if the given array is a NumPy array of floats.
    """
    if not isinstance(X, np.ndarray) or X.dtype.kind not in "iuf":
        raise TypeError("X must be a NumPy array of floats.")
    return X


def check_array_size(X: np.ndarray) -> np.ndarray:
    """
    Check if the given array contains at least two elements.
    """
    if X.size < 2:
        raise ValueError("X must contain at least two elements.")
    return X


def check_array_shape(X: np.ndarray, model_is_var: bool, allow_multi_column: bool) -> np.ndarray:
    """
    Check if the given array meets the required shape constraints.

    Parameters
    ----------
    X : np.ndarray
        The input array to be checked.
    model_is_var : bool
        Flag indicating if the model is a VAR (Vector Autoregression) model.
    allow_multi_column : bool
        Flag indicating if multiple columns are allowed in the array.

    Returns
    -------
    np.ndarray
        The original array if it meets the constraints.

    Raises
    ------
    ValueError
        If the array does not meet the required shape constraints.

    Examples
    --------
    >>> check_array_shape(np.array([[1, 2], [3, 4]]), True, True)
    array([[1, 2], [3, 4]])

    >>> check_array_shape(np.array([1, 2, 3]), False, False)
    array([1, 2, 3])
    """
    if model_is_var:
        if X.shape[1] < 2:
            raise ValueError("X must be 2-dimensional with at least 2 columns")
        return X

    if allow_multi_column:
        return X

    if X.ndim > 2 or (X.ndim == 2 and X.shape[1] != 1):
        raise ValueError("X must be 1-dimensional or 2-dimensional with a single column")

    return X


def add_newaxis_if_needed(X: np.ndarray) -> np.ndarray:
    """
    Add a new axis to the given array if it's needed.
    """
    if X.ndim == 1:  # and model_is_var:
        X = X[:, np.newaxis]
    return X


def validate_single_integer(
    value: Integral,
    min_value: Optional[Integral] = None,
    max_value: Optional[Integral] = None,
) -> None:
    """Validate a single integer value against an optional minimum value."""
    if not isinstance(value, Integral):
        raise TypeError(f"Input must be an integer. Got {value}.")
    if min_value is not None and value < min_value:
        raise ValueError(f"Integer must be at least {min_value}. Got {value}.")
    if max_value is not None and value > max_value:
        raise ValueError(f"Integer must be at most {max_value}. Got {value}.")


def validate_list_of_integers(
    value,
    min_value: Optional[Integral] = None,
    max_value: Optional[Integral] = None,
) -> None:
    """Validate a list of integer values against an optional minimum value."""
    if not value:
        raise TypeError(f"list must not be empty. Got {value}.")

    if not all(isinstance(x, Integral) for x in value):
        raise TypeError(f"All elements in the list must be integers. Got {value}.")

    if min_value is not None and any(x < min_value for x in value):
        raise ValueError(f"All integers in the list must be at least {min_value}. Got {value}.")

    if max_value is not None and any(x > max_value for x in value):
        raise ValueError(f"All integers in the list must be at most {max_value}. Got {value}.")


def validate_integer_array(
    value: np.ndarray,
    min_value: Optional[Integral] = None,
    max_value: Optional[Integral] = None,
) -> None:
    """Validate a 1D numpy array of integers against an optional minimum value."""
    if value.size == 0:
        raise TypeError(f"Array must not be empty. Got {value}.")

    if value.ndim != 1 or value.dtype.kind not in "iu":
        raise TypeError(f"Array must be 1D and contain only integers. Got {value}.")

    if min_value is not None and any(value < min_value):
        raise ValueError(f"All integers in the array must be at least {min_value}. Got {value}.")

    if max_value is not None and any(value > max_value):
        raise ValueError(f"All integers in the array must be at most {max_value}. Got {value}.")


def validate_integers(
    *values,
    min_value: Optional[Integral] = None,
    max_value: Optional[Integral] = None,
) -> None:
    """
    Validates that all input values are integers and optionally, above a minimum value.

    Each value can be an integer, a list of integers, or a 1D numpy array of integers.
    If min_value is provided, all integers must be greater than or equal to min_value.

    Parameters
    ----------
    *values : Union[Integral, list[Integral], np.ndarray]
        One or more values to validate.
    min_value : Integral, optional
        If provided, all integers must be greater than or equal to min_value.
    max_value : Integral, optional
        If provided, all integers must be less than or equal to max_value.

    Raises
    ------
    TypeError
        If a value is not an integer, list of integers, or 1D array of integers,
        or if any integer is less than min_value or greater than max_value.
    """
    for value in values:
        if isinstance(value, Integral):
            validate_single_integer(value, min_value, max_value)
        elif isinstance(value, list):
            validate_list_of_integers(value, min_value, max_value)
        elif isinstance(value, np.ndarray):
            validate_integer_array(value, min_value, max_value)
        else:
            raise TypeError(
                f"Input must be an integer, a list of integers, or a 1D array of integers. Got {value}."
            )


def validate_X(
    X: np.ndarray,
    model_is_var: bool,
    allow_multi_column: Optional[bool] = None,
) -> np.ndarray:
    """
    Validate the input array X based on the given model type.

    Parameters
    ----------
    X : np.ndarray
        The input array to be validated. It must be a NumPy array of floats (i, u, or f type).
    model_is_var : bool
        A flag to determine whether the model is of VAR (Vector Autoregression) type.
        If True, the function will validate it as a VAR array.
        If False, the function will validate it as a non-VAR array.
    allow_multi_column : bool, optional
        A flag to determine whether the array is allowed to have more than one column.
        If not specified, it defaults to the value of `model_is_var`.

    Returns
    -------
    np.ndarray
        A validated array.

    Raises
    ------
    TypeError
        If X is not a NumPy array or its data type is not float.
    ValueError
        If X contains fewer than two elements, or does not meet the dimensionality requirements.
    """
    if allow_multi_column is None:
        allow_multi_column = model_is_var

    X = check_array_type(X)
    X = check_array_size(X)
    X = add_newaxis_if_needed(X)
    # print(X.shape)
    X = check_array(
        X,
        ensure_2d=True,  # model_is_var or allow_multi_column, #True,
        force_all_finite=True,
        dtype=[np.float64, np.float32],
    )
    X = check_array_shape(X, model_is_var, allow_multi_column)

    return X


def validate_exog(exog: np.ndarray) -> np.ndarray:
    """
    Validate the exogenous variable array `exog`, ensuring its dimensionality and dtype.

    Parameters
    ----------
    exog : np.ndarray
        The exogenous variable array to be validated. Must be a NumPy array of floats.

    Returns
    -------
    np.ndarray
        A validated exogenous variable array.

    Raises
    ------
    TypeError
        If `exog` is not a NumPy array or its data type is not float.
    ValueError
        If `exog` contains fewer than two elements.
    """
    return validate_X(exog, model_is_var=False, allow_multi_column=True)


def validate_X_and_y(
    X: np.ndarray,
    y: Optional[np.ndarray],
    model_is_var: bool = False,
    model_is_arch: bool = False,
):
    """
    Validate and reshape input data and exogenous variables.

    This function uses :func:`validate_X` and :func:`validate_exog` to perform detailed validation.

    Parameters
    ----------
    X : np.ndarray
        The input array to be validated.
    y : Optional[np.ndarray]
        The exogenous variable array to be validated. Can be None.
    model_is_var : bool, optional
        A flag to determine if the model is of VAR type. Default is False.
    model_is_arch : bool, optional
        A flag to determine if the model is of ARCH type. Default is False.

    Returns
    -------
    Tuple[np.ndarray, Optional[np.ndarray]]
        A tuple containing the validated X array and optionally the validated exog array.

    See Also
    --------
    validate_X : Function for validating the input array X.
    validate_exog : Function for validating the exogenous variable array.
    """
    X = validate_X(X, model_is_var)

    if y is not None:
        y = validate_exog(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of rows in y must be equal to the number of rows in X.")
    # Ensure contiguous arrays for ARCH models
    if model_is_arch:
        X = np.ascontiguousarray(X)
        if y is not None:
            y = np.ascontiguousarray(y)

    return X, y


def validate_block_indices(block_indices: list[np.ndarray], input_length: int) -> None:
    """
    Validate the input block indices. Each block index must be a 1D NumPy array with at least one index and all indices must be within the range of X.

    Parameters
    ----------
    block_indices : list[np.ndarray]
        The input block indices.
    input_length : int
        The length of the input data.

    Raises
    ------
    TypeError
        If block_indices is not a list or if it contains non-NumPy arrays.
    ValueError
        If block_indices is empty or if it contains NumPy arrays with non-integer values,
        or if it contains NumPy arrays with no indices, or if it contains NumPy arrays
        with indices outside the range of X.
    """
    block_indices = check_is_list(block_indices, "block_indices")
    block_indices = check_is_nonempty(block_indices, "block_indices")
    block_indices = check_are_np_arrays(block_indices, "block_indices")
    block_indices = check_are_1d_integer_arrays(block_indices, "block_indices")
    # Allow individual blocks to be empty, as long as they are 1D integer arrays.
    # block_indices = check_have_at_least_one_index(
    #     block_indices, "block_indices"
    # )
    block_indices = check_indices_within_range(block_indices, input_length, "block_indices")


def validate_blocks(blocks: list[np.ndarray]) -> None:
    """
    Validate the input blocks. Each block must be a 2D NumPy array with at least one element.

    Parameters
    ----------
    blocks : list[np.ndarray]
        The input blocks.

    Raises
    ------
    TypeError
        If blocks is not a list or if it contains non-NumPy arrays.
    ValueError
        If blocks is empty or if it contains NumPy arrays with non-finite values,
        or if it contains NumPy arrays with no elements, or if it contains NumPy arrays
        with no features, or if it contains NumPy arrays with different number of features.
    """
    blocks = check_is_list(blocks, "blocks")
    blocks = check_is_nonempty(blocks, "blocks")
    blocks = check_are_np_arrays(blocks, "blocks")
    blocks = check_are_2d_arrays(blocks, "blocks")
    blocks = check_have_at_least_one_element(blocks, "blocks")
    blocks = check_have_at_least_one_feature(blocks, "blocks")
    blocks = check_have_same_num_of_features(blocks, "blocks")
    blocks = check_are_finite(blocks, "blocks")


def validate_weights(weights: np.ndarray) -> None:
    """
    Validate the input weights. Each weight must be a non-negative finite value.

    Parameters
    ----------
    weights : np.ndarray
        The input weights.

    Raises
    ------
    TypeError
        If weights is not a NumPy array.
    ValueError
        If weights contains any non-finite values, or if it contains any negative values,
        or if it contains any complex values, or if it contains all zeros,
        or if it is a 2D array with more than one column.
    """
    weights = check_is_np_array(weights, "weights")
    weights = check_is_finite(weights, "weights")
    weights = check_are_nonnegative(weights, "weights")
    weights = check_are_real(weights, "weights")
    weights = check_is_not_all_zero(weights, "weights")
    weights = check_is_1d_or_2d_single_column(weights, "weights")


def validate_fitted_model(fitted_model) -> None:
    """
    Validate the input fitted model. It must be an instance of a fitted model class.

    Parameters
    ----------
    fitted_model : FittedModelTypes
        The input fitted model.

    Raises
    ------
    TypeError
        If fitted_model is not an instance of a fitted model class.
    """
    valid_types = FittedModelTypes()
    if not isinstance(fitted_model, valid_types):
        valid_names = ", ".join([t.__name__ for t in valid_types])
        raise TypeError(
            f"fitted_model must be an instance of {valid_names}. Got {type(fitted_model).__name__} instead."
        )


def validate_literal_type(input_value: str, literal_type: Any) -> None:
    """
    Validate the type of `input_value` against a Literal type or dictionary keys.

    Parameters
    ----------
    input_value : str
        The value to validate.
    literal_type : type, or list
        if type: Literal type or dictionary against which to validate the `input_value`.
        if list: list of valid values against which to validate the `input_value`.

    Raises
    ------
    TypeError
        If `input_value` is not a string.
    ValueError
        If `input_value` is not among the valid types in `literal_type` or dictionary keys.

    Examples
    --------
    >>> validate_literal_type("a", Literal["a", "b", "c"])
    >>> validate_literal_type("x", {"x": 1, "y": 2})
    >>> validate_literal_type("z", Literal["a", "b", "c"])
    ValueError: Invalid input_value 'z'. Expected one of 'a', 'b', 'c'.
    >>> validate_literal_type("z", {"x": 1, "y": 2})
    ValueError: Invalid input_value 'z'. Expected one of 'x', 'y'.
    """
    if not isinstance(input_value, str):
        raise TypeError(f"input_value must be a string. Got {type(input_value).__name__} instead.")

    if isinstance(literal_type, Mapping):
        valid_types = [str(key) for key in literal_type]
    elif isinstance(literal_type, list):
        valid_types = literal_type
    else:
        valid_types = [str(arg) for arg in get_args(literal_type)]

    if input_value.lower() not in valid_types:
        raise ValueError(
            f"Invalid input_value '{input_value}'. Expected one of {', '.join(valid_types)}."
        )


def validate_rng(rng: RngTypes, allow_seed: bool = True) -> Generator:
    """
    Validate and convert input to a numpy.random.Generator instance.

    Parameters
    ----------
    rng : {None, int, numpy.random.Generator}
        Random number generator or seed.
        If None, a new default Generator is returned.
        If int and allow_seed is True, it's used to seed a new Generator.
        If Generator, it's returned unchanged.
    allow_seed : bool, optional
        Whether to allow integer seeds. Default is True.

    Returns
    -------
    numpy.random.Generator
        A valid numpy random number generator.

    Raises
    ------
    TypeError
        If rng is not of an allowed type based on the allow_seed parameter.
    ValueError
        If rng is an integer outside the range [0, 2**32 - 1].

    Notes
    -----
    This function ensures that a valid numpy.random.Generator is always returned,
    either by creating a new one or validating an existing one.
    """
    # Case 1: rng is None, return a new default Generator
    if rng is None:
        return np.random.default_rng()

    # Case 2: rng is already a Generator, return it
    if isinstance(rng, Generator):
        return rng

    # Case 3: allow_seed is True, check if rng is a valid integer seed
    if allow_seed:
        if isinstance(rng, int):
            if not (0 <= rng < 2**32):
                raise ValueError(f"The random seed must be between 0 and 2**32 - 1. Got {rng}")
            return np.random.default_rng(rng)
        else:
            raise TypeError("The input must be None, a Generator instance, or an integer seed.")

    # Case 4: allow_seed is False, and rng is not None or a Generator
    else:
        raise TypeError("The input must be None or a Generator instance.")


def validate_order(order) -> None:
    """
    Validates the type of the resids_order order.

    This function validates the order but doesn't return the normalized value.
    For validation with value normalization, use tsbootstrap.validators.validate_order.

    Parameters
    ----------
    order : Any
        The order to validate.

    Raises
    ------
    TypeError
        If the order is not of the expected type (Integral, list, or tuple).
    ValueError
        If the order is an integral but is negative or zero.
        If the order is a list and not all elements are positive integers.
        If the order is a tuple and not all elements are non-negative integers.
    """
    from tsbootstrap.validators import validate_order as validate_order_full

    # Use the full validator which returns normalized value
    validate_order_full(order)
