import numpy as np
from numpy import ndarray
from sklearn.utils import check_array, check_X_y
from typing import Union, List, Optional, Tuple


def validate_integers(*values: Union[int, List[int], np.ndarray], positive: bool = False) -> None:
    for value in values:
        if isinstance(value, int):
            # If value is an integer, check if positive if required
            if positive and value <= 0:
                raise TypeError("All integers must be positive.")
            continue

        if isinstance(value, list):
            # Check if the list is empty
            if not value:
                raise TypeError("List must not be empty.")

            # Check if every element in the list is an integer
            if not all(isinstance(x, int) for x in value):
                raise TypeError("All elements in the list must be integers.")

            # Check if every element in the list is positive if required
            if positive and any(x <= 0 for x in value):
                raise TypeError("All integers in the list must be positive.")
            continue

        if isinstance(value, np.ndarray):
            # Check if the array is empty
            if value.size == 0:
                raise TypeError("Array must not be empty.")

            # Check if the array is 1D and if every element is an integer
            if value.ndim != 1 or not np.issubdtype(value.dtype, np.integer):
                raise TypeError("Array must be 1D and contain only integers.")

            # Check if every element in the array is positive if required
            if positive and np.any(value <= 0):
                raise TypeError("All integers in the array must be positive.")
            continue

        # If none of the above conditions are met, the input is invalid
        raise TypeError(
            "Input must be an integer, a list of integers, or a 1D array of integers.")


def validate_X_and_exog(X: ndarray, exog: Optional[np.ndarray], model_is_var: bool = False, model_is_arch: bool = False) -> Tuple[ndarray, Optional[np.ndarray]]:
    """
    Validate and reshape input data and exogenous variables.

    Args:
        X (ndarray): The input data.
        exog (Optional[np.ndarray]): Optional exogenous variables.
        model_is_var (bool): Indicates if the model is a VAR model.
        model_is_arch (bool): Indicates if the model is an ARCH model.

    Returns:
        Tuple[ndarray, Optional[np.ndarray]]: The validated and reshaped X and exog arrays.
    """
    # Validate and reshape X
    if not model_is_var:
        X = check_array(X, ensure_2d=False, force_all_finite=True)
        X = np.squeeze(X)
        if X.ndim != 1:
            raise ValueError("X must be 1-dimensional")
    else:
        X = check_array(X, ensure_2d=True, force_all_finite=True)
        if X.shape[1] < 2:
            raise ValueError("X must be 2-dimensional with at least 2 columns")

    # Validate and reshape exog if necessary
    if exog is not None:
        if exog.ndim == 1:
            exog = exog[:, np.newaxis]
        exog = check_array(exog, ensure_2d=True, force_all_finite=True)
        exog, X = check_X_y(exog, X, force_all_finite=True,
                            multi_output=model_is_var)

    # Ensure contiguous arrays for ARCH models
    if model_is_arch:
        X = np.ascontiguousarray(X)
        if exog is not None:
            exog = np.ascontiguousarray(exog)

    return X, exog


def validate_blocks(blocks: List[np.ndarray]) -> None:
    """
    Validate the input blocks.
    """
    # Check if 'blocks' is a list
    if not isinstance(blocks, list):
        raise TypeError("Input 'blocks' must be a list.")

    # Check if 'blocks' is empty
    if len(blocks) == 0:
        raise ValueError("Input 'blocks' must not be empty.")

    # Check if 'blocks' contains only NumPy arrays
    if not all(isinstance(block, np.ndarray) for block in blocks):
        raise TypeError("Input 'blocks' must be a list of NumPy arrays.")

    # Check if 'blocks' contains only 2D NumPy arrays
    if not all(block.ndim == 2 for block in blocks):
        raise ValueError(
            "Input 'blocks' must be a list of 2D NumPy arrays.")

    # Check if 'blocks' contains only NumPy arrays with at least one timestamp
    if not all(block.shape[0] > 0 for block in blocks):
        raise ValueError(
            "Input 'blocks' must be a list of 2D NumPy arrays with at least one timestamp.")

    # Check if 'blocks' contains only NumPy arrays with at least one feature
    if not all(block.shape[1] > 0 for block in blocks):
        raise ValueError(
            "Input 'blocks' must be a list of 2D NumPy arrays with at least one feature.")

    # Check if 'blocks' contains only NumPy arrays with the same number of features
    if not all(block.shape[1] == blocks[0].shape[1] for block in blocks):
        raise ValueError(
            "Input 'blocks' must be a list of 2D NumPy arrays with the same number of features.")

    # Check if 'blocks' contains NumPy arrays with finite values
    if not all(np.all(np.isfinite(block)) for block in blocks):
        raise ValueError(
            "Input 'blocks' must be a list of 2D NumPy arrays with finite values.")


def validate_weights(weights: np.ndarray) -> None:
    # Check if weights contains any non-finite values
    if not np.isfinite(weights).all():
        raise ValueError(
            "The provided callable function or array resulted in non-finite values. Please check your inputs.")
    # Check if weights contains any negative values
    if np.any(weights < 0):
        raise ValueError(
            "The provided callable function resulted in negative values. Please check your function.")
    # Check if weights contains any complex values
    if np.any(np.iscomplex(weights)):
        raise ValueError(
            "The provided callable function resulted in complex values. Please check your function.")
    # Check if weights contains all zeros
    if np.all(weights == 0):
        raise ValueError(
            "The provided callable function resulted in all zero values. Please check your function.")
    # Check if tapered_weights_arr is a 1D array or a 2D array with a single column
    if weights.ndim == 2 and weights.shape[1] != 1:
        raise ValueError(
            "The provided callable function resulted in a 2D array with more than one column. Please check your function.")
