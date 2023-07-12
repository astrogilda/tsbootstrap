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
