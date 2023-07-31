
from typing import get_args, TypeVar, Literal
import numpy as np
from numpy import ndarray
from sklearn.utils import check_array, check_X_y
from typing import Union, List, Optional, Tuple, Literal, Any
from numbers import Integral
from utils.types import FittedModelType, ModelTypes, RngTypes, BlockCompressorTypes
from numpy.random import Generator
from utils.odds_and_ends import check_generator


def validate_integers(*values: Union[Integral, List[Integral], np.ndarray], positive: bool = False) -> None:
    for value in values:
        if isinstance(value, Integral):
            # If value is an integer, check if positive if required
            if positive and value <= 0:
                raise TypeError(f"All integers must be positive. Got {value}.")
            continue

        if isinstance(value, list):
            # Check if the list is empty
            if not value:
                raise TypeError(f"List must not be empty. Got {value}.")

            # Check if every element in the list is an integer
            if not all(isinstance(x, Integral) for x in value):
                raise TypeError(
                    f"All elements in the list must be integers. Got {value}.")

            # Check if every element in the list is positive if required
            if positive and any(x <= 0 for x in value):
                raise TypeError(
                    f"All integers in the list must be positive. Got {value}.")
            continue

        if isinstance(value, np.ndarray):
            # Check if the array is empty
            if value.size == 0:
                raise TypeError(f"Array must not be empty. Got {value}.")

            # Check if the array is 1D and if every element is an integer
            # i for signed integer, u for unsigned integer
            if value.ndim != 1 or not value.dtype.kind in 'iu':
                raise TypeError(
                    f"Array must be 1D and contain only integers. Got {value}.")

            # Check if every element in the array is positive if required
            if positive and any(value <= 0):
                raise TypeError(
                    f"All integers in the array must be positive. Got {value}.")
            continue

        # If none of the above conditions are met, the input is invalid
        raise TypeError(
            f"Input must be an integer, a list of integers, or a 1D array of integers. Got {value}.")


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


def validate_block_indices(block_indices: List[np.ndarray], input_length: Integral) -> None:
    """
    Validate the input block indices.
    """
    # Check if 'block_indices' is a list
    if not isinstance(block_indices, list):
        raise TypeError("Input 'block_indices' must be a list.")

    # Check if 'block_indices' is empty
    if len(block_indices) == 0:
        raise ValueError("Input 'block_indices' must not be empty.")

    # Check if 'block_indices' contains only NumPy arrays
    if not all(isinstance(block, np.ndarray) for block in block_indices):
        raise TypeError(
            "Input 'block_indices' must be a list of NumPy arrays.")

    # Check if 'block_indices' contains only 1D NumPy arrays with integer values
    if not all(block.ndim == 1 and np.issubdtype(block.dtype, np.integer) for block in block_indices):
        raise ValueError(
            "Input 'block_indices' must be a list of 1D NumPy arrays with integer values.")

    # Check if 'block_indices' contains only NumPy arrays with at least one index
    if not all(block.size > 0 for block in block_indices):
        raise ValueError(
            "Input 'block_indices' must be a list of 1D NumPy arrays with at least one index.")

    # Check if 'block_indices' contains only NumPy arrays with indices within the range of X
    if not all(np.all(block < input_length) for block in block_indices):
        raise ValueError(
            "Input 'block_indices' must be a list of 1D NumPy arrays with indices within the range of X.")


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

    # Check if 'blocks' contains only NumPy arrays with at least one element
    if not all(block.shape[0] > 0 for block in blocks):
        raise ValueError(
            "Input 'blocks' must be a list of 2D NumPy arrays with at least one element.")

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
    if (weights.ndim == 2 and weights.shape[1] != 1) or weights.ndim > 2:
        raise ValueError(
            "The provided callable function resulted in a 2D array with more than one column. Please check your function.")


def validate_fitted_model(fitted_model: FittedModelType) -> None:
    valid_types = FittedModelType.__args__
    if not isinstance(fitted_model, valid_types):
        valid_names = ', '.join([t.__name__ for t in valid_types])
        raise ValueError(
            f"fitted_model must be an instance of {valid_names}.")


# LiteralTypeVar = TypeVar("LiteralTypeVar", bound=Literal)


def validate_literal_type(input_value: str, literal_type: Any) -> None:
    """Validate the type of input_value against a Literal type.

    This function validates if the input_value is among the valid types defined 
    in the literal_type.

    Parameters
    ----------
    input_value : str
        The value to validate.
    literal_type : LiteralTypeVar
        The Literal type against which to validate the input_value.

    Raises
    ------
    TypeError
        If input_value is not a string.
    ValueError
        If input_value is not among the valid types in literal_type.
    """
    valid_types = get_args(literal_type)
    if not isinstance(input_value, str):
        raise TypeError(
            f"input_value must be a string. Got {type(input_value).__name__} instead.")
    if input_value.lower() not in valid_types:
        raise ValueError(
            f"Invalid input_value '{input_value}'. Expected one of {', '.join(valid_types)}.")


def validate_X(X: np.ndarray, model_is_var: bool = False) -> None:
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D NumPy array.")
    if model_is_var and X.shape[1] < 2:
        raise ValueError("X must have at least two columns.")
    if np.isnan(X).any():
        raise ValueError("X must not contain NaN values.")
    if np.isinf(X).any():
        raise ValueError("X must not contain infinite values.")
    if np.iscomplex(X).any():
        raise ValueError("X must not contain complex values.")


def validate_rng(rng: RngTypes) -> None:
    if rng is not None and not isinstance(rng, (Generator, Integral)):
        raise TypeError(
            'The random number generator must be an instance of the numpy.random.Generator class, or an integer.')
    rng = check_generator(rng)
    return rng
