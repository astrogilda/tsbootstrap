
import numpy as np
from numba import njit
from typing import Tuple


def is_callable(obj: object) -> bool:
    """
    Check if the provided object is callable.

    This function checks whether a provided object is callable 
    by using the built-in Python function `callable`.

    Parameters
    ----------
    obj : object
        The object to check for callability.

    Returns
    -------
    bool
        True if the object is callable, False otherwise.

    Examples
    --------
    >>> is_callable(lambda x: x + 1)
    True
    >>> is_callable(5)
    False

    """
    return callable(obj)


def is_numba_compiled(fn: callable) -> bool:
    """
    Check if a function is compiled by Numba.

    This function checks whether a provided callable object 
    (usually a function or a method) is compiled by Numba.

    Parameters
    ----------
    fn : callable
        The callable object to check.

    Returns
    -------
    bool
        True if the function is compiled by Numba, False otherwise.

    Examples
    --------
    >>> from numba import njit
    >>> @njit
    ... def add(x, y):
    ...     return x + y
    ...
    >>> is_numba_compiled(add)
    True
    >>> def subtract(x, y):
    ...     return x - y
    ...
    >>> is_numba_compiled(subtract)
    False

    """
    # Check if the callable object has the "__numba__" attribute
    return getattr(fn, "__numba__", False)


@njit
def normalize_array(array: np.ndarray) -> np.ndarray:
    """
    Normalize the block_weights array.

    Parameters
    ----------
    array : np.ndarray
        1d array.

    Returns
    -------
    np.ndarray
        An array of normalized values, shape == (-1,1).

    Notes
    -----
    This function is used to normalize the block_weights array. Please note that the input array is modified in-place. It expects a 1d array, or a 2d array with only one column. The array should not contain NaN or infinite values, and should not contain complex values.
    """

    sum_array = np.sum(array)

    if sum_array == 0:  # Handle zero-sum case
        return np.full_like(array, 1.0/len(array)).reshape(-1, 1)
    else:
        return (array / sum_array).reshape(-1, 1)


@njit
def choice_with_p(weights: np.ndarray) -> int:
    """
    Given an array of weights, this function returns a single index
    sampled with probabilities proportional to the input weights.

    Parameters
    ----------
    weights : np.ndarray
        A 1D array, or a 2D array with one column, of probabilities for each index. The array should not contain NaN or infinite values,
        should not contain complex values, and all elements should be non-negative.

    Returns
    -------
    int
        A single sampled index.

    Notes
    -----
    This function is used to sample indices from the block_weights array. The array is normalized before sampling.
    Only call this function with the output of '_prepare_block_weights' or '_prepare_taper_weights'.
    """

    # Normalize weights
    p = weights / weights.sum()
    # Create cumulative sum of normalized weights (these will now act as probabilities)
    cum_weights = np.cumsum(p)
    # Draw a single random value
    random_value = np.random.rand()
    # Find index where drawn random value would be inserted in cumulative weight array
    chosen_index = np.searchsorted(cum_weights, random_value)
    return chosen_index


def time_series_split(X: np.ndarray, test_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits a given time series into training and test sets

    Parameters
    ----------
    X : np.ndarray
        The input time series.
    test_ratio : float
        The ratio of the test set size to the total size of the series.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the training set and the test set.
    """
    # Validate test_ratio
    if not 0 <= test_ratio <= 1:
        raise ValueError("Test ratio must be between 0 and 1.")

    split_index = int(len(X) * (1 - test_ratio))
    return X[:split_index], X[split_index:]


@njit
def mean_axis_0(x: np.ndarray) -> np.ndarray:
    """
    Compute the mean along the zeroth axis of a 2D numpy array.

    Parameters
    ----------
    x : np.ndarray
        A 2D numpy array for which the mean along the zeroth axis will be computed.

    Returns
    -------
    np.ndarray
        A 1D numpy array of the means along the zeroth axis.

    Notes
    -----
    This function is decorated with the numba `@njit` decorator, 
    and thus is compiled to machine code at runtime for performance.

    """
    # Get the shape of the input array
    n, k = x.shape
    # Initialize an array of zeros to hold the mean values
    mean = np.zeros(k)

    # Iterate over the zeroth axis
    for i in range(n):
        # Increment the mean values by the current row of x
        mean += x[i]

    # Divide the mean values by the length of the zeroth axis to get the mean
    mean /= n

    return mean
