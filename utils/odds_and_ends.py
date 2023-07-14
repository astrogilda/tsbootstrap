
import numpy as np
from numba import njit
from typing import Tuple


def is_callable(obj):
    return callable(obj)


def is_numba_compiled(fn):
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
    """

    sum_array = np.sum(array)

    if sum_array == 0:  # Handle zero-sum case
        return np.full_like(array, 1.0/len(array)).reshape(-1, 1)
    else:
        return (array / sum_array).reshape(-1, 1)


@njit
def choice_with_p(weights: np.ndarray) -> np.ndarray:
    """
    Given an array of weights, this function returns an array of indices
    sampled with probabilities proportional to the input weights.

    Parameters
    ----------
    weights : np.ndarray
        An array of probabilities for each index.

    Returns
    -------
    np.ndarray
        An array of sampled indices.
    """
    if weights.ndim != 1:
        raise ValueError("Weights must be a 1-dimensional array.")

    if np.any(weights < 0):
        raise ValueError("All elements of weights must be non-negative.")

    size = len(weights)

    # Normalize weights
    p = weights / weights.sum()
    # Create cumulative sum of normalized weights (these will now act as probabilities)
    cum_weights = np.cumsum(p)
    # Draw random values
    random_values = np.random.rand(size)
    # Find indices where drawn random values would be inserted in cumulative weight array
    chosen_indices = np.searchsorted(cum_weights, random_values)
    return chosen_indices


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
def mean_axis_0(x):
    n, k = x.shape
    mean = np.zeros(k)
    for i in range(n):
        mean += x[i]
    mean /= n
    return mean
