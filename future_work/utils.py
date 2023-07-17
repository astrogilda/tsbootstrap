

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
