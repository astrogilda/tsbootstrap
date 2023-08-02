
import sys
from contextlib import contextmanager
import os
from numpy.random import Generator
import numpy as np
from typing import Tuple, Optional
from numbers import Integral


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
        raise ValueError(
            f"Test ratio must be between 0 and 1. Got {test_ratio}")

    split_index = int(len(X) * (1 - test_ratio))
    return X[:split_index], X[split_index:]


def check_generator(seed_or_rng, seed_allowed: bool = True) -> Generator:
    """Turn seed into a np.random.Generator instance.

    Parameters
    ----------
    seed_or_rng : int, Generator, or None
        If seed_or_rng is None, return the Generator singleton used by np.random.
        If seed_or_rng is an int, return a new Generator instance seeded with seed_or_rng.
        If seed_or_rng is already a Generator instance, return it.
        Otherwise raise ValueError.
    """
    if seed_or_rng is None:
        return np.random.default_rng()
    if isinstance(seed_or_rng, Generator):
        return seed_or_rng
    if seed_allowed:
        if isinstance(seed_or_rng, Integral):
            if not (0 <= seed_or_rng < 2**32):
                raise ValueError(
                    f"The random seed must be between 0 and 2**32 - 1. Got {seed_or_rng}")
            return np.random.default_rng(seed_or_rng)

    raise ValueError(
        f"{seed_or_rng} cannot be used to seed a numpy.random.Generator instance"
    )


def generate_random_indices(num_samples: int, rng: Optional[Generator] = None) -> np.ndarray:
    """
    Generate random indices with replacement.

    This function generates random indices from 0 to `num_samples-1` with replacement.
    The generated indices can be used for bootstrap sampling, etc.

    Parameters
    ----------
    num_samples : int
        The number of samples for which the indices are to be generated. 
        This must be a positive integer.
    rng : int, optional
        The seed for the random number generator. If provided, this must be a non-negative integer.
        Default is None, which does not set the numpy's random seed and the results will be non-deterministic.

    Returns
    -------
    np.ndarray
        A numpy array of shape (`num_samples`,) containing randomly generated indices.

    Raises
    ------
    ValueError
        If `num_samples` is not a positive integer or if `random_seed` is provided and 
        it is not a non-negative integer.

    Examples
    --------
    >>> generate_random_indices(5, random_seed=0)
    array([4, 0, 3, 3, 3])
    >>> generate_random_indices(5)
    array([2, 1, 4, 2, 0])  # random
    """

    # Check types and values of num_samples and random_seed
    from utils.validate import validate_integers
    validate_integers(num_samples, min_value=1)
    rng = check_generator(rng, seed_allowed=False)

    # Generate random indices with replacement
    in_bootstrap_indices = rng.choice(
        np.arange(num_samples), size=num_samples, replace=True)

    return in_bootstrap_indices


@contextmanager
def suppress_stdout_stderr():
    """A context manager for doing a "deep suppression" of stdout and stderr, 
    i.e. will suppress all print, even if the print originates in a compiled 
    C/Fortran sub-function.
    """
    # Open a pair of null files
    null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
    # Save the actual stdout (1) and stderr (2) file descriptors.
    save_fds = [os.dup(1), os.dup(2)]
    try:
        # Assign the null pointers to stdout and stderr.
        os.dup2(null_fds[0], 1)
        os.dup2(null_fds[1], 2)
        yield
    finally:
        # Re-assign the real stdout/stderr back to (1) and (2)
        for fd, save_fd in zip(null_fds, save_fds):
            os.dup2(save_fd, fd)
        # Close the null files
        for fd in null_fds + save_fds:
            os.close(fd)


@contextmanager
def suppress_stdout():
    """A context manager that redirects stdout and stderr to devnull"""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        yield
    finally:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def assert_arrays_compare(a, b, rtol=1e-5, atol=1e-8, check_same=True):
    # Identify the locations of nans and infs in both arrays
    a_nan_locs = np.isnan(a)
    b_nan_locs = np.isnan(b)
    a_inf_locs = np.isinf(a)
    b_inf_locs = np.isinf(b)

    # Check that the location of nans and infs matches in both arrays
    if check_same:
        assert np.array_equal(
            a_nan_locs, b_nan_locs), "NaNs in different locations"
        assert np.array_equal(
            a_inf_locs, b_inf_locs), "Infs in different locations"
    else:
        if not np.array_equal(a_nan_locs, b_nan_locs):
            return True
        if not np.array_equal(a_inf_locs, b_inf_locs):
            return True

    # Check that the signs of the infinite values match in both arrays
    if check_same:
        assert np.array_equal(np.sign(a[a_inf_locs]), np.sign(
            b[b_inf_locs])), "Infs with different signs"
    else:
        if not np.array_equal(np.sign(a[a_inf_locs]), np.sign(b[b_inf_locs])):
            return True

    # Mask the nans and infs in both arrays
    a_masked = np.ma.masked_where(a_nan_locs | a_inf_locs, a)
    b_masked = np.ma.masked_where(b_nan_locs | b_inf_locs, b)

    # Now use np.allclose or np.any and np.isclose on the finite values, based on check_same
    if check_same:
        assert np.allclose(a_masked, b_masked, rtol=rtol,
                           atol=atol), "Arrays are not almost equal"
    else:
        if np.any(~np.isclose(a_masked, b_masked, rtol=rtol, atol=atol)):
            return True

    # If we're checking for differences and haven't returned True yet, then the arrays are the same
    if not check_same:
        return False

    # If we're checking for sameness and haven't raised an AssertionError, then the arrays are the same
    return True
