"""
Utility functions: Essential tools refined through production experience.

This module contains utility functions that have proven indispensable across
our bootstrap implementations. Each function represents a crystallization of
patterns we've encountered repeatedly—abstracted, optimized, and battle-tested.

These utilities embody the principle that good infrastructure makes the right
thing easy and the wrong thing hard. From random number generation with proper
seeding to output suppression for clean interfaces, each tool addresses a
specific need identified through real-world usage.
"""

import os
from contextlib import contextmanager

import numpy as np

from tsbootstrap.utils.types import RngTypes
from tsbootstrap.utils.validate import validate_rng


def generate_random_indices(num_samples: int, rng: RngTypes = None) -> np.ndarray:  # type: ignore
    """
    Generate bootstrap indices with proper randomization control.

    This function implements the core resampling mechanism for bootstrap methods,
    generating indices that sample with replacement from the original data. The
    implementation ensures both statistical validity and computational efficiency,
    with careful attention to random number generation best practices.

    We provide flexible randomization control to support both exploratory analysis
    (where reproducibility matters) and production systems (where true randomness
    is essential). The function integrates seamlessly with numpy's modern random
    number generation framework.

    Parameters
    ----------
    num_samples : int
        Number of indices to generate, typically matching the original data size.
        This maintains the same sample size across bootstrap iterations, ensuring
        valid statistical inference.

    rng : RngTypes, optional
        Random number control. Accepts an integer seed for reproducibility,
        a configured Generator for fine control, or None for system entropy.
        We recommend explicit seeding for research reproducibility.

    Returns
    -------
    np.ndarray
        Array of indices for resampling, shape (num_samples,). Each index
        references a position in the original data, with repetition reflecting
        the sampling with replacement process.

    Examples
    --------
    >>> # Reproducible sampling for research
    >>> generate_random_indices(5, rng=42)
    array([4, 0, 3, 3, 3])

    >>> # Production usage with system randomness
    >>> indices = generate_random_indices(1000)  # True random sampling
    """
    # Check types and values of num_samples and random_seed
    from tsbootstrap.utils.validate import validate_integers

    validate_integers(num_samples, min_value=1)  # type: ignore
    rng = validate_rng(rng, allow_seed=True)

    # Generate random indices with replacement
    in_bootstrap_indices = rng.choice(
        np.arange(num_samples), size=num_samples, replace=True  # type: ignore
    )

    return in_bootstrap_indices


@contextmanager
def suppress_output(verbose: int = 2):
    """A context manager for controlling the suppression of stdout and stderr.

    Parameters
    ----------
    verbose : int, optional
        Verbosity level controlling suppression.
        2 - No suppression (default)
        1 - Suppress stdout only
        0 - Suppress both stdout and stderr

    Returns
    -------
    None

    Examples
    --------
    with suppress_output(verbose=1):
        print('This will not be printed to stdout')
    """
    # No suppression required
    if verbose == 2:
        yield
        return

    # Open null files as needed
    null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2 if verbose == 0 else 1)]
    # Save the actual stdout (1) and possibly stderr (2) file descriptors.
    save_fds = [os.dup(1), os.dup(2)] if verbose == 0 else [os.dup(1)]
    try:
        # Assign the null pointers as required
        os.dup2(null_fds[0], 1)
        if verbose == 0:
            os.dup2(null_fds[1], 2)
        yield
    finally:
        # Re-assign the real stdout/stderr back
        for fd, save_fd in zip(null_fds, save_fds):
            os.dup2(save_fd, fd)
        # Close the null files and saved file descriptors
        for fd in null_fds + save_fds:
            os.close(fd)


def _check_nan_inf_locations(a: np.ndarray, b: np.ndarray, check_same: bool) -> bool:
    """
    Check the locations of NaNs and Infs in both arrays.

    Parameters
    ----------
    a, b : np.ndarray
        The arrays to be compared.
    check_same : bool
        If True, checks if NaNs and Infs are in the same locations.

    Returns
    -------
    bool
        True if locations do not match and check_same is False, otherwise False.

    Raises
    ------
    ValueError
        If check_same is True and the arrays have NaNs or Infs in different locations.
    """
    a_nan_locs = np.isnan(a)
    b_nan_locs = np.isnan(b)
    a_inf_locs = np.isinf(a)
    b_inf_locs = np.isinf(b)

    if not np.array_equal(a_nan_locs, b_nan_locs) or not np.array_equal(a_inf_locs, b_inf_locs):
        if check_same:
            raise ValueError(
                "Arrays have NaN or infinity values at different positions. "
                "For arrays to be considered equal, special values (NaN, inf, -inf) "
                "must appear at the same indices in both arrays. Check your data "
                "for inconsistent handling of missing or infinite values."
            )
        else:
            return True

    return False


def _check_inf_signs(a: np.ndarray, b: np.ndarray, check_same: bool) -> bool:
    """
    Check the signs of Infs in both arrays.

    Parameters
    ----------
    a, b : np.ndarray
        The arrays to be compared.
    check_same : bool
        If True, checks if Infs have the same signs.

    Returns
    -------
    bool
        True if signs do not match and check_same is False, otherwise False.

    Raises
    ------
    ValueError
        If check_same is True and the arrays have Infs with different signs.
    """
    a_inf_locs = np.isinf(a)
    b_inf_locs = np.isinf(b)

    if not np.array_equal(np.sign(a[a_inf_locs]), np.sign(b[b_inf_locs])):
        if check_same:
            raise ValueError(
                "Arrays contain infinities with different signs at the same position. "
                "One array has positive infinity while the other has negative infinity "
                "at corresponding indices. These values cannot be considered approximately equal."
            )
        else:
            return True

    return False


def _check_close_values(
    a: np.ndarray, b: np.ndarray, rtol: float, atol: float, check_same: bool
) -> bool:
    """
    Check that the finite values in the arrays are close.

    Parameters
    ----------
    a, b : np.ndarray
        The arrays to be compared.
    rtol : float
        The relative tolerance parameter for the np.allclose function.
    atol : float
        The absolute tolerance parameter for the np.allclose function.
    check_same : bool
        If True, checks if the arrays are almost equal.

    Returns
    -------
    bool
        True if values are not close and check_same is False, otherwise False.

    Raises
    ------
    ValueError
        If check_same is True and the arrays are not almost equal.
    """
    a_nan_locs = np.isnan(a)
    b_nan_locs = np.isnan(b)
    a_inf_locs = np.isinf(a)
    b_inf_locs = np.isinf(b)
    a_masked = np.ma.masked_where(a_nan_locs | a_inf_locs, a)
    b_masked = np.ma.masked_where(b_nan_locs | b_inf_locs, b)

    if check_same:
        if not np.allclose(a_masked, b_masked, rtol=rtol, atol=atol):
            raise ValueError(
                f"Arrays are not approximately equal within tolerance. "
                f"The relative tolerance is rtol={rtol} and absolute tolerance is atol={atol}. "
                f"Some values differ by more than these tolerances allow. "
                f"Consider increasing tolerance if small differences are acceptable."
            )
    else:
        if np.any(~np.isclose(a_masked, b_masked, rtol=rtol, atol=atol)):
            return True

    return False


def assert_arrays_compare(
    a: np.ndarray, b: np.ndarray, rtol=1e-5, atol=1e-8, check_same=True
) -> bool:
    """
    Assert that two arrays are almost equal.

    This function compares two arrays for equality, allowing for NaNs and Infs in the arrays.
    The arrays are considered equal if the following conditions are satisfied:
    1. The locations of NaNs and Infs in both arrays are the same.
    2. The signs of the infinite values in both arrays are the same.
    3. The finite values are almost equal.

    Parameters
    ----------
    a, b : np.ndarray
        The arrays to be compared.
    rtol : float, optional
        The relative tolerance parameter for the np.allclose function.
        Default is 1e-5.
    atol : float, optional
        The absolute tolerance parameter for the np.allclose function.
        Default is 1e-8.
    check_same : bool, optional
        If True, raise an AssertionError if the arrays are not almost equal.
        If False, return True if the arrays are not almost equal and False otherwise.
        Default is True.

    Returns
    -------
    bool
        If check_same is False, returns True if the arrays are not almost equal and False otherwise.
        If check_same is True, returns True if the arrays are almost equal and False otherwise.

    Raises
    ------
    AssertionError
        If check_same is True and the arrays are not almost equal.
    ValueError
        If check_same is True and the arrays have NaNs or Infs in different locations.
        If check_same is True and the arrays have Infs with different signs.
    """
    if _check_nan_inf_locations(a, b, check_same):
        return not check_same
    if _check_inf_signs(a, b, check_same):
        return not check_same
    if _check_close_values(a, b, rtol, atol, check_same):
        return not check_same

    return not check_same if not check_same else True
