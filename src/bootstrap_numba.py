
from typing import List, Union, Optional
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper
from statsmodels.tsa.arima.model import ARIMAResultsWrapper
from arch.univariate.base import ARCHModelResult

import numpy as np
from numba import njit
from numpy.random import Generator
from numbers import Integral

from utils.markov_sampler import MarkovSampler
from utils.odds_and_ends import *

# TODO: block_weights=p with block_length=1 should be equivalent to the iid bootstrap

"""
Call the below functions from src.bootstrap.py. These functions have not had their inputs checked.
"""


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
    random_seed : int, optional
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
    if not (isinstance(num_samples, Integral) and num_samples > 0):
        raise ValueError("num_samples must be a positive integer.")

    if rng is not None and not isinstance(rng, Generator):
        raise ValueError("rng must be either None or a Geneartor instance.")

    # Use the global state if rng is not provided if provided
    if rng is None:
        np.random.default_rng()

    # Generate random indices with replacement
    in_bootstrap_indices = rng.choice(
        np.arange(num_samples), size=num_samples, replace=True)

    return in_bootstrap_indices


'''
def generate_samples_markov(blocks: List[np.ndarray], n_clusters: int, random_seed: int, **kwargs) -> np.ndarray:
    """
    Generate a bootstrapped time series based on the Markov chain bootstrapping method.

    Parameters
    ----------
    blocks : List[np.ndarray]
        A list of numpy arrays representing the original time series blocks. The last block may have fewer samples than block_length.
    n_clusters : int
        The number of clusters for the Hidden Markov Model.
    random_seed : int
        The seed for the random number generator.

    Other Parameters
    ----------------
    apply_pca : bool, optional
        Whether to apply PCA, by default False.
    pca : object, optional
        PCA object to apply, by default None.
    n_iter_hmm : int, optional
        Number of iterations for the HMM model, by default 100.
    n_fits_hmm : int, optional
        Number of fits for the HMM model, by default 10.
    method : str
        The method to be used for block summarization.
    blocks_as_hidden_states_flag : bool, optional

    Returns
    -------
    np.ndarray
        A numpy array representing the bootstrapped time series.

    """

    markov_sampler = MarkovSampler(random_seed=random_seed, **kwargs)
    bootstrapped_series = markov_sampler.sample(blocks=blocks, n_states=n_clusters,
                                                random_seed=random_seed)

    return bootstrapped_series

'''


# @njit
def simulate_ar_process(n_samples: int, lags: np.ndarray, coefs: np.ndarray, init: np.ndarray, rng: Optional[Union[Integral, Generator]] = None) -> np.ndarray:
    """
    Simulates an Autoregressive (AR) process with given lags, coefficients, initial values, and random errors.

    Args:
        lags (np.ndarray): The lags to be used in the AR process. Can be non-consecutive, but when called from `generate_samples_sieve_autoreg`, it will be sorted.
        coefs (np.ndarray): The coefficients corresponding to each lag. Of shape (1, len(lags)). Sorted by `generate_samples_sieve_autoreg` corresponding to the sorted `lags`.
        init (np.ndarray): The initial values for the simulation. Should be at least as long as the maximum lag.
        rng (Optional[Union[Integral, Generator]], optional): The random number generator. Defaults to None.

    Returns:
        np.ndarray: The simulated AR process as a 1D NumPy array.

    Raises:
        ValueError: If `init` is not long enough to cover the maximum lag.
    """
    # Set the random seed
    rng = check_generator(rng)
    random_errors = rng.normal(size=n_samples)

    # Convert lags to a NumPy array if it is not already. When called from `generate_samples_sieve_autoreg`, it will be sorted.
    lags = np.array(sorted(lags))
    max_lag = np.max(lags)
    if len(init) < max_lag:
        raise ValueError(
            "Length of 'init' must be at least as long as the maximum lag in 'lags'")
    if coefs.shape[0] != 1:
        raise ValueError(
            "AR coefficients must be a 1D NumPy array of shape (1, X)")
    if coefs.shape[1] != len(lags):
        raise ValueError(
            "Length of 'coefs' must be the same as the length of 'lags'")

    # In case init is 2d with shape (X, 1), convert it to 1d
    init = init.ravel()
    series = np.zeros(n_samples, dtype=init.dtype)
    series[:max_lag] = init

    # Loop through the series, calculating each value based on the lagged values, coefficients, and random error
    for t in range(max_lag, n_samples):
        ar_term = 0
        for i in range(len(lags)):
            ar_term += coefs[0, i] * series[t - lags[i]]
        series[t] = ar_term + random_errors[t]

    return series


def simulate_arima_process(n_samples: int, fitted_model: ARIMAResultsWrapper, rng: Optional[Union[Integral, Generator]] = None) -> np.ndarray:
    """
    Simulate a time series from an ARIMA model.

    Args:
        n_samples (int): The number of samples to simulate.
        fitted_model (ARIMAResultsWrapper): The fitted ARIMA model.
        random_seed (int): The seed for the random number generator.

    Returns:
        np.ndarray: The simulated time series.

    Raises:
        ValueError: If the fitted_model is not an instance of ARIMAResultsWrapper.
    """
    if not isinstance(fitted_model, ARIMAResultsWrapper):
        raise ValueError(
            "fitted_model must be an instance of ARIMAResultsWrapper.")

    # Set the random seed
    rng = check_generator(rng)
    simulated_series = fitted_model.simulate(
        nsimulations=n_samples, random_state=rng)
    return simulated_series


def simulate_sarima_process(n_samples: int, fitted_model: SARIMAXResultsWrapper, rng: Optional[Union[Integral, Generator]] = None) -> np.ndarray:
    """
    Simulate a time series from a SARIMA model.

    Args:
        n_samples (int): The number of samples to simulate.
        fitted_model (SARIMAXResultsWrapper): The fitted SARIMA model.
        random_seed (int): The seed for the random number generator.

    Returns:
        np.ndarray: The simulated time series.

    Raises:
        ValueError: If the fitted_model is not an instance of SARIMAXResultsWrapper.
    """
    if not isinstance(fitted_model, SARIMAXResultsWrapper):
        raise ValueError(
            "fitted_model must be an instance of SARIMAXResultsWrapper.")

    # Set the random seed
    rng = check_generator(rng)
    simulated_series = fitted_model.simulate(
        nsimulations=n_samples, random_state=rng)
    return simulated_series


def simulate_var_process(n_samples: int, fitted_model: VARResultsWrapper, rng: Optional[Union[Integral, Generator]] = None) -> np.ndarray:
    """
    Simulate a time series from a VAR model.

    Args:
        n_samples (int): The number of samples to simulate.
        fitted_model (VARResultsWrapper): The fitted VAR model.
        random_seed (int): The seed for the random number generator.

    Returns:
        np.ndarray: The simulated time series.

    Raises:
        ValueError: If the fitted_model is not an instance of VARResultsWrapper.
    """
    if not isinstance(fitted_model, VARResultsWrapper):
        raise ValueError(
            "fitted_model must be an instance of VARResultsWrapper.")
    # Set the random seed
    rng = check_generator(rng)
    simulated_series = fitted_model.simulate_var(
        steps=n_samples, random_state=rng)
    return simulated_series


# TODO: review this function -- definitely more complex than this. for instance, there is an initial_value and a burn argument that are not used here
def simulate_arch_process(n_samples: int, fitted_model: ARCHModelResult, rng: Optional[Union[Integral, Generator]] = None) -> np.ndarray:
    """
    Simulate a time series from an ARCH/GARCH model.

    Args:
        n_samples (int): The number of samples to simulate.
        fitted_model (ARCHModelResult): The fitted ARCH/GARCH model.
        random_seed (int): The seed for the random number generator.

    Returns:
        np.ndarray: The simulated time series.

    Raises:
        ValueError: If the fitted_model is not an instance of ARCHModelResult.
    """
    if not isinstance(fitted_model, ARCHModelResult):
        raise ValueError(
            "fitted_model must be an instance of ARCHModelResult.")
    # Set the random seed
    rng = check_generator(rng)
    simulated_data = fitted_model.model.simulate(
        params=fitted_model.params, nobs=n_samples, random_state=rng)
    return simulated_data['data'].values


def generate_samples_sieve_autoreg(
    X_fitted: np.ndarray,
    resids_lags: Union[Integral, List[int]],
    resids_coefs: np.ndarray,
    resids: np.ndarray,
    rng: Optional[Union[Integral, Generator]] = None,
) -> np.ndarray:

    # print(f"resids_coefs.shape: {resids_coefs.shape}")
    # print(f"resids_lags: {resids_lags}")

    n_samples, n_features = X_fitted.shape

    if n_features > 1:
        raise ValueError(
            "Only univariate time series are supported for the AR model.")
    if n_samples != len(resids):
        raise ValueError(
            "Length of 'resids' must be the same as the number of samples in 'X_fitted'.")
    # In case resids is 2d with shape (X, 1), convert it to 1d
    resids = resids.ravel()
    # In case X_fitted is 2d with shape (X, 1), convert it to 1d
    X_fitted = X_fitted.ravel()

    # Generate the bootstrap series
    bootstrap_series = np.zeros(n_samples, dtype=X_fitted.dtype)

    # Convert resids_lags to a NumPy array if it is not already
    resids_lags = np.arange(1, resids_lags + 1) if isinstance(
        resids_lags, Integral) else np.array(sorted(resids_lags))

    # Simulate residuals using the AR model
    max_lag = max(resids_lags)
    simulated_residuals = simulate_ar_process(
        n_samples=n_samples, lags=resids_lags, coefs=resids_coefs, init=resids[:max_lag], rng=rng)
    # simulated_residuals.shape: (n_samples,)

    bootstrap_series[:max_lag] = X_fitted[:max_lag]

    # Loop through the series, calculating each value based on the lagged values, coefficients, and random error
    for t in range(max_lag, n_samples):
        lagged_values = bootstrap_series[t - resids_lags]
        # lagged_values.shape: (n_lags,)
        lagged_values = lagged_values.reshape(-1, 1)
        # lagged_values.shape: (n_lags, 1)
        # print(f"lagged_values.shape: {lagged_values.shape}")
        bootstrap_series[t] = resids_coefs @ lagged_values + \
            simulated_residuals[t]

    return bootstrap_series.reshape(-1, 1)


def generate_samples_sieve_arima(
    X_fitted: np.ndarray,
    resids_fit_model: ARIMAResultsWrapper,
    rng: Optional[Union[Integral, Generator]] = None,
) -> np.ndarray:
    n_samples, n_features = X_fitted.shape

    # Simulate residuals using the ARIMA model
    simulated_residuals = simulate_arima_process(
        n_samples, resids_fit_model, rng)

    # Add the simulated residuals to the original series
    bootstrap_series = X_fitted + simulated_residuals

    return bootstrap_series


def generate_samples_sieve_sarima(
    X_fitted: np.ndarray,
    resids_fit_model: SARIMAXResultsWrapper,
    rng: Optional[Union[Integral, Generator]] = None,
) -> np.ndarray:
    n_samples, n_features = X_fitted.shape

    # Simulate residuals using the SARIMA model
    simulated_residuals = simulate_sarima_process(
        n_samples, resids_fit_model, rng)

    # Add the simulated residuals to the original series
    bootstrap_series = X_fitted + simulated_residuals

    return bootstrap_series


def generate_samples_sieve_var(
    X_fitted: np.ndarray,
    resids_fit_model: VARResultsWrapper,
    rng: Optional[Union[Integral, Generator]] = None,
) -> np.ndarray:
    n_samples, n_features = X_fitted.shape

    # Simulate residuals using the SARIMA model
    simulated_residuals = simulate_var_process(
        n_samples, resids_fit_model, rng)

    # Add the simulated residuals to the original series
    bootstrap_series = X_fitted + simulated_residuals

    return bootstrap_series


def generate_samples_sieve_arch(
    X_fitted: np.ndarray,
    resids_fit_model: ARCHModelResult,
    rng: Optional[Union[Integral, Generator]] = None,
) -> np.ndarray:
    n_samples, n_features = X_fitted.shape

    # Simulate residuals using the SARIMA model
    simulated_residuals = simulate_arch_process(
        n_samples, resids_fit_model, rng)

    # Add the simulated residuals to the original series
    bootstrap_series = X_fitted + simulated_residuals

    return bootstrap_series


def generate_samples_sieve(
    X_fitted: np.ndarray,
    model_type: str,
    rng: Optional[Union[Integral, Generator]] = None,
    resids_lags: Optional[Union[int, List[int]]] = None,
    resids_coefs: Optional[np.ndarray] = None,
    resids: Optional[np.ndarray] = None,
    resids_fit_model: Optional[Union[ARIMAResultsWrapper,
                                     SARIMAXResultsWrapper, VARResultsWrapper, ARCHModelResult]] = None,
) -> np.ndarray:
    """
    Generate a bootstrap sample using the sieve bootstrap.
    """
    if model_type not in ['ar', 'arima', 'sarima', 'var', 'arch']:
        raise ValueError(
            "model_type must be one of 'ar', 'arima', 'sarima', 'var', 'arch'.")
    if model_type == 'ar' and (resids_lags is None or resids_coefs is None or resids is None):
        raise ValueError(
            "resids_lags, resids_coefs and resids must be provided for the AR model.")
    if model_type in ['arima', 'sarima', 'var', 'arch'] and resids_fit_model is None:
        raise ValueError(
            "resids_fit_model must be provided for the ARIMA, SARIMA, VAR and ARCH models.")

    if model_type == 'ar':
        return generate_samples_sieve_autoreg(
            X_fitted, resids_lags, resids_coefs, resids, rng)
    elif model_type == 'arima':
        return generate_samples_sieve_arima(
            X_fitted, resids_fit_model, rng)
    elif model_type == 'sarima':
        return generate_samples_sieve_sarima(
            X_fitted, resids_fit_model, rng)
    elif model_type == 'var':
        return generate_samples_sieve_var(
            X_fitted, resids_fit_model, rng)
    elif model_type == 'arch':
        return generate_samples_sieve_arch(
            X_fitted, resids_fit_model, rng)
