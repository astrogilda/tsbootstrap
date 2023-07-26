
from typing import List, Union, Optional
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper
from statsmodels.tsa.arima.model import ARIMAResultsWrapper
from arch.univariate.base import ARCHModelResult

import numpy as np
from numba import njit
from numpy.random import Generator

# from future_work.numba_base import *
from utils.markov_sampler import MarkovSampler
from utils.odds_and_ends import *

# TODO: block_weights=p with block_length=1 should be equivalent to the iid bootstrap

"""
Call the below functions from src.bootstrap.py. These functions have not had their inputs checked.
"""


@njit
def generate_random_indices(num_samples: int, random_seed: Optional[int] = None) -> np.ndarray:
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
    if not (isinstance(num_samples, int) and num_samples > 0):
        raise ValueError("num_samples must be a positive integer.")
    if random_seed is not None and not (isinstance(random_seed, int) and random_seed >= 0):
        raise ValueError("random_seed must be a non-negative integer.")

    # Set the random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate random indices with replacement
    in_bootstrap_indices = np.random.choice(
        np.arange(num_samples), size=num_samples, replace=True)

    return in_bootstrap_indices


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


@njit
def simulate_ar_process(lags: np.ndarray, coefs: np.ndarray, init: np.ndarray, random_seed: int) -> np.ndarray:
    """
    Simulates an Autoregressive (AR) process with given lags, coefficients, initial values, and random errors.

    Args:
        lags (np.ndarray): The lags to be used in the AR process. Can be non-consecutive.
        coefs (np.ndarray): The coefficients corresponding to each lag. Should be the same length as `lags`.
        init (np.ndarray): The initial values for the simulation. Should be at least as long as the maximum lag.
        random_seed (int): The seed for the random number generator.

    Returns:
        np.ndarray: The simulated AR process as a 1D NumPy array.

    Raises:
        ValueError: If `init` is not long enough to cover the maximum lag.
    """
    # Set the random seed
    np.random.seed(random_seed)
    random_errors = np.random.normal(size=n_samples)
    max_lag = np.max(lags)
    assert len(
        init) >= max_lag, "Length of 'init' must be at least as long as the maximum lag in 'lags'"
    n_samples = len(random_errors)
    series = np.zeros(n_samples)
    series[:max_lag] = init

    # Loop through the series, calculating each value based on the lagged values, coefficients, and random error
    for t in range(max_lag, n_samples):
        ar_term = 0
        for i in range(len(lags)):
            ar_term += coefs[i] * series[t - lags[i]]
        series[t] = ar_term + random_errors[t]

    return series


def simulate_arima_process(n_samples: int, fitted_model: ARIMAResultsWrapper, random_seed: int) -> np.ndarray:
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
    np.random.seed(random_seed)
    simulated_series = fitted_model.simulate(nsimulations=n_samples)
    return simulated_series


def simulate_sarima_process(n_samples: int, fitted_model: SARIMAXResultsWrapper, random_seed: int) -> np.ndarray:
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
    np.random.seed(random_seed)
    simulated_series = fitted_model.simulate(nsimulations=n_samples)
    return simulated_series


def simulate_var_process(n_samples: int, fitted_model: VARResultsWrapper, random_seed: int) -> np.ndarray:
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
    np.random.seed(random_seed)
    simulated_series = fitted_model.simulate_var(steps=n_samples)
    return simulated_series


# TODO: review this function -- definitely more complex than this. for instance, there is an initial_value and a burn argument that are not used here
def simulate_arch_process(n_samples: int, fitted_model: ARCHModelResult, random_seed: int) -> np.ndarray:
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
    np.random.seed(random_seed)
    simulated_data = fitted_model.model.simulate(
        params=fitted_model.params, nobs=n_samples)
    return simulated_data['data'].values


def generate_samples_sieve_autoreg(
    X_fitted: np.ndarray,
    resids_lags: Union[int, List[int]],
    resids_coefs: np.ndarray,
    resids: np.ndarray,
    random_seed: int,
) -> np.ndarray:

    n_samples, n_features = X_fitted.shape

    # Generate the bootstrap series
    bootstrap_series = np.zeros((n_samples, n_features), dtype=np.float64)

    max_lag = max(resids_lags) if isinstance(
        resids_lags, list) else resids_lags
    simulated_residuals = simulate_ar_process(
        resids_lags, resids_coefs, resids[:max_lag], random_seed)

    bootstrap_series[:max_lag] = X_fitted[:max_lag]
    for t in range(max_lag, n_samples):
        lagged_values = bootstrap_series[t - np.array(resids_lags)]
        bootstrap_series[t] = resids_coefs @ lagged_values.T + \
            simulated_residuals[t]

    return bootstrap_series


def generate_samples_sieve_arima(
    X_fitted: np.ndarray,
    resids_fit_model: ARIMAResultsWrapper,
    random_seed: int,
) -> np.ndarray:
    n_samples, n_features = X_fitted.shape

    # Simulate residuals using the ARIMA model
    simulated_residuals = simulate_arima_process(
        n_samples, resids_fit_model, random_seed)

    # Add the simulated residuals to the original series
    bootstrap_series = X_fitted + simulated_residuals

    return bootstrap_series


def generate_samples_sieve_sarima(
    X_fitted: np.ndarray,
    resids_fit_model: SARIMAXResultsWrapper,
    random_seed: int,
) -> np.ndarray:
    n_samples, n_features = X_fitted.shape

    # Simulate residuals using the SARIMA model
    simulated_residuals = simulate_sarima_process(
        n_samples, resids_fit_model, random_seed)

    # Add the simulated residuals to the original series
    bootstrap_series = X_fitted + simulated_residuals

    return bootstrap_series


def generate_samples_sieve_var(
    X_fitted: np.ndarray,
    resids_fit_model: VARResultsWrapper,
    random_seed: int,
) -> np.ndarray:
    n_samples, n_features = X_fitted.shape

    # Simulate residuals using the SARIMA model
    simulated_residuals = simulate_var_process(
        n_samples, resids_fit_model, random_seed)

    # Add the simulated residuals to the original series
    bootstrap_series = X_fitted + simulated_residuals

    return bootstrap_series


def generate_samples_sieve_arch(
    X_fitted: np.ndarray,
    resids_fit_model: ARCHModelResult,
    random_seed: int,
) -> np.ndarray:
    n_samples, n_features = X_fitted.shape

    # Simulate residuals using the SARIMA model
    simulated_residuals = simulate_arch_process(
        n_samples, resids_fit_model, random_seed)

    # Add the simulated residuals to the original series
    bootstrap_series = X_fitted + simulated_residuals

    return bootstrap_series


def generate_samples_sieve(
    X_fitted: np.ndarray,
    random_seed: int,
    model_type: str,
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
            X_fitted, resids_lags, resids_coefs, resids, random_seed)
    elif model_type == 'arima':
        return generate_samples_sieve_arima(
            X_fitted, resids_fit_model, random_seed)
    elif model_type == 'sarima':
        return generate_samples_sieve_sarima(
            X_fitted, resids_fit_model, random_seed)
    elif model_type == 'var':
        return generate_samples_sieve_var(
            X_fitted, resids_fit_model, random_seed)
    elif model_type == 'arch':
        return generate_samples_sieve_arch(
            X_fitted, resids_fit_model, random_seed)
