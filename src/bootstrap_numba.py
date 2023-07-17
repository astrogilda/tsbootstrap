
from numba.core.errors import TypingError
from typing import List, Callable, Union, Tuple, Optional
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper
from statsmodels.tsa.arima.model import ARIMAResultsWrapper
from arch.univariate.base import ARCHModelResult

import numpy as np
from numba import njit

from utils.block_length_sampler import BlockLengthSampler
from future_work.numba_base import *
from utils.markov_sampler import MarkovSampler
from utils.odds_and_ends import *
from utils.validate import validate_weights

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


def _prepare_block_weights(block_weights: Optional[Union[np.ndarray, Callable]], X: np.ndarray) -> np.ndarray:
    """
    Prepare the block_weights array by normalizing it or generating it
    based on the callable function provided.

    Parameters
    ----------
    block_weights : Union[np.ndarray, Callable]
        An array of weights or a callable function to generate weights.
    X : np.ndarray
        Input data array.

    Returns
    -------
    np.ndarray
        An array of normalized block_weights.
    """

    size = X.shape[0]

    if callable(block_weights):
        X_copy = X.copy()
        try:
            block_weights_jitted = njit(block_weights)
            block_weights_arr = block_weights_jitted(X_copy)
        except TypingError:
            block_weights_arr = block_weights(X_copy)
        if not np.array_equal(X, X_copy):
            raise ValueError(
                "'block_weights' function must not have side effects")

    elif isinstance(block_weights, np.ndarray):
        if block_weights.shape[0] == 0:
            block_weights_arr = np.full((size, 1), 1 / size)
        else:
            if block_weights.shape[0] != X.shape[0]:
                raise ValueError(
                    "block_weights array must have the same size as X")
            block_weights_arr = block_weights

    elif block_weights is None:
        block_weights_arr = np.full((size, 1), 1 / size)

    else:
        raise TypeError(
            "'block_weights' must be a numpy array or a callable function")

    # Validate the block_weights array
    validate_weights(block_weights_arr)
    # Normalize the block_weights array
    block_weights_arr = normalize_array(block_weights_arr)

    return block_weights_arr


def _prepare_tapered_weights(tapered_weights: Optional[Union[np.ndarray, Callable]], block_length: int) -> np.ndarray:
    """
    Prepare the tapered_weights array by normalizing it or generating it
    based on the callable function provided.

    Parameters
    ----------
    tapered_weights : Union[np.ndarray, Callable]
        An array of weights or a callable function to generate weights.
    block_length : int
        Length of each block.

    Returns
    -------
    np.ndarray
        An array of normalized tapered_weights.
    """

    # Check if 'block_length' is a positive integer
    if not (isinstance(block_length, int) and block_length > 0):
        raise ValueError("block_length must be a positive integer.")

    if callable(tapered_weights):
        try:
            tapered_weights_jitted = njit(tapered_weights)
            tapered_weights_arr = tapered_weights_jitted(block_length)
        except TypingError:
            tapered_weights_arr = tapered_weights(block_length)

    elif isinstance(tapered_weights, np.ndarray):
        if tapered_weights.size == 0:
            tapered_weights_arr = np.full((block_length, 1), 1 / block_length)
        else:
            if tapered_weights.size != block_length:
                raise ValueError(
                    "tapered_weights array must have the same size as block_length")
            tapered_weights_arr = tapered_weights

    elif tapered_weights is None:
        tapered_weights_arr = np.full((block_length, 1), 1 / block_length)

    else:
        raise TypeError(
            "'tapered_weights' must be a numpy array or a callable function")

    # Validate the tapered_weights array
    validate_weights(tapered_weights_arr)
    # Normalize the tapered_weights array
    tapered_weights_arr = normalize_array(tapered_weights_arr)

    return tapered_weights_arr


@njit
def resample_blocks(blocks: List[np.ndarray], n: int, block_weights: np.ndarray, random_seed: int) -> List[np.ndarray]:
    """
    Resamples blocks with replacement to create a new list of blocks with total length equal to n.

    Parameters
    ----------
    blocks : list of 1d numpy arrays
        Blocks to be resampled. Each block represents a collection of unique index positions.
    n : int
        The total number of samples in the newly generated list of blocks.
    block_weights : np.ndarray
        2D array of probabilities for each unique element being the first element of a block.

    Returns
    -------
    list of 1d numpy arrays
        The newly generated list of blocks with total length equal to n.
    """
    # Set the random seed
    np.random.seed(random_seed)
    new_blocks = []
    total_samples = 0
    # Create a dictionary mapping first indices to their blocks
    block_dict = {block[0]: block for block in blocks}
    # Get the first indices and their weights
    first_indices = list(block_dict.keys())
    while total_samples < n:
        # Filter out the blocks that are too large or have zero weight
        eligible_indices = [index for index in first_indices if len(
            block_dict[index]) <= n - total_samples and block_weights[index] > 0]
        # If there are no eligible complete blocks
        if len(eligible_indices) == 0:
            # Get the incomplete eligible blocks
            incomplete_eligible_indices = [index for index in first_indices if len(
                block_dict[index]) > 0 and block_weights[index] > 0]
            # Get the weights of the incomplete eligible indices
            incomplete_eligible_weights = np.array(
                [block_weights[index] for index in incomplete_eligible_indices])
            # Select an index based on the provided weights
            index = choice_with_p(incomplete_eligible_weights)
            # Find the block that starts with the selected index
            selected_block = block_dict[incomplete_eligible_indices[index]]
            # Add the first n - total_samples samples from the selected block
            new_blocks.append(selected_block[:n - total_samples])
            break
        # Get the weights of the eligible indices
        eligible_weights = np.array([block_weights[index]
                                    for index in eligible_indices])
        # Select an index based on the provided weights
        index = choice_with_p(eligible_weights)
        # Find the block that starts with the selected index
        selected_block = block_dict[eligible_indices[index]]
        new_blocks.append(selected_block)
        total_samples += len(selected_block)
    return new_blocks


def generate_block_indices_and_data(X: np.ndarray, block_length: int, blocks: List[np.ndarray], block_weights: Optional[Union[np.ndarray, Callable]] = None, tapered_weights: Optional[Union[np.ndarray, Callable]] = None, random_seed: int = 42) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate block indices and corresponding data for the input data array X.

    Parameters
    ----------
    X : np.ndarray
        Input data array.
    block_length : int
        Length of each block.
    block_weights : Union[np.ndarray, Callable], optional
        An array of weights or a callable function to generate weights.
    tapered_weights : Union[np.ndarray, Callable], optional
        An array of weights to apply to the data within the blocks.
    overlap_flag : bool, optional
        Whether to allow overlapping blocks, by default True.
    wrap_around_flag :bool, optional
        Whether to allow wrap-around in the block sampling, by default False.
    random_seed : int, optional
        Random seed for reproducibility, by default 42.

    Other Parameters
    ----------------
    block_length_distribution : str, optional
        The block length distribution function to use, represented by its name as a string, by default "none".
    overlap_length : int, optional
        The length of overlap between consecutive blocks, by default 1.
    min_block_length : int, optional
        The minimum length of a block, by default 1.

    Returns
    -------
    Tuple[List[np.ndarray], List[np.ndarray]]
        A tuple containing a list of block indices and a list of corresponding
        modified data blocks.

    Notes
    -----
    The block indices are generated using the following steps:
    1. Generate block weights using the block_weights argument.
    2. Generate block indices using the block_weights and block_length arguments.
    3. Apply tapered_weights to the data within the blocks if provided.
    """

    n = X.shape[0]

    # if wrap_around_flag:
    #    n += block_length - 1

    # Prepare the block_weights array
    block_weights = _prepare_block_weights(block_weights, X)
    # Resample blocks with replacement to create a new list of blocks with total length equal to n
    resampled_block_indices = resample_blocks(
        block_blocks=blocks, n=n, block_weights=block_weights, random_seed=random_seed)

    # Apply tapered_weights to the data within the blocks if provided
    tapered_weights = _prepare_tapered_weights(tapered_weights, block_length)
    modified_blocks = [X[block] *
                       tapered_weights for block in resampled_block_indices]
    return resampled_block_indices, modified_blocks


def generate_samples_markov(blocks: List[np.ndarray], method: str, block_length: int, n_clusters: int, random_seed: int, **kwargs) -> np.ndarray:
    """
    Generate a bootstrapped time series based on the Markov chain bootstrapping method.

    Parameters
    ----------
    blocks : List[np.ndarray]
        A list of numpy arrays representing the original time series blocks. The last block may have fewer samples than block_length.
    method : str
        The method to be used for block summarization.
    block_length : int
        The number of samples in each block, except possibly for the last block.
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
    kmedians_max_iter : int, optional
        Maximum number of iterations for K-Medians, by default 300.
    n_iter_hmm : int, optional
        Number of iterations for the HMM model, by default 100.
    n_fits_hmm : int, optional
        Number of fits for the HMM model, by default 10.

    Returns
    -------
    np.ndarray
        A numpy array representing the bootstrapped time series.

    """
    total_length = sum(block.shape[0] for block in blocks)

    transmat_init = MarkovSampler.calculate_transition_probabilities(
        blocks=blocks)
    blocks_summarized = MarkovSampler.summarize_blocks(
        blocks=blocks, method=method,
        apply_pca=kwargs.get('apply_pca', False),
        pca=kwargs.get('pca', None),
        kmedians_max_iter=kwargs.get('kmedians_max_iter', 300),
        random_seed=random_seed)
    fit_hmm_model = MarkovSampler.fit_hidden_markov_model(
        blocks_summarized=blocks_summarized,
        n_states=n_clusters,
        random_seed=random_seed,
        transmat_init=transmat_init,
        n_iter_hmm=kwargs.get('n_iter_hmm', 100),
        n_fits_hmm=kwargs.get('n_fits_hmm', 10)
    )
    transition_probabilities, cluster_centers, cluster_covars, cluster_assignments = MarkovSampler.get_cluster_transitions_centers_assignments(
        blocks_summarized=blocks_summarized,
        hmm_model=fit_hmm_model,
        transmat_init=transmat_init)

    # Initialize the random number generator
    rng = np.random.default_rng(seed=random_seed)

    # Choose a random starting block from the original blocks
    start_block_idx = 0
    start_block = blocks[start_block_idx]

    # Initialize the bootstrapped time series with the starting block
    bootstrapped_series = start_block.copy()

    # Get the state of the starting block
    current_state = cluster_assignments[start_block_idx]

    # Generate synthetic blocks and concatenate them to the bootstrapped time series until it matches the total length
    while bootstrapped_series.shape[0] < total_length:
        # Predict the next block's state using the HMM model
        next_state = rng.choice(
            n_clusters, p=transition_probabilities[current_state])

        # Determine the length of the synthetic block
        synthetic_block_length = block_length if bootstrapped_series.shape[0] + \
            block_length <= total_length else total_length - bootstrapped_series.shape[0]

        # Generate a synthetic block corresponding to the predicted state
        synthetic_block_mean = cluster_centers[next_state]
        synthetic_block_cov = cluster_covars[next_state]
        synthetic_block = rng.multivariate_normal(
            synthetic_block_mean, synthetic_block_cov, size=synthetic_block_length)

        # Concatenate the generated synthetic block to the bootstrapped time series
        bootstrapped_series = np.vstack((bootstrapped_series, synthetic_block))

        # Update the current state
        current_state = next_state

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
