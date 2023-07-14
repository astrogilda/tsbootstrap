
from typing import List, Callable, Union, Tuple, Optional
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper
from statsmodels.tsa.arima.model import ARIMAResultsWrapper
from arch.univariate.base import ARCHModelResult

import numpy as np
from numba import njit
from sklearn.utils import check_random_state

from utils.block_length_sampler import BlockLengthSampler
from utils.numba_base import *
from utils.markov_sampler import MarkovSampler

# TODO: block_weights=p with block_length=1 should be equivalent to the iid bootstrap

"""
Call the below functions from src.bootstrap.py. These functions have not had their inputs checked.
"""


@njit
def generate_indices_random(num_samples: int, random_seed: int) -> np.ndarray:
    np.random.seed(random_seed)
    in_bootstrap_indices = np.random.choice(
        np.arange(num_samples), size=num_samples, replace=True)
    return in_bootstrap_indices


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
    return (array / np.sum(array)).reshape(-1, 1)


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
    if is_callable(block_weights):
        if not is_numba_compiled(block_weights):
            block_weights = njit(block_weights)
        block_weights_arr = block_weights(X)
    else:
        if block_weights.size == 0:
            block_weights_arr = np.full((size, 1), 1 / size)
        else:
            assert block_weights.size == X.shape[0], "block_weights array must have the same size as X"
            block_weights_arr = block_weights

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
    if is_callable(tapered_weights):
        if not is_numba_compiled(tapered_weights):
            tapered_weights = njit(tapered_weights)
        tapered_weights_arr = tapered_weights(block_length)
    else:
        if tapered_weights.size == 0:
            tapered_weights_arr = np.full((block_length, 1), 1 / block_length)
        else:
            assert tapered_weights.size == block_length, "tapered_weights array must have the same size as block_length"
            tapered_weights_arr = tapered_weights

    tapered_weights_arr = normalize_array(tapered_weights_arr)
    return tapered_weights_arr


@njit
def _generate_non_overlapping_indices(n: int, block_length_sampler: BlockLengthSampler, block_weights: np.ndarray, wrap_around_flag: bool) -> List[np.ndarray]:
    """
    Generate block indices for the non-overlapping case.

    Parameters
    ----------
    n : int
        The length of the input data array.
    block_length : int
        Length of each block.
    block_weights : np.ndarray
        An array of normalized block_weights.
    wrap_around_flag : bool
        Whether to allow wrap-around in the block sampling.

    Returns
    -------
    List[np.ndarray]
        A list of block indices for the non-overlapping case.
    """
    block_indices = []
    total_elements_covered = 0
    while total_elements_covered < n:
        block_length = min(
            block_length_sampler.sample_block_length(), n - total_elements_covered)
        block_starts = np.arange(
            total_elements_covered, total_elements_covered + block_length)

        if wrap_around_flag:
            if total_elements_covered + block_length > n:
                wrap_around_point = (total_elements_covered + block_length) % n
                adjusted_block_weights = np.concatenate(
                    (block_weights[block_starts], [(block_weights[-1] + block_weights[wrap_around_point]) / 2]))
            else:
                adjusted_block_weights = block_weights[block_starts]
        else:
            adjusted_block_weights = block_weights[block_starts]

        sampled_block_start = choice_with_p(adjusted_block_weights)
        block_indices.append(
            np.arange(sampled_block_start, (sampled_block_start + block_length) % n))
        total_elements_covered += block_length
    return block_indices


@njit
def _generate_overlapping_indices(n: int, block_length_sampler: BlockLengthSampler, block_weights: np.ndarray, wrap_around_flag: bool) -> List[np.ndarray]:
    """
    Generate block indices for the overlapping case.

    Parameters
    ----------
    n : int
        The length of the input data array.
    block_length : int
        Length of each block.
    block_weights : np.ndarray
        An array of normalized block_weights.

    Returns
    -------
    List[np.ndarray]
        A list of block indices for the overlapping case.
    """
    block_indices = []
    total_elements_covered = 0
    while total_elements_covered < n:
        block_length = min(
            block_length_sampler.sample_block_length(), n - total_elements_covered)
        if wrap_around_flag:
            block_starts = np.arange(
                total_elements_covered, total_elements_covered + block_length) % n
        else:
            block_starts = np.arange(total_elements_covered, min(
                total_elements_covered + block_length, n))

        sampled_block_start = choice_with_p(block_weights[block_starts])
        block_indices_iter = np.arange(sampled_block_start, min(
            sampled_block_start + block_length, n)) % (n if wrap_around_flag else 1)
        block_indices_iter = np.reshape(
            block_indices_iter, (block_indices_iter.size, 1))
        block_indices.append(block_indices_iter)
        total_elements_covered += block_length
    return block_indices


def generate_block_indices_and_data(X: np.ndarray, block_length: int, block_weights: Optional[Union[np.ndarray, Callable]] = None, tapered_weights: Optional[Union[np.ndarray, Callable]] = None, block_length_distribution: str = 'none', overlap_flag: bool = True, wrap_around_flag: bool = False, random_seed: int = 42) -> Tuple[List[np.ndarray], List[np.ndarray]]:
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

    np.random.seed(random_seed)
    n = X.shape[0]

    if wrap_around_flag:
        n += block_length - 1

    block_length_sampler = BlockLengthSampler(
        block_length_distribution=block_length_distribution, avg_block_length=block_length)

    block_weights = _prepare_block_weights(block_weights, X)

    if not overlap_flag:
        block_indices = _generate_non_overlapping_indices(
            n, block_length_sampler, block_weights, wrap_around_flag)
    else:
        block_indices = _generate_overlapping_indices(
            n, block_length_sampler, block_weights, wrap_around_flag)

    # Apply tapered_weights to the data within the blocks if provided
    tapered_weights = _prepare_tapered_weights(tapered_weights, block_length)
    modified_blocks = [X[block] * tapered_weights for block in block_indices]
    return block_indices, modified_blocks


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


"""
 this is a somewhat simplified version of the spectral bootstrap, and more advanced versions could involve operations such as adjusting the amplitude and phase of the FFT components, filtering, or other forms of spectral manipulation. Also, this version of spectral bootstrap will not work with signals that have frequency components exceeding half of the sampling frequency (Nyquist frequency) due to aliasing.
"""


@njit
def generate_block_indices_spectral(X: np.ndarray, block_length: int, random_seed: int) -> List[np.ndarray]:
    np.random.seed(random_seed)
    n = X.shape[0]
    num_blocks = int(np.ceil(n / block_length))

    # Generate Fourier frequencies
    freqs = rfftfreq_numba(n)
    freq_indices = np.arange(len(freqs))

    # Sample frequencies with replacement
    sampled_freqs = np.random.choice(freq_indices, num_blocks, replace=True)

    # Generate blocks using the sampled frequencies
    block_indices = []
    for freq in sampled_freqs:
        time_indices = np.where(freqs == freq)[0]
        if time_indices.size > 0:  # Check if time_indices is not empty
            start = np.random.choice(time_indices)
            block = np.arange(start, min(start + block_length, n))
            block_indices.append(block)

    return block_indices


'''
import numpy as np
from numba import njit, prange
from typing import List, Optional, Union
import pyfftw.interfaces

@njit
def rfftfreq_numba(n: int, d: float = 1.0) -> np.ndarray:
    """Compute the one-dimensional n-point discrete Fourier Transform sample frequencies.
    This is a Numba-compatible implementation of numpy.fft.rfftfreq.
    """
    val = 1.0 / (n * d)
    N = n // 2 + 1
    results = np.arange(0, N, dtype=np.float64)
    return results * val

@njit
def generate_block_indices_spectral(
    X: np.ndarray,
    block_length: int,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    amplitude_adjustment: bool = False,
    phase_randomization: bool = False
) -> List[np.ndarray]:
    if random_state is None:
        random_state = np.random.default_rng()
    elif isinstance(random_state, int):
        random_state = np.random.default_rng(random_state)

    n = X.shape[0]
    num_blocks = n // block_length

    # Compute the FFT of the input signal X
    X_freq = pyfftw.interfaces.numpy_fft.rfft(X)

    # Generate Fourier frequencies
    freqs = rfftfreq_numba(n)
    freq_indices = np.arange(len(freqs))

    # Filter out frequency components above the Nyquist frequency
    nyquist_index = n // 2
    freq_indices = freq_indices[:nyquist_index]

    # Sample frequencies with replacement
    sampled_freqs = random_state.choice(freq_indices, num_blocks, replace=True)

    # Generate blocks using the sampled frequencies
    block_indices = []
    for freq in prange(num_blocks):
        time_indices = np.where(freqs == sampled_freqs[freq])[0]
        if time_indices.size > 0:  # Check if time_indices is not empty
            start = random_state.choice(time_indices)
            block = np.arange(start, start + block_length)

            # Amplitude adjustment
            if amplitude_adjustment:
                block_amplitude = np.abs(X_freq[block])
                X_freq[block] *= random_state.uniform(0, 2, size=block_amplitude.shape) * block_amplitude

            # Phase randomization
            if phase_randomization:
                random_phase = random_state.uniform(0, 2 * np.pi, size=X_freq[block].shape)
                X_freq[block] *= np.exp(1j * random_phase)

            block_indices.append(block)

    return block_indices
'''


@njit
def cholesky_numba(A):
    return np.linalg.cholesky(A)


@njit
def generate_har_decomposition(X: np.ndarray, bandwidth: int, lambda_value: float = 1e-6) -> np.ndarray:
    """
    Compute the Cholesky decomposition of the HAR covariance matrix.

    This function first computes the HAR covariance matrix for a given time
    series X and bandwidth h. Then, it regularizes the covariance matrix by
    adding a small multiple of the identity matrix. Finally, it calculates
    the Cholesky decomposition of the regularized covariance matrix.

    Parameters
    ----------
    X : np.ndarray
        The input data array (time series).
    bandwidth : int
        The number of lags to consider in the autocovariance estimation.
    lambda_value : float, optional
        The regularization parameter (default is 1e-6).

    Returns
    -------
    np.ndarray
        The lower triangular matrix obtained from the Cholesky decomposition.
    """
    # Use the HAR estimator to compute the long-run covariance matrix
    long_run_cov = har_cov(X, bandwidth)

    # Regularize the covariance matrix
    regularized_matrix = long_run_cov + lambda_value * \
        np.identity(long_run_cov.shape[0])

    # Calculate the Cholesky decomposition of the regularized long-run covariance matrix
    cholesky_decomposition = cholesky_numba(regularized_matrix, lower=True)

    return cholesky_decomposition


@njit
def generate_har_errors(X: np.ndarray, cholesky_decomposition: np.ndarray, random_seed: int) -> np.ndarray:
    """
    Generate bootstrapped errors using the Cholesky decomposition.

    Parameters
    ----------
    X : np.ndarray
        The input data array (time series).
    cholesky_decomposition : np.ndarray
        The lower triangular matrix obtained from Cholesky decomposition.
    random_seed : int
        The seed for the random number generator.

    Returns
    -------
    np.ndarray
        The bootstrapped errors.
    """
    np.random.seed(random_seed)
    # Generate the bootstrapped errors
    normal_errors = np.random.randn(X.shape[0], X.shape[1])
    bootstrapped_errors = normal_errors @ cholesky_decomposition.T

    return bootstrapped_errors


'''
# TODO: ensure that this function works for autoreg, arima, sarima, var, and arch models
# TODO: ensure that it works when lag_order is a List of ints
@njit
def generate_samples_residual(X: np.ndarray, X_fitted: np.ndarray, residuals: np.ndarray, lag: Union[int, List[int]], model_type: str, random_seed: int) -> np.ndarray:
    # Resample residuals
    resampled_indices = generate_indices_random(
        residuals.shape[0], random_seed)
    resampled_residuals = residuals[resampled_indices]
    # Add the bootstrapped residuals to the fitted values
    bootstrapped_X = X_fitted + resampled_residuals
    if model_type in ['ar', 'var']:
        # For AutoReg models, `lag` can possibly be a list of ints
        lag = np.max(lag)
        # Prepend the first 'lag_order' original observations to the bootstrapped series
        extended_bootstrapped_X = np.vstack((X[:lag], bootstrapped_X))
        # Prepend the indices of the first 'lag_order' original observations to resampled_indices
        initial_indices = np.arange(lag, dtype=np.int64)
        extended_resampled_indices = np.hstack(
            (initial_indices, resampled_indices + lag))
        return extended_resampled_indices, extended_bootstrapped_X
    else:
        return resampled_indices, bootstrapped_X
'''


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


'''
@njit
def generate_samples_fractional(
    X: np.ndarray,
    block_length: int,
    random_seed: int,
) -> np.ndarray:
    np.random.seed(random_seed)

    # Construct the bootstrapped series
    bootstrap_series = np.empty((0, X.shape[1]))
    for block in block_indices:
        block_data = X_diff[block]
        bootstrap_series = np.vstack((bootstrap_series, block_data))

    # Truncate the bootstrapped series if necessary
    if bootstrap_series.shape[0] > X_diff.shape[0]:
        bootstrap_series = bootstrap_series[: X_diff.shape[0], :]

    # Add back the fractional differences
    bootstrap_series = np.cumsum(bootstrap_series, axis=0)

    return bootstrap_series
'''

'''
@njit
def generate_boolean_mask(X: np.ndarray, random_seed: int) -> np.ndarray:
    np.random.seed(random_seed)
    residuals = X - np.mean(X)
    num_samples = X.shape[0]
    boolean_mask = np.empty(num_samples, dtype=np.bool_)
    for i in range(num_samples):
        boolean_mask[i] = residuals[np.random.randint(
            residuals.shape[0])].item() >= 0
    return boolean_mask



@njit
def generate_bayesian_bootstrap_samples(X: np.ndarray, random_seed: int) -> np.ndarray:
    np.random.seed(random_seed)
    alpha = np.ones(X.shape[0], dtype=np.float64)
    weights = dirichlet_numba(alpha)

    # Initialize the bootstrap sample array
    bootstrap_samples = np.zeros_like(X)

    for i in range(X.shape[0]):
        bootstrap_samples[i] = X[i] * weights[i]

    return bootstrap_samples



@njit
def generate_test_mask_bayesian(X: np.ndarray, random_seed: int) -> np.ndarray:
    alpha = np.ones(X.shape[0], dtype=np.float64)
    weights = dirichlet_numba(alpha, random_seed)
    X_weighted = X * weights[:, np.newaxis]

    mask = np.empty(X.shape[0], dtype=np.bool_)

    for i in range(X.shape[0]):
        mask[i] = np.all(X_weighted[i] == X[i])

    return mask



@njit
def generate_test_mask_bayesian_block(X: np.ndarray, block_length: int, random_seed: int) -> np.ndarray:
    np.random.seed(random_seed)
    num_blocks = X.shape[0] // block_length
    X_blocks = X.reshape(num_blocks, block_length, -1)

    alpha = np.ones(num_blocks, dtype=np.float64)
    weights = dirichlet_numba(alpha, random_seed)

    X_weighted_blocks = X_blocks * weights[:, np.newaxis, np.newaxis]

    mask = np.empty_like(X, dtype=np.bool_)

    for block_i in range(num_blocks):
        mask[block_i * block_length:(block_i + 1) * block_length] = np.all(
            X_weighted_blocks[block_i] == X_blocks[block_i], axis=1)

    return mask


@njit
def generate_test_mask_subsampling(X: np.ndarray, sample_fraction: float, random_seed: int) -> np.ndarray:
    np.random.seed(random_seed)
    num_samples = X.shape[0]
    sample_size = int(sample_fraction * num_samples)
    sampled_indices = np.random.choice(
        num_samples, size=sample_size, replace=False)
    test_mask = np.full(num_samples, False, dtype=np.bool_)
    for idx in sampled_indices:
        test_mask[idx] = True
    return test_mask


@njit
def generate_test_mask_poisson_bootstrap(X: np.ndarray, random_seed: int) -> np.ndarray:
    np.random.seed(random_seed)
    sampled_indices = np.random.poisson(lam=1, size=X.shape[0])
    test_mask = np.zeros(X.shape[0], dtype=np.bool_)
    test_mask[sampled_indices] = True
    return test_mask


def generate_test_mask_polynomial_fit_bootstrap(X: np.ndarray, degree: int, random_seed: int) -> np.ndarray:
    np.random.seed(random_seed)
    x = np.arange(X.shape[0]).reshape(-1, 1)
    poly = PolynomialFeatures(degree=degree)
    x_poly = poly.fit_transform(x)
    model = LinearRegression().fit(x_poly, X)
    fit = model.predict(x_poly)
    residuals = X - fit

    bootstrapped_residuals = np.empty_like(residuals)
    for i in range(residuals.shape[0]):
        bootstrapped_residuals[i] = np.random.choice(
            residuals, size=residuals.shape[1], replace=True)
    bootstrapped_series = fit + bootstrapped_residuals

    test_mask = np.zeros_like(X, dtype=np.bool_)
    for i in range(X.shape[0]):
        test_mask[i] = np.any(np.all(bootstrapped_series == X[i], axis=1))
    return test_mask




@njit
def generate_test_mask_banded(X: np.ndarray, band_width: int, random_seed: int) -> np.ndarray:
    np.random.seed(random_seed)
    banded_matrix = hankel_numba(X[:band_width], X[band_width - 1:])
    sampled_rows = np.random.choice(
        len(banded_matrix), size=len(banded_matrix), replace=True)
    bootstrap_series = banded_matrix[sampled_rows].flatten()

    test_mask = np.zeros((X.shape[0],), dtype=np.bool_)
    for i in range(bootstrap_series.shape[0]):
        pos = np.nonzero(np.all(X == bootstrap_series[i], axis=1))[0]
        if pos.size > 0:
            test_mask[pos[0]] = True
    return test_mask



#########
#########


def generate_test_mask_weibull(X: np.ndarray, random_seed: int) -> np.ndarray:
    np.random.seed(random_seed)
    shape, scale = scipy.stats.weibull_min.fit(X)
    return np.random.weibull(shape, size=len(X)) * scale


def generate_test_mask_gamma(X: np.ndarray, random_seed: int) -> np.ndarray:
    np.random.seed(random_seed)
    shape, loc, scale = scipy.stats.gamma.fit(X)
    return np.random.gamma(shape, scale, size=len(X))

@njit
def generate_test_mask_exponential(X: np.ndarray, random_seed: int) -> np.ndarray:
    np.random.seed(random_seed)
    rate = 1 / np.mean(X)
    return np.random.exponential(scale=1/rate, size=len(X))


@njit
def generate_test_mask_double(X: np.ndarray, random_seed: int) -> np.ndarray:
    np.random.seed(random_seed)
    # Generate first bootstrap sample
    sample1 = np.random.choice(X, size=len(X), replace=True)
    # Generate second bootstrap sample from the first
    return np.random.choice(sample1, size=len(X), replace=True)


@njit
def generate_test_mask_trimmed(X: np.ndarray, percentile: float, random_seed: int) -> np.ndarray:
    np.random.seed(random_seed)
    # Trim the data by removing outliers
    X_trimmed = trimboth_numba(X, percentile)
    return np.random.choice(X_trimmed, size=len(X), replace=True)


#####
#####

     

@njit
def generate_block_indices_nonoverlapping(X: np.ndarray, block_length: int, random_seed: int) -> List[np.ndarray]:
    np.random.seed(random_seed)
    n = X.shape[0]
    num_full_blocks = n // block_length
    remainder = n % block_length

    block_starts = np.arange(0, n - block_length + 1, block_length)
    sampled_block_starts = np.random.choice(
        block_starts, num_full_blocks, replace=True)
    block_indices = [np.arange(start, start + block_length)
                     for start in sampled_block_starts]

    if remainder > 0:
        # Add the remaining data as an additional block
        block_indices.append(np.arange(n - remainder, n))

    return block_indices


@njit
def generate_block_indices_moving_weighted(X: np.ndarray, block_length: int, weights: np.ndarray = None, random_seed: int = 42) -> List[np.ndarray]:
    n = X.shape[0] - block_length + 1
    block_indices = []
    if weights.size == 0:
        weights = np.full(n, 1 / n)
    sampled_block_starts = choice_with_p(
        weights, size=n, random_seed=random_seed)
    block_indices = [np.arange(start, start + block_length)
                     for start in sampled_block_starts]
    return block_indices


@njit
def generate_block_indices_circular(X: np.ndarray, block_length: int, random_seed: int) -> List[np.ndarray]:
    np.random.seed(random_seed)
    n = X.shape[0]
    block_start_indices = np.random.randint(0, X.shape[0], n)
    # Can start block from anywhere in series
    block_indices = [(np.arange(start, start + block_length) % X.shape[0])
                     for start in block_start_indices]
    # Wraparound if block crosses series boundary
    return block_indices



@njit
def generate_block_indices_randomblocklength(X: np.ndarray, block_length: int, random_seed: int) -> List[np.ndarray]:
    np.random.seed(random_seed)
    # The average block length determines the parameter for geometric distribution
    avg_block_length = block_length
    p = 1.0 / avg_block_length
    # Initiate empty list to store the blocks
    block_indices = []
    total_elements_covered = 0
    # Track total elements covered
    while total_elements_covered < len(X):
        # Block length is a random draw from a geometric distribution
        # To prevent block length from being too long, we take the minimum of the random draw and the remaining length of the series. This is because the geometric distribution is unbounded. Helps prevent memory errors.
        block_length = min(np.random.geometric(
            p), len(X) - total_elements_covered)
        # Ensure the block does not cross series boundary
        start_index = np.random.randint(len(X) - block_length + 1)
        # Block ends where it begins + block length
        end_index = start_index + block_length
        # Create indices without wraparound
        indices = np.arange(start_index, end_index)
        # Append block indices
        block_indices.append(indices)
        # Update the total number of elements covered
        total_elements_covered += block_length
    return block_indices


# geometric distribution of block lengths, block cannot wrap around, overlap between blocks possible
@njit
def generate_block_indices_stationary(X: np.ndarray, block_length: int, random_seed: int) -> List[np.ndarray]:
    np.random.seed(random_seed)
    # The average block length determines the parameter for geometric distribution
    avg_block_length = block_length
    p = 1.0 / avg_block_length
    # Initiate empty list to store the blocks
    block_indices = []
    total_elements_covered = 0
    # Track total elements covered
    while total_elements_covered < len(X):
        # Block length is a random draw from a geometric distribution
        # To prevent block length from being too long, we take the minimum of the random draw and the remaining length of the series. This is because the geometric distribution is unbounded. Helps prevent memory errors.
        block_length = min(np.random.geometric(
            p), len(X) - total_elements_covered)
        # Ensure the block does not cross series boundary
        start_index = np.random.randint(len(X) - block_length + 1)
        # Block ends where it begins + block length
        end_index = start_index + block_length
        # Create indices without wraparound
        indices = np.arange(start_index, end_index)
        # Append block indices
        block_indices.append(indices)
        # Update the total number of elements covered
        total_elements_covered += block_length
    return block_indices

    

# @njit
def diff_inv(series_diff, lag, xi=None):
    """Reverse the differencing operation.
    Args:
        series_diff (np.ndarray): Differenced series.
        lag (int): Order of differencing.
        xi (np.ndarray): Initial values of the original series, of length `lag`.

    Returns:
        np.ndarray: Original series.
    """
    n = len(series_diff)
    series = np.zeros_like(series_diff)
    if xi is not None:
        series[:lag] = xi
    else:
        series[:lag] = series_diff[:lag]

    for i in range(lag, n):
        series[i] = series_diff[i] + series[i - lag]
    return series




def generate_samples_sieve(
    X: np.ndarray,
    lags: Union[int, List[int], Tuple[int, int, int]],
    seasonal_lags: Tuple[int, int, int, int],
    coefs: np.ndarray,
    resids: np.ndarray,
    resids_lags: Union[int, List[int]],
    resids_coefs: np.ndarray,
    model: str,
    fitted_model: Union[AutoRegResultsWrapper, ARIMAResultsWrapper, SARIMAXResultsWrapper, VARResultsWrapper, ARCHModelResult],
    random_seed: int,
    orig_X: np.ndarray
) -> np.ndarray:

    n_samples, n_features = X.shape
    np.random.seed(random_seed)

    # Generate the bootstrap series
    bootstrap_series = np.zeros((n_samples, n_features), dtype=np.float64)

    if model == 'ar':
        max_lag = max(lags)
        bootstrap_series[:max_lag] = X[:max_lag]
        simulated_residuals = simulate_ar_process(
            resids_lags, resids_coefs, resids[:max(resids_lags)], np.random.normal(size=n_samples))
    elif model == 'arima':
        p, d, q = lags
        max_lag = p + d + q
        bootstrap_series[:max_lag] = orig_X[:max_lag]
        simulated_residuals = simulate_arima_process(n_samples, fitted_model)
        simulated_residuals = diff_inv(simulated_residuals, d, xi=orig_X[:d])
    elif model == 'sarima':
        if seasonal_lags is None:
            raise ValueError(f"SARIMA model requires 'seasonal_lags' argument")
        P, D, Q, s = seasonal_lags
        # considering the max lag between ARIMA and seasonal components
        max_lag = max([max(lags), s])
        bootstrap_series[:max_lag] = orig_X[:max_lag]
        simulated_residuals = simulate_sarima_process(n_samples, fitted_model)
        simulated_residuals = diff_inv(simulated_residuals, D, xi=orig_X[:D])
        if s > 0:
            simulated_residuals = diff_inv(
                simulated_residuals, s*D, xi=orig_X[:s*D])
    elif model == 'var':
        max_lag = fitted_model.model.k_ar
        bootstrap_series[:max_lag] = orig_X[:max_lag]
        simulated_residuals = simulate_var_process(n_samples, fitted_model)
    elif model == 'arch':
        if isinstance(resids_lags, int):
            max_lag = resids_lags
        else:
            # Set max_lag to the highest lag in the ARCH model
            max_lag = max(resids_lags)
        bootstrap_series[:max_lag] = orig_X[:max_lag]
        simulated_residuals = simulate_arch_process(
            n_samples, fitted_model)
    else:
        raise ValueError(f"Unknown model: {model}")

    for t in range(max_lag, n_samples):
        lagged_values = bootstrap_series[t - np.array(lags)]
        bootstrap_series[t] = coefs @ lagged_values.T + simulated_residuals[t]

    return bootstrap_series


'''
