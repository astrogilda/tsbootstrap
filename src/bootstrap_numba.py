from arch import arch_model
import scipy
from typing import Optional, List, Callable
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from numba import njit
from numba.core.registry import CPUDispatcher
from sklearn.utils import check_random_state

from utils.block_length_sampler import BlockLengthSampler
from utils.numba_base import *

# TODO: block_weights=p with block_length=1 should be equivalent to the iid bootstrap

"""
Call the below functions from src.bootstrap.py. These functions have not had their inputs checked.
"""

"""
The key difference between the two methods lies in how they select the starting point for each block:

For Moving Block Bootstrap, the starting point of each block is drawn from all possible starting points (from 0 to len(X) - block_length), which allows overlap between blocks.

For Stationary Bootstrap, the starting point of each block is drawn uniformly at random from the entire series (from 0 to len(X) - block_length), and blocks can wrap around to the start of the series.
"""


@njit
def generate_indices_random(num_samples: int, random_seed: int) -> np.ndarray:
    np.random.seed(random_seed)
    in_bootstrap_indices = np.random.choice(
        np.arange(num_samples), size=num_samples, replace=True)
    return in_bootstrap_indices


"""
        # if len(tapered_weights) != block_length:
        #    raise ValueError(
        #        "Size of tapered_weights must be equal to block_length.")
        # else:
        # Apply tapered_weights to each block

"""

# Allows the blocks to overlap or not, depending on the overlap_flag
# Allows the blocks to wrap around to the start of the series, depending on the wrap_around flag
# Allows the user to specify weights for the blocks, which are used to sample the blocks
# block_weights are used when picking the blocks and the tapered_weights are used for weighting the samples within the block. Note that tapered_weights must be of size equal to block_length.

"""

import numpy as np
from src.bootstrap_numba import generate_block_indices_and_data
X = np.random.random((10,2))
block_length = 3
q,w = generate_block_indices_and_data(X, block_length, block_weights=np.array([]), tapered_weights=np.array([]), overlap_flag=True, wrap_around_flag=True, random_seed=0)

"""


@njit
def _prepare_block_weights(block_weights: Union[np.ndarray, Callable], X: np.ndarray) -> np.ndarray:
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
    if callable(block_weights):
        if not isinstance(block_weights, CPUDispatcher):
            block_weights = njit(block_weights)
        block_weights = block_weights(X)
    elif block_weights.size == 0:
        size = X.shape[0]
        block_weights = np.full(size, 1 / size)
    else:  # normalize block_weights
        assert block_weights.size == X.shape[0], "block_weights array must have the same size as X"
        block_weights = block_weights / np.sum(block_weights)
    return block_weights


@njit
def _prepare_tapered_weights(tapered_weights: Union[np.ndarray, Callable], block_length: int) -> np.ndarray:
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
    if callable(tapered_weights):
        if not isinstance(tapered_weights, CPUDispatcher):
            tapered_weights = njit(tapered_weights)
        tapered_weights = tapered_weights(block_length)
    elif tapered_weights.size == 0:
        tapered_weights = np.full(block_length, 1 / block_length)
    else:
        assert tapered_weights.size == block_length, "tapered_weights array must have the same size as block_length"
        tapered_weights = tapered_weights / np.sum(tapered_weights)
    return tapered_weights


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
        block_indices.append(np.arange(sampled_block_start, min(
            sampled_block_start + block_length, n)) % (n if wrap_around_flag else 1))
        total_elements_covered += block_length
    return block_indices


@njit
def generate_block_indices_and_data(X: np.ndarray, block_length: int, block_weights: Union[np.ndarray, Callable], tapered_weights: np.ndarray, block_length_distribution: str = 'none', overlap_flag: bool = True, wrap_around_flag: bool = False, random_seed: int = 42) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate block indices and corresponding data for the input data array X.

    Parameters
    ----------
    X : np.ndarray
        Input data array.
    block_length : int
        Length of each block.
    block_weights : np.ndarray
        An array of weights or a callable function to generate weights.
    tapered_weights : np.ndarray
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
    This function is optimized using Numba, so it does not raise any errors.
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


def generate_samples_markov(X: np.ndarray, method: str, block_length: int, n_clusters: int, random_seed: int) -> np.ndarray:
    method = method.lower()
    if method not in ['random', 'clustering', 'hmm']:
        raise ValueError(
            "Method must be one of 'random', 'clustering', or 'hmm'")

    random_state = check_random_state(random_seed)
    transition_probabilities, cluster_centers, cluster_assignments = calculate_transition_probabilities(
        X, block_length, method, n_clusters, random_seed)

    num_blocks = len(X) // block_length
    remainder = len(X) % block_length
    bootstrap_sample = []

    # Select the initial block based on the clustered blocks
    current_block = random_state.choice(num_blocks)
    current_cluster = cluster_assignments[current_block]

    for _ in range(num_blocks):
        # Append the cluster center (representative block) to the bootstrap sample
        bootstrap_sample.append(cluster_centers[current_cluster])

        # Update the current cluster based on the transition probabilities
        current_cluster = random_state.choice(
            n_clusters, p=transition_probabilities[current_cluster])

    # Handle the case when the length of X is not an integer multiple of block_length
    if remainder > 0:
        bootstrap_sample.append(
            cluster_centers[current_cluster][:remainder, :])

    return np.concatenate(bootstrap_sample, axis=0)


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
def generate_hac_errors(X: np.ndarray, bandwidth: int, random_seed: int) -> np.ndarray:
    np.random.seed(random_seed)

    # Use the HAR estimator to compute the long-run covariance matrix
    # Rule-of-thumb for the number of lags
    h = bandwidth
    long_run_cov = har_cov(X, h)

    # Calculate the Cholesky decomposition of the long-run covariance matrix
    cholesky_decomposition = cholesky_numba(long_run_cov)

    # Generate the bootstrapped errors
    normal_errors = np.random.randn(X.shape[0], X.shape[1])
    bootstrapped_errors = normal_errors @ cholesky_decomposition.T

    return bootstrapped_errors


@njit
def generate_samples_residual(X: np.ndarray, X_fitted: np.ndarray, residuals: np.ndarray, lag_order: int, random_seed: int) -> np.ndarray:
    # Resample residuals
    resampled_indices = generate_indices_random(
        residuals.shape[0], random_seed)
    resampled_residuals = residuals[resampled_indices]
    # Add the bootstrapped residuals to the fitted values
    bootstrapped_X = X_fitted + resampled_residuals
    # Prepend the first 'lag_order' original observations to the bootstrapped series
    extended_bootstrapped_X = np.vstack((X[:lag_order], bootstrapped_X))
    # Prepend the indices of the first 'lag_order' original observations to resampled_indices
    initial_indices = np.arange(lag_order, dtype=np.int64)
    extended_resampled_indices = np.hstack(
        (initial_indices, resampled_indices + lag_order))
    return extended_resampled_indices, extended_bootstrapped_X


@njit
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


@njit
def simulate_ar_process(lags: List[int], coefs: np.ndarray, init: np.ndarray, random_errors: np.ndarray) -> np.ndarray:
    """
    Simulates an Autoregressive (AR) process with given lags, coefficients, initial values, and random errors.

    Args:
        lags (List[int]): The lags to be used in the AR process. Can be non-consecutive.
        coefs (np.ndarray): The coefficients corresponding to each lag. Should be the same length as `lags`.
        init (np.ndarray): The initial values for the simulation. Should be at least as long as the maximum lag.
        random_errors (np.ndarray): The random errors to be added at each step.

    Returns:
        np.ndarray: The simulated AR process as a 1D NumPy array.

    Raises:
        ValueError: If `coefs` is not the same length as `lags`.
        ValueError: If `init` is not long enough to cover the maximum lag.
    """
    max_lag = max(lags)
    if len(coefs) != len(lags):
        raise ValueError("Length of 'coefs' must match length of 'lags'")
    if len(init) < max_lag:
        raise ValueError(
            "Length of 'init' must be at least as long as the maximum lag")

    n_samples = len(random_errors)
    series = np.zeros(n_samples)
    series[:max_lag] = init

    # Loop through the series, calculating each value based on the lagged values, coefficients, and random error
    for t in range(max_lag, n_samples):
        lagged_values = series[t - np.array(lags)]
        series[t] = coefs @ lagged_values + random_errors[t]

    return series


def simulate_arima_process(n_samples: int, fitted_model: ARIMAResultsWrapper) -> np.ndarray:
    """Simulate residuals for an ARIMA model."""
    rng = np.random.default_rng()
    return fitted_model.simulate(nsimulations=n_samples, error_gen=rng.normal)


def simulate_sarima_process(n_samples: int, fitted_model: SARIMAXResultsWrapper) -> np.ndarray:
    """Simulate residuals for a SARIMA model."""
    rng = np.random.default_rng()
    return fitted_model.simulate(nsimulations=n_samples, error_gen=rng.normal)


def simulate_var_process(n_samples: int, fitted_model: VARResultsWrapper) -> np.ndarray:
    """Simulate residuals for a VAR model."""
    rng = np.random.default_rng()
    # Assuming the VAR model is fitted with lags = k
    initial_value = fitted_model.y[-fitted_model.model.k_ar:]
    return fitted_model.model.simulate_var(fitted_model.params, n_steps=n_samples, initial_value=initial_value, error_gen=rng.normal)


def simulate_arch_process(n_samples: int, arch_model: ARCHModelResult) -> np.ndarray:
    """Simulate residuals for an ARCH/GARCH model."""
    rng = np.random.default_rng()
    return arch_model.model.simulate(arch_model.params, n_samples, rng.normal)


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


def generate_samples_sieve(
    X: np.ndarray,
    resids_lags: Union[int, List[int]],
    resids_coefs: np.ndarray,
    resids_fit_model: AutoRegResultsWrapper,
    random_seed: int,
) -> np.ndarray:

    n_samples, n_features = X.shape
    np.random.seed(random_seed)

    # Generate the bootstrap series
    bootstrap_series = np.zeros((n_samples, n_features), dtype=np.float64)

    max_lag = max(resids_lags) if isinstance(
        resids_lags, list) else resids_lags
    bootstrap_series[:max_lag] = X[:max_lag]
    simulated_residuals = simulate_ar_process(
        resids_lags, resids_coefs, resids_fit_model.resid[:max_lag], np.random.normal(size=n_samples))

    for t in range(max_lag, n_samples):
        lagged_values = bootstrap_series[t - np.array(resids_lags)]
        bootstrap_series[t] = resids_coefs @ lagged_values.T + \
            simulated_residuals[t]

    return bootstrap_series


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


def estimate_bias_implementation(X, block_length, n_bootstraps, bias_scope, jitted_statistic, random_seed):
    np.random.seed(random_seed)
    n = X.shape[0]
    if bias_scope == 'whole':
        original_stat = jitted_statistic(X)
        bootstrap_stats = np.zeros(n_bootstraps)
        for i in range(n_bootstraps):
            test_samples = X[np.random.randint(n, size=n)]
            bootstrap_stats[i] = jitted_statistic(test_samples)
        mean_bootstrap_stat = np.mean(bootstrap_stats)
        bias = mean_bootstrap_stat - original_stat
        return np.array([bias])
    elif bias_scope == 'block':
        n_blocks = n // block_length
        original_stats = np.array([jitted_statistic(
            X[i * block_length:(i + 1) * block_length]) for i in range(n_blocks)])
        bootstrap_stats = np.zeros((n_blocks, n_bootstraps))
        for i in range(n_bootstraps):
            test_samples = X[np.random.randint(n, size=n)]
            for j in range(n_blocks):
                start = j * block_length
                end = start + block_length
                block_stat = jitted_statistic(test_samples[start:end])
                bootstrap_stats[j, i] = block_stat
        biases = np.mean(bootstrap_stats, axis=1) - original_stats
        return biases


@generated_jit(nopython=True)
def estimate_bias(X: np.ndarray, block_length: int, n_bootstraps: int,
                  bias_scope: str, statistic: Callable, random_seed: int) -> np.ndarray:
    # Check if the statistic function is Numba-compatible (has a Numba-compiled implementation)
    if hasattr(statistic, 'py_func') and hasattr(statistic, 'inspect_llvm'):
        jitted_statistic = statistic
    else:
        raise ValueError(
            "The statistic function must be a Numba-compiled function.")

    return estimate_bias_implementation(X=X, block_length=block_length, n_bootstraps=n_bootstraps, bias_scope=bias_scope, jitted_statistic=jitted_statistic, random_seed=random_seed)


     return test_mask

     

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

'''
