from arch.univariate.mean import ARX
from arch.univariate import GARCH
from typing import Union, List, Optional, Tuple
from numpy import ndarray
from arch import arch_model
from numba import njit
from numba import float64, prange, int32
from numba.types import Array
from functools import partial
from statsmodels.tsa.stattools import acf, pacf
from numpy.random import RandomState
from hmmlearn import hmm
from sklearn.cluster import KMeans
import math
import numpy as np
import numba

from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper
from statsmodels.tsa.ar_model import AutoReg, AutoRegResultsWrapper
from statsmodels.tsa.vector_ar.var_model import VAR, VARResultsWrapper
from statsmodels.tsa.arima.model import ARIMA, ARIMAResultsWrapper
from arch.univariate.base import ARCHModelResult


# For Spectral Bootstrap
@njit
def rfftfreq_numba(n: int, d: float = 1.0) -> np.ndarray:
    if n % 2 == 0:
        N = n // 2 + 1
    else:
        N = (n + 1) // 2
    return np.arange(N) / (n * d)


# For Bayesian Bootstrap
@njit
def dirichlet_numba(alpha: np.ndarray, random_seed: int) -> np.ndarray:
    np.random.seed(random_seed)
    sample = np.empty(alpha.shape, dtype=np.float64)
    for i in range(len(alpha)):
        sample[i] = np.random.gamma(alpha[i], 1)
    return sample / np.sum(sample)


@njit
def choice_with_p(weights: np.ndarray, size: int) -> np.ndarray:
    """
    Given an array of weights, this function returns an array of indices
    sampled with probabilities proportional to the input weights.

    Parameters
    ----------
    weights : np.ndarray
        An array of probabilities for each index.
    size : int
        The number of indices to sample.

    Returns
    -------
    np.ndarray
        An array of sampled indices.
    """
    # Normalize weights
    p = weights / weights.sum()
    # Create cumulative sum of normalized weights (these will now act as probabilities)
    cum_weights = np.cumsum(p)
    # Draw random values
    random_values = np.random.rand(size)
    # Find indices where drawn random values would be inserted in cumulative weight array
    chosen_indices = np.searchsorted(cum_weights, random_values)
    return chosen_indices


# For banded bootstrap
@njit
def hankel_numba(c: np.ndarray, r: np.ndarray) -> np.ndarray:
    hankel_matrix = np.empty((len(c), len(r)), dtype=c.dtype)
    for i in range(len(c)):
        for j in range(len(r)):
            if i + j < len(c):
                hankel_matrix[i, j] = c[i + j]
            else:
                hankel_matrix[i, j] = r[i + j - len(c) + 1]
    return hankel_matrix


def fit_ar(X: np.ndarray, lags: Union[int, List[int]] = 1, **kwargs) -> AutoRegResultsWrapper:
    """Fits an AR model to the input data.

    Args:
        X (ndarray): The input data.
        ar_order (int): The order of the AR model.

    Returns:
        AutoRegResultsWrapper: The fitted AR model.
    """
    # X has to be 1d
    # lags can be a list of non-consecutive positive ints
    X = X.squeeze()
    if X.ndim != 1:
        raise ValueError("X must be 1-dimensional")
    model = AutoReg(endog=X, lags=lags, **kwargs)
    model_fit = model.fit()
    return model_fit


def fit_arima(X: ndarray, arima_order: Tuple[int, int, int] = (1, 0, 0), exog: Optional[np.ndarray] = None, **kwargs) -> ARIMAResultsWrapper:
    """Fits an ARIMA model to the input data.

    Args:
        X (ndarray): The input data.
        arima_order (Tuple[int, int, int]): The order of the ARIMA model.

    Returns:
        ARIMAResultsWrapper: The fitted ARIMA model.
    """
    # X has to be 1d
    X = X.squeeze()
    if X.ndim != 1:
        raise ValueError("X must be 1-dimensional")
    if exog is not None:
        if exog.ndim == 1:
            exog = exog.reshape(-1, 1)
        if exog.ndim != 2:
            raise ValueError("exog must be 2-dimensional")
        if exog.shape[0] != X.shape[0]:
            raise ValueError("exog must have the same number of rows as X")
    ar_order, diff_deg, ma_order = arima_order
    model = ARIMA(endog=X, order=arima_order, exog=exog, **kwargs)
    model_fit = model.fit()
    return model_fit


def fit_sarima(X: ndarray, sarima_order: Tuple[int, int, int, int] = (0, 0, 0, 2), arima_order: Optional[Tuple[int, int, int]] = (1, 0, 0), exog: Optional[np.ndarray] = None, **kwargs) -> SARIMAXResultsWrapper:
    """Fits a SARIMA model to the input data.

    Args:
        X (ndarray): The input data.
        sarima_order (Tuple[int, int, int, int]): The order of the SARIMA model.

    Returns:
        SARIMAXResultsWrapper: The fitted SARIMA model.
    """
    # X has to be 1d
    X = X.squeeze()
    if X.ndim != 1:
        raise ValueError("X must be 1-dimensional")
    if exog is not None:
        if exog.ndim == 1:
            exog = exog.reshape(-1, 1)
        if exog.ndim != 2:
            raise ValueError("exog must be 2-dimensional")
        if exog.shape[0] != X.shape[0]:
            raise ValueError("exog must have the same number of rows as X")
    if sarima_order[-1] < 2:
        raise ValueError("The seasonal periodicity must be greater than 1")
    if arima_order is None:
        arima_order = sarima_order[:3]
    model = SARIMAX(endog=X, order=arima_order,
                    seasonal_order=sarima_order, exog=exog, **kwargs)
    model_fit = model.fit()
    return model_fit


def fit_var(X: ndarray, lags: Optional[int] = None, exog: Optional[np.ndarray] = None, **kwargs) -> VARResultsWrapper:
    """Fits a Vector Autoregression (VAR) model to the input data.
    Args:
        X (ndarray): The input data.
        lags (int): The number of lags to include in the VAR model. We let the model determine the optimal number of lags.
    Returns:
        VARResultsWrapper: The fitted VAR model.
    """
    # X has to be 2d with at least 2 columns
    if X.ndim != 1 and X.shape[1] < 2:
        raise ValueError("X must be 2-dimensional with at least 2 columns")
    if exog is not None:
        if exog.ndim == 1:
            exog = exog.reshape(-1, 1)
        if exog.ndim != 2:
            raise ValueError("exog must be 2-dimensional")
        if exog.shape[0] != X.shape[0]:
            raise ValueError("exog must have the same number of rows as X")
    model = VAR(endog=X, exog=exog, **kwargs)
    model_fit = model.fit()  # maxlags=lags)
    return model_fit


def fit_arch(X: np.ndarray, p: int = 1, q: int = 1, model_type: str = 'GARCH', lags: Union[int, List[int]] = 0, exog: Optional[np.ndarray] = None, **kwargs) -> ARCHModelResult:
    """
    Fits a GARCH, GARCH-M, EGARCH, TARCH, or AGARCH model to the input data.

    Args:
        X (ndarray): The input data.
        p (int): The number of lags in the GARCH part of the model.
        q (int): The number of lags in the ARCH part of the model.
        model_type (str): The type of GARCH model to fit. Options are 'GARCH', 'GARCH-M', 'EGARCH', 'TARCH', and 'AGARCH'.
        lags (Union[int, List[int]]): The number of lags or a list of lag indices to include in the AR part of the model.

    Returns:
        ARCHModelResult: The fitted GARCH model.
    """
    # X has to be 1d
    X = X.squeeze()
    if X.ndim != 1:
        raise ValueError("X must be 1-dimensional")
    X = np.ascontiguousarray(X)  # Make sure the input array is C-contiguous
    if exog is not None:
        if exog.ndim == 1:
            exog = exog.reshape(-1, 1)
        if exog.ndim != 2:
            raise ValueError("exog must be 2-dimensional")
        if exog.shape[0] != X.shape[0]:
            raise ValueError("exog must have the same number of rows as X")
        exog = np.ascontiguousarray(exog)
    if model_type == 'GARCH':
        model = ARX(y=X, x=exog, lags=lags)
        model.volatility = GARCH(p=p, q=q)
    elif model_type == 'GARCH-M':
        model = arch_model(y=X, x=exog, mean='AR', lags=lags,
                           vol='GARCH-M', p=p, q=q, **kwargs)
    elif model_type == 'EGARCH':
        model = arch_model(y=X, x=exog, mean='Zero', lags=lags,
                           vol='EGARCH', p=p, q=q, **kwargs)
    elif model_type == 'TARCH':
        model = arch_model(y=X, x=exog, mean='Zero', lags=lags,
                           vol='TARCH', p=p, q=q, **kwargs)
    else:
        raise ValueError(
            "model_type must be one of 'GARCH', 'GARCH-M', 'EGARCH', 'TARCH', or 'AGARCH'")
    options = {"maxiter": 200}
    model_fit = model.fit(disp='off', options=options)
    return model_fit


# TODO: introduce gap parameters before and after the split.
def time_series_split(X: np.ndarray, test_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    split_index = int(len(X) * (1-test_ratio))
    return X[:split_index], X[split_index:]


@njit
def cholesky_numba(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    L = np.zeros((n, n))

    for i in prange(n):
        for j in range(i + 1):
            s = np.dot(L[j, :j], L[j, :j])
            if i == j:
                L[i, i] = np.sqrt(A[i, i] - s)
            else:
                L[j, i] = (A[i, j] - s) / L[j, j]

    return L


@njit
def har_cov(X: np.ndarray, h: int) -> np.ndarray:
    """
    Calculate the Heteroskedasticity-Autocorrelation Robust (HAR) covariance matrix estimator.

    Parameters
    ----------
    X : np.ndarray
        The input data array (time series)
    h : int
        The number of lags to consider in the autocovariance estimation

    Returns
    -------
    np.ndarray
        The estimated covariance matrix
    """
    n = X.shape[0]
    p = X.shape[1]
    cov_matrix = np.zeros((p, p))

    for t in range(n):
        x_t = X[t].reshape(-1, 1)
        for lag in range(1, h + 1):
            if t - lag >= 0:
                x_lag = X[t - lag].reshape(-1, 1)
                cov_matrix += (1 / (n * lag ** 2)) * np.outer(x_t, x_lag)

    return cov_matrix


###############
# For fractional differentiation
@njit
def _gamma_lanczos(x: float) -> float:
    p = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
         771.32342877765313, -176.61502916214059, 12.507343278686905,
         -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
    x -= 1.0
    y = p[0]
    for i in range(1, len(p)):
        y += p[i] / (x + i)
    t = x + len(p) - 0.5
    return np.sqrt(2 * np.pi) * t**(x + 0.5) * np.exp(-t) * y


@njit
def _comb_numba(n: float, k: int) -> float:
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    num = _gamma_lanczos(n + 1)
    den = _gamma_lanczos(k + 1) * _gamma_lanczos(n - k + 1)
    return num / den


@njit
def _fft_numba(x: np.ndarray) -> np.ndarray:
    n = x.size
    if n <= 1:
        return x
    even = _fft_numba(x[0::2])
    odd = _fft_numba(x[1::2])
    factor = np.exp(-2j * np.pi * np.arange(n) / n)
    return np.concatenate([even + factor[:n // 2] * odd, even + factor[n // 2:] * odd])


@njit
def _ifft_numba(x: np.ndarray) -> np.ndarray:
    n = x.size
    if n <= 1:
        return x
    even = _ifft_numba(x[0::2])
    odd = _ifft_numba(x[1::2])
    factor = np.exp(2j * np.pi * np.arange(n) / n)
    return np.concatenate([even + factor[:n // 2] * odd, even + factor[n // 2:] * odd]) / n


@njit
def _convolve_numba(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    n_sig = len(signal)
    n_ker = len(kernel)
    n_res = n_sig + n_ker - 1
    n_fft = int(2 ** np.ceil(np.log2(n_res)))

    signal_padded = np.pad(signal, (0, n_fft - n_sig))
    kernel_padded = np.pad(kernel, (0, n_fft - n_ker))

    fft_signal = _fft_numba(signal_padded)
    fft_kernel = _fft_numba(kernel_padded)

    fft_result = fft_signal * fft_kernel
    result = _ifft_numba(fft_result).real

    return result[:n_res]


@njit
def fractional_diff_numba(X: np.ndarray, d: float) -> np.ndarray:
    n = X.shape[0]
    weights = np.array([((-1)**k) * _comb_numba(d, k) for k in range(n)])
    conv_result = _convolve_numba(X, weights[::-1])

    # Calculate the number of initial values to drop
    num_to_drop = int(math.ceil(d))

    return conv_result[num_to_drop:-(n - 1)]


###############
# For Markov bootstrap

@njit
def calculate_transition_probs(assignments, n_components):
    num_blocks = len(assignments)
    transitions = np.zeros((n_components, n_components))
    for i in range(num_blocks - 1):
        transitions[assignments[i], assignments[i + 1]] += 1
    transition_probabilities = transitions / \
        np.sum(transitions, axis=1)[:, np.newaxis]
    return transition_probabilities


def fit_hidden_markov_model(X: np.ndarray, n_states: int, n_iter=1000) -> hmm.GaussianHMM:
    model = hmm.GaussianHMM(n_components=n_states,
                            covariance_type="diag", n_iter=n_iter)
    model.fit(X)
    return model


def calculate_transition_probabilities(X: np.ndarray, block_length: int, method: str = 'clustering', n_components: Optional[int] = 3, random_state: RandomState = RandomState(42)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    method = method.lower()
    if method not in ['random', 'clustering', 'hmm']:
        raise ValueError(
            "Method must be one of 'random', 'clustering', or 'hmm'")

    num_blocks = X.shape[0] // block_length
    remainder = X.shape[0] % block_length
    blocks = [X[i*block_length:(i+1)*block_length, :]
              for i in range(num_blocks)]
    if remainder > 0:
        blocks.append(X[num_blocks*block_length:, :])

    blocks = np.asarray(blocks)

    if method == 'random':
        cluster_assignments = random_state.randint(
            n_components, size=len(blocks))
        cluster_centers = np.asarray(
            [blocks[cluster_assignments == i].mean(axis=0) for i in range(n_components)])

    elif method == 'clustering':
        kmeans = KMeans(n_clusters=n_components,
                        random_state=random_state).fit(np.vstack(blocks))
        cluster_centers = kmeans.cluster_centers_
        cluster_assignments = kmeans.labels_

    elif method == 'hmm':
        model = fit_hidden_markov_model(np.vstack(blocks), n_components)
        cluster_centers = model.means_
        cluster_assignments = model.predict(np.vstack(blocks))

    transition_probabilities = calculate_transition_probs(
        cluster_assignments, n_components)

    return transition_probabilities, cluster_centers, cluster_assignments

################


@njit
def weibull_mle_single_feature(x: np.ndarray, max_iter: int = 100, tol: float = 1e-8):
    # Calculate L-moments for a more robust initial guess
    x_sorted = np.sort(x)
    w = np.arange(1, len(x) + 1) / (len(x) + 1)
    lmom1 = np.sum(x_sorted * w)
    lmom2 = np.sum(x_sorted * (w - 0.35))

    # Initial guess for the shape and scale parameters
    shape = np.log(2) / np.log(lmom2 / lmom1)
    scale = lmom1 / np.exp(math.gamma(1 + 1 / shape))

    # BFGS optimizer
    bfgs = BFGS(max_iter, tol)
    params = np.array([shape, scale], dtype=np.float64)
    params_opt = bfgs.optimize(
        weibull_neg_log_likelihood, weibull_neg_log_likelihood_grad, params, x)

    return params_opt


@njit
def weibull_mle(X: np.ndarray, max_iter: int = 100, tol: float = 1e-8) -> np.ndarray:
    n_features = X.shape[1]
    params = np.zeros((n_features, 2))

    for feature_idx in range(n_features):
        params[feature_idx] = weibull_mle_single_feature(
            X[:, feature_idx], max_iter, tol)

    return params


@njit
def weibull_neg_log_likelihood(params: np.ndarray, X: np.ndarray):
    shape, scale = params
    n = len(X)
    return n * np.log(scale) + (shape - 1) * np.sum(np.log(X)) - np.sum((X / scale) ** shape)


@njit
def weibull_neg_log_likelihood_grad(params: np.ndarray, X: np.ndarray):
    shape, scale = params
    n = len(X)
    X_scaled = X / scale
    X_scaled_shape = X_scaled ** shape
    dshape = -n / shape + np.sum(np.log(X) - X_scaled_shape * np.log(X_scaled))
    dscale = n / scale - np.sum(X_scaled_shape * (X / scale))
    return np.array([dshape, dscale])


spec = [
    ('max_iter', int32),
    ('tol', float64),
    ('alpha', float64),
    ('last_fval', float64),
    ('last_grad', Array(float64, 1, 'C')),
]


@numba.experimental.jitclass(spec)
class BFGS:
    def __init__(self, max_iter: int, tol: float, alpha: float = 0.1):
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.last_fval = 0.0
        self.last_grad = np.zeros(1)

    def optimize(self, f, grad_f, x0, *args):
        x = x0.copy()
        n = len(x)
        I = np.eye(n)
        H = I.copy()

        for k in range(self.max_iter):
            self.last_fval = f(x, *args)
            self.last_grad = grad_f(x, *args)

            if np.linalg.norm(self.last_grad) < self.tol:
                break

            p = -H @ self.last_grad
            x_new = x + self.alpha * p
            s = x_new - x
            x = x_new

            y = grad_f(x, *args) - self.last_grad
            rho = 1 / np.dot(y, s)
            H = (I - rho * np.outer(s, y)) @ H @ (I -
                                                  rho * np.outer(y, s)) + rho * np.outer(s, s)

        return x


@njit
def generate_test_mask_weibull(X: np.ndarray, random_seed: int) -> np.ndarray:
    np.random.seed(random_seed)
    n_samples, n_features = X.shape
    params = weibull_mle(X)
    samples = np.zeros((n_samples, n_features))

    for feature_idx in range(n_features):
        shape, scale = params[feature_idx]
        samples[:, feature_idx] = np.random.weibull(
            shape, size=n_samples) * scale

    return samples


# for trimmed bootstrap
@njit
def trimboth_numba(X: np.ndarray, proportiontocut: float):
    lower_limit = np.percentile(X, proportiontocut * 100.0)
    upper_limit = np.percentile(X, 100.0 - proportiontocut * 100.0)
    return X[(X >= lower_limit) & (X <= upper_limit)]


# for time-varying bootstrap
def acf_error_multidimensional(block_length: int, data: np.ndarray) -> float:
    n_vars = data.shape[1]  # number of time series (columns)
    acf_errors = []

    for i in range(n_vars):
        var_data = data[:, i]
        acf_values = acf(var_data, nlags=block_length)
        acf_sq = np.square(acf_values)
        acf_errors.append(acf_sq)

    mean_acf_errors = np.mean(acf_errors, axis=0)
    error = np.sum(mean_acf_errors)
    return error


def pacf_error_multidimensional(block_length: int, data: np.ndarray) -> float:
    n_vars = data.shape[1]  # number of time series (columns)
    pacf_errors = []

    for i in range(n_vars):
        var_data = data[:, i]
        pacf_values = pacf(var_data, nlags=block_length)
        pacf_sq = np.square(pacf_values)
        pacf_errors.append(pacf_sq)

    mean_pacf_errors = np.mean(pacf_errors, axis=0)
    error = np.sum(mean_pacf_errors)
    return error


# for random blocklength bootstrap


class BlockLengthSampler:
    """
    A class for sampling block lengths for the random block length bootstrap.

    Attributes
    ----------
    block_length_distributions : Dict[str, Callable]
        A dictionary containing the supported block length distribution functions.
    block_length_distribution : str
        The selected block length distribution function.
    avg_block_length : int
        The average block length to be used for sampling.
    """

    def __init__(self, block_length_distribution: str, avg_block_length: int):
        """
        Initialize the BlockLengthSampler with the selected distribution and average block length.

        Parameters
        ----------
        block_length_distribution : str
            The block length distribution function to use.
        avg_block_length : int
            The average block length to be used for sampling.
        """
        self.block_length_distribution = block_length_distribution
        self.avg_block_length = avg_block_length
        self.block_length_distributions = {
            'poisson': partial(np.random.poisson, lam=avg_block_length),
            'exponential': partial(np.random.exponential, scale=avg_block_length),
            'normal': partial(np.random.normal, loc=avg_block_length, scale=avg_block_length / 3),
            'gamma': partial(np.random.gamma, shape=2, scale=avg_block_length / 2),
            'beta': partial(np.random.beta, a=2, b=2),
            'lognormal': partial(np.random.lognormal, mean=np.log(avg_block_length / 2), sigma=np.log(2)),
            'weibull': partial(np.random.weibull, a=1.5),
            'pareto': partial(np.random.pareto, a=1),
            'geometric': partial(np.random.geometric, p=1 / avg_block_length),
            'uniform': partial(np.random.randint, low=1, high=2 * avg_block_length)
        }

        if block_length_distribution not in self.block_length_distributions:
            raise ValueError(f"Unknown block_length_distribution '{block_length_distribution}', "
                             f"available options are: {list(self.block_length_distributions.keys())}")

    def sample_block_length(self, random_seed: int = None) -> int:
        """
        Sample a block length from the selected distribution.

        Parameters
        ----------
        random_seed : int, optional
            Random seed for reproducibility, by default None. If None, the global random state is used.

        Returns
        -------
        int
            A sampled block length.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        return int(self.block_length_distributions[self.block_length_distribution]())
