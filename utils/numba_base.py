from numba import njit
import numpy as np
from utils.odds_and_ends import mean_axis_0


@njit
def har_cov(X: np.ndarray, h: int) -> np.ndarray:
    """
    Compute the Heteroskedasticity-Autocorrelation Robust (HAR) covariance matrix estimator.

    Parameters
    ----------
    X : np.ndarray
        The input data array (time series).
    h : int
        The number of lags to consider in the autocovariance estimation.

    Returns
    -------
    np.ndarray
        The HAR covariance matrix.
    """
    assert X.ndim == 2, "X must be a 2-dimensional array."
    n, k = X.shape
    assert h >= 0, "h must be non-negative."
    assert h < n, "h must be less than the number of time steps in X."
    X_centered = X - mean_axis_0(X)
    S = np.zeros((k, k))

    for j in range(h + 1):
        gamma_j = np.zeros((k, k))
        for t in range(j, n):
            gamma_j += np.outer(X_centered[t], X_centered[t - j])

        gamma_j /= (n - j)

        if j == 0:
            S += gamma_j
        else:
            S += (1 - j / (h + 1)) * (gamma_j + gamma_j.T)

    return S


'''

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

'''

'''
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
'''

'''

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


'''
