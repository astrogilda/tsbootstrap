from numba import njit
import numpy as np
from future_work.utils import mean_axis_0


@njit
def cholesky_numba(A):
    return np.linalg.cholesky(A)


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
# Heteroskedasticity-Autocorrelation Robust (HAR) Bootstrap
class BaseHARBootstrap(BaseTimeSeriesBootstrap):
    def __init__(self, *args, bandwidth: Optional[int] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if bandwidth is not None:
            if not isinstance(bandwidth, int):
                raise TypeError("Bandwidth must be an integer")
            if bandwidth < 1:
                raise ValueError(
                    "Bandwidth must be a positive integer greater than or equal to 1")
        self.bandwidth = bandwidth

    def _generate_cholesky_matrix(self, X: np.ndarray) -> np.ndarray:

        if self.bandwidth is None:
            # Rule-of-thumb for the number of lags
            h = int(np.ceil(4 * (X.shape[0] / 100) ** (2 / 9)))
        else:
            h = self.bandwidth
        return generate_har_decomposition(X, h)

    def _generate_bootstrapped_errors(self, X: np.ndarray, cholesky: np.ndarray, random_seed: int) -> np.ndarray:
        return generate_har_errors(X, cholesky, random_seed)


class WholeHARBootstrap(BaseHARBootstrap):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cholesky = None

    def _generate_samples_single_bootstrap(self, X: np.ndarray, random_seed: Optional[int], **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if random_seed is None:
            random_seed = self.random_seed

        if self.cholesky is None:
            self.cholesky = self._generate_cholesky_matrix(X)

        in_bootstrap_errors = self._generate_bootstrapped_errors(
            X, self.cholesky, random_seed)
        in_bootstrap_samples_hac = X + in_bootstrap_errors
        return [np.arange(X.shape[0])], [in_bootstrap_samples_hac]


class BlockHARBootstrap(BaseHARBootstrap, BaseBlockBootstrap):
    def _generate_samples_single_bootstrap(self, X: np.ndarray, random_seed: Optional[int], **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        block_indices, block_data = super()._generate_samples_single_bootstrap(X=X,
                                                                               random_seed=random_seed, **kwargs)

        bootstrap_samples = []

        for block_data_iter in block_data:
            cholesky = self._generate_cholesky_matrix(block_data_iter)
            # block_data[block] will always be 2d by design, see `generate_block_indices_and_data`
            bootstrap_errors = self._generate_bootstrapped_errors(
                block_data_iter, cholesky, random_seed)
            bootstrap_samples.append(block_data_iter + bootstrap_errors)

        return block_indices, bootstrap_samples
'''
