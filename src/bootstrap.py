from __future__ import annotations
from fracdiff.sklearn import Fracdiff, FracdiffStat
from statsmodels.tsa.stattools import adfuller
from numpy.random import RandomState
import warnings
from statsmodels.tsa.ar_model import AutoRegResultsWrapper
from arch.univariate.base import ARCHModelResult
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from statsmodels.tsa.arima.model import ARIMAResultsWrapper
from typing import Callable, List, Optional, Tuple, Union
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, RegressorMixin

from functools import lru_cache
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.ar_model import AutoReg
from typing import Iterator, Type
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod

from src.bootstrap_numba import *

# TODO: use TypeVar to allow for different types of data (e.g. numpy arrays, pandas dataframes, lists, etc.)
# TODO: add option for block length to be a fraction of the data length
# TODO: use Type to indicate the type of the data (even classes), instead of directly reference the instance
# TODO: add option for multiprocessing using mp.pool in _iter_test_masks and _generate_samples
# TODO: add an option for VECM (vector error correction model) bootstrap
# TODO: write a data abstraction layer that converts data to a common format (e.g. numpy arrays) and back. also convert 1d to 2d arrays, and check for dimensionality and type consistency and convert if necessary.
# TODO: test vs out-of-bootstrap samples; should we split in the abstract layer itself and only pass the training set to the bootstrap generator?
# TODO: __init__ files for all submodules
# TODO: explain how _iter_test_mask is only kept for backwards compatibility and is not used in the bootstrap generator
# TODO: add option for block_length to be None, in which case it is set to the square root of the data length
# TODO: add option for block_length to be a list of integers, in which case it is used as the block length for each bootstrap sample
# TODO: Hierarchical Archimedean Copula


class BaseTimeSeriesBootstrap(metaclass=ABCMeta):
    """
    Base class for time series bootstrap cross-validators for time series data.

    Parameters
    ----------
    n_bootstraps : int, default=1
        The number of bootstrap samples to create.
    random_seed : int, default=42
        The seed used by the random number generator.

    Raises
    ------
    ValueError
        If n_bootstraps is not greater than 0.
    NotImplementedError
        If either _iter_test_masks or _generate_samples is not implemented in a derived class.
    """

    def __init__(self,  n_bootstraps: int = 1,
                 random_seed: int = 42) -> None:

        if n_bootstraps <= 0:
            raise ValueError(
                "Number of bootstrap iterations should be greater than 0")

        self.n_bootstraps = n_bootstraps
        self.random_seed = random_seed

    def _generate_samples(self, X: np.ndarray, random_seed: int, **kwargs) -> Iterator[np.ndarray]:
        """Generates bootstrapped samples directly.

        Should be implemented in derived classes if applicable.
        """
        for _ in range(self.n_bootstraps):
            indices, data = self._generate_samples_single_bootstrap(
                X, random_seed, **kwargs)
            data = np.concatenate(data, axis=0)
            random_seed += 1
            yield data

    @abstractmethod
    def _generate_samples_single_bootstrap(self, X: np.ndarray, random_seed: int, **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Generates list of bootstrapped indices and samples for a single bootstrap iteration.
        Should be implemented in derived classes.
        """

    def split(self, X: Union[np.ndarray, pd.DataFrame, List], y: Optional[np.ndarray] = None,
              groups: Optional[np.ndarray] = None) -> Iterator[Tuple[np.ndarray]]:
        """Generate indices to split data into training and test set."""
        X = np.asarray(X)
        if len(X.shape) < 2:
            X = np.expand_dims(X, 1)

        self._check_input(X)

        self.train_index, self.test_index = time_series_split(
            X, test_ratio=0.2)

        samples_iter = self._generate_samples(X, y, groups)

        for train_samples in samples_iter:
            yield train_samples  # , test_samples

    def _check_input(self, X):
        if np.any(np.diff([len(x) for x in X]) != 0):
            raise ValueError("All time series must be of the same length.")

    def get_n_splits(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
                     groups: Optional[np.ndarray] = None) -> int:
        """Returns the number of bootstrapping iterations."""
        return self.n_bootstraps


class BlockBootstrap(BaseTimeSeriesBootstrap):
    """
    Block Bootstrap base class for time series data.

    Parameters
    ----------
    block_length : int, default=None
        The length of the blocks to sample. If None, the block length is automatically set to the square root of the number of observations.

    Raises
    ------
    ValueError
        If block_length is not greater than 0.
    """

    def __init__(self, block_length: Optional[int] = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if block_length is not None:
            if block_length <= 0:
                raise ValueError("Block length should be greater than 0")

        self.block_length = block_length

    def _check_input(self, X):
        super()._check_input(X)
        if self.block_length is not None and self.block_length > X.shape[0]:
            raise ValueError(
                "block_length cannot be greater than the size of the input array X.")


class MovingBlockBootstrap(BlockBootstrap):
    def _generate_samples_single_bootstrap(self, X: np.ndarray, block_length: int, random_seed: Optional[int]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if random_seed is None:
            random_seed = self.random_seed
        block_indices, block_data = generate_block_indices_and_data(
            X=X, block_length=block_length, random_seed=random_seed, tapered_weights=np.array([]), block_weights=np.array([]))
        return block_indices, block_data


class StationaryBootstrap(BlockBootstrap):
    def _generate_samples_single_bootstrap(self, X: np.ndarray, block_length: int, random_seed: Optional[int]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if random_seed is None:
            random_seed = self.random_seed
        block_indices, block_data = generate_block_indices_and_data(
            X=X, block_length=block_length, random_seed=random_seed, tapered_weights=np.array([]), block_weights=np.array([]), block_length_distribution='geometric')
        return block_indices, block_data


class CircularBootstrap(BlockBootstrap):
    def _generate_samples_single_bootstrap(self, X: np.ndarray, block_length: int, random_seed: Optional[int]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if random_seed is None:
            random_seed = self.random_seed
        block_indices, block_data = generate_block_indices_and_data(
            X=X, block_length=block_length, random_seed=random_seed, tapered_weights=np.array([]), block_weights=np.array([]), wrap_around_flag=True)
        return block_indices, block_data


class NonOverlappingBootstrap(BlockBootstrap):
    def _generate_samples_single_bootstrap(self, X: np.ndarray, block_length: int, random_seed: Optional[int]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if random_seed is None:
            random_seed = self.random_seed
        block_indices, block_data = generate_block_indices_and_data(
            X=X, block_length=block_length, random_seed=random_seed, tapered_weights=np.array([]), block_weights=np.array([]), overlap_flag=False)
        return block_indices, block_data


base_block_bootstraps = [NonOverlappingBootstrap, MovingBlockBootstrap,
                         StationaryBootstrap, CircularBootstrap]


class BaseBlockBootstrap(BaseTimeSeriesBootstrap):
    def __init__(self, bootstrap_type: Optional[Type[BlockBootstrap]], block_length: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (bootstrap_type is not None) and (bootstrap_type not in base_block_bootstraps):
            raise ValueError(
                "bootstrap_type should be one of {}".format(base_block_bootstraps))
        self.bootstrap_type = bootstrap_type
        self.block_length = block_length

    def _generate_samples_single_bootstrap(self, X: np.ndarray, random_seed: Optional[int], **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if random_seed is None:
            random_seed = self.random_seed

        n = X.shape[0]
        block_length = self.block_length if self.block_length is not None else int(
            np.sqrt(n))

        if self.bootstrap_type is None:
            block_indices, block_data = generate_block_indices_and_data(
                X=X, block_length=block_length, random_seed=random_seed, tapered_weights=np.array([]), block_weights=np.array([]), **kwargs)
        else:
            block_indices, block_data = self.bootstrap_type._generate_samples_single_bootstrap(
                X=X, block_length=block_length, random_seed=random_seed)

        return block_indices, block_data


class AdaptiveBlockLengthBootstrap(BaseBlockBootstrap):
    def __init__(self,
                 block_length_bins: List[int],
                 error_function: Callable,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.block_length_bins = block_length_bins
        self.error_function = error_function

    def _select_block_length(self, X: np.ndarray) -> int:
        errors = [self.error_function(block_length, X)
                  for block_length in self.block_length_bins]
        best_block_length = self.block_length_bins[np.argmin(errors)]
        return best_block_length

    def _generate_samples_single_bootstrap(self, X: np.ndarray, random_seed: Optional[int], **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        block_length = self._select_block_length(X)
        return super()._generate_samples_single_bootstrap(X=X, random_seed=random_seed, block_length=block_length, **kwargs)


class TaperedBlockBootstrap(BaseBlockBootstrap):
    def __init__(self,
                 tapered_weights: Union[np.ndarray, callable] = np.bartlett,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tapered_weights = tapered_weights

    def _generate_samples_single_bootstrap(self, X: np.ndarray, random_seed: Optional[int], **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        return super()._generate_samples_single_bootstrap(X=X, random_seed=random_seed, tapered_weights=self.tapered_weights, **kwargs)


class BartlettsBootstrap(TaperedBlockBootstrap):
    def __init__(self, *args, **kwargs):
        super().__init__(tapered_weights=np.bartlett,
                         bootstrap_type=MovingBlockBootstrap, *args, **kwargs)


class HammingBootstrap(TaperedBlockBootstrap):
    def __init__(self, *args, **kwargs):
        super().__init__(tapered_weights=np.hamming,
                         bootstrap_type=MovingBlockBootstrap, *args, **kwargs)


class HanningBootstrap(TaperedBlockBootstrap):
    def __init__(self, *args, **kwargs):
        super().__init__(tapered_weights=np.hanning,
                         bootstrap_type=MovingBlockBootstrap, *args, **kwargs)


class BlackmanBootstrap(TaperedBlockBootstrap):
    def __init__(self, *args, **kwargs):
        super().__init__(tapered_weights=np.blackman,
                         bootstrap_type=MovingBlockBootstrap, *args, **kwargs)


# TODO: logic might need some changing to incorporate genearted block_indices into `generate_samples_markov`
class MarkovBootstrap(BaseBlockBootstrap):
    def __init__(self,
                 method: str,
                 n_clusters: int,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        if method not in ['random', 'clustering', 'hmm']:
            raise ValueError(
                "Method must be one of 'random', 'clustering', or 'hmm'")

        self.method = method
        self.n_clusters = n_clusters

    def _generate_samples_single_bootstrap(self, X: np.ndarray, random_seed: Optional[int], **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        block_indices, block_data = super()._generate_samples_single_bootstrap(X=X,
                                                                               random_seed=random_seed, **kwargs)
        # Generate markov samples from the bootstrap samples
        bootstrap_samples = generate_samples_markov(
            block_data, method=self.method, block_length=self.block_length, n_clusters=self.n_clusters, random_seed=random_seed)

        return block_indices, bootstrap_samples


class BaseBiasCorrectedBootstrap(BaseTimeSeriesBootstrap):
    def __init__(self, *args, statistic: Callable = np.mean, **kwargs):
        super().__init__(*args, **kwargs)
        self.statistic = statistic


class WholeBiasCorrectedBootstrap(BaseBiasCorrectedBootstrap):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias = None

    def _generate_samples_single_bootstrap(self, X: np.ndarray, random_seed: Optional[int], **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if random_seed is None:
            random_seed = self.random_seed

        if self.bias is None:
            self.bias = np.mean(self.statistic(X), axis=0, keepdims=True)

        bootstrap_samples = X - self.bias
        return [np.arange(X.shape[0])], [bootstrap_samples]


class BlockBiasCorrectedBootstrap(BaseBiasCorrectedBootstrap, BaseBlockBootstrap):
    def _generate_samples_single_bootstrap(self, X: np.ndarray, random_seed: Optional[int], **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        block_indices, block_data = super()._generate_samples_single_bootstrap(X=X,
                                                                               random_seed=random_seed, **kwargs)

        bootstrap_samples = []

        for block_data_iter in block_data:
            bias = np.mean(self.statistic(block_data_iter),
                           axis=0, keepdims=True)
            bootstrap_samples.append(block_data_iter - bias)

        return block_indices, bootstrap_samples


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


class BaseResidualBootstrap(BaseTimeSeriesBootstrap):
    def __init__(self, model_type: str, order: Optional[Union[int, List[int], Tuple[int, int, int], Tuple[int, int, int, int]]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_type = model_type.lower()
        if model_type == 'arch':
            raise ValueError(
                "Do not use ARCH models to fit the data; they are meant for fitting to residuals.")
        self.model_type = model_type
        # order is a tuple of (p, o, q) for ARIMA and (p, d, q, s) for SARIMAX. It is either a single int or a list of non-consecutive ints for AR, and an int for VAR and ARCH. If None, the best order is chosen via TSFitBestLag. Do note that TSFitBestLag only chooses the best lag, not the best order, so for the tuple values, it only chooses the best p, not the best (p, o, q) or (p, d, q, s). The rest of the values are set to 0.
        self.order = order


class WholeResidualBootstrap(BaseResidualBootstrap):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fit_model = None
        self.resids = None
        self.X_fitted = None
        self.coefs = None

    def _generate_samples_single_bootstrap(self, X: np.ndarray, random_seed: Optional[int], **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if random_seed is None:
            random_seed = self.random_seed

        if self.resids is None or self.X_fitted is None or self.order is None or self.coefs is None:
            fit_obj = TSFitBestLag(model_type=self.model_type, order=self.order,
                                   save_models=kwargs.get('save_models', False))
            self.fit_model = fit_obj.fit(X=X, exog=kwargs.get('exog', None))
            self.X_fitted = fit_obj.get_fitted_X()
            self.resids = fit_obj.get_residuals()
            self.order = fit_obj.get_order()
            self.coefs = fit_obj.get_coefs()

        # Resample residuals
        resampled_indices = generate_indices_random(
            self.resids.shape[0], random_seed)
        resampled_residuals = self.resids[resampled_indices]
        # Add the bootstrapped residuals to the fitted values
        bootstrap_samples = self.X_fitted + resampled_residuals
        return [resampled_indices], [bootstrap_samples]


class BlockResidualBootstrap(BaseResidualBootstrap, BaseBlockBootstrap):
    def _generate_samples_single_bootstrap(self, X: np.ndarray, random_seed: Optional[int], **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        block_indices, block_data = super()._generate_samples_single_bootstrap(X=X,
                                                                               random_seed=random_seed, **kwargs)

        bootstrap_samples = []

        for block_data_iter in enumerate(block_data):
            fit_obj = TSFitBestLag(model_type=self.model_type, order=self.order,
                                   save_models=kwargs.get('save_models', False))
            fit_obj.fit(
                X=block_data_iter, exog=kwargs.get('exog', None))
            X_fitted = fit_obj.get_fitted_X()
            resids = fit_obj.get_residuals()

            # Resample residuals
            resampled_indices = generate_indices_random(
                resids.shape[0], random_seed)
            resampled_resids = resids[resampled_indices]
            # Add the bootstrapped residuals to the fitted values
            bootstrap_samples_iter = X_fitted + resampled_resids
            bootstrap_samples.append(bootstrap_samples_iter)

        return block_indices, bootstrap_samples


VALID_MODELS = [AutoReg, ARIMA, SARIMAX, VAR, arch_model]

# TODO: return indices from `generate_samples_sieve`


class BaseSieveBootstrap(BaseResidualBootstrap):
    def __init__(self, resids_model_type: str, resids_order: Optional[Union[int, List[int], Tuple[int, int, int], Tuple[int, int, int, int]]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resids_model_type = resids_model_type.lower()
        self.resids_order = resids_order


class WholeSieveBootstrap(BaseSieveBootstrap):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.resids_coefs = None
        self.resids_fit_model = None

    def _generate_samples_single_bootstrap(self, X: np.ndarray, random_seed: Optional[int], **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        if random_seed is None:
            random_seed = self.random_seed

        if self.resids_order is None or self.resids_coefs is None:
            fit_obj = TSFitBestLag(model_type=self.model_type, order=self.order,
                                   save_models=kwargs.get('save_models', False))
            self.fit_model = fit_obj.fit(X=X, exog=kwargs.get('exog', None))
            self.X_fitted = fit_obj.get_fitted_X()
            self.resids = fit_obj.get_residuals()

            resids_fit_obj = TSFitBestLag(
                model_type=self.resids_model_type, order=self.resids_order, save_models=kwargs.get('save_models', False))
            self.resids_fit_model = resids_fit_obj.fit(
                self.resids, exog=None)
            self.resids_order = resids_fit_obj.get_order()
            self.resids_coefs = resids_fit_obj.get_coefs()

        bootstrap_samples = generate_samples_sieve(
            X_fitted=self.X_fitted, random_seed=random_seed, model_type=self.resids_model_type, resids_lags=self.resids_order, resids_coefs=self.resids_coefs, resids=self.resids, resids_fit_model=self.resids_fit_model)
        return [np.arange(X.shape[0])], [bootstrap_samples]


class BlockSieveBootstrap(BaseSieveBootstrap, BaseBlockBootstrap):
    def _generate_samples_single_bootstrap(self, X: np.ndarray, random_seed: Optional[int], **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        block_indices, block_data = super()._generate_samples_single_bootstrap(X=X,
                                                                               random_seed=random_seed, **kwargs)

        bootstrap_samples = []

        for block_data_iter in enumerate(block_data):
            fit_obj = TSFitBestLag(model_type=self.model_type, order=self.order,
                                   save_models=kwargs.get('save_models', False))
            fit_obj.fit(X=block_data_iter, exog=kwargs.get('exog', None))
            X_fitted = fit_obj.get_fitted_X()
            resids = fit_obj.get_residuals()

            resids_fit_obj = TSFitBestLag(
                model_type=self.resids_model_type, order=self.resids_order, save_models=kwargs.get('save_models', False))
            resids_fit_model = resids_fit_obj.fit(resids, exog=None)
            resids_order = resids_fit_obj.get_order()
            resids_coefs = resids_fit_obj.get_coefs()

            bootstrap_samples_iter = generate_samples_sieve(
                X_fitted=X_fitted, random_seed=random_seed, model_type=self.resids_model_type, resids_lags=resids_order, resids_coefs=resids_coefs, resids=resids, resids_fit_model=resids_fit_model)
            bootstrap_samples.append(bootstrap_samples_iter)

        return block_indices, bootstrap_samples


class BaseFractionalDifferencingBootstrap(BaseTimeSeriesBootstrap):
    """
    Implementation of the Fractional Differencing Bootstrap (FDB) method for time series data.

    Parameters
    ----------
    d: float, default=0.5
        The order of differencing to be applied on the time series before bootstrap sampling.
    """

    def __init__(self, diff_order: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)
        self.diff_order = diff_order
        if diff_order is None:
            self.fracdiff_transformer = FracdiffStat()
        else:
            if diff_order <= 0.5:
                self.fracdiff_transformer = Fracdiff(self.diff_order)
            else:
                raise ValueError("differencing order must be <= 0.5")


# Does not respect temporal order
class WholeFractionalDifferencingBootstrap(BaseFractionalDifferencingBootstrap):
    def _generate_samples_single_bootstrap(self, X: np.ndarray, random_seed: Optional[int], **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Fractionally difference the series, perform standard bootstrap on the differenced series,
        then apply the inverse fractional differencing to get the bootstrap sample.
        """
        X_diff = self.fracdiff_transformer.fit_transform(X)

        if random_seed is None:
            random_seed = self.random_seed
        # Resample residuals
        bootstrap_indices = generate_indices_random(
            X_diff.shape[0], random_seed)
        X_diff_bootstrapped = X_diff[bootstrap_indices]
        bootstrap_samples = self.fracdiff_transformer.inverse_transform(
            X_diff_bootstrapped)
        return bootstrap_indices, bootstrap_samples


class BlockFractionalDifferencingBootstrap(BaseFractionalDifferencingBootstrap, BaseBlockBootstrap):
    def _generate_samples_single_bootstrap(self, X: np.ndarray, random_seed: Optional[int], **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        block_indices, block_data = super()._generate_samples_single_bootstrap(X=X_diff,
                                                                               random_seed=random_seed, **kwargs)
        if random_seed is None:
            random_seed = self.random_seed

        bootstrap_samples = []
        for block_data_iter in block_data:
            data_diff = self.fracdiff_transformer.fit_transform(
                block_data_iter)
            # Resample residuals
            bootstrap_indices = generate_indices_random(
                data_diff.shape[0], random_seed)
            data_diff_bootstrapped = data_diff[bootstrap_indices]
            bootstrap_samples_iter = self.fracdiff_transformer.inverse_transform(
                data_diff_bootstrapped)
            bootstrap_samples.append(bootstrap_samples_iter)
        return block_indices, bootstrap_samples


def adf_test(X: np.ndarray) -> bool:
    """
    Run the Augmented Dickey-Fuller test on the time series to check for stationarity.
    If p-value < 0.05, return True. Else, return False.
    """
    result = adfuller(X)
    return result[1] < 0.05


######################

"""
. In a typical Markov bootstrap, the block length is set to 2 because we're considering pairs of observations to maintain the first order Markov property, which states that the future state depends only on the current state and not on the sequence of events that preceded it.

If you're looking to extend this to include a higher order Markov process, where the future state depends on more than just the immediate previous state, then yes, we would need a block_length parameter to indicate the order of the Markov process.

For instance, if you're considering a second order Markov process, the block_length would be 3, as each block would consist of three consecutive observations. In this case, the next state depends on the current state and the state before it. This can be generalized to an nth order Markov process, where the block_length would be n+1.

"""

'''
class SpectralBootstrap(BlockBootstrap):
    def _generate_block_indices(self, X: np.ndarray) -> List[np.ndarray]:
        return generate_block_indices_spectral(X, self.block_length, self.random_seed)
'''


class TSFit(BaseEstimator, RegressorMixin):
    """
    This class performs fitting for various time series models including 'ar', 'arima', 'sarima', 'var', and 'arch'.
    """

    def __init__(self, order: Union[int, List[int], Tuple[int, int, int], Tuple[int, int, int, int]], model_type: str, **kwargs) -> None:
        if model_type not in ['ar', 'arima', 'sarima', 'var', 'arch']:
            raise ValueError(
                f"Invalid model type '{model_type}', should be one of ['ar', 'arima', 'sarima', 'var', 'arch']")

        if type(order) == tuple and model_type not in ['arima', 'sarima']:
            raise ValueError(
                f"Invalid order '{order}', should be an integer for model type '{model_type}'")

        if type(order) == int and model_type in ['arima', 'sarima']:
            order = (order, 0, 0, 0)
            warnings.warn(
                f"{model_type.upper()} model requires a tuple of order (p, d, q, s), where d is the order of differencing and s is the seasonal period. Setting d=0, q=0 and s=0.")

        self.order = order
        self.model_type = model_type.lower()
        self.rescale_factors = {}
        self.model = None
        self.model_params = kwargs

    def get_params(self, deep=True):
        # deep argument ignored as no nested estimators are used.
        return {"order": self.order, "model_type": self.model_type, **self.model_params}

    def set_params(self, **params):
        # Iterate over all parameters, those not found in the model are considered model parameters
        self.model_params = {}
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.model_params[key] = value
        return self

    def __repr__(self):
        return f"TSFit(order={self.order}, model_type='{self.model_type}')"

    def fit_func(self, model_type):
        if model_type == 'arima':
            return fit_arima
        elif model_type == 'ar':
            return fit_ar
        elif model_type == 'var':
            return fit_var
        elif model_type == 'sarima':
            return fit_sarima
        elif model_type == 'arch':
            return fit_arch
        else:
            raise ValueError(f"Invalid model type {model_type}")

    @lru_cache(maxsize=None)
    def fit(self, X: np.ndarray, exog: Optional[np.ndarray] = None) -> Union[AutoRegResultsWrapper, ARIMAResultsWrapper, SARIMAXResultsWrapper, VARResultsWrapper, ARCHModelResult]:
        """
        Fit the chosen model to the data.

        Args:
            X: The input data.
            exog: Exogenous variables, optional.

        Raises:
            ValueError: If the model type or the model order is invalid.
        """
        # Check if the input shapes are valid
        if len(X.shape) != 2 or X.shape[1] < 1:
            raise ValueError(
                "X should be 2-D with the second dimension greater than or equal to 1.")
        if exog is not None:
            # checking whether X and exog have compatible shapes
            check_X_y(X, exog)
            if len(exog.shape) != 2 or exog.shape[1] < 1:
                raise ValueError(
                    "exog should be 2-D with the second dimension greater than or equal to 1.")

        def _rescale_inputs(X: np.ndarray, exog: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[float, Optional[List[float]]]]:
            def rescale_array(arr: np.ndarray) -> Tuple[np.ndarray, float]:
                variance = np.var(arr)
                rescale_factor = 1
                if variance < 1 or variance > 1000:
                    rescale_factor = np.sqrt(100 / variance)
                    arr_rescaled = arr * rescale_factor
                return arr_rescaled, rescale_factor

            X, x_rescale_factor = rescale_array(X)

            if exog is not None:
                exog_rescale_factors = []
                for i in range(exog.shape[1]):
                    exog[:, i], factor = rescale_array(exog[:, i])
                    exog_rescale_factors.append(factor)
            else:
                exog_rescale_factors = None

            return X, exog, (x_rescale_factor, exog_rescale_factors)

        fit_func = self.fit_func(self.model_type)

        if self.model_type == 'arch':
            X, exog, (x_rescale_factor,
                      exog_rescale_factors) = _rescale_inputs(X, exog)
            self.model = fit_func(
                X, self.order, exog=exog, **self.model_params)
            self.rescale_factors['x'] = x_rescale_factor
            self.rescale_factors['exog'] = exog_rescale_factors
        else:
            self.model = fit_func(
                X, self.order, exog=exog, **self.model_params)

        return self

    def get_coefs(self) -> np.ndarray:
        n_features = self.model.model.endog.shape[1] if len(
            self.model.model.endog.shape) > 1 else 1
        return self._get_coefs_helper(self.model, n_features)

    def get_residuals(self) -> np.ndarray:
        return self._get_residuals_helper(self.model)

    def get_fitted_X(self) -> np.ndarray:
        return self._get_fitted_X_helper(self.model)

    def get_order(self) -> Union[int, List[int], Tuple[int, int, int], Tuple[int, int, int, int]]:
        return self._get_order_helper(self.model)

    def predict(self, X: np.ndarray, n_steps: int = 1):
        # Check if the model is already fitted
        check_is_fitted(self, ['model'])
        if self.model_type == 'var':
            return self.model.forecast(X, n_steps)
        else:
            n_features = X.shape[1] if len(X.shape) > 1 else 1
            coefs = self.get_coefs().T.reshape(n_features, -1)
            X_lagged = self._lag(X, coefs.shape[1])
            return np.dot(X_lagged, coefs)

    def score(self, X: np.ndarray, y_true: np.ndarray):
        y_pred = self.predict(X)
        # Use r2 as the score
        return r2_score(y_true, y_pred)

    # These helper methods are internal and still take the model as a parameter.
    # They can be used by the public methods above which do not take the model parameter.

    def _get_coefs_helper(self, model, n_features) -> np.ndarray:
        if self.model_type == 'var':
            return model.params[1:].reshape(self.get_order(), n_features, n_features).transpose(1, 0, 2)
        elif self.model_type == 'ar':
            if isinstance(self.order, list):
                coefs = np.zeros((n_features, len(self.order)))
                for i, lag in enumerate(self.order):
                    coefs[:, i] = model.params[1 + i::len(self.order)]
            else:
                coefs = model.params[1:].reshape(n_features, self.order)
            return coefs
        elif self.model_type in ['arima', 'sarima', 'arch']:
            return model.params

    def _get_residuals_helper(self, model) -> np.ndarray:
        model_resid = model.resid

        # Ensure model_resid has the correct shape, (n, 1) or (n, k)
        if model_resid.ndim == 1:
            model_resid = model_resid.reshape(-1, 1)

        if self.model_type in ['ar', 'var']:
            max_lag = np.max(self.get_order())
            values_to_add_back = self.model.model.endog[:max_lag]

            # Ensure values_to_add_back has the same shape as model_resid
            if values_to_add_back.ndim != model_resid.ndim:
                values_to_add_back = values_to_add_back.reshape(-1, 1)

            model_resid = np.vstack((values_to_add_back, model_resid))

        if self.model_type == 'arch':
            model_resid = model_resid / self.rescale_factors['x']

        return model_resid

    def _get_fitted_X_helper(self, model) -> np.ndarray:
        model_fittedvalues = model.fittedvalues

        # Ensure model_fittedvalues has the correct shape, (n, 1) or (n, k)
        if model_fittedvalues.ndim == 1:
            model_fittedvalues = model_fittedvalues.reshape(-1, 1)

        if self.model_type in ['ar', 'var']:
            max_lag = np.max(self.get_order())
            values_to_add_back = self.model.model.endog[:max_lag]

            # Ensure values_to_add_back has the same shape as model_fittedvalues
            if values_to_add_back.ndim != model_fittedvalues.ndim:
                values_to_add_back = values_to_add_back.reshape(-1, 1)

            model_fittedvalues = np.vstack(
                (values_to_add_back, model_fittedvalues))

        if self.model_type == 'arch':
            return (model.resid + model.conditional_volatility) / self.rescale_factors['x']
        else:
            return model_fittedvalues

    def _get_order_helper(self, model) -> Union[int, List[int], Tuple[int, int, int], Tuple[int, int, int, int]]:
        return model.k_ar if self.model == 'var' else self.order

    def _lag(self, X: np.ndarray, n_lags: int):
        if len(X) < n_lags:
            raise ValueError(
                "Number of lags is greater than the length of the input data.")
        return np.column_stack([X[i:-(n_lags - i), :] for i in range(n_lags)])


# TODO: use the already created multidimensional versions of acf and pacf on numba_base to get this working for multivariate data
class RankLags:
    def __init__(self, X: np.ndarray, model_type: str, max_lag: int = 10, exog: Optional[np.ndarray] = None, save_models=False) -> None:
        self.X = X
        self.max_lag = max_lag
        self.model_type = model_type.lower()
        self.exog = exog
        self.save_models = save_models
        self.models = []

    def rank_lags_by_aic_bic(self) -> Tuple[np.ndarray, np.ndarray]:
        aic_values = []
        bic_values = []
        for lag in range(1, self.max_lag + 1):
            fit_obj = TSFit(order=lag, model_type=self.model_type)
            model = fit_obj.fit(X=self.X, exog=self.exog)
            if self.save_models:
                self.models.append(model)
            aic_values.append(model.aic)
            bic_values.append(model.bic)

        aic_ranked_lags = np.argsort(aic_values)
        bic_ranked_lags = np.argsort(bic_values)

        return aic_ranked_lags + 1, bic_ranked_lags + 1

    def rank_lags_by_pacf(self) -> np.ndarray:
        pacf_values = pacf(self.X, nlags=self.max_lag)[1:]
        ci = 1.96 / np.sqrt(len(self.X))
        significant_lags = np.where(np.abs(pacf_values) > ci)[0]
        return significant_lags + 1

    def estimate_conservative_lag(self) -> int:
        aic_ranked_lags, bic_ranked_lags = self.rank_lags_by_aic_bic()
        pacf_ranked_lags = self.rank_lags_by_pacf()
        highest_ranked_lags = set(aic_ranked_lags).intersection(
            bic_ranked_lags, pacf_ranked_lags)

        if not highest_ranked_lags:
            return aic_ranked_lags[-1]
        else:
            return min(highest_ranked_lags)

    def get_model(self, order) -> Union[AutoRegResultsWrapper, ARIMAResultsWrapper, SARIMAXResultsWrapper, VARResultsWrapper, ARCHModelResult]:
        return self.models[order - 1]


class TSFitBestLag(BaseEstimator, RegressorMixin):
    def __init__(self, model_type: str, max_lag: int = 10, order: Optional[Union[int, List[int], Tuple[int, int, int], Tuple[int, int, int, int]]] = None, save_models=False):
        self.model_type = model_type
        self.max_lag = max_lag
        self.order = order
        self.save_models = save_models
        self.rank_lagger = None
        self.ts_fit = None
        self.model = None

    def _compute_best_order(self, X) -> int:
        self.rank_lagger = RankLags(
            X=X, max_lag=self.max_lag, model_type=self.model_type, save_models=self.save_models)
        best_order = self.rank_lagger.estimate_conservative_lag()
        return best_order

    def fit(self, X: np.ndarray, exog: Optional[np.ndarray] = None) -> Union[AutoRegResultsWrapper, ARIMAResultsWrapper, SARIMAXResultsWrapper, VARResultsWrapper, ARCHModelResult]:
        if self.order is None:
            self.order = self._compute_best_order(X)
            if self.save_models:
                self.model = self.rank_lagger.get_model(self.order)
        self.ts_fit = TSFit(order=self.order, model_type=self.model_type)
        self.model = self.ts_fit.fit(X, exog=exog)
        return self

    def get_coefs(self) -> np.ndarray:
        return self.ts_fit.get_coefs()

    def get_residuals(self) -> np.ndarray:
        return self.ts_fit.get_residuals()

    def get_fitted_X(self) -> np.ndarray:
        return self.ts_fit.get_fitted_X()

    def get_order(self) -> Union[int, List[int], Tuple[int, int, int], Tuple[int, int, int, int]]:
        return self.ts_fit.get_order()

    def get_model(self) -> Union[AutoRegResultsWrapper, ARIMAResultsWrapper, SARIMAXResultsWrapper, VARResultsWrapper, ARCHModelResult]:
        if self.save_models:
            return self.rank_lagger.get_model(self.order)
        else:
            raise ValueError(
                'Models were not saved. Please set save_models=True during initialization.')

    def predict(self, X: np.ndarray, n_steps: int = 1):
        return self.ts_fit.predict(X, n_steps)

    def score(self, X: np.ndarray, y_true: np.ndarray):
        return self.ts_fit.score(X, y_true)
