from functools import lru_cache
from statsmodels.tsa.stattools import pacf, acf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.ar_model import AutoReg
from typing import Optional, Union, Callable, List, Tuple, Iterator, Type, Dict
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

        '''
        if not any([
            callable(getattr(self, "_iter_test_masks", None)),
            callable(getattr(self, "_generate_samples", None))
        ]) or all([
            callable(getattr(self, "_iter_test_masks", None)),
            callable(getattr(self, "_generate_samples", None))
        ]):
            raise NotImplementedError(
                "Either _iter_test_masks or _generate_samples (but not both) must be implemented in derived classes."
            )
        '''

    def _iter_test_masks(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                         groups: Optional[np.ndarray] = None, **kwargs) -> Iterator[np.ndarray]:
        """Returns a boolean mask corresponding to the test set."""
        test_mask = np.zeros(X.shape[0], dtype=bool)
        test_mask[self.test_index] = True
        for _ in range(self.n_bootstraps):
            yield test_mask

    def _generate_samples(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                          groups: Optional[np.ndarray] = None, **kwargs) -> Iterator[np.ndarray]:
        """Generates bootstrapped samples directly.

        Should be implemented in derived classes if applicable.
        """
        random_seed = self.random_seed
        for _ in range(self.n_bootstraps):
            block_indices, block_data = self._generate_samples_single_bootstrap(
                X, random_seed, **kwargs)
            random_seed += 1
            yield block_data

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

        # test_masks_iter = self._iter_test_masks(X, y, groups)
        samples_iter = self._generate_samples(X, y, groups)

        '''
        if test_masks_iter:
            # Use test masks to generate train and test indices
            n = X.shape[0]
            indices = np.arange(n)
            for test_mask in test_masks_iter:
                # test_indices = indices[test_mask]
                train_indices = np.setdiff1d(
                    indices, indices[test_mask], assume_unique=True)
                train_samples = X[train_indices]
                # test_samples = X[test_indices]
                yield train_samples  # , test_samples
        '''
        # elif samples_iter:
        # Generate bootstrapped samples directly
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

    '''
    @abstractmethod
    def _generate_block_indices(self, X: np.ndarray, block_length: int, random_seed: int) -> List[np.ndarray]:
        """Generates a list of index arrays for data blocks to be sampled.

        Should be implemented in derived classes.
        """

    def _iter_test_masks(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                         groups: Optional[np.ndarray] = None) -> Iterator[np.ndarray]:

        n = X.shape[0]
        block_length = self.block_length if self.block_length is not None else int(
            np.sqrt(n))
        n_blocks = n // block_length

        for _ in range(self.n_bootstraps):
            block_indices = self._generate_block_indices(
                X, block_length, self.random_seed)
            test_mask = np.zeros(n, dtype=bool)
            for block in block_indices[:n_blocks]:
                test_mask[block] = True

            # Handle remaining samples
            if n_blocks < len(block_indices):
                remaining_block = block_indices[n_blocks]
                remaining_block = remaining_block[:(
                    n - n_blocks * block_length)]
                test_mask[remaining_block] = True

            self.random_seed += 1
            yield test_mask
    '''


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


class NonOverlappingSubseriesBootstrap(BlockBootstrap):
    def _generate_samples_single_bootstrap(self, X: np.ndarray, block_length: int, random_seed: Optional[int]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if random_seed is None:
            random_seed = self.random_seed
        block_indices, block_data = generate_block_indices_and_data(
            X=X, block_length=block_length, random_seed=random_seed, tapered_weights=np.array([]), block_weights=np.array([]), overlap_flag=False)
        return block_indices, block_data


base_block_bootstraps = [NonOverlappingSubseriesBootstrap, MovingBlockBootstrap,
                         StationaryBootstrap, CircularBootstrap]


class AdaptiveBlockLengthBootstrap(BaseTimeSeriesBootstrap):
    def __init__(self,
                 block_length_bins: List[int],
                 error_function: Callable,
                 bootstrap_type: Optional[Type[BlockBootstrap]] = None,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if (bootstrap_type is not None) and (bootstrap_type not in base_block_bootstraps):
            raise ValueError(
                "bootstrap_type should be one of {}".format(base_block_bootstraps))
        self.bootstrap_type = bootstrap_type
        self.block_length_bins = block_length_bins
        self.error_function = error_function

    def _select_block_length(self, X: np.ndarray) -> int:
        errors = [self.error_function(block_length, X)
                  for block_length in self.block_length_bins]
        best_block_length = self.block_length_bins[np.argmin(errors)]
        return best_block_length

    def _generate_samples_single_bootstrap(self, X: np.ndarray, random_seed: Optional[int], **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if random_seed is None:
            random_seed = self.random_seed
        block_length = self._select_block_length(X)
        if self.bootstrap_type is None:
            block_indices, block_data = generate_block_indices_and_data(
                X=X, block_length=block_length, tapered_weights=np.array([]), block_weights=np.array([]), random_seed=random_seed, **kwargs)
        else:
            block_indices, block_data = self.bootstrap_type._generate_samples_single_bootstrap(
                X=X, block_length=block_length, random_seed=random_seed)
        return block_indices, block_data


class TaperedBlockBootstrap(BaseTimeSeriesBootstrap):
    def __init__(self,
                 block_length: int,
                 bootstrap_type: Optional[Type[BlockBootstrap]] = None,
                 tapered_weights: Union[np.ndarray, callable] = np.bartlett,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if (bootstrap_type is not None) and (bootstrap_type not in base_block_bootstraps):
            raise ValueError(
                "bootstrap_type should be one of {}".format(base_block_bootstraps))
        self.block_length = block_length
        self.bootstrap_type = bootstrap_type
        self.tapered_weights = tapered_weights

    def _generate_samples_single_bootstrap(self, X: np.ndarray, random_seed: Optional[int], **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if random_seed is None:
            random_seed = self.random_seed
        if self.bootstrap_type is None:
            block_indices, block_data = generate_block_indices_and_data(
                X=X, block_length=self.block_length, tapered_weights=self.tapered_weights, block_weights=np.array([]), random_seed=random_seed, **kwargs)
        else:
            block_indices, block_data = self.bootstrap_type._generate_samples_single_bootstrap(
                X=X, block_length=self.block_length, random_seed=self.random_seed, tapered_weights=self.tapered_weights)
        return block_indices, block_data


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


class MarkovBootstrap(BaseTimeSeriesBootstrap):
    def __init__(self,
                 method: str,
                 block_length: int,
                 n_clusters: int,
                 bootstrap_type: Optional[Type[BlockBootstrap]] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        if method not in ['random', 'clustering', 'hmm']:
            raise ValueError(
                "Method must be one of 'random', 'clustering', or 'hmm'")
        if (bootstrap_type is not None) and (bootstrap_type not in base_block_bootstraps):
            raise ValueError(
                "bootstrap_type should be one of {}".format(base_block_bootstraps))

        self.bootstrap_type = bootstrap_type
        self.method = method
        self.block_length = block_length
        self.n_clusters = n_clusters

    def _generate_samples_single_bootstrap(self, X: np.ndarray, random_seed: Optional[int], **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if random_seed is None:
            random_seed = self.random_seed
        # Get block indices from the specific bootstrap type
        if self.bootstrap_type is None:
            block_indices, block_data = generate_block_indices_and_data(
                X=X, block_length=self.block_length, random_seed=random_seed, tapered_weights=np.array([]), block_weights=np.array([]), **kwargs)
        else:
            block_indices, block_data = self.bootstrap_type._generate_samples_single_bootstrap(
                X=X, block_length=self.block_length, random_seed=random_seed)

        # Generate markov samples from the bootstrap samples
        in_bootstrap_samples = generate_samples_markov(
            block_data, method=self.method, block_length=self.block_length, n_clusters=self.n_clusters, random_seed=random_seed)

        return block_indices, in_bootstrap_samples


class BaseBiasCorrectedBootstrap(BaseTimeSeriesBootstrap):
    def __init__(self, *args, statistic: Callable, **kwargs):
        super().__init__(*args, **kwargs)
        self.statistic = statistic


class WholeBiasCorrectedBootstrap(BaseBiasCorrectedBootstrap):
    def _generate_samples(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                          groups: Optional[np.ndarray] = None) -> Iterator[np.ndarray]:
        num_samples = X.shape[0]
        random_seed = self.random_seed
        for _ in range(self.n_bootstraps):
            bias = np.mean(self.statistic(X), axis=0)
            in_bootstrap_samples_bias_corrected = X - bias
            random_seed += 1
            yield in_bootstrap_samples_bias_corrected


class BlockBiasCorrectedBootstrap(BaseBiasCorrectedBootstrap):
    def __init__(self, bootstrap_type: BlockBootstrap, block_length: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if bootstrap_type not in base_block_bootstraps:
            raise ValueError(
                "bootstrap_type should be one of {}".format(base_block_bootstraps))
        self.bootstrap_type = bootstrap_type
        self.block_length = block_length

    def _generate_samples_single_bootstrap(self, X: np.ndarray, random_seed: Optional[int], **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if random_seed is None:
            random_seed = self.random_seed

        if self.bootstrap_type is None:
            block_indices, block_data = generate_block_indices_and_data(
                X=X, block_length=self.block_length, random_seed=random_seed, tapered_weights=np.array([]), block_weights=np.array([]), **kwargs)
        else:
            block_indices, block_data = self.bootstrap_type._generate_samples_single_bootstrap(
                X=X, block_length=self.block_length, random_seed=random_seed)

        train_bias_corrected = np.zeros_like(block_data)

        for i, block in enumerate(block_indices):
            bias = np.mean(self.statistic(block_data[block]), axis=0)
            train_bias_corrected[block] = block_data[block] - bias

        yield block_indices, train_bias_corrected


class BaseHACBootstrap(BaseTimeSeriesBootstrap):
    def __init__(self, *args, bandwidth: Optional[int] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if bandwidth is not None:
            if not isinstance(bandwidth, int):
                raise TypeError("Bandwidth must be an integer")
            if bandwidth < 1:
                raise ValueError(
                    "Bandwidth must be a positive integer greater than or equal to 1")
        self.bandwidth = bandwidth

    def _generate_bootstrapped_errors(self, X: np.ndarray, random_seed: int) -> np.ndarray:

        if self.bandwidth is None:
            h = int(np.ceil(4 * (X.shape[0] / 100) ** (2 / 9)))
        else:
            h = self.bandwidth

        return generate_hac_errors(X, h, random_seed)


class WholeHACBootstrap(BaseHACBootstrap):
    def _generate_samples_single_bootstrap(self, X: np.ndarray, random_seed: Optional[int], **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if random_seed is None:
            random_seed = self.random_seed

        in_bootstrap_errors = self._generate_bootstrapped_errors(
            X, random_seed)
        in_bootstrap_samples_hac = X + in_bootstrap_errors
        yield in_bootstrap_samples_hac


class BlockHACBootstrap(BaseHACBootstrap):
    def __init__(self, bootstrap_type: BlockBootstrap, block_length: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if bootstrap_type not in base_block_bootstraps:
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
            block_indices, in_bootstrap_samples = generate_block_indices_and_data(
                X=X, block_length=block_length, random_seed=random_seed, tapered_weights=np.array([]), block_weights=np.array([]), **kwargs)
        else:
            block_indices, in_bootstrap_samples = self.bootstrap_type._generate_samples_single_bootstrap(
                X=X, block_length=block_length, random_seed=random_seed)

        bootstrapped_errors = self._generate_bootstrapped_errors(
            in_bootstrap_samples)
        bootstrapped_samples = in_bootstrap_samples + bootstrapped_errors

        yield block_indices, bootstrapped_samples


class BaseResidualBootstrap(BaseTimeSeriesBootstrap):
    def __init__(self, model_type: str, order: Optional[Union[int, Tuple[int, int, int], Tuple[int, int, int, int]]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_type = model_type.lower()
        if model_type == 'arch':
            raise ValueError(
                "Do not use ARCH models to fit the data; they are meant for fitting to residuals.")
        self.model_type = model_type
        self.fit_obj = TSFit(order=order, model_type=self.model_type)
        self.residuals = None
        self.X_fitted = None
        self.order = None

    def _generate_samples_single_bootstrap(self, X: np.ndarray, random_seed: Optional[int]) -> Tuple[np.ndarray, np.ndarray, int]:
        if random_seed is None:
            random_seed = self.random_seed

        if self.residuals is None or self.X_fitted is None:
            model = self.fit_obj.get_fit_func(X, self.order)
            self.X_fitted = self.fit_obj.get_fitted_X(model)
            self.residuals = self.fit_obj.get_residuals(model)
            self.order = self.fit_obj.get_order(model)


class WholeResidualBootstrap(BaseResidualBootstrap):
    def _generate_samples_single_bootstrap(self, X: np.ndarray, random_seed: Optional[int]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        super()._generate_samples_single_bootstrap(X, random_seed)
        in_bootstrap_indices, in_bootstrap_samples = generate_samples_residual(
            self.X_fitted, self.residuals, self.order, random_seed=random_seed)
        return in_bootstrap_indices, in_bootstrap_samples


class BlockResidualBootstrap(BaseResidualBootstrap):
    def __init__(self, bootstrap_type: Optional[Type[BlockBootstrap]], block_length: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (bootstrap_type is not None) and (bootstrap_type not in base_block_bootstraps):
            raise ValueError(
                "bootstrap_type should be one of {}".format(base_block_bootstraps))
        self.bootstrap_type = bootstrap_type
        self.block_length = block_length

    def _generate_samples_single_bootstrap(self, X: np.ndarray, random_seed: Optional[int], **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        super()._generate_samples_single_bootstrap(X, random_seed)

        if self.bootstrap_type is None:
            block_indices, block_data = generate_block_indices_and_data(
                X=self.residuals, block_length=self.block_length, random_seed=random_seed, tapered_weights=np.array([]), block_weights=np.array([]), **kwargs)
        else:
            block_indices, block_data = self.bootstrap_type._generate_samples_single_bootstrap(
                X=self.residuals, block_length=self.block_length, random_seed=random_seed)

        in_bootstrap_samples = self.X_fitted + block_data
        # Prepend the first 'self.order' original observations to the bootstrapped series
        extended_bootstrapped_samples = np.vstack(
            (X[:self.order], in_bootstrap_samples))
        # Prepend the indices of the first 'self.order' original observations to resampled_indices
        initial_indices = np.arange(self.order, dtype=np.int64)
        extended_block_indices = np.hstack(
            (initial_indices, block_indices + self.order))
        yield extended_block_indices, extended_bootstrapped_samples


# TODO: return indices from `generate_samples_sieve`
class BaseSieveBootstrap(BaseResidualBootstrap):
    def __init__(self, resids_model_type: str, resids_order: Optional[Union[int, Tuple[int, int, int], Tuple[int, int, int, int]]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resids_model_type = resids_model_type.lower()
        self.fit_obj_resids = TSFit(order=resids_order,
                                    model_type=self.resids_model_type)
        self.resids_order = None
        self.coefs = None
        self.resids_coefs = None

    def _generate_samples_single_bootstrap(self, X: np.ndarray, random_seed: Optional[int]) -> Tuple[np.ndarray, np.ndarray, int]:
        if random_seed is None:
            random_seed = self.random_seed

        if self.residuals is None or self.X_fitted is None:
            model = self.fit_obj.get_fit_func(X, self.order)
            self.X_fitted = self.fit_obj.get_fitted_X(model)
            self.residuals = self.fit_obj.get_residuals(model)
            self.k_ar = self.fit_obj.get_order(model)


class SieveBootstrap(BaseTimeSeriesBootstrap):
    def __init__(self, order: Optional[Union[int, Tuple[int, int, int], Tuple[int, int, int, int]]], resids_order: Optional[Union[int, Tuple[int, int, int], Tuple[int, int, int, int]]], model_type: str, resids_model_type: str = 'ar', *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_type = model_type.lower()
        if model_type == "arch":
            raise ValueError(
                "Do not use ARCH models to fit the data; they are meant for fitting to residuals")
        self.order = order
        self.resids_order = resids_order
        self.model_type = model_type
        self.fit_obj = TSFit(order=self.order, model_type=self.model_type)
        # In tridational sieve bootstrap, the residuals are generated from an AR(p) model. However, we extend this to allow a number of different models to be used for fitting the residuals.
        self.resid_model_type = resids_model_type.lower()
        self.fit_obj_resids = TSFit(
            order=self.resids_order, model_type=self.resid_model_type)
        self.residuals = None
        self.coefs = None
        self.resids_coefs = None
        self.k_ar = None
        self.k_resids_ar = None

    def _generate_samples_single_bootstrap(self, X: np.ndarray, random_seed: Optional[int], **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if random_seed is None:
            random_seed = self.random_seed

        n_samples, n_features = X.shape

        if self.residuals is None or self.coefs is None or self.resids_coefs is None or self.residuals is None or self.resids_coefs is None:
            # Fit a multivariate autoregressive model of order self.order
            model = self.fit_obj.get_fit_func(X=X)
            self.coefs = self.fit_obj.get_coefs(model, n_features)
            self.residuals = self.fit_obj.get_residuals(model)
            self.k_ar = self.fit_obj.get_order(model)

            # Fit an AR model to the residuals
            model = self.fit_obj_resids.get_fit_func(X=self.residuals)
            n_features_resids = 1 if self.residuals.ndim == 1 else self.residuals.shape[1]
            self.resids_coefs = self.fit_obj_resids.get_coefs(
                model, n_features_resids)
            self.k_resids_ar = self.fit_obj_resids.get_order(model)

        bootstrap_samples = generate_samples_sieve(
            X=X, order=self.order, coefs=self.coefs, resids=self.residuals, resids_order=self.resids_order, resids_coefs=self.resids_coefs, random_seed=random_seed)
        yield bootstrap_samples


class TSFit():
    """
    This class performs fitting for various time series models including 'ar', 'arima', 'sarima', 'var', and 'arch'.
    """

    def __init__(self, order: Union[int, List[int], Tuple[int, int, int], Tuple[int, int, int, int]], model_type: str):
        """
        Initialize TSFit with model order, model type, and maximum lag.

        Args:
            order: The order of the model. Can be int, list, or tuple depending on the model type.
            model_type: The type of the model. Can be 'ar', 'arima', 'sarima', 'var', or 'arch'.

        Raises:
            ValueError: If model_type is not one of the allowed types.
        """
        self.order = order
        self.model_type = model_type.lower()
        self.rescale_factors = {}
        self.model = None
        if self.model_type not in ['ar', 'arima', 'sarima', 'var', 'arch']:
            raise ValueError(
                f"Invalid model type '{self.model_type}', should be one of ['ar', 'arima', 'sarima', 'var', 'arch']")

    def get_fit_func(self):
        """
        Fetch the appropriate fit function based on the model type.

        Returns:
            The fitting function.

        Raises:
            ValueError: If the model type or the model order is invalid.
        """
        fit_funcs = {
            'arima': fit_arima,
            'ar': fit_ar,
            'var': fit_var,
            'sarima': fit_sarima,
            'arch': fit_arch
        }

        if type(self.order) == tuple and self.model_type not in ['arima', 'sarima']:
            raise ValueError(
                f"Invalid order '{self.order}', should be an integer for model type '{self.model_type}'")

        if type(self.order) == int and self.model_type in ['arima', 'sarima']:
            self.order = (self.order, 0, 0, 0)
            raise Warning(
                f"{self.model_type.upper()} model requires a tuple of order (p, d, q, s), where d is the order of differencing and s is the seasonal period. Setting d=0, q=0 and s=0.")

        return fit_funcs.get(self.model_type)

    @lru_cache(maxsize=None)
    def fit_model(self, X: np.ndarray, exog: Optional[np.ndarray] = None, **kwargs):
        """
        Fit the chosen model to the data.

        Args:
            X: The input data.
            exog: Exogenous variables, optional.

        Raises:
            ValueError: If the model type or the model order is invalid.
        """
        fit_func = self.get_fit_func()

        if self.model_type == 'arch':
            X, exog, (x_rescale_factor,
                      exog_rescale_factors) = self._rescale_inputs(X, exog)
            self.model = fit_func(X, self.order, exog=exog)
            self.rescale_factors['x'] = x_rescale_factor
            self.rescale_factors['exog'] = exog_rescale_factors
        else:
            self.model = fit_func(X, self.order, exog=exog, **kwargs)

        return self.model

    def get_coefs(self, n_features):
        # Now uses self.model
        return self._get_coefs_helper(self.model, n_features)

    def get_residuals(self):
        # Now uses self.model
        return self._get_residuals_helper(self.model)

    def get_fitted_X(self):
        # Now uses self.model
        return self._get_fitted_X_helper(self.model)

    def get_order(self):
        # Now uses self.model
        return self._get_order_helper(self.model)

    # These helper methods are internal and still take the model as a parameter.
    # They can be used by the public methods above which do not take the model parameter.
    def _get_coefs_helper(self, model, n_features):
        if self.model_type == 'var':
            return model.params[1:].reshape(self.order, n_features, n_features).transpose(1, 0, 2)
        elif self.model_type == 'ar':
            return model.params[1:].reshape(n_features, self.order)
        elif self.model_type in ['arima', 'sarima', 'arch']:
            return model.params
        else:
            raise ValueError(f"Invalid model type '{self.model_type}'")

    def _get_residuals_helper(self, model):
        if self.model_type == 'arch':
            return model.resid / self.rescale_factors['x']
        return model.resid

    def _get_fitted_X_helper(self, model):
        if self.model_type == 'arch':
            return (model.resid + model.conditional_volatility) / self.rescale_factors['x']
        return model.fittedvalues

    def _get_order_helper(self, model):
        return model.order

    @staticmethod
    def _rescale_inputs(X: np.ndarray, exog: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[float, Optional[List[float]]]]:
        def rescale_array(arr: np.ndarray) -> Tuple[np.ndarray, float]:
            variance = np.var(arr)
            rescale_factor = 1
            if variance < 1 or variance > 1000:
                rescale_factor = np.sqrt(100 / variance)
                arr = arr * rescale_factor
            return arr, rescale_factor

        X, x_rescale_factor = rescale_array(X)

        if exog is not None:
            exog_rescale_factors = []
            for i in range(exog.shape[1]):
                exog[:, i], factor = rescale_array(exog[:, i])
                exog_rescale_factors.append(factor)
        else:
            exog_rescale_factors = None

        return X, exog, (x_rescale_factor, exog_rescale_factors)


class RankLags():
    """
    When we find the best lag using this model_type, we return just one integer. If the user wants to pass in a list of lags, they need to either modify this model_type to return a list of lags, or pass in a list of lags to the TSFit object.
    """

    def __init__(self, X: np.ndarray, model_type: str, max_lag: int = 10, exog: Optional[np.ndarray] = None, save_models=False):
        self.X = X
        self.max_lag = max_lag
        self.model_type = model_type.lower()
        self.exog = exog
        self.save_models = save_models
        self.models = []

    def rank_lags_by_aic_bic(self):
        aic_values = []
        bic_values = []
        for lag in range(1, self.max_lag + 1):
            fit_obj = TSFit(order=lag, model_type=self.model_type)
            model = fit_obj.fit_model(X=self.X, exog=self.exog)
            if self.save_models:
                self.models.append(model)
            aic_values.append(model.aic)
            bic_values.append(model.bic)

        aic_ranked_lags = np.argsort(aic_values)
        bic_ranked_lags = np.argsort(bic_values)

        return aic_ranked_lags + 1, bic_ranked_lags + 1

    def rank_lags_by_pacf(self):
        # Compute PACF values
        pacf_values = pacf(self.X, nlags=self.max_lag)[1:]  # exclude lag 0

        # Calculate the confidence interval
        ci = 1.96 / np.sqrt(len(self.X))

        # Select only the lags where the absolute PACF value is greater than the confidence interval
        significant_lags = np.where(np.abs(pacf_values) > ci)[0]

        return significant_lags + 1

    def estimate_conservative_lag(self):
        aic_ranked_lags, bic_ranked_lags = self.rank_lags_by_aic_bic()
        pacf_ranked_lags = self.rank_lags_by_pacf()

        # Return the minimum of the last (highest) ranked lag that is common in all three metrics
        highest_ranked_lags = set(aic_ranked_lags).intersection(
            bic_ranked_lags, pacf_ranked_lags)

        if not highest_ranked_lags:
            # return aic lag if there is no common lag
            return aic_ranked_lags[-1]
        else:
            return min(highest_ranked_lags)

    def get_model(self, order):
        return self.models[order - 1]


class TSFitBestLag():
    """
    This class computes the best order for a model if no order is provided, 
    then uses the order to fit a time series model of a given type.
    """

    def __init__(self, X: np.ndarray, model_type: str, max_lag: int = 10, order: Optional[Union[int, List[int], Tuple[int, int, int], Tuple[int, int, int, int]]] = None, exog: Optional[np.ndarray] = None, save_models=False):
        self.X = X
        self.model_type = model_type
        self.max_lag = max_lag
        self.order = order
        self.exog = exog
        self.save_models = save_models

        if self.order is None:
            self.order = self._compute_best_order()
            if self.save_models:
                self.model = self.rank_lagger.get_model(self.order)

        self.ts_fit = TSFit(order=self.order, model_type=self.model_type)

    def _compute_best_order(self) -> int:
        self.rank_lagger = RankLags(
            X=self.X, max_lag=self.max_lag, model_type=self.model_type, save_models=self.save_models)
        best_order = self.rank_lagger.estimate_conservative_lag()
        return best_order

    def get_fit_func(self, **kwargs) -> Callable:
        if self.model is None:
            self.model = self.ts_fit.get_fit_func(
                self.X, exog=self.exog, **kwargs)
        return self.model

    def get_coefs(self, n_features: int):
        return self.ts_fit.get_coefs(self.model, n_features)

    def get_residuals(self):
        return self.ts_fit.get_residuals(self.model)

    def get_fitted_X(self):
        return self.ts_fit.get_fitted_X(self.model)

    def get_order(self):
        return self.ts_fit.get_order(self.model)

    def get_model(self):
        if self.save_models:
            return self.rank_lagger.get_model(self.order)
        else:
            raise ValueError(
                'Models were not saved. Please set save_models=True during initialization.')


'''
class WildBootstrap(BaseTimeSeriesBootstrap):
    def _iter_test_masks(self, X: np.ndarray) -> Iterator[np.ndarray]:
        for _ in range(self.n_bootstraps):
            boolean_mask = generate_boolean_mask(
                X, self.random_seed)
            self.random_seed += 1
            yield boolean_mask


class PolynomialFitBootstrap(BaseTimeSeriesBootstrap):
    def __init__(self, degree: int, **kwargs):
        super().__init__(**kwargs)
        self.degree = degree

    def _iter_test_masks(self, X: np.ndarray) -> Iterator[np.ndarray]:
        for _ in range(self.n_bootstraps):
            test_mask = generate_test_mask_polynomial_fit_bootstrap(
                X, self.degree, self.random_seed)
            self.random_seed += 1
            yield test_mask


class BandedBootstrap(BaseTimeSeriesBootstrap):
    def __init__(self, band_width: int, **kwargs):
        super().__init__(**kwargs)
        self.band_width = band_width

    def _iter_test_masks(self, X: np.ndarray) -> Iterator[np.ndarray]:
        for _ in range(self.n_bootstraps):
            test_mask = generate_test_mask_banded(
                X, self.band_width, self.random_seed)
            self.random_seed += 1
            yield test_mask





class BayesianBootstrap(BaseTimeSeriesBootstrap):
    def _iter_test_masks(self, X: np.ndarray) -> Iterator[np.ndarray]:
        for _ in range(self.n_bootstraps):
            test_mask = generate_test_mask_bayesian(X, self.random_seed)
            self.random_seed += 1
            yield test_mask


class SubsamplingBootstrap(BaseTimeSeriesBootstrap):
    def __init__(self, sample_fraction: float, **kwargs):
        super().__init__(**kwargs)
        self.sample_fraction = sample_fraction

    def _iter_test_masks(self, X: np.ndarray) -> Iterator[np.ndarray]:
        for _ in range(self.n_bootstraps):
            test_mask = generate_test_mask_subsampling(
                X, self.sample_fraction, self.random_seed)
            self.random_seed += 1
            yield test_mask


class PoissonBootstrap(BaseTimeSeriesBootstrap):
    def _iter_test_masks(self, X: np.ndarray) -> Iterator[np.ndarray]:
        for _ in range(self.n_bootstraps):
            test_mask = generate_test_mask_poisson_bootstrap(
                X, self.random_seed)
            self.random_seed += 1
            yield test_mask
'''

######################

'''
class FractionalBootstrap(BaseTimeSeriesBootstrap):
    def __init__(self, d: float, block_length: int, block_indices_func_provided: bool = False, block_indices_func: Optional[Callable] = None, **kwargs):
        super().__init__(**kwargs)
        self.d = d
        self.block_length = block_length
        self.block_indices_func_provided = block_indices_func_provided
        self.block_indices_func = block_indices_func

    def _iter_test_masks(self, X: np.ndarray) -> Iterator[np.ndarray]:
        for _ in range(self.n_bootstraps):
            yield generate_test_mask_fractional_diff(X, self.d, self.block_length, self.block_indices_func_provided, self.block_indices_func, self.random_seed)


class BaseBiasCorrectedBootstrap(BaseTimeSeriesBootstrap):
    def __init__(self, *args, statistic: Callable, **kwargs):
        super().__init__(*args, **kwargs)
        self.statistic = statistic


class WholeBiasCorrectedBootstrap(BaseBiasCorrectedBootstrap):
    def _generate_samples(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                          groups: Optional[np.ndarray] = None) -> Iterator[np.ndarray]:
        num_samples = X.shape[0]
        for _ in range(self.n_bootstraps):
            in_bootstrap_indices = generate_indices_random(
                num_samples, self.random_seed)
            out_of_bootstrap_indices = np.array(
                list(set(np.arange(X.shape[0])) - set(in_bootstrap_indices)))
            in_bootstrap_samples = X[in_bootstrap_indices]
            bias = np.mean(self.statistic(in_bootstrap_samples), axis=0)
            in_bootstrap_samples_bias_corrected = in_bootstrap_samples - bias
            self.random_seed += 1
            yield in_bootstrap_samples_bias_corrected, X[out_of_bootstrap_indices]


class BlockBiasCorrectedBootstrap(BaseBiasCorrectedBootstrap):
    def __init__(self, bootstrap_type: BlockBootstrap, block_length: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if bootstrap_type not in base_block_bootstraps:
            raise ValueError(
                "bootstrap_type should be one of {}".format(base_block_bootstraps))
        self.bootstrap_type = bootstrap_type
        self.block_length = block_length

    def _generate_samples(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                          groups: Optional[np.ndarray] = None) -> Iterator[np.ndarray]:
        for _ in range(self.n_bootstraps):
            block_indices = self.bootstrap_type._generate_block_indices(
                X, self.block_length, self.random_seed)
            train = np.concatenate([X[block] for block in block_indices])
            train_bias_corrected = np.zeros_like(train)

            for i, block in enumerate(block_indices):
                bias = np.mean(self.statistic(train[block]), axis=0)
                train_bias_corrected[block] = train[block] - bias

            self.random_seed += 1
            yield train_bias_corrected, X[np.setdiff1d(np.arange(X.shape[0]), train)]


class BaseFractionalBootstrap(BaseTimeSeriesBootstrap):
    def __init__(self, diff_order: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diff_order = diff_order


class WholeFractionalBootstrap(BaseFractionalBootstrap):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_length = 1
        self.block_indices_func_provided = False
        self.block_indices_func = None

    def _iter_test_masks(self, X: np.ndarray) -> Iterator[np.ndarray]:
        for _ in range(self.n_bootstraps):
            mask = self.generate_test_mask_fractional_diff(X)
            yield mask


class BlockFractionalBootstrap(BaseFractionalBootstrap):
    def __init__(self, bootstrap_type: BlockBootstrap, block_length: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if bootstrap_type not in base_block_bootstraps:
            raise ValueError(
                "bootstrap_type should be one of {}".format(base_block_bootstraps))
        self.bootstrap_type = bootstrap_type
        self.block_length = block_length

    def _generate_samples(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                          groups: Optional[np.ndarray] = None) -> Iterator[np.ndarray]:
        # Apply fractional differencing
        X = fractional_diff_numba(X, self.diff_order)

        for _ in range(self.n_bootstraps):
            # Generate block indices using the specified block_indices_func
            block_indices = self.bootstrap_type._generate_block_indices(
                X, self.block_length, self.random_seed)
            in_bootstrap_indices = np.concatenate(
                [block for block in block_indices])
            out_of_bootstrap_indices = np.setdiff1d(
                np.arange(X.shape[0]), in_bootstrap_indices)
            out_of_bootstrap_samples = X[out_of_bootstrap_indices]

            train = np.concatenate([X[block] for block in block_indices])
            train_bias_corrected = np.zeros_like(train)

            for i, block in enumerate(block_indices):
                bias = np.mean(self.statistic(train[block]), axis=0)
                train_bias_corrected[block] = train[block] - bias

            self.random_seed += 1
            yield train_bias_corrected, out_of_bootstrap_samples

'''

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
