from __future__ import annotations
from scipy.stats import distributions
from fracdiff.sklearn import Fracdiff, FracdiffStat
from typing import Callable, List, Optional, Tuple, Union, Iterator

from statsmodels.tsa.ar_model import AutoReg
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from numpy.random import Generator

from utils.block_length_sampler import BlockLengthSampler
from src.block_resampler import BlockResampler
from src.block_generator import BlockGenerator
from utils.tsfit import *
from utils.tsmodels import *
from src.bootstrap_numba import *
from utils.odds_and_ends import check_generator, time_series_split


class BaseTimeSeriesBootstrap(metaclass=ABCMeta):
    """
    Base class for time series bootstrapping.
    """

    def __init__(self,  n_bootstraps: int = 10,
                 random_state: Optional[Union[Generator, int]] = None) -> None:
        """
        Parameters
        ----------
        n_bootstraps : int, default=10
            The number of bootstrap samples to create.
        random_state : int or Generator, default=np.random.default_rng()
            The random number generator or seed used to generate the bootstrap samples.
        """
        self.n_bootstraps = n_bootstraps
        self.rng = random_state

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, seed: Optional[Union[Generator, int]]) -> None:
        if seed is None:
            rng = np.random.default_rng()
        elif not (isinstance(rng, Generator) or isinstance(rng, int)):
            raise TypeError(
                'The random number generator must be an instance of the numpy.random.Generator class, or an integer.')
        self._rng = check_generator(seed)

    @property
    def n_bootstraps(self):
        return self._n_bootstraps

    @n_bootstraps.setter
    def n_bootstraps(self, value):
        if not isinstance(value, int):
            raise TypeError(
                'The number of bootstrap iterations must be an integer.')
        if not value > 0:
            raise ValueError(
                'The number of bootstrap iterations must be greater than 0.')
        self._n_bootstraps = value

    def _generate_samples(self, X: np.ndarray, return_indices: bool = False) -> Iterator[np.ndarray]:
        """Generates bootstrapped samples directly.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Yields
        ------
        Iterator[np.ndarray]
            An iterator over the bootstrapped samples.

        """
        for _ in range(self.n_bootstraps):
            indices, data = self._generate_samples_single_bootstrap(X)
            data = np.concatenate(data, axis=0)
            if return_indices:
                yield indices, data
            else:
                yield data

    @abstractmethod
    def _generate_samples_single_bootstrap(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Generates list of bootstrapped indices and samples for a single bootstrap iteration.
        Should be implemented in derived classes.
        """

    def split(self, X: Union[np.ndarray, pd.DataFrame, List], return_indices: bool = False, y: Optional[np.ndarray] = None,
              groups: Optional[np.ndarray] = None) -> Iterator[np.ndarray] | Iterator[Tuple[List[np.ndarray], np.ndarray]]:
        """Generate indices to split data into training and test set."""
        X = np.asarray(X)
        if len(X.shape) < 2:
            X = np.expand_dims(X, 1)

        self._check_input(X)

        X_train, X_test = time_series_split(
            X, test_ratio=0.2)

        samples_iter = self._generate_samples(
            X=X_train, return_indices=return_indices)

        for train_samples in samples_iter:
            yield train_samples

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

    def __init__(self, n_bootstraps: int = 10, block_length: Optional[int] = None, block_length_distribution: Optional[str] = None, wrap_around_flag: bool = False, overlap_flag: bool = False, combine_generation_and_sampling_flag: bool = False, random_state: Optional[Union[int, Generator]] = None, **kwargs) -> None:
        """
        Parameters
        ----------
        block_length : int or None, default=None
            The length of the blocks to sample. If None, the block length is automatically set to the square root of the number of observations.
        block_length_distribution : str or None, default=None
            The block length distribution function to use. If None, the block length distribution is automatically set to None (internally converted to "none" in BlockLengthSampler).
        wrap_around_flag : bool, default=False
            Whether to wrap around the input data when generating blocks.
        overlap_flag : bool, default=False
            Whether to allow blocks to overlap.
        combine_generation_and_sampling : bool, default=False

        Additional Parameters
        -----
        overlap_length : int or None
            The length of the overlap between blocks. Defaults to 1.
        min_block_length : int or None
            The minimum block length. Defaults to 1.
        block_weights : array-like of shape (X.shape[0],), or Callable, or None
            The weights with which to sample blocks, where block_weights[i] is the weight assigned to the ith block. Defaults to None.
        tapered_weights : array-like of shape (n_blocks,), or Callable, or None
            The tapered weights to assign to each block. Defaults to None.

        BlockLengthSampler Parameters
        -----
        block_length : int or None
        block_length_distribution : str or None

        BlockGenerator Parameters
        -----
        wrap_around_flag : bool, default=False
        overlap_flag : bool, default=False
        overlap_length : int or None
        min_block_length : int or None

        BlockResampler Parameters
        -----
        blocks : list of arrays
        block_weights : array-like of shape (X.shape[0],), or Callable, or None
        tapered_weights : array-like of shape (n_blocks,), or Callable, or None

        Common Parameters
        -----
        X : array-like of shape (n_samples, n_features)
        random_state : int or Generator, default=None
        """

        super().__init__(n_bootstraps=n_bootstraps, random_state=random_state)
        self.block_length_distribution = block_length_distribution
        self.block_length = block_length
        self.wrap_around_flag = wrap_around_flag
        self.overlap_flag = overlap_flag
        self.combine_generation_and_sampling_flag = combine_generation_and_sampling_flag

        self.block_weights = kwargs.get("block_weights", None)
        self.tapered_weights = kwargs.get("tapered_weights", None)
        self.overlap_length = kwargs.get('overlap_length', None)
        self.min_block_length = kwargs.get('min_block_length', None)

        self.blocks = None
        self.block_resampler = None

    @property
    def block_length(self) -> Optional[int]:
        """Getter for block_length."""
        return self._block_length

    @block_length.setter
    def block_length(self, value) -> None:
        """
        Setter for block_length. Performs validation on assignment.
        Parameters
        ----------
        value : int or None
        """
        if value is not None:
            if not isinstance(value, int) or value < 1:
                raise ValueError(
                    "Block length needs to be None or an integer >= 1.")
        self._block_length = value

    @property
    def block_length_distribution(self) -> Optional[str]:
        """Getter for block_length_distribution."""
        return self._block_length_distribution

    @block_length_distribution.setter
    def block_length_distribution(self, value) -> None:
        """
        Setter for block_length_distribution. Performs validation on assignment.
        Parameters
        ----------
        value : str
            The block length distribution function to use.
        """
        if value is not None and not isinstance(value, str):
            raise ValueError("block_length_distribution must be a string.")
        self._block_length_distribution = value

    def _check_input(self, X: np.ndarray) -> None:
        super()._check_input(X)
        if self.block_length is not None and self.block_length > X.shape[0]:
            raise ValueError(
                "block_length cannot be greater than the size of the input array X.")

    def _generate_blocks(self, X: np.ndarray) -> List[np.ndarray]:
        """Generates blocks of indices.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        blocks : list of arrays
            The generated blocks.

        """

        block_length_sampler = BlockLengthSampler(
            avg_block_length=self.block_length if self.block_length is not None else int(
                np.sqrt(X.shape[0])),
            block_length_distribution=self.block_length_distribution,
            rng=self.rng)

        block_generator = BlockGenerator(
            block_length_sampler=block_length_sampler, input_length=X.shape[0], rng=self.rng, wrap_around_flag=self.wrap_around_flag, overlap_length=self.overlap_length, min_block_length=self.min_block_length)

        blocks = block_generator.generate_blocks(
            overlap_flag=self.overlap_flag)

        return blocks


class MovingBlockBootstrap(BlockBootstrap):

    def __init__(self, n_bootstraps: int = 10, block_length: int | None = None, block_length_distribution: str | None = None, wrap_around_flag: bool = False, overlap_flag: bool = True, combine_generation_and_sampling_flag: bool = False, random_state: Optional[Union[int, Generator]] = None, **kwargs) -> None:
        min_block_length = kwargs.pop("min_block_length", 1)
        super().__init__(n_bootstraps, block_length, block_length_distribution,
                         wrap_around_flag, overlap_flag, combine_generation_and_sampling_flag, random_state, min_block_length=min_block_length, **kwargs)

    def _generate_samples_single_bootstrap(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generate a single bootstrap sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray]]
            A tuple containing the indices and data of the generated blocks.
        """

        if self.combine_generation_and_sampling_flag or self.blocks is None:
            blocks = self._generate_blocks(X=X)

            block_resampler = BlockResampler(
                X=X,
                blocks=blocks,
                rng=self.rng,
                block_weights=self.block_weights,
                tapered_weights=self.tapered_weights
            )
        else:
            blocks = self.blocks
            block_resampler = self.block_resampler

        block_indices, block_data = block_resampler.resample_block_indices_and_data()

        if not self.combine_generation_and_sampling_flag:
            self.blocks = blocks
            self.block_resampler = block_resampler

        return block_indices, block_data


'''
class StationaryBootstrap(BlockBootstrap):
    def _generate_samples_single_bootstrap(self, X: np.ndarray, block_length: Optional[int], **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:

        if self.blocks is None:
            self.blocks = self._generate_blocks(block_length=block_length, block_length_distribution="geometric",
                                                X=X, rng=self.rng, wrap_around_flag=False, overlap_flag=True, **kwargs)
            self.block_resampler = BlockResampler(X=X,
                                                  blocks=self.blocks, rng=self.rng,
                                                  block_weights=kwargs.get(
                                                      "block_weights", None),
                                                  tapered_weights=kwargs.get("tapered_weights", None))

        block_indices, block_data = self.block_resampler.resample_block_indices_and_data()
        return block_indices, block_data


class CircularBootstrap(BlockBootstrap):
    def _generate_samples_single_bootstrap(self, X: np.ndarray, block_length: Optional[int], **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:

        if self.blocks is None:
            self.blocks = self._generate_blocks(block_length=block_length, block_length_distribution=None,
                                                X=X, rng=self.rng, wrap_around_flag=True, overlap_flag=True, **kwargs)
            self.block_resampler = BlockResampler(X=X,
                                                  blocks=self.blocks, rng=self.rng,
                                                  block_weights=kwargs.get(
                                                      "block_weights", None),
                                                  tapered_weights=kwargs.get("tapered_weights", None))

        block_indices, block_data = self.block_resampler.resample_block_indices_and_data()
        return block_indices, block_data


class NonOverlappingBootstrap(BlockBootstrap):
    def _generate_samples_single_bootstrap(self, X: np.ndarray, block_length: Optional[int], **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:

        if self.blocks is None:
            self.blocks = self._generate_blocks(block_length=block_length, block_length_distribution=None,
                                                X=X, rng=self.rng, wrap_around_flag=False, overlap_flag=False, **kwargs)
            self.block_resampler = BlockResampler(X=X,
                                                  blocks=self.blocks, rng=self.rng,
                                                  block_weights=kwargs.get(
                                                      "block_weights", None),
                                                  tapered_weights=kwargs.get("tapered_weights", None))

        block_indices, block_data = self.block_resampler.resample_block_indices_and_data()
        return block_indices, block_data


base_block_bootstraps = [NonOverlappingBootstrap, MovingBlockBootstrap,
                         StationaryBootstrap, CircularBootstrap]


class BaseBlockBootstrap(BlockBootstrap):
    def __init__(self, bootstrap_type: Optional[Type[BlockBootstrap]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (bootstrap_type is not None) and (bootstrap_type not in base_block_bootstraps):
            raise ValueError(
                "bootstrap_type should be one of {}".format(base_block_bootstraps))
        self.bootstrap_type = bootstrap_type

    def _generate_samples_single_bootstrap(self, X: np.ndarray, random_seed: Optional[int], block_length: Optional[int], **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if random_seed is None:
            random_seed = self.random_seed

        n = X.shape[0]
        block_length = self.block_length if self.block_length is not None else int(
            np.sqrt(n))

        if self.bootstrap_type is None:
            blocks = self._generate_blocks(
                block_length=block_length, X=X, random_seed=random_seed, **kwargs)
            block_indices, block_data = resample_block_indices_and_data(
                X=X, block_length=block_length, blocks=blocks, random_seed=random_seed, **kwargs)
        else:
            block_indices, block_data = self.bootstrap_type._generate_samples_single_bootstrap(
                X=X, block_length=block_length, random_seed=random_seed, **kwargs)

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


# TODO: higher-order Markov models or conditional variants of HMMs (e.g., Input-Output HMMs or Factorial HMMs).
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


class BaseResidualBootstrap(BaseTimeSeriesBootstrap):
    def __init__(self, model_type: Literal["ar", "arima", "sarima", "var"], order: Optional[Union[int, List[int], Tuple[int, int, int], Tuple[int, int, int, int]]], *args, **kwargs):
        """
        order is a tuple of (p, o, q) for ARIMA and (p, d, q, s) for SARIMAX. It is either a single int or a list of non-consecutive ints for AR, and an int for VAR and ARCH. If None, the best order is chosen via TSFitBestLag. Do note that TSFitBestLag only chooses the best lag, not the best order, so for the tuple values, it only chooses the best p, not the best (p, o, q) or (p, d, q, s). The rest of the values are set to 0.
        """
        super().__init__(*args, **kwargs)
        model_type = model_type.lower()
        if model_type == "arch":
            raise ValueError(
                "Do not use ARCH models to fit the data; they are meant for fitting to residuals.")
        self.model_type = model_type
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
        resampled_indices = generate_random_indices(
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
            resampled_indices = generate_random_indices(
                resids.shape[0], random_seed)
            resampled_resids = resids[resampled_indices]
            # Add the bootstrapped residuals to the fitted values
            bootstrap_samples_iter = X_fitted + resampled_resids
            bootstrap_samples.append(bootstrap_samples_iter)

        return block_indices, bootstrap_samples


class BaseDistributionBootstrap(BaseResidualBootstrap):
    """
    Implementation of the Distribution Bootstrap (DB) method for time series data.

    Parameters
    ----------
    distribution: str, default='normal'
        The distribution
    """

    def __init__(self, distribution: str = 'normal', **kwargs):
        super().__init__(**kwargs)
        # Check if the distribution exists in scipy.stats
        distribution = distribution.lower()
        if not hasattr(distributions, self.distribution):
            raise ValueError(
                f"Invalid distribution: {self.distribution}; must be a valid distribution in scipy.stats.")
        self.distribution = distribution


class WholeDistributionBootstrap(BaseDistributionBootstrap):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fit_model = None
        self.resids = None
        self.X_fitted = None
        self.resids_dist = None
        self.resids_dist_params = ()

    def _fit_distribution(self):
        # Check if the distribution exists in scipy.stats
        dist = getattr(distributions, self.distribution)
        # Fit the distribution to the residuals
        params = dist.fit(self.resids)
        self.resids_dist = dist
        self.resids_dist_params = params

    def _generate_samples_single_bootstrap(self, X: np.ndarray, random_seed: Optional[int], **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if random_seed is None:
            random_seed = self.random_seed

        if self.resids is None or self.X_fitted is None or self.resids_dist is None or self.resids_dist_params == ():
            fit_obj = TSFitBestLag(model_type=self.model_type, order=self.order,
                                   save_models=kwargs.get('save_models', False))
            self.fit_model = fit_obj.fit(X=X, exog=kwargs.get('exog', None))
            self.X_fitted = fit_obj.get_fitted_X()
            self.resids = fit_obj.get_residuals()

            # Fit the specified distribution to the residuals
            self._fit_distribution()

        # Generate new residuals from the fitted distribution
        bootstrap_residuals = self.resids_dist.rvs(
            size=X.shape[0], *self.resids_dist_params)

        # Add new residuals to the fitted values to create the bootstrap time series
        bootstrap_samples = self.X_fitted + bootstrap_residuals
        return [np.arange(0, X.shape[0])], [bootstrap_samples]


class BlockDistributionBootstrap(BaseDistributionBootstrap, BaseBlockBootstrap):
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

            # Fit the specified distribution to the residuals
            resids_dist = getattr(distributions, self.distribution)
            # Fit the distribution to the residuals
            resids_dist_params = resids_dist.fit(resids)
            # Generate new residuals from the fitted distribution
            bootstrap_residuals = resids_dist.rvs(
                size=X.shape[0], *resids_dist_params)
            # Add new residuals to the fitted values to create the bootstrap time series
            bootstrap_samples_iter = X_fitted + bootstrap_residuals
            bootstrap_samples.append(bootstrap_samples_iter)

        return block_indices, bootstrap_samples


VALID_MODELS = [AutoReg, ARIMA, SARIMAX, VAR, arch_model]


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
        bootstrap_indices = generate_random_indices(
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
            bootstrap_indices = generate_random_indices(
                data_diff.shape[0], random_seed)
            data_diff_bootstrapped = data_diff[bootstrap_indices]
            bootstrap_samples_iter = self.fracdiff_transformer.inverse_transform(
                data_diff_bootstrapped)
            bootstrap_samples.append(bootstrap_samples_iter)
        return block_indices, bootstrap_samples

'''
