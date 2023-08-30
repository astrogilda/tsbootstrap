from __future__ import annotations

import inspect
from abc import ABCMeta, abstractmethod
from functools import partial
from numbers import Integral
from typing import Any, Callable, Iterator, Optional, Union

import numpy as np
from scipy.signal.windows import tukey
from scipy.stats import (
    beta,
    expon,
    gamma,
    geom,
    lognorm,
    norm,
    pareto,
    poisson,
    rv_continuous,
    uniform,
    weibull_min,
)
from sklearn.decomposition import PCA

from ts_bs import (
    BlockGenerator,
    BlockLengthSampler,
    BlockResampler,
    MarkovSampler,
    TimeSeriesSimulator,
    TSFitBestLag,
)
from ts_bs.bootstrap_configs import *
from ts_bs.utils.odds_and_ends import (
    generate_random_indices,
    time_series_split,
)
from ts_bs.utils.types import (
    FittedModelTypes,
    ModelTypes,
    ModelTypesWithoutArch,
    OrderTypes,
    OrderTypesWithoutNone,
    RngTypes,
)
from ts_bs.utils.validate import (
    validate_integers,
    validate_literal_type,
    validate_order,
    validate_rng,
)

# TODO: add a check if generated block is only one unit long
# TODO: ensure docstrings align with functionality
# TODO: test -- check len(returned_indices) == X.shape[0]
# TODO: ensure x is 2d only for var, otherwise 1d or 2d with 1 feature
# TODO: block_weights=p with block_length=1 should be equivalent to the iid bootstrap
# TODO: add test to fit_ar to ensure input lags, if list, are unique


class BaseTimeSeriesBootstrap(metaclass=ABCMeta):
    """
    Base class for time series bootstrapping.

    Raises
    ------
    ValueError
        If n_bootstraps is not greater than 0.
    """

    def __init__(self, config: BaseTimeSeriesBootstrapConfig) -> None:
        self.config = config

    def bootstrap(
        self,
        X: np.ndarray,
        return_indices: bool = False,
        exog: np.ndarray | None = None,
    ) -> Iterator[np.ndarray] | Iterator[tuple[list[np.ndarray], np.ndarray]]:
        """Generate indices to split data into training and test set."""
        X = np.asarray(X)
        if len(X.shape) < 2:
            X = np.expand_dims(X, 1)

        self._check_input(X)

        X_train, X_test = time_series_split(X, test_ratio=0.2)

        if exog is not None:
            self._check_input(exog)
            exog_train, _ = time_series_split(exog, test_ratio=0.2)
        else:
            exog_train = None
            # exog_test = None

        samples_iter = self._generate_samples(
            X=X_train, return_indices=return_indices, exog=exog_train
        )

        yield from samples_iter

    def _generate_samples(
        self,
        X: np.ndarray,
        return_indices: bool = False,
        exog: np.ndarray | None = None,
    ) -> Iterator[np.ndarray]:
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
        for _ in range(self.config.n_bootstraps):
            indices, data = self._generate_samples_single_bootstrap(
                X=X, exog=exog
            )
            data = np.concatenate(data, axis=0)
            if return_indices:
                yield indices, data  # type: ignore
            else:
                yield data

    @abstractmethod
    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, exog: np.ndarray | None = None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Generates list of bootstrapped indices and samples for a single bootstrap iteration.

        Should be implemented in derived classes.
        """

    def _check_input(self, X):
        """Checks if the input is valid."""
        if np.any(np.diff([len(x) for x in X]) != 0):
            raise ValueError("All time series must be of the same length.")

    def get_n_bootstraps(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> Integral:
        """Returns the number of bootstrapping iterations."""
        return self.config.n_bootstraps  # type: ignore


class BlockBootstrap(BaseTimeSeriesBootstrap):
    """
    Block Bootstrap base class for time series data.

    Parameters
    ----------
    block_length : Integral, default=None
        The length of the blocks to sample. If None, the block length is automatically set to the square root of the number of observations.

    Raises
    ------
    ValueError
        If block_length is not greater than 0.
    """

    def __init__(
        self,
        config: BlockBootstrapConfig,
    ) -> None:
        """
        Block Bootstrap class for time series data.
        """
        super().__init__(config=config)
        self.config = config

        self.blocks = None
        self.block_resampler = None

    def _check_input(self, X: np.ndarray) -> None:
        super()._check_input(X)
        if self.config.block_length is not None and self.config.block_length > X.shape[0]:  # type: ignore
            raise ValueError(
                "block_length cannot be greater than the size of the input array X."
            )

    def _generate_blocks(self, X: np.ndarray) -> list[np.ndarray]:
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
            avg_block_length=self.config.block_length
            if self.config.block_length is not None
            else int(np.sqrt(X.shape[0])),  # type: ignore
            block_length_distribution=self.config.block_length_distribution,
            rng=self.config.rng,
        )

        block_generator = BlockGenerator(
            block_length_sampler=block_length_sampler,
            input_length=X.shape[0],  # type: ignore
            rng=self.config.rng,
            wrap_around_flag=self.config.wrap_around_flag,
            overlap_length=self.config.overlap_length,
            min_block_length=self.config.min_block_length,
        )

        blocks = block_generator.generate_blocks(
            overlap_flag=self.config.overlap_flag
        )

        return blocks

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, exog: np.ndarray | None = None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
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
        if (
            self.config.combine_generation_and_sampling_flag
            or self.blocks is None
        ):
            blocks = self._generate_blocks(X=X)

            block_resampler = BlockResampler(
                X=X,
                blocks=blocks,
                rng=self.config.rng,
                block_weights=self.config.block_weights,
                tapered_weights=self.config.tapered_weights,
            )
        else:
            blocks = self.blocks
            block_resampler = self.block_resampler

        (
            block_indices,
            block_data,
        ) = block_resampler.resample_block_indices_and_data()  # type: ignore

        if not self.config.combine_generation_and_sampling_flag:
            self.blocks = blocks
            self.block_resampler = block_resampler

        return block_indices, block_data


class MovingBlockBootstrap(BlockBootstrap):
    r"""
    Moving Block Bootstrap class for time series data.

    This class functions similarly to the base `BlockBootstrap` class, with
    the following modifications to the default behavior:
    * `overlap_flag` is always set to True, meaning that blocks can overlap.
    * `wrap_around_flag` is always set to False, meaning that the data will not
    wrap around when generating blocks.
    * `block_length_distribution` is always None, meaning that the block length
    distribution is not utilized.
    * `combine_generation_and_sampling_flag` is always False, meaning that the block
    generation and resampling are performed separately.

    Raises
    ------
    ValueError
        If block_length is not greater than 0.

    Notes
    -----
    The Moving Block Bootstrap is defined as:

    .. math::
        \\hat{X}_t = \\frac{1}{L}\\sum_{i=1}^L X_{t + \\lfloor U_i \\rfloor}

    where :math:`L` is the block length, :math:`U_i` is a uniform random variable on :math:`[0, 1]`, and :math:`\\lfloor \\cdot \\rfloor` is the floor function.

    References
    ----------
    .. [^1^] https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Moving_block_bootstrap
    """

    def __init__(self, config: MovingBlockBootstrapConfig, **kwargs) -> None:
        super().__init__(config=config, **kwargs)


class StationaryBlockBootstrap(BlockBootstrap):
    r"""
    Stationary Block Bootstrap class for time series data.

    This class functions similarly to the base `BlockBootstrap` class, with
    the following modifications to the default behavior:
    * `overlap_flag` is always set to True, meaning that blocks can overlap.
    * `wrap_around_flag` is always set to False, meaning that the data will not
    wrap around when generating blocks.
    * `block_length_distribution` is always "geometric", meaning that the block
    length distribution is geometrically distributed.
    * `combine_generation_and_sampling_flag` is always False, meaning that the block
    generation and resampling are performed separately.

    Raises
    ------
    ValueError
        If block_length is not greater than 0.

    Notes
    -----
    The Stationary Block Bootstrap is defined as:

    .. math::
        \\hat{X}_t = \\frac{1}{L}\\sum_{i=1}^L X_{t + \\lfloor U_i \\rfloor}

    where :math:`L` is the block length, :math:`U_i` is a uniform random variable on :math:`[0, 1]`, and :math:`\\lfloor \\cdot \\rfloor` is the floor function.

    References
    ----------
    .. [^1^] https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Moving_block_bootstrap
    """

    def __init__(
        self, config: StationaryBlockBootstrapConfig, **kwargs
    ) -> None:
        super().__init__(config=config, **kwargs)


class CircularBlockBootstrap(BlockBootstrap):
    r"""
    Circular Block Bootstrap class for time series data.

    This class functions similarly to the base `BlockBootstrap` class, with
    the following modifications to the default behavior:
    * `overlap_flag` is always set to True, meaning that blocks can overlap.
    * `wrap_around_flag` is always set to True, meaning that the data will wrap
    around when generating blocks.
    * `block_length_distribution` is always None, meaning that the block length
    distribution is not utilized.
    * `combine_generation_and_sampling_flag` is always False, meaning that the block
    generation and resampling are performed separately.

    Parameters
    ----------
    block_length : Integral, default=None
        The length of the blocks to sample. If None, the block length is automatically set to the square root of the number of observations.

    Raises
    ------
    ValueError
        If block_length is not greater than 0.

    Notes
    -----
    The Circular Block Bootstrap is defined as:

    .. math::
        \\hat{X}_t = \\frac{1}{L}\\sum_{i=1}^L X_{t + \\lfloor U_i \\rfloor}

    where :math:`L` is the block length, :math:`U_i` is a uniform random variable on :math:`[0, 1]`, and :math:`\\lfloor \\cdot \\rfloor` is the floor function.

    References
    ----------
    .. [^1^] https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Moving_block_bootstrap
    """

    def __init__(self, config: CircularBlockBootstrapConfig, **kwargs) -> None:
        super().__init__(config=config, **kwargs)


class NonOverlappingBlockBootstrap(BlockBootstrap):
    r"""
    Non-Overlapping Block Bootstrap class for time series data.

    This class functions similarly to the base `BlockBootstrap` class, with
    the following modifications to the default behavior:
    * `overlap_flag` is always set to False, meaning that blocks cannot overlap.
    * `wrap_around_flag` is always set to False, meaning that the data will not
    wrap around when generating blocks.
    * `block_length_distribution` is always None, meaning that the block length
    distribution is not utilized.
    * `combine_generation_and_sampling_flag` is always False, meaning that the block
    generation and resampling are performed separately.

    Parameters
    ----------
    block_length : Integral, default=None
        The length of the blocks to sample. If None, the block length is automatically set to the square root of the number of observations.

    Raises
    ------
    ValueError
        If block_length is not greater than 0.

    Notes
    -----
    The Non-Overlapping Block Bootstrap is defined as:

    .. math::
        \\hat{X}_t = \\frac{1}{L}\\sum_{i=1}^L X_{t + i}

    where :math:`L` is the block length.

    References
    ----------
    .. [^1^] https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Moving_block_bootstrap
    """

    def __init__(
        self, config: NonOverlappingBlockBootstrapConfig, **kwargs
    ) -> None:
        super().__init__(config=config, **kwargs)


class BaseBlockBootstrap(BlockBootstrap):
    """
    Base class for block bootstrapping.
    """

    def __init__(self, config: BaseBlockBootstrapConfig, **kwargs):
        super().__init__(config=config, **kwargs)
        self.bootstrap_instance: BlockBootstrap | None = None

        if config.bootstrap_type:
            self.bootstrap_instance = config.bootstrap_type_dict[
                config.bootstrap_type
            ](config=config)

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, exog: np.ndarray | None = None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Generate a single bootstrap sample using either the base BlockBootstrap method or the specified bootstrap_type.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray]]
            A tuple containing the indices and data of the generated blocks.
        """
        if self.bootstrap_instance is None:
            # Generate samples using the base BlockBootstrap method
            (
                block_indices,
                block_data,
            ) = super()._generate_samples_single_bootstrap(X=X, exog=exog)
        else:
            # Generate samples using the specified bootstrap_type
            if hasattr(
                self.bootstrap_instance, "_generate_samples_single_bootstrap"
            ):
                (
                    block_indices,
                    block_data,
                ) = self.bootstrap_instance._generate_samples_single_bootstrap(
                    X=X, exog=exog
                )
            else:
                raise NotImplementedError(
                    f"The bootstrap class '{type(self.bootstrap_instance).__name__}' does not implement '_generate_samples_single_bootstrap' method."
                )

        return block_indices, block_data


# Be cautious when using the default windowing functions from numpy, as they drop to 0 at the edges.This could be particularly problematic for smaller block_lengths. In the current implementation, we have clipped the min to 0.1, in block_resampler.py.


class BartlettsBootstrap(BaseBlockBootstrap):
    """Bartlett's Bootstrap class for time series data.

    This class is a specialized bootstrapping class that uses
    Bartlett's window for tapered weights.
    """

    def __init__(self, config: BartlettsBootstrapConfig):
        """Initialize BartlettsBootstrap.

        Parameters
        ----------
        config : BartlettsBootstrapConfig
            The configuration object.
        """
        if config is None:
            config = BartlettsBootstrapConfig()
        super().__init__(config=config)


class HammingBootstrap(BaseBlockBootstrap):
    r"""
    Hamming Bootstrap class for time series data.

    This class is a specialized bootstrapping class that uses
    Hamming window for tapered weights.

    Notes
    -----
    The Hamming window is defined as:

    .. math::
        w(n) = 0.54 - 0.46 \\cos\\left(\\frac{2\\pi n}{N - 1}\\right)

    where :math:`N` is the block length.

    References
    ----------
    .. [^1^] https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Moving_block_bootstrap
    """

    def __init__(self, config: HammingBootstrapConfig):
        """Initialize HammingBootstrap.

        Parameters
        ----------
        config : HammingBootstrapConfig
            The configuration object.
        """
        if config is None:
            config = HammingBootstrapConfig()
        super().__init__(config=config)


class HanningBootstrap(BaseBlockBootstrap):
    r"""
    Hanning Bootstrap class for time series data.

    This class is a specialized bootstrapping class that uses
    Hanning window for tapered weights.

    Notes
    -----
    The Hanning window is defined as:

    .. math::
        w(n) = 0.5 - 0.5 \\cos\\left(\\frac{2\\pi n}{N - 1}\\right)

    where :math:`N` is the block length.

    References
    ----------
    .. [^1^] https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Moving_block_bootstrap
    """

    def __init__(self, config: HanningBootstrapConfig):
        """Initialize HanningBootstrap.

        Parameters
        ----------
        config : HanningBootstrapConfig
            The configuration object.
        """
        if config is None:
            config = HanningBootstrapConfig()
        super().__init__(config=config)


class BlackmanBootstrap(BaseBlockBootstrap):
    r"""
    Blackman Bootstrap class for time series data.

    This class is a specialized bootstrapping class that uses
    Blackman window for tapered weights.

    Notes
    -----
    The Blackman window is defined as:

    .. math::
        w(n) = 0.42 - 0.5 \\cos\\left(\\frac{2\\pi n}{N - 1}\\right) + 0.08 \\cos\\left(\\frac{4\\pi n}{N - 1}\\right)

    where :math:`N` is the block length.

    References
    ----------
    .. [^1^] https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Moving_block_bootstrap
    """

    def __init__(self, config: BlackmanBootstrapConfig):
        """Initialize BlackmanBootstrap.

        Parameters
        ----------
        config : BlackmanBootstrapConfig
            The configuration object.
        """
        if config is None:
            config = BlackmanBootstrapConfig()
        super().__init__(config=config)


class TukeyBootstrap(BaseBlockBootstrap):
    r"""
    Tukey Bootstrap class for time series data.

    This class is a specialized bootstrapping class that uses
    Tukey window for tapered weights.

    Notes
    -----
    The Tukey window is defined as:

    .. math::
        w(n) = \\begin{cases}
            0.5\\left[1 + \\cos\\left(\\frac{2\\pi n}{\\alpha(N - 1)}\\right)\\right], & \\text{if } n < \\frac{\\alpha(N - 1)}{2}\\\\
            1, & \\text{if } \\frac{\\alpha(N - 1)}{2} \\leq n \\leq (N - 1)\\left(1 - \\frac{\\alpha}{2}\\right)\\\\
            0.5\\left[1 + \\cos\\left(\\frac{2\\pi n}{\\alpha(N - 1)}\\right)\\right], & \\text{if } n > (N - 1)\\left(1 - \\frac{\\alpha}{2}\\right)
        \\end{cases}

    where :math:`N` is the block length and :math:`\\alpha` is the parameter
    controlling the shape of the window.
    """

    def __init__(self, config: TukeyBootstrapConfig):
        """Initialize TukeyBootstrap.

        Parameters
        ----------
        config : TukeyBootstrapConfig
            The configuration object.
        """
        if config is None:
            config = TukeyBootstrapConfig()
        super().__init__(config=config)


class BaseResidualBootstrap(BaseTimeSeriesBootstrap):
    """
    Base class for residual bootstrap.

    Attributes
    ----------
    fit_model : TSFitBestLag
        The fitted model.
    resids : np.ndarray
        The residuals of the fitted model.
    X_fitted : np.ndarray
        The fitted values of the fitted model.
    coefs : np.ndarray
        The coefficients of the fitted model.

    Methods
    ------- 
    __init__ : Initialize self.
    _fit_model : Fits the model to the data and stores the residuals.
    """

    def __init__(
        self,
        config: BaseResidualBootstrapConfig,
    ):
        """
        Initialize self.

        Parameters
        ----------
        config : BaseResidualBootstrapConfig
            The configuration object.
        """
        super().__init__(config=config)
        self.config = config

        self.fit_model = None
        self.resids = None
        self.X_fitted = None
        self.coefs = None

    def _fit_model(
        self, X: np.ndarray, exog: np.ndarray | None = None
    ) -> None:
        """Fits the model to the data and stores the residuals."""
        if (
            self.resids is None
            or self.X_fitted is None
            or self.fit_model is None
            or self.coefs is None
        ):
            fit_obj = TSFitBestLag(
                model_type=self.config.model_type,
                order=self.config.order,
                save_models=self.config.save_models,
                **self.config.model_params,
            )
            self.fit_model = fit_obj.fit(X=X, exog=exog).model
            self.X_fitted = fit_obj.get_fitted_X()
            self.resids = fit_obj.get_residuals()
            self.order = fit_obj.get_order()
            self.coefs = fit_obj.get_coefs()


# Fit, then resample residuals.
class WholeResidualBootstrap(BaseResidualBootstrap):
    """
    Whole Residual Bootstrap class for time series data.

    This class applies residual bootstrapping to the entire time series,
    without any block structure. This is the most basic form of residual
    bootstrapping. The residuals are resampled with replacement and added to
    the fitted values to generate new samples.

    Methods
    -------
    __init__ : Initialize self.
    _generate_samples_single_bootstrap : Generate a single bootstrap sample.
    """

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, exog: np.ndarray | None =
         None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        self._fit_model(X=X, exog=exog)

        # Resample residuals
        resampled_indices = generate_random_indices(
            self.resids.shape[0], self.config.rng
        )
        resampled_residuals = self.resids[resampled_indices]
        # Add the bootstrapped residuals to the fitted values
        bootstrap_samples = self.X_fitted + resampled_residuals
        return [resampled_indices], [bootstrap_samples]



class BlockResidualBootstrap(BaseResidualBootstrap, BaseBlockBootstrap):
    """
    Block Residual Bootstrap class for time series data.

    This class applies residual bootstrapping to blocks of the time series.
    The residuals are bootstrapped using the specified block structure and
    added to the fitted values to generate new samples.

    Methods
    -------
    __init__ : Initialize self.
    _generate_samples_single_bootstrap : Generate a single bootstrap sample.
    """

    def __init__(
        self,
        residual_config: BaseResidualBootstrapConfig, block_config: BaseBlockBootstrapConfig,
    ) -> None:
        """
        Initialize self.

        Parameters
        ----------
        residual_config : BaseResidualBootstrapConfig
            The configuration object for the residual bootstrap.
        block_config : BaseBlockBootstrapConfig
            The configuration object for the block bootstrap.
        """
        BaseResidualBootstrap.__init__(self, config=residual_config)
        BaseBlockBootstrap.__init__(self, config=block_config)

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, exog: Optional[np.ndarray] = None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        # Fit the model and store residuals, fitted values, etc.
        super()._fit_model(X=X, exog=exog)

        # Generate blocks of residuals
        block_indices, block_data = BaseBlockBootstrap._generate_samples_single_bootstrap(
            self, X=self.resids
        )

        # Add the bootstrapped residuals to the fitted values
        bootstrap_samples = self.X_fitted + np.concatenate(block_data, axis=0)

        return block_indices, [bootstrap_samples]


class BaseMarkovBootstrap(BaseResidualBootstrap):
    """
    Base class for Markov bootstrap.

    Attributes
    ----------
    hmm_object : MarkovSampler or None
        The MarkovSampler object used for sampling.

    Methods
    -------
    __init__ : Initialize the Markov bootstrap.

    Notes
    -----
    Fitting Markov models is expensive, hence we do not allow re-fititng. We instead fit once to the residuals and generate new samples by changing the random_seed.
    """

    def __init__(
        self,
        config: BaseMarkovBootstrapConfig,
    ):
        """
        Initialize self.

        Parameters
        ----------
        config : BaseMarkovBootstrapConfig
            The configuration object.
        """
        super().__init__(config=config)
        self.config = config

        self.hmm_object = None



# Fit HMM to residuals, then sample from the HMM with different random seeds.
class WholeMarkovBootstrap(BaseMarkovBootstrap):
    """
    Whole Markov Bootstrap class for time series data.

    This class applies Markov bootstrapping to the entire time series,
    without any block structure. This is the most basic form of Markov
    bootstrapping. The residuals are fit to a Markov model, and then
    resampled using the Markov model. The resampled residuals are added to
    the fitted values to generate new samples.

    Methods
    -------
    __init__ : Initialize self.
    _generate_samples_single_bootstrap : Generate a single bootstrap sample.
    """

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, exog: np.ndarray | None = None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        self._fit_model(X=X, exog=exog)

        random_seed = self.config.rng.integers(0, 1000)
        if self.hmm_object is None:
            markov_sampler = MarkovSampler(
                apply_pca_flag=self.config.apply_pca_flag,
                pca=self.config.pca,
                n_iter_hmm=self.config.n_iter_hmm,
                n_fits_hmm=self.config.n_fits_hmm,
                method=self.config.method,
                blocks_as_hidden_states_flag=self.config.blocks_as_hidden_states_flag,
                random_seed=random_seed,
            )

            markov_sampler.fit(blocks=self.resids, n_states=self.config.n_states)
            self.hmm_object = markov_sampler

        bootstrapped_resids = self.hmm_object.sample(
            random_seed=random_seed + self.config.rng.integers(0, 1000)
        )[0]
        bootstrap_samples = self.X_fitted + bootstrapped_resids

        return [np.arange(X.shape[0])], [bootstrap_samples]


# Fit HMM to residuals, then resample using blocks once, then sample from the HMM with different random seeds.
class BlockMarkovBootstrap(BaseMarkovBootstrap, BaseBlockBootstrap):
    def __init__(
        self,
        *args_base_residual,
        method: str = "mean",
        apply_pca_flag: bool = False,
        pca: PCA | None = None,
        n_iter_hmm: Integral = 100,
        n_fits_hmm: Integral = 10,
        blocks_as_hidden_states_flag: bool = False,
        n_states: Integral = 5,
        bootstrap_type: str | None = None,
        kwargs_base_block: dict[str, Any] | None = None,
        **kwargs_base_residual,
    ) -> None:
        kwargs_base_block = (
            {} if kwargs_base_block is None else kwargs_base_block
        )
        super().__init__(
            *args_base_residual,
            method=method,
            apply_pca_flag=apply_pca_flag,
            pca=pca,
            n_iter_hmm=n_iter_hmm,
            n_fits_hmm=n_fits_hmm,
            blocks_as_hidden_states_flag=blocks_as_hidden_states_flag,
            n_states=n_states,
            **kwargs_base_residual,
        )
        BaseBlockBootstrap.__init__(
            self, bootstrap_type=bootstrap_type, **kwargs_base_block
        )

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, exog: np.ndarray | None = None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        super()._fit_model(X=X, exog=exog)
        (
            block_indices,
            block_data,
        ) = BaseBlockBootstrap._generate_samples_single_bootstrap(
            self, X=self.resids
        )

        random_seed = self.rng.integers(0, 1000)
        if self.hmm_object is None:
            markov_sampler = MarkovSampler(
                apply_pca_flag=self.apply_pca_flag,
                pca=self.pca,
                n_iter_hmm=self.n_iter_hmm,
                n_fits_hmm=self.n_fits_hmm,
                method=self.method,
                blocks_as_hidden_states_flag=self.blocks_as_hidden_states_flag,
                random_seed=random_seed,
            )

            markov_sampler.fit(blocks=block_data, n_states=self.n_states)
            self.hmm_object = markov_sampler

        # Add the bootstrapped residuals to the fitted values
        bootstrapped_resids = self.hmm_object.sample(
            random_seed=random_seed + self.rng.integers(0, 1000)
        )[0]
        bootstrap_samples = self.X_fitted + bootstrapped_resids

        return block_indices, [bootstrap_samples]


# Preserve the statistic/bias in the original data.
class BaseBiasCorrectedBootstrap(BaseTimeSeriesBootstrap):
    """Bootstrap class that generates bootstrapped samples preserving a specific statistic.

    This class generates bootstrapped time series data, preserving a given statistic (such as mean, median, etc.)
    The statistic is calculated from the original data and then used as a parameter for generating the bootstrapped samples.
    The statistic is calculated from the original data and then used as a parameter for generating the bootstrapped samples.
    For example, if the statistic is np.mean, then the mean of the original data is calculated and then used as a parameter for generating the bootstrapped samples.

    Attributes
    ----------
    statistic_X : np.ndarray, default=None
        The statistic calculated from the original data. This is used as a parameter for generating the bootstrapped samples.

    Methods
    -------
    __init__ : Initialize the BaseBiasCorrectedBootstrap class.
    _calculate_statistic(X: np.ndarray) -> np.ndarray : Calculate the statistic from the input data.
    """

    def __init__(
        self,
        n_bootstraps: Integral = 10,
        statistic: Callable = np.mean,
        statistic_axis: Integral = 0,
        statistic_keepdims: bool = True,
        rng: np.random.Generator | Integral | None = None,
    ) -> None:
        """
        Initialize the BaseBiasCorrectedBootstrap class.

        Parameters
        ----------
        n_bootstraps : Integral, default=10
            The number of bootstrapped samples to generate.
        statistic : Callable, default np.mean
            A callable function to compute the statistic that should be preserved. This function should take a numpy array as input,
            and return a scalar or a numpy array as output. For example, you can pass np.mean, np.median, or any other custom function.
        statistic_axis : Integral, default=0
            The axis along which the statistic should be computed.
        statistic_keepdims : bool, default=True
            Whether to keep the dimensions of the statistic or not.
        rng : Integral or np.random.Generator, default=None
            A random number generator or an integer seed. If None, the default RNG from numpy will be used.
        """
        super().__init__(n_bootstraps=n_bootstraps, rng=rng)
        self.statistic = statistic
        self.statistic_axis = statistic_axis
        self.statistic_keepdims = statistic_keepdims

        self.statistic_X = None

    @property
    def statistic(self) -> Callable:
        """Get the current statistic function."""
        return self._statistic

    @statistic.setter
    def statistic(self, value: Callable) -> None:
        """Set a new statistic function.

        The new function must be callable. If it is not, a TypeError will be raised.
        """
        if not callable(value):
            raise TypeError("statistic must be a callable.")
        self._statistic = value

    def _calculate_statistic(self, X: np.ndarray) -> np.ndarray:
        params = inspect.signature(self.statistic).parameters
        kwargs_stat = {
            "axis": self.statistic_axis,
            "keepdims": self.statistic_keepdims,
        }
        kwargs_stat = {k: v for k, v in kwargs_stat.items() if k in params}
        statistic_X = self.statistic(X, **kwargs_stat)
        return statistic_X


class WholeBiasCorrectedBootstrap(BaseBiasCorrectedBootstrap):
    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, exog: np.ndarray | None = None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if self.statistic_X is None:
            self.statistic_X = self._calculate_statistic(X=X)

        # Resample residuals
        resampled_indices = generate_random_indices(X.shape[0], self.rng)
        bootstrapped_sample = X[resampled_indices]
        # Calculate the bootstrapped statistic
        statistic_bootstrapped = self._calculate_statistic(bootstrapped_sample)
        # Calculate the bias
        bias = self.statistic_X - statistic_bootstrapped
        # Add the bias to the bootstrapped sample
        bootstrap_sample_bias_corrected = bootstrapped_sample + bias.reshape(
            bootstrapped_sample.shape
        )
        return [resampled_indices], [bootstrap_sample_bias_corrected]


class BlockBiasCorrectedBootstrap(
    BaseBiasCorrectedBootstrap, BaseBlockBootstrap
):
    def __init__(
        self,
        n_bootstraps: Integral = 10,
        statistic: Callable = np.mean,
        statistic_axis: Integral = 0,
        statistic_keepdims: bool = True,
        rng: np.random.Generator | Integral | None = None,
        bootstrap_type: str | None = None,
        kwargs_base_block: dict[str, Any] | None = None,
    ) -> None:
        kwargs_base_block = (
            {} if kwargs_base_block is None else kwargs_base_block
        )
        super().__init__(
            n_bootstraps=n_bootstraps,
            statistic=statistic,
            statistic_axis=statistic_axis,
            statistic_keepdims=statistic_keepdims,
            rng=rng,
        )
        BaseBlockBootstrap.__init__(
            self, bootstrap_type=bootstrap_type, **kwargs_base_block
        )

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, exog: np.ndarray | None = None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if self.statistic_X is None:
            self.statistic_X = super()._calculate_statistic(X=X)
        (
            block_indices,
            block_data,
        ) = BaseBlockBootstrap._generate_samples_single_bootstrap(self, X=X)

        block_data_concat = np.concatenate(block_data, axis=0)
        # Calculate the bootstrapped statistic
        statistic_bootstrapped = self._calculate_statistic(block_data_concat)
        # Calculate the bias
        bias = self.statistic_X - statistic_bootstrapped
        # Add the bias to the bootstrapped sample
        bootstrap_samples = block_data_concat + bias
        return block_indices, [bootstrap_samples]


# We can only fit uni-variate distributions, so X must be a 1D array, and `model_type` in BaseResidualBootstrap must not be "var".
class BaseDistributionBootstrap(BaseResidualBootstrap):
    """
    Implementation of the Distribution Bootstrap (DB) method for time series data.

    The DB method is a non-parametric method that generates bootstrapped samples by fitting a distribution to the residuals and then generating new residuals from the fitted distribution. The new residuals are then added to the fitted values to create the bootstrapped samples.

    Attributes
    ----------
    resids_dist : scipy.stats.rv_continuous or None
        The distribution object used to generate the bootstrapped samples. If None, the distribution has not been fit yet.
    resids_dist_params : tuple or None
        The parameters of the distribution used to generate the bootstrapped samples. If None, the distribution has not been fit yet.

    Methods
    -------
    __init__ : Initialize the BaseDistributionBootstrap class.
    fit_distribution(resids: np.ndarray) -> tuple[rv_continuous, tuple]
        Fit the specified distribution to the residuals and return the distribution object and the parameters of the distribution.
    """

    distribution_methods = {
        "poisson": poisson,
        "exponential": expon,
        "normal": norm,
        "gamma": gamma,
        "beta": beta,
        "lognormal": lognorm,
        "weibull": weibull_min,
        "pareto": pareto,
        "geometric": geom,
        "uniform": uniform,
    }

    def __init__(
        self,
        *args_base_residual,
        distribution: str = "normal",
        refit: bool = False,
        **kwargs_base_residual,
    ) -> None:
        """
        Initialize the BaseDistributionBootstrap class.

        Parameters
        ----------
        distribution: str, default='normal'
            The distribution to use for generating the bootstrapped samples. Must be one of 'poisson', 'exponential', 'normal', 'gamma', 'beta', 'lognormal', 'weibull', 'pareto', 'geometric', or 'uniform'.
        refit: bool, default=False
            Whether to refit the distribution to the resampled residuals for each bootstrap. If False, the distribution is fit once to the residuals and the same distribution is used for all bootstraps.

        Notes
        -----
        The distribution is fit to the residuals using the `fit` method of the distribution object. The parameters of the distribution are then used to generate new residuals using the `rvs` method of the distribution object.
        """
        super().__init__(*args_base_residual, **kwargs_base_residual)

        if self.model_type == "var":
            raise ValueError(
                "model_type cannot be 'var' for distribution bootstrap, since we can only fit uni-variate distributions."
            )

        self.distribution = distribution
        self.refit = refit

        self.resids_dist = None
        self.resids_dist_params = ()

    @property
    def distribution(self) -> str:
        """Getter for distribution."""
        return self._distribution

    @distribution.setter
    def distribution(self, value: str) -> None:
        """Setter for distribution. Performs validation on assignment."""
        if not isinstance(value, str):
            raise TypeError("distribution must be a string.")

        # Check if the distribution exists in the dictionary
        if value.lower() not in self.distribution_methods:
            raise ValueError(
                f"Invalid distribution: {value}; must be a valid distribution name in distribution_methods."
            )
        self._distribution = value.lower()

    def fit_distribution(
        self, resids: np.ndarray
    ) -> tuple[rv_continuous, tuple]:
        """
        Fit the specified distribution to the residuals and return the distribution object and the parameters of the distribution.

        Parameters
        ----------
        resids : np.ndarray
            The residuals to fit the distribution to.

        Returns
        -------
        resids_dist : scipy.stats.rv_continuous
            The distribution object used to generate the bootstrapped samples.
        resids_dist_params : tuple
            The parameters of the distribution used to generate the bootstrapped samples.
        """
        dist = self.distribution_methods[self.distribution]
        # Fit the distribution to the residuals
        params = dist.fit(resids)
        resids_dist = dist
        resids_dist_params = params
        return resids_dist, resids_dist_params


# Either fit distribution to resids once and generate new samples from fitted distribution with new random seed, or resample resids once and fit distribution to resampled resids, then generate new samples from fitted distribution with the same random seed n_bootstrap times.
class WholeDistributionBootstrap(BaseDistributionBootstrap):
    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, exog: np.ndarray | None = None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        # Fit the model and residuals
        self._fit_model(X=X, exog=exog)
        # Fit the specified distribution to the residuals
        if not self.refit:
            if self.resids_dist is None or self.resids_dist_params == ():
                (
                    self.resids_dist,
                    self.resids_dist_params,
                ) = super().fit_distribution(self.resids)

            # Generate new residuals from the fitted distribution
            bootstrap_residuals = self.resids_dist.rvs(
                *self.resids_dist_params,
                size=X.shape[0],
                random_state=self.rng.integers(0, 2**32 - 1),
            ).reshape(-1, 1)

            # Add new residuals to the fitted values to create the bootstrap time series
            bootstrap_samples = self.X_fitted + bootstrap_residuals
            return [np.arange(0, X.shape[0])], [bootstrap_samples]

        else:
            # Resample residuals
            resampled_indices = generate_random_indices(
                self.resids.shape[0], self.rng
            )
            resampled_residuals = self.resids[resampled_indices]
            resids_dist, resids_dist_params = super().fit_distribution(
                resampled_residuals
            )
            # Generate new residuals from the fitted distribution
            bootstrap_residuals = resids_dist.rvs(
                *resids_dist_params, size=X.shape[0], random_state=self.rng
            ).reshape(-1, 1)

            # Add the bootstrapped residuals to the fitted values
            bootstrap_samples = self.X_fitted + resampled_residuals
            return [resampled_indices], [bootstrap_samples]


class BlockDistributionBootstrap(
    BaseDistributionBootstrap, BaseBlockBootstrap
):
    def __init__(
        self,
        *args_base_residual,
        distribution: str = "normal",
        refit: bool = False,
        bootstrap_type: str | None = None,
        kwargs_base_block: dict[str, Any] | None = None,
        **kwargs_base_residual,
    ) -> None:
        kwargs_base_block = (
            {} if kwargs_base_block is None else kwargs_base_block
        )
        super().__init__(
            *args_base_residual,
            distribution=distribution,
            refit=refit,
            **kwargs_base_residual,
        )
        BaseBlockBootstrap.__init__(
            self, bootstrap_type=bootstrap_type, **kwargs_base_block
        )

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, exog: np.ndarray | None = None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        # Fit the model and residuals
        super()._fit_model(X=X, exog=exog)
        (
            block_indices,
            block_data,
        ) = BaseBlockBootstrap._generate_samples_single_bootstrap(
            self, X=self.resids
        )
        block_data_concat = np.concatenate(block_data, axis=0)
        # Fit the specified distribution to the residuals
        if not self.refit:
            if self.resids_dist is None or self.resids_dist_params == ():
                (
                    self.resids_dist,
                    self.resids_dist_params,
                ) = super().fit_distribution(block_data_concat)

            # Generate new residuals from the fitted distribution
            bootstrap_residuals = self.resids_dist.rvs(
                *self.resids_dist_params,
                size=block_data_concat.shape[0],
                random_state=self.rng.integers(0, 2**32 - 1),
            ).reshape(-1, 1)

            # Add new residuals to the fitted values to create the bootstrap time series
            bootstrap_samples = self.X_fitted + bootstrap_residuals
            return [np.arange(0, block_data_concat.shape[0])], [
                bootstrap_samples
            ]

        else:
            # Resample residuals
            resids_dist, resids_dist_params = super().fit_distribution(
                block_data_concat
            )
            # Generate new residuals from the fitted distribution
            bootstrap_residuals = resids_dist.rvs(
                *resids_dist_params,
                size=block_data_concat.shape[0],
                random_state=self.rng,
            ).reshape(-1, 1)

            # Add the bootstrapped residuals to the fitted values
            bootstrap_samples = self.X_fitted + bootstrap_residuals
            return block_indices, [bootstrap_samples]


class BaseSieveBootstrap(BaseResidualBootstrap):
    """
    Base class for Sieve bootstrap.

    This class provides the core functionalities for implementing the Sieve
    bootstrap method, allowing for the fitting of various models to the residuals
    and generation of bootstrapped samples.
    The Sieve bootstrap is a parametric method that generates bootstrapped samples by fitting a model to the residuals and then generating new residuals from the fitted model. The new residuals are then added to the fitted values to create the bootstrapped samples.

    Attributes
    ----------
    resids_model_params : dict or None
        Parameters for the residual model, specified through `kwargs_base_sieve`.
    resids_coefs : type or None
        Coefficients of the fitted residual model. Replace "type" with the specific type if known.
    resids_fit_model : type or None
        Fitted residual model object. Replace "type" with the specific type if known.

    Methods
    -------
    __init__ : Initialize the BaseSieveBootstrap class.
    _fit_resids_model : Fit the residual model to the residuals.
    """

    def __init__(
        self,
        *args_base_residual,
        resids_model_type: ModelTypes = "ar",
        resids_order: OrderTypes = None,
        save_resids_models: bool = False,
        kwargs_base_sieve: dict | None = None,
        **kwargs_base_residual,
    ) -> None:
        """
        Initialize the BaseSieveBootstrap class.

        Parameters
        ----------
        *args_base_residual:
            Variable length argument list.
        resids_model_type : str, default="ar"
            The model type to use for fitting the residuals. Must be one of "ar", "arima", "sarima", "var", or "arch".
        resids_order : Integral or list or tuple, default=None
            The order of the model to use for fitting the residuals. If None, the order is automatically determined.
        save_resids_models : bool, default=False
            Whether to save the fitted models for the residuals.
        kwargs_base_sieve : dict, default=None
            Keyword arguments to pass to the Sieve bootstrap class.
        **kwargs_base_residual:
            Arbitrary keyword arguments.
        """
        kwargs_base_sieve = (
            {} if kwargs_base_sieve is None else kwargs_base_sieve
        )
        super().__init__(*args_base_residual, **kwargs_base_residual)
        if self.model_type == "var":
            self._resids_model_type = "var"
        else:
            self.resids_model_type = resids_model_type
        self.resids_order = resids_order
        self.save_resids_models = save_resids_models
        self.resids_model_params = kwargs_base_sieve

        self.resids_coefs = None
        self.resids_fit_model = None

    @property
    def resids_model_type(self) -> ModelTypes:
        return self._resids_model_type

    @resids_model_type.setter
    def resids_model_type(self, value: ModelTypes) -> None:
        validate_literal_type(value, ModelTypes)
        value = value.lower()
        if value == "var" and self.model_type != "var":
            raise ValueError(
                "resids_model_type can be 'var' only if model_type is also 'var'."
            )
        self._resids_model_type = value

    @property
    def resids_order(self) -> OrderTypes:
        return self._resids_order

    @resids_order.setter
    def resids_order(self, value) -> None:
        """
        Set the order of residuals.

        Parameters
        ----------
        value : Integral or list or tuple
            The order value to be set. Must be a positive integer, or a list/tuple of positive integers.

        Raises
        ------
        TypeError
            If the value is not of the expected type (Integral, list, or tuple).
        ValueError
            If the value is an integral but is negative.
            If the value is a list/tuple and not all elements are positive integers.
        """
        validate_order(value)
        self._resids_order = value

    def _fit_resids_model(
        self, X: np.ndarray
    ) -> tuple[FittedModelTypes, OrderTypesWithoutNone, np.ndarray]:
        """
        Fit the residual model to the residuals.

        Parameters
        ----------
        X : np.ndarray
            The residuals to fit the model to.

        Returns
        -------
        resids_fit_model : type
            The fitted residual model object. Replace "type" with the specific type if known.
        resids_order : Integral or list or tuple
            The order of the fitted residual model.
        resids_coefs : np.ndarray
            The coefficients of the fitted residual model.
        """
        if self.resids_fit_model is None or self.resids_coefs is None:
            resids_fit_obj = TSFitBestLag(
                model_type=self.resids_model_type,
                order=self.resids_order,
                save_models=self.save_resids_models,
                **self.resids_model_params,
            )
            resids_fit_model = resids_fit_obj.fit(X, exog=None).model
            resids_order = resids_fit_obj.get_order()
            resids_coefs = resids_fit_obj.get_coefs()
            self.resids_fit_model = resids_fit_model
            self.resids_order = resids_order
            self.resids_coefs = resids_coefs


class WholeSieveBootstrap(BaseSieveBootstrap):
    """
    Implementation of the Sieve bootstrap method for time series data.

    Methods
    -------
    _generate_samples_single_bootstrap : Generate a single bootstrapped sample.
    """

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, exog: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        self._fit_model(X=X, exog=exog)
        self._fit_resids_model(X=self.resids)

        ts_simulator = TimeSeriesSimulator(
            X_fitted=self.X_fitted,
            rng=self.rng,
            fitted_model=self.resids_fit_model,
        )

        simulated_samples = ts_simulator.generate_samples_sieve(
            model_type=self.resids_model_type,
            resids_lags=self.resids_order,
            resids_coefs=self.resids_coefs,
            resids=self.resids,
        )

        return [np.arange(0, X.shape[0])], [simulated_samples]


class BlockSieveBootstrap(BaseSieveBootstrap, BaseBlockBootstrap):
    """
    Implementation of the Sieve bootstrap method for time series data.

    Methods
    -------
    _generate_samples_single_bootstrap : Generate a single bootstrapped sample.
    """

    def __init__(
        self,
        *args_base_residual,
        resids_model_type: ModelTypes = "ar",
        resids_order: OrderTypes = None,
        save_resids_models: bool = False,
        bootstrap_type: str | None = None,
        kwargs_base_sieve: dict[str, Any] | None = None,
        kwargs_base_block: dict[str, Any] | None = None,
        **kwargs_base_residual,
    ) -> None:
        """
        Initialize the BaseSieveBootstrap class.

        Parameters
        ----------
        *args_base_residual:
            Variable length argument list.
        resids_model_type : str, default="ar"
            The model type to use for fitting the residuals. Must be one of "ar", "arima", "sarima", "var", or "arch".
        resids_order : Integral or list or tuple, default=None
            The order of the model to use for fitting the residuals. If None, the order is automatically determined.
        save_resids_models : bool, default=False
            Whether to save the fitted models for the residuals.
        bootstrap_type : str, default=None
            The type of bootstrap to use. Must be one of "moving", "circular", "stationary", or "non-overlapping".
        kwargs_base_sieve : dict, default=None
            Keyword arguments to pass to the Sieve bootstrap class.
        kwargs_base_block : dict, default=None
            Keyword arguments to pass to the Block bootstrap class.
        **kwargs_base_residual:
            Arbitrary keyword arguments.
        """
        kwargs_base_block = (
            {} if kwargs_base_block is None else kwargs_base_block
        )
        kwargs_base_sieve = (
            {} if kwargs_base_sieve is None else kwargs_base_sieve
        )

        super().__init__(
            *args_base_residual,
            resids_model_type=resids_model_type,
            resids_order=resids_order,
            save_resids_models=save_resids_models,
            kwargs_base_sieve=kwargs_base_sieve,
            **kwargs_base_residual,
        )
        BaseBlockBootstrap.__init__(
            self, bootstrap_type=bootstrap_type, **kwargs_base_block
        )

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, exog: np.ndarray | None = None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        # Fit the model and residuals
        super()._fit_model(X=X, exog=exog)
        super()._fit_resids_model(X=self.resids)

        ts_simulator = TimeSeriesSimulator(
            X_fitted=self.X_fitted,
            rng=self.rng,
            fitted_model=self.resids_fit_model,
        )

        simulated_samples = ts_simulator.generate_samples_sieve(
            model_type=self.resids_model_type,
            resids_lags=self.resids_order,
            resids_coefs=self.resids_coefs,
            resids=self.resids,
        )

        resids_resids = self.X_fitted - simulated_samples
        (
            block_indices,
            resids_resids_resampled,
        ) = BaseBlockBootstrap._generate_samples_single_bootstrap(
            self, X=resids_resids
        )
        resids_resids_resampled_concat = np.concatenate(
            resids_resids_resampled, axis=0
        )

        bootstrapped_samples = self.X_fitted + resids_resids_resampled_concat

        return block_indices, [bootstrapped_samples]
