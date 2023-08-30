from __future__ import annotations

from functools import partial
from numbers import Integral
from typing import Callable, Optional, Union, Literal

import numpy as np
from scipy.signal.windows import tukey
from sklearn.decomposition import PCA

from ts_bs.bootstrap import (
    CircularBlockBootstrap,
    MovingBlockBootstrap,
    NonOverlappingBlockBootstrap,
    StationaryBlockBootstrap,
)
from ts_bs.utils.types import (
    BlockCompressorTypes,
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


class BaseTimeSeriesBootstrapConfig:
    def __init__(
        self,
        n_bootstraps: Integral = 10, # type: ignore
        rng: Integral | np.random.Generator | None = None,
    ):
        self.n_bootstraps = n_bootstraps
        self.rng = rng

    @property
    def rng(self) -> np.random.Generator:
        """Getter for rng."""
        return self._rng

    @rng.setter
    def rng(self, value: RngTypes) -> None:
        """Setter for rng. Performs validation on assignment."""
        self._rng = validate_rng(value)

    @property
    def n_bootstraps(self) -> Integral:
        """Getter for n_bootstraps."""
        return self._n_bootstraps

    @n_bootstraps.setter
    def n_bootstraps(self, value) -> None:
        """Setter for n_bootstraps. Performs validation on assignment."""
        validate_integers(value, min_value=1)  # type: ignore
        self._n_bootstraps = value


class BlockBootstrapConfig(BaseTimeSeriesBootstrapConfig):
    """
    Block Bootstrap base class for time series data.
    """

    def __init__(
        self,
        block_length: Integral | None = None,
        block_length_distribution: str | None = None,
        wrap_around_flag: bool = False,
        overlap_flag: bool = False,
        combine_generation_and_sampling_flag: bool = False,
        block_weights: Callable | np.ndarray | None = None,
        tapered_weights: Callable | None = None,
        overlap_length: Integral | None = None,
        min_block_length: Integral | None = None,
        n_bootstraps: Integral = 10, # type: ignore
        rng: Integral | np.random.Generator | None = None,
    ) -> None:
        """
        Block Bootstrap class for time series data.

        Parameters
        ----------
        n_bootstraps : Integral, default=10
            The number of bootstrap samples to create.
        block_length : Integral, default=None
            The length of the blocks to sample. If None, the block length is automatically set to the square root of the number of observations.
        block_length_distribution : str, default=None
            The block length distribution function to use. If None, the block length distribution is not utilized.
        wrap_around_flag : bool, default=False
            Whether to wrap around the data when generating blocks.
        overlap_flag : bool, default=False
            Whether to allow blocks to overlap.
        combine_generation_and_sampling_flag : bool, default=False
            Whether to combine the block generation and sampling steps.
        rng : Integral or np.random.Generator, default=np.random.default_rng()
            The random number generator or seed used to generate the bootstrap samples.
        block_weights : array-like of shape (n_blocks,), default=None
            The weights to use when sampling blocks.
        tapered_weights : callable, default=None
            The tapered weights to use when sampling blocks.
        overlap_length : Integral, default=None
            The length of the overlap between blocks.
        min_block_length : Integral, default=None
            The minimum length of the blocks.
        """
        # Initialize the parent class
        super().__init__(n_bootstraps=n_bootstraps, rng=rng)

        self.block_length_distribution = block_length_distribution
        self.block_length = block_length
        self.wrap_around_flag = wrap_around_flag
        self.overlap_flag = overlap_flag
        self.combine_generation_and_sampling_flag = (
            combine_generation_and_sampling_flag
        )

        self.block_weights = block_weights
        self.tapered_weights = tapered_weights
        self.overlap_length = overlap_length
        self.min_block_length = min_block_length

    @property
    def block_length(self) -> Integral | None:
        """Getter for block_length."""
        return self._block_length

    @block_length.setter
    def block_length(self, value) -> None:
        """
        Setter for block_length. Performs validation on assignment.

        Parameters
        ----------
        value : Integral or None.
        """
        if value is not None and (
            not isinstance(value, Integral) or value < 1
        ):
            raise ValueError(
                "Block length needs to be None or an integer >= 1."
            )
        self._block_length = value

    @property
    def block_length_distribution(self) -> str | None:
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

    @property
    def wrap_around_flag(self) -> bool:
        """Getter for wrap_around_flag."""
        return self._wrap_around_flag

    @wrap_around_flag.setter
    def wrap_around_flag(self, value) -> None:
        """
        Setter for wrap_around_flag. Performs validation on assignment.

        Parameters
        ----------
        value : bool
            Whether to wrap around the data when generating blocks.
        """
        if not isinstance(value, bool):
            raise TypeError("wrap_around_flag must be a boolean.")
        self._wrap_around_flag = value

    @property
    def overlap_flag(self) -> bool:
        """Getter for overlap_flag."""
        return self._overlap_flag

    @overlap_flag.setter
    def overlap_flag(self, value) -> None:
        """
        Setter for overlap_flag. Performs validation on assignment.

        Parameters
        ----------
        value : bool
            Whether to allow blocks to overlap.
        """
        if not isinstance(value, bool):
            raise TypeError("overlap_flag must be a boolean.")
        self._overlap_flag = value

    @property
    def combine_generation_and_sampling_flag(self) -> bool:
        """Getter for combine_generation_and_sampling_flag."""
        return self._combine_generation_and_sampling_flag

    @combine_generation_and_sampling_flag.setter
    def combine_generation_and_sampling_flag(self, value) -> None:
        """
        Setter for combine_generation_and_sampling_flag. Performs validation on assignment.

        Parameters
        ----------
        value : bool
            Whether to combine the block generation and sampling steps.
        """
        if not isinstance(value, bool):
            raise TypeError(
                "combine_generation_and_sampling_flag must be a boolean."
            )
        self._combine_generation_and_sampling_flag = value

    @property
    def block_weights(self) -> Callable | np.ndarray | None:
        """Getter for block_weights."""
        return self._block_weights

    @block_weights.setter
    def block_weights(self, value) -> None:
        """
        Setter for block_weights. Performs validation on assignment.

        Parameters
        ----------
        value : array-like of shape (n_blocks,)
            The weights to use when sampling blocks.
        """
        if value is not None and (
            not isinstance(value, np.ndarray) or not callable(value)
        ):
            raise TypeError("block_weights must be a numpy array or callable.")
        self._block_weights = value

    @property
    def tapered_weights(self) -> Callable | None:
        """Getter for tapered_weights."""
        return self._tapered_weights

    @tapered_weights.setter
    def tapered_weights(self, value) -> None:
        """
        Setter for tapered_weights. Performs validation on assignment.

        Parameters
        ----------
        value : callable
            The tapered weights to use when sampling blocks.
        """
        if value is not None and not callable(value):
            raise TypeError("tapered_weights must be a callable.")
        self._tapered_weights = value

    @property
    def overlap_length(self) -> Integral | None:
        """Getter for overlap_length."""
        return self._overlap_length

    @overlap_length.setter
    def overlap_length(self, value) -> None:
        """
        Setter for overlap_length. Performs validation on assignment.

        Parameters
        ----------
        value : Integral or None.
        """
        if value is not None and (
            not isinstance(value, Integral) or value < 1
        ):
            raise ValueError("overlap_length must be None or an integer >= 1.")
        self._overlap_length = value

    @property
    def min_block_length(self) -> Integral | None:
        """Getter for min_block_length."""
        return self._min_block_length

    @min_block_length.setter
    def min_block_length(self, value) -> None:
        """
        Setter for min_block_length. Performs validation on assignment.

        Parameters
        ----------
        value : Integral or None.
        """
        if value is not None and (
            not isinstance(value, Integral) or value < 1
        ):
            raise ValueError(
                "min_block_length must be None or an integer >= 1."
            )
        self._min_block_length = value


class MovingBlockBootstrapConfig(BlockBootstrapConfig):
    """
    Configuration class for MovingBlockBootstrap.

    This class is a specialized configuration class that sets
    `wrap_around_flag` to False, `overlap_flag` to True, and
    `block_length_distribution` to None.
    """
    def __init__(self, block_length: Integral | None = None, **kwargs) -> None:
        """
        Initialize self.

        Parameters
        ----------
        block_length : Integral, default=None
            The length of the blocks to sample.
        kwargs
            Additional keyword arguments to pass to the parent BlockBootstrapConfig class.
            See the documentation for BlockBootstrapConfig for more information.
        """
        super().__init__(
            block_length=block_length,
            wrap_around_flag=False,
            overlap_flag=True,
            block_length_distribution=None,
            **kwargs,
        )

    @BlockBootstrapConfig.overlap_flag.setter
    def overlap_flag(self, value):
        raise ValueError(
            "overlap_flag cannot be modified in a MovingBlockBootstrapConfig instance."
        )

    @BlockBootstrapConfig.wrap_around_flag.setter
    def wrap_around_flag(self, value):
        raise ValueError(
            "wrap_around_flag cannot be modified in a MovingBlockBootstrapConfig instance."
        )

    @BlockBootstrapConfig.block_length_distribution.setter
    def block_length_distribution(self, value):
        raise ValueError(
            "block_length_distribution cannot be modified in a MovingBlockBootstrapConfig instance."
        )


class StationaryBlockBootstrapConfig(BlockBootstrapConfig):
    """
    Configuration class for StationaryBlockBootstrap.

    This class is a specialized configuration class that sets
    `wrap_around_flag` to False, `overlap_flag` to True, and
    `block_length_distribution` to "geometric".
    """
    def __init__(self, block_length: Integral | None = None, **kwargs) -> None:
        """
        Initialize self.

        Parameters
        ----------
        block_length : Integral, default=None
            The length of the blocks to sample.
        kwargs
            Additional keyword arguments to pass to the parent BlockBootstrapConfig class.
            See the documentation for BlockBootstrapConfig for more information.
        """
        super().__init__(
            block_length=block_length,
            wrap_around_flag=False,
            overlap_flag=True,
            block_length_distribution="geometric",
            **kwargs,
        )


    @BlockBootstrapConfig.overlap_flag.setter
    def overlap_flag(self, value):
        raise ValueError(
            "overlap_flag cannot be modified in a StationaryBlockBootstrapConfig instance."
        )

    @BlockBootstrapConfig.wrap_around_flag.setter
    def wrap_around_flag(self, value):
        raise ValueError(
            "wrap_around_flag cannot be modified in a StationaryBlockBootstrapConfig instance."
        )

    @BlockBootstrapConfig.block_length_distribution.setter
    def block_length_distribution(self, value):
        raise ValueError(
            "block_length_distribution cannot be modified in a StationaryBlockBootstrapConfig instance."
        )


class CircularBlockBootstrapConfig(BlockBootstrapConfig):
    """
    Configuration class for CircularBlockBootstrap.

    This class is a specialized configuration class that sets
    `wrap_around_flag` to True, `overlap_flag` to True, and
    `block_length_distribution` to None.
    """
    def __init__(self, block_length: Integral | None = None, **kwargs) -> None:
        """
        Initialize self.

        Parameters
        ----------
        block_length : Integral, default=None
            The length of the blocks to sample.
        kwargs
            Additional keyword arguments to pass to the parent BlockBootstrapConfig class.
            See the documentation for BlockBootstrapConfig for more information.
        """
        super().__init__(
            block_length=block_length,
            wrap_around_flag=True,
            overlap_flag=True,
            block_length_distribution=None,
            **kwargs,
        )


    @BlockBootstrapConfig.overlap_flag.setter
    def overlap_flag(self, value):
        raise ValueError(
            "overlap_flag cannot be modified in a CircularBlockBootstrapConfig instance."
        )

    @BlockBootstrapConfig.wrap_around_flag.setter
    def wrap_around_flag(self, value):
        raise ValueError(
            "wrap_around_flag cannot be modified in a CircularBlockBootstrapConfig instance."
        )

    @BlockBootstrapConfig.block_length_distribution.setter
    def block_length_distribution(self, value):
        raise ValueError(
            "block_length_distribution cannot be modified in a CircularBlockBootstrapConfig instance."
        )



class NonOverlappingBlockBootstrapConfig(BlockBootstrapConfig):
    """
    Configuration class for NonOverlappingBlockBootstrap.

    This class is a specialized configuration class that sets
    `wrap_around_flag` to False, `overlap_flag` to False, and
    `block_length_distribution` to None.
    """
    def __init__(self, block_length: Integral | None = None, **kwargs) -> None:
        """
        Initialize self.

        Parameters
        ----------
        block_length : Integral, default=None
            The length of the blocks to sample.
        kwargs
            Additional keyword arguments to pass to the parent BlockBootstrapConfig class.
            See the documentation for BlockBootstrapConfig for more information.
        """
        super().__init__(
            block_length=block_length,
            wrap_around_flag=False,
            overlap_flag=False,
            block_length_distribution=None,
            **kwargs,
        )

    @BlockBootstrapConfig.overlap_flag.setter
    def overlap_flag(self, value):
        raise ValueError(
            "overlap_flag cannot be modified in a NonOverlappingBlockBootstrapConfig instance."
        )

    @BlockBootstrapConfig.wrap_around_flag.setter
    def wrap_around_flag(self, value):
        raise ValueError(
            "wrap_around_flag cannot be modified in a NonOverlappingBlockBootstrapConfig instance."
        )

    @BlockBootstrapConfig.block_length_distribution.setter
    def block_length_distribution(self, value):
        raise ValueError(
            "block_length_distribution cannot be modified in a NonOverlappingBlockBootstrapConfig instance."
        )


class BaseBlockBootstrapConfig(BlockBootstrapConfig):
    """
    Configuration class for BaseBlockBootstrap.

    This class is a specialized configuration class that allows for the
    `bootstrap_type` parameter to be set. The `bootstrap_type` parameter
    determines the type of block bootstrap to use.
    """
    bootstrap_type_dict = {
        "nonoverlapping": NonOverlappingBlockBootstrap,
        "moving": MovingBlockBootstrap,
        "stationary": StationaryBlockBootstrap,
        "circular": CircularBlockBootstrap,
    }

    def __init__(
        self,
        bootstrap_type: str | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize self.

        Parameters
        ----------
        bootstrap_type : str, default=None
            The type of block bootstrap to use. Must be one of "nonoverlapping", "moving", "stationary", or "circular".
        kwargs
            Additional keyword arguments to pass to the parent BlockBootstrapConfig class.
            See the documentation for BlockBootstrapConfig for more information.
        """
        super().__init__(**kwargs)
        self.bootstrap_type = bootstrap_type

    @property
    def bootstrap_type(self) -> str | None:
        return self._bootstrap_type

    @bootstrap_type.setter
    def bootstrap_type(self, value: str | None):
        valid_types = set(self.bootstrap_type_dict.keys())

        if value is not None and value not in valid_types:
            raise ValueError(f"bootstrap_type must be one of {valid_types}.")
        self._bootstrap_type = value


class BartlettsBootstrapConfig(BaseBlockBootstrapConfig):
    """Config class for BartlettsBootstrap.

    This class is a specialized configuration class that sets
    `tapered_weights` to Bartlett's window and `bootstrap_type` to "moving".
    """

    def __init__(self, block_length: None | Integral = None, **kwargs) -> None:
        """Initialize BartlettsBootstrapConfig.

        Parameters
        ----------
        block_length : Optional[Integral], default=None
            The length of the blocks to sample.
        kwargs
            Additional keyword arguments to pass to the parent BaseBlockBootstrapConfig class.
            See the documentation for BaseBlockBootstrapConfig for more information.
        """
        super().__init__(
            bootstrap_type="moving",  # Forced to "moving"
            block_length=block_length,
            tapered_weights=np.bartlett,  # Forced to np.bartlett
            **kwargs,
        )

    @BaseBlockBootstrapConfig.tapered_weights.setter
    def tapered_weights(self, value):
        raise ValueError(
            "tapered_weights cannot be modified in a BartlettsBootstrapConfig instance."
        )

    @BaseBlockBootstrapConfig.bootstrap_type.setter
    def bootstrap_type(self, value):
        raise ValueError(
            "bootstrap_type cannot be modified in a BartlettsBootstrapConfig instance."
        )


class HammingBootstrapConfig(BaseBlockBootstrapConfig):
    """Config class for HammingBootstrap.

    This class is a specialized configuration class that sets
    `tapered_weights` to Hamming window and `bootstrap_type` to "moving".
    """

    def __init__(self, block_length: None | Integral = None, **kwargs) -> None:
        """Initialize HammingBootstrapConfig.

        Parameters
        ----------
        block_length : Optional[Integral], default=None
            The length of the blocks to sample.
        kwargs
            Additional keyword arguments to pass to the parent BaseBlockBootstrapConfig class.
            See the documentation for BaseBlockBootstrapConfig for more information.
        """
        super().__init__(
            bootstrap_type="moving",  # Forced to "moving"
            block_length=block_length,
            tapered_weights=np.hamming,  # Forced to np.hamming
            **kwargs,
        )

    @BaseBlockBootstrapConfig.tapered_weights.setter
    def tapered_weights(self, value):
        raise ValueError(
            "tapered_weights cannot be modified in a HammingBootstrapConfig instance."
        )

    @BaseBlockBootstrapConfig.bootstrap_type.setter
    def bootstrap_type(self, value):
        raise ValueError(
            "bootstrap_type cannot be modified in a HammingBootstrapConfig instance."
        )


class HanningBootstrapConfig(BaseBlockBootstrapConfig):
    """Config class for HanningBootstrap.

    This class is a specialized configuration class that sets
    `tapered_weights` to Hanning window and `bootstrap_type` to "moving".
    """

    def __init__(self, block_length: None | Integral = None, **kwargs) -> None:
        """Initialize HanningBootstrapConfig.

        Parameters
        ----------
        block_length : Optional[Integral], default=None
            The length of the blocks to sample.
        kwargs
            Additional keyword arguments to pass to the parent BaseBlockBootstrapConfig class.
            See the documentation for BaseBlockBootstrapConfig for more information.
        """
        super().__init__(
            bootstrap_type="moving",  # Forced to "moving"
            block_length=block_length,
            tapered_weights=np.hanning,  # Forced to np.hanning
            **kwargs,
        )

    @BaseBlockBootstrapConfig.tapered_weights.setter
    def tapered_weights(self, value):
        raise ValueError(
            "tapered_weights cannot be modified in a HanningBootstrapConfig instance."
        )

    @BaseBlockBootstrapConfig.bootstrap_type.setter
    def bootstrap_type(self, value):
        raise ValueError(
            "bootstrap_type cannot be modified in a HanningBootstrapConfig instance."
        )


class BlackmanBootstrapConfig(BaseBlockBootstrapConfig):
    """Config class for BlackmanBootstrap.

    This class is a specialized configuration class that sets
    `tapered_weights` to Blackman window and `bootstrap_type` to "moving".
    """

    def __init__(self, block_length: None | Integral = None, **kwargs) -> None:
        """Initialize BlackmanBootstrapConfig.

        Parameters
        ----------
        block_length : Optional[Integral], default=None
            The length of the blocks to sample.
        kwargs
            Additional keyword arguments to pass to the parent BaseBlockBootstrapConfig class.
            See the documentation for BaseBlockBootstrapConfig for more information.
        """
        super().__init__(
            bootstrap_type="moving",  # Forced to "moving"
            block_length=block_length,
            tapered_weights=np.blackman,  # Forced to np.blackman
            **kwargs,
        )

    @BaseBlockBootstrapConfig.tapered_weights.setter
    def tapered_weights(self, value):
        raise ValueError(
            "tapered_weights cannot be modified in a BlackmanBootstrapConfig instance."
        )

    @BaseBlockBootstrapConfig.bootstrap_type.setter
    def bootstrap_type(self, value):
        raise ValueError(
            "bootstrap_type cannot be modified in a BlackmanBootstrapConfig instance."
        )


class TukeyBootstrapConfig(BaseBlockBootstrapConfig):
    """Config class for TukeyBootstrap.

    This class is a specialized configuration class that sets
    `tapered_weights` to Tukey window and `bootstrap_type` to "moving".
    """

    tukey_alpha = staticmethod(partial(tukey, alpha=0.5))

    def __init__(self, block_length: None | Integral = None, **kwargs) -> None:
        """Initialize TukeyBootstrapConfig.

        Parameters
        ----------
        block_length : Optional[Integral], default=None
            The length of the blocks to sample.
        kwargs
            Additional keyword arguments to pass to the parent BaseBlockBootstrapConfig class.
            See the documentation for BaseBlockBootstrapConfig for more information.
        """
        super().__init__(
            bootstrap_type="moving",  # Forced to "moving"
            block_length=block_length,
            tapered_weights=self.tukey_alpha,  # Forced to tukey_alpha
            **kwargs,
        )

    @BaseBlockBootstrapConfig.tapered_weights.setter
    def tapered_weights(self, value):
        raise ValueError(
            "tapered_weights cannot be modified in a TukeyBootstrapConfig instance."
        )

    @BaseBlockBootstrapConfig.bootstrap_type.setter
    def bootstrap_type(self, value):
        raise ValueError(
            "bootstrap_type cannot be modified in a TukeyBootstrapConfig instance."
        )



class BaseResidualBootstrapConfig(BaseTimeSeriesBootstrapConfig):
    """
    Configuration class for BaseResidualBootstrap.

    This class is a specialized configuration class that enables residual
    time series bootstrapping.
    """

    def __init__(
        self,
        n_bootstraps: Integral = 10, # type: ignore
        rng: Integral | np.random.Generator | None = None,
        model_type: ModelTypesWithoutArch = "ar",
        order: OrderTypes = None,
        save_models: bool = False,
        **kwargs,
    ):
        """
        Initialize self.

        Parameters
        ----------
        n_bootstraps : Integral, default=10
            The number of bootstrap samples to create.
        model_type : str, default="ar"
            The model type to use. Must be one of "ar", "arima", "sarima", "var", or "arch".
        order : Integral or list or tuple, default=None
            The order of the model. If None, the best order is chosen via TSFitBestLag. If Integral, it is the lag order for AR, ARIMA, and SARIMA, and the lag order for ARCH. If list or tuple, the order is a tuple of (p, o, q) for ARIMA and (p, d, q, s) for SARIMAX. It is either a single Integral or a list of non-consecutive ints for AR, and an Integral for VAR and ARCH. If None, the best order is chosen via TSFitBestLag. Do note that TSFitBestLag only chooses the best lag, not the best order, so for the tuple values, it only chooses the best p, not the best (p, o, q) or (p, d, q, s). The rest of the values are set to 0.
        save_models : bool, default=False
            Whether to save the fitted models.
        rng : Integral or np.random.Generator, default=np.random.default_rng()
            The random number generator or seed used to generate the bootstrap samples.
        **kwargs
            Additional keyword arguments to pass to the TSFit model.

        Raises
        ------
        ValueError
            If model_type is not one of "ar", "arima", "sarima", "var", or "arch".

        Notes
        -----
        The model_type and order parameters are passed to TSFitBestLag, which
        chooses the best lag and order for the model. The best lag and order are
        then used to fit the model to the data. The residuals are then stored
        for use in the bootstrap.

        References
        ----------
        .. [^1^] https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Residual_bootstrap
        """
        super().__init__(n_bootstraps=n_bootstraps, rng=rng)
        self.model_type = model_type
        self.order = order
        self.save_models = save_models
        self.model_params = kwargs

    @property
    def model_type(self) -> str:
        """Getter for model_type."""
        return self._model_type

    @model_type.setter
    def model_type(self, value: str) -> None:
        """Setter for model_type. Performs validation on assignment."""
        value = value.lower()
        validate_literal_type(value, ModelTypesWithoutArch) # type: ignore
        self._model_type = value

    @property
    def order(self) -> OrderTypes:
        """Getter for order."""
        return self._order

    @order.setter
    def order(self, value) -> None:
        """Setter for order. Performs validation on assignment."""
        validate_order(value)
        self._order = value

    @property
    def save_models(self) -> bool:
        """Getter for save_models."""
        return self._save_models

    @save_models.setter
    def save_models(self, value: bool) -> None:
        """Setter for save_models. Performs validation on assignment."""
        if not isinstance(value, bool):
            raise TypeError("save_models must be a boolean.")
        self._save_models = value



class BaseMarkovBootstrapConfig(BaseResidualBootstrapConfig):
    """
    Configuration class for BaseMarkovBootstrap.
    """

    def __init__(
        self,
        n_bootstraps: Integral = 10, # type: ignore
        rng: Integral | np.random.Generator | None = None,
        method: BlockCompressorTypes = "middle",
        apply_pca_flag: bool = False,
        pca: Optional[PCA] = None,
        n_iter_hmm: Integral = 10, # type: ignore
        n_fits_hmm: Integral = 1, # type: ignore
        blocks_as_hidden_states_flag: bool = False,
        n_states: Integral = 2, # type: ignore
        **kwargs,
    ):
        """
        Initialize self.

        Parameters
        ----------
        n_bootstraps : Integral, default=10
            The number of bootstrap samples to create.
        rng : Integral or np.random.Generator, default=np.random.default_rng()
            The random number generator or seed used to generate the bootstrap samples.
        method : str, default="middle"
            The method to use for compressing the blocks. Must be one of "first", "middle", "last", "mean", "mode", "median", "kmeans", "kmedians", "kmedoids".
        apply_pca_flag : bool, default=False
            Whether to apply PCA to the residuals before fitting the HMM.
        pca : PCA, default=None
            The PCA object to use for applying PCA to the residuals.
        n_iter_hmm : Integral, default=10
            Number of iterations for fitting the HMM.
        n_fits_hmm : Integral, default=1
            Number of times to fit the HMM.
        blocks_as_hidden_states_flag : bool, default=False
            Whether to use blocks as hidden states.
        n_states : Integral, default=2
            Number of states for the HMM.
        **kwargs
            Additional keyword arguments to pass to the BaseResidualBootstrapConfig class,
            except for n_bootstraps and rng, which are passed directly to the parent BaseTimeSeriesBootstrapConfig class.
            See the documentation for BaseResidualBootstrapConfig for more information.
        """
        super().__init__(n_bootstraps=n_bootstraps, rng=rng, **kwargs)
        self.method = method
        self.apply_pca_flag = apply_pca_flag
        self.pca = pca
        self.n_iter_hmm = n_iter_hmm
        self.n_fits_hmm = n_fits_hmm
        self.blocks_as_hidden_states_flag = blocks_as_hidden_states_flag
        self.n_states = n_states

    @property
    def method(self) -> str:
        """Getter for method."""
        return self._method

    @method.setter
    def method(self, value: BlockCompressorTypes) -> None:
        """Setter for method. Performs validation on assignment."""
        validate_literal_type(value, BlockCompressorTypes) # type: ignore
        self._method = value.lower()

    @property
    def apply_pca_flag(self) -> bool:
        """Getter for apply_pca_flag."""
        return self._apply_pca_flag

    @apply_pca_flag.setter
    def apply_pca_flag(self, value: bool) -> None:
        """Setter for apply_pca_flag. Performs validation on assignment."""
        if not isinstance(value, bool):
            raise TypeError("apply_pca_flag must be a boolean.")
        self._apply_pca_flag = value

    @property
    def pca(self) -> PCA | None:
        """Getter for pca."""
        return self._pca

    @pca.setter
    def pca(self, value: PCA | None) -> None:
        """Setter for pca. Performs validation on assignment."""
        if value is not None and not isinstance(value, PCA):
            raise TypeError("pca must be an instance of PCA.")
        self._pca = value

    @property
    def n_iter_hmm(self) -> Integral:
        """Getter for n_iter_hmm."""
        return self._n_iter_hmm

    @n_iter_hmm.setter
    def n_iter_hmm(self, value: Integral) -> None:
        """Setter for n_iter_hmm. Performs validation on assignment."""
        validate_integers(value, min_value=10) # type: ignore
        self._n_iter_hmm = value

    @property
    def n_fits_hmm(self) -> Integral:
        """Getter for n_fits_hmm."""
        return self._n_fits_hmm

    @n_fits_hmm.setter
    def n_fits_hmm(self, value: Integral) -> None:
        """Setter for n_fits_hmm. Performs validation on assignment."""
        validate_integers(value, min_value=1) # type: ignore
        self._n_fits_hmm = value

    @property
    def blocks_as_hidden_states_flag(self) -> bool:
        """Getter for blocks_as_hidden_states_flag."""
        return self._blocks_as_hidden_states_flag

    @blocks_as_hidden_states_flag.setter
    def blocks_as_hidden_states_flag(self, value: bool) -> None:
        """Setter for blocks_as_hidden_states_flag. Performs validation on assignment."""
        if not isinstance(value, bool):
            raise TypeError("blocks_as_hidden_states_flag must be a boolean.")
        self._blocks_as_hidden_states_flag = value

    @property
    def n_states(self) -> Integral:
        """Getter for n_states."""
        return self._n_states

    @n_states.setter
    def n_states(self, value: Integral) -> None:
        """Setter for n_states. Performs validation on assignment."""
        validate_integers(value, min_value=2)
        self._n_states = value


class BaseBiasCorrectedBootstrapConfig:
    """
    Configuration class for BaseBiasCorrectedBootstrap.

    Parameters
    ----------
    statistic : Callable
        A callable function to compute the statistic that should be preserved.
    statistic_axis : int, default=0
        The axis along which the statistic should be computed.
    statistic_keepdims : bool, default=False
        Whether to keep the dimensions of the statistic or not.
    """

    def __init__(self, statistic: Callable, statistic_axis: Optional[int] = 0, statistic_keepdims: bool = False):
        self.statistic = statistic
        self.statistic_axis = statistic_axis
        self.statistic_keepdims = statistic_keepdims
        self.validate()

    def validate(self):
        """Validate the configuration parameters."""
        if not callable(self.statistic):
            raise ValueError("statistic must be a callable function.")
 : Callable
        A callable function to compute the statistic that should be preserved.
    statistic_axis : int, default=0
        The axis along which the statistic should be computed.
    statistic_keepdims : bool, default=False
        Whether to keep the dimensions of the statistic or not.
    """

    def __init__(self, statistic: Callable, statistic_axis: Optional[int] = 0, statistic_keepdims: bool = False):
        self.statistic = statistic
        self.statistic_axis = statistic_axis
        self.statistic_keepdims = statistic_keepdims
        self.validate()

    def validate(self):
        """Validate the configuration parameters."""
        if not callable(self.statistic):
            raise ValueError("statistic must be a callable function.")
