from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Optional, Type, Union

import numpy as np
from pydantic import (
    ConfigDict,
    Field,
    PositiveInt,
    ValidationInfo,
    field_validator,
)
from scipy.signal.windows import tukey

from tsbootstrap.base_bootstrap import BaseTimeSeriesBootstrap
from tsbootstrap.block_generator import BlockGenerator
from tsbootstrap.block_length_sampler import BlockLengthSampler
from tsbootstrap.block_resampler import BlockResampler
from tsbootstrap.utils.types import DistributionTypes


class BlockBootstrap(BaseTimeSeriesBootstrap):
    """
    Block Bootstrap base class for time series data.

    Attributes
    ----------
    block_length : Optional[int]
        The length of the blocks to sample. If None, the block length is automatically set to the square root of the number of observations.
    block_length_distribution : Optional[str]
        The block length distribution function to use. If None, the block length distribution is not utilized.
    wrap_around_flag : bool
        Whether to wrap around the data when generating blocks.
    overlap_flag : bool
        Whether to allow blocks to overlap.
    combine_generation_and_sampling_flag : bool
        Whether to combine the block generation and sampling steps.
    block_weights : Optional[Union[np.ndarray, Callable]]
        The weights to use when sampling blocks.
    tapered_weights : Optional[Callable]
        The tapered weights to use when sampling blocks.
    overlap_length : Optional[int]
        The length of the overlap between blocks.
    min_block_length : Optional[int]
        The minimum length of the blocks.
    blocks : Optional[list[np.ndarray]]
        The generated blocks. Initialized as None.
    block_resampler : Optional[BlockResampler]
        The block resampler object. Initialized as None.

    Notes
    -----
    This class uses Pydantic for data validation. The `block_length`, `overlap_length`,
    and `min_block_length` fields must be greater than or equal to 1 if provided.

    The `blocks` and `block_resampler` attributes are not included in the initialization
    and are set during the bootstrap process.

    Raises
    ------
    ValueError
        If validation fails for any of the fields, e.g., if block_length is less than 1.
    """

    _tags = {"bootstrap_type": "block"}

    # Model configuration
    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
    }

    block_length: Optional[PositiveInt] = Field(default=None, ge=1)
    block_length_distribution: Optional[DistributionTypes] = Field(
        default=None
    )
    wrap_around_flag: bool = Field(default=False)
    overlap_flag: bool = Field(default=False)
    combine_generation_and_sampling_flag: bool = Field(default=False)
    block_weights: Optional[Union[np.ndarray, Callable]] = Field(default=None)
    tapered_weights: Optional[Callable] = Field(default=None)
    overlap_length: Optional[PositiveInt] = Field(default=None, ge=1)
    min_block_length: Optional[PositiveInt] = Field(default=None, ge=1)

    blocks: Optional[list[np.ndarray]] = Field(default=None, init=False)
    block_resampler: Optional[BlockResampler] = Field(default=None, init=False)

    def _check_input_bb(self, X: np.ndarray, enforce_univariate=True) -> None:
        if self.block_length is not None and self.block_length > X.shape[0]:
            raise ValueError(
                "block_length cannot be greater than the size of the input array X."
            )

    def _generate_blocks(self, X: np.ndarray) -> list[np.ndarray]:
        """Generates blocks of indices.

        Parameters
        ----------
        X : array-like of shape (n_timepoints, n_features)
            The input samples.

        Returns
        -------
        blocks : list of arrays
            The generated blocks.

        """
        self._check_input_bb(X)
        block_length_sampler = BlockLengthSampler(
            avg_block_length=(
                self.block_length
                if self.block_length is not None
                else int(np.sqrt(X.shape[0]))
            ),
            block_length_distribution=self.block_length_distribution,
            rng=self.rng,
        )

        block_generator = BlockGenerator(
            block_length_sampler=block_length_sampler,
            input_length=X.shape[0],
            rng=self.rng,
            wrap_around_flag=self.wrap_around_flag,
            overlap_length=self.overlap_length,
            min_block_length=self.min_block_length,
        )

        blocks = block_generator.generate_blocks(
            overlap_flag=self.overlap_flag
        )

        return blocks

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y=None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Generate a single bootstrap sample.

        Parameters
        ----------
        X : array-like of shape (n_timepoints, n_features)
            The input samples.

        Returns
        -------
        tuple[list[np.ndarray], list[np.ndarray]]
            A tuple containing the indices and data of the generated blocks.
        """
        if self.combine_generation_and_sampling_flag or self.blocks is None:
            blocks = self._generate_blocks(X=X)

            block_resampler = BlockResampler(
                X=X,
                blocks=blocks,
                rng=self.rng,
                block_weights=self.block_weights,
                tapered_weights=self.tapered_weights,
            )
        else:
            blocks = self.blocks
            block_resampler = self.block_resampler

        (
            block_indices,
            block_data,
        ) = block_resampler.resample_block_indices_and_data()  # type: ignore

        if not self.combine_generation_and_sampling_flag:
            self.blocks = blocks
            self.block_resampler = block_resampler

        return block_indices, block_data


class BaseBlockBootstrap(BlockBootstrap):
    """
    Base class for block bootstrapping.

    This class is a specialized class that allows for the
    `bootstrap_type` parameter to be set. The `bootstrap_type` parameter
    determines the type of block bootstrap to use.

    Parameters
    ----------
    bootstrap_type : str, optional
        The type of block bootstrap to use.
        Must be one of "nonoverlapping", "moving", "stationary", or "circular".
        Default is "moving".

    Attributes
    ----------
    bootstrap_instance : BlockBootstrap or None
        An instance of the specified bootstrap type class.

    Methods
    -------
    validate_bootstrap_type
    model_post_init
    _generate_samples_single_bootstrap
    __repr__
    """

    bootstrap_type: Optional[str] = Field(default="moving")

    @field_validator("bootstrap_type")
    @classmethod
    def validate_bootstrap_type(cls, v: Optional[str]) -> Optional[str]:
        """
        Validate the bootstrap_type.
        """
        valid_types = get_bootstrap_types_dict()
        if v is not None and v not in valid_types:
            raise ValueError(
                f"Invalid bootstrap_type: {v}. Must be one of {list(valid_types.keys())}"
            )
        return v

    def model_post_init(self, __context: ConfigDict) -> None:
        """
        Post-initialization method called after the model is fully initialized.

        Parameters
        ----------
        __context : ConfigDict
            Configuration context (unused in this implementation).
        """
        super().__init__()
        self.bootstrap_instance: Optional[BlockBootstrap] = None
        if self.bootstrap_type:
            # Get the bootstrap class based on the specified type
            bcls: Type[BlockBootstrap] = BLOCK_BOOTSTRAP_TYPES_DICT[
                self.bootstrap_type
            ]
            # Create an instance of the specified bootstrap class
            self.bootstrap_instance = bcls()

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Generate a single bootstrap sample using either the base BlockBootstrap method or the specified bootstrap_type.

        Parameters
        ----------
        X : np.ndarray
            The input samples of shape (n_timepoints, n_features).
        y : np.ndarray, optional
            The target values. By default None.

        Returns
        -------
        tuple[list[np.ndarray], list[np.ndarray]]
            A tuple containing the indices and data of the generated blocks.

        Raises
        ------
        NotImplementedError
            If the specified bootstrap class does not implement the '_generate_samples_single_bootstrap' method.
        """
        if self.bootstrap_instance is None:
            # If no specific bootstrap instance is set, use the base class method
            return super()._generate_samples_single_bootstrap(X=X, y=y)
        else:
            if hasattr(
                self.bootstrap_instance, "_generate_samples_single_bootstrap"
            ):
                # Use the specific bootstrap instance's method
                return (
                    self.bootstrap_instance._generate_samples_single_bootstrap(
                        X=X, y=y
                    )
                )
            else:
                raise NotImplementedError(
                    f"The bootstrap class '{type(
                        self.bootstrap_instance).__name__}' does not implement '_generate_samples_single_bootstrap' method."
                )

    def __repr__(self) -> str:
        """
        Return a string representation of the BaseBlockBootstrap instance.
        """
        return f"BaseBlockBootstrap(bootstrap_type='{self.bootstrap_type}', bootstrap_instance={type(self.bootstrap_instance).__name__ if self.bootstrap_instance else None})"


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

    Parameters
    ----------
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    block_length : Integral, default=None
        The length of the blocks to sample.
        If None, the block length is the square root of the number of observations.
    block_length_distribution : str, default=None
        The block length distribution function to use.
        If None, the block length distribution is not utilized.
    wrap_around_flag : bool, default=False
        Whether to wrap around the data when generating blocks.
    overlap_flag : bool, default=False
        Whether to allow blocks to overlap.
    combine_generation_and_sampling_flag : bool, default=False
        Whether to combine the block generation and sampling steps.
    block_weights : array-like of shape (n_blocks,), default=None
        The weights to use when sampling blocks.
    tapered_weights : callable, default=None
        The tapered weights to use when sampling blocks.
    overlap_length : Integral, default=None
        The length of the overlap between blocks.
    min_block_length : Integral, default=None
        The minimum length of the blocks.
    rng : Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

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

    @field_validator(
        "wrap_around_flag",
        "overlap_flag",
        "block_length_distribution",
        mode="before",
    )
    @classmethod
    def set_fixed_values(cls, v, info):
        if info.field_name == "wrap_around_flag":
            return False
        elif info.field_name == "overlap_flag":
            return True
        elif info.field_name == "block_length_distribution":
            return None
        elif info.field_name == "combine_generation_and_sampling_flag":
            return False
        return v


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

    Parameters
    ----------
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    block_length : Integral, default=None
        The length of the blocks to sample.
        If None, the block length is the square root of the number of observations.
    block_length_distribution : str, default=None
        The block length distribution function to use.
        If None, the block length distribution is not utilized.
    wrap_around_flag : bool, default=False
        Whether to wrap around the data when generating blocks.
    overlap_flag : bool, default=False
        Whether to allow blocks to overlap.
    combine_generation_and_sampling_flag : bool, default=False
        Whether to combine the block generation and sampling steps.
    block_weights : array-like of shape (n_blocks,), default=None
        The weights to use when sampling blocks.
    tapered_weights : callable, default=None
        The tapered weights to use when sampling blocks.
    overlap_length : Integral, default=None
        The length of the overlap between blocks.
    min_block_length : Integral, default=None
        The minimum length of the blocks.
    rng : Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

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

    @field_validator(
        "wrap_around_flag",
        "overlap_flag",
        "block_length_distribution",
        mode="before",
    )
    @classmethod
    def set_fixed_values(cls, v, info):
        if info.field_name == "wrap_around_flag":
            return False
        elif info.field_name == "overlap_flag":
            return True
        elif info.field_name == "block_length_distribution":
            return "geometric"
        elif info.field_name == "combine_generation_and_sampling_flag":
            return False
        return v


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
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    block_length : Integral, default=None
        The length of the blocks to sample.
        If None, the block length is the square root of the number of observations.
    block_length_distribution : str, default=None
        The block length distribution function to use.
        If None, the block length distribution is not utilized.
    wrap_around_flag : bool, default=False
        Whether to wrap around the data when generating blocks.
    overlap_flag : bool, default=False
        Whether to allow blocks to overlap.
    combine_generation_and_sampling_flag : bool, default=False
        Whether to combine the block generation and sampling steps.
    block_weights : array-like of shape (n_blocks,), default=None
        The weights to use when sampling blocks.
    tapered_weights : callable, default=None
        The tapered weights to use when sampling blocks.
    overlap_length : Integral, default=None
        The length of the overlap between blocks.
    min_block_length : Integral, default=None
        The minimum length of the blocks.
    rng : Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

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

    @field_validator(
        "wrap_around_flag",
        "overlap_flag",
        "block_length_distribution",
        mode="before",
    )
    @classmethod
    def set_fixed_values(cls, v, info):
        if (
            info.field_name == "wrap_around_flag"
            or info.field_name == "overlap_flag"
        ):
            return True
        elif info.field_name == "block_length_distribution":
            return None
        elif info.field_name == "combine_generation_and_sampling_flag":
            return False
        return v


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
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    block_length : Integral, default=None
        The length of the blocks to sample.
        If None, the block length is the square root of the number of observations.
    block_length_distribution : str, default=None
        The block length distribution function to use.
        If None, the block length distribution is not utilized.
    wrap_around_flag : bool, default=False
        Whether to wrap around the data when generating blocks.
    overlap_flag : bool, default=False
        Whether to allow blocks to overlap.
    combine_generation_and_sampling_flag : bool, default=False
        Whether to combine the block generation and sampling steps.
    block_weights : array-like of shape (n_blocks,), default=None
        The weights to use when sampling blocks.
    tapered_weights : callable, default=None
        The tapered weights to use when sampling blocks.
    overlap_length : Integral, default=None
        The length of the overlap between blocks.
    min_block_length : Integral, default=None
        The minimum length of the blocks.
    rng : Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

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

    @field_validator(
        "wrap_around_flag",
        "overlap_flag",
        "block_length_distribution",
        mode="before",
    )
    @classmethod
    def set_fixed_values(cls, v, info):
        if (
            info.field_name == "wrap_around_flag"
            or info.field_name == "overlap_flag"
        ):
            return False
        elif info.field_name == "block_length_distribution":
            return None
        elif info.field_name == "combine_generation_and_sampling_flag":
            return False
        return v


# Be cautious when using the default windowing functions from numpy, as they drop to 0 at the edges.This could be particularly problematic for smaller block_lengths. In the current implementation, we have clipped the min to 0.1, in block_resampler.py.


class BartlettsBootstrap(BaseBlockBootstrap):
    r"""Bartlett's Bootstrap class for time series data.

    This class is a specialized bootstrapping class that uses
    Bartlett's window for tapered weights.

    Parameters
    ----------
    bootstrap_type : str
        Always set to 'moving' for Bartlett's bootstrap.
    tapered_weights : callable
        Always set to np.bartlett for Bartlett's bootstrap.
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    block_length : Integral, default=None
        The length of the blocks to sample.
        If None, the block length is the square root of the number of observations.
    block_length_distribution : str, default=None
        The block length distribution function to use.
        If None, the block length distribution is not utilized.
    wrap_around_flag : bool, default=False
        Whether to wrap around the data when generating blocks.
    overlap_flag : bool, default=False
        Whether to allow blocks to overlap.
    combine_generation_and_sampling_flag : bool, default=False
        Whether to combine the block generation and sampling steps.
    block_weights : array-like of shape (n_blocks,), default=None
        The weights to use when sampling blocks.
    overlap_length : Integral, default=None
        The length of the overlap between blocks.
    min_block_length : Integral, default=None
        The minimum length of the blocks.
    rng : Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

    Notes
    -----
    The Bartlett window is defined as:

    .. math::
        w(n) = 1 - \\frac{|n - (N - 1) / 2|}{(N - 1) / 2}

    where :math:`N` is the block length.

    References
    ----------
    .. [^1^] https://en.wikipedia.org/wiki/Window_function#Triangular_window
    """

    @field_validator("bootstrap_type", mode="before")
    @classmethod
    def set_bootstrap_type(cls, v: Optional[str]) -> str:
        return "moving"

    @field_validator("tapered_weights", mode="before")
    @classmethod
    def set_tapered_weights(cls, v: Optional[Callable]) -> Callable:
        return np.bartlett

    def __repr__(self) -> str:
        return f"BartlettsBootstrap(bootstrap_type='{self.bootstrap_type}', tapered_weights=np.bartlett)"


class HammingBootstrap(BaseBlockBootstrap):
    r"""
    Hamming Bootstrap class for time series data.

    This class is a specialized bootstrapping class that uses
    Hamming window for tapered weights.

    Parameters
    ----------
    bootstrap_type : str
        Always set to 'moving' for Hamming bootstrap.
    tapered_weights : callable
        Always set to np.hamming for Hamming bootstrap.
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    block_length : Integral, default=None
        The length of the blocks to sample.
        If None, the block length is the square root of the number of observations.
    block_length_distribution : str, default=None
        The block length distribution function to use.
        If None, the block length distribution is not utilized.
    wrap_around_flag : bool, default=False
        Whether to wrap around the data when generating blocks.
    overlap_flag : bool, default=False
        Whether to allow blocks to overlap.
    combine_generation_and_sampling_flag : bool, default=False
        Whether to combine the block generation and sampling steps.
    block_weights : array-like of shape (n_blocks,), default=None
        The weights to use when sampling blocks.
    overlap_length : Integral, default=None
        The length of the overlap between blocks.
    min_block_length : Integral, default=None
        The minimum length of the blocks.
    rng : Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

    Notes
    -----
    The Hamming window is defined as:

    .. math::
        w(n) = 0.54 - 0.46 \\cos\\left(\\frac{2\\pi n}{N - 1}\\right)

    where :math:`N` is the block length.

    References
    ----------
    .. [^1^] https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows
    """

    @field_validator("bootstrap_type", mode="before")
    @classmethod
    def set_bootstrap_type(cls, v: Optional[str]) -> str:
        return "moving"

    @field_validator("tapered_weights", mode="before")
    @classmethod
    def set_tapered_weights(cls, v: Optional[Callable]) -> Callable:
        return np.hamming

    def __repr__(self) -> str:
        return f"HammingBootstrap(bootstrap_type='{self.bootstrap_type}', tapered_weights=np.hamming)"


class HanningBootstrap(BaseBlockBootstrap):
    r"""
    Hanning Bootstrap class for time series data.

    This class is a specialized bootstrapping class that uses
    Hanning window for tapered weights.

    Parameters
    ----------
    bootstrap_type : str
        Always set to 'moving' for Hanning bootstrap.
    tapered_weights : callable
        Always set to np.hamming for Hanning bootstrap.
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    block_length : Integral, default=None
        The length of the blocks to sample.
        If None, the block length is the square root of the number of observations.
    block_length_distribution : str, default=None
        The block length distribution function to use.
        If None, the block length distribution is not utilized.
    wrap_around_flag : bool, default=False
        Whether to wrap around the data when generating blocks.
    overlap_flag : bool, default=False
        Whether to allow blocks to overlap.
    combine_generation_and_sampling_flag : bool, default=False
        Whether to combine the block generation and sampling steps.
    block_weights : array-like of shape (n_blocks,), default=None
        The weights to use when sampling blocks.
    overlap_length : Integral, default=None
        The length of the overlap between blocks.
    min_block_length : Integral, default=None
        The minimum length of the blocks.
    rng : Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

    Notes
    -----
    The Hanning window is defined as:

    .. math::
        w(n) = 0.5 - 0.5 \\cos\\left(\\frac{2\\pi n}{N - 1}\\right)

    where :math:`N` is the block length.

    References
    ----------
    .. [^1^] https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows
    """

    @field_validator("bootstrap_type", mode="before")
    @classmethod
    def set_bootstrap_type(cls, v: Optional[str]) -> str:
        return "moving"

    @field_validator("tapered_weights", mode="before")
    @classmethod
    def set_tapered_weights(cls, v: Optional[Callable]) -> Callable:
        return np.hanning

    def __repr__(self) -> str:
        return f"HanningBootstrap(bootstrap_type='{self.bootstrap_type}', tapered_weights=np.hanning)"


class BlackmanBootstrap(BaseBlockBootstrap):
    r"""
    Blackman Bootstrap class for time series data.

    This class is a specialized bootstrapping class that uses
    Blackman window for tapered weights.

    Parameters
    ----------
    bootstrap_type : str
        Always set to 'moving' for Blackman bootstrap.
    tapered_weights : callable
        Always set to np.blackman for Blackman bootstrap.
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    block_length : Integral, default=None
        The length of the blocks to sample.
        If None, the block length is the square root of the number of observations.
    block_length_distribution : str, default=None
        The block length distribution function to use.
        If None, the block length distribution is not utilized.
    wrap_around_flag : bool, default=False
        Whether to wrap around the data when generating blocks.
    overlap_flag : bool, default=False
        Whether to allow blocks to overlap.
    combine_generation_and_sampling_flag : bool, default=False
        Whether to combine the block generation and sampling steps.
    block_weights : array-like of shape (n_blocks,), default=None
        The weights to use when sampling blocks.
    overlap_length : Integral, default=None
        The length of the overlap between blocks.
    min_block_length : Integral, default=None
        The minimum length of the blocks.
    rng : Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

    Notes
    -----
    The Blackman window is defined as:

    .. math::
        w(n) = 0.42 - 0.5 \\cos\\left(\\frac{2\\pi n}{N - 1}\\right) + 0.08 \\cos\\left(\\frac{4\\pi n}{N - 1}\\right)

    where :math:`N` is the block length.

    References
    ----------
    .. [^1^] https://en.wikipedia.org/wiki/Window_function#Blackman_window
    """

    @field_validator("bootstrap_type", mode="before")
    @classmethod
    def set_bootstrap_type(cls, v: Optional[str]) -> str:
        return "moving"

    @field_validator("tapered_weights", mode="before")
    @classmethod
    def set_tapered_weights(cls, v: Optional[Callable]) -> Callable:
        return np.blackman

    def __repr__(self) -> str:
        return f"BlackmanBootstrap(bootstrap_type='{self.bootstrap_type}', tapered_weights=np.blackman)"


class TukeyBootstrap(BaseBlockBootstrap):
    r"""
    Tukey Bootstrap class for time series data.

    This class is a specialized bootstrapping class that uses
    Tukey window for tapered weights.

    Parameters
    ----------
    bootstrap_type : str
        Always set to 'moving' for Tukey bootstrap.
    tapered_weights : callable
        Always set to scipy.signal.windows.tukey for Tukey bootstrap.
    alpha : float, default=0.5
        Shape parameter of the Tukey window. Must be between 0 and 1 (exclusive).
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    block_length : Integral, default=None
        The length of the blocks to sample.
        If None, the block length is the square root of the number of observations.
    block_length_distribution : str, default=None
        The block length distribution function to use.
        If None, the block length distribution is not utilized.
    wrap_around_flag : bool, default=False
        Whether to wrap around the data when generating blocks.
    overlap_flag : bool, default=False
        Whether to allow blocks to overlap.
    combine_generation_and_sampling_flag : bool, default=False
        Whether to combine the block generation and sampling steps.
    block_weights : array-like of shape (n_blocks,), default=None
        The weights to use when sampling blocks.
    tapered_weights : callable, default=None
        The tapered weights to use when sampling blocks.
    overlap_length : Integral, default=None
        The length of the overlap between blocks.
    min_block_length : Integral, default=None
        The minimum length of the blocks.
    rng : Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

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

    References
    ----------
    .. [^1^] https://en.wikipedia.org/wiki/Window_function#Tukey_window
    """

    alpha: float = Field(default=0.5, gt=0, lt=1)

    @field_validator("bootstrap_type", mode="before")
    @classmethod
    def set_bootstrap_type(cls, v: Optional[str]) -> str:
        return "moving"

    @field_validator("tapered_weights", mode="before")
    @classmethod
    def set_tapered_weights(
        cls, v: Optional[Callable], info: ValidationInfo
    ) -> Callable:
        alpha = info.data.get("alpha", 0.5)
        return partial(tukey, alpha=alpha)

    def __repr__(self) -> str:
        return f"TukeyBootstrap(bootstrap_type='{self.bootstrap_type}', tapered_weights=scipy.signal.windows.tukey)"


BLOCK_BOOTSTRAP_TYPES_DICT = {
    "nonoverlapping": NonOverlappingBlockBootstrap,
    "moving": MovingBlockBootstrap,
    "stationary": StationaryBlockBootstrap,
    "circular": CircularBlockBootstrap,
}


def get_bootstrap_types_dict():
    return BLOCK_BOOTSTRAP_TYPES_DICT
