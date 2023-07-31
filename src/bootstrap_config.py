from __future__ import annotations
import dataclasses
from functools import partial
from scipy.signal.windows import tukey
from typing import Callable, List, Optional, Tuple, Union, Iterator, Any, Dict, Type
import numpy as np
from numpy.random import Generator
from numbers import Integral
from utils.validate import validate_rng, validate_integers
from dataclasses import dataclass
from utils.types import RngTypes


@dataclass
class BaseTimeSeriesBootstrapConfig:
    _n_bootstraps: Integral = 10
    _rng: Optional[Union[Integral, Generator]] = None

    @property
    def rng(self) -> Generator:
        return self._rng

    @rng.setter
    def rng(self, value: RngTypes) -> None:
        self._rng = validate_rng(value)

    @property
    def n_bootstraps(self) -> Integral:
        return self._n_bootstraps

    @n_bootstraps.setter
    def n_bootstraps(self, value) -> None:
        validate_integers(value, positive=True)
        self._n_bootstraps = value


@dataclass
class BlockBootstrapConfig(BaseTimeSeriesBootstrapConfig):
    _block_length: Optional[Integral] = None
    _block_length_distribution: Optional[str] = None
    _wrap_around_flag: bool = False
    _overlap_flag: bool = False
    _combine_generation_and_sampling_flag: bool = False
    _overlap_length: Optional[Integral] = None
    _min_block_length: Optional[Integral] = None
    _block_weights: Optional[Union[np.ndarray, Callable]] = None
    _tapered_weights: Optional[Callable] = None

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
    rng : int or Generator, default=None
    """

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
            if not isinstance(value, Integral) or value < 1:
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
        """
        if not isinstance(value, bool):
            raise ValueError("wrap_around_flag must be a boolean.")
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
        """
        if not isinstance(value, bool):
            raise ValueError("overlap_flag must be a boolean.")
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
        """
        if not isinstance(value, bool):
            raise ValueError(
                "combine_generation_and_sampling_flag must be a boolean.")
        self._combine_generation_and_sampling_flag = value

    @property
    def overlap_length(self) -> Optional[Integral]:
        """Getter for overlap_length."""
        return self._overlap_length

    @overlap_length.setter
    def overlap_length(self, value) -> None:
        """
        Setter for overlap_length. Performs validation on assignment.
        Parameters
        ----------
        value : int or None
        """
        if value is not None:
            validate_integers(value, positive=True)
        self._overlap_length = value

    @property
    def min_block_length(self) -> Optional[Integral]:
        """Getter for min_block_length."""
        return self._min_block_length

    @min_block_length.setter
    def min_block_length(self, value) -> None:
        """
        Setter for min_block_length. Performs validation on assignment.
        Parameters
        ----------
        value : int or None
        """
        if value is not None:
            validate_integers(value, positive=True)
        self._min_block_length = value

    @property
    def block_weights(self) -> Optional[Union[np.ndarray, Callable]]:
        """Getter for block_weights."""
        return self._block_weights

    @block_weights.setter
    def block_weights(self, value) -> None:
        """
        Setter for block_weights. Performs validation on assignment.
        Parameters
        ----------
        value : array-like of shape (X.shape[0],), or Callable, or None
        """
        if value is not None:
            if not isinstance(value, (np.ndarray, Callable)):
                raise ValueError(
                    "block_weights must be an array-like of shape (X.shape[0],), or a Callable, or None.")
        self._block_weights = value

    @property
    def tapered_weights(self) -> Optional[Callable]:
        """Getter for tapered_weights."""
        return self._tapered_weights

    @tapered_weights.setter
    def tapered_weights(self, value) -> None:
        """
        Setter for tapered_weights. Performs validation on assignment.
        Parameters
        ----------
        value : array-like of shape (n_blocks,), or Callable, or None
        """
        if value is not None:
            if not isinstance(value, Callable):
                raise ValueError(
                    "tapered_weights must be a Callable, or None.")
        self._tapered_weights = value


@dataclass
class MovingBlockBootstrapConfig(BlockBootstrapConfig):
    """Configuration class for MovingBlockBootstrap."""

    def __init__(self) -> None:
        super().__init__()
        self._overlap_flag = True
        self._wrap_around_flag = False
        self._block_length_distribution = None
        self._combine_generation_and_sampling_flag = False

        """
        This class functions similarly to the base `BlockBootstrapConfig` class, with
        the following modifications to the default behavior:
        * `overlap_flag` is always set to True, meaning that blocks can overlap.
        * `wrap_around_flag` is always set to False, meaning that the data will not
        wrap around when generating blocks.
        * `block_length_distribution` is always None, meaning that the block length
        distribution is not utilized.
        * `combine_generation_and_sampling_flag` is always False, meaning that the block
        generation and resampling are performed separately.
        """


@dataclass
class StationaryBlockBootstrapConfig(BlockBootstrapConfig):
    """Configuration class for MovingBlockBootstrap."""

    def __init__(self) -> None:
        super().__init__()
        self.overlap_flag = True
        self.wrap_around_flag = False
        self.block_length_distribution = "geometric"
        self._combine_generation_and_sampling_flag = False

        """
        This class functions similarly to the base `BlockBootstrap` class, with
        the following modifications to the default behavior:
        * `overlap_flag` is always set to True, meaning that blocks can overlap.
        * `wrap_around_flag` is always set to False, meaning that the data will not
            wrap around when generating blocks.
        * `block_length_distribution` is always "geometric", meaning that the block
            length distribution is geometrically distributed.
        * `combine_generation_and_sampling_flag` is always False, meaning that the block
            generation and resampling are performed separately.
        """


@dataclass
class CircularBlockBootstrapConfig(BlockBootstrapConfig):
    """Configuration class for MovingBlockBootstrap."""

    def __init__(self) -> None:
        super().__init__()
        self.overlap_flag = True
        self.wrap_around_flag = True
        self.block_length_distribution = None
        self._combine_generation_and_sampling_flag = False

        """
        This class functions similarly to the base `BlockBootstrapConfig` class, with
        the following modifications to the default behavior:
        * `overlap_flag` is always set to True, meaning that blocks can overlap.
        * `wrap_around_flag` is always set to False, meaning that the data will not
        wrap around when generating blocks.
        * `block_length_distribution` is always None, meaning that the block length
        distribution is not utilized.
        * `combine_generation_and_sampling_flag` is always False, meaning that the block
        generation and resampling are performed separately.
        """


@dataclass
class NonOverlappingBlockBootstrapConfig(BlockBootstrapConfig):
    """Configuration class for MovingBlockBootstrap."""

    def __init__(self) -> None:
        super().__init__()
        self.overlap_flag = False
        self.wrap_around_flag = False
        self.block_length_distribution = None
        self._combine_generation_and_sampling_flag = False

        """
        This class functions similarly to the base `BlockBootstrapConfig` class, with
        the following modifications to the default behavior:
        * `overlap_flag` is always set to True, meaning that blocks can overlap.
        * `wrap_around_flag` is always set to False, meaning that the data will not
        wrap around when generating blocks.
        * `block_length_distribution` is always None, meaning that the block length
        distribution is not utilized.
        * `combine_generation_and_sampling_flag` is always False, meaning that the block
        generation and resampling are performed separately.
        """


@dataclass
class BaseBlockBootstrapConfig(BlockBootstrapConfig):
    from src.bootstrap import BlockBootstrap
    _bootstrap_type: Optional[str] = None
    _bootstrap_type_dict: Dict[str, (Type[BlockBootstrapConfig], Type[BlockBootstrap])] = dataclasses.field(
        init=False)

    def __init__(self, _bootstrap_type: Optional[str] = None):
        from src.bootstrap import NonOverlappingBlockBootstrap, MovingBlockBootstrap, StationaryBlockBootstrap, CircularBlockBootstrap
        self._bootstrap_type = _bootstrap_type
        self._bootstrap_type_dict = {
            'nonoverlapping': (NonOverlappingBlockBootstrapConfig, NonOverlappingBlockBootstrap),
            'moving': (MovingBlockBootstrapConfig, MovingBlockBootstrap),
            'stationary': (StationaryBlockBootstrapConfig, StationaryBlockBootstrap),
            'circular': (CircularBlockBootstrapConfig, CircularBlockBootstrap)
        }

    @property
    def bootstrap_type(self) -> Optional[str]:
        return self._bootstrap_type

    @bootstrap_type.setter
    def bootstrap_type(self, value: Optional[str]) -> None:
        if value is not None and not isinstance(value, str):
            raise ValueError("bootstrap_type must be a string.")

        if value not in self._bootstrap_type_dict:
            raise ValueError(
                "bootstrap_type should be one of {}".format(list(self._bootstrap_type_dict.keys())))

        self._bootstrap_type = value


@dataclass
class BartlettBootstrapConfig(BaseBlockBootstrapConfig):
    def __init__(self) -> None:
        super().__init__()
        self.tapered_weights = np.bartlett
        self.bootstrap_type = "moving"


@dataclass
class HammingBootstrapConfig(BaseBlockBootstrapConfig):
    def __init__(self) -> None:
        super().__init__()
        self.tapered_weights = np.hamming
        self.bootstrap_type = "moving"


@dataclass
class HanningBootstrapConfig(BaseBlockBootstrapConfig):
    def __init__(self) -> None:
        super().__init__()
        self.tapered_weights = np.hanning
        self.bootstrap_type = "moving"


@dataclass
class BlackmanBootstrapConfig(BaseBlockBootstrapConfig):
    def __init__(self) -> None:
        super().__init__()
        self.tapered_weights = np.blackman
        self.bootstrap_type = "moving"


@dataclass
class TukeyBootstrapConfig(BaseBlockBootstrapConfig):

    def __init__(self) -> None:
        super().__init__()
        self.tapered_weights = partial(tukey, alpha=0.5)
        self.bootstrap_type = 'moving'
