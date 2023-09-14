from __future__ import annotations

from collections.abc import Callable
from functools import partial
from numbers import Integral

import numpy as np
from scipy.signal.windows import tukey

from tsbootstrap.base_bootstrap_configs import BaseTimeSeriesBootstrapConfig
from tsbootstrap.block_bootstrap import BLOCK_BOOTSTRAP_TYPES_DICT
from tsbootstrap.utils.validate import validate_single_integer


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
        n_bootstraps: Integral = 10,  # type: ignore
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
        if value is not None:
            validate_single_integer(value, min_value=1)
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
        if value is not None:
            validate_single_integer(value, min_value=1)
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
        if value is not None:
            validate_single_integer(value, min_value=1)
        self._min_block_length = value

    def __repr__(self) -> str:
        """Return repr(self)."""
        base_repr = super().__repr__()
        return (
            f"{base_repr[:-1]}, "
            f"block_length={self.block_length}, "
            f"block_length_distribution={self.block_length_distribution}, "
            f"wrap_around_flag={self.wrap_around_flag}, "
            f"overlap_flag={self.overlap_flag}, "
            f"combine_generation_and_sampling_flag={self.combine_generation_and_sampling_flag}, "
            f"block_weights={self.block_weights}, "
            f"tapered_weights={self.tapered_weights}, "
            f"overlap_length={self.overlap_length}, "
            f"min_block_length={self.min_block_length})"
        )

    def __str__(self) -> str:
        """Return str(self)."""
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        """Return self == other."""
        if not isinstance(other, BlockBootstrapConfig):
            return False
        return (
            super().__eq__(other)
            and self.block_length == other.block_length
            and self.block_length_distribution
            == other.block_length_distribution
            and self.wrap_around_flag == other.wrap_around_flag
            and self.overlap_flag == other.overlap_flag
            and self.combine_generation_and_sampling_flag
            == other.combine_generation_and_sampling_flag
            and self.block_weights == other.block_weights
            and self.tapered_weights == other.tapered_weights
            and self.overlap_length == other.overlap_length
            and self.min_block_length == other.min_block_length
        )


class BaseBlockBootstrapConfig(BlockBootstrapConfig):
    """
    Configuration class for BaseBlockBootstrap.

    This class is a specialized configuration class that allows for the
    `bootstrap_type` parameter to be set. The `bootstrap_type` parameter
    determines the type of block bootstrap to use.
    """

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
        valid_types = set(BLOCK_BOOTSTRAP_TYPES_DICT.keys())

        if value is not None and value not in valid_types:
            raise ValueError(f"bootstrap_type must be one of {valid_types}.")
        self._bootstrap_type = value

    def __repr__(self) -> str:
        """Return repr(self)."""
        base_repr = super().__repr__()
        return f"{base_repr[:-1]}, bootstrap_type={self.bootstrap_type})"

    def __str__(self) -> str:
        """Return str(self)."""
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        """Return self == other."""
        if not isinstance(other, BaseBlockBootstrapConfig):
            return False
        return (
            super().__eq__(other)
            and self.bootstrap_type == other.bootstrap_type
        )


class MovingBlockBootstrapConfig(BlockBootstrapConfig):
    """
    Configuration class for MovingBlockBootstrap.

    This class is a specialized configuration class that sets
    `wrap_around_flag` to False, `overlap_flag` to True, and
    `block_length_distribution` to None.
    """

    def __init__(
        self,
        block_length: Integral,
        **kwargs,
    ) -> None:
        """
        Initialize self.

        Parameters
        ----------
        block_length : Integral, default=None
            The length of the blocks to sample.
        kwargs
            Additional keyword arguments, except for wrap_around_flag, overlap_flag, and block_length_distribution, to pass to the parent BlockBootstrapConfig class.
            See BlockBootstrapConfig for more information.
        """
        # Prevent these properties from being passed to the parent class
        if "wrap_around_flag" in kwargs:
            del kwargs["wrap_around_flag"]
        if "overlap_flag" in kwargs:
            del kwargs["overlap_flag"]
        if "block_length_distribution" in kwargs:
            del kwargs["block_length_distribution"]
        # Initialize the parent class
        super().__init__(block_length=block_length, **kwargs)
        # Set the properties directly
        self._wrap_around_flag = False
        self._overlap_flag = True
        self._block_length_distribution = None

    def __repr__(self) -> str:
        """Return repr(self)."""
        base_repr = super().__repr__()
        return (
            f"{base_repr[:-1]}, "
            f"block_length={self.block_length}, "
            f"wrap_around_flag={self.wrap_around_flag}, "
            f"overlap_flag={self.overlap_flag}, "
            f"block_length_distribution={self.block_length_distribution})"
        )

    def __str__(self) -> str:
        """Return str(self)."""
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        """Return self == other."""
        if not isinstance(other, MovingBlockBootstrapConfig):
            return False
        return (
            super().__eq__(other)
            and self.block_length == other.block_length
            and self.wrap_around_flag == other.wrap_around_flag
            and self.overlap_flag == other.overlap_flag
            and self.block_length_distribution
            == other.block_length_distribution
        )


class StationaryBlockBootstrapConfig(BlockBootstrapConfig):
    """
    Configuration class for StationaryBlockBootstrap.

    This class is a specialized configuration class that sets
    `wrap_around_flag` to False, `overlap_flag` to True, and
    `block_length_distribution` to "geometric".
    """

    def __init__(
        self,
        block_length: Integral,
        **kwargs,
    ) -> None:
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
        # Prevent these properties from being passed to the parent class
        if "wrap_around_flag" in kwargs:
            del kwargs["wrap_around_flag"]
        if "overlap_flag" in kwargs:
            del kwargs["overlap_flag"]
        if "block_length_distribution" in kwargs:
            del kwargs["block_length_distribution"]
        # Initialize the parent class
        super().__init__(block_length=block_length, **kwargs)
        # Set the properties directly
        self._wrap_around_flag = False
        self._overlap_flag = True
        self._block_length_distribution = "geometric"

    def __repr__(self) -> str:
        """Return repr(self)."""
        base_repr = super().__repr__()
        return (
            f"{base_repr[:-1]}, "
            f"block_length={self.block_length}, "
            f"wrap_around_flag={self.wrap_around_flag}, "
            f"overlap_flag={self.overlap_flag}, "
            f"block_length_distribution={self.block_length_distribution})"
        )

    def __str__(self) -> str:
        """Return str(self)."""
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        """Return self == other."""
        if not isinstance(other, StationaryBlockBootstrapConfig):
            return False
        return (
            super().__eq__(other)
            and self.block_length == other.block_length
            and self.wrap_around_flag == other.wrap_around_flag
            and self.overlap_flag == other.overlap_flag
            and self.block_length_distribution
            == other.block_length_distribution
        )


class CircularBlockBootstrapConfig(BlockBootstrapConfig):
    """
    Configuration class for CircularBlockBootstrap.

    This class is a specialized configuration class that sets
    `wrap_around_flag` to True, `overlap_flag` to True, and
    `block_length_distribution` to None.
    """

    def __init__(
        self,
        block_length: Integral,
        **kwargs,
    ) -> None:
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
        # Prevent these properties from being passed to the parent class
        if "wrap_around_flag" in kwargs:
            del kwargs["wrap_around_flag"]
        if "overlap_flag" in kwargs:
            del kwargs["overlap_flag"]
        if "block_length_distribution" in kwargs:
            del kwargs["block_length_distribution"]
        # Initialize the parent class
        super().__init__(block_length=block_length, **kwargs)
        # Set the properties directly
        self._wrap_around_flag = True
        self._overlap_flag = True
        self._block_length_distribution = None

    def __repr__(self) -> str:
        """Return repr(self)."""
        base_repr = super().__repr__()
        return (
            f"{base_repr[:-1]}, "
            f"block_length={self.block_length}, "
            f"wrap_around_flag={self.wrap_around_flag}, "
            f"overlap_flag={self.overlap_flag}, "
            f"block_length_distribution={self.block_length_distribution})"
        )

    def __str__(self) -> str:
        """Return str(self)."""
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        """Return self == other."""
        if not isinstance(other, CircularBlockBootstrapConfig):
            return False
        return (
            super().__eq__(other)
            and self.block_length == other.block_length
            and self.wrap_around_flag == other.wrap_around_flag
            and self.overlap_flag == other.overlap_flag
            and self.block_length_distribution
            == other.block_length_distribution
        )


class NonOverlappingBlockBootstrapConfig(BlockBootstrapConfig):
    """
    Configuration class for NonOverlappingBlockBootstrap.

    This class is a specialized configuration class that sets
    `wrap_around_flag` to False, `overlap_flag` to False, and
    `block_length_distribution` to None.
    """

    def __init__(
        self,
        block_length: Integral,
        **kwargs,
    ) -> None:
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
        # Prevent these properties from being passed to the parent class
        if "wrap_around_flag" in kwargs:
            del kwargs["wrap_around_flag"]
        if "overlap_flag" in kwargs:
            del kwargs["overlap_flag"]
        if "block_length_distribution" in kwargs:
            del kwargs["block_length_distribution"]
        # Initialize the parent class
        super().__init__(block_length=block_length, **kwargs)
        # Set the properties directly
        self._wrap_around_flag = False
        self._overlap_flag = False
        self._block_length_distribution = None

    def __repr__(self) -> str:
        """Return repr(self)."""
        base_repr = super().__repr__()
        return (
            f"{base_repr[:-1]}, "
            f"block_length={self.block_length}, "
            f"wrap_around_flag={self.wrap_around_flag}, "
            f"overlap_flag={self.overlap_flag}, "
            f"block_length_distribution={self.block_length_distribution})"
        )

    def __str__(self) -> str:
        """Return str(self)."""
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        """Return self == other."""
        if not isinstance(other, NonOverlappingBlockBootstrapConfig):
            return False
        return (
            super().__eq__(other)
            and self.block_length == other.block_length
            and self.wrap_around_flag == other.wrap_around_flag
            and self.overlap_flag == other.overlap_flag
            and self.block_length_distribution
            == other.block_length_distribution
        )


class BartlettsBootstrapConfig(BaseBlockBootstrapConfig):
    """Config class for BartlettBootstrap.

    This class is a specialized configuration class that sets
    `tapered_weights` to Bartlett window and `bootstrap_type` to "moving".
    """

    def __init__(
        self,
        block_length: None | Integral = None,
        **kwargs,
    ) -> None:
        """Initialize BartlettsBootstrapConfig.

        Parameters
        ----------
        block_length : Optional[Integral], default=None
            The length of the blocks to sample.
        kwargs
            Additional keyword arguments to pass to the parent BaseBlockBootstrapConfig class.
            See the documentation for BaseBlockBootstrapConfig for more information.
        """
        # Prevent these properties from being passed to the parent class
        if "bootstrap_type" in kwargs:
            del kwargs["bootstrap_type"]
        if "tapered_weights" in kwargs:
            del kwargs["tapered_weights"]

        # Initialize the parent class
        super().__init__(block_length=block_length, **kwargs)

        # Set the properties directly
        self._bootstrap_type = "moving"
        self._tapered_weights = np.bartlett

    def __repr__(self) -> str:
        """Return repr(self)."""
        base_repr = super().__repr__()
        return (
            f"{base_repr[:-1]}, "
            f"bootstrap_type={self.bootstrap_type}, "
            f"tapered_weights={self.tapered_weights})"
        )

    def __str__(self) -> str:
        """Return str(self)."""
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        """Return self == other."""
        if not isinstance(other, BartlettsBootstrapConfig):
            return False
        return (
            super().__eq__(other)
            and self.bootstrap_type == other.bootstrap_type
            and self.tapered_weights == other.tapered_weights
        )


class HammingBootstrapConfig(BaseBlockBootstrapConfig):
    """Config class for HammingBootstrap.

    This class is a specialized configuration class that sets
    `tapered_weights` to Hamming window and `bootstrap_type` to "moving".
    """

    def __init__(
        self,
        block_length: None | Integral = None,
        **kwargs,
    ) -> None:
        """Initialize HammingBootstrapConfig.

        Parameters
        ----------
        block_length : Optional[Integral], default=None
            The length of the blocks to sample.
        kwargs
            Additional keyword arguments to pass to the parent BaseBlockBootstrapConfig class.
            See the documentation for BaseBlockBootstrapConfig for more information.
        """
        # Prevent these properties from being passed to the parent class
        if "bootstrap_type" in kwargs:
            del kwargs["bootstrap_type"]
        if "tapered_weights" in kwargs:
            del kwargs["tapered_weights"]

        # Initialize the parent class
        super().__init__(block_length=block_length, **kwargs)

        # Set the properties directly
        self._bootstrap_type = "moving"
        self._tapered_weights = np.hamming

    def __repr__(self) -> str:
        """Return repr(self)."""
        base_repr = super().__repr__()
        return (
            f"{base_repr[:-1]}, "
            f"bootstrap_type={self.bootstrap_type}, "
            f"tapered_weights={self.tapered_weights})"
        )

    def __str__(self) -> str:
        """Return str(self)."""
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        """Return self == other."""
        if not isinstance(other, HammingBootstrapConfig):
            return False
        return (
            super().__eq__(other)
            and self.bootstrap_type == other.bootstrap_type
            and self.tapered_weights == other.tapered_weights
        )


class HanningBootstrapConfig(BaseBlockBootstrapConfig):
    """Config class for HanningBootstrap.

    This class is a specialized configuration class that sets
    `tapered_weights` to Hanning window and `bootstrap_type` to "moving".
    """

    def __init__(
        self,
        block_length: None | Integral = None,
        **kwargs,
    ) -> None:
        """Initialize HanningBootstrapConfig.

        Parameters
        ----------
        block_length : Optional[Integral], default=None
            The length of the blocks to sample.
        kwargs
            Additional keyword arguments to pass to the parent BaseBlockBootstrapConfig class.
            See the documentation for BaseBlockBootstrapConfig for more information.
        """
        # Prevent these properties from being passed to the parent class
        if "bootstrap_type" in kwargs:
            del kwargs["bootstrap_type"]
        if "tapered_weights" in kwargs:
            del kwargs["tapered_weights"]

        # Initialize the parent class
        super().__init__(block_length=block_length, **kwargs)

        # Set the properties directly
        self._bootstrap_type = "moving"
        self._tapered_weights = np.hanning

    def __repr__(self) -> str:
        """Return repr(self)."""
        base_repr = super().__repr__()
        return (
            f"{base_repr[:-1]}, "
            f"bootstrap_type={self.bootstrap_type}, "
            f"tapered_weights={self.tapered_weights})"
        )

    def __str__(self) -> str:
        """Return str(self)."""
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        """Return self == other."""
        if not isinstance(other, HanningBootstrapConfig):
            return False
        return (
            super().__eq__(other)
            and self.bootstrap_type == other.bootstrap_type
            and self.tapered_weights == other.tapered_weights
        )


class BlackmanBootstrapConfig(BaseBlockBootstrapConfig):
    """Config class for BlackmanBootstrap.

    This class is a specialized configuration class that sets
    `tapered_weights` to Blackman window and `bootstrap_type` to "moving".
    """

    def __init__(
        self,
        block_length: None | Integral = None,
        **kwargs,
    ) -> None:
        """Initialize BlackmanBootstrapConfig.

        Parameters
        ----------
        block_length : Optional[Integral], default=None
            The length of the blocks to sample.
        kwargs
            Additional keyword arguments to pass to the parent BaseBlockBootstrapConfig class.
            See the documentation for BaseBlockBootstrapConfig for more information.
        """
        # Prevent these properties from being passed to the parent class
        if "bootstrap_type" in kwargs:
            del kwargs["bootstrap_type"]
        if "tapered_weights" in kwargs:
            del kwargs["tapered_weights"]

        # Initialize the parent class
        super().__init__(block_length=block_length, **kwargs)

        # Set the properties directly
        self._bootstrap_type = "moving"
        self._tapered_weights = np.blackman

    def __repr__(self) -> str:
        """Return repr(self)."""
        base_repr = super().__repr__()
        return (
            f"{base_repr[:-1]}, "
            f"bootstrap_type={self.bootstrap_type}, "
            f"tapered_weights={self.tapered_weights})"
        )

    def __str__(self) -> str:
        """Return str(self)."""
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        """Return self == other."""
        if not isinstance(other, BlackmanBootstrapConfig):
            return False
        return (
            super().__eq__(other)
            and self.bootstrap_type == other.bootstrap_type
            and self.tapered_weights == other.tapered_weights
        )


class TukeyBootstrapConfig(BaseBlockBootstrapConfig):
    """Config class for TukeyBootstrap.

    This class is a specialized configuration class that sets
    `tapered_weights` to Tukey window and `bootstrap_type` to "moving".
    """

    def __init__(
        self,
        block_length: None | Integral = None,
        **kwargs,
    ) -> None:
        """Initialize TukeyBootstrapConfig.

        Parameters
        ----------
        block_length : Optional[Integral], default=None
            The length of the blocks to sample.
        kwargs
            Additional keyword arguments to pass to the parent BaseBlockBootstrapConfig class.
            See the documentation for BaseBlockBootstrapConfig for more information.
        """
        # Prevent these properties from being passed to the parent class
        if "bootstrap_type" in kwargs:
            del kwargs["bootstrap_type"]
        if "tapered_weights" in kwargs:
            del kwargs["tapered_weights"]

        # Initialize the parent class
        super().__init__(block_length=block_length, **kwargs)

        # Set the properties directly
        alpha = kwargs.get("alpha", 0.5)
        self._bootstrap_type = "moving"
        self._tapered_weights = partial(tukey, alpha=alpha)

    def __repr__(self) -> str:
        """Return repr(self)."""
        base_repr = super().__repr__()
        return (
            f"{base_repr[:-1]}, "
            f"bootstrap_type={self.bootstrap_type}, "
            f"tapered_weights={self.tapered_weights})"
        )

    def __str__(self) -> str:
        """Return str(self)."""
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        """Return self == other."""
        if not isinstance(other, TukeyBootstrapConfig):
            return False
        return (
            super().__eq__(other)
            and self.bootstrap_type == other.bootstrap_type
            and self.tapered_weights == other.tapered_weights
        )
