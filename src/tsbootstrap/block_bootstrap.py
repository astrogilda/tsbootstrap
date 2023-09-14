from __future__ import annotations

from email.mime import base
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tsbootstrap.block_bootstrap_configs import (
        BartlettsBootstrapConfig,
        BaseBlockBootstrapConfig,
        BlackmanBootstrapConfig,
        BlockBootstrapConfig,
        CircularBlockBootstrapConfig,
        HammingBootstrapConfig,
        HanningBootstrapConfig,
        MovingBlockBootstrapConfig,
        NonOverlappingBlockBootstrapConfig,
        StationaryBlockBootstrapConfig,
        TukeyBootstrapConfig,
    )


from tsbootstrap.base_bootstrap import BaseTimeSeriesBootstrap
from tsbootstrap.block_generator import BlockGenerator
from tsbootstrap.block_length_sampler import BlockLengthSampler
from tsbootstrap.block_resampler import BlockResampler


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

    def __repr__(self) -> str:
        base_repr = super().__repr__()
        return f"{base_repr}\n{self.config}"

    def __str__(self) -> str:
        base_str = super().__str__()
        return f"{base_str}\n{self.config}"

    def __eq__(self, obj: object) -> bool:
        if not isinstance(obj, BlockBootstrap):
            return False
        return super().__eq__(obj) and self.config == obj.config

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.config))


class BaseBlockBootstrap(BlockBootstrap):
    """
    Base class for block bootstrapping.
    """

    def __init__(self, config: BaseBlockBootstrapConfig, **kwargs):
        super().__init__(config=config, **kwargs)
        self.bootstrap_instance: BlockBootstrap | None = None

        if config.bootstrap_type:
            self.bootstrap_instance = BLOCK_BOOTSTRAP_TYPES_DICT[
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

    def __repr__(self) -> str:
        base_repr = super().__repr__()
        return f"{base_repr}\n{self.config}"

    def __str__(self) -> str:
        base_str = super().__str__()
        return f"{base_str}\n{self.config}"

    def __eq__(self, obj: object) -> bool:
        if not isinstance(obj, BaseBlockBootstrap):
            return False
        return super().__eq__(obj) and self.config == obj.config

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.config))


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

    def __repr__(self) -> str:
        return super().__repr__() + "\n" + self.config.__repr__()

    def __str__(self) -> str:
        return super().__str__() + "\n" + self.config.__str__()

    def __eq__(self, obj: object) -> bool:
        if not isinstance(obj, MovingBlockBootstrap):
            return False
        return super().__eq__(obj) and self.config == obj.config

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.config))

    """
    def __getstate__(self) -> dict[str, object]:
        return {
            "config": self.config,
            "blocks": self.blocks,
            "block_resampler": self.block_resampler,
        }
    """


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

    def __repr__(self) -> str:
        return super().__repr__() + "\n" + self.config.__repr__()

    def __str__(self) -> str:
        return super().__str__() + "\n" + self.config.__str__()

    def __eq__(self, obj: object) -> bool:
        if not isinstance(obj, StationaryBlockBootstrap):
            return False
        return super().__eq__(obj) and self.config == obj.config

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.config))

    """
    def __getstate__(self) -> dict[str, object]:
        return {
            "config": self.config,
            "blocks": self.blocks,
            "block_resampler": self.block_resampler,
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        self.config = state["config"]
        self.blocks = state["blocks"]
        self.block_resampler = state["block_resampler"]
    """


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

    def __repr__(self) -> str:
        return super().__repr__() + "\n" + self.config.__repr__()

    def __str__(self) -> str:
        return super().__str__() + "\n" + self.config.__str__()

    def __eq__(self, obj: object) -> bool:
        if not isinstance(obj, CircularBlockBootstrap):
            return False
        return super().__eq__(obj) and self.config == obj.config

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.config))

    """
    def __getstate__(self) -> dict[str, object]:
        return {
            "config": self.config,
            "blocks": self.blocks,
            "block_resampler": self.block_resampler,
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        self.config = state["config"]
        self.blocks = state["blocks"]
        self.block_resampler = state["block_resampler"]
    """


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

    def __repr__(self) -> str:
        return super().__repr__() + "\n" + self.config.__repr__()

    def __str__(self) -> str:
        return super().__str__() + "\n" + self.config.__str__()

    def __eq__(self, obj: object) -> bool:
        if not isinstance(obj, NonOverlappingBlockBootstrap):
            return False
        return super().__eq__(obj) and self.config == obj.config

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.config))

    """
    def __getstate__(self) -> dict[str, object]:
        return {
            "config": self.config,
            "blocks": self.blocks,
            "block_resampler": self.block_resampler,
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        self.config = state["config"]
        self.blocks = state["blocks"]
        self.block_resampler = state["block_resampler"]
    """


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
        super().__init__(config=config)

    def __repr__(self) -> str:
        return super().__repr__() + "\n" + self.config.__repr__()

    def __str__(self) -> str:
        return super().__str__() + "\n" + self.config.__str__()

    def __eq__(self, obj: object) -> bool:
        if not isinstance(obj, BartlettsBootstrap):
            return False
        return super().__eq__(obj) and self.config == obj.config

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.config))

    """
    def __getstate__(self) -> dict[str, object]:
        return {
            "config": self.config,
            "bootstrap_instance": self.bootstrap_instance,
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        self.config = state["config"]
        self.bootstrap_instance = state["bootstrap_instance"]
    """


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
        super().__init__(config=config)

    def __repr__(self) -> str:
        return super().__repr__() + "\n" + self.config.__repr__()

    def __str__(self) -> str:
        return super().__str__() + "\n" + self.config.__str__()

    def __eq__(self, obj: object) -> bool:
        if not isinstance(obj, HammingBootstrap):
            return False
        return super().__eq__(obj) and self.config == obj.config

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.config))

    """
    def __getstate__(self) -> dict[str, object]:
        return {
            "config": self.config,
            "bootstrap_instance": self.bootstrap_instance,
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        self.config = state["config"]
        self.bootstrap_instance = state["bootstrap_instance"]
    """


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
        super().__init__(config=config)

    def __repr__(self) -> str:
        return super().__repr__() + "\n" + self.config.__repr__()

    def __str__(self) -> str:
        return super().__str__() + "\n" + self.config.__str__()

    def __eq__(self, obj: object) -> bool:
        if not isinstance(obj, HanningBootstrap):
            return False
        return super().__eq__(obj) and self.config == obj.config

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.config))

    """
    def __getstate__(self) -> dict[str, object]:
        return {
            "config": self.config,
            "bootstrap_instance": self.bootstrap_instance,
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        if ["config", "bootstrap_instance"] != state.keys():
            raise ValueError(
                "The state dictionary must contain the keys 'config' and 'bootstrap_instance'."
            )
        self.config = state["config"]
        self.bootstrap_instance = state["bootstrap_instance"]
    """


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
        super().__init__(config=config)

    def __repr__(self) -> str:
        return super().__repr__() + "\n" + self.config.__repr__()

    def __str__(self) -> str:
        return super().__str__() + "\n" + self.config.__str__()

    def __eq__(self, obj: object) -> bool:
        if not isinstance(obj, BlackmanBootstrap):
            return False
        return super().__eq__(obj) and self.config == obj.config

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.config))

    """
    def __getstate__(self) -> dict[str, object]:
        return {
            "config": self.config,
            "bootstrap_instance": self.bootstrap_instance,
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        if ["config", "bootstrap_instance"] != state.keys():
            raise ValueError(
                "The state dictionary must contain the keys 'config' and 'bootstrap_instance'."
            )
        self.config = state["config"]
        self.bootstrap_instance = state["bootstrap_instance"]
    """


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
        super().__init__(config=config)

    def __repr__(self) -> str:
        return super().__repr__() + "\n" + self.config.__repr__()

    def __str__(self) -> str:
        return super().__str__() + "\n" + self.config.__str__()

    def __eq__(self, obj: object) -> bool:
        if not isinstance(obj, TukeyBootstrap):
            return False
        return super().__eq__(obj) and self.config == obj.config

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.config))

    """
    def __getstate__(self) -> dict[str, object]:
        return {
            "config": self.config,
            "bootstrap_instance": self.bootstrap_instance,
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        if ["config", "bootstrap_instance"] != state.keys():
            raise ValueError(
                "The state dictionary must contain the keys 'config' and 'bootstrap_instance'."
            )
        self.config = state["config"]
        self.bootstrap_instance = state["bootstrap_instance"]
    """


BLOCK_BOOTSTRAP_TYPES_DICT = {
    "nonoverlapping": NonOverlappingBlockBootstrap,
    "moving": MovingBlockBootstrap,
    "stationary": StationaryBlockBootstrap,
    "circular": CircularBlockBootstrap,
}
