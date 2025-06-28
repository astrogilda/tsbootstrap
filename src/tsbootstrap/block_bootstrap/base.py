"""
Base classes for block bootstrap methods.
"""

from __future__ import annotations

import logging
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Type,
    Union,
)

import numpy as np
from pydantic import ConfigDict, Field, PositiveInt, field_validator

from tsbootstrap.block_generator import BlockGenerator
from tsbootstrap.block_length_sampler import BlockLengthSampler
from tsbootstrap.block_resampler import BlockResampler
from tsbootstrap.bootstrap_intermediate import BlockBasedBootstrap
from tsbootstrap.common_fields import (
    BLOCK_LENGTH_DISTRIBUTION_FIELD,
    BLOCK_LENGTH_FIELD,
    MIN_BLOCK_LENGTH_FIELD,
    OVERLAP_FLAG_FIELD,
    WRAP_AROUND_FLAG_FIELD,
)
from tsbootstrap.utils.types import DistributionTypes

logger = logging.getLogger(__name__)


class BlockBootstrap(BlockBasedBootstrap):
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

    block_length: Optional[PositiveInt] = BLOCK_LENGTH_FIELD
    block_length_distribution: Optional[DistributionTypes] = BLOCK_LENGTH_DISTRIBUTION_FIELD
    wrap_around_flag: bool = WRAP_AROUND_FLAG_FIELD
    overlap_flag: bool = OVERLAP_FLAG_FIELD
    combine_generation_and_sampling_flag: bool = Field(default=False)
    block_weights: Optional[Union[np.ndarray, Callable]] = Field(default=None)
    tapered_weights: Optional[Callable] = Field(default=None)
    overlap_length: Optional[PositiveInt] = Field(default=None, ge=1)
    min_block_length: Optional[PositiveInt] = MIN_BLOCK_LENGTH_FIELD

    blocks: Optional[list[np.ndarray]] = Field(default=None, init=False)
    block_resampler: Optional[BlockResampler] = Field(default=None, init=False)

    def _check_input_bb(self, X: np.ndarray, enforce_univariate=True) -> None:
        if self.block_length is not None and self.block_length > X.shape[0]:
            raise ValueError("block_length cannot be greater than the size of the input array X.")

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
                self.block_length if self.block_length is not None else int(np.sqrt(X.shape[0]))
            ),
            block_length_distribution=self.block_length_distribution,
            rng=self.rng if self.rng is not None else np.random.default_rng(),
        )

        block_generator = BlockGenerator(
            block_length_sampler=block_length_sampler,
            input_length=X.shape[0],
            rng=self.rng if self.rng is not None else np.random.default_rng(),
            wrap_around_flag=self.wrap_around_flag,
            overlap_length=self.overlap_length,
            min_block_length=self.min_block_length,
        )

        blocks = block_generator.generate_blocks(overlap_flag=self.overlap_flag)

        logger.debug(f"DEBUG: BlockGenerator generated {len(blocks)} blocks.")
        return blocks

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y=None, n: Optional[int] = None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Generate a single bootstrap sample.
        """
        if n is None:
            n = X.shape[0]
        logger.debug(
            f"BlockBootstrap._generate_samples_single_bootstrap: X.shape = {X.shape}, target n = {n}"
        )
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
            if self.block_resampler is None:
                raise RuntimeError(
                    "Internal invariant broken: self.block_resampler should not be None here when "
                    "self.blocks is populated and combine_generation_and_sampling_flag is False."
                )
            block_resampler = self.block_resampler

        (
            block_indices,
            block_data,
        ) = block_resampler.resample_block_indices_and_data(n=n)

        if not self.combine_generation_and_sampling_flag:
            self.blocks = blocks
            self.block_resampler = block_resampler

        return block_indices, block_data

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator."""
        params = super().get_params(deep=deep)
        # Ensure n_bootstraps is always included
        if "n_bootstraps" not in params:
            params["n_bootstraps"] = self.n_bootstraps

        # Ensure all Pydantic model fields are included
        model_fields = type(self).model_fields
        for field_name, field_info in model_fields.items():
            if field_info.init_var:
                continue
            if (
                field_info.init is not False
                and field_name not in params
                and hasattr(self, field_name)
            ):
                params[field_name] = getattr(self, field_name)

        return params


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
    n_bootstraps : int, default=10
        The number of bootstrap samples to create.
    rng : int or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.
    block_length : Optional[int], default=None
        The length of the blocks to sample. If None, the block length is automatically
        set to the square root of the number of observations.
    block_length_distribution : Optional[str], default=None
        The block length distribution function to use. If None, the block length
        distribution is not utilized.
    wrap_around_flag : bool, default=False
        Whether to wrap around the data when generating blocks.
    overlap_flag : bool, default=False
        Whether to allow blocks to overlap.
    combine_generation_and_sampling_flag : bool, default=False
        Whether to combine the block generation and sampling steps.
    block_weights : Optional[Union[np.ndarray, Callable]], default=None
        The weights to use when sampling blocks.
    tapered_weights : Optional[Callable], default=None
        The tapered weights to use when sampling blocks.
    overlap_length : Optional[int], default=None
        The length of the overlap between blocks.
    min_block_length : Optional[int], default=None
        The minimum length of the blocks.

    Attributes
    ----------
    bootstrap_instance: Optional[BlockBootstrap] = Field(default=None, init=False, validate_default=False)
        An instance of the specified bootstrap type class.

    Methods
    -------
    validate_bootstrap_type
    model_post_init
    _generate_samples_single_bootstrap
    __repr__
    """

    bootstrap_type: Optional[str] = Field(default="moving")
    bootstrap_instance: Optional[BlockBootstrap] = Field(
        default=None, init=False, validate_default=False
    )

    @field_validator("bootstrap_type")
    @classmethod
    def validate_bootstrap_type(cls, v: Optional[str]) -> Optional[str]:
        """Validate the bootstrap_type."""
        from tsbootstrap.block_bootstrap.registry import (
            get_bootstrap_types_dict,
        )

        valid_types = get_bootstrap_types_dict()
        if v is not None and v not in valid_types:
            raise ValueError(
                f"Invalid bootstrap_type: {v}. Must be one of {list(valid_types.keys())}"
            )
        return v

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """
        Get parameters for this estimator.

        Ensures n_bootstraps is always included in the parameters.
        """
        # Get parameters from parent classes
        params = super().get_params(deep=deep)

        # Ensure n_bootstraps is always present
        if "n_bootstraps" not in params:
            params["n_bootstraps"] = self.n_bootstraps

        # Ensure all Pydantic model fields are included
        model_fields = type(self).model_fields
        for field_name, field_info in model_fields.items():
            if field_info.init_var:
                continue
            if (
                field_info.init is not False
                and field_name not in params
                and hasattr(self, field_name)
            ):
                params[field_name] = getattr(self, field_name)

        return params

    def model_post_init(self, __context: ConfigDict) -> None:
        """Post-initialization method called after the model is fully initialized."""
        from tsbootstrap.block_bootstrap.registry import (
            BLOCK_BOOTSTRAP_TYPES_DICT,
        )

        super().model_post_init(__context)  # Call parent's post_init first
        if self.bootstrap_type:
            # Get the bootstrap class based on the specified type
            bcls: Type[BlockBootstrap] = BLOCK_BOOTSTRAP_TYPES_DICT[self.bootstrap_type]

            # Collect all relevant parameters from self to pass to the bcls constructor
            init_kwargs = {
                # From BaseTimeSeriesBootstrap (via BlockBootstrap)
                "n_bootstraps": self.n_bootstraps,
                "rng": self.rng,  # self.rng is already a validated Generator instance
                # From BlockBootstrap - these are generally safe to pass from self
                "block_length": self.block_length,
                "block_weights": self.block_weights,
                # Carries validated func from self (e.g., for Bartletts)
                "tapered_weights": self.tapered_weights,
                "overlap_length": self.overlap_length,
                "min_block_length": self.min_block_length,
            }

            # For these parameters, only pass them if the target class 'bcls'
            # does NOT define them with a Literal type. If 'bcls' uses Literal,
            # it means it has a fixed value, and we should let it use that fixed default.
            conditional_params_from_self = {
                "wrap_around_flag": self.wrap_around_flag,
                "overlap_flag": self.overlap_flag,
                "block_length_distribution": self.block_length_distribution,
                "combine_generation_and_sampling_flag": self.combine_generation_and_sampling_flag,
            }

            for param_name, self_value in conditional_params_from_self.items():
                bcls_field = bcls.model_fields.get(param_name)
                is_bcls_literal = False
                if (
                    bcls_field
                    and bcls_field.annotation is not None
                    and hasattr(bcls_field.annotation, "__origin__")
                    and bcls_field.annotation.__origin__ is Literal
                ):
                    is_bcls_literal = True

                if not is_bcls_literal:
                    # If bcls does not use Literal for this param, pass the value from self
                    init_kwargs[param_name] = self_value

            # Create an instance of the specified bootstrap class with the collected parameters
            self.bootstrap_instance = bcls(**init_kwargs)

    def _generate_samples_single_bootstrap(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        n: Optional[int] = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Generate a single bootstrap sample using either the base BlockBootstrap method or the specified bootstrap_type."""
        if self.bootstrap_instance is None:
            # If no specific bootstrap instance is set, use the base class method
            return super()._generate_samples_single_bootstrap(X=X, y=y, n=n)
        else:
            if hasattr(self.bootstrap_instance, "_generate_samples_single_bootstrap"):
                # Use the specific bootstrap instance's method
                return self.bootstrap_instance._generate_samples_single_bootstrap(X=X, y=y, n=n)
            else:
                raise NotImplementedError(
                    f"The bootstrap class '{type(self.bootstrap_instance).__name__}' does not implement '_generate_samples_single_bootstrap' method."
                )

    def __repr__(self) -> str:
        """Return a string representation of the BaseBlockBootstrap instance."""
        instance_repr = None
        if hasattr(self, "bootstrap_instance") and self.bootstrap_instance is not None:
            instance_repr = type(self.bootstrap_instance).__name__
        return f"BaseBlockBootstrap(bootstrap_type='{self.bootstrap_type}', bootstrap_instance={instance_repr})"
