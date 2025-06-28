"""
Enhanced bootstrap configuration types using Pydantic 2.x advanced features.

This module provides improved type safety and validation using custom
Annotated types and advanced Pydantic features.
"""

from typing import Any, Dict, Literal, Optional, Union

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_serializer,
    model_serializer,
    model_validator,
)

from tsbootstrap.utils.types import ModelTypesWithoutArch
from tsbootstrap.validators import (
    BlockLengthDistribution,
    Fraction,
    ModelOrder,
    PositiveInt,
    RngType,
    StatisticType,
)


class BaseBootstrapConfig(BaseModel):
    """Enhanced base configuration for all bootstrap types."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        # Performance optimizations
        validate_default=False,
        use_enum_values=True,
        # Better error messages
        validate_return=True,
        str_strip_whitespace=True,
    )

    n_bootstraps: PositiveInt = Field(
        default=10,
        description="The number of bootstrap samples to create",
    )

    rng: RngType = Field(
        default=None,
        description="Random number generator or seed",
    )

    @computed_field
    @property
    def is_parallel_capable(self) -> bool:
        """Check if parallel processing would be beneficial."""
        return self.n_bootstraps > 10

    @computed_field
    @property
    def estimated_memory_mb(self) -> float:
        """Estimate memory usage in MB (to be overridden by subclasses)."""
        # Base estimate: ~8MB per bootstrap sample
        return self.n_bootstraps * 8.0

    @field_serializer("rng", when_used="json")
    def serialize_rng(self, rng: RngType) -> Optional[int]:
        """Serialize RNG for JSON compatibility."""
        if isinstance(rng, np.random.Generator):
            # Can't serialize Generator, return None
            return None
        return rng

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization validation."""
        # Can be overridden by subclasses for additional validation


class WholeBootstrapConfig(BaseBootstrapConfig):
    """Enhanced configuration for whole bootstrap methods."""

    bootstrap_type: Literal["whole"] = Field(
        default="whole",
        frozen=True,  # Make immutable
        description="Bootstrap type identifier",
    )

    @computed_field
    @property
    def block_structure(self) -> bool:
        """Whether this bootstrap uses block structure."""
        return False


class BlockBootstrapConfig(BaseBootstrapConfig):
    """Enhanced configuration for block bootstrap methods."""

    bootstrap_type: Literal["block"] = Field(
        default="block",
        frozen=True,
    )

    block_length: Optional[PositiveInt] = Field(
        default=None,
        description="Fixed block length if not using distribution",
    )

    block_length_distribution: BlockLengthDistribution = Field(
        default=None,
        description="Distribution for random block lengths",
    )

    avg_block_length: Optional[PositiveInt] = Field(
        default=None,
        description="Average block length for random distributions",
    )

    min_block_length: PositiveInt = Field(
        default=1,
        description="Minimum block length for random distributions",
    )

    wrap_around_flag: bool = Field(
        default=False,
        description="Whether to wrap around the time series",
    )

    overlap_flag: bool = Field(
        default=False,
        description="Whether blocks can overlap",
    )

    @model_validator(mode="after")
    def validate_block_config(self) -> "BlockBootstrapConfig":
        """Validate block configuration consistency."""
        if self.block_length is None and self.block_length_distribution is None:
            raise ValueError("Either block_length or block_length_distribution must be specified")

        if self.block_length is not None and self.block_length_distribution is not None:
            raise ValueError("Cannot specify both block_length and block_length_distribution")

        if self.block_length_distribution is not None and self.avg_block_length is None:
            raise ValueError(
                "avg_block_length must be specified when using block_length_distribution"
            )

        if self.avg_block_length is not None and self.avg_block_length < self.min_block_length:
            raise ValueError("avg_block_length must be >= min_block_length")

        return self

    @computed_field
    @property
    def effective_block_length(self) -> int:
        """Get the effective block length for memory estimation."""
        if self.block_length is not None:
            return self.block_length
        return self.avg_block_length or 10

    @computed_field
    @property
    def block_structure(self) -> bool:
        """Whether this bootstrap uses block structure."""
        return True


class ResidualBootstrapConfig(BaseBootstrapConfig):
    """Enhanced configuration for residual bootstrap methods."""

    bootstrap_type: Literal["residual"] = Field(
        default="residual",
        frozen=True,
    )

    model_type: ModelTypesWithoutArch = Field(
        default="ar",
        description="Time series model type",
    )

    model_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional model parameters",
    )

    order: Optional[ModelOrder] = Field(
        default=None,
        description="Model order (auto-detected if None)",
    )

    seasonal_order: Optional[ModelOrder] = Field(
        default=None,
        description="Seasonal order for SARIMA models",
    )

    save_models: bool = Field(
        default=False,
        description="Whether to save fitted models",
    )

    @model_validator(mode="after")
    def validate_model_config(self) -> "ResidualBootstrapConfig":
        """Validate model configuration."""
        if self.model_type == "sarima" and self.seasonal_order is None:
            raise ValueError("seasonal_order must be specified for SARIMA models")

        if self.model_type != "sarima" and self.seasonal_order is not None:
            raise ValueError(f"seasonal_order should not be specified for {self.model_type} models")

        return self

    @computed_field
    @property
    def requires_model_fitting(self) -> bool:
        """Whether this bootstrap requires model fitting."""
        return True


class MarkovBootstrapConfig(BaseBootstrapConfig):
    """Enhanced configuration for Markov bootstrap methods."""

    bootstrap_type: Literal["markov"] = Field(
        default="markov",
        frozen=True,
    )

    method: Literal["first", "middle", "last", "biased_coinflip"] = Field(
        default="middle",
        description="Method for selecting values within states",
    )

    transition_probs_method: Literal["mle", "bootstrap"] = Field(
        default="mle",
        description="Method for estimating transition probabilities",
    )

    @computed_field
    @property
    def uses_transition_matrix(self) -> bool:
        """Whether this bootstrap uses a transition matrix."""
        return True


class DistributionBootstrapConfig(BaseBootstrapConfig):
    """Enhanced configuration for distribution bootstrap methods."""

    bootstrap_type: Literal["distribution"] = Field(
        default="distribution",
        frozen=True,
    )

    distribution: Literal[
        "normal",
        "exponential",
        "gamma",
        "beta",
        "lognormal",
        "weibull",
        "pareto",
        "geometric",
        "poisson",
        "uniform",
    ] = Field(
        default="normal",
        description="Distribution to fit to the data",
    )

    refit_each_time: bool = Field(
        default=False,
        description="Whether to refit distribution for each bootstrap",
    )

    @computed_field
    @property
    def parametric(self) -> bool:
        """Whether parametric bootstrap."""
        return True


class SieveBootstrapConfig(ResidualBootstrapConfig):
    """Enhanced configuration for sieve bootstrap methods."""

    bootstrap_type: Literal["sieve"] = Field(
        default="sieve",
        frozen=True,
    )

    model_type: Literal["ar"] = Field(
        default="ar",
        frozen=True,  # Sieve only works with AR
        description="Sieve bootstrap only supports AR models",
    )

    min_lag: PositiveInt = Field(
        default=1,
        description="Minimum lag for AR order selection",
    )

    max_lag: Optional[PositiveInt] = Field(
        default=None,
        description="Maximum lag for AR order selection",
    )

    criterion: Literal["aic", "bic", "hqic"] = Field(
        default="aic",
        description="Information criterion for order selection",
    )

    @model_validator(mode="after")
    def validate_lag_config(self) -> "SieveBootstrapConfig":
        """Validate lag configuration."""
        if self.max_lag is not None and self.max_lag < self.min_lag:
            raise ValueError("max_lag must be >= min_lag")
        return self


class StatisticPreservingBootstrapConfig(BaseBootstrapConfig):
    """Enhanced configuration for statistic preserving bootstrap."""

    bootstrap_type: Literal["statistic_preserving"] = Field(
        default="statistic_preserving",
        frozen=True,
    )

    statistic: StatisticType = Field(
        default="mean",
        description="Statistic to preserve",
    )

    statistic_axis: Literal[0, 1] = Field(
        default=0,
        description="Axis for statistic computation",
    )

    statistic_keepdims: bool = Field(
        default=False,
        description="Whether to keep dimensions in statistic",
    )

    tolerance: Fraction = Field(
        default=0.001,
        description="Tolerance for statistic matching",
    )

    max_iterations: PositiveInt = Field(
        default=100,
        description="Maximum iterations for matching",
    )

    @computed_field
    @property
    def preserves_dependencies(self) -> bool:
        """Whether this method preserves temporal dependencies."""
        return self.statistic in ["cov"]

    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        """Custom serialization with additional metadata."""
        data = {
            "bootstrap_type": self.bootstrap_type,
            "n_bootstraps": self.n_bootstraps,
            "statistic": self.statistic,
            "statistic_config": {
                "axis": self.statistic_axis,
                "keepdims": self.statistic_keepdims,
                "tolerance": self.tolerance,
                "max_iterations": self.max_iterations,
            },
            "metadata": {
                "preserves_dependencies": self.preserves_dependencies,
                "is_parallel_capable": self.is_parallel_capable,
            },
        }
        return data


# Type alias for all bootstrap configurations
BootstrapConfig = Union[
    WholeBootstrapConfig,
    BlockBootstrapConfig,
    ResidualBootstrapConfig,
    MarkovBootstrapConfig,
    DistributionBootstrapConfig,
    SieveBootstrapConfig,
    StatisticPreservingBootstrapConfig,
]
