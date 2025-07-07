"""
Configuration architecture: Type-safe blueprints for bootstrap methods.

When we designed the bootstrap configuration system, we faced a fundamental
challenge: how to provide flexibility for dozens of bootstrap variants while
maintaining type safety and preventing invalid configurations. Our solution
leverages Pydantic's advanced features to create a configuration framework
that guides users toward valid setups while catching errors before they
reach computational code.

Each configuration class here represents years of experience about what
parameters make sense together. We encode constraints like "block length
distributions require an average length" or "sieve bootstrap only works
with AR models" directly into the type system. This approach transforms
runtime errors into immediate validation feedback, dramatically improving
the developer experience.

The architecture follows a compositional pattern where base configurations
provide common functionality, while specialized configs add method-specific
constraints. We've found this design scales elegantly as new bootstrap
methods are added to the library.
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
    """Foundation for all bootstrap configurations: shared wisdom across methods.

    We've distilled the common requirements of all bootstrap methods into this
    base configuration. Every bootstrap variant, regardless of its specific
    algorithm, needs to control sample size and randomness. This class captures
    those universal needs while providing extension points for method-specific
    requirements.

    The computed fields here reflect patterns we've observed across thousands
    of bootstrap applications: when parallel processing becomes beneficial,
    how memory scales with sample size, and how to handle random number
    generators in distributed settings.
    """

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
        """Determine if parallel processing would improve performance.

        Through benchmarking, we've found that parallel overhead only pays off
        above 10 bootstrap samples. Below that threshold, the coordination cost
        exceeds the computational savings.
        """
        return self.n_bootstraps > 10

    @computed_field
    @property
    def estimated_memory_mb(self) -> float:
        """Estimate memory footprint for resource planning.

        We use 8MB per sample as our baseline, derived from profiling typical
        time series lengths. Subclasses refine this estimate based on their
        specific memory patterns—block methods need more, whole methods less.
        """
        return self.n_bootstraps * 8.0

    @field_serializer("rng", when_used="json")
    def serialize_rng(self, rng: RngType) -> Optional[int]:
        """Serialize RNG for JSON compatibility."""
        if isinstance(rng, np.random.Generator):
            # Can't serialize Generator, return None
            return None
        return rng

    def model_post_init(self, __context: Any) -> None:
        """Hook for subclass-specific validation after Pydantic's checks.

        We provide this extension point for bootstrap methods that need
        complex cross-field validation beyond what validators can express.
        The double underscore in __context follows Pydantic conventions.
        """
        pass  # Subclasses override as needed


class WholeBootstrapConfig(BaseBootstrapConfig):
    """Configuration for whole sample bootstrap: the simplest approach.

    Whole bootstrap methods resample entire time series observations,
    treating each as an independent unit. While this breaks temporal
    dependencies, it remains valuable for certain analyses where we
    care more about the marginal distribution than the time structure.
    """

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
    """Configuration for block bootstrap: preserving temporal dependencies.

    Block bootstrap represents our primary solution to the dependency
    problem in time series resampling. By sampling contiguous blocks
    rather than individual observations, we preserve local correlation
    structures. The configuration options here reflect decades of research
    into optimal block selection strategies.
    """

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
        """Ensure block parameters form a coherent configuration.

        We've learned from user feedback that certain parameter combinations
        lead to confusion or errors. This validator encodes those lessons,
        preventing specifications like both fixed and random block lengths,
        or random lengths without an average.
        """
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
    """Configuration for model-based residual bootstrap.

    Residual bootstrap combines parametric modeling with resampling,
    offering a middle ground between fully parametric and nonparametric
    approaches. We fit a time series model, extract residuals, resample
    them, and generate new series. This preserves the model structure
    while allowing for non-parametric error distributions.
    """

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
    """Configuration for Markov chain bootstrap.

    The Markov bootstrap captures state-dependent dynamics by treating
    the time series as transitions between discrete states. We build
    a transition matrix and generate new series by sampling from these
    transitions. The method choices here reflect different philosophies
    about state representation and transition estimation.
    """

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
    """Configuration for parametric distribution bootstrap.

    Sometimes we know (or assume) the underlying distribution of our data.
    Distribution bootstrap leverages this knowledge by fitting a parametric
    distribution and sampling from it. We support a wide range of distributions,
    each suited to different data characteristics—exponential for durations,
    lognormal for prices, beta for proportions.
    """

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
    """Configuration for sieve bootstrap: adaptive AR modeling.

    The sieve bootstrap addresses a key challenge in residual methods:
    choosing the right model order. Rather than fixing the order, we let
    it grow with sample size, approximating infinite-order processes with
    finite AR models. This configuration controls that adaptive selection
    process.
    """

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
    """Configuration for bootstrap that maintains specific statistical properties.

    We developed statistic-preserving bootstrap to address cases where
    standard resampling destroys important data characteristics. By iteratively
    adjusting samples to match target statistics, we ensure bootstrap samples
    reflect key properties of the original data. This proves especially valuable
    for risk metrics and correlation structures.
    """

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
