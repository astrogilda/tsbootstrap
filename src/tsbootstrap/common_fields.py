"""
Shared field definitions: Maintaining consistency across bootstrap implementations.

We created this module after noticing the same field definitions scattered
across dozens of bootstrap classes. Each duplicate definition was a potential
source of inconsistency—different descriptions, validation rules, or default
values for what should be identical parameters. By centralizing these
definitions, we ensure that a block_length field behaves identically whether
it appears in MovingBlockBootstrap or StationaryBlockBootstrap.

The field definitions here encode hard-won knowledge about sensible defaults
and constraints. For instance, we default to sqrt(n) for block length because
theoretical results suggest this scaling balances bias and variance. Each
field's validation rules prevent common mistakes we've observed in practice.

Beyond consistency, this approach simplifies maintenance. When we discover
a better default or need to clarify a description, we update it once here
rather than hunting through every bootstrap class.
"""
from __future__ import annotations

from typing import Optional

from pydantic import Field

from tsbootstrap.utils.types import (
    ModelTypes,
)

# Model-related fields
MODEL_TYPE_FIELD = Field(
    default="ar",
    description="The model type to use. Options are 'ar', 'ma', 'arma', 'arima', 'sarima', 'var', 'arch'.",
)

MODEL_TYPE_NO_ARCH_FIELD = Field(
    default="ar",
    description="The model type to use. Options are 'ar', 'ma', 'arma', 'arima', 'sarima', 'var' (no 'arch').",
)

ORDER_FIELD = Field(
    default=None,
    description="The order of the model. For AR/MA/ARCH/VAR: integer. For ARIMA/SARIMA: tuple of (p,d,q).",
)

SEASONAL_ORDER_FIELD = Field(
    default=None,
    description="The seasonal order for SARIMA models as tuple of (P,D,Q,s).",
)

SAVE_MODELS_FIELD = Field(
    default=False,
    description="Whether to save fitted models during bootstrap.",
)


# Block-related fields
BLOCK_LENGTH_FIELD = Field(
    default=None,
    ge=1,
    description="Length of blocks for block bootstrap. If None, defaults to sqrt(n).",
)

BLOCK_LENGTH_REQUIRED_FIELD = Field(
    ...,
    ge=1,
    description="Length of blocks for block bootstrap. Must be specified.",
)

BLOCK_LENGTH_DISTRIBUTION_FIELD = Field(
    default=None,
    description="Distribution used for sampling block lengths. Options: 'geometric', 'poisson', 'uniform', 'normal', 'gamma', 'beta', 'lognormal', 'weibull', 'pareto', 'exponential'.",
)

AVG_BLOCK_LENGTH_FIELD = Field(
    default=None,
    ge=1,
    description="Average block length for variable-length block methods.",
)

MIN_BLOCK_LENGTH_FIELD = Field(
    default=None,
    ge=1,
    description="Minimum block length when using variable-length blocks.",
)

OVERLAP_FLAG_FIELD = Field(
    default=True,
    description="Whether blocks are allowed to overlap.",
)

WRAP_AROUND_FLAG_FIELD = Field(
    default=False,
    description="Whether to wrap around the data when generating blocks.",
)


# Bootstrap configuration fields
N_BOOTSTRAPS_FIELD = Field(
    default=10,
    ge=1,
    description="The number of bootstrap samples to generate.",
)

RNG_FIELD = Field(
    default=None,
    description="Random number generator or seed for reproducibility.",
)


# Data validation fields
X_FIELD = Field(
    ...,
    description="The input time series data.",
)


# Factory function for creating field with custom defaults
def create_model_type_field(
    default: ModelTypes = "ar",
    include_arch: bool = True,
) -> Field:
    """
    Generate a model type field with context-appropriate constraints.

    We discovered that ARCH models don't play well with certain bootstrap
    methods—the volatility clustering they capture requires special handling.
    This factory lets bootstrap classes easily exclude ARCH when it's not
    supported, preventing confusing error messages deep in the computation.

    Parameters
    ----------
    default : ModelTypes, default="ar"
        The default model type. We chose AR as it's the simplest and most
        universally supported across bootstrap methods.
    include_arch : bool, default=True
        Whether to include 'arch' in allowed model types. Set False for
        methods that can't handle volatility models.

    Returns
    -------
    Field
        A configured Pydantic Field with appropriate validation.
    """
    if include_arch:
        description = "The model type to use. Options are 'ar', 'ma', 'arma', 'arima', 'sarima', 'var', 'arch'."
    else:
        description = "The model type to use. Options are 'ar', 'ma', 'arma', 'arima', 'sarima', 'var' (no 'arch')."

    return Field(default=default, description=description)


def create_block_length_field(
    default: Optional[int] = None,
    required: bool = False,
    ge: int = 1,
) -> Field:
    """
    Generate a block length field tailored to specific bootstrap needs.

    Block length selection remains one of the trickiest aspects of block
    bootstrap. Too short and we lose dependencies; too long and we have
    too few blocks to resample. This factory encodes our recommended
    practices while allowing methods to override based on their specific
    requirements.

    Parameters
    ----------
    default : Optional[int], default=None
        The default block length. When None, we compute sqrt(n) at runtime,
        following theoretical guidance for optimal bias-variance tradeoff.
    required : bool, default=False
        Whether users must explicitly specify block length. Some methods
        need this to prevent accidental misuse.
    ge : int, default=1
        The minimum allowed value. We enforce positive lengths to catch
        configuration errors early.

    Returns
    -------
    Field
        A configured Pydantic Field with block-specific validation.
    """
    if required:
        return Field(
            ...,
            ge=ge,
            description="Length of blocks for block bootstrap. Must be specified.",
        )
    else:
        return Field(
            default=default,
            ge=ge,
            description="Length of blocks for block bootstrap. If None, defaults to sqrt(n).",
        )
