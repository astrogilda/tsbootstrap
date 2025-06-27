"""
Common field definitions for bootstrap classes.

This module centralizes the definition of commonly used Pydantic fields
across bootstrap implementations to reduce code duplication and ensure
consistency.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import Field, PositiveInt

from tsbootstrap.utils.types import (
    DistributionTypes,
    ModelTypes,
    ModelTypesWithoutArch,
    OrderTypes,
)
from tsbootstrap.validators import BlockLengthDistribution, ModelOrder

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
    Create a model_type field with custom defaults.

    Parameters
    ----------
    default : ModelTypes, default="ar"
        The default model type.
    include_arch : bool, default=True
        Whether to include 'arch' in allowed model types.

    Returns
    -------
    Field
        A Pydantic Field instance.
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
    Create a block_length field with custom defaults.

    Parameters
    ----------
    default : Optional[int], default=None
        The default block length. If None, will be computed as sqrt(n).
    required : bool, default=False
        Whether the field is required.
    ge : int, default=1
        The minimum allowed value.

    Returns
    -------
    Field
        A Pydantic Field instance.
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
