"""
Custom validators using Pydantic 2.x Annotated types.

This module provides reusable type annotations with built-in validation
for common bootstrap parameters, leveraging Pydantic 2.x features.
"""
from __future__ import annotations

from typing import Annotated, Any, List, Optional, Union

import numpy as np
from pydantic import (
    AfterValidator,
    BeforeValidator,
    Field,
    PlainSerializer,
    StringConstraints,
)
from pydantic_core import core_schema

from tsbootstrap.utils.types import OrderTypes


def validate_positive_int(v: Any) -> int:
    """Validate that a value is a positive integer."""
    if not isinstance(v, (int, np.integer)):
        raise TypeError(f"Expected integer, got {type(v).__name__}")
    value = int(v)
    if value <= 0:
        raise ValueError(f"Value must be positive, got {value}")
    return value


def validate_non_negative_int(v: Any) -> int:
    """Validate that a value is a non-negative integer."""
    if not isinstance(v, (int, np.integer)):
        raise TypeError(f"Expected integer, got {type(v).__name__}")
    value = int(v)
    if value < 0:
        raise ValueError(f"Value must be non-negative, got {value}")
    return value


def validate_probability(v: Any) -> float:
    """Validate that a value is a valid probability [0, 1]."""
    try:
        value = float(v)
    except (TypeError, ValueError) as err:
        raise TypeError(f"Expected numeric value, got {type(v).__name__}") from err

    if not 0 <= value <= 1:
        raise ValueError(f"Probability must be between 0 and 1, got {value}")
    return value


def validate_fraction(v: Any) -> float:
    """Validate that a value is a valid fraction (0, 1)."""
    try:
        value = float(v)
    except (TypeError, ValueError) as err:
        raise TypeError(f"Expected numeric value, got {type(v).__name__}") from err

    if not 0 < value < 1:
        raise ValueError(f"Fraction must be between 0 and 1 (exclusive), got {value}")
    return value


def validate_rng(v: Any) -> Optional[Union[int, np.random.Generator]]:
    """
    Validate and normalize RNG input.

    lightweight version that returns the input as-is if valid.
    For full validation and conversion to Generator, use utils.validate.validate_rng.

    Parameters
    ----------
    v : Any
        The input to validate.

    Returns
    -------
    Optional[Union[int, np.random.Generator]]
        The validated RNG value.

    Raises
    ------
    TypeError
        If the input is not a valid RNG type.
    """
    if v is None:
        return None
    if isinstance(v, np.random.Generator):
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    raise TypeError(f"RNG must be None, int, or np.random.Generator, got {type(v).__name__}")


def validate_block_length_distribution(v: Any) -> Optional[str]:
    """Validate block length distribution type."""
    if v is None:
        return None
    if not isinstance(v, str):
        raise TypeError(f"Expected string, got {type(v).__name__}")

    valid_distributions = {"uniform", "geometric", "exponential", "poisson"}
    if v not in valid_distributions:
        raise ValueError(f"Invalid distribution '{v}'. Must be one of {valid_distributions}")
    return v


def validate_order(v: Any) -> OrderTypes:
    """Validate model order parameter."""
    # Handle single integer
    if isinstance(v, (int, np.integer)):
        value = int(v)
        if value <= 0:
            raise ValueError(f"Order must be positive, got {value}")
        return value

    # Handle list of integers
    if isinstance(v, list):
        if not v:
            raise ValueError("Order list cannot be empty")
        validated = []
        for item in v:
            if not isinstance(item, (int, np.integer)):
                raise TypeError(f"Order list must contain only integers, got {type(item).__name__}")
            val = int(item)
            if val <= 0:
                raise ValueError(f"All orders must be positive, got {val}")
            validated.append(val)
        return validated

    # Handle tuples (for ARIMA/SARIMA orders)
    if isinstance(v, tuple):
        if len(v) not in [3, 4]:
            raise ValueError(f"Order tuple must have 3 or 4 elements, got {len(v)}")
        validated = []
        for _i, item in enumerate(v):
            if not isinstance(item, (int, np.integer)):
                raise TypeError(
                    f"Order tuple must contain only integers, got {type(item).__name__}"
                )
            val = int(item)
            if val < 0:
                raise ValueError(f"Order values must be non-negative, got {val}")
            validated.append(val)
        return tuple(validated)

    raise TypeError(f"Order must be int, List[int], or tuple, got {type(v).__name__}")


def serialize_numpy_array(v: np.ndarray) -> List:
    """Serialize numpy array to list for JSON compatibility."""
    return v.tolist()


def validate_array_input(v: Any) -> np.ndarray:
    """Validate and convert input to numpy array."""
    if isinstance(v, np.ndarray):
        return v
    try:
        arr = np.asarray(v)
        if arr.ndim == 0:
            raise
    except Exception as e:
        raise TypeError(f"Cannot convert to numpy array: {e}") from e
    else:
        return arr


# Annotated type definitions with validators
PositiveInt = Annotated[
    int,
    AfterValidator(validate_positive_int),
    Field(description="A positive integer value"),
]

NonNegativeInt = Annotated[
    int,
    AfterValidator(validate_non_negative_int),
    Field(description="A non-negative integer value"),
]

Probability = Annotated[
    float,
    AfterValidator(validate_probability),
    Field(description="A probability value between 0 and 1"),
]

Fraction = Annotated[
    float,
    AfterValidator(validate_fraction),
    Field(description="A fraction value between 0 and 1 (exclusive)"),
]

RngType = Annotated[
    Optional[Union[int, np.random.Generator]],
    BeforeValidator(validate_rng),
    Field(description="Random number generator or seed"),
]

BlockLengthDistribution = Annotated[
    Optional[str],
    AfterValidator(validate_block_length_distribution),
    Field(description="Distribution type for random block lengths"),
]

ModelOrder = Annotated[
    OrderTypes,
    AfterValidator(validate_order),
    Field(description="Model order specification"),
]

NumpyArray = Annotated[
    np.ndarray,
    BeforeValidator(validate_array_input),
    PlainSerializer(serialize_numpy_array, when_used="json"),
    Field(description="Numpy array data"),
]


# Custom validator for bootstrap-specific constraints
def validate_bootstrap_params(n_bootstraps: int) -> bool:
    """Validate bootstrap parameters are consistent."""
    return True


# String constraints for method names
MethodName = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9_]*$", min_length=1, max_length=50),
    Field(description="Valid method name in snake_case"),
]


# Custom type for statistic specification
StatisticType = Annotated[
    str,
    StringConstraints(pattern=r"^(mean|var|cov|median|std)$"),
    Field(description="Type of statistic to preserve"),
]


# Array shape validator
def validate_2d_array(v: np.ndarray) -> np.ndarray:
    """Ensure array is 2D."""
    if v.ndim == 1:
        return v.reshape(-1, 1)
    elif v.ndim == 2:
        return v
    else:
        raise ValueError(f"Array must be 1D or 2D, got {v.ndim}D")


Array2D = Annotated[
    np.ndarray,
    BeforeValidator(validate_array_input),
    AfterValidator(validate_2d_array),
    PlainSerializer(serialize_numpy_array, when_used="json"),
    Field(description="2D numpy array"),
]


# Bootstrap-specific compound types
class BootstrapIndices:
    """Custom type for bootstrap indices with validation."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, handler: Any
    ) -> core_schema.CoreSchema:
        """Define schema for bootstrap indices."""

        def validate_indices(v: Any) -> np.ndarray:
            if isinstance(v, (list, tuple)):
                v = np.array(v)
            if not isinstance(v, np.ndarray):
                raise TypeError("Indices must be array-like")
            if v.ndim != 1:
                raise ValueError("Indices must be 1D")
            if not np.issubdtype(v.dtype, np.integer):
                raise TypeError("Indices must be integers")
            if np.any(v < 0):
                raise ValueError("Indices must be non-negative")
            return v

        return core_schema.no_info_after_validator_function(
            validate_indices, core_schema.any_schema()
        )
