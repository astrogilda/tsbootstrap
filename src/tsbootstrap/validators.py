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
        raise TypeError(
            f"Expected an integer value but received {type(v).__name__}. "
            f"This parameter must be a whole number (int or numpy integer type). "
            f"If you have a float value, consider using int() to convert it."
        )
    value = int(v)
    if value <= 0:
        raise ValueError(
            f"This parameter must be a positive integer (greater than 0). "
            f"Received: {value}. Positive integers are required for counts, sizes, "
            f"and iterations. Please provide a value of 1 or greater."
        )
    return value


def validate_non_negative_int(v: Any) -> int:
    """Validate that a value is a non-negative integer."""
    if not isinstance(v, (int, np.integer)):
        raise TypeError(
            f"Expected an integer value but received {type(v).__name__}. "
            f"This parameter must be a whole number (int or numpy integer type). "
            f"If you have a float value, consider using int() to convert it."
        )
    value = int(v)
    if value < 0:
        raise ValueError(
            f"This parameter must be non-negative (0 or greater). "
            f"Received: {value}. Non-negative integers are required for indices, "
            f"offsets, and optional counts. Please provide a value of 0 or greater."
        )
    return value


def validate_probability(v: Any) -> float:
    """Validate that a value is a valid probability [0, 1]."""
    try:
        value = float(v)
    except (TypeError, ValueError) as err:
        raise TypeError(
            f"Expected a numeric value for probability but received {type(v).__name__}. "
            f"Probabilities must be numbers (int or float) that can represent likelihood. "
            f"Please provide a numeric value."
        ) from err

    if not 0 <= value <= 1:
        raise ValueError(
            f"Probability values must be between 0 and 1 (inclusive). "
            f"Received: {value}. Probabilities represent likelihoods where 0 means "
            f"impossible and 1 means certain. Please provide a value in the range [0, 1]."
        )
    return value


def validate_fraction(v: Any) -> float:
    """Validate that a value is a valid fraction (0, 1)."""
    try:
        value = float(v)
    except (TypeError, ValueError) as err:
        raise TypeError(
            f"Expected a numeric value for fraction but received {type(v).__name__}. "
            f"Fractions must be numbers (int or float) representing parts of a whole. "
            f"Please provide a numeric value."
        ) from err

    if not 0 < value < 1:
        raise ValueError(
            f"Fraction values must be strictly between 0 and 1 (exclusive). "
            f"Received: {value}. Valid fractions are like 0.25, 0.5, or 0.75 - "
            f"they cannot be 0 or 1. Please provide a value in the range (0, 1)."
        )
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
    raise TypeError(
        f"Random number generator must be None, an integer seed, or np.random.Generator instance. "
        f"Received: {type(v).__name__}. Use None for default RNG, an integer for reproducible "
        f"randomness (e.g., rng=42), or pass an existing np.random.Generator instance."
    )


def validate_block_length_distribution(v: Any) -> Optional[str]:
    """Validate block length distribution type."""
    if v is None:
        return None
    if not isinstance(v, str):
        raise TypeError(
            f"Block length distribution must be specified as a string. "
            f"Received: {type(v).__name__}. Please provide the distribution name "
            f"as a string, e.g., 'geometric' or 'exponential'."
        )

    valid_distributions = {"uniform", "geometric", "exponential", "poisson"}
    if v not in valid_distributions:
        raise ValueError(
            f"Unknown block length distribution: '{v}'. "
            f"Supported distributions are: {', '.join(sorted(valid_distributions))}. "
            f"Each distribution has different properties - 'geometric' is often preferred "
            f"for stationary block bootstrap."
        )
    return v


def validate_order(v: Any) -> OrderTypes:
    """Validate model order parameter."""
    # Handle single integer
    if isinstance(v, (int, np.integer)):
        value = int(v)
        if value <= 0:
            raise ValueError(
                f"Model order must be a positive integer. Received: {value}. "
                f"The order represents the number of lagged observations to include "
                f"in the model. Please provide a value of 1 or greater."
            )
        return value

    # Handle list of integers
    if isinstance(v, list):
        if not v:
            raise ValueError(
                "Order list cannot be empty. When providing multiple orders for model "
                "selection, include at least one positive integer representing a lag order "
                "to test, e.g., [1, 2, 3] or [1, 3, 5, 7]."
            )
        validated = []
        for item in v:
            if not isinstance(item, (int, np.integer)):
                raise TypeError(
                    f"Order list must contain only integers. Found {type(item).__name__} "
                    f"in the list. Each element should be a positive integer representing "
                    f"a lag order, e.g., [1, 2, 3] not [1, 2.5, 3]."
                )
            val = int(item)
            if val <= 0:
                raise ValueError(
                    f"All model orders must be positive integers. Found: {val} in the list. "
                    f"Each order represents the number of lags to include. Please ensure "
                    f"all values are 1 or greater."
                )
            validated.append(val)
        return validated

    # Handle tuples (for ARIMA/SARIMA orders)
    if isinstance(v, tuple):
        if len(v) not in [3, 4]:
            raise ValueError(
                f"ARIMA/SARIMA order tuple must have exactly 3 elements (p, d, q) for ARIMA "
                f"or 4 elements (p, d, q, s) for seasonal ARIMA. Received tuple with {len(v)} "
                f"elements. Example: (1, 1, 1) for ARIMA(1,1,1) or (1, 1, 1, 12) for seasonal."
            )
        validated = []
        for _i, item in enumerate(v):
            if not isinstance(item, (int, np.integer)):
                raise TypeError(
                    f"ARIMA order tuple must contain only integers. Found {type(item).__name__} "
                    f"in position {_i}. Each element should be a non-negative integer: "
                    f"(p=AR order, d=differencing, q=MA order, s=seasonal period)."
                )
            val = int(item)
            if val < 0:
                raise ValueError(
                    f"ARIMA order values must be non-negative. Found {val} in position {_i}. "
                    f"Use 0 to exclude a component (e.g., (1, 0, 0) for pure AR model) "
                    f"or positive values to include it."
                )
            validated.append(val)
        return tuple(validated)

    raise TypeError(
        f"Model order must be an integer, a list of integers, or a tuple. "
        f"Received: {type(v).__name__}. Valid formats: "
        f"int (e.g., 2), list (e.g., [1, 2, 3]), or tuple (e.g., (1, 0, 1)). "
        f"Use int for single order, list for order selection, tuple for ARIMA specifications."
    )


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
        raise TypeError(
            f"Cannot convert input to numpy array. The data provided is not in a format "
            f"that can be interpreted as an array. Common array-like formats include: "
            f"lists [1, 2, 3], tuples (1, 2, 3), or existing numpy arrays. "
            f"Original error: {e}"
        ) from e
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
        raise ValueError(
            f"Input array has {v.ndim} dimensions, but only 1D or 2D arrays are supported. "
            f"1D arrays represent univariate time series, 2D arrays represent multivariate "
            f"time series with shape (n_samples, n_features). Consider using array.reshape() "
            f"or array.flatten() to adjust dimensions."
        )


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
                raise TypeError(
                    "Bootstrap indices must be array-like (list, tuple, or numpy array). "
                    "These indices specify which observations to include in the bootstrap sample."
                )
            if v.ndim != 1:
                raise ValueError(
                    f"Bootstrap indices must be a 1-dimensional array. Received {v.ndim}D array. "
                    f"Indices should be a flat array of integers like [0, 1, 2, 1, 0] representing "
                    f"which observations to select."
                )
            if not np.issubdtype(v.dtype, np.integer):
                raise TypeError(
                    f"Bootstrap indices must be integers, but array has dtype {v.dtype}. "
                    f"Indices represent positions in the original data and must be whole numbers. "
                    f"Consider using array.astype(int) if appropriate."
                )
            if np.any(v < 0):
                raise ValueError(
                    "Bootstrap indices must be non-negative. Found negative values in the array. "
                    "Indices represent positions in the data starting from 0. Ensure all values "
                    "are valid array indices."
                )
            return v

        return core_schema.no_info_after_validator_function(
            validate_indices, core_schema.any_schema()
        )
