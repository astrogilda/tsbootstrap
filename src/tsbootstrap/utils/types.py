"""
Type definitions: Building a shared vocabulary for time series bootstrapping.

When we started this project, type confusion was a constant source of bugs.
What exactly is an "order"—an integer, a tuple, a list? Can RNG be None or
must it be a Generator? These ambiguities led to runtime errors that proper
typing could have prevented at development time.

This module establishes our type vocabulary, leveraging Python's type system
to encode constraints that make invalid states unrepresentable. We use Literal
types for closed sets of options, Union types for flexible parameters, and
careful Optional annotations to distinguish "can be None" from "must have value".

The type definitions here serve as both documentation and enforcement. When
you see OrderTypes in a function signature, you immediately know it accepts
integers for simple models, tuples for ARIMA specifications, or lists for
order selection ranges. This clarity propagates throughout the codebase.

We've also navigated Python version compatibility here, providing rich types
for modern Python while maintaining compatibility with older versions through
careful feature detection and fallbacks.
"""

from __future__ import annotations

import sys
from enum import Enum
from numbers import Integral
from typing import Any, List, Literal, Optional, Union

from numpy.random import Generator
from packaging.specifiers import SpecifierSet

# Define model and block compressor types using Literal for clearer enum-style typing.
ModelTypesWithoutArch = Literal["ar", "arima", "sarima", "var"]

ModelTypes = Literal["ar", "arima", "sarima", "var", "arch"]

BlockCompressorTypes = Literal[
    "first",
    "middle",
    "last",
    "mean",
    "mode",
    "median",
    "kmeans",
    "kmedians",
    "kmedoids",
]


class DistributionTypes(Enum):
    """
    Supported distributions for variable block length sampling.

    Each distribution here represents a different philosophy about block
    length variability. We've curated this list based on theoretical results
    and empirical performance across diverse time series applications.

    GEOMETRIC stands out as theoretically motivated—it's the only distribution
    yielding a stationary bootstrap. EXPONENTIAL approximates geometric for
    continuous contexts. UNIFORM provides bounded randomness when you know
    reasonable limits. The others serve specialized needs we've encountered
    in practice.
    """

    NONE = "none"
    POISSON = "poisson"
    EXPONENTIAL = "exponential"
    NORMAL = "normal"
    GAMMA = "gamma"
    BETA = "beta"
    LOGNORMAL = "lognormal"
    WEIBULL = "weibull"
    PARETO = "pareto"
    GEOMETRIC = "geometric"
    UNIFORM = "uniform"


# Version detection for conditional type definitions
# We check runtime Python version to provide the richest possible
# types while maintaining backward compatibility.
sys_version = sys.version.split(" ")[0]
new_typing_available = sys_version in SpecifierSet(">=3.10")


def FittedModelTypes() -> tuple:
    """
    Gather all fitted model types for runtime type checking.

    We face a challenge: different statistical packages return different
    result objects after model fitting. This function provides a unified
    way to check "is this a fitted model?" regardless of its origin.

    The lazy import pattern here prevents circular dependencies while
    still providing comprehensive type coverage. We've included all the
    major model result types we support across statsmodels and arch.

    Returns
    -------
    tuple
        All supported fitted model result types for isinstance checks.
    """
    from arch.univariate.base import ARCHModelResult
    from statsmodels.tsa.ar_model import AutoRegResultsWrapper
    from statsmodels.tsa.arima.model import ARIMAResultsWrapper
    from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
    from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

    fmt = (
        AutoRegResultsWrapper,
        ARIMAResultsWrapper,
        SARIMAXResultsWrapper,
        VARResultsWrapper,
        ARCHModelResult,
    )
    return fmt


# Type definitions for complex parameter types
#
# We define RngTypes unconditionally to satisfy static type checkers.
# This represents our flexible approach to random number generation:
# users can pass None (use default), an integer seed (reproducible),
# or a configured Generator (full control).
RngTypes = Optional[Union[Generator, Integral]]

if new_typing_available:
    OrderTypesWithoutNone = Union[
        Integral,
        List[Integral],
        tuple[Integral, Integral, Integral],
        tuple[Integral, Integral, Integral, Integral],
    ]
    OrderTypes = Optional[OrderTypesWithoutNone]

else:
    OrderTypesWithoutNone = Any
    OrderTypes = Any
