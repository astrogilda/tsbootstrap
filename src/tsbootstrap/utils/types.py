# Use future annotations for better handling of forward references.
from __future__ import annotations

from enum import Enum
from numbers import Integral
from typing import Literal, Optional, Union

from numpy.random import Generator


class DistributionTypes(str, Enum):
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

# Define type aliases for specific tuple lengths
Tuple3 = tuple[Integral, Integral, Integral]
Tuple4 = tuple[Integral, Integral, Integral, Integral]

OrderTypesWithoutNone = Union[Integral, list[Integral] | Tuple3 | Tuple4]
OrderTypes = Optional[OrderTypesWithoutNone]
RngTypes = Optional[Union[Generator, Integral]]


def FittedModelTypes() -> tuple:
    """
    Return a tuple of fitted model types for use in isinstance checks.

    Returns
    -------
        tuple: A tuple containing the result wrapper types for fitted models.
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
