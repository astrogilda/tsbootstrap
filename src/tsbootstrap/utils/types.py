from numbers import Integral
from typing import Literal

from arch.univariate.base import ARCHModelResult
from numpy.random import Generator
from statsmodels.tsa.ar_model import AutoRegResultsWrapper
from statsmodels.tsa.arima.model import ARIMAResultsWrapper
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

ModelTypesWithoutArch = Literal["ar", "arima", "sarima", "var"]

ModelTypes = Literal["ar", "arima", "sarima", "var", "arch"]

FittedModelTypes = (
    AutoRegResultsWrapper
    | ARIMAResultsWrapper
    | SARIMAXResultsWrapper
    | VARResultsWrapper
    | ARCHModelResult
)

OrderTypesWithoutNone = (
    Integral
    | list[Integral]
    | tuple[Integral, Integral, Integral]
    | tuple[Integral, Integral, Integral, Integral]
)

OrderTypes = None | OrderTypesWithoutNone

RngTypes = None | Generator | Integral

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
