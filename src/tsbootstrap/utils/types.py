from numbers import Integral
from typing import Literal

from numpy.random import Generator

ModelTypesWithoutArch = Literal["ar", "arima", "sarima", "var"]

ModelTypes = Literal["ar", "arima", "sarima", "var", "arch"]


def FittedModelTypes():
    from arch.univariate.base import ARCHModelResult
    from statsmodels.tsa.ar_model import AutoRegResultsWrapper
    from statsmodels.tsa.arima.model import ARIMAResultsWrapper
    from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
    from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

    fmt = (AutoRegResultsWrapper
        | ARIMAResultsWrapper
        | SARIMAXResultsWrapper
        | VARResultsWrapper
        | ARCHModelResult
    )
    return fmt

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
