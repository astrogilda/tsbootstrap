from numbers import Integral
import sys
from typing import Any, Literal

from numpy.random import Generator
from packaging.specifiers import SpecifierSet

ModelTypesWithoutArch = Literal["ar", "arima", "sarima", "var"]

ModelTypes = Literal["ar", "arima", "sarima", "var", "arch"]


sys_version = sys.version.split(" ")[0]
new_typing_available = sys_version in SpecifierSet(">=3.10")


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

if new_typing_available:
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
else:
    OrderTypesWithoutNone = Any
    OrderTypes = Any
    RngTypes = Any
    BlockCompressorTypes = [
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
