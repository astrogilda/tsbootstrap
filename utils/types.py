from typing import List, Union, Optional, Literal, Tuple
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper
from statsmodels.tsa.arima.model import ARIMAResultsWrapper
from arch.univariate.base import ARCHModelResult
from statsmodels.tsa.ar_model import AutoRegResultsWrapper
from numpy.random import Generator
from numbers import Integral

ModelTypes = Literal["ar", "arima", "sarima", "var", "arch"]

FittedModelType = Union[
    AutoRegResultsWrapper,
    ARIMAResultsWrapper,
    SARIMAXResultsWrapper,
    VARResultsWrapper,
    ARCHModelResult,
]

OrderTypes = Optional[Union[Integral, List[Integral],
                            Tuple[Integral, Integral, Integral], Tuple[Integral, Integral, Integral, Integral]]]

OrderTypesWithoutNone = Union[Integral, List[Integral],
                              Tuple[Integral, Integral, Integral], Tuple[Integral, Integral, Integral, Integral]]

RngTypes = Optional[Union[Generator, Integral]]

BlockCompressorTypes = Literal["first", "middle", "last",
                               "mean", "mode", "median", "kmeans", "kmedians", "kmedoids"]
