from typing import List, Union, Optional, Literal, Dict, Any
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper
from statsmodels.tsa.arima.model import ARIMAResultsWrapper
from arch.univariate.base import ARCHModelResult
from statsmodels.tsa.ar_model import AutoRegResultsWrapper

ModelTypes = Literal["ar", "arima", "sarima", "var", "arch"]
FittedModelType = Union[
    AutoRegResultsWrapper,
    ARIMAResultsWrapper,
    SARIMAXResultsWrapper,
    VARResultsWrapper,
    ARCHModelResult,
]
