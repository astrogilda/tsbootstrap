"""Backend abstraction for time series models.

This module provides a protocol-based abstraction layer for different
time series modeling backends (statsmodels, statsforecast, etc.).
"""

from tsbootstrap.backends.adapter import BackendToStatsmodelsAdapter, fit_with_backend
from tsbootstrap.backends.factory import create_backend, get_backend_info
from tsbootstrap.backends.protocol import FittedModelBackend, ModelBackend
from tsbootstrap.backends.statsforecast_backend import (
    StatsForecastBackend,
    StatsForecastFittedBackend,
)
from tsbootstrap.backends.statsmodels_backend import StatsModelsBackend, StatsModelsFittedBackend

__all__ = [
    "BackendToStatsmodelsAdapter",
    "FittedModelBackend",
    "ModelBackend",
    "StatsForecastBackend",
    "StatsForecastFittedBackend",
    "StatsModelsBackend",
    "StatsModelsFittedBackend",
    "create_backend",
    "fit_with_backend",
    "get_backend_info",
]
