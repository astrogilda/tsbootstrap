"""
Performance utilities: Future capability for backend benchmarking.

This module will provide performance measurement and benchmarking utilities
for comparing backend implementations. Currently a stub implementation.

The performance utilities will eventually enable:
- Backend performance benchmarking
- Memory usage profiling
- Scaling characteristic analysis
- Performance regression detection
"""

from typing import Any, Dict, List

import numpy as np


def benchmark_backend(backend: str, model_type: str, data: np.ndarray, **kwargs: Any) -> float:
    """Benchmark backend performance.

    Parameters
    ----------
    backend : str
        Backend to benchmark
    model_type : str
        Type of model
    data : np.ndarray
        Time series data
    **kwargs
        Model parameters

    Returns
    -------
    float
        Execution time in seconds
    """
    _not_implemented_msg = (
        "benchmark_backend is a planned feature that is not yet implemented. "
        "This stub exists to maintain test structure for future development."
    )
    raise NotImplementedError(_not_implemented_msg)


def measure_memory_usage(backend: str, model_type: str, data_size: int, **kwargs: Any) -> float:
    """Measure memory usage of backend.

    Parameters
    ----------
    backend : str
        Backend to measure
    model_type : str
        Type of model
    data_size : int
        Size of data to test
    **kwargs
        Model parameters

    Returns
    -------
    float
        Memory usage in MB
    """
    _not_implemented_msg = (
        "measure_memory_usage is a planned feature that is not yet implemented. "
        "This stub exists to maintain test structure for future development."
    )
    raise NotImplementedError(_not_implemented_msg)


def measure_scaling(
    backend: str, model_type: str, data_sizes: List[int], **kwargs: Any
) -> Dict[str, List[float]]:
    """Measure scaling characteristics.

    Parameters
    ----------
    backend : str
        Backend to measure
    model_type : str
        Type of model
    data_sizes : List[int]
        Sizes to test
    **kwargs
        Model parameters

    Returns
    -------
    Dict[str, List[float]]
        Scaling results with 'sizes' and 'times' keys
    """
    _not_implemented_msg = (
        "measure_scaling is a planned feature that is not yet implemented. "
        "This stub exists to maintain test structure for future development."
    )
    raise NotImplementedError(_not_implemented_msg)
