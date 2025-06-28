"""
Block bootstrap module for time series bootstrap methods.

This module provides various block bootstrap methods for time series data.
"""

# Import only the most essential classes eagerly
from tsbootstrap.block_bootstrap.base import BaseBlockBootstrap, BlockBootstrap
from tsbootstrap.block_bootstrap.registry import (
    BLOCK_BOOTSTRAP_TYPES_DICT,
    get_bootstrap_types_dict,
)

# Lazy import mapping
_lazy_imports = {
    "BartlettsBootstrap": "bartletts",
    "BlackmanBootstrap": "blackman",
    "CircularBlockBootstrap": "circular",
    "HammingBootstrap": "hamming",
    "HanningBootstrap": "hanning",
    "MovingBlockBootstrap": "moving",
    "NonOverlappingBlockBootstrap": "non_overlapping",
    "StationaryBlockBootstrap": "stationary",
    "TukeyBootstrap": "tukey",
}


def __getattr__(name):
    """Lazy loading of block bootstrap implementations."""
    if name in _lazy_imports:
        import importlib

        module = importlib.import_module(f".{_lazy_imports[name]}", package=__name__)
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseBlockBootstrap",
    "BlockBootstrap",
    "MovingBlockBootstrap",
    "StationaryBlockBootstrap",
    "CircularBlockBootstrap",
    "NonOverlappingBlockBootstrap",
    "BartlettsBootstrap",
    "HammingBootstrap",
    "HanningBootstrap",
    "BlackmanBootstrap",
    "TukeyBootstrap",
    "BLOCK_BOOTSTRAP_TYPES_DICT",
    "get_bootstrap_types_dict",
]
