"""Time Series Bootstrap package."""

from importlib.metadata import version
from typing import TYPE_CHECKING

__version__ = version("tsbootstrap")

# Import only the most essential classes eagerly
from .base_bootstrap import BaseTimeSeriesBootstrap
from .bootstrap_factory import BootstrapFactory

if TYPE_CHECKING:
    from .bootstrap import (
        BlockResidualBootstrap as BlockResidualBootstrap,
    )
    from .bootstrap import (
        BlockSieveBootstrap as BlockSieveBootstrap,
    )
    from .bootstrap import (
        WholeResidualBootstrap as WholeResidualBootstrap,
    )
    from .bootstrap import (
        WholeSieveBootstrap as WholeSieveBootstrap,
    )


# Lazy import implementation
_lazy_imports = {
    # Async bootstrap classes
    "AsyncBootstrap": "async_bootstrap",
    "AsyncBlockResidualBootstrap": "async_bootstrap",
    "AsyncWholeResidualBootstrap": "async_bootstrap",
    "AsyncWholeSieveBootstrap": "async_bootstrap",
    "DynamicAsyncBootstrap": "async_bootstrap",
    # Block bootstrap classes
    "BartlettsBootstrap": "block_bootstrap",
    "BaseBlockBootstrap": "block_bootstrap",
    "BlackmanBootstrap": "block_bootstrap",
    "BlockBootstrap": "block_bootstrap",
    "CircularBlockBootstrap": "block_bootstrap",
    "HammingBootstrap": "block_bootstrap",
    "HanningBootstrap": "block_bootstrap",
    "MovingBlockBootstrap": "block_bootstrap",
    "NonOverlappingBlockBootstrap": "block_bootstrap",
    "StationaryBlockBootstrap": "block_bootstrap",
    "TukeyBootstrap": "block_bootstrap",
    # Block utilities
    "BlockGenerator": "block_generator",
    "BlockLengthSampler": "block_length_sampler",
    "BlockResampler": "block_resampler",
    # Bootstrap implementations
    "BlockResidualBootstrap": "bootstrap",
    "BlockSieveBootstrap": "bootstrap",
    "WholeResidualBootstrap": "bootstrap",
    "WholeSieveBootstrap": "bootstrap",
    # Extended bootstrap implementations
    "BlockDistributionBootstrap": "bootstrap_ext",
    "BlockMarkovBootstrap": "bootstrap_ext",
    "BlockStatisticPreservingBootstrap": "bootstrap_ext",
    "WholeDistributionBootstrap": "bootstrap_ext",
    "WholeMarkovBootstrap": "bootstrap_ext",
    "WholeStatisticPreservingBootstrap": "bootstrap_ext",
    # Markov sampler components
    "BlockCompressor": "markov_sampler",
    "MarkovSampler": "markov_sampler",
    "MarkovTransitionMatrixCalculator": "markov_sampler",
    # Model selection and utilities
    "TSFitBestLag": "model_selection",
    "RankLags": "ranklags",
    "TimeSeriesModel": "time_series_model",
    "TimeSeriesSimulator": "time_series_simulator",
    "TSFit": "tsfit.base",
}


def __getattr__(name):
    """Lazy loading of modules to improve import time."""
    if name in _lazy_imports:
        import importlib

        module_path = _lazy_imports[name]
        if "." in module_path:
            # Handle submodule imports like tsfit.base
            parts = module_path.split(".")
            module = importlib.import_module(f".{parts[0]}", package=__name__)
            for part in parts[1:]:
                module = getattr(module, part)
        else:
            module = importlib.import_module(f".{module_path}", package=__name__)

        # Get the actual class/function from the module
        attr = getattr(module, name)

        # Cache it for future use
        globals()[name] = attr
        return attr

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseTimeSeriesBootstrap",
    "BartlettsBootstrap",
    "BaseBlockBootstrap",
    "BlackmanBootstrap",
    "BlockBootstrap",
    "CircularBlockBootstrap",
    "HammingBootstrap",
    "HanningBootstrap",
    "MovingBlockBootstrap",
    "NonOverlappingBlockBootstrap",
    "StationaryBlockBootstrap",
    "TukeyBootstrap",
    "BlockGenerator",
    "BlockLengthSampler",
    "BlockResampler",
    "BlockResidualBootstrap",
    "WholeResidualBootstrap",
    "WholeSieveBootstrap",
    "BlockSieveBootstrap",
    "BlockCompressor",
    "MarkovSampler",
    "MarkovTransitionMatrixCalculator",
    "RankLags",
    "TimeSeriesModel",
    "TimeSeriesSimulator",
    "TSFit",
    "TSFitBestLag",
    # Factory and async classes
    "BootstrapFactory",
    "AsyncBootstrap",
    "AsyncWholeResidualBootstrap",
    "AsyncBlockResidualBootstrap",
    "AsyncWholeSieveBootstrap",
    "DynamicAsyncBootstrap",
    # Extended bootstrap implementations
    "WholeMarkovBootstrap",
    "BlockMarkovBootstrap",
    "WholeDistributionBootstrap",
    "BlockDistributionBootstrap",
    "WholeStatisticPreservingBootstrap",
    "BlockStatisticPreservingBootstrap",
]
