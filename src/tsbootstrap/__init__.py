"""Time Series Bootstrap package.

We provide a comprehensive suite of bootstrapping methods for time series analysis,
designed to handle the unique challenges of temporal dependencies and non-stationarity.
Our implementation emphasizes both computational efficiency and statistical rigor,
offering researchers and practitioners a flexible toolkit for uncertainty quantification
in time series modeling.

The package architecture follows a modular design where we separate concerns between
core bootstrapping algorithms, block generation strategies, and model interfaces.
This separation allows us to compose different techniques while maintaining
consistent behavior across the library.
"""

from importlib.metadata import version
from typing import TYPE_CHECKING

__version__ = version("tsbootstrap")

# We import only the most essential classes eagerly to minimize startup time.
# The BaseTimeSeriesBootstrap provides our foundational interface, while
# BootstrapFactory offers a convenient entry point for users who prefer
# configuration-based initialization over direct class instantiation.
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


# Our lazy import mapping allows us to defer loading heavyweight modules
# until they're actually needed. This dramatically improves import performance
# for users who only need a subset of our functionality. We organize imports
# by category to make the structure clear and maintainable.
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
    # Utilities
    "AutoOrderSelector": "utils",
    "RankLags": "ranklags",
    "TimeSeriesModel": "time_series_model",
    "TimeSeriesSimulator": "time_series_simulator",
}


def __getattr__(name):
    """Implement lazy loading to improve import performance.

    We intercept attribute access at the module level to defer imports until
    they're actually needed. This approach reduces initial import time from
    several seconds to milliseconds for typical use cases. Once loaded,
    we cache the imported objects to avoid repeated import overhead.

    The implementation handles both simple module imports and nested submodule
    access, though we currently keep our module structure flat for simplicity.
    """
    if name in _lazy_imports:
        import importlib

        module_path = _lazy_imports[name]
        if "." in module_path:
            # We handle potential future submodule imports, though our current
            # architecture keeps modules at a single level for clarity
            parts = module_path.split(".")
            module = importlib.import_module(f".{parts[0]}", package=__name__)
            for part in parts[1:]:
                module = getattr(module, part)
        else:
            module = importlib.import_module(f".{module_path}", package=__name__)

        # Extract the requested attribute from its containing module
        attr = getattr(module, name)

        # Cache the imported object to avoid repeated import costs
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
    "AutoOrderSelector",
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
