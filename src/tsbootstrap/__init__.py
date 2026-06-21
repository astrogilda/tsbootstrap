"""tsbootstrap: correct, typed time-series bootstrap methods.

The public surface is one function, :func:`bootstrap`, configured with a typed
method specification:

    from tsbootstrap import bootstrap, MovingBlock
    result = bootstrap(x, method=MovingBlock(block_length="auto"), n_bootstraps=999)

See :mod:`tsbootstrap.methods` for the available specifications and
:mod:`tsbootstrap.results` for the structured return type.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from tsbootstrap.api import bootstrap, bootstrap_reduce
from tsbootstrap.diagnostics import Diagnosis, diagnose
from tsbootstrap.errors import (
    BackendError,
    Codes,
    DegenerateBlockBootstrapWarning,
    InputDataError,
    MethodConfigError,
    ModelStabilityError,
    NearUnitRootWarning,
    OOBUnavailableError,
    RNGContractError,
    TSBootstrapError,
    TSBootstrapWarning,
)
from tsbootstrap.methods import (
    AR,
    ARIMA,
    IID,
    VAR,
    CircularBlock,
    MethodSpec,
    MovingBlock,
    NonOverlappingBlock,
    ResidualBootstrap,
    SieveAR,
    StationaryBlock,
    TaperedBlock,
)
from tsbootstrap.results import (
    BootstrapResult,
    BootstrapRunMetadata,
    BootstrapSample,
    ReducedResult,
)

try:
    __version__ = _version("tsbootstrap")
except PackageNotFoundError:  # pragma: no cover - editable/uninstalled
    __version__ = "0.2.0.dev0"

__all__ = [
    "__version__",
    # entry point
    "bootstrap",
    "bootstrap_reduce",
    # diagnostics
    "diagnose",
    "Diagnosis",
    # method specifications
    "IID",
    "MovingBlock",
    "CircularBlock",
    "StationaryBlock",
    "NonOverlappingBlock",
    "TaperedBlock",
    "AR",
    "ARIMA",
    "VAR",
    "ResidualBootstrap",
    "SieveAR",
    "MethodSpec",
    # results
    "BootstrapResult",
    "BootstrapSample",
    "BootstrapRunMetadata",
    "ReducedResult",
    # errors and warnings
    "Codes",
    "TSBootstrapError",
    "InputDataError",
    "MethodConfigError",
    "ModelStabilityError",
    "BackendError",
    "OOBUnavailableError",
    "RNGContractError",
    "TSBootstrapWarning",
    "NearUnitRootWarning",
    "DegenerateBlockBootstrapWarning",
]
