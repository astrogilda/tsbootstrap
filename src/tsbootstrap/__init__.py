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

from tsbootstrap.api import (
    bootstrap,
    bootstrap_iter,
    bootstrap_reduce,
    bootstrap_reduce_panel,
)
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
from tsbootstrap.metadata import MethodMetadata, metadata_for
from tsbootstrap.methods import (
    AR,
    ARIMA,
    IID,
    VAR,
    BaseMethodSpec,
    BlockWild,
    CircularBlock,
    MethodSpec,
    MovingBlock,
    NonOverlappingBlock,
    ResidualBootstrap,
    SieveAR,
    StationaryBlock,
    TaperedBlock,
    Wild,
)
from tsbootstrap.results import (
    BootstrapResult,
    BootstrapRunMetadata,
    BootstrapSample,
    ReducedResult,
)

# Uncertainty quantification. These are safe to import on a core-only install:
# the UQ modules are numpy-only at import time, and scikit-learn (the ``uq`` extra)
# is imported lazily inside the out-of-bag path, raising a clear BackendError only
# when an EnbPI ensemble is actually fitted without it.
from tsbootstrap.uq import (
    EnbPIEnsemble,
    aci_halfwidths,
    basic_interval,
    bca_interval,
    block_jackknife_se,
    conf_int,
    conf_int_panel,
    enbpi_intervals,
    fit_predict_oob,
    forecast_intervals,
    jackknife_acceleration,
    jackknife_statistics,
    nexcp_quantile,
    percentile_interval,
    sliding_window_halfwidths,
    static_halfwidths,
    studentized_interval,
)

try:
    __version__ = _version("tsbootstrap")
except PackageNotFoundError:  # pragma: no cover - editable/uninstalled
    __version__ = "0.2.0.dev0"

__all__ = [
    "__version__",
    # entry point
    "bootstrap",
    "bootstrap_iter",
    "bootstrap_reduce",
    "bootstrap_reduce_panel",
    # diagnostics and method introspection
    "diagnose",
    "Diagnosis",
    "metadata_for",
    "MethodMetadata",
    # method specifications
    "IID",
    "MovingBlock",
    "CircularBlock",
    "StationaryBlock",
    "NonOverlappingBlock",
    "TaperedBlock",
    "Wild",
    "BlockWild",
    "AR",
    "ARIMA",
    "VAR",
    "ResidualBootstrap",
    "SieveAR",
    "BaseMethodSpec",
    "MethodSpec",
    # results
    "BootstrapResult",
    "BootstrapSample",
    "BootstrapRunMetadata",
    "ReducedResult",
    # uncertainty quantification
    "EnbPIEnsemble",
    "enbpi_intervals",
    "fit_predict_oob",
    "forecast_intervals",
    "aci_halfwidths",
    "nexcp_quantile",
    "static_halfwidths",
    "sliding_window_halfwidths",
    "percentile_interval",
    "basic_interval",
    "jackknife_statistics",
    "block_jackknife_se",
    "studentized_interval",
    "jackknife_acceleration",
    "bca_interval",
    "conf_int",
    "conf_int_panel",
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
