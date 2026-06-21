"""sktime / skbase adapter classes over the functional ``bootstrap()`` core.

These are thin ``skbase.BaseObject`` estimators (discoverable by sktime and
validated by ``check_estimator``) that delegate to the pure functional core. The
core stays stateless; all estimator state lives here, so non-sktime users and
future accelerated backends use ``bootstrap()`` directly without the OOP layer.
"""

from __future__ import annotations

from tsbootstrap.adapters.estimators import (
    ARIMAResidualBootstrap,
    ARResidualBootstrap,
    BaseTimeSeriesBootstrap,
    CircularBlockBootstrap,
    IIDBootstrap,
    MovingBlockBootstrap,
    NonOverlappingBlockBootstrap,
    SieveBootstrap,
    StationaryBlockBootstrap,
    TaperedBlockBootstrap,
    VARResidualBootstrap,
)

__all__ = [
    "BaseTimeSeriesBootstrap",
    "IIDBootstrap",
    "MovingBlockBootstrap",
    "CircularBlockBootstrap",
    "StationaryBlockBootstrap",
    "NonOverlappingBlockBootstrap",
    "TaperedBlockBootstrap",
    "ARResidualBootstrap",
    "ARIMAResidualBootstrap",
    "VARResidualBootstrap",
    "SieveBootstrap",
]
