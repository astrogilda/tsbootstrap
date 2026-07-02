"""Concrete sktime/skbase bootstrap adapters over the functional core.

Each class is a thin ``skbase.BaseObject`` that stores its parameters, builds a
:class:`~tsbootstrap.methods.MethodSpec`, and delegates generation to
:func:`tsbootstrap.bootstrap`. The shared base holds the delegation logic so the
concrete classes carry only their parameters (shape "concrete over a shared base").
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from skbase.base import BaseObject

from tsbootstrap.api import bootstrap as _bootstrap
from tsbootstrap.methods import (
    AR,
    ARIMA,
    IID,
    VAR,
    BlockLength,
    BlockWild,
    CircularBlock,
    Innovation,
    MethodSpec,
    MovingBlock,
    NonOverlappingBlock,
    ResidualBootstrap,
    SieveAR,
    StationaryBlock,
    TaperedBlock,
    Wild,
)

_Sample = NDArray[np.floating]


class BaseTimeSeriesBootstrap(BaseObject):
    """Base adapter: delegate generation to the functional ``bootstrap()``."""

    _tags = {
        "object_type": "bootstrap",
        "bootstrap_type": "other",
        "capability:multivariate": False,
    }

    def __init__(self, n_bootstraps: int = 999, random_state: int | None = None) -> None:
        self.n_bootstraps = n_bootstraps
        self.random_state = random_state
        super().__init__()

    def _make_spec(self) -> MethodSpec:
        raise NotImplementedError("concrete adapters must implement _make_spec")

    def bootstrap(
        self, X: object, y: object = None, return_indices: bool = False
    ) -> Iterator[_Sample] | Iterator[tuple[_Sample, NDArray[np.int32] | None]]:
        """Yield ``n_bootstraps`` bootstrap samples of ``X`` (optionally with indices)."""
        result = _bootstrap(
            X,
            method=self._make_spec(),
            n_bootstraps=self.n_bootstraps,
            random_state=self.random_state,
        )
        # sktime convention: the yielded element type depends on the return_indices
        # flag, expressed as a union of iterators rather than an iterator of a union.
        for sample in result:
            yield (  # pyright: ignore[reportReturnType]
                (sample.values, sample.indices) if return_indices else sample.values
            )

    def get_n_bootstraps(self) -> int:
        """Number of bootstrap replicates this estimator generates."""
        return self.n_bootstraps

    @classmethod
    def get_test_params(cls) -> list[dict[str, Any]]:
        return []


class IIDBootstrap(BaseTimeSeriesBootstrap):
    """i.i.d. resampling (baseline; assumes no serial dependence)."""

    _tags = {
        **BaseTimeSeriesBootstrap._tags,
        "bootstrap_type": "iid",
        "capability:multivariate": True,
    }

    def _make_spec(self) -> MethodSpec:
        return IID()

    @classmethod
    def get_test_params(cls) -> list[dict[str, Any]]:
        return [{"n_bootstraps": 10}, {"n_bootstraps": 5, "random_state": 0}]


class MovingBlockBootstrap(BaseTimeSeriesBootstrap):
    """Moving block bootstrap (overlapping fixed-length blocks)."""

    _tags = {
        **BaseTimeSeriesBootstrap._tags,
        "bootstrap_type": "block",
        "capability:multivariate": True,
    }

    def __init__(
        self,
        block_length: BlockLength = "auto",
        n_bootstraps: int = 999,
        random_state: int | None = None,
    ) -> None:
        self.block_length: BlockLength = block_length
        super().__init__(n_bootstraps=n_bootstraps, random_state=random_state)

    def _make_spec(self) -> MethodSpec:
        return MovingBlock(block_length=self.block_length)

    @classmethod
    def get_test_params(cls) -> list[dict[str, Any]]:
        return [
            {"block_length": 5, "n_bootstraps": 10},
            {"block_length": "auto", "n_bootstraps": 5},
        ]


class CircularBlockBootstrap(BaseTimeSeriesBootstrap):
    """Circular block bootstrap (wrap-around blocks)."""

    _tags = {
        **BaseTimeSeriesBootstrap._tags,
        "bootstrap_type": "block",
        "capability:multivariate": True,
    }

    def __init__(
        self,
        block_length: BlockLength = "auto",
        n_bootstraps: int = 999,
        random_state: int | None = None,
    ) -> None:
        self.block_length: BlockLength = block_length
        super().__init__(n_bootstraps=n_bootstraps, random_state=random_state)

    def _make_spec(self) -> MethodSpec:
        return CircularBlock(block_length=self.block_length)

    @classmethod
    def get_test_params(cls) -> list[dict[str, Any]]:
        return [
            {"block_length": 5, "n_bootstraps": 10},
            {"block_length": "auto", "n_bootstraps": 5},
        ]


class StationaryBlockBootstrap(BaseTimeSeriesBootstrap):
    """Stationary bootstrap (Politis-Romano; geometric block lengths)."""

    _tags = {
        **BaseTimeSeriesBootstrap._tags,
        "bootstrap_type": "block",
        "capability:multivariate": True,
    }

    def __init__(
        self,
        avg_block_length: BlockLength = "auto",
        n_bootstraps: int = 999,
        random_state: int | None = None,
    ) -> None:
        self.avg_block_length: BlockLength = avg_block_length
        super().__init__(n_bootstraps=n_bootstraps, random_state=random_state)

    def _make_spec(self) -> MethodSpec:
        return StationaryBlock(avg_block_length=self.avg_block_length)

    @classmethod
    def get_test_params(cls) -> list[dict[str, Any]]:
        return [
            {"avg_block_length": 5, "n_bootstraps": 10},
            {"avg_block_length": "auto", "n_bootstraps": 5},
        ]


class NonOverlappingBlockBootstrap(BaseTimeSeriesBootstrap):
    """Non-overlapping block bootstrap (Carlstein)."""

    _tags = {
        **BaseTimeSeriesBootstrap._tags,
        "bootstrap_type": "block",
        "capability:multivariate": True,
    }

    def __init__(
        self,
        block_length: BlockLength = "auto",
        n_bootstraps: int = 999,
        random_state: int | None = None,
    ) -> None:
        self.block_length: BlockLength = block_length
        super().__init__(n_bootstraps=n_bootstraps, random_state=random_state)

    def _make_spec(self) -> MethodSpec:
        return NonOverlappingBlock(block_length=self.block_length)

    @classmethod
    def get_test_params(cls) -> list[dict[str, Any]]:
        return [
            {"block_length": 5, "n_bootstraps": 10},
            {"block_length": "auto", "n_bootstraps": 5},
        ]


class TaperedBlockBootstrap(BaseTimeSeriesBootstrap):
    """Tapered block bootstrap (energy-normalized window)."""

    _tags = {
        **BaseTimeSeriesBootstrap._tags,
        "bootstrap_type": "block",
        "capability:multivariate": True,
    }

    def __init__(
        self,
        window: Literal["bartlett", "blackman", "hamming", "hann", "tukey"] = "bartlett",
        block_length: BlockLength = "auto",
        alpha: float = 0.5,
        n_bootstraps: int = 999,
        random_state: int | None = None,
    ) -> None:
        self.window: Literal["bartlett", "blackman", "hamming", "hann", "tukey"] = window
        self.block_length: BlockLength = block_length
        self.alpha = alpha
        super().__init__(n_bootstraps=n_bootstraps, random_state=random_state)

    def _make_spec(self) -> MethodSpec:
        return TaperedBlock(window=self.window, block_length=self.block_length, alpha=self.alpha)

    @classmethod
    def get_test_params(cls) -> list[dict[str, Any]]:
        return [{"window": "hann", "block_length": 5, "n_bootstraps": 10}]


class ARResidualBootstrap(BaseTimeSeriesBootstrap):
    """Recursive AR residual bootstrap."""

    _tags = {
        **BaseTimeSeriesBootstrap._tags,
        "bootstrap_type": "model",
        "capability:multivariate": False,
    }

    def __init__(
        self,
        order: int = 1,
        innovation: Innovation | None = None,
        n_bootstraps: int = 999,
        random_state: int | None = None,
    ) -> None:
        self.order = order
        self.innovation = innovation
        super().__init__(n_bootstraps=n_bootstraps, random_state=random_state)

    def _make_spec(self) -> MethodSpec:
        return ResidualBootstrap(
            model=AR(order=self.order),
            innovation=self.innovation if self.innovation is not None else IID(),
        )

    @classmethod
    def get_test_params(cls) -> list[dict[str, Any]]:
        return [
            {"order": 1, "n_bootstraps": 10},
            {"order": 2, "n_bootstraps": 5},
            {"order": 1, "innovation": Wild(), "n_bootstraps": 5},
            {"order": 1, "innovation": BlockWild(block_length=5), "n_bootstraps": 5},
        ]


class ARIMAResidualBootstrap(BaseTimeSeriesBootstrap):
    """Recursive ARIMA residual bootstrap (differenced scale)."""

    _tags = {
        **BaseTimeSeriesBootstrap._tags,
        "bootstrap_type": "model",
        "capability:multivariate": False,
    }

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        innovation: Innovation | None = None,
        n_bootstraps: int = 999,
        random_state: int | None = None,
    ) -> None:
        self.order = order
        self.innovation = innovation
        super().__init__(n_bootstraps=n_bootstraps, random_state=random_state)

    def _make_spec(self) -> MethodSpec:
        return ResidualBootstrap(
            model=ARIMA(order=self.order),
            innovation=self.innovation if self.innovation is not None else IID(),
        )

    @classmethod
    def get_test_params(cls) -> list[dict[str, Any]]:
        return [{"order": (1, 1, 1), "n_bootstraps": 10}]


class VARResidualBootstrap(BaseTimeSeriesBootstrap):
    """Recursive VAR residual bootstrap (multivariate)."""

    _tags = {
        **BaseTimeSeriesBootstrap._tags,
        "bootstrap_type": "model",
        "capability:multivariate": True,
    }

    def __init__(
        self,
        order: int = 1,
        innovation: Innovation | None = None,
        n_bootstraps: int = 999,
        random_state: int | None = None,
    ) -> None:
        self.order = order
        self.innovation = innovation
        super().__init__(n_bootstraps=n_bootstraps, random_state=random_state)

    def _make_spec(self) -> MethodSpec:
        return ResidualBootstrap(
            model=VAR(order=self.order),
            innovation=self.innovation if self.innovation is not None else IID(),
        )

    @classmethod
    def get_test_params(cls) -> list[dict[str, Any]]:
        return [{"order": 1, "n_bootstraps": 10}]


class SieveBootstrap(BaseTimeSeriesBootstrap):
    """Sieve bootstrap (AR order selected on the original series)."""

    _tags = {
        **BaseTimeSeriesBootstrap._tags,
        "bootstrap_type": "model",
        "capability:multivariate": False,
    }

    def __init__(
        self,
        min_lag: int = 1,
        max_lag: int | None = None,
        criterion: Literal["aic", "bic", "hqic"] = "bic",
        innovation: Innovation | None = None,
        n_bootstraps: int = 999,
        random_state: int | None = None,
    ) -> None:
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.criterion: Literal["aic", "bic", "hqic"] = criterion
        self.innovation = innovation
        super().__init__(n_bootstraps=n_bootstraps, random_state=random_state)

    def _make_spec(self) -> MethodSpec:
        return SieveAR(
            min_lag=self.min_lag,
            max_lag=self.max_lag,
            criterion=self.criterion,
            innovation=self.innovation if self.innovation is not None else IID(),
        )

    @classmethod
    def get_test_params(cls) -> list[dict[str, Any]]:
        return [{"n_bootstraps": 10}]


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
