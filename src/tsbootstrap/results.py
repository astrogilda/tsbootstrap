"""Structured result objects returned by the public API.

A :class:`BootstrapResult` is a sequence of :class:`BootstrapSample`, carries a
:class:`BootstrapRunMetadata` provenance record, and exposes vectorised views
(``values()``, ``indices()``) plus out-of-bag / in-bag primitives for downstream
conformal-prediction use.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import overload

import numpy as np
from numpy.typing import NDArray

from tsbootstrap.errors import OOBUnavailableError


@dataclass(frozen=True, slots=True)
class BootstrapRunMetadata:
    """Provenance for a bootstrap run; everything needed to reproduce or cite it."""

    method: str
    method_params: dict[str, object]
    n_bootstraps: int
    n_obs: int
    n_series: int
    random_state_kind: str
    seed_entropy: int | Sequence[int] | None
    backend: str | None = None
    versions: dict[str, str] = field(default_factory=dict)
    references: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    failed: bool = False
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class BootstrapSample:
    """One bootstrap replicate.

    Attributes
    ----------
    values : ndarray
        The resampled/regenerated series, shape ``(n,)`` or ``(n, d)``.
    sample_id : int
        Replicate index ``i`` (also identifies the RNG stream that produced it).
    indices : ndarray or None
        Original-observation indices used, shape ``(n,)``, when the method
        resamples observations (block/IID). ``None`` for recursive methods,
        which have no observation-index provenance.
    metadata : dict
        Optional per-sample detail (e.g. block starts/lengths).
    """

    values: NDArray[np.float64]
    sample_id: int
    indices: NDArray[np.intp] | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class BootstrapResult(Sequence[BootstrapSample]):
    """An ordered, materialised collection of bootstrap samples plus metadata."""

    __slots__ = ("_samples", "metadata")

    def __init__(self, samples: Iterable[BootstrapSample], metadata: BootstrapRunMetadata) -> None:
        self._samples: list[BootstrapSample] = list(samples)
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self._samples)

    @overload
    def __getitem__(self, index: int) -> BootstrapSample: ...

    @overload
    def __getitem__(self, index: slice) -> list[BootstrapSample]: ...

    def __getitem__(self, index: int | slice) -> BootstrapSample | list[BootstrapSample]:
        return self._samples[index]

    def __iter__(self) -> Iterator[BootstrapSample]:
        return iter(self._samples)

    def iter_samples(self) -> Iterator[BootstrapSample]:
        """Iterate over the individual :class:`BootstrapSample` objects."""
        return iter(self._samples)

    def values(self) -> NDArray[np.float64]:
        """Stack the samples into one array, shape ``(n_bootstraps, n[, d])``."""
        if not self._samples:
            return np.empty((0,), dtype=np.float64)
        return np.stack([s.values for s in self._samples], axis=0)

    def indices(self) -> NDArray[np.intp] | None:
        """Stacked observation indices, or ``None`` if any sample lacks them (recursive)."""
        per_sample = [s.indices for s in self._samples]
        if any(idx is None for idx in per_sample):
            return None
        present = [idx for idx in per_sample if idx is not None]
        return np.stack(present, axis=0).astype(np.intp, copy=False)

    def inbag_counts(self) -> NDArray[np.intp]:
        """How many times each original observation appears per replicate.

        Shape ``(n_bootstraps, n_obs)``. Raises :class:`OOBUnavailableError` for
        methods without observation-index provenance.
        """
        idx = self.indices()
        if idx is None:
            raise OOBUnavailableError(
                f"in-bag/out-of-bag counts require observation indices, which "
                f"method {self.metadata.method!r} does not produce",
                hint="Use an observation-resampling method (IID or a block method) for OOB.",
            )
        n_obs = self.metadata.n_obs
        counts = np.empty((idx.shape[0], n_obs), dtype=np.intp)
        for b in range(idx.shape[0]):
            counts[b] = np.bincount(idx[b], minlength=n_obs)[:n_obs]
        return counts

    def get_oob_mask(self) -> NDArray[np.bool_]:
        """Boolean out-of-bag mask ``(n_bootstraps, n_obs)`` (True = never sampled)."""
        return self.inbag_counts() == 0

    def __repr__(self) -> str:
        return (
            f"BootstrapResult(method={self.metadata.method!r}, "
            f"n_bootstraps={len(self._samples)}, "
            f"n_obs={self.metadata.n_obs}, n_series={self.metadata.n_series})"
        )


@dataclass(frozen=True, slots=True)
class ReducedResult:
    """Per-replicate statistics from :func:`~tsbootstrap.bootstrap_reduce`, plus provenance.

    ``statistics`` has shape ``(n_bootstraps, |theta|)`` — the value of the per-replicate
    statistic for every replicate — or is ``None`` when the run failed preparation. Peak
    memory is ``O(B * |theta|)``, never the ``O(B * n * d)`` of the materialised paths, so
    very large ``n_bootstraps`` stays in RAM.
    """

    statistics: NDArray[np.float64] | None
    metadata: BootstrapRunMetadata

    @property
    def failed(self) -> bool:
        """Whether preparation failed (e.g. a non-stationary fit under ``stability_policy='skip'``)."""
        return self.metadata.failed

    @property
    def failure_reason(self) -> str | None:
        """Human-readable reason when :attr:`failed`, else ``None``."""
        return self.metadata.failure_reason

    def quantile(
        self,
        q: float | Sequence[float] | NDArray[np.float64],
        *,
        axis: int = 0,
    ) -> NDArray[np.float64]:
        """Exact quantile(s) over the ``n_bootstraps`` replicates."""
        if self.statistics is None:
            raise ValueError("the bootstrap run failed; there are no statistics to reduce")
        return np.quantile(self.statistics, q, axis=axis)

    def __len__(self) -> int:
        return 0 if self.statistics is None else int(self.statistics.shape[0])


__all__ = ["BootstrapRunMetadata", "BootstrapSample", "BootstrapResult", "ReducedResult"]
