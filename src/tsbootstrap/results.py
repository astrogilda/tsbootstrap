"""Structured result objects returned by the public API.

A :class:`BootstrapResult` holds the stacked replicate arrays the engine wrote, is a
sequence of lazily built :class:`BootstrapSample` views, carries a
:class:`BootstrapRunMetadata` provenance record, and exposes the arrays (``values()``,
``indices()``) plus out-of-bag / in-bag primitives for downstream conformal-prediction use.
"""

from __future__ import annotations

import operator
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar, overload

import numpy as np
from numpy.typing import NDArray

from tsbootstrap.errors import OOBUnavailableError

_T = TypeVar("_T", bound="np.generic[Any]")


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
    dtype: str = "float64"
    versions: dict[str, str] = field(default_factory=dict)
    references: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    failed: bool = False
    failure_reason: str | None = None
    # Reconstruct with np.random.SeedSequence(**seed_state) to replay a run made
    # with a reused or spawned SeedSequence; entropy alone omits its spawn state.
    seed_state: dict[str, object] | None = None
    # The random stream the backend drew from, versioned: "pcg64/seedsequence" for the
    # numpy backend, tsbootstrap.prng_keys.STREAM_ID for the compiled backend.
    rng_stream: str | None = None


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
        Original-observation indices used, shape ``(n,)``, **dtype ``int32``**,
        when the method resamples observations (block/IID). ``None`` for
        recursive methods, which have no observation-index provenance. The
        ``int32`` dtype is contractual: the producer caps the series length below
        ``2**31`` observations so every index fits, and ``.indices()`` is
        guaranteed to return ``int32`` (not platform ``intp``).
    metadata : dict
        Optional per-sample detail (e.g. block starts/lengths).
    """

    values: NDArray[np.floating]
    sample_id: int
    indices: NDArray[np.int32] | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class BootstrapResult(Sequence[BootstrapSample]):
    """An ordered collection of bootstrap replicates, backed by one stacked array.

    The result holds the ``(n_bootstraps, n[, d])`` values array and the
    ``(n_bootstraps, n)`` ``int32`` index array the engine wrote, and nothing per
    replicate. :meth:`values` and :meth:`indices` return read-only views of those
    arrays (no copy); pass ``copy=True`` for a writeable copy. Indexing or iterating
    builds each :class:`BootstrapSample` on demand, as read-only views into the same
    arrays.
    """

    __slots__ = ("_indices", "_values", "metadata")

    def __init__(
        self,
        values: NDArray[np.floating],
        indices: NDArray[np.int32] | None,
        metadata: BootstrapRunMetadata,
    ) -> None:
        if indices is not None:
            if indices.shape[0] != values.shape[0]:
                raise ValueError(
                    f"values and indices disagree on the replicate count: "
                    f"{values.shape[0]} != {indices.shape[0]}"
                )
            indices = _read_only(indices.astype(np.int32, copy=False))
        self._values = _read_only(values)
        self._indices = indices
        self.metadata = metadata

    @classmethod
    def empty(cls, metadata: BootstrapRunMetadata) -> BootstrapResult:
        """A result with no replicates, as returned when preparation failed."""
        return cls(np.empty((0,), dtype=np.float64), None, metadata)

    def __len__(self) -> int:
        return int(self._values.shape[0])

    def _sample(self, i: int) -> BootstrapSample:
        idx = None if self._indices is None else self._indices[i]
        return BootstrapSample(values=self._values[i], sample_id=i, indices=idx)

    @overload
    def __getitem__(self, index: int) -> BootstrapSample: ...

    @overload
    def __getitem__(self, index: slice) -> list[BootstrapSample]: ...

    def __getitem__(self, index: int | slice) -> BootstrapSample | list[BootstrapSample]:
        n = len(self)
        if isinstance(index, slice):
            return [self._sample(i) for i in range(*index.indices(n))]
        i = operator.index(index)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError(f"replicate index {index} out of range for {n} replicates")
        return self._sample(i)

    def __iter__(self) -> Iterator[BootstrapSample]:
        return (self._sample(i) for i in range(len(self)))

    def iter_samples(self) -> Iterator[BootstrapSample]:
        """Iterate over the individual :class:`BootstrapSample` objects."""
        return iter(self)

    def values(self, *, copy: bool = False) -> NDArray[np.floating]:
        """The replicate values, shape ``(n_bootstraps, n[, d])``.

        Returns a read-only view of the stored array; ``copy=True`` returns a writeable
        copy. A failed run (no replicates) returns an empty ``(0,)`` array.
        """
        return self._values.copy() if copy else self._values

    def indices(self, *, copy: bool = False) -> NDArray[np.int32] | None:
        """Observation indices (``int32``), shape ``(n_bootstraps, n)``, or ``None``.

        A read-only view of the stored array (``copy=True`` for a writeable copy). The
        array is ``int32`` end to end: the engines emit ``int32`` indices and the producer
        caps the series length below ``2**31``. Recursive methods (no observation-index
        provenance) return ``None``.
        """
        if self._indices is None:
            return None
        return self._indices.copy() if copy else self._indices

    def inbag_counts(self) -> NDArray[np.intp]:
        """How many times each original observation appears per replicate.

        Shape ``(n_bootstraps, n_obs)``, computed with one ``bincount`` over row-offset
        indices. Raises :class:`OOBUnavailableError` for methods without
        observation-index provenance.
        """
        idx = self._indices
        if idx is None:
            raise OOBUnavailableError(
                f"in-bag/out-of-bag counts require observation indices, which "
                f"method {self.metadata.method!r} does not produce",
                hint="Use an observation-resampling method (IID or a block method) for OOB.",
            )
        n_obs = self.metadata.n_obs
        n_rep = idx.shape[0]
        if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= n_obs):
            raise ValueError(f"observation indices fall outside [0, {n_obs})")
        offsets = np.arange(n_rep, dtype=np.intp)[:, None] * n_obs
        flat = (idx + offsets).ravel()
        counts = np.bincount(flat, minlength=n_rep * n_obs)
        return counts.reshape(n_rep, n_obs).astype(np.intp, copy=False)

    def get_oob_mask(self) -> NDArray[np.bool_]:
        """Boolean out-of-bag mask ``(n_bootstraps, n_obs)`` (True = never sampled)."""
        return self.inbag_counts() == 0

    def __repr__(self) -> str:
        return (
            f"BootstrapResult(method={self.metadata.method!r}, "
            f"n_bootstraps={len(self)}, "
            f"n_obs={self.metadata.n_obs}, n_series={self.metadata.n_series})"
        )


def _read_only(arr: NDArray[_T]) -> NDArray[_T]:
    """A read-only view of ``arr``; the caller's array keeps its own flags."""
    view = arr.view()
    view.flags.writeable = False
    return view


@dataclass(frozen=True, slots=True)
class ReducedResult:
    """Per-replicate statistics from :func:`~tsbootstrap.bootstrap_reduce`, plus provenance.

    ``statistics`` has shape ``(n_bootstraps, |theta|)``, the value of the per-replicate
    statistic for every replicate, or is ``None`` when the run failed preparation. Peak
    memory is ``O(B * |theta|)``, never the ``O(B * n * d)`` of the materialised paths, so
    very large ``n_bootstraps`` stays in RAM.
    """

    statistics: NDArray[np.floating] | None
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
        q: float | Sequence[float] | NDArray[np.floating],
        *,
        axis: int = 0,
    ) -> NDArray[np.floating]:
        """Exact quantile(s) over the ``n_bootstraps`` replicates."""
        if self.statistics is None:
            raise ValueError("the bootstrap run failed; there are no statistics to reduce")
        return np.quantile(self.statistics, q, axis=axis)

    def __len__(self) -> int:
        return 0 if self.statistics is None else int(self.statistics.shape[0])


__all__ = ["BootstrapRunMetadata", "BootstrapSample", "BootstrapResult", "ReducedResult"]
