"""Spec-type + backend -> executor registries, and the numpy chunk driver.

Decouples the typed configuration (:mod:`tsbootstrap.methods`) from execution. Each
method registers its numeric kernel keyed by spec type; the entry point looks an
executor up by ``(spec_type, backend)`` and calls it with no backend branch of its
own. The registries live in their own module so engine modules can register without
importing the entry point (no import cycles).

There are two executor contracts, both producing all ``B`` replicates in one call and
owning their own RNG derivation from the run's root ``SeedSequence`` (Option D):

- a **values executor** ``(prepared, spec, root_ss, n_bootstraps, n_obs, sim_dtype)``
  returns ``(values (B, n[, d]), indices (B, n) int32 or None)``;
- a **reduce executor** additionally takes a :class:`ReduceRequest` and returns the
  ``(B, |theta|)`` statistics array without materialising the full sample.

The numpy backend registers ONE per-spec numeric kernel via
:func:`register_chunk_executor`; that kernel produces one CHUNK of replicates
from its replicate streams, and this module wraps it into both a numpy values executor
and a generic numpy reduce executor (the streaming fallback) by driving it over the
byte-budgeted chunk loop. A compiled/GPU backend instead registers a full-``B`` fused
kernel directly via :func:`register_values_executor` / :func:`register_reduce_executor`
under its own backend key. A future JAX/threefry backend registers a ``(spec, "jax")``
executor the same way, with no change to :mod:`tsbootstrap.api`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from tsbootstrap.errors import Codes, MethodConfigError
from tsbootstrap.results import IndexSource
from tsbootstrap.rng import ROW_GROUP_ELEMENTS, ReplicateStreams

# Chunking. Every numpy engine is chunk-invariant (each replicate's bytes depend only on its
# own stream; the VAR fallback runs the numba kernel's per-replicate accumulation order), so
# the chunk size is a memory decision, not a reproducibility one. A chunk holds about
# ``_CHUNK_BYTES`` of kernel output plus float64 working space, capped at ``_CHUNK_SIZE``
# replicates; the first chunk is sized from the series length and later chunks from the
# measured bytes per replicate of the first.
_CHUNK_BYTES = 32 * 2**20
_CHUNK_SIZE = 65536


@dataclass(frozen=True, slots=True)
class PreparationFailed:
    """Sentinel a preparer returns instead of a context when setup fails recoverably.

    Used by ``stability_policy="skip"``: a non-stationary fit produces this instead
    of raising, and the entry point returns an empty result flagged as failed.
    """

    reason: str


@dataclass(frozen=True, slots=True)
class ReduceRequest:
    """A resolved reduction request handed to a reduce executor.

    The numpy backend uses ``fn`` (the per-replicate or, when ``vectorized``, per-chunk
    callable) and ignores ``name``/``q``; a compiled backend uses the built-in ``name``
    (and ``q`` for the quantile) and cannot run an arbitrary ``fn``. The entry point
    resolves the public ``statistic`` argument into this once and both backends read the
    field they support, so the reduce seam carries no backend-specific branching.
    """

    fn: Callable[[NDArray[np.floating], NDArray[np.int32] | None], object] | None
    name: str | None
    q: float | None
    vectorized: bool


# A per-spec numpy CHUNK kernel: build one chunk of replicates from its replicate streams.
# ``streams`` (a :class:`tsbootstrap.rng.ReplicateStreams`) yields, replicate by replicate, one
# reused Generator positioned at that replicate's stream; the kernel draws per replicate and
# vectorises the numeric work. Returns
# ``(values (chunk, n[, d]) sim_dtype, indices (chunk, n) int32 or None)``.
# ``out_values`` and ``out_indices`` are either None (the kernel allocates its result) or
# C-contiguous row slices of the run's full output: the kernel then writes its final bytes
# there, once, and returns those same arrays.
ChunkExecutor = Callable[
    [
        object,
        object,
        ReplicateStreams,
        int,
        "np.dtype[np.floating]",
        "NDArray[np.floating] | None",
        "NDArray[np.int32] | None",
    ],
    "tuple[NDArray[np.floating], NDArray[np.int32] | None]",
]
# Where a chunk's rows go: ``(start, size) -> (out_values, out_indices)`` row slices of the
# run's full output, or ``(None, None)`` when the kernel should allocate.
OutputSlots = Callable[
    [int, int], "tuple[NDArray[np.floating] | None, NDArray[np.int32] | None]"
]
# A full-B values executor (Option D): owns RNG derivation from the root and chunking.
ValuesExecutor = Callable[
    [object, object, "np.random.SeedSequence", int, int, "np.dtype[np.floating]"],
    "tuple[NDArray[np.floating], NDArray[np.int32] | IndexSource | None]",
]
# A full-B reduce executor: fuses generation and reduction, never materialising (B, n, d).
ReduceExecutor = Callable[
    [object, object, "np.random.SeedSequence", int, int, "np.dtype[np.floating]", ReduceRequest],
    NDArray[np.float64],
]
# Preparer: (data, spec, exog) -> prepared. Runs ONCE per bootstrap() call (e.g. fit a
# model). The prepared value is passed to every executor. ``exog`` is the optional (n, k)
# exogenous array (None for most methods). The default preparer returns the data unchanged.
Preparer = Callable[[NDArray[np.float64], object, object], object]

_CHUNK_KERNELS: dict[type, ChunkExecutor] = {}
_VALUES_EXECUTORS: dict[tuple[type, str], ValuesExecutor] = {}
_REDUCE_EXECUTORS: dict[tuple[type, str], ReduceExecutor] = {}
# Optional per-registration predicate: a registered executor may still refuse some
# instances of its spec type (e.g. a compiled recursive kernel and an innovation it does
# not draw). Keyed (spec_type, backend, kind) with kind "values" or "reduce"; absent means
# every instance is supported.
SupportPredicate = Callable[[object], bool]
_SUPPORT: dict[tuple[type, str, str], SupportPredicate] = {}
_KINDS = ("values", "reduce")
_PREPARERS: dict[type, Preparer] = {}

# Each concrete kernel/executor/preparer is registered with its precise parameter types
# and returned UNCHANGED, so the registration site keeps full type checking; the registry
# stores it type-erased via one localized ``cast`` and resolves the spec at lookup time.
_K = TypeVar(
    "_K",
    bound=Callable[..., "tuple[NDArray[np.floating], NDArray[np.int32] | None]"],
)
_V = TypeVar("_V", bound=Callable[..., object])
_P = TypeVar("_P", bound=Callable[..., object])


def gather_rows(
    data: NDArray[np.float64],
    idx: NDArray[np.int32],
    sim_dtype: np.dtype[np.floating],
    out: NDArray[np.floating] | None = None,
) -> NDArray[np.floating]:
    """``data[idx]`` in ``sim_dtype``, written once into ``out`` when it is given.

    The source is cast first when ``sim_dtype`` is narrower, so the gather copies bytes
    already in the target type and ``np.take`` needs no buffered copy of ``out``; every
    value equals ``data[idx].astype(sim_dtype)``. Indices are in range by construction, so
    the unchecked ``clip`` mode is exact.
    """
    source = data if data.dtype == sim_dtype else data.astype(sim_dtype)
    if out is None:
        out = np.empty((*idx.shape, *data.shape[1:]), dtype=sim_dtype)
    rows = max(1, ROW_GROUP_ELEMENTS // max(1, idx.shape[-1]))
    for lo in range(0, idx.shape[0], rows):
        # np.take widens the index block to intp; one group at a time keeps that copy small.
        np.take(source, idx[lo : lo + rows], axis=0, out=out[lo : lo + rows], mode="clip")
    return out


def _identity_preparer(data: NDArray[np.float64], spec: object, exog: object) -> object:
    return data


def _chunk_runs(
    kernel: ChunkExecutor,
    prepared: object,
    spec: object,
    root_ss: np.random.SeedSequence,
    n_bootstraps: int,
    n_obs: int,
    sim_dtype: np.dtype[np.floating],
    slots: OutputSlots | None = None,
) -> Iterator[tuple[int, NDArray[np.floating], NDArray[np.int32] | None]]:
    """Run ``kernel`` over byte-budgeted chunks, yielding ``(start, values, indices)``.

    Replicate ``i`` is bound to child ``i`` of the root (the numpy determinism contract).
    Streams are derived per chunk, so a streaming caller never holds ``B`` seed states,
    and the root is never advanced. ``slots``, when given, is asked for each chunk's output
    row slices just before the kernel runs, so the kernel writes into the final array.
    """
    # Before the first chunk only the length is known: assume one float64 series, indices and
    # a float64 working copy (20 bytes per observation).
    per_replicate = max(1, n_obs * 20)
    start = 0
    while start < n_bootstraps:
        size = max(1, min(_CHUNK_SIZE, _CHUNK_BYTES // per_replicate, n_bootstraps - start))
        out_values, out_indices = (None, None) if slots is None else slots(start, size)
        values, indices = kernel(
            prepared,
            spec,
            ReplicateStreams(root_ss, start, size),
            n_obs,
            sim_dtype,
            out_values,
            out_indices,
        )
        # Engines return C-contiguous values in the requested sim_dtype (and contiguous
        # int32 indices); assert the contract once at the seam rather than re-coercing.
        assert values.dtype == sim_dtype and values.flags["C_CONTIGUOUS"]  # noqa: S101
        if start == 0:
            row = values[0]
            per_replicate = max(
                1, row.nbytes + row.size * 8 + (0 if indices is None else indices[0].nbytes)
            )
        yield start, values, indices
        del values, indices  # the next chunk must not be built beside this one
        start += size


class _FullOutput:
    """The run's full ``(B, ...)`` output, allocated once the first chunk fixes the row shape."""

    __slots__ = ("indices", "values")

    def __init__(self) -> None:
        self.values: NDArray[np.floating] | None = None
        self.indices: NDArray[np.int32] | None = None

    def adopt_first_chunk(
        self,
        n_bootstraps: int,
        values: NDArray[np.floating],
        indices: NDArray[np.int32] | None,
    ) -> None:
        """Allocate the full arrays and move the first chunk into them."""
        size = values.shape[0]
        self.values = np.empty((n_bootstraps, *values.shape[1:]), dtype=values.dtype)
        self.values[:size] = values
        if indices is not None:
            self.indices = np.empty((n_bootstraps, *indices.shape[1:]), dtype=np.int32)
            self.indices[:size] = indices

    def slots(
        self, start: int, size: int
    ) -> tuple[NDArray[np.floating] | None, NDArray[np.int32] | None]:
        """The row slices the next chunk should be written into (None before adoption)."""
        if self.values is None:
            return None, None
        stop = start + size
        return self.values[start:stop], None if self.indices is None else self.indices[start:stop]

    def result(self) -> tuple[NDArray[np.floating], NDArray[np.int32] | None]:
        assert self.values is not None  # noqa: S101  (n_bootstraps >= 1)
        return self.values, self.indices


def _make_numpy_values(kernel: ChunkExecutor) -> ValuesExecutor:
    """Wrap a per-spec chunk kernel into a full-B numpy values executor."""

    def _numpy_values(
        prepared: object,
        spec: object,
        root_ss: np.random.SeedSequence,
        n_bootstraps: int,
        n_obs: int,
        sim_dtype: np.dtype[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.int32] | None]:
        # One output allocation per run. The first chunk fixes the row shape, so the kernel
        # allocates it; a run that fits one chunk returns the kernel's own arrays. Otherwise
        # the first chunk moves into the full output and every later chunk is written into
        # its slot by the kernel itself, so neither a list of chunks nor a second chunk-sized
        # copy exists at the peak.
        out = _FullOutput()
        for start, values, indices in _chunk_runs(
            kernel, prepared, spec, root_ss, n_bootstraps, n_obs, sim_dtype, out.slots
        ):
            if start == 0:
                if values.shape[0] == n_bootstraps:
                    return values, indices
                out.adopt_first_chunk(n_bootstraps, values, indices)
            del values, indices  # a later chunk is already in place; the first has been copied
        return out.result()

    return _numpy_values


def _make_numpy_reduce(kernel: ChunkExecutor) -> ReduceExecutor:
    """Wrap a per-spec chunk kernel into the generic streaming numpy reduce executor.

    This is the reduce fallback for every numpy-backed method: it drives the same chunk
    kernel over the same chunk loop as the values path (so its generation is bit-identical)
    and applies ``request.fn`` to each replicate, keeping only the ``(B, |theta|)`` result.
    Peak memory is O(chunk), independent of B in the paths.
    """

    def _numpy_reduce(
        prepared: object,
        spec: object,
        root_ss: np.random.SeedSequence,
        n_bootstraps: int,
        n_obs: int,
        sim_dtype: np.dtype[np.floating],
        request: ReduceRequest,
    ) -> NDArray[np.float64]:
        assert request.fn is not None  # noqa: S101  (numpy reduce always carries a callable)
        fn = request.fn
        statistics: NDArray[np.float64] | None = None
        k = 0
        for _start, values, indices in _chunk_runs(
            kernel, prepared, spec, root_ss, n_bootstraps, n_obs, sim_dtype
        ):
            chunk_b = values.shape[0]
            if request.vectorized:
                block = np.asarray(fn(values, indices), dtype=np.float64)
                if block.ndim == 0 or block.shape[0] != chunk_b:
                    got = "a scalar" if block.ndim == 0 else f"leading axis {block.shape[0]}"
                    raise ValueError(
                        f"a vectorized statistic must return one row per replicate, shape "
                        f"(chunk, *theta) with leading axis {chunk_b}; got {got}. Either return a "
                        f"per-replicate-stacked result or call without vectorized=True."
                    )
                if statistics is None:
                    statistics = np.empty((n_bootstraps, *block.shape[1:]), dtype=np.float64)
                statistics[k : k + chunk_b] = block
                k += chunk_b
                continue
            for i in range(chunk_b):
                idx_i = None if indices is None else indices[i]
                theta = np.asarray(fn(values[i], idx_i), dtype=np.float64)
                if statistics is None:
                    statistics = np.empty((n_bootstraps, *theta.shape), dtype=np.float64)
                statistics[k] = theta  # broadcasts/raises identically to np.stack on mismatch
                k += 1
        if statistics is None:  # no replicates produced (defensive; n_bootstraps >= 1)
            statistics = np.empty((0,), dtype=np.float64)
        return statistics

    return _numpy_reduce


def register_chunk_executor(spec_type: type) -> Callable[[_K], _K]:
    """Register a numpy per-chunk kernel for ``spec_type`` (the default backend).

    Registering the kernel also builds and registers its full-B numpy values executor and
    the generic streaming numpy reduce executor, so every numpy method gets both seams from
    one kernel. Returns the kernel unchanged.
    """

    def decorator(fn: _K) -> _K:
        _CHUNK_KERNELS[spec_type] = cast(ChunkExecutor, fn)
        _VALUES_EXECUTORS[(spec_type, "numpy")] = _make_numpy_values(cast(ChunkExecutor, fn))
        _REDUCE_EXECUTORS[(spec_type, "numpy")] = _make_numpy_reduce(cast(ChunkExecutor, fn))
        return fn

    return decorator


def stream_numpy_values(
    spec: object,
    prepared: object,
    root_ss: np.random.SeedSequence,
    n_bootstraps: int,
    n_obs: int,
    sim_dtype: np.dtype[np.floating],
) -> Iterator[tuple[NDArray[np.floating], NDArray[np.int32] | None]]:
    """Yield ``(values, indices)`` per byte-budgeted chunk for the numpy streaming iterator.

    Drives the same per-spec chunk kernel over the same :func:`_chunk_runs` loop as the
    numpy values executor, but yields each chunk instead of concatenating, so a caller can
    stream a very large ``B`` without ever holding the full ``(B, n[, d])`` tensor.
    Determinism is identical to the materialised path: replicate ``i`` is bound to child
    ``i`` of the root regardless of the chunk boundary.
    """
    try:
        kernel = _CHUNK_KERNELS[type(spec)]
    except KeyError:
        raise _unsupported(spec, "numpy") from None
    for _start, values, indices in _chunk_runs(
        kernel, prepared, spec, root_ss, n_bootstraps, n_obs, sim_dtype
    ):
        yield values, indices


def register_values_executor(
    spec_type: type, backend: str, *, supports: SupportPredicate | None = None
) -> Callable[[_V], _V]:
    """Register a full-B values executor for ``(spec_type, backend)`` (e.g. a compiled kernel).

    ``supports`` narrows the registration to the instances it returns True for; the
    capability check :func:`supports` reads it, so no hand-written list can go stale.
    """

    def decorator(fn: _V) -> _V:
        _VALUES_EXECUTORS[(spec_type, backend)] = cast(ValuesExecutor, fn)
        if supports is not None:
            _SUPPORT[(spec_type, backend, "values")] = supports
        return fn

    return decorator


def register_reduce_executor(
    spec_type: type, backend: str, *, supports: SupportPredicate | None = None
) -> Callable[[_V], _V]:
    """Register a full-B fused reduce executor for ``(spec_type, backend)``."""

    def decorator(fn: _V) -> _V:
        _REDUCE_EXECUTORS[(spec_type, backend)] = cast(ReduceExecutor, fn)
        if supports is not None:
            _SUPPORT[(spec_type, backend, "reduce")] = supports
        return fn

    return decorator


def _check_kind(kind: str) -> None:
    if kind not in _KINDS:
        raise MethodConfigError(
            f"capability kind must be one of {_KINDS}; got {kind!r}",
            code=Codes.INVALID_PARAMETER,
            context={"kind": kind},
        )


def supports(spec: object, backend: str, kind: str) -> bool:
    """Whether ``backend`` runs ``spec`` for ``kind`` ("values" or "reduce").

    True exactly when an executor of that kind is registered for the spec's type and
    backend and its registration predicate (if any) accepts the instance.
    """
    _check_kind(kind)
    table = _VALUES_EXECUTORS if kind == "values" else _REDUCE_EXECUTORS
    if (type(spec), backend) not in table:
        return False
    predicate = _SUPPORT.get((type(spec), backend, kind))
    return predicate is None or bool(predicate(spec))


def registered_spec_names(backend: str, kind: str) -> list[str]:
    """Sorted names of the spec types with a ``kind`` executor on ``backend``."""
    _check_kind(kind)
    table = _VALUES_EXECUTORS if kind == "values" else _REDUCE_EXECUTORS
    return sorted(spec_type.__name__ for spec_type, b in table if b == backend)


def get_values_executor(spec: object, backend: str) -> ValuesExecutor:
    """Look up the values executor for a spec instance and backend, or raise structured."""
    try:
        return _VALUES_EXECUTORS[(type(spec), backend)]
    except KeyError:
        raise _unsupported(spec, backend) from None


def get_reduce_executor(spec: object, backend: str) -> ReduceExecutor:
    """Look up the reduce executor for a spec instance and backend, or raise structured."""
    try:
        return _REDUCE_EXECUTORS[(type(spec), backend)]
    except KeyError:
        raise _unsupported(spec, backend) from None


def _unsupported(spec: object, backend: str) -> MethodConfigError:
    values = registered_spec_names(backend, "values")
    reduce = registered_spec_names(backend, "reduce")
    return MethodConfigError(
        f"method {type(spec).__name__!r} is not implemented for backend {backend!r}",
        code=Codes.UNSUPPORTED_MODEL_FEATURE,
        context={"method": type(spec).__name__, "backend": backend},
        hint=f"Backend {backend!r} runs values for {values} and reduce for {reduce}.",
    )


def has_values_executor(spec_type: type, backend: str) -> bool:
    """Whether a values executor is registered for ``(spec_type, backend)``."""
    return (spec_type, backend) in _VALUES_EXECUTORS


def has_reduce_executor(spec_type: type, backend: str) -> bool:
    """Whether a fused reduce executor is registered for ``(spec_type, backend)``."""
    return (spec_type, backend) in _REDUCE_EXECUTORS


def register_preparer(spec_type: type) -> Callable[[_P], _P]:
    """Decorator: register a one-time setup function for ``spec_type``."""

    def decorator(fn: _P) -> _P:
        _PREPARERS[spec_type] = cast(Preparer, fn)
        return fn

    return decorator


def get_preparer(spec: object) -> Preparer:
    """Return the preparer for a spec instance (the identity preparer by default)."""
    return _PREPARERS.get(type(spec), _identity_preparer)


__all__ = [
    "ChunkExecutor",
    "OutputSlots",
    "gather_rows",
    "ValuesExecutor",
    "ReduceExecutor",
    "ReduceRequest",
    "Preparer",
    "PreparationFailed",
    "register_chunk_executor",
    "register_values_executor",
    "register_reduce_executor",
    "register_preparer",
    "get_values_executor",
    "get_reduce_executor",
    "get_preparer",
    "has_values_executor",
    "has_reduce_executor",
    "registered_spec_names",
    "stream_numpy_values",
    "SupportPredicate",
    "supports",
]
