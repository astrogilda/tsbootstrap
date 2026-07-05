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
:func:`register_chunk_executor`; that kernel produces a fixed-size CHUNK of replicates
from a list of child seeds, and this module wraps it into both a numpy values executor
and a generic numpy reduce executor (the streaming fallback) by driving it over the
``_CHUNK_SIZE`` chunk loop. A compiled/GPU backend instead registers a full-``B`` fused
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
from tsbootstrap.rng import spawn_seed_sequences

# Generate B in fixed-size chunks. A constant (never RAM-derived) chunk size keeps the
# matrix shapes the BLAS kernels see identical across machines, so floating-point
# accumulation order, and therefore results, stay reproducible. The numpy values and
# reduce executors both drive their per-spec kernel over this chunk size, so the two are
# bit-identical to each other and independent of B.
_CHUNK_SIZE = 2048


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


# A per-spec numpy CHUNK kernel: build one fixed-size chunk of replicates from its child
# seeds. ``seeds`` is the chunk's list of per-replicate ``SeedSequence`` objects (seed i
# bound to replicate i); the kernel materialises its generators via
# :func:`tsbootstrap.rng.generators_from_seeds` and vectorises the numeric work. Returns
# ``(values (chunk, n[, d]) sim_dtype, indices (chunk, n) int32 or None)``.
ChunkExecutor = Callable[
    [object, object, "list[np.random.SeedSequence]", int, "np.dtype[np.floating]"],
    "tuple[NDArray[np.floating], NDArray[np.int32] | None]",
]
# A full-B values executor (Option D): owns RNG derivation from the root and chunking.
ValuesExecutor = Callable[
    [object, object, "np.random.SeedSequence", int, int, "np.dtype[np.floating]"],
    "tuple[NDArray[np.floating], NDArray[np.int32] | None]",
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


def _identity_preparer(data: NDArray[np.float64], spec: object, exog: object) -> object:
    return data


def _chunked_seeds(
    root_ss: np.random.SeedSequence, n_bootstraps: int
) -> list[list[np.random.SeedSequence]]:
    """Spawn the B per-replicate child seeds upfront and slice them into fixed-size chunks.

    The upfront spawn is the locked numpy determinism contract (replicate i is bound to
    child i of the root), and the fixed ``_CHUNK_SIZE`` keeps BLAS shapes, and therefore
    accumulation order, identical regardless of B. Both are what the ``_NUMPY_BITS``
    byte-freeze pins.
    """
    seeds = spawn_seed_sequences(root_ss, n_bootstraps)
    return [seeds[start : start + _CHUNK_SIZE] for start in range(0, n_bootstraps, _CHUNK_SIZE)]


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
        value_chunks: list[NDArray[np.floating]] = []
        index_chunks: list[NDArray[np.int32]] = []
        indices_present = True
        for chunk in _chunked_seeds(root_ss, n_bootstraps):
            values, indices = kernel(prepared, spec, chunk, n_obs, sim_dtype)
            # Engines return C-contiguous values in the requested sim_dtype (and contiguous
            # int32 indices); assert the contract once at the seam rather than re-coercing.
            assert values.dtype == sim_dtype and values.flags["C_CONTIGUOUS"]  # noqa: S101
            value_chunks.append(values)
            if indices is None:
                indices_present = False
            else:
                index_chunks.append(indices)
        # A single chunk needs no concatenate (which would copy the whole (B, n[, d]) array).
        values_b = (
            value_chunks[0] if len(value_chunks) == 1 else np.concatenate(value_chunks, axis=0)
        )
        if not indices_present:
            indices_b: NDArray[np.int32] | None = None
        elif len(index_chunks) == 1:
            indices_b = index_chunks[0]
        else:
            indices_b = np.concatenate(index_chunks, axis=0)
        return values_b, indices_b

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
        for chunk in _chunked_seeds(root_ss, n_bootstraps):
            values, indices = kernel(prepared, spec, chunk, n_obs, sim_dtype)
            assert values.dtype == sim_dtype and values.flags["C_CONTIGUOUS"]  # noqa: S101
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
    """Yield ``(values, indices)`` per fixed-size chunk for the numpy streaming iterator.

    Drives the same per-spec chunk kernel over the same :func:`_chunked_seeds` loop as the
    numpy values executor, but yields each chunk instead of concatenating, so a caller can
    stream a very large ``B`` without ever holding the full ``(B, n[, d])`` tensor.
    Determinism is identical to the materialised path: replicate ``i`` is bound to child
    ``i`` of the root regardless of the chunk boundary.
    """
    try:
        kernel = _CHUNK_KERNELS[type(spec)]
    except KeyError:
        raise _unsupported(spec, "numpy") from None
    for chunk in _chunked_seeds(root_ss, n_bootstraps):
        values, indices = kernel(prepared, spec, chunk, n_obs, sim_dtype)
        assert values.dtype == sim_dtype and values.flags["C_CONTIGUOUS"]  # noqa: S101
        yield values, indices


def register_values_executor(spec_type: type, backend: str) -> Callable[[_V], _V]:
    """Register a full-B values executor for ``(spec_type, backend)`` (e.g. a compiled kernel)."""

    def decorator(fn: _V) -> _V:
        _VALUES_EXECUTORS[(spec_type, backend)] = cast(ValuesExecutor, fn)
        return fn

    return decorator


def register_reduce_executor(spec_type: type, backend: str) -> Callable[[_V], _V]:
    """Register a full-B fused reduce executor for ``(spec_type, backend)``."""

    def decorator(fn: _V) -> _V:
        _REDUCE_EXECUTORS[(spec_type, backend)] = cast(ReduceExecutor, fn)
        return fn

    return decorator


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
    return MethodConfigError(
        f"method {type(spec).__name__!r} is not implemented for backend {backend!r}",
        code=Codes.UNSUPPORTED_MODEL_FEATURE,
        hint="Supported methods are the MethodSpec union in tsbootstrap.methods; "
        "the compiled backend covers only the observation and AR-residual methods.",
    )


def has_values_executor(spec_type: type, backend: str) -> bool:
    """Whether a values executor is registered for ``(spec_type, backend)``."""
    return (spec_type, backend) in _VALUES_EXECUTORS


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
    "stream_numpy_values",
]
