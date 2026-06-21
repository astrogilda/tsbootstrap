"""Public entry point: ``bootstrap(X, *, method=MethodSpec, ...)``.

One typed function dispatches on the method spec, runs each replicate on its own
index-bound RNG stream (serial or via joblib), and returns a structured
:class:`~tsbootstrap.results.BootstrapResult`. Observation indices are intrinsic
to the sampling plan, so they are always attached when meaningful (recursive
methods attach ``None``); there is no ``return_indices`` flag.
"""

from __future__ import annotations

import contextlib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

import numpy as np
from numpy.typing import NDArray

from tsbootstrap.dispatch import (
    PreparationFailed,
    get_executor,
    get_preparer,
    register_executor,
)
from tsbootstrap.errors import Codes, MethodConfigError
from tsbootstrap.metadata import metadata_for
from tsbootstrap.methods import IID, MethodSpec
from tsbootstrap.results import BootstrapResult, BootstrapRunMetadata, BootstrapSample
from tsbootstrap.rng import (
    RandomStateLike,
    resolve_and_describe,
    spawn_generators,
    warmup_kernels,
)
from tsbootstrap.validation import coerce_observations


def _versions() -> dict[str, str]:
    out: dict[str, str] = {"numpy": np.__version__}
    with contextlib.suppress(ImportError):
        import scipy

        out["scipy"] = scipy.__version__
    with contextlib.suppress(PackageNotFoundError):  # editable/uninstalled
        out["tsbootstrap"] = _pkg_version("tsbootstrap")
    return out


@register_executor(IID)
def _iid_executor(
    data: NDArray[np.float64],
    spec: IID,
    generators: list[np.random.Generator],
    n_obs: int,
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Plain i.i.d. resampling of observation rows. Baseline; breaks dependence."""
    idx = np.stack([g.integers(0, n_obs, size=n_obs).astype(np.intp) for g in generators])
    return data[idx], idx


_executors_ready = False

# Generate B in fixed-size chunks. A constant (never RAM-derived) chunk size keeps the
# matrix shapes the BLAS kernels see identical across machines, so floating-point
# accumulation order — and therefore results — stay reproducible.
_CHUNK_SIZE = 2048


def _ensure_executors() -> None:
    """Import engine modules so they register their executors (idempotent).

    IID registers at import of this module. Block and recursive engines are
    imported here so dispatch sees them without an import cycle.
    """
    global _executors_ready
    if _executors_ready:
        return
    import tsbootstrap.block  # noqa: F401  (registers block executors)
    import tsbootstrap.model  # noqa: F401  (registers recursive executors)

    _executors_ready = True


def bootstrap(
    X: object,
    *,
    method: MethodSpec,
    n_bootstraps: int = 999,
    random_state: RandomStateLike = None,
) -> BootstrapResult:
    """Generate bootstrap replicates of a time series.

    Parameters
    ----------
    X : array-like
        Observations, shape ``(n,)`` or ``(n, d)``.
    method : MethodSpec
        A method spec (e.g. ``MovingBlock(block_length="auto")``).
    n_bootstraps : int, default 999
        Number of replicates.
    random_state : int | numpy Generator | SeedSequence | None
        Reproducibility seed. Replicate ``i`` is bound to its own generator, so
        results are reproducible for a given seed and environment (OS, hardware,
        BLAS, NumPy), as with NumPy/scikit-learn.

    Returns
    -------
    BootstrapResult
        Sequence of :class:`~tsbootstrap.results.BootstrapSample` plus metadata.
    """
    if not isinstance(n_bootstraps, int) or isinstance(n_bootstraps, bool) or n_bootstraps < 1:
        raise MethodConfigError(
            "n_bootstraps must be an integer >= 1",
            code=Codes.INVALID_PARAMETER,
            context={"n_bootstraps": n_bootstraps},
        )

    arr, was_1d = coerce_observations(X)
    n_obs, n_series = arr.shape

    _ensure_executors()
    executor = get_executor(method)

    root_ss, rs_info = resolve_and_describe(random_state)
    prepared = get_preparer(method)(arr, method)  # one-time setup (e.g. model fit)
    meta = metadata_for(method)

    def _metadata(**extra: object) -> BootstrapRunMetadata:
        return BootstrapRunMetadata(
            method=meta.name,
            method_params=method.model_dump(),
            n_bootstraps=n_bootstraps,
            n_obs=n_obs,
            n_series=n_series,
            random_state_kind=rs_info.kind,
            seed_entropy=rs_info.entropy,
            versions=_versions(),
            references=meta.references,
            **extra,  # type: ignore[arg-type]
        )

    # stability_policy="skip": a non-stationary fit fails the whole run honestly
    # rather than crashing a pipeline; no replicates are generated.
    if isinstance(prepared, PreparationFailed):
        return BootstrapResult([], _metadata(failed=True, failure_reason=prepared.reason))

    generators = spawn_generators(root_ss, n_bootstraps)
    warmup_kernels()

    # Each replicate draws its randoms from its own index-bound generator (so
    # determinism is independent of batching); the numeric work is vectorised, and B
    # is processed in fixed-size chunks to bound peak memory.
    value_chunks: list[NDArray[np.float64]] = []
    index_chunks: list[NDArray[np.intp]] = []
    indices_present = True
    for start in range(0, n_bootstraps, _CHUNK_SIZE):
        chunk = generators[start : start + _CHUNK_SIZE]
        v_chunk, idx_chunk = executor(prepared, method, chunk, n_obs)
        value_chunks.append(np.asarray(v_chunk, dtype=np.float64))
        if idx_chunk is None:
            indices_present = False
        else:
            index_chunks.append(np.asarray(idx_chunk))
    values_b = np.concatenate(value_chunks, axis=0)
    indices_b = np.concatenate(index_chunks, axis=0) if indices_present else None

    samples: list[BootstrapSample] = []
    for i in range(n_bootstraps):
        v = np.ascontiguousarray(values_b[i], dtype=np.float64)
        if was_1d and v.ndim == 2 and v.shape[1] == 1:
            v = v[:, 0]
        idx_i = None if indices_b is None else np.ascontiguousarray(indices_b[i])
        samples.append(BootstrapSample(values=v, sample_id=i, indices=idx_i))

    return BootstrapResult(samples, _metadata())


__all__ = ["bootstrap", "register_executor"]
