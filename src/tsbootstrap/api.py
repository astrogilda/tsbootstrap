"""Public entry point: ``bootstrap(X, *, method=BaseMethodSpec, ...)``.

``bootstrap`` returns a structured :class:`~tsbootstrap.results.BootstrapResult`;
``bootstrap_reduce`` reduces each replicate to a statistic without materialising the
full ``(B, n[, d])`` array. Both dispatch on the method spec and run each replicate on
its own index-bound RNG stream. Observation indices are intrinsic to the sampling plan,
so they are always attached when meaningful (recursive methods attach ``None``); there
is no ``return_indices`` flag.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
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
from tsbootstrap.methods import IID, BaseMethodSpec, ResidualBootstrap, SieveAR
from tsbootstrap.results import (
    BootstrapResult,
    BootstrapRunMetadata,
    BootstrapSample,
    ReducedResult,
)
from tsbootstrap.rng import (
    RandomStateLike,
    generators_from_seeds,
    resolve_and_describe,
    spawn_seed_sequences,
    warmup_kernels,
)
from tsbootstrap.validation import coerce_exog, coerce_observations


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
    seeds: list[np.random.SeedSequence],
    n_obs: int,
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Plain i.i.d. resampling of observation rows. Baseline; breaks dependence."""
    generators = generators_from_seeds(seeds)
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


@dataclass(frozen=True, slots=True)
class _RunSetup:
    """Everything bootstrap() and bootstrap_reduce() share after the one-time setup."""

    executor: object
    prepared: object
    method: object
    n_obs: int
    n_series: int
    n_bootstraps: int
    was_1d: bool
    seeds: list[np.random.SeedSequence]
    metadata: Callable[..., BootstrapRunMetadata]


def _setup_run(
    X: object,
    method: BaseMethodSpec,
    n_bootstraps: int,
    random_state: RandomStateLike,
    exog: object,
) -> _RunSetup | BootstrapRunMetadata:
    """Validate, fit the model once, and spawn the per-replicate RNG streams.

    Returns a ready :class:`_RunSetup`, or a failed :class:`BootstrapRunMetadata` when
    preparation fails under ``stability_policy="skip"`` (no replicates are generated).
    """
    if not isinstance(n_bootstraps, int) or isinstance(n_bootstraps, bool) or n_bootstraps < 1:
        raise MethodConfigError(
            "n_bootstraps must be an integer >= 1",
            code=Codes.INVALID_PARAMETER,
            context={"n_bootstraps": n_bootstraps},
        )

    arr, was_1d = coerce_observations(X)
    n_obs, n_series = arr.shape

    exog_arr = None
    if exog is not None:
        if not isinstance(method, (ResidualBootstrap, SieveAR)):
            raise MethodConfigError(
                "exogenous regressors are only supported for model-based methods "
                "(ResidualBootstrap, SieveAR)",
                code=Codes.UNSUPPORTED_EXOG,
            )
        exog_arr = coerce_exog(exog, n_obs)

    _ensure_executors()
    executor = get_executor(method)
    root_ss, rs_info = resolve_and_describe(random_state)
    prepared = get_preparer(method)(arr, method, exog_arr)  # one-time setup (e.g. model fit)
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

    # stability_policy="skip": a non-stationary fit fails the whole run honestly.
    if isinstance(prepared, PreparationFailed):
        return _metadata(failed=True, failure_reason=prepared.reason)

    seeds = spawn_seed_sequences(root_ss, n_bootstraps)
    warmup_kernels()
    return _RunSetup(
        executor=executor,
        prepared=prepared,
        method=method,
        n_obs=n_obs,
        n_series=n_series,
        n_bootstraps=n_bootstraps,
        was_1d=was_1d,
        seeds=seeds,
        metadata=_metadata,
    )


def _iter_chunks(setup: _RunSetup):
    """Yield ``(values, indices)`` per fixed-size chunk of replicates (bounds peak memory).

    Each replicate draws from its own index-bound generator, so determinism is independent
    of the chunking; the numeric work inside the executor is vectorised over the chunk.
    """
    for start in range(0, setup.n_bootstraps, _CHUNK_SIZE):
        chunk = setup.seeds[start : start + _CHUNK_SIZE]
        yield setup.executor(setup.prepared, setup.method, chunk, setup.n_obs)


def bootstrap(
    X: object,
    *,
    method: BaseMethodSpec,
    n_bootstraps: int = 999,
    random_state: RandomStateLike = None,
    exog: object = None,
) -> BootstrapResult:
    """Generate bootstrap replicates of a time series.

    Parameters
    ----------
    X : array-like
        Observations, shape ``(n,)`` or ``(n, d)``.
    method : BaseMethodSpec
        A method spec (e.g. ``MovingBlock(block_length="auto")``).
    n_bootstraps : int, default 999
        Number of replicates.
    random_state : int | numpy Generator | SeedSequence | None
        Reproducibility seed. Replicate ``i`` is bound to its own generator, so
        results are reproducible for a given seed and environment (OS, hardware,
        BLAS, NumPy), as with NumPy/scikit-learn.
    exog : array-like or None
        Optional exogenous regressors, shape ``(n,)`` or ``(n, k)``, aligned with ``X``,
        held fixed during regeneration. Supported for ``ResidualBootstrap`` with an ``AR``
        (ARX), ``VAR`` (VARX), or ``ARIMA`` (ARIMAX) model, and for ``SieveAR``. ARX/VARX
        require ``initial="fixed"`` and ``burn_in=0`` (the exog must align with each step);
        ARIMAX has no such constraint (exog enters at the level after inverse-differencing).

    Returns
    -------
    BootstrapResult
        Sequence of :class:`~tsbootstrap.results.BootstrapSample` plus metadata.
    """
    setup = _setup_run(X, method, n_bootstraps, random_state, exog)
    if isinstance(setup, BootstrapRunMetadata):  # preparation failed (stability skip)
        return BootstrapResult([], setup)

    value_chunks: list[NDArray[np.float64]] = []
    index_chunks: list[NDArray[np.intp]] = []
    indices_present = True
    for v_chunk, idx_chunk in _iter_chunks(setup):
        value_chunks.append(np.asarray(v_chunk, dtype=np.float64))
        if idx_chunk is None:
            indices_present = False
        else:
            index_chunks.append(np.asarray(idx_chunk))
    values_b = np.concatenate(value_chunks, axis=0)
    indices_b = np.concatenate(index_chunks, axis=0) if indices_present else None

    samples: list[BootstrapSample] = []
    for i in range(setup.n_bootstraps):
        v = np.ascontiguousarray(values_b[i], dtype=np.float64)
        if setup.was_1d and v.ndim == 2 and v.shape[1] == 1:
            v = v[:, 0]
        idx_i = None if indices_b is None else np.ascontiguousarray(indices_b[i])
        samples.append(BootstrapSample(values=v, sample_id=i, indices=idx_i))
    return BootstrapResult(samples, setup.metadata())


def bootstrap_reduce(
    X: object,
    *,
    method: BaseMethodSpec,
    statistic: Callable[[NDArray[np.float64], NDArray[np.intp] | None], object],
    n_bootstraps: int = 999,
    random_state: RandomStateLike = None,
    exog: object = None,
) -> ReducedResult:
    """Bootstrap, reducing each replicate to a statistic instead of materialising paths.

    For very large ``n_bootstraps`` the full ``(B, n[, d])`` array does not fit in RAM.
    ``bootstrap_reduce`` evaluates ``statistic`` on each replicate inside the same
    fixed-size chunk loop as :func:`bootstrap` and keeps only the ``(B, |theta|)`` array
    of results, so peak memory is independent of ``B`` in the paths. Take exact quantiles
    over the replicates afterward (``result.quantile(...)``) — the basis for scaling
    conformal / UQ calibration to very large ``B``.

    Parameters
    ----------
    statistic : callable ``(values, indices) -> scalar | array``
        Applied to each replicate. ``values`` is the replicate, shape ``(n,)`` or
        ``(n, d)``; ``indices`` is its original-observation indices ``(n,)`` for
        observation-resampling methods, or ``None`` for recursive methods (so e.g. EnbPI
        can build the out-of-bag mask). It MUST be independent across replicates — it is
        evaluated one replicate at a time, so any dependence on the chunk boundary
        (``_CHUNK_SIZE``) would make the result irreproducible.

    Returns
    -------
    ReducedResult
        ``.statistics`` of shape ``(n_bootstraps, |theta|)``, or a failed result
        (``.statistics is None``, ``.failed``) when preparation fails.
    """
    setup = _setup_run(X, method, n_bootstraps, random_state, exog)
    if isinstance(setup, BootstrapRunMetadata):  # preparation failed (stability skip)
        return ReducedResult(statistics=None, metadata=setup)

    rows: list[NDArray[np.float64]] = []
    for v_chunk, idx_chunk in _iter_chunks(setup):
        v_chunk = np.asarray(v_chunk, dtype=np.float64)
        for i in range(v_chunk.shape[0]):
            v = np.ascontiguousarray(v_chunk[i], dtype=np.float64)
            if setup.was_1d and v.ndim == 2 and v.shape[1] == 1:
                v = v[:, 0]
            idx_i = None if idx_chunk is None else np.ascontiguousarray(idx_chunk[i])
            rows.append(np.asarray(statistic(v, idx_i), dtype=np.float64))
    statistics = np.stack(rows, axis=0) if rows else np.empty((0,), dtype=np.float64)
    return ReducedResult(statistics=statistics, metadata=setup.metadata())


__all__ = ["bootstrap", "bootstrap_reduce", "register_executor"]
