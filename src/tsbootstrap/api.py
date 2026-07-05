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
import functools
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from tsbootstrap.dispatch import (
    PreparationFailed,
    ReduceRequest,
    get_preparer,
    get_reduce_executor,
    get_values_executor,
    register_chunk_executor,
    stream_numpy_values,
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
    root_key_from,
    spawn_seed_sequences,
    warmup_kernels,
)
from tsbootstrap.validation import coerce_exog, coerce_observations


@functools.lru_cache(maxsize=1)
def _versions_cached() -> dict[str, str]:
    out: dict[str, str] = {"numpy": np.__version__}
    with contextlib.suppress(ImportError):
        import scipy

        out["scipy"] = scipy.__version__
    with contextlib.suppress(PackageNotFoundError):  # editable/uninstalled
        out["tsbootstrap"] = _pkg_version("tsbootstrap")
    return out


def _versions() -> dict[str, str]:
    # Package versions never change within a process; resolve them once and hand back a
    # fresh copy so a caller mutating the metadata dict cannot corrupt the cached value.
    return dict(_versions_cached())


@register_chunk_executor(IID)
def _iid_executor(
    data: NDArray[np.float64],
    spec: IID,
    seeds: list[np.random.SeedSequence],
    n_obs: int,
    sim_dtype: np.dtype[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.int32]]:
    """Plain i.i.d. resampling of observation rows. Baseline; breaks dependence."""
    generators = generators_from_seeds(seeds)
    out = np.empty((len(generators), n_obs), dtype=np.int32)
    for b, g in enumerate(generators):
        out[b] = g.integers(0, n_obs, size=n_obs, dtype=np.int32)
    return data[out].astype(sim_dtype, copy=False), out


_executors_ready = False
_compiled_ready = False

# Observation indices are int32 end to end (see block/indices.py); a series longer than
# this cannot be addressed by an int32 index, so the producer refuses it loudly rather
# than silently wrapping to a negative index.
_MAX_N_OBS = 2**31

# Precision of the simulation/path tensor. The fit, autocovariance, and all reductions
# always run in float64; only the returned values array is cast to one of these at the
# executor boundary. Lower precisions (e.g. bfloat16) are reserved for a future GPU backend.
_SIM_DTYPES: dict[str, np.dtype[np.floating]] = {
    "float64": np.dtype(np.float64),
    "float32": np.dtype(np.float32),
}


def _resolve_sim_dtype(dtype: str) -> np.dtype[np.floating]:
    """Map the public ``dtype`` string to a NumPy dtype, or raise a structured error."""
    try:
        return _SIM_DTYPES[dtype]
    except (KeyError, TypeError):
        raise MethodConfigError(
            f"dtype must be one of {sorted(_SIM_DTYPES)} (lower precisions such as "
            f"'bfloat16'/'bf16' are reserved for a future GPU backend); got {dtype!r}",
            code=Codes.INVALID_PARAMETER,
            context={"dtype": dtype},
        ) from None


def _mean_reducer(values: NDArray[np.floating], indices: NDArray[np.int32] | None) -> object:
    """Column mean of one replicate: ``(d,)`` for ``(n, d)`` input, scalar for ``(n,)``."""
    return values.mean(axis=0)


def _var_reducer(values: NDArray[np.floating], indices: NDArray[np.int32] | None) -> object:
    """Population variance (ddof=0) per column of one replicate, matching the compiled kernel."""
    return values.var(axis=0)


def _std_reducer(values: NDArray[np.floating], indices: NDArray[np.int32] | None) -> object:
    """Population standard deviation (ddof=0) per column of one replicate."""
    return values.std(axis=0)


# Reducers selectable by name. Passing ``statistic`` as one of these names lets the library
# choose the implementation: the default numpy backend maps the name to the callable below, and
# the compiled backend runs the matching fused kernel (it cannot introspect an arbitrary Python
# callable, so only these named reducers are available there). The numpy callables use ddof=0,
# matching the compiled kernels, so the two backends agree in distribution.
_BUILTIN_REDUCERS: dict[str, Callable[[NDArray[np.floating], NDArray[np.int32] | None], object]] = {
    "mean": _mean_reducer,
    "var": _var_reducer,
    "std": _std_reducer,
}


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


def _ensure_compiled_executors() -> None:
    """Register the compiled (numba) executors on first use of ``backend="compiled"``.

    Kept separate from :func:`_ensure_executors` and called only on the compiled path so
    numba is never imported for a pure-numpy run. Idempotent (the underlying
    ``register_compiled_executors`` is itself idempotent).
    """
    global _compiled_ready
    if _compiled_ready:
        return
    from tsbootstrap.block._compiled import register_compiled_executors

    register_compiled_executors()
    _compiled_ready = True


@dataclass(frozen=True, slots=True)
class _RunSetup:
    """Everything bootstrap() and bootstrap_reduce() share after the one-time setup.

    The run's root :class:`~numpy.random.SeedSequence` and ``n_bootstraps`` cross the
    executor seam unchanged; each executor derives its own per-replicate RNG from them
    (the numpy backend spawns ``B`` children, the compiled backend packs the root and
    keys in-kernel), so this setup holds no backend-specific RNG material.
    """

    prepared: object
    method: BaseMethodSpec
    n_obs: int
    n_series: int
    n_bootstraps: int
    was_1d: bool
    sim_dtype: np.dtype[np.floating]
    root_ss: np.random.SeedSequence
    metadata: Callable[..., BootstrapRunMetadata]


def _setup_run(
    X: object,
    method: BaseMethodSpec,
    n_bootstraps: int,
    random_state: RandomStateLike,
    exog: object,
    dtype: str = "float64",
) -> _RunSetup | BootstrapRunMetadata:
    """Validate, fit the model once, and resolve the run's root RNG.

    Returns a ready :class:`_RunSetup` carrying the root :class:`~numpy.random.SeedSequence`
    (executors derive their own per-replicate RNG from it), or a failed
    :class:`BootstrapRunMetadata` when preparation fails under ``stability_policy="skip"``
    (no replicates are generated).
    """
    if not isinstance(n_bootstraps, int) or isinstance(n_bootstraps, bool) or n_bootstraps < 1:
        raise MethodConfigError(
            "n_bootstraps must be an integer >= 1",
            code=Codes.INVALID_PARAMETER,
            context={"n_bootstraps": n_bootstraps},
        )

    sim_dtype = _resolve_sim_dtype(dtype)

    arr, was_1d = coerce_observations(X)
    n_obs, n_series = arr.shape

    # Single boundary for every method: observation indices are int32, so a series at or
    # above 2**31 observations cannot be addressed without overflow. Refuse it here rather
    # than letting the index arrays silently wrap to negative positions downstream.
    if n_obs >= _MAX_N_OBS:
        raise ValueError(
            f"series length {n_obs} >= {_MAX_N_OBS} exceeds the int32 index limit; "
            "bootstrap index arrays cannot address this many observations"
        )

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
            dtype=dtype,
            versions=_versions(),
            references=meta.references,
            **extra,  # type: ignore[arg-type]
        )

    # stability_policy="skip": a non-stationary fit fails the whole run honestly.
    if isinstance(prepared, PreparationFailed):
        return _metadata(failed=True, failure_reason=prepared.reason)

    return _RunSetup(
        prepared=prepared,
        method=method,
        n_obs=n_obs,
        n_series=n_series,
        n_bootstraps=n_bootstraps,
        was_1d=was_1d,
        sim_dtype=sim_dtype,
        root_ss=root_ss,
        metadata=_metadata,
    )


def bootstrap(
    X: object,
    *,
    method: BaseMethodSpec,
    n_bootstraps: int = 999,
    random_state: RandomStateLike = None,
    exog: object = None,
    dtype: Literal["float64", "float32"] = "float64",
    backend: Literal["numpy", "compiled"] = "numpy",
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
    dtype : {"float64", "float32"}, default "float64"
        Precision of the returned replicate values. The model fit, autocovariance, and
        every reduction always run in ``float64``; only the final simulation/path tensor
        is cast to ``dtype``, halving its memory at ``float32`` for large ``B``. Lower
        precisions are reserved for a future GPU backend.
    backend : {"numpy", "compiled"}, default "numpy"
        ``"numpy"`` is the default reproducible path (one PCG64 stream per replicate).
        ``"compiled"`` selects an opt-in numba kernel that builds indices and gathers in one
        replicate-parallel pass, a large speed-up on the observation methods (IID and the
        block families). It uses a distinct counter-based RNG stream with its own
        reproducibility goldens, so its replicates are equal in distribution to the numpy
        path but not bit-identical, and it is never engaged unless requested. It does not
        support the recursive (model-based) methods and requires the ``[accel]`` extra.

    Returns
    -------
    BootstrapResult
        Sequence of :class:`~tsbootstrap.results.BootstrapSample` plus metadata.
    """
    if backend not in ("numpy", "compiled"):
        raise MethodConfigError(
            f"backend must be 'numpy' or 'compiled'; got {backend!r}",
            code=Codes.INVALID_PARAMETER,
            context={"backend": backend},
        )
    if backend == "compiled":
        # Register the compiled executors, then reject an unsupported method up front, before
        # any model fit, so the error is the same regardless of the data (a recursive fit
        # could otherwise raise its own error first).
        _ensure_compiled_executors()
        from tsbootstrap.block._compiled import compiled_supports, unsupported_method_error

        if not compiled_supports(method):
            raise unsupported_method_error(method)
    setup = _setup_run(X, method, n_bootstraps, random_state, exog, dtype)
    if isinstance(setup, BootstrapRunMetadata):  # preparation failed (stability skip)
        return BootstrapResult([], setup)

    # One registry lookup, no backend branch in the execution path: the numpy values
    # executor drives the per-spec chunk kernel over the fixed chunk loop; the compiled
    # executor (registered above) runs the fused kernel. Each owns its RNG derivation from
    # the run's root SeedSequence (Option D).
    values_b, indices_b = get_values_executor(setup.method, backend)(
        setup.prepared,
        setup.method,
        setup.root_ss,
        setup.n_bootstraps,
        setup.n_obs,
        setup.sim_dtype,
    )
    meta = setup.metadata(backend="compiled") if backend == "compiled" else setup.metadata()
    samples = _assemble_samples(values_b, indices_b, setup.was_1d, setup.n_bootstraps)
    return BootstrapResult(samples, meta)


def _assemble_samples(
    values_b: NDArray[np.floating],
    indices_b: NDArray[np.int32] | None,
    was_1d: bool,
    n_bootstraps: int,
) -> list[BootstrapSample]:
    """Build the per-replicate :class:`BootstrapSample` list from the stacked arrays.

    Collapses the trailing length-1 axis for a univariate (``was_1d``) series so each sample
    is ``(n,)`` rather than ``(n, 1)``, mirroring the input shape.
    """
    samples: list[BootstrapSample] = []
    for i in range(n_bootstraps):
        v = values_b[i]
        if was_1d and v.ndim == 2 and v.shape[1] == 1:
            v = v[:, 0]
        idx_i = None if indices_b is None else indices_b[i]
        samples.append(BootstrapSample(values=v, sample_id=i, indices=idx_i))
    return samples


def _resolve_reducer_selection(
    statistic: str
    | tuple[str, float]
    | Callable[[NDArray[np.floating], NDArray[np.int32] | None], object],
    backend: str,
) -> tuple[str | None, float | None]:
    """Validate the public ``statistic`` to a ``(reducer_name, q)`` selection.

    Returns the built-in reducer name and its quantile level, or ``(None, None)`` for a
    user callable (allowed only on the numpy backend). Raises the same typed errors as
    before: a malformed tuple, an out-of-range ``q``, an unknown named reducer, or a
    callable on the compiled backend.
    """
    if isinstance(statistic, tuple):
        # The only parametrized reducer is the quantile: ("quantile", q).
        if len(statistic) != 2 or statistic[0] != "quantile":
            raise MethodConfigError(
                f"a tuple statistic must be ('quantile', q); got {statistic!r}",
                code=Codes.INVALID_PARAMETER,
                context={"statistic": statistic},
            )
        q = float(statistic[1])
        if not 0.0 <= q <= 1.0:
            raise MethodConfigError(
                f"quantile level q must lie in [0, 1]; got {q}",
                code=Codes.INVALID_PARAMETER,
                context={"q": q},
            )
        return "quantile", q
    if isinstance(statistic, str):
        if statistic not in _BUILTIN_REDUCERS:
            raise MethodConfigError(
                f"unknown built-in reducer {statistic!r}; available: {sorted(_BUILTIN_REDUCERS)} "
                "(the quantile reducer is selected as the tuple ('quantile', q))",
                code=Codes.INVALID_PARAMETER,
                context={"statistic": statistic},
            )
        return statistic, None
    if backend == "compiled":
        raise MethodConfigError(
            "backend='compiled' requires a built-in reducer (e.g. statistic='mean' or "
            "('quantile', q)); it cannot run an arbitrary Python callable",
            code=Codes.INVALID_PARAMETER,
        )
    return None, None


def bootstrap_reduce(
    X: object,
    *,
    method: BaseMethodSpec,
    statistic: str
    | tuple[str, float]
    | Callable[[NDArray[np.floating], NDArray[np.int32] | None], object],
    n_bootstraps: int = 999,
    random_state: RandomStateLike = None,
    exog: object = None,
    dtype: Literal["float64", "float32"] = "float64",
    vectorized: bool = False,
    backend: Literal["numpy", "compiled"] = "numpy",
) -> ReducedResult:
    """Bootstrap, reducing each replicate to a statistic instead of materialising paths.

    For very large ``n_bootstraps`` the full ``(B, n[, d])`` array does not fit in RAM.
    ``bootstrap_reduce`` evaluates ``statistic`` on each replicate inside the same
    fixed-size chunk loop as :func:`bootstrap` and keeps only the ``(B, |theta|)`` array
    of results, so peak memory is independent of ``B`` in the paths. Take exact quantiles
    over the replicates afterward (``result.quantile(...)``), the basis for scaling
    conformal / UQ calibration to very large ``B``.

    Parameters
    ----------
    statistic : str, ("quantile", q) tuple, or callable ``(values, indices) -> scalar | array``
        A callable applied to each replicate; the name of a built-in reducer (``"mean"``,
        ``"var"``, or ``"std"``); or ``("quantile", q)`` for the per-replicate quantile at
        level ``q`` in ``[0, 1]``. For a callable, ``values`` is the replicate, shape ``(n,)``
        or ``(n, d)``; ``indices`` is its original-observation indices ``(n,)`` for
        observation-resampling methods, or ``None`` for recursive methods (so e.g. EnbPI
        can build the out-of-bag mask). It MUST be independent across replicates, it is
        evaluated one replicate at a time, so any dependence on the chunk boundary
        (``_CHUNK_SIZE``) would make the result irreproducible. A built-in reducer (name or
        the quantile tuple) is required for ``backend="compiled"``, which cannot run an
        arbitrary Python callable.
    backend : {"numpy", "compiled"}, default "numpy"
        ``"numpy"`` is the default reproducible path (one PCG64 stream per replicate).
        ``"compiled"`` selects an opt-in numba kernel that fuses index build, gather, and
        reduce and never materialises the full sample, for a large speed-up on supported
        ``(method, reducer)`` pairs: the observation methods (IID and the block families) and
        the AR residual bootstrap, with the mean, variance, standard-deviation, and quantile
        reducers. It uses a distinct counter-based RNG stream with its own reproducibility
        goldens, so its results are equal in distribution to the numpy path but not
        bit-identical; it is never engaged unless you ask for it. Requires the ``[accel]`` extra.
    dtype : {"float64", "float32"}, default "float64"
        Precision of the replicate values handed to ``statistic``. The model fit,
        autocovariance, and every reduction inside the engines stay ``float64``; only the
        simulation/path tensor is cast. A ``float32`` path is a faithful down-cast of the
        ``float64`` path, not a different computation.
    vectorized : bool, default False
        If ``True``, ``statistic`` is called once per chunk over the whole batch:
        ``values`` is ``(chunk, n[, d])``, ``indices`` is ``(chunk, n)`` or ``None``, and it
        must return ``(chunk, *theta)`` (the statistic stacked over the chunk's replicates).
        This collapses the per-replicate Python call into one vectorised call, the fast path
        for large ``B`` and panel-scale use. The default per-replicate mode is the simple,
        always-correct path; for a genuinely batch-equivalent statistic ``vectorized=True``
        is byte-identical to it.

    Returns
    -------
    ReducedResult
        ``.statistics`` of shape ``(n_bootstraps, |theta|)``, or a failed result
        (``.statistics is None``, ``.failed``) when preparation fails.
    """
    if backend not in ("numpy", "compiled"):
        raise MethodConfigError(
            f"backend must be 'numpy' or 'compiled'; got {backend!r}",
            code=Codes.INVALID_PARAMETER,
            context={"backend": backend},
        )
    reducer_name, reducer_q = _resolve_reducer_selection(statistic, backend)
    if backend == "compiled":
        # Register the compiled executors, then reject an unsupported method up front, before
        # any model fit, so the error is the same regardless of the data (a recursive fit
        # could otherwise raise its own error first).
        _ensure_compiled_executors()
        from tsbootstrap.block._compiled import compiled_supports, unsupported_method_error

        if not compiled_supports(method):
            raise unsupported_method_error(method)

    setup = _setup_run(X, method, n_bootstraps, random_state, exog, dtype)
    if isinstance(setup, BootstrapRunMetadata):  # preparation failed (stability skip)
        return ReducedResult(statistics=None, metadata=setup)

    # Resolve the public statistic into one backend-agnostic request, then dispatch with no
    # backend branch: the numpy reduce executor streams request.fn over the chunk loop; the
    # compiled executor runs its fused kernel keyed by request.name/q.
    request = _resolve_reduce_request(
        statistic, reducer_name, reducer_q, vectorized=vectorized, was_1d=setup.was_1d
    )
    stats = get_reduce_executor(setup.method, backend)(
        setup.prepared,
        setup.method,
        setup.root_ss,
        setup.n_bootstraps,
        setup.n_obs,
        setup.sim_dtype,
        request,
    )
    # A univariate (was_1d) series reduces to (B,): the compiled kernel returns (B, 1) and the
    # numpy reducer (whose request.fn squeezes the single column) already returns (B,) for a
    # scalar statistic, so collapse only the trailing length-1 axis. No-op for the numpy path.
    if setup.was_1d and stats.ndim == 2 and stats.shape[1] == 1:
        stats = stats[:, 0]
    meta = setup.metadata(backend="compiled") if backend == "compiled" else setup.metadata()
    return ReducedResult(statistics=stats, metadata=meta)


def _resolve_reduce_request(
    statistic: str
    | tuple[str, float]
    | Callable[[NDArray[np.floating], NDArray[np.int32] | None], object],
    reducer_name: str | None,
    reducer_q: float | None,
    *,
    vectorized: bool,
    was_1d: bool,
) -> ReduceRequest:
    """Resolve the public ``statistic`` into one backend-agnostic :class:`ReduceRequest`.

    ``fn`` is the numpy per-replicate (or, when ``vectorized``, per-chunk) callable with the
    univariate ``was_1d`` squeeze baked in, so a 1-D series presents each replicate as ``(n,)``
    exactly as the materialised path does; ``name``/``q`` are the compiled-path fields. Both
    sides are populated for a built-in reducer so one request serves either backend. A user
    callable sets ``name=None`` (already rejected for the compiled backend before this point).
    """
    core: Callable[[NDArray[np.floating], NDArray[np.int32] | None], object]
    if reducer_name == "quantile":
        assert reducer_q is not None  # noqa: S101  (the tuple branch always sets q for "quantile")
        _q = reducer_q

        def _quantile_reducer(
            values: NDArray[np.floating], indices: NDArray[np.int32] | None
        ) -> object:
            return np.quantile(values, _q, axis=0)

        core = _quantile_reducer
    elif reducer_name is not None:
        core = _BUILTIN_REDUCERS[reducer_name]
    else:
        assert callable(statistic)  # noqa: S101  (the str/tuple/compiled branches are exhaustive)
        core = statistic

    def _fn(values: NDArray[np.floating], indices: NDArray[np.int32] | None) -> object:
        # Squeeze the trailing length-1 axis for a univariate series so the reducer sees the
        # documented (n,) replicate (vectorized: (chunk, n)); a no-op for multivariate input.
        if was_1d and vectorized and values.ndim == 3 and values.shape[2] == 1:
            values = values[:, :, 0]
        elif was_1d and not vectorized and values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        return core(values, indices)

    return ReduceRequest(fn=_fn, name=reducer_name, q=reducer_q, vectorized=vectorized)


def _coerce_panel(
    panel: object,
    indptr: object,
) -> tuple[NDArray[np.float64], NDArray[np.int64], int, int, bool]:
    """Resolve a ragged-panel input to ``(flat_data, indptr, num_series, d, was_1d)``.

    Two input forms are accepted:

    - ``panel`` a list/sequence of per-series arrays (each ``(n_s,)`` or ``(n_s, d)``)
      and ``indptr=None``: the flat data and CSR ``indptr`` are built internally.
    - ``panel`` a flat ``(total_N,)`` or ``(total_N, d)`` array and an explicit
      ``indptr`` of length ``num_series + 1``.

    ``was_1d`` is True when the per-series observations are univariate (so the result
    collapses its trailing axis), mirroring the rectangular ``was_1d`` collapse.
    """
    if indptr is None:
        series_list = list(panel)  # type: ignore[call-overload]
        if len(series_list) < 1:
            raise MethodConfigError(
                "a panel must contain at least one series",
                code=Codes.TOO_FEW_OBSERVATIONS,
                context={"num_series": len(series_list)},
            )
        coerced: list[NDArray[np.float64]] = []
        was_1d = True
        d = 1
        for s, series in enumerate(series_list):
            arr = np.ascontiguousarray(series, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            elif arr.ndim != 2:
                raise MethodConfigError(
                    f"series {s} must be 1-D or 2-D; got {arr.ndim} dimensions",
                    code=Codes.INVALID_SHAPE,
                    context={"series": s, "ndim": arr.ndim},
                )
            else:
                was_1d = False
            if s == 0:
                d = arr.shape[1]
            elif arr.shape[1] != d:
                raise MethodConfigError(
                    f"every series must have the same number of columns; series 0 has {d}, "
                    f"series {s} has {arr.shape[1]}",
                    code=Codes.INVALID_SHAPE,
                    context={"series": s, "d": arr.shape[1], "expected": d},
                )
            coerced.append(arr)
        offsets = np.zeros(len(coerced) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([a.shape[0] for a in coerced])
        flat = np.concatenate(coerced, axis=0) if len(coerced) > 1 else coerced[0]
        return flat, offsets, len(coerced), d, was_1d

    arr = np.asarray(panel)
    was_1d = arr.ndim == 1
    if was_1d:
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise MethodConfigError(
            f"a flat panel must be 1-D or 2-D (total_N[, d]); got {arr.ndim} dimensions",
            code=Codes.INVALID_SHAPE,
            context={"ndim": arr.ndim},
        )
    flat = np.ascontiguousarray(arr, dtype=np.float64)
    indptr_arr = np.ascontiguousarray(indptr)
    if indptr_arr.ndim != 1 or indptr_arr.shape[0] < 2:
        raise MethodConfigError(
            "indptr must be 1-D of length num_series + 1 (>= 2)",
            code=Codes.INVALID_SHAPE,
            context={"shape": tuple(indptr_arr.shape)},
        )
    return (
        flat,
        indptr_arr.astype(np.int64, copy=False),
        int(indptr_arr.shape[0]) - 1,
        flat.shape[1],
        was_1d,
    )


def bootstrap_reduce_panel(
    panel: object,
    *,
    indptr: object = None,
    method: BaseMethodSpec,
    statistic: str
    | tuple[str, float]
    | Callable[[NDArray[np.floating], NDArray[np.int32] | None], object],
    n_bootstraps: int = 999,
    random_state: RandomStateLike = None,
    dtype: Literal["float64", "float32"] = "float64",
    backend: Literal["numpy", "compiled"] = "numpy",
) -> ReducedResult:
    """Bootstrap a ragged PANEL of series and reduce each to a statistic, in one pass.

    The panel-scale counterpart of :func:`bootstrap_reduce`: it conformalises a whole
    collection of unequal-length (ragged) series at once and returns a dense
    ``(n_bootstraps, num_series, |theta|)`` array of per-replicate, per-series statistics,
    never materialising any ``(B, num_series, n)`` path. This is the route for calibrating
    a global / foundation forecaster over thousands of series without a Python loop over
    series.

    A separate function (rather than a flag on :func:`bootstrap_reduce`) is justified
    because the input contract differs fundamentally: a panel is ragged, so its natural
    representation is a list of per-series arrays (or a flat array plus a CSR ``indptr``),
    not the rectangular ``(n, d)`` that :func:`bootstrap_reduce` assumes. The output gains
    a series axis, and ``.values()`` is mathematically incoherent across unequal lengths,
    so the reduce IS the panel API.

    Parameters
    ----------
    panel : sequence of arrays, or a flat array
        Either a list of per-series observations (each ``(n_s,)`` or ``(n_s, d)``) with
        ``indptr=None``, or a flat ``(total_N,)`` / ``(total_N, d)`` array paired with an
        explicit ``indptr``.
    indptr : array of shape ``(num_series + 1,)`` or None
        CSR offsets when ``panel`` is a flat array; ``None`` when ``panel`` is a list of
        per-series arrays (the offsets are then built internally).
    method : BaseMethodSpec
        An observation method spec (IID, MovingBlock, CircularBlock, StationaryBlock, or
        NonOverlappingBlock). Recursive (model-based) methods are out of v1 panel scope and
        raise a typed error.
    statistic : str, ("quantile", q) tuple, or callable
        A built-in reducer name (``"mean"``, ``"var"``, ``"std"``), the ``("quantile", q)``
        tuple, or a callable ``(values, indices) -> scalar | array`` applied per series per
        replicate. ``backend="compiled"`` requires a built-in reducer.
    n_bootstraps : int, default 999
        Number of replicates ``B``.
    random_state : int | numpy Generator | SeedSequence | None
        Reproducibility seed. Reproducibility is tied to ``(seed, panel slot order)``:
        each series is keyed by its SLOT in the panel, so reordering, adding, or removing
        a series shifts the downstream slots and changes their streams (exactly as column
        order is part of a run's identity today).
    dtype : {"float64", "float32"}, default "float64"
        Precision of the returned statistics. The reductions run in float64; only the
        returned array is cast.
    backend : {"numpy", "compiled"}, default "numpy"
        ``"numpy"`` is the default reproducible reference: it loops over series calling the
        per-series :func:`bootstrap_reduce` (one PCG64 stream per replicate per series),
        slower but the reproducible default. ``"compiled"`` runs the fused, fully parallel
        panel kernel (a distinct counter-based Philox stream with its own goldens, equal in
        distribution to the numpy path but not bit-identical); it is the panel-scale moat
        and requires the ``[accel]`` extra.

    Returns
    -------
    ReducedResult
        ``.statistics`` of shape ``(n_bootstraps, num_series, |theta|)``, collapsed to
        ``(n_bootstraps, num_series)`` when the series are univariate and the statistic is
        scalar (mirroring the ``(B,)`` collapse of the rectangular reduce).
    """
    if backend not in ("numpy", "compiled"):
        raise MethodConfigError(
            f"backend must be 'numpy' or 'compiled'; got {backend!r}",
            code=Codes.INVALID_PARAMETER,
            context={"backend": backend},
        )
    if not isinstance(n_bootstraps, int) or isinstance(n_bootstraps, bool) or n_bootstraps < 1:
        raise MethodConfigError(
            "n_bootstraps must be an integer >= 1",
            code=Codes.INVALID_PARAMETER,
            context={"n_bootstraps": n_bootstraps},
        )
    sim_dtype = _resolve_sim_dtype(dtype)

    _ensure_executors()
    from tsbootstrap.block._compiled import (
        compiled_panel_reduce,
        compiled_panel_supports,
        unsupported_panel_method_error,
    )

    if not compiled_panel_supports(method):
        # The panel backend is observation-methods-only in both modes: the numpy reference
        # loops over the same per-series observation reduce, so a recursive method has no
        # coherent ragged-panel meaning here either. Reject it identically up front.
        raise unsupported_panel_method_error(method)

    flat, indptr_arr, num_series, _d, was_1d = _coerce_panel(panel, indptr)
    root_ss, rs_info = resolve_and_describe(random_state)
    total_n = int(flat.shape[0])

    def _metadata() -> BootstrapRunMetadata:
        meta = metadata_for(method)
        return BootstrapRunMetadata(
            method=meta.name,
            method_params=method.model_dump(),
            n_bootstraps=n_bootstraps,
            n_obs=total_n,
            n_series=num_series,
            random_state_kind=rs_info.kind,
            seed_entropy=rs_info.entropy,
            dtype=dtype,
            versions=_versions(),
            references=meta.references,
            backend=backend,
        )

    if backend == "compiled":
        # Validate the reducer is a built-in (the compiled kernel cannot run an arbitrary
        # callable), reusing the same translation as the rectangular path.
        reducer_name, reducer_q = _panel_compiled_reducer(statistic)
        warmup_kernels()
        root_key = root_key_from(root_ss)
        stats = compiled_panel_reduce(
            method,
            flat,
            indptr_arr,
            root_key,
            sim_dtype,
            reducer=reducer_name,
            q=reducer_q,
            n_bootstraps=n_bootstraps,
        )
        if was_1d:  # univariate panel collapses the trailing column axis
            stats = stats[:, :, 0]
        return ReducedResult(statistics=stats, metadata=_metadata())  # type: ignore[arg-type]

    # numpy reference: loop over series, reduce each with the per-series bootstrap_reduce on
    # its own child SeedSequence (reproducible PCG64). Slower, the reproducible default.
    series_seeds = spawn_seed_sequences(root_ss, num_series)
    statistics: NDArray[np.float64] | None = None
    for s in range(num_series):
        lo = int(indptr_arr[s])
        hi = int(indptr_arr[s + 1])
        series = flat[lo:hi, 0] if was_1d else flat[lo:hi]
        res = bootstrap_reduce(
            series,
            method=method,
            statistic=statistic,
            n_bootstraps=n_bootstraps,
            random_state=series_seeds[s],
            dtype=dtype,
            backend="numpy",
        )
        col = res.statistics
        if col is None:  # preparation failed for this series (defensive; obs methods never fail)
            raise MethodConfigError(
                f"the per-series reduce failed for series {s}",
                code=Codes.INVALID_PARAMETER,
                context={"series": s, "reason": res.failure_reason},
            )
        if statistics is None:
            statistics = cast(
                "NDArray[np.float64]",
                np.empty((n_bootstraps, num_series, *col.shape[1:]), dtype=col.dtype),
            )
        statistics[:, s] = col
    return ReducedResult(statistics=statistics, metadata=_metadata())


def _panel_compiled_reducer(statistic: object) -> tuple[str, float | None]:
    """Translate a ``statistic`` argument to a built-in reducer name + q for the panel kernel.

    The compiled panel kernel runs only the named reducers, so an arbitrary callable is
    rejected here (the same contract as the rectangular compiled path).
    """
    if isinstance(statistic, tuple):
        if len(statistic) != 2 or statistic[0] != "quantile":
            raise MethodConfigError(
                f"a tuple statistic must be ('quantile', q); got {statistic!r}",
                code=Codes.INVALID_PARAMETER,
                context={"statistic": statistic},
            )
        q = float(statistic[1])
        if not 0.0 <= q <= 1.0:
            raise MethodConfigError(
                f"quantile level q must lie in [0, 1]; got {q}",
                code=Codes.INVALID_PARAMETER,
                context={"q": q},
            )
        return "quantile", q
    if isinstance(statistic, str):
        if statistic not in _BUILTIN_REDUCERS:
            raise MethodConfigError(
                f"unknown built-in reducer {statistic!r}; available: {sorted(_BUILTIN_REDUCERS)} "
                "(the quantile reducer is selected as the tuple ('quantile', q))",
                code=Codes.INVALID_PARAMETER,
                context={"statistic": statistic},
            )
        return statistic, None
    raise MethodConfigError(
        "backend='compiled' requires a built-in reducer (e.g. statistic='mean' or "
        "('quantile', q)); it cannot run an arbitrary Python callable",
        code=Codes.INVALID_PARAMETER,
    )


def bootstrap_iter(
    X: object,
    *,
    method: BaseMethodSpec,
    n_bootstraps: int = 999,
    random_state: RandomStateLike = None,
    exog: object = None,
) -> Iterator[tuple[NDArray[np.floating], NDArray[np.int32] | None]]:
    """Yield bootstrap replicates in fixed-size chunks, bounding peak memory.

    Like :func:`bootstrap` but yields ``(values, indices)`` one chunk of replicates at a
    time instead of materialising all ``B`` at once, so a caller can stream a very large
    ``B`` (or feed batches to an array framework) without ever holding the full
    ``(B, n[, d])`` tensor. ``values`` is ``(chunk, n)`` or ``(chunk, n, d)``; ``indices`` is
    ``(chunk, n)`` int32, or ``None`` for recursive methods. Determinism matches
    :func:`bootstrap` exactly: replicate ``i`` is bound to the same RNG stream regardless of
    the chunk size. Yields nothing if preparation fails (e.g. a non-stationary fit under
    ``stability_policy='skip'``).
    """
    setup = _setup_run(X, method, n_bootstraps, random_state, exog)
    if isinstance(setup, BootstrapRunMetadata):  # preparation failed (stability skip)
        return
    for v_chunk, idx_chunk in stream_numpy_values(
        setup.method,
        setup.prepared,
        setup.root_ss,
        setup.n_bootstraps,
        setup.n_obs,
        setup.sim_dtype,
    ):
        if setup.was_1d and v_chunk.ndim == 3 and v_chunk.shape[2] == 1:
            v_chunk = v_chunk[:, :, 0]
        yield v_chunk, idx_chunk


__all__ = [
    "bootstrap",
    "bootstrap_iter",
    "bootstrap_reduce",
    "bootstrap_reduce_panel",
    "register_chunk_executor",
]
