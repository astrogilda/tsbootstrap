"""Before-and-after measurements for the performance and RNG design study.

Every cell runs in a fresh subprocess so its peak RSS is its own, and every timed cell is
preceded by a small warm-up call in the same process (imports, numba compilation, first
allocations) so the timing covers steady-state work only. Each cell is repeated
``--repeats`` times and the minimum is reported, with the spread beside it.

Measured, per the study's list:

- end-to-end ``bootstrap(...).values()`` time and peak RSS, with GC on and with GC off;
- fresh-buffer first write against a write into an already-touched buffer;
- per-replicate seed cost (child derivation plus Generator construction);
- VAR with three series;
- compiled reduce against the numpy streaming reduce and a stdlib process pool;
- latency cells: import time with and without the ``[accel]`` extra, cold first call with an
  empty and a warm numba cache, small-B call overhead, a named reducer against a Python callable
  per replicate and vectorised, panel cost per series, per-chunk overhead of the numpy chunk
  loop (the chunk contract check included), and ``conf_int`` for every interval kind with the
  BCa jackknife timed on its own.

Run on a dedicated box::

    python benchmarks/bench_perf_study.py --json benchmarks/results/perf_study_<label>.json

Only the public API (plus :mod:`tsbootstrap.rng` for the seed micro-benchmark) is used, so
the same script measures any commit.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

# Thread layers are pinned before numpy is used anywhere, so BLAS and numba do not fight
# over cores and every cell sees the same thread budget.
_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")


def _ar1(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + e[t]
    return x


def _var1(n: int, d: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    a = 0.4 * np.eye(d) + 0.1
    x = np.zeros((n, d))
    for t in range(1, n):
        x[t] = a @ x[t - 1] + rng.standard_normal(d)
    return x


def _method(name: str) -> Any:
    import tsbootstrap as tsb

    return {
        "iid": tsb.IID(),
        "moving": tsb.MovingBlock(block_length=10),
        "stationary": tsb.StationaryBlock(block_length=10),
        "ar": tsb.ResidualBootstrap(model=tsb.AR(order=2)),
        "sieve": tsb.SieveAR(max_lag=10),
        "var": tsb.ResidualBootstrap(model=tsb.VAR(order=1)),
    }[name]


# name -> (method, n, B, d, backend)
VALUES_CELLS: dict[str, tuple[str, int, int, int, str]] = {
    "iid_n200_B100000": ("iid", 200, 100_000, 1, "numpy"),
    "iid_n2000_B10000": ("iid", 2000, 10_000, 1, "numpy"),
    "moving_n200_B100000": ("moving", 200, 100_000, 1, "numpy"),
    "moving_n2000_B10000": ("moving", 2000, 10_000, 1, "numpy"),
    "stationary_n200_B100000": ("stationary", 200, 100_000, 1, "numpy"),
    "stationary_n2000_B10000": ("stationary", 2000, 10_000, 1, "numpy"),
    "ar_n200_B100000": ("ar", 200, 100_000, 1, "numpy"),
    "ar_n2000_B10000": ("ar", 2000, 10_000, 1, "numpy"),
    "sieve_n200_B100000": ("sieve", 200, 100_000, 1, "numpy"),
    "sieve_n20000_B1000": ("sieve", 20_000, 1000, 1, "numpy"),
    "var3_n200_B20000": ("var", 200, 20_000, 3, "numpy"),
    "iid_n200_B100000_compiled": ("iid", 200, 100_000, 1, "compiled"),
    "moving_n200_B100000_compiled": ("moving", 200, 100_000, 1, "compiled"),
    "stationary_n200_B100000_compiled": ("stationary", 200, 100_000, 1, "compiled"),
    "ar_n200_B100000_compiled": ("ar", 200, 100_000, 1, "compiled"),
    "sieve_n200_B100000_compiled": ("sieve", 200, 100_000, 1, "compiled"),
    "var3_n200_B20000_compiled": ("var", 200, 20_000, 3, "compiled"),
}

# name -> (method, n, B, d, driver) ; driver in numpy | compiled | process
REDUCE_CELLS: dict[str, tuple[str, int, int, int, str]] = {
    f"reduce_{m}_n2000_B{b}_{drv}": (m, 2000, b, d, drv)
    for m, d, b in (("ar", 1, 20_000), ("var", 3, 10_000), ("sieve", 1, 20_000))
    for drv in ("numpy", "compiled", "process")
}


def _data(method: str, n: int, d: int) -> np.ndarray:
    return _var1(n, d) if method == "var" else _ar1(n)


def _supported(method: str, backend: str, kind: str) -> bool:
    if backend != "compiled":
        return True
    from tsbootstrap.block._compiled import compiled_supports

    return bool(compiled_supports(_method(method), kind))


def _run_values(method: str, n: int, b: int, d: int, backend: str, gc_off: bool) -> dict[str, Any]:
    import tsbootstrap as tsb

    x = _data(method, n, d)
    spec = _method(method)
    tsb.bootstrap(x, method=spec, n_bootstraps=64, random_state=1, backend=backend).values()
    gc.collect()
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if gc_off:
        gc.disable()
    t0 = time.perf_counter()
    res = tsb.bootstrap(x, method=spec, n_bootstraps=b, random_state=0, backend=backend)
    t1 = time.perf_counter()
    vals = res.values()
    t2 = time.perf_counter()
    if gc_off:
        gc.enable()
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "seconds": t2 - t0,
        "bootstrap_s": t1 - t0,
        "values_s": t2 - t1,
        "peak_rss_mb": rss1 / 1024.0,
        "rss_before_mb": rss0 / 1024.0,
        "array_mb": vals.nbytes / 2**20,
        "checksum": float(np.asarray(vals, dtype=np.float64)[:: max(1, b // 97)].sum()),
    }


def _process_reduce(spec: Any, x: np.ndarray, b: int, workers: int) -> np.ndarray:
    """A stdlib process-pool mean reduce over the numpy chunk kernels (forkserver start)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    from tsbootstrap import dispatch as dsp
    from tsbootstrap.api import _ensure_executors

    _ensure_executors()
    prepared = dsp.get_preparer(spec)(
        np.asarray(x, dtype=np.float64).reshape(len(x), -1), spec, None
    )
    root = np.random.SeedSequence(0)
    chunk = 1024
    starts = list(range(0, b, chunk))
    ctx = mp.get_context("forkserver")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = [
            ex.submit(_process_chunk, prepared, spec, root, s, min(chunk, b - s), len(x))
            for s in starts
        ]
        return np.concatenate([f.result() for f in futs])


def _process_chunk(
    prepared: Any, spec: Any, root: Any, start: int, count: int, n: int
) -> np.ndarray:
    from tsbootstrap import dispatch as dsp
    from tsbootstrap.api import _ensure_executors
    from tsbootstrap.rng import ReplicateStreams

    _ensure_executors()
    kernel = dsp._CHUNK_KERNELS[type(spec)]
    vals, _ = kernel(prepared, spec, ReplicateStreams(root, start, count), n, np.dtype(np.float64))
    return vals.mean(axis=1, dtype=np.float64).reshape(count, -1)


def _run_reduce(method: str, n: int, b: int, d: int, driver: str) -> dict[str, Any]:
    import tsbootstrap as tsb

    x = _data(method, n, d)
    spec = _method(method)
    workers = int(os.environ.get("BENCH_WORKERS", "4"))
    if driver == "process":
        _process_reduce(spec, x, 256, workers)  # start the server and warm the workers' imports
        t0 = time.perf_counter()
        out = _process_reduce(spec, x, b, workers)
        t1 = time.perf_counter()
    else:
        tsb.bootstrap_reduce(
            x, method=spec, statistic="mean", n_bootstraps=64, random_state=1, backend=driver
        )
        t0 = time.perf_counter()
        out = tsb.bootstrap_reduce(
            x, method=spec, statistic="mean", n_bootstraps=b, random_state=0, backend=driver
        ).statistics
        t1 = time.perf_counter()
    return {
        "seconds": t1 - t0,
        "workers": workers if driver == "process" else 1,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "checksum": float(np.asarray(out).sum()),
    }


def _seed_micro(b: int) -> dict[str, Any]:
    """Per-replicate cost of deriving each replicate's generator, then of its first draw."""
    from tsbootstrap import rng as rngmod

    root = np.random.SeedSequence(0)
    out: dict[str, Any] = {"B": b}
    t0 = time.perf_counter()
    children = rngmod.child_seed_sequences(root, 0, b)
    t1 = time.perf_counter()
    gens = [np.random.default_rng(c) for c in children]
    t2 = time.perf_counter()
    for g in gens:
        g.integers(0, 200, size=200, dtype=np.int32)
    t3 = time.perf_counter()
    out["legacy_child_us"] = (t1 - t0) / b * 1e6
    out["legacy_generator_us"] = (t2 - t1) / b * 1e6
    out["legacy_draw_us"] = (t3 - t2) / b * 1e6
    streams_cls = getattr(rngmod, "ReplicateStreams", None)
    if streams_cls is not None:
        t4 = time.perf_counter()
        chunks = [streams_cls(root, s, min(2048, b - s)) for s in range(0, b, 2048)]
        t5 = time.perf_counter()
        for chunk in chunks:
            for g in chunk:
                g.integers(0, 200, size=200, dtype=np.int32)
        t6 = time.perf_counter()
        out["batched_state_us"] = (t5 - t4) / b * 1e6
        out["batched_load_and_draw_us"] = (t6 - t5) / b * 1e6
    return out


def _select_by_svd_loop(x: np.ndarray, max_lag: int | None) -> int:
    """The per-order SVD order selection the QR path replaced (the before column)."""
    n = x.shape[0]
    upper = max_lag if max_lag is not None else int(np.ceil(10 * np.log10(n)))
    upper = max(1, min(upper, n // 2 - 1))
    y = x[upper:]
    m = y.shape[0]
    full = np.column_stack([np.ones(m), *(x[upper - j : n - j] for j in range(1, upper + 1))])
    best, best_k = np.inf, 1
    for k in range(1, upper + 1):
        des = full[:, : k + 1]
        beta = np.linalg.lstsq(des, y, rcond=None)[0]
        r = y - des @ beta
        ic = m * float(np.log(float(r @ r) / m)) + np.log(m) * (k + 1)
        if ic < best:
            best, best_k = ic, k
    return best_k


def _sieve_setup(n: int) -> dict[str, Any]:
    import tsbootstrap as tsb
    from tsbootstrap.model.fit import select_ar_order

    x = _ar1(n)
    tsb.bootstrap(x[:500], method=tsb.SieveAR(), n_bootstraps=2, random_state=0).values()
    out: dict[str, Any] = {}
    t0 = time.perf_counter()
    k_qr = select_ar_order(x)
    out["select_qr_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    k_svd = _select_by_svd_loop(x, None)
    out["select_svd_loop_s"] = time.perf_counter() - t0
    out["same_order"] = k_qr == k_svd
    out["order"] = k_qr
    t0 = time.perf_counter()
    tsb.bootstrap(x, method=tsb.SieveAR(), n_bootstraps=1, random_state=0).values()
    out["seconds"] = time.perf_counter() - t0  # setup-inclusive bootstrap, B=1
    return out


def _buffer_micro(mb: int) -> dict[str, Any]:
    """Fresh-buffer first write versus a write into an already-touched buffer."""
    src = np.random.default_rng(0).standard_normal(mb * 2**20 // 8)
    touched = np.empty_like(src)
    touched[:] = 0.0
    t0 = time.perf_counter()
    touched[:] = src
    t1 = time.perf_counter()
    fresh = np.empty_like(src)
    t2 = time.perf_counter()
    fresh[:] = src
    t3 = time.perf_counter()
    t4 = time.perf_counter()
    both = np.concatenate([src[: src.size // 2], src[src.size // 2 :]])
    t5 = time.perf_counter()
    del both
    return {
        "megabytes": mb,
        "touched_write_s": t1 - t0,
        "fresh_alloc_s": t2 - t1,
        "fresh_first_write_s": t3 - t2,
        "concatenate_s": t5 - t4,
    }


def _timed(fn: Callable[[], Any], repeats: int = 5) -> float:
    """The median wall time of ``fn`` over ``repeats`` calls, after one untimed call."""
    fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def _import_cell(accel: bool) -> dict[str, Any]:
    """Seconds to ``import tsbootstrap``, with numba importable or blocked."""
    if not accel:
        sys.modules["numba"] = None  # type: ignore[assignment]
    t0 = time.perf_counter()
    import tsbootstrap  # noqa: F401

    return {"seconds": time.perf_counter() - t0, "accel": accel}


def _cold_cell(method: str, backend: str, fresh_cache: bool) -> dict[str, Any]:
    """Seconds for the first ``bootstrap`` call of a fresh process."""
    import tempfile

    if fresh_cache:
        os.environ["NUMBA_CACHE_DIR"] = tempfile.mkdtemp(prefix="tsb-bench-numba-")
    import tsbootstrap as tsb

    x = _data(method, 200, 2 if method == "var" else 1)
    spec = _method(method)
    t0 = time.perf_counter()
    tsb.bootstrap(x, method=spec, n_bootstraps=100, random_state=0, backend=backend).values()
    return {"seconds": time.perf_counter() - t0, "fresh_numba_cache": fresh_cache}


def _small_b_cell() -> dict[str, Any]:
    """Per-call time of a tiny run: the fixed cost of one API call."""
    import tsbootstrap as tsb

    x = _ar1(100)
    out: dict[str, Any] = {}
    for method in ("iid", "moving", "ar"):
        spec = _method(method)
        for b in (1, 10):
            out[f"bootstrap_{method}_B{b}_us"] = 1e6 * _timed(
                lambda spec=spec, b=b: tsb.bootstrap(x, method=spec, n_bootstraps=b, random_state=0),
                repeats=200,
            )
    spec = _method("iid")
    out["reduce_mean_B10_us"] = 1e6 * _timed(
        lambda: tsb.bootstrap_reduce(x, method=spec, statistic="mean", n_bootstraps=10, random_state=0),
        repeats=200,
    )
    return out


def _reducer_cell() -> dict[str, Any]:
    """A named reducer against a Python callable per replicate and a vectorised one."""
    import tsbootstrap as tsb

    x = _ar1(1000)
    spec = _method("moving")
    kw: dict[str, Any] = {"method": spec, "n_bootstraps": 2000, "random_state": 0}
    return {
        "named_s": _timed(lambda: tsb.bootstrap_reduce(x, statistic="mean", **kw), repeats=3),
        "callable_s": _timed(
            lambda: tsb.bootstrap_reduce(x, statistic=lambda v, i: v.mean(), **kw), repeats=3
        ),
        "vectorized_s": _timed(
            lambda: tsb.bootstrap_reduce(
                x, statistic=lambda v, i: v.mean(axis=1), vectorized=True, **kw
            ),
            repeats=3,
        ),
    }


def _panel_cell(n_series: int) -> dict[str, Any]:
    """Seconds per series of a ragged panel reduce, numpy backend."""
    import tsbootstrap as tsb

    rng = np.random.default_rng(0)
    panel = [rng.standard_normal(int(rng.integers(50, 150))) for _ in range(n_series)]
    spec = _method("moving")
    secs = _timed(
        lambda: tsb.bootstrap_reduce_panel(
            panel, method=spec, statistic="mean", n_bootstraps=200, random_state=0
        ),
        repeats=3,
    )
    return {"seconds": secs, "n_series": n_series, "per_series_ms": 1e3 * secs / n_series}


def _chunk_cell() -> dict[str, Any]:
    """Per-chunk overhead of the numpy chunk loop, contract check included.

    The same run is timed with one replicate per chunk and with one chunk; the difference
    over the number of extra chunks is the cost of a chunk (stream derivation, the contract
    assertion, slot hand-out).
    """
    import tsbootstrap as tsb
    from tsbootstrap import dispatch

    x = _ar1(200)
    spec = _method("moving")
    b = 2000
    big = _timed(lambda: tsb.bootstrap(x, method=spec, n_bootstraps=b, random_state=0), repeats=3)
    saved = dispatch._CHUNK_BYTES
    dispatch._CHUNK_BYTES = 1
    try:
        small = _timed(
            lambda: tsb.bootstrap(x, method=spec, n_bootstraps=b, random_state=0), repeats=3
        )
    finally:
        dispatch._CHUNK_BYTES = saved
    return {
        "one_chunk_s": big,
        "one_replicate_chunks_s": small,
        "per_chunk_us": 1e6 * (small - big) / (b - 1),
    }


def _conf_int_cell() -> dict[str, Any]:
    """``conf_int`` for each interval kind, with the BCa jackknife timed on its own."""
    import tsbootstrap as tsb
    from tsbootstrap.uq import conf_int
    from tsbootstrap.uq.classical import jackknife_acceleration

    x = _ar1(500)
    out: dict[str, Any] = {}
    for kind, spec in (
        ("percentile", _method("moving")),
        ("basic", _method("moving")),
        ("studentized", _method("moving")),
        ("bca", _method("iid")),
    ):
        out[f"{kind}_s"] = _timed(
            lambda kind=kind, spec=spec: conf_int(
                x, "mean", method=spec, kind=kind, n_bootstraps=999, random_state=0
            ),
            repeats=3,
        )
    out["bca_jackknife_only_s"] = _timed(
        lambda: jackknife_acceleration(x, lambda v, _idx: np.atleast_1d(v.mean())), repeats=3
    )
    return out


def _latency_cell(name: str) -> dict[str, Any]:
    cells: dict[str, Callable[[], dict[str, Any]]] = {
        "import_accel": lambda: _import_cell(True),
        "import_noaccel": lambda: _import_cell(False),
        "cold_iid_numpy": lambda: _cold_cell("iid", "numpy", False),
        "cold_iid_compiled_fresh": lambda: _cold_cell("iid", "compiled", True),
        "cold_iid_compiled_cached": lambda: _cold_cell("iid", "compiled", False),
        "cold_var_compiled_fresh": lambda: _cold_cell("var", "compiled", True),
        "small_b": _small_b_cell,
        "reducer_modes": _reducer_cell,
        "panel_200_series": lambda: _panel_cell(200),
        "chunk_overhead": _chunk_cell,
        "conf_int_kinds": _conf_int_cell,
    }
    return cells[name]()


# The field that ranks the repeats of a cell whose result has no ``seconds`` field.
_LATENCY_KEY = {
    "small_b": "bootstrap_iid_B1_us",
    "reducer_modes": "named_s",
    "chunk_overhead": "one_chunk_s",
    "conf_int_kinds": "percentile_s",
}

LATENCY_CELLS = (
    "import_accel",
    "import_noaccel",
    "cold_iid_numpy",
    "cold_iid_compiled_fresh",
    "cold_iid_compiled_cached",
    "cold_var_compiled_fresh",
    "small_b",
    "reducer_modes",
    "panel_200_series",
    "chunk_overhead",
    "conf_int_kinds",
)


def _child(argv: list[str]) -> None:
    kind = argv[0]
    if kind == "values":
        method, n, b, d, backend = VALUES_CELLS[argv[1]]
        if not _supported(method, backend, "values"):
            print(json.dumps({"unsupported": True}))
            return
        print(json.dumps(_run_values(method, n, b, d, backend, gc_off=argv[2] == "gcoff")))
    elif kind == "reduce":
        method, n, b, d, driver = REDUCE_CELLS[argv[1]]
        if not _supported(method, driver if driver == "compiled" else "numpy", "reduce"):
            print(json.dumps({"unsupported": True}))
            return
        print(json.dumps(_run_reduce(method, n, b, d, driver)))
    elif kind == "seed":
        print(json.dumps(_seed_micro(int(argv[1]))))
    elif kind == "buffer":
        print(json.dumps(_buffer_micro(int(argv[1]))))
    elif kind == "sieve_setup":
        print(json.dumps(_sieve_setup(int(argv[1]))))
    elif kind == "latency":
        print(json.dumps(_latency_cell(argv[1])))


def _spawn(args: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, __file__, "--child", *args], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        return {"error": proc.stderr[-2000:]}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _summarise(runs: list[dict[str, Any]], key: str = "seconds") -> dict[str, Any]:
    good = [r for r in runs if "error" not in r and not r.get("unsupported")]
    if not good:
        return {"runs": runs}
    vals = sorted(r[key] for r in good)
    best = min(good, key=lambda r: r[key])
    return {
        **best,
        "min": vals[0],
        "max": vals[-1],
        "n": len(vals),
        "peak_rss_mb_max": max(r.get("peak_rss_mb", 0.0) for r in good),
    }


def _provenance() -> dict[str, Any]:
    import numba
    import scipy

    import tsbootstrap

    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    return {
        "git_sha": os.environ.get("BOX_RUN_SOURCE_HEAD") or sha,
        "snapshot": os.environ.get("BOX_RUN_SNAPSHOT"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "scipy": scipy.__version__,
        "tsbootstrap": getattr(tsbootstrap, "__version__", "?"),
        "cpu_count": os.cpu_count(),
        "machine": platform.machine(),
        "loadavg_start": os.getloadavg(),
        "threads": {k: os.environ.get(k) for k in _THREAD_ENV},
        "start_method_process_cells": "forkserver",
    }


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _child(sys.argv[2:])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--only", default="", help="comma-separated substrings of cell names")
    args = ap.parse_args()
    for k in _THREAD_ENV:
        os.environ.setdefault(k, "1" if k != "NUMBA_NUM_THREADS" else str(os.cpu_count()))
    want: Callable[[str], bool] = (
        (lambda name: any(s in name for s in args.only.split(",")))
        if args.only
        else (lambda _: True)
    )
    result: dict[str, Any] = {
        "provenance": _provenance(),
        "values": {},
        "values_gcoff": {},
        "reduce": {},
    }
    for name in VALUES_CELLS:
        if not want(name):
            continue
        for mode, slot in (("gcon", "values"), ("gcoff", "values_gcoff")):
            if mode == "gcoff" and name.endswith("_compiled"):
                continue
            result[slot][name] = _summarise(
                [_spawn(["values", name, mode]) for _ in range(args.repeats)]
            )
            print(slot, name, json.dumps(result[slot][name])[:240], flush=True)
    for name in REDUCE_CELLS:
        if want(name):
            result["reduce"][name] = _summarise(
                [_spawn(["reduce", name]) for _ in range(args.repeats)]
            )
            print("reduce", name, json.dumps(result["reduce"][name])[:240], flush=True)
    if want("seed"):
        result["seed"] = [_spawn(["seed", "100000"]) for _ in range(args.repeats)]
    for n_setup in (2_000, 20_000, 200_000):
        name = f"sieve_setup_n{n_setup}"
        if want(name):
            result.setdefault("sieve_setup", {})[name] = _summarise(
                [_spawn(["sieve_setup", str(n_setup)]) for _ in range(args.repeats)]
            )
            print("sieve_setup", name, json.dumps(result["sieve_setup"][name])[:240], flush=True)
    for name in LATENCY_CELLS:
        if want(name):
            if name.endswith("_compiled_cached"):
                _spawn(["latency", name])  # populates the on-disk numba cache; not timed
            result.setdefault("latency", {})[name] = _summarise(
                [_spawn(["latency", name]) for _ in range(args.repeats)], key=_LATENCY_KEY.get(name, "seconds")
            )
            print("latency", name, json.dumps(result["latency"][name])[:240], flush=True)
    if want("buffer"):
        result["buffer"] = [_spawn(["buffer", "160"]) for _ in range(args.repeats)]
    result["provenance"]["loadavg_end"] = os.getloadavg()
    Path(args.json).write_text(json.dumps(result, indent=1))
    print("WROTE", args.json)


if __name__ == "__main__":
    main()
