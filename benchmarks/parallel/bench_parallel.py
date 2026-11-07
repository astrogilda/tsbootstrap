# /// script
# requires-python = ">=3.13"  # SharedMemory(track=False) needs 3.13
# dependencies = ["tsbootstrap[accel,models]", "pymizu[numpy]==0.1.0"]
# [tool.uv.sources]
# tsbootstrap = { path = "../..", editable = true }
# [tool.uv]
# # pymizu 0.1.0 (2026-11-04) is younger than the project's 7-day cooldown; this scoped
# # override applies to this benchmark-only script, never to the package lock.
# exclude-newer-package = { pymizu = "2026-11-05T00:00:00Z" }
# ///
"""Stage 2: process-parallel drivers against the numpy and compiled backends.

For each workload cell, times backend="numpy" (serial reference), backend="compiled"
(numba threads, where supported), and the four process drivers in proto.py at several
worker counts, on persistent warmed pools (pool start-up is timed separately). Every
process result is compared byte for byte with the numpy reference.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from statistics import median

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pymizu  # noqa: E402

import tsbootstrap.api as api  # noqa: E402
from benchmarks.parallel import proto  # noqa: E402
from benchmarks.parallel.profile_time import METHODS, _ar1, _var1  # noqa: E402
from tsbootstrap import bootstrap  # noqa: E402
from tsbootstrap.dispatch import get_values_executor  # noqa: E402
from tsbootstrap.rng import resolve_seed_sequence  # noqa: E402


def best(fn, repeats):
    fn()
    gc.collect()
    s = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        s.append(time.perf_counter() - t)
        gc.collect()  # release pickled results and mapped regions before the next sample
    return round(min(s), 5), round(median(s), 5)


def reference(x, spec, B):
    setup = api._setup_run(x, spec, B, 0, None)
    v, i = get_values_executor(spec, "numpy")(
        setup.prepared, spec, setup.root_ss, B, setup.n_obs, setup.sim_dtype
    )
    return setup, v, i


def run_ppe(ex, setup, spec, B, shm, idx=True):
    desc = proto.root_descriptor(resolve_seed_sequence(0), B, advance=False)
    nc = proto.n_chunks(B)
    args = (setup.prepared, spec, desc, B, setup.n_obs, setup.sim_dtype.str)
    if not shm:
        futs = [ex.submit(proto.chunk_task, c, *args) for c in range(nc)]
        return proto.assemble([f.result() for f in futs], B)
    vshape = (B, *VTAIL[0])
    out = proto.ShmOut(vshape, setup.sim_dtype, B, setup.n_obs, idx)
    try:
        futs = [
            ex.submit(proto.chunk_task_shm, c, *args, out.v.name, vshape, None if out.i is None else out.i.name) for c in range(nc)
        ]
        for f in futs:
            f.result()
        return out.arrays(copy=True)
    finally:
        out.close()


def run_mizu(pool, setup, spec, B, shm, idx=True):
    desc = proto.root_descriptor(resolve_seed_sequence(0), B, advance=False)
    nc = proto.n_chunks(B)
    args = (setup.prepared, spec, desc, B, setup.n_obs, setup.sim_dtype.str)
    if not shm:
        res = pool.map(proto.chunk_task, range(nc), args=args, n_chunks=nc).collect()
        return proto.assemble(res, B)
    vshape = (B, *VTAIL[0])
    out = proto.ShmOut(vshape, setup.sim_dtype, B, setup.n_obs, idx)
    try:
        pool.map(
            proto.chunk_task_shm, range(nc), args=(*args, out.v.name, vshape, None if out.i is None else out.i.name), n_chunks=nc
        ).collect()
        return out.arrays(copy=True)
    finally:
        out.close()


VTAIL = [()]


def same(a, b):
    va, ia = a
    vb, ib = b
    if va.shape != vb.shape or va.tobytes() != vb.tobytes():
        return False
    if ia is None:
        return True  # recursive methods: indices None in the reference
    return ib is not None and ia.tobytes() == ib.tobytes()


def run_reduce(kind, pool_or_ex, setup, spec, B):
    desc = proto.root_descriptor(resolve_seed_sequence(0), B, advance=False)
    nc = proto.n_chunks(B)
    args = (setup.prepared, spec, desc, B, setup.n_obs, setup.sim_dtype.str)
    if kind == "ppe":
        parts = [f.result() for f in [pool_or_ex.submit(proto.reduce_task, c, *args) for c in range(nc)]]
    else:
        parts = pool_or_ex.map(proto.reduce_task, range(nc), args=args, n_chunks=nc).collect()
    return np.concatenate(parts)


def serial_reduce(setup, spec, B):
    desc = proto.root_descriptor(resolve_seed_sequence(0), B, advance=False)
    args = (setup.prepared, spec, desc, B, setup.n_obs, setup.sim_dtype.str)
    return np.concatenate([proto.reduce_task(c, *args) for c in range(proto.n_chunks(B))])


CELLS = [
    (m, n, B)
    for m in ("iid", "moving", "stationary", "ar2", "sieve", "var3")
    for (n, B) in ((200, 10000), (200, 40000), (2000, 10000), (5000, 4096))
]


def main():
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    workers = [int(w) for w in os.environ.get("WORKERS", "1,2,4,8").split(",")]
    print(json.dumps({"python": sys.version.split()[0], "numpy": np.__version__,
                      "cpus": os.cpu_count()}), flush=True)
    for m, n, B in CELLS:  # single-process baselines, once per cell
        make, kind = METHODS[m]
        spec = make()
        x = _ar1(n) if kind == "uni" else _var1(n, 3)
        setup, v, i = reference(x, spec, B)
        row = {"base": m, "n": n, "B": B, "out_mb": round((v.nbytes + (0 if i is None else i.nbytes)) / 1e6, 1)}
        row["numpy_values"] = best(lambda: get_values_executor(spec, "numpy")(
            setup.prepared, spec, resolve_seed_sequence(0), B, setup.n_obs, setup.sim_dtype), repeats)
        row["numpy_reduce_mean"] = best(lambda: serial_reduce(setup, spec, B), repeats)
        try:
            bootstrap(x, method=spec, n_bootstraps=B, random_state=0, backend="compiled")
            row["compiled_values"] = best(lambda: get_values_executor(spec, "compiled")(
                setup.prepared, spec, resolve_seed_sequence(0), B, setup.n_obs, setup.sim_dtype), repeats)
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
            row["compiled_values"] = f"unsupported: {type(exc).__name__}"
        print(json.dumps(row), flush=True)
        del v, i
        gc.collect()

    for w in workers:  # one pool pair alive at a time
        t = time.perf_counter()
        pool = pymizu.Pool.create(w)
        pool.map(proto.warm, range(w * 4)).collect()
        t_m = time.perf_counter() - t
        t = time.perf_counter()
        ex = ProcessPoolExecutor(max_workers=w)
        list(ex.map(proto.warm, range(w * 4)))
        t_e = time.perf_counter() - t
        print(json.dumps({"startup": w, "mizu_s": round(t_m, 4), "ppe_s": round(t_e, 4)}), flush=True)
        for m, n, B in CELLS:
            make, kind = METHODS[m]
            spec = make()
            x = _ar1(n) if kind == "uni" else _var1(n, 3)
            setup, v, i = reference(x, spec, B)
            VTAIL[0] = v.shape[1:]
            ref = (v, i)
            has_idx = i is not None
            row = {"method": m, "n": n, "B": B, "w": w}
            ident = {}
            for name, fn in (
                ("ppe", lambda: run_ppe(ex, setup, spec, B, False)),
                ("ppe_shm", lambda: run_ppe(ex, setup, spec, B, True, has_idx)),
                ("mizu", lambda: run_mizu(pool, setup, spec, B, False)),
                ("mizu_shm", lambda: run_mizu(pool, setup, spec, B, True, has_idx)),
            ):
                ident[name] = same(ref, fn())
                gc.collect()
                row[name] = best(fn, repeats)
            red_ref = serial_reduce(setup, spec, B)
            for name, fn in (
                ("ppe_reduce", lambda: run_reduce("ppe", ex, setup, spec, B)),
                ("mizu_reduce", lambda: run_reduce("mizu", pool, setup, spec, B)),
            ):
                ident[name] = fn().tobytes() == red_ref.tobytes()
                row[name] = best(fn, repeats)
            row["bit_identical"] = all(ident.values())
            row["not_identical"] = [k for k, ok in ident.items() if not ok]
            print(json.dumps(row), flush=True)
            del ref, v, i
            gc.collect()
        pool.stop()
        ex.shutdown()


if __name__ == "__main__":
    main()
