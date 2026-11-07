# /// script
# requires-python = ">=3.10"
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
    s = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        s.append(time.perf_counter() - t)
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


def main():
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    workers = [int(w) for w in os.environ.get("WORKERS", "1,2,4,8").split(",")]
    cells = [
        (m, n, B)
        for m in ("iid", "moving", "stationary", "ar2", "sieve", "var3")
        for (n, B) in ((200, 10000), (200, 100000), (2000, 10000), (2000, 40000), (20000, 4096))
    ]

    # Pool start-up (create + import tsbootstrap in every worker), per worker count.
    pools, exs = {}, {}
    for w in workers:
        t = time.perf_counter()
        p = pymizu.Pool.create(w)
        p.map(proto.warm, range(w * 4)).collect()
        t_m = time.perf_counter() - t
        t = time.perf_counter()
        e = ProcessPoolExecutor(max_workers=w)
        list(e.map(proto.warm, range(w * 4)))
        t_e = time.perf_counter() - t
        pools[w], exs[w] = p, e
        print(json.dumps({"startup": w, "mizu_s": round(t_m, 4), "ppe_s": round(t_e, 4)}), flush=True)

    for m, n, B in cells:
        make, kind = METHODS[m]
        spec = make()
        x = _ar1(n) if kind == "uni" else _var1(n, 3)
        setup, v, i = reference(x, spec, B)
        ref = (v, i)
        VTAIL[0] = v.shape[1:]
        row = {"method": m, "n": n, "B": B, "out_mb": round((v.nbytes + (0 if i is None else i.nbytes)) / 1e6, 1)}
        row["numpy_values"] = best(
            lambda: get_values_executor(spec, "numpy")(
                setup.prepared, spec, resolve_seed_sequence(0), B, setup.n_obs, setup.sim_dtype
            ),
            repeats,
        )
        try:
            bootstrap(x, method=spec, n_bootstraps=B, random_state=0, backend="compiled")
            from tsbootstrap.dispatch import get_values_executor as gve

            row["compiled_values"] = best(
                lambda: gve(spec, "compiled")(
                    setup.prepared, spec, resolve_seed_sequence(0), B, setup.n_obs, setup.sim_dtype
                ),
                repeats,
            )
        except Exception as exc:  # noqa: BLE001
            row["compiled_values"] = f"unsupported: {type(exc).__name__}"
        ident = {}
        for w in workers:
            for name, fn in (
                ("ppe", lambda: run_ppe(exs[w], setup, spec, B, False)),
                ("ppe_shm", lambda: run_ppe(exs[w], setup, spec, B, True, i is not None)),
                ("mizu", lambda: run_mizu(pools[w], setup, spec, B, False)),
                ("mizu_shm", lambda: run_mizu(pools[w], setup, spec, B, True, i is not None)),
            ):
                ident[f"{name}_w{w}"] = same(ref, fn())
                row[f"{name}_w{w}"] = best(fn, repeats)
        row["bit_identical_all"] = all(ident.values())
        row["not_identical"] = [k for k, ok in ident.items() if not ok]
        print(json.dumps(row), flush=True)

    for p in pools.values():
        p.stop()
    for e in exs.values():
        e.shutdown()


if __name__ == "__main__":
    main()
