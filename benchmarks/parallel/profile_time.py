# /// script
# requires-python = ">=3.10"
# dependencies = ["tsbootstrap[accel,models]"]
# [tool.uv.sources]
# tsbootstrap = { path = "../..", editable = true }
# ///
"""Where does tsbootstrap's time go? Stage 1 of the process-parallel backend study.

Decomposes one ``bootstrap()`` call on the numpy backend into its serial phases (setup and
fit, seed spawn, generator construction, the vectorised chunk kernel, concatenation, and
sample assembly), times the compiled backend beside it, and prices the hand-off a
process-parallel backend would pay: pickling the prepared context out, and returning the
(B, n[, d]) values and (B, n) indices either by pickle or by one copy into shared memory.

Run on a quiet many-core box; prints one JSON line per cell.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from statistics import median

import numpy as np

import tsbootstrap.api as api
from tsbootstrap import (
    AR,
    IID,
    VAR,
    MovingBlock,
    ResidualBootstrap,
    SieveAR,
    StationaryBlock,
    bootstrap,
    bootstrap_reduce,
)
from tsbootstrap.dispatch import _CHUNK_KERNELS, _CHUNK_SIZE, _chunked_seeds
from tsbootstrap.rng import generators_from_seeds, resolve_seed_sequence, spawn_seed_sequences


def _ar1(n: int) -> np.ndarray:
    rng = np.random.default_rng(0)
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


METHODS = {
    "iid": (lambda: IID(), "uni"),
    "moving": (lambda: MovingBlock(block_length=20), "uni"),
    "stationary": (lambda: StationaryBlock(avg_block_length=20), "uni"),
    "ar2": (lambda: ResidualBootstrap(model=AR(order=2)), "uni"),
    "sieve": (lambda: SieveAR(), "uni"),
    "var3": (lambda: ResidualBootstrap(model=VAR(order=1)), "var"),
}


def best(fn, repeats: int) -> tuple[float, float]:
    fn()  # settle (JIT, caches)
    s = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        s.append(time.perf_counter() - t)
    return min(s), median(s)


def cell(name: str, n: int, B: int, repeats: int) -> dict:
    make, kind = METHODS[name]
    spec = make()
    x = _ar1(n) if kind == "uni" else _var1(n, 3)
    out: dict = {"method": name, "n": n, "B": B}

    out["numpy_total"] = best(lambda: bootstrap(x, method=spec, n_bootstraps=B, random_state=0), repeats)
    try:
        out["compiled_total"] = best(
            lambda: bootstrap(x, method=spec, n_bootstraps=B, random_state=0, backend="compiled"),
            repeats,
        )
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        out["compiled_total"] = f"unsupported: {type(exc).__name__}"
    out["numpy_reduce_mean"] = best(
        lambda: bootstrap_reduce(x, method=spec, statistic="mean", n_bootstraps=B, random_state=0),
        repeats,
    )

    # Phase decomposition of the numpy values path.
    setup = api._setup_run(x, spec, B, 0, None)
    out["setup_fit"] = best(lambda: api._setup_run(x, spec, B, 0, None), repeats)
    out["spawn"] = best(lambda: spawn_seed_sequences(resolve_seed_sequence(0), B), repeats)
    seeds = spawn_seed_sequences(resolve_seed_sequence(0), B)
    out["generators"] = best(lambda: generators_from_seeds(seeds), repeats)
    kernel = _CHUNK_KERNELS[type(spec)]
    chunks = _chunked_seeds(resolve_seed_sequence(0), B)

    def run_kernels():
        return [kernel(setup.prepared, spec, c, setup.n_obs, setup.sim_dtype) for c in chunks]

    out["kernels_incl_generators"] = best(run_kernels, repeats)
    res = run_kernels()
    vals = [r[0] for r in res]
    idxs = [r[1] for r in res if r[1] is not None]
    out["concat"] = best(lambda: (np.concatenate(vals) if len(vals) > 1 else vals[0]), repeats)
    values_b = np.concatenate(vals) if len(vals) > 1 else vals[0]
    indices_b = (np.concatenate(idxs) if len(idxs) > 1 else idxs[0]) if idxs else None
    out["assemble"] = best(lambda: api._assemble_samples(values_b, indices_b, setup.was_1d, B), repeats)

    # Hand-off a process backend would pay.
    out_bytes = values_b.nbytes + (0 if indices_b is None else indices_b.nbytes)
    out["out_bytes"] = out_bytes
    payload = (values_b, indices_b)

    def pickle_rt():
        return pickle.loads(pickle.dumps(payload, protocol=5))

    out["out_pickle_roundtrip"] = best(pickle_rt, repeats)
    dst_v = np.empty_like(values_b)
    dst_i = None if indices_b is None else np.empty_like(indices_b)

    def one_copy():
        np.copyto(dst_v, values_b)
        if dst_i is not None:
            np.copyto(dst_i, indices_b)

    out["out_one_copy"] = best(one_copy, repeats)
    prep = pickle.dumps((setup.prepared, spec), protocol=5)
    out["prepared_bytes"] = len(prep)
    out["prepared_pickle_roundtrip"] = best(lambda: pickle.loads(pickle.dumps((setup.prepared, spec), protocol=5)), repeats)
    out["n_chunks"] = len(chunks)
    out["chunk_size"] = _CHUNK_SIZE
    return out


def main() -> None:
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    grid = [
        (m, n, B)
        for m in METHODS
        for n in (200, 2000, 20000)
        for B in (999, 10000, 100000)
        if n * B <= 100_000_000
    ]
    for m, n, B in grid:
        print(json.dumps(cell(m, n, B, repeats)), flush=True)


if __name__ == "__main__":
    main()
