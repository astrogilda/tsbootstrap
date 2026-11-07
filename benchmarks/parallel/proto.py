"""Prototype process-parallel drivers for the numpy chunk kernels (stage 2 of the study).

Every driver runs the SAME per-spec numpy chunk kernel over the SAME fixed chunk grid
(``dispatch._CHUNK_SIZE`` replicates per chunk, replicate i bound to child i of the root
SeedSequence), so each must reproduce ``backend="numpy"`` byte for byte for any worker count.
They differ only in how work and results cross processes:

- ``ppe``        ProcessPoolExecutor, results pickled back (stdlib baseline);
- ``ppe_shm``    ProcessPoolExecutor, workers write into parent-owned SharedMemory outputs;
- ``mizu``       pymizu Pool.map over chunk ids, results pickled back;
- ``mizu_shm``   pymizu Pool.map, workers write into parent-owned SharedMemory outputs.

Worker functions live at module top level so both pools can import them by reference.
"""

from __future__ import annotations

from multiprocessing import shared_memory

import numpy as np

from tsbootstrap.dispatch import _CHUNK_KERNELS, _CHUNK_SIZE


def _ensure() -> None:
    import tsbootstrap.api as api

    api._ensure_executors()


def child_seeds(root_desc: tuple, start: int, size: int) -> list[np.random.SeedSequence]:
    """Children ``start .. start+size`` of the root, built directly (no O(B) spawn).

    numpy's ``SeedSequence.spawn`` constructs child ``i`` as
    ``SeedSequence(entropy, spawn_key=spawn_key + (n_children_spawned + i,), pool_size)``;
    building the same object here makes worker-side derivation O(chunk), not O(B).
    """
    entropy, spawn_key, pool_size, first = root_desc
    return [
        np.random.SeedSequence(entropy, spawn_key=(*spawn_key, first + start + i), pool_size=pool_size)
        for i in range(size)
    ]


def root_descriptor(root: np.random.SeedSequence, n: int, advance: bool) -> tuple:
    """Describe the root for workers.

    ``n_children_spawned`` is read-only, so a caller-owned root is advanced by a real
    ``spawn(n)`` (O(n) objects, discarded) to keep its counter exactly where the serial path
    leaves it. A root private to this call (an int, ``None`` or Generator seed) needs no advance.
    """
    desc = (root.entropy, tuple(root.spawn_key), root.pool_size, root.n_children_spawned)
    if advance:
        root.spawn(n)
    return desc


def chunk_task(c, prepared, spec, root_desc, B, n_obs, dtype_str, chunk_size=_CHUNK_SIZE):
    _ensure()
    start = c * chunk_size
    size = min(chunk_size, B - start)
    kernel = _CHUNK_KERNELS[type(spec)]
    return kernel(prepared, spec, child_seeds(root_desc, start, size), n_obs, np.dtype(dtype_str))


def chunk_task_shm(c, prepared, spec, root_desc, B, n_obs, dtype_str, vname, vshape, iname,
                   chunk_size=_CHUNK_SIZE):
    values, indices = chunk_task(c, prepared, spec, root_desc, B, n_obs, dtype_str, chunk_size)
    start = c * chunk_size
    vs = shared_memory.SharedMemory(name=vname, track=False)
    try:
        out = np.ndarray(vshape, dtype=np.dtype(dtype_str), buffer=vs.buf)
        out[start : start + values.shape[0]] = values
        del out
    finally:
        vs.close()
    if iname is not None and indices is not None:
        ish = shared_memory.SharedMemory(name=iname, track=False)
        try:
            outi = np.ndarray((B, n_obs), dtype=np.int32, buffer=ish.buf)
            outi[start : start + indices.shape[0]] = indices
            del outi
        finally:
            ish.close()
    return None


def reduce_task(c, prepared, spec, root_desc, B, n_obs, dtype_str, chunk_size=_CHUNK_SIZE):
    values, _ = chunk_task(c, prepared, spec, root_desc, B, n_obs, dtype_str, chunk_size)
    return np.ascontiguousarray(values.mean(axis=1).reshape(values.shape[0], -1), dtype=np.float64)


def warm(_i: int = 0) -> int:
    _ensure()
    return 0


def n_chunks(B: int, chunk_size: int = _CHUNK_SIZE) -> int:
    return -(-B // chunk_size)


def assemble(results, B):
    vals = [r[0] for r in results]
    idxs = [r[1] for r in results]
    v = vals[0] if len(vals) == 1 else np.concatenate(vals)
    i = None if idxs[0] is None else (idxs[0] if len(idxs) == 1 else np.concatenate(idxs))
    return v, i


class ShmOut:
    """Parent-owned shared output buffers for one run."""

    def __init__(self, vshape, dtype, B, n_obs, with_indices):
        self.dtype = np.dtype(dtype)
        nbytes = int(np.prod(vshape)) * self.dtype.itemsize
        self.v = shared_memory.SharedMemory(create=True, size=max(nbytes, 1))
        self.i = (
            shared_memory.SharedMemory(create=True, size=max(B * n_obs * 4, 1)) if with_indices else None
        )
        self.vshape = vshape
        self.B, self.n_obs = B, n_obs

    def arrays(self, copy=True):
        v = np.ndarray(self.vshape, dtype=self.dtype, buffer=self.v.buf)
        i = None if self.i is None else np.ndarray((self.B, self.n_obs), dtype=np.int32, buffer=self.i.buf)
        if copy:
            v = v.copy()
            i = None if i is None else i.copy()
        return v, i

    def close(self):
        for s in (self.v, self.i):
            if s is not None:
                s.close()
                s.unlink()
