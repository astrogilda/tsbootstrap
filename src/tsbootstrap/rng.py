"""Deterministic RNG contract for reproducible, parallel-safe bootstrapping.

The contract guarantees reproducible, worker-count-invariant streams:

- Each bootstrap replicate ``i`` gets its own independent ``Generator``, derived
  from ``SeedSequence.spawn(n)[i]`` and **bound to the sample index before
  dispatch**. Worker assignment, ``n_jobs``, and chunking therefore cannot
  change which stream sample ``i`` uses.
- ``random_state=int`` ⇒ identical samples for any ``n_jobs``.
- ``random_state=Generator`` ⇒ a 128-bit seed is drawn from it **once** (and
  recorded), so the run is reproducible from the recorded entropy and the live
  generator is never shared across processes.
- ``random_state=None`` ⇒ OS entropy; non-reproducible, but the drawn entropy is
  still recorded in run metadata for provenance.

Numerical determinism additionally requires single-threaded BLAS per worker
(see :func:`single_threaded_blas`), because multithreaded linear algebra can
reorder floating-point reductions.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass

import numpy as np

from tsbootstrap.errors import RNGContractError

RandomStateLike = int | np.random.Generator | np.random.SeedSequence | None


@dataclass(frozen=True, slots=True)
class RandomStateInfo:
    """Provenance of the RNG root used for a run (recorded in run metadata)."""

    kind: str  # "int" | "generator" | "seed_sequence" | "none"
    # SeedSequence.entropy is int | Sequence[int] | None; we mirror that so no
    # lossy conversion is forced at the call site.
    entropy: int | Sequence[int] | None


def resolve_seed_sequence(random_state: RandomStateLike) -> np.random.SeedSequence:
    """Resolve any accepted ``random_state`` to a :class:`numpy.random.SeedSequence`.

    A ``Generator`` is consumed once (a 128-bit seed is drawn) so the live
    generator is never shared across workers.
    """
    if random_state is None:
        return np.random.SeedSequence()
    if isinstance(random_state, np.random.SeedSequence):
        return random_state
    if isinstance(random_state, (int, np.integer)):
        seed = int(random_state)
        if seed < 0:
            raise RNGContractError(
                "random_state integer seed must be non-negative",
                context={"random_state": seed},
            )
        return np.random.SeedSequence(seed)
    if isinstance(random_state, np.random.Generator):
        entropy = tuple(int(x) for x in random_state.integers(0, 2**32, size=4))
        return np.random.SeedSequence(entropy)
    raise RNGContractError(
        f"Unsupported random_state type {type(random_state).__name__!r}",
        context={"type": type(random_state).__name__},
        hint="Pass an int, a numpy Generator or SeedSequence, or None.",
    )


def resolve_and_describe(
    random_state: RandomStateLike,
) -> tuple[np.random.SeedSequence, RandomStateInfo]:
    """Resolve the root SeedSequence once and describe it for run metadata.

    Resolving exactly once is important: a ``Generator`` input is consumed by
    resolution, so describing must read the resolved sequence, not re-draw.
    """
    root = resolve_seed_sequence(random_state)
    if random_state is None:
        kind = "none"
    elif isinstance(random_state, np.random.SeedSequence):
        kind = "seed_sequence"
    elif isinstance(random_state, (int, np.integer)):
        kind = "int"
    else:
        kind = "generator"
    entropy = root.entropy
    if isinstance(entropy, np.ndarray):  # pragma: no cover - numpy may box it
        entropy = tuple(int(x) for x in entropy)
    return root, RandomStateInfo(kind=kind, entropy=entropy)


def spawn_seed_sequences(root: np.random.SeedSequence, n: int) -> list[np.random.SeedSequence]:
    """Return ``n`` independent child SeedSequences, one per replicate index.

    These stateless seed objects -- not live Generators -- cross the executor seam, so a
    future compiled/GPU backend can derive its own counter-based keys from the seed entropy
    without instantiating NumPy Generators. ``seeds[i]`` is bound to bootstrap sample ``i``
    and is stable regardless of how the work is parallelised or chunked.
    """
    if n < 0:
        raise RNGContractError("n must be non-negative", context={"n": n})
    return list(root.spawn(n))


def generators_from_seeds(seeds: list[np.random.SeedSequence]) -> list[np.random.Generator]:
    """Materialize one NumPy Generator per seed -- the first step inside a NumPy executor."""
    return [np.random.default_rng(s) for s in seeds]


def spawn_generators(root: np.random.SeedSequence, n: int) -> list[np.random.Generator]:
    """Return ``n`` independent generators, one per replicate index (stable under chunking).

    Convenience for the non-executor forward-simulation path; the executor seam itself carries
    the SeedSequences (see :func:`spawn_seed_sequences`) and materializes generators locally.
    """
    return generators_from_seeds(spawn_seed_sequences(root, n))


def root_key_from(root_ss: np.random.SeedSequence) -> tuple[int, int]:
    """Pack a run's root SeedSequence into two uint64 halves for the compiled seam.

    ``generate_state`` is a pure, non-consuming read of the sequence's entropy (it does
    not touch the spawn counter), so it carries the full 128-bit root across the seam
    without disturbing the numpy path's child derivation. The compiled kernels derive each
    replicate's Philox key from these two scalars in their parallel loop (see
    :func:`tsbootstrap.block._compiled._replicate_key`), so the compiled path spawns no
    per-replicate SeedSequence and runs no O(B) Python key loop.
    """
    words = root_ss.generate_state(4, dtype=np.uint32)
    root_a = (int(words[0]) << 32) | int(words[1])
    root_b = (int(words[2]) << 32) | int(words[3])
    return root_a, root_b


@contextlib.contextmanager
def single_threaded_blas() -> Generator[None, None, None]:
    """Force single-threaded BLAS within the block, for bitwise reproducibility.

    No-op if ``threadpoolctl`` is unavailable (it ships with scikit-learn, a
    core dependency, but we degrade gracefully rather than hard-require it).
    """
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - threadpoolctl is normally present
        yield
        return
    with threadpool_limits(limits=1):
        yield


# --- compiled-kernel warm-up -------------------------------------------------
# Numba kernels compile on first call. Under a spawn-based parallel backend
# every worker would otherwise re-JIT, which can make parallel runs slower than
# serial. Engines register their kernels here; the entry point warms them once
# in the parent process before dispatching, so fork inherits the compiled code
# and spawn hits the on-disk cache.

_WARMUP_HOOKS: list[Callable[[], None]] = []
_WARMED: set[Callable[[], None]] = set()


def register_warmup(fn: Callable[[], None]) -> Callable[[], None]:
    """Register a no-argument callable that warms a compiled kernel."""
    _WARMUP_HOOKS.append(fn)
    return fn


def warmup_kernels() -> None:
    """Run each registered warm-up exactly once.

    Tracking which hooks have run (rather than a single "warmed" flag) means a
    hook registered after an earlier warm-up still runs on the next call, and
    repeated calls are no-ops. Warm-up failure never breaks a run.
    """
    for fn in _WARMUP_HOOKS:
        if fn in _WARMED:
            continue
        with contextlib.suppress(Exception):
            fn()
        _WARMED.add(fn)


__all__ = [
    "RandomStateLike",
    "RandomStateInfo",
    "resolve_seed_sequence",
    "resolve_and_describe",
    "spawn_seed_sequences",
    "generators_from_seeds",
    "spawn_generators",
    "root_key_from",
    "single_threaded_blas",
    "register_warmup",
    "warmup_kernels",
]
