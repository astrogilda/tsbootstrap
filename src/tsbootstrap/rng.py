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
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from tsbootstrap.errors import RNGContractError

RandomStateLike = int | np.random.Generator | np.random.SeedSequence | None


@dataclass(frozen=True, slots=True)
class RandomStateInfo:
    """Provenance of the RNG root used for a run (recorded in run metadata)."""

    kind: str  # "int" | "generator" | "seed_sequence" | "none"
    entropy: int | tuple[int, ...] | None


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


def spawn_generators(root: np.random.SeedSequence, n: int) -> list[np.random.Generator]:
    """Return ``n`` independent generators, one per replicate index.

    ``generators[i]`` is the stream for bootstrap sample ``i`` and is stable
    regardless of how the work is parallelised.
    """
    if n < 0:
        raise RNGContractError("n must be non-negative", context={"n": n})
    return [np.random.default_rng(child) for child in root.spawn(n)]


@contextlib.contextmanager
def single_threaded_blas() -> Iterator[None]:
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
_warmed = False


def register_warmup(fn: Callable[[], None]) -> Callable[[], None]:
    """Register a no-argument callable that warms a compiled kernel."""
    _WARMUP_HOOKS.append(fn)
    return fn


def warmup_kernels() -> None:
    """Run every registered warm-up once. Warm-up failure never breaks a run."""
    global _warmed
    if _warmed:
        return
    for fn in _WARMUP_HOOKS:
        with contextlib.suppress(Exception):
            fn()
    _warmed = True


__all__ = [
    "RandomStateLike",
    "RandomStateInfo",
    "resolve_seed_sequence",
    "resolve_and_describe",
    "spawn_generators",
    "single_threaded_blas",
    "register_warmup",
    "warmup_kernels",
]
