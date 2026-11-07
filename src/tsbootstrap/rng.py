"""Deterministic RNG contract for reproducible bootstrapping.

- Each bootstrap replicate ``i`` gets its own independent stream, derived from child
  ``i`` of the run's root ``SeedSequence`` and **bound to the replicate index**, so the
  chunk size and ``B`` cannot change which stream replicate ``i`` uses.
- ``random_state=int`` or ``SeedSequence``: the run is a pure function of that value.
  A ``SeedSequence`` is read, never advanced (the same convention as
  ``numpy.random.default_rng(seed_sequence)``), so passing the same object twice repeats
  the run on both backends. For independent runs, pass ``seed_sequence.spawn(k)[j]``.
- ``random_state=Generator``: a 128-bit seed is drawn from it **once** (and recorded),
  so the run is reproducible from the recorded state and the live generator is never
  shared.
- ``random_state=None``: OS entropy; not reproducible by itself, but the drawn entropy
  is recorded in run metadata, so the run can be replayed.

Every run records ``seed_state`` (entropy, spawn key, pool size and the child offset), and
``np.random.SeedSequence(**seed_state)`` replays it exactly.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
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
    # A reused/spawned SeedSequence is not reconstructible from entropy alone:
    # spawn_key and n_children_spawned identify the exact point in its stream.
    state: dict[str, object]


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
    state = dict(root.state)
    state["entropy"] = entropy  # plain Python values, including array-backed user entropy
    return root, RandomStateInfo(kind=kind, entropy=entropy, state=state)


def child_seed_sequences(
    root: np.random.SeedSequence, start: int, count: int
) -> list[np.random.SeedSequence]:
    """Return children ``start .. start + count - 1`` of ``root`` without advancing it.

    Child ``i`` is exactly what ``root.spawn`` would return in position ``i``: the same
    entropy and pool size with ``spawn_key + (n_children_spawned + i,)``. Building it
    directly leaves ``root`` untouched (so a ``SeedSequence`` input is a value) and lets a
    chunked or streaming caller hold only the children of the current chunk.
    """
    if start < 0 or count < 0:
        raise RNGContractError(
            "start and count must be non-negative", context={"start": start, "count": count}
        )
    base = root.n_children_spawned + start
    return [
        np.random.SeedSequence(
            root.entropy, spawn_key=(*root.spawn_key, base + i), pool_size=root.pool_size
        )
        for i in range(count)
    ]


def spawn_seed_sequences(root: np.random.SeedSequence, n: int) -> list[np.random.SeedSequence]:
    """Return the first ``n`` child SeedSequences of ``root``, one per replicate index.

    ``seeds[i]`` is bound to bootstrap sample ``i`` and is stable however the work is
    chunked. ``root`` is not advanced (see :func:`child_seed_sequences`).
    """
    if n < 0:
        raise RNGContractError("n must be non-negative", context={"n": n})
    return child_seed_sequences(root, 0, n)


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
    # generate_state ignores the child offset, but the offset is part of the seed's value
    # (the numpy backend's children start there), so fold it in; offset 0 is the identity.
    offset = root_ss.n_children_spawned
    if offset:
        from tsbootstrap.prng_keys import offset_word

        root_b ^= offset_word(offset)
    return root_a, root_b


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
    "child_seed_sequences",
    "register_warmup",
    "warmup_kernels",
]
