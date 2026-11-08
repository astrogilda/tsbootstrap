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
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

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


# --- batched replicate streams ------------------------------------------------
# Replicate i's stream is exactly ``np.random.default_rng(child_i)`` where child_i is child
# ``n_children_spawned + i`` of the root. Building B SeedSequence objects and B Generators
# costs tens of microseconds per replicate, more than the draws themselves for short series.
# The functions below compute the same PCG64 states for a whole chunk at once (NumPy's
# SeedSequence hash and PCG64 seeding, vectorized over the child index) and load each state
# into ONE reused Generator. The draw methods stay NumPy's own, so the bytes are NumPy's.

_MASK32 = 0xFFFFFFFF
_INIT_A = 0x43B0D7E5
_MULT_A = 0x931E8875
_INIT_B = 0x8B51F9DD
_MULT_B = 0x58F38DED
_MIX_MULT_L = 0xCA01F9DD
_MIX_MULT_R = 0x4973F715
_XSHIFT = 16
# PCG64's 128-bit LCG multiplier as four 32-bit limbs, least significant first.
_PCG_MULT = (2549297995355413924 << 64) + 4865540595714422341
_PCG_MULT_LIMBS = tuple((_PCG_MULT >> (32 * k)) & _MASK32 for k in range(4))
_U64 = np.uint64


def _uint32_words(value: object) -> list[int]:
    """NumPy's SeedSequence coercion of entropy or a spawn key to 32-bit words."""
    if isinstance(value, np.ndarray) and value.dtype == np.dtype(np.uint32):
        return [int(v) for v in value]
    if isinstance(value, (int, np.integer)):
        n = int(value)
        if n == 0:
            return [0]
        words = []
        while n > 0:
            words.append(n & _MASK32)
            n >>= 32
        return words
    out: list[int] = []
    for item in value:  # type: ignore[attr-defined]
        out.extend(_uint32_words(item))
    return out


def _hashmix(value: NDArray[np.uint64], hash_const: list[int]) -> NDArray[np.uint64]:
    value = value ^ _U64(hash_const[0])
    hash_const[0] = (hash_const[0] * _MULT_A) & _MASK32
    value = (value * _U64(hash_const[0])) & _U64(_MASK32)
    return value ^ (value >> _U64(_XSHIFT))


def _mix(x: NDArray[np.uint64], y: NDArray[np.uint64]) -> NDArray[np.uint64]:
    result = (_U64(_MIX_MULT_L) * x - _U64(_MIX_MULT_R) * y) & _U64(_MASK32)
    return result ^ (result >> _U64(_XSHIFT))


def _mixed_pools(entropy: list[NDArray[np.uint64]], pool_size: int) -> list[NDArray[np.uint64]]:
    """SeedSequence.mix_entropy over columns: ``entropy`` is one (m,) word array per position."""
    hash_const = [_INIT_A]
    zero = np.zeros_like(entropy[0])
    mixer = [
        _hashmix(entropy[i] if i < len(entropy) else zero, hash_const) for i in range(pool_size)
    ]
    for i_src in range(pool_size):
        for i_dst in range(pool_size):
            if i_src != i_dst:
                mixer[i_dst] = _mix(mixer[i_dst], _hashmix(mixer[i_src], hash_const))
    for i_src in range(pool_size, len(entropy)):
        for i_dst in range(pool_size):
            mixer[i_dst] = _mix(mixer[i_dst], _hashmix(entropy[i_src], hash_const))
    return mixer


def _add128(a: list[NDArray[np.uint64]], b: list[NDArray[np.uint64]]) -> list[NDArray[np.uint64]]:
    out, carry = [], _U64(0)
    for k in range(4):
        s = a[k] + b[k] + carry
        out.append(s & _U64(_MASK32))
        carry = s >> _U64(32)
    return out


def _mul128_const(a: list[NDArray[np.uint64]]) -> list[NDArray[np.uint64]]:
    cols = [np.zeros_like(a[0]) for _ in range(4)]
    for i in range(4):
        for j in range(4 - i):
            p = a[i] * _U64(_PCG_MULT_LIMBS[j])  # both < 2**32, so p < 2**64
            cols[i + j] += p & _U64(_MASK32)
            if i + j + 1 < 4:
                cols[i + j + 1] += p >> _U64(32)
    out, carry = [], _U64(0)
    for k in range(4):
        s = cols[k] + carry
        out.append(s & _U64(_MASK32))
        carry = s >> _U64(32)
    return out


def replicate_states(root: np.random.SeedSequence, start: int, count: int) -> NDArray[np.uint64]:
    """PCG64 states of replicates ``start .. start + count - 1``, shape ``(count, 4)``.

    Row ``i`` is ``(state_hi, state_lo, inc_hi, inc_lo)`` of
    ``np.random.default_rng(child).bit_generator`` for the child SeedSequence that
    :func:`child_seed_sequences` builds at position ``start + i``, computed without building
    any SeedSequence or Generator. ``root`` is not advanced.
    """
    if start < 0 or count < 0:
        raise RNGContractError(
            "start and count must be non-negative", context={"start": start, "count": count}
        )
    out = np.empty((count, 4), dtype=np.uint64)
    if count == 0:
        return out
    run = _uint32_words(root.entropy)
    run += [0] * max(0, root.pool_size - len(run))  # children always carry a spawn key
    prefix = run + _uint32_words(tuple(root.spawn_key))
    first = root.n_children_spawned + start
    # Group by how many 32-bit words the child index takes (one below 2**32).
    widths = [len(_uint32_words(first)), len(_uint32_words(first + count - 1))]
    for width in range(widths[0], widths[1] + 1):
        lo_idx = max(first, (1 << (32 * (width - 1))) if width > 1 else 0)
        hi_idx = min(first + count, 1 << (32 * width))
        if lo_idx >= hi_idx:
            continue
        idx = np.arange(lo_idx, hi_idx, dtype=np.uint64)
        m = idx.shape[0]
        entropy = [np.full(m, w, dtype=np.uint64) for w in prefix]
        entropy += [(idx >> _U64(32 * k)) & _U64(_MASK32) for k in range(width)]
        pool = _mixed_pools(entropy, root.pool_size)
        hash_const = _INIT_B
        words = []
        for i_dst in range(8):  # generate_state(4, uint64) == 8 uint32 words
            v = pool[i_dst % root.pool_size] ^ _U64(hash_const)
            hash_const = (hash_const * _MULT_B) & _MASK32
            v = (v * _U64(hash_const)) & _U64(_MASK32)
            words.append(v ^ (v >> _U64(_XSHIFT)))
        # PCG64 seeding: initstate = (w1:w0, w3:w2) as (hi, lo); initseq likewise from w4..w7.
        initstate = [words[2], words[3], words[0], words[1]]
        seq = [words[6], words[7], words[4], words[5]]
        inc = [((seq[0] << _U64(1)) & _U64(_MASK32)) | _U64(1)] + [
            ((seq[k] << _U64(1)) | (seq[k - 1] >> _U64(31))) & _U64(_MASK32) for k in (1, 2, 3)
        ]
        # pcg_setseq_128_srandom_r: state = ((0 * M + inc) + initstate) * M + inc.
        state = _add128(_mul128_const(_add128(inc, initstate)), inc)
        rows = slice(lo_idx - first, hi_idx - first)
        out[rows, 0] = (state[3] << _U64(32)) | state[2]
        out[rows, 1] = (state[1] << _U64(32)) | state[0]
        out[rows, 2] = (inc[3] << _U64(32)) | inc[2]
        out[rows, 3] = (inc[1] << _U64(32)) | inc[0]
    return out


def _legacy_state(child: np.random.SeedSequence) -> tuple[int, int, int, int]:
    s = np.random.PCG64(child).state["state"]
    return (s["state"] >> 64, s["state"] & (2**64 - 1), s["inc"] >> 64, s["inc"] & (2**64 - 1))


class _StateLoader:
    """Loads a precomputed PCG64 state into one Generator, as fast as the layout allows.

    The fast path writes the 128-bit state and increment straight into the bit generator's C
    struct and clears its buffered 32-bit half. It is used only after a probe proves, on this
    NumPy build, that raw writes and the official ``state`` setter agree word for word;
    otherwise every load goes through the official setter, which is slower but exact.
    """

    __slots__ = ("bitgen", "generator", "_words", "_flags", "_order")

    def __init__(self) -> None:
        self.bitgen = np.random.PCG64(0)
        self.generator = np.random.Generator(self.bitgen)
        self._words: NDArray[np.uint64] | None = None
        self._flags: NDArray[np.uint32] | None = None
        self._order: NDArray[np.intp] = np.arange(4)
        with contextlib.suppress(Exception):
            self._probe()

    def _setter(self, hi: int, lo: int, ihi: int, ilo: int, has: int = 0, uint: int = 0) -> None:
        self.bitgen.state = {
            "bit_generator": "PCG64",
            "state": {"state": (hi << 64) | lo, "inc": (ihi << 64) | ilo},
            "has_uint32": has,
            "uinteger": uint,
        }

    def _probe(self) -> None:
        import ctypes

        known = (0x1111111122222223, 0x3333333344444445, 0x5555555566666667, 0x7777777788888889)
        self._setter(*known, has=1, uint=0x5A5A1234)
        base = self.bitgen.ctypes.state_address
        ptr_size = ctypes.sizeof(ctypes.c_void_p)
        pcg = ctypes.c_void_p.from_address(base).value
        if not pcg:
            return
        words = np.ctypeslib.as_array((ctypes.c_uint64 * 4).from_address(pcg))
        flags = np.ctypeslib.as_array((ctypes.c_uint32 * 2).from_address(base + ptr_size))
        found = [int(w) for w in words]
        if sorted(found) != sorted(known) or int(flags[0]) != 1 or int(flags[1]) != 0x5A5A1234:
            return
        order = tuple(found.index(k) for k in known)  # memory slot of hi, lo, ihi, ilo
        probe = (0x0123456789ABCDEF, 0xFEDCBA9876543210, 0x0F1E2D3C4B5A6978, 0x8796A5B4C3D2E1F1)
        words[list(order)] = probe
        flags[:] = 0
        got = self.bitgen.state
        want = {"state": (probe[0] << 64) | probe[1], "inc": (probe[2] << 64) | probe[3]}
        if got["state"] != want or got["has_uint32"] != 0 or got["uinteger"] != 0:
            self._setter(0, 0, 0, 1)
            return
        self._words, self._flags = words, flags
        self._order = np.asarray(order, dtype=np.intp)

    @property
    def fast(self) -> bool:
        """Whether loads write the C struct directly (the probe passed)."""
        return self._words is not None

    def load(self, row: NDArray[np.uint64]) -> np.random.Generator:
        """Position the shared Generator at the start of the stream whose state is ``row``."""
        if self._words is not None and self._flags is not None:
            self._words[self._order] = row
            self._flags[0] = 0
            self._flags[1] = 0
        else:
            self._setter(int(row[0]), int(row[1]), int(row[2]), int(row[3]))
        return self.generator


# Array elements a numpy chunk kernel works on at once (its temporaries, not its output).
ROW_GROUP_ELEMENTS = 2**16


class ReplicateStreams:
    """The random streams of a contiguous run of replicates, for a numpy chunk kernel.

    ``len(streams)`` is the number of replicates. Iterating yields, for each replicate in
    order, a Generator positioned at the start of that replicate's stream, byte-identical to
    ``np.random.default_rng(child)`` for the replicate's child SeedSequence. The SAME
    Generator object is yielded every time and repositioned on the next step, so a kernel
    must finish its draws for replicate ``i`` before advancing and must never keep the
    Generator across steps (``list(streams)`` would hold one object ``B`` times).
    """

    __slots__ = ("_states", "_loader")

    def __init__(self, root: np.random.SeedSequence, start: int, count: int) -> None:
        self._states = replicate_states(root, start, count)
        self._loader = _StateLoader()
        if count:
            # Cheap guard on every chunk: the first and last replicates must match NumPy's own
            # seeding, or the batched hash is wrong for this NumPy and the chunk fails loudly.
            ends = (0, count - 1)
            kids = [child_seed_sequences(root, start + i, 1)[0] for i in ends]
            for i, kid in zip(ends, kids, strict=True):
                if tuple(int(w) for w in self._states[i]) != _legacy_state(kid):
                    raise RNGContractError(
                        "batched replicate seeding disagrees with numpy's SeedSequence",
                        context={"replicate": start + i, "numpy": np.__version__},
                    )

    def __len__(self) -> int:
        return int(self._states.shape[0])

    def __iter__(self) -> Iterator[np.random.Generator]:
        load = self._loader.load
        for row in self._states:
            yield load(row)

    def groups(self, row_elements: int) -> Iterator[tuple[int, int, ReplicateStreams]]:
        """Yield ``(lo, hi, sub_streams)`` covering the replicates in groups of whole rows.

        A group holds about :data:`ROW_GROUP_ELEMENTS` array elements, given ``row_elements``
        elements per replicate, so a kernel that builds its temporaries one group at a time
        keeps its working set small and in cache whatever the chunk size. Each replicate's
        stream is independent, so grouping never changes a byte.
        """
        rows = max(1, ROW_GROUP_ELEMENTS // max(1, row_elements))
        for lo in range(0, len(self), rows):
            hi = min(len(self), lo + rows)
            sub = object.__new__(ReplicateStreams)
            sub._states = self._states[lo:hi]
            sub._loader = self._loader
            yield lo, hi, sub


# What a chunk kernel iterates: the streams of a run, or a plain sequence of Generators.
StreamsLike = ReplicateStreams | Sequence[np.random.Generator]


def stream_groups(
    streams: StreamsLike, row_elements: int
) -> Iterator[tuple[int, int, StreamsLike]]:
    """Split ``streams`` into groups of whole replicates, as :meth:`ReplicateStreams.groups`.

    A kernel may be handed a plain sequence of Generators instead of a
    :class:`ReplicateStreams` (tests do); such a sequence is sliced the same way.
    """
    if isinstance(streams, ReplicateStreams):
        yield from streams.groups(row_elements)
        return
    rows = max(1, ROW_GROUP_ELEMENTS // max(1, row_elements))
    for lo in range(0, len(streams), rows):
        hi = min(len(streams), lo + rows)
        yield lo, hi, streams[lo:hi]


def replicate_streams(root: np.random.SeedSequence, n: int) -> ReplicateStreams:
    """The streams of replicates ``0 .. n - 1`` of ``root`` (see :class:`ReplicateStreams`)."""
    return ReplicateStreams(root, 0, n)


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
    # generate_state ignores the child offset. A root that has already spawned k children
    # means "start at replicate k" on the numpy backend (its children continue from k), and
    # it means the same here: replicate b's key hashes root_a + (b + 1) * golden, so adding
    # k * golden to root_a makes replicate b of this root replicate k + b of the fresh root
    # (exactly, mod 2**64). Offset 0 is the identity.
    from tsbootstrap.prng_keys import REPLICATE_GOLDEN

    root_a = (root_a + root_ss.n_children_spawned * REPLICATE_GOLDEN) & 0xFFFFFFFFFFFFFFFF
    return root_a, root_b


__all__ = [
    "RandomStateLike",
    "RandomStateInfo",
    "resolve_seed_sequence",
    "resolve_and_describe",
    "spawn_seed_sequences",
    "ROW_GROUP_ELEMENTS",
    "StreamsLike",
    "stream_groups",
    "ReplicateStreams",
    "replicate_states",
    "replicate_streams",
    "root_key_from",
    "child_seed_sequences",
]
