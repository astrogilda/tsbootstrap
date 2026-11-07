"""Pure, numba-free reference for the compiled backend's counter-based random stream.

This module is the single source of truth for the compiled stream (version
:data:`STREAM_ID`): the per-replicate Philox-4x32-10 key, the counter layout, the
raw round function and the unbiased bounded-integer draw. It is numpy-only and imports
nothing from the rest of the package, so three consumers share one definition without
pulling in numba:

- :mod:`tsbootstrap.block._compiled` recomputes the same arithmetic inside its
  ``@njit`` kernels (in ``np.uint64``, for speed); the two implementations are held
  bit-equal by the same-stream tests in ``tests/unit/test_compiled.py``.
- Those tests use :class:`PhiloxStream` as the INDEPENDENT ORACLE (explicit masking on
  Python ints, a different execution path from the kernel's ``np.uint64``), so a kernel
  bug cannot hide behind a self-referential check.
- A future array backend (JAX ``philox4x32``, cupy) reimplements the same math on its
  own array type and verifies it against these references.

Stream layout. One stream per (replicate ``b``, panel series ``s``; ``s = 0`` for a
single series): the key is :func:`replicate_key` of ``b`` and the 128-bit counter is
``(block low32, block high32, s low32, s high32)``. Draw ``k`` is word ``k % 4`` of
Philox block ``k // 4``. The replicate key is injective in ``b`` and the series index
owns counter words 2 and 3, so distinct ``(b, s)`` never share a stream, and with a
64-bit block counter a stream never wraps.
"""

from __future__ import annotations

from typing import Final

import numpy as np

# Philox-4x32-10 constants (canonical Random123 / Salmon et al. 2011,
# include/Random123/philox.h). np.uint64-typed so the numba kernels can import them as
# compile-time constants; the pinned known-answer test forbids changing them.
PHILOX_M0: Final = np.uint64(0xD2511F53)
PHILOX_M1: Final = np.uint64(0xCD9E8D57)
PHILOX_W0: Final = np.uint32(0x9E3779B9)
PHILOX_W1: Final = np.uint32(0xBB67AE85)
MASK32: Final = np.uint64(0xFFFFFFFF)
SH32: Final = np.uint64(32)

# SplitMix64 finalizer mix constants (Steele et al. 2014).
_SPLITMIX_A: Final = 0xBF58476D1CE4E5B9
_SPLITMIX_B: Final = 0x94D049BB133111EB
_MASK64: Final = (1 << 64) - 1

# Odd multiplier of the replicate axis (and of the SeedSequence child-offset fold).
REPLICATE_GOLDEN: Final = 0xD1B54A32D192ED03

# Version of the compiled stream. Bump it, with a CHANGELOG entry, whenever the bytes a
# given (root, replicate, series) produces change.
STREAM_ID: Final = "philox4x32-10/v2"


def _splitmix64_finalize(z: int) -> int:
    """One SplitMix64 finalizer pass on a uint64 (bijection mod 2**64)."""
    z = ((z ^ (z >> 30)) * _SPLITMIX_A) & _MASK64
    z = ((z ^ (z >> 27)) * _SPLITMIX_B) & _MASK64
    return z ^ (z >> 31)


def replicate_key(root_a: int, root_b: int, b: int) -> tuple[int, int]:
    """Replicate ``b``'s two 32-bit Philox key words from the packed 128-bit root.

    Two-round SplitMix64 over ``root_a + (b + 1) * REPLICATE_GOLDEN`` with the upper 64
    root bits (``root_b``) folded in between the rounds. ``(b + 1) * golden`` is a
    bijection mod 2**64 for the odd golden and every step is a bijection on uint64, so
    ``b -> key`` is injective for a fixed root (single-series replicate keys are
    exact-distinct). Explicit masking on Python ints keeps this a reference independent
    of the kernel's ``np.uint64`` recomputation.
    """
    z = (root_a + (b + 1) * REPLICATE_GOLDEN) & _MASK64
    z = _splitmix64_finalize(z)
    z = z ^ root_b
    z = _splitmix64_finalize(z)
    return (z >> 32) & 0xFFFFFFFF, z & 0xFFFFFFFF


def offset_word(offset: int) -> int:
    """Avalanche a SeedSequence child offset into a 64-bit word folded into the root.

    A bijection of ``offset`` (odd multiplier, then the SplitMix64 finalizer), so distinct
    offsets of one root give distinct compiled streams.
    """
    return _splitmix64_finalize((offset * REPLICATE_GOLDEN) & _MASK64)


def philox4x32_10(c: tuple[int, int, int, int], k: tuple[int, int]) -> tuple[int, int, int, int]:
    """Ten canonical Philox-4x32 rounds over a full 4-word counter and 2-word key.

    The reference oracle for the kernel's ``_philox_round4`` and for the published
    Random123 known-answer vectors. The Weyl key increment runs BEFORE rounds 2..10 (the
    first round uses the un-bumped key; nine bumps, none after the last round).
    """
    m0, m1 = 0xD2511F53, 0xCD9E8D57
    w0, w1 = 0x9E3779B9, 0xBB67AE85
    c0 = c[0] & 0xFFFFFFFF
    c1 = c[1] & 0xFFFFFFFF
    c2 = c[2] & 0xFFFFFFFF
    c3 = c[3] & 0xFFFFFFFF
    k0, k1 = k[0] & 0xFFFFFFFF, k[1] & 0xFFFFFFFF
    for r in range(10):
        if r > 0:
            k0 = (k0 + w0) & 0xFFFFFFFF
            k1 = (k1 + w1) & 0xFFFFFFFF
        p0 = m0 * c0
        hi0 = p0 >> 32
        lo0 = p0 & 0xFFFFFFFF
        p1 = m1 * c2
        hi1 = p1 >> 32
        lo1 = p1 & 0xFFFFFFFF
        c0 = (hi1 ^ c1 ^ k0) & 0xFFFFFFFF
        c1 = lo1
        c2 = (hi0 ^ c3 ^ k1) & 0xFFFFFFFF
        c3 = lo0
    return c0, c1, c2, c3


class PhiloxStream:
    """Reference draw stream for one (replicate, series): words and unbiased integers.

    Mirrors the kernel's ``_next_word`` and ``_bounded`` on Python ints.
    """

    def __init__(self, key_hi: int, key_lo: int, series: int = 0) -> None:
        self._k = (key_lo & 0xFFFFFFFF, key_hi & 0xFFFFFFFF)  # Random123 (k0, k1)
        self._series = series
        self._block = 0
        self._buf: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._pos = 4

    def next_word(self) -> int:
        """The next 32-bit output word of the stream."""
        if self._pos == 4:
            blk, ser = self._block, self._series
            self._buf = philox4x32_10(
                (blk & 0xFFFFFFFF, blk >> 32, ser & 0xFFFFFFFF, ser >> 32), self._k
            )
            self._block += 1
            self._pos = 0
        word = self._buf[self._pos]
        self._pos += 1
        return word

    def bounded(self, n: int) -> int:
        """Exactly uniform integer in ``[0, n)`` (32-bit multiply-high with rejection)."""
        m = self.next_word() * n
        low = m & 0xFFFFFFFF
        if low < n:
            thresh = (2**32 - n) % n
            while low < thresh:
                m = self.next_word() * n
                low = m & 0xFFFFFFFF
        return m >> 32


__all__ = [
    "PHILOX_M0",
    "PHILOX_M1",
    "PHILOX_W0",
    "PHILOX_W1",
    "MASK32",
    "SH32",
    "REPLICATE_GOLDEN",
    "STREAM_ID",
    "PhiloxStream",
    "replicate_key",
    "offset_word",
    "philox4x32_10",
]
