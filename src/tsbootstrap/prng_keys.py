"""Pure, numba-free reference for the compiled backend's counter-based key math.

This module is the single source of truth for the Philox-4x32-10 key derivation the
compiled fast paths use: the per-replicate key, the panel per-series key fold, and the
raw round function. It is numpy-only and imports nothing from the rest of the package,
so three consumers can share one definition without pulling in numba:

- :mod:`tsbootstrap.block._compiled` recomputes the same arithmetic inside its
  ``@njit`` kernels (in ``np.uint64``, for speed); the two implementations are held
  bit-equal by the same-stream tests in ``tests/unit/test_compiled.py``.
- Those tests use the functions here as the INDEPENDENT ORACLE (explicit ``uint64``
  masking on Python ints, a different execution path from the kernel's ``np.uint64``),
  so a kernel bug cannot hide behind a self-referential check.
- A future array backend (JAX ``threefry``/``philox4x32``, cupy) reimplements the same
  math on its own array type and verifies it against these references, so the integer
  index stream can be made bit-identical across CPU and GPU.

The two multipliers below are load-bearing: ``REPLICATE_GOLDEN`` (replicate axis) MUST
differ from ``SERIES_GOLDEN`` (panel-series axis), or the panel key
``fold_in_key(replicate_key(root, b), s)`` becomes symmetric in ``(b, s)`` at an
all-zero root and replicate ``b``/series ``s`` would alias replicate ``s``/series ``b``.
Keeping both here co-locates that invariant in one file.
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
TWO32_INV: Final = 1.0 / 4294967296.0  # 1 / 2**32

# SplitMix64 finalizer mix constants (Steele et al. 2014).
_SPLITMIX_A: Final = 0xBF58476D1CE4E5B9
_SPLITMIX_B: Final = 0x94D049BB133111EB
_MASK64: Final = (1 << 64) - 1

# Distinct odd multipliers per axis (the anti-aliasing invariant; see the module
# docstring). Python ints here; the kernel imports and wraps them as np.uint64.
REPLICATE_GOLDEN: Final = 0xD1B54A32D192ED03
SERIES_GOLDEN: Final = 0x9E3779B97F4A7C15

# Slot-0 avalanche of SERIES_GOLDEN, XOR-ed out so panel series slot 0 folds to the
# identity (a lone-series panel reproduces the single-series stream bitwise).
SPLITMIX_S0_HI: Final = 0xE220A839
SPLITMIX_S0_LO: Final = 0x7B1DCDAF


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


def hash_series_words(s: int) -> tuple[int, int]:
    """Avalanche panel-slot series index ``s`` into two 32-bit words.

    A SplitMix64 finalizer over ``(s + 1) * SERIES_GOLDEN``, with the slot-0 avalanche
    XOR-ed out so ``s = 0`` maps to ``(0, 0)`` (the identity fold). The finalizer is a
    bijection and XOR-ing a constant preserves injectivity, so distinct ``s`` map to
    distinct word pairs; folding both words into the Philox key gives every ``(b, s)``
    a distinct stream (no 32-bit birthday risk).
    """
    z = _splitmix64_finalize(((s + 1) * SERIES_GOLDEN) & _MASK64)
    hi = ((z >> 32) & 0xFFFFFFFF) ^ SPLITMIX_S0_HI
    lo = (z & 0xFFFFFFFF) ^ SPLITMIX_S0_LO
    return hi, lo


def fold_in_key(key_hi: int, key_lo: int, s: int) -> tuple[int, int]:
    """Per-(replicate, series) Philox key: the replicate key XOR the series-slot words.

    A pure function of ``(key, s)`` (never ``num_series`` or the work-item order), which
    is what makes "series ``s`` inside a panel == series ``s`` standalone at the same key
    and slot" hold bitwise.
    """
    h_hi, h_lo = hash_series_words(s)
    return key_hi ^ h_hi, key_lo ^ h_lo


def philox4x32_10(
    c: tuple[int, int, int, int], k: tuple[int, int]
) -> tuple[int, int, int, int]:
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


def u01_from_word(word: int) -> float:
    """32-bit uniform in ``[0, 1)``: one Philox output word divided by ``2**32``."""
    return (word & 0xFFFFFFFF) * TWO32_INV


__all__ = [
    "PHILOX_M0",
    "PHILOX_M1",
    "PHILOX_W0",
    "PHILOX_W1",
    "MASK32",
    "SH32",
    "TWO32_INV",
    "REPLICATE_GOLDEN",
    "SERIES_GOLDEN",
    "SPLITMIX_S0_HI",
    "SPLITMIX_S0_LO",
    "replicate_key",
    "hash_series_words",
    "fold_in_key",
    "philox4x32_10",
    "u01_from_word",
]
