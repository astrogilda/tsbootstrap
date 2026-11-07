"""Byte oracle: batched replicate streams equal NumPy's own per-child construction.

Every check compares :class:`tsbootstrap.rng.ReplicateStreams` against
``np.random.default_rng(child)`` for the same child SeedSequence: the state before any draw,
the draws for a mix of methods (odd int32 counts use the buffered 32-bit half), and the state
after the draws.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsbootstrap.errors import RNGContractError
from tsbootstrap.rng import (
    ReplicateStreams,
    _StateLoader,
    child_seed_sequences,
    replicate_states,
)


def _mixed_draws(g: np.random.Generator) -> list[object]:
    return [
        g.integers(0, 2_000_003, size=7, dtype=np.int32).tolist(),  # odd count: buffered half
        g.integers(0, 3, size=5).tolist(),  # rejection bound
        g.standard_normal(3).tolist(),
        g.random(2).tolist(),
        int(g.integers(0, 2**40)),
    ]


ROOTS = [
    np.random.SeedSequence(0),
    np.random.SeedSequence(2**200 + 12345),
    np.random.SeedSequence([1, 2, 3, 4, 5, 6, 7]),  # entropy longer than the pool
    np.random.SeedSequence(np.array([9, 8], dtype=np.uint32)),
    np.random.SeedSequence(77, spawn_key=(3, 2**40)),  # nested, multi-word spawn key
    np.random.SeedSequence(5, pool_size=8),  # non-default pool size
]


@pytest.mark.parametrize("root", ROOTS, ids=range(len(ROOTS)))
@pytest.mark.parametrize("start", [0, 5])
def test_streams_equal_numpy_children(root, start):
    count = 9
    kids = child_seed_sequences(root, start, count)
    want = [_mixed_draws(np.random.default_rng(k)) for k in kids]
    want_after = []
    for k in kids:
        g = np.random.default_rng(k)
        _mixed_draws(g)
        want_after.append(g.bit_generator.state)
    streams = ReplicateStreams(root, start, count)
    got, got_after = [], []
    for g in streams:
        got.append(_mixed_draws(g))
        got_after.append(g.bit_generator.state)
    assert got == want
    assert got_after == want_after


def test_previously_spawned_root_starts_at_its_offset():
    root = np.random.SeedSequence(42)
    root.spawn(3)
    kids = root.spawn(4)  # what numpy hands out next; the copy below is not advanced
    fresh = np.random.SeedSequence(42)
    fresh.spawn(3)
    got = [g.integers(0, 10**9, size=4).tolist() for g in ReplicateStreams(fresh, 0, 4)]
    assert got == [np.random.default_rng(k).integers(0, 10**9, size=4).tolist() for k in kids]
    assert fresh.n_children_spawned == 3  # never advanced


def test_child_index_crossing_32_bits():
    root = np.random.SeedSequence(11)
    start = 2**32 - 2  # two one-word children, then two two-word children
    states = replicate_states(root, start, 4)
    for i, kid in enumerate(child_seed_sequences(root, start, 4)):
        s = np.random.PCG64(kid).state["state"]
        assert int(states[i, 0]) << 64 | int(states[i, 1]) == s["state"]
        assert int(states[i, 2]) << 64 | int(states[i, 3]) == s["inc"]


def test_fast_loader_is_used_and_setter_fallback_agrees():
    fast = _StateLoader()
    assert fast.fast  # the probe passes on every supported platform
    slow = _StateLoader()
    slow._words = None  # force the official setter
    row = replicate_states(np.random.SeedSequence(3), 0, 1)[0]
    a = fast.load(row)
    a.integers(0, 5, size=3, dtype=np.int32)  # leave a buffered half behind
    a = fast.load(row)  # reload must clear it
    b = slow.load(row)
    assert (
        a.integers(0, 10**6, size=9, dtype=np.int32).tolist()
        == b.integers(0, 10**6, size=9, dtype=np.int32).tolist()
    )


def test_len_and_empty():
    assert len(ReplicateStreams(np.random.SeedSequence(1), 0, 0)) == 0
    assert len(ReplicateStreams(np.random.SeedSequence(1), 3, 6)) == 6
    with pytest.raises(RNGContractError):
        replicate_states(np.random.SeedSequence(1), -1, 2)


@pytest.mark.parametrize("backend", ["numpy", "compiled"])
def test_offset_root_continues_the_run_on_both_backends(backend):
    """A root that spawned k children gives replicates k.. of the fresh root, on both backends."""
    pytest.importorskip("numba") if backend == "compiled" else None
    import tsbootstrap as tsb
    from tsbootstrap.methods import MovingBlock

    x = np.random.default_rng(3).standard_normal(60)
    spec = MovingBlock(block_length=4)
    full = tsb.bootstrap(x, method=spec, n_bootstraps=10, random_state=np.random.SeedSequence(7),
                         backend=backend).indices()
    advanced = np.random.SeedSequence(7)
    advanced.spawn(4)
    tail = tsb.bootstrap(x, method=spec, n_bootstraps=6, random_state=advanced,
                         backend=backend).indices()
    np.testing.assert_array_equal(np.asarray(tail), np.asarray(full)[4:])
