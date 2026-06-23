"""Tests for the deterministic, parallel-safe RNG contract (tsbootstrap.rng)."""

from __future__ import annotations

import numpy as np
import pytest

from tsbootstrap.errors import RNGContractError
from tsbootstrap.rng import (
    generators_from_seeds,
    register_warmup,
    resolve_and_describe,
    resolve_seed_sequence,
    single_threaded_blas,
    spawn_generators,
    spawn_seed_sequences,
    warmup_kernels,
)


def _draws(gen: np.random.Generator) -> list[int]:
    return gen.integers(0, 1_000_000, size=20).tolist()


class TestRNGDeterminism:
    def test_int_seed_is_fully_deterministic(self):
        a = spawn_generators(resolve_seed_sequence(42), 5)
        b = spawn_generators(resolve_seed_sequence(42), 5)
        assert [_draws(g) for g in a] == [_draws(g) for g in b]

    def test_distinct_indices_give_distinct_streams(self):
        gens = spawn_generators(resolve_seed_sequence(42), 6)
        draws = [tuple(_draws(g)) for g in gens]
        assert len(set(draws)) == 6  # every replicate stream is independent

    def test_prefix_stability_more_samples_keeps_earlier_streams(self):
        # Sample i must use the same stream regardless of total n_bootstraps, so a
        # SeedSequence spawned for n=10 must share its first 4 children with n=4.
        root_small = resolve_seed_sequence(123)
        root_big = resolve_seed_sequence(123)
        small = spawn_generators(root_small, 4)
        big = spawn_generators(root_big, 10)
        assert [_draws(g) for g in small] == [_draws(g) for g in big[:4]]

    def test_generator_input_is_consumed_once_and_reproducible(self):
        root, info = resolve_and_describe(np.random.default_rng(7))
        assert info.kind == "generator"
        assert info.entropy is not None
        g_from_root = spawn_generators(root, 3)
        g_from_recorded = spawn_generators(np.random.SeedSequence(info.entropy), 3)
        assert [_draws(g) for g in g_from_root] == [_draws(g) for g in g_from_recorded]


class TestG2StreamRouting:
    """G2: the chunked-spawn / stream-routing invariant that makes chunking determinism-safe.

    ``bootstrap`` spawns one root SeedSequence then SLICES its full child set per chunk
    (see ``api._iter_chunks``). These pin that this positional binding is load-bearing:
    slicing a single root's spawn is bit-for-bit identical to a pre-materialized full
    spawn and invariant to chunk size, while spawning a FRESH root per chunk would only
    reproduce the first chunk and silently corrupt the tail.
    """

    def test_chunked_spawn_equals_full_spawn_bit_for_bit(self):
        # Slicing one root's full spawn into chunks == pre-materialized full spawn, exactly.
        root_full = resolve_seed_sequence(123)
        full = [_draws(g) for g in generators_from_seeds(spawn_seed_sequences(root_full, 10))]

        root_chunked = resolve_seed_sequence(123)
        all_seeds = spawn_seed_sequences(root_chunked, 10)
        chunked: list[list[int]] = []
        for start in range(0, 10, 3):  # chunk size 3
            chunk = all_seeds[start : start + 3]
            chunked.extend(_draws(g) for g in generators_from_seeds(chunk))
        assert chunked == full

    @pytest.mark.parametrize("chunk_size", [1, 2, 3, 4, 7, 10, 100])
    def test_chunk_size_invariance(self, chunk_size):
        # The spawned streams must be identical regardless of how they are chunked.
        root_ref = resolve_seed_sequence(2024)
        reference = [_draws(g) for g in generators_from_seeds(spawn_seed_sequences(root_ref, 10))]

        root = resolve_seed_sequence(2024)
        all_seeds = spawn_seed_sequences(root, 10)
        got: list[list[int]] = []
        for start in range(0, 10, chunk_size):
            chunk = all_seeds[start : start + chunk_size]
            got.extend(_draws(g) for g in generators_from_seeds(chunk))
        assert got == reference

    def test_prefix_stability_under_chunking(self):
        # A run of n=10 chunked must share its first 4 streams with a run of n=4, so adding
        # replicates never reshuffles earlier ones regardless of the chunk boundary.
        small = [
            _draws(g)
            for g in generators_from_seeds(spawn_seed_sequences(resolve_seed_sequence(9), 4))
        ]
        big_seeds = spawn_seed_sequences(resolve_seed_sequence(9), 10)
        big: list[list[int]] = []
        for start in range(0, 10, 3):
            big.extend(_draws(g) for g in generators_from_seeds(big_seeds[start : start + 3]))
        assert big[:4] == small

    def test_fresh_root_per_chunk_does_not_reproduce_tail(self):
        # NEGATIVE assertion: re-spawning a fresh root per chunk (the WRONG design) reproduces
        # only the first chunk and diverges on every later chunk. This proves the single-root
        # positional binding is load-bearing, not incidental.
        root_full = resolve_seed_sequence(123)
        full = [_draws(g) for g in generators_from_seeds(spawn_seed_sequences(root_full, 10))]

        chunk_size = 3
        bad: list[list[int]] = []
        for start in range(0, 10, chunk_size):
            fresh_root = resolve_seed_sequence(123)  # the bug: a new root each chunk
            n_this = min(chunk_size, 10 - start)
            bad.extend(
                _draws(g) for g in generators_from_seeds(spawn_seed_sequences(fresh_root, n_this))
            )

        assert bad[:chunk_size] == full[:chunk_size]  # first chunk coincidentally matches
        assert bad[chunk_size:] != full[chunk_size:]  # but the tail is corrupted


class TestRNGInputValidation:
    def test_describe_kinds(self):
        assert resolve_and_describe(5)[1].kind == "int"
        assert resolve_and_describe(None)[1].kind == "none"
        assert resolve_and_describe(np.random.SeedSequence(1))[1].kind == "seed_sequence"

    def test_negative_seed_rejected(self):
        with pytest.raises(RNGContractError):
            resolve_seed_sequence(-1)

    def test_unsupported_random_state_rejected(self):
        with pytest.raises(RNGContractError):
            resolve_seed_sequence("not-a-seed")  # type: ignore[arg-type]


class TestRNGUtilities:
    def test_single_threaded_blas_is_a_noop_safe_context(self):
        with single_threaded_blas():
            _ = np.linalg.svd(np.eye(3))  # must run without error

    def test_warmup_runs_each_hook_once(self):
        calls = []
        register_warmup(lambda: calls.append(1))
        warmup_kernels()
        warmup_kernels()  # idempotent
        assert calls.count(1) == 1
