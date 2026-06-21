"""Tests for the deterministic, parallel-safe RNG contract (tsbootstrap.rng)."""

from __future__ import annotations

import numpy as np
import pytest

from tsbootstrap.errors import RNGContractError
from tsbootstrap.rng import (
    register_warmup,
    resolve_and_describe,
    resolve_seed_sequence,
    single_threaded_blas,
    spawn_generators,
    warmup_kernels,
)


def _draws(gen: np.random.Generator) -> list[int]:
    return gen.integers(0, 1_000_000, size=20).tolist()


def test_int_seed_is_fully_deterministic():
    a = spawn_generators(resolve_seed_sequence(42), 5)
    b = spawn_generators(resolve_seed_sequence(42), 5)
    assert [_draws(g) for g in a] == [_draws(g) for g in b]


def test_distinct_indices_give_distinct_streams():
    gens = spawn_generators(resolve_seed_sequence(42), 6)
    draws = [tuple(_draws(g)) for g in gens]
    assert len(set(draws)) == 6  # every replicate stream is independent


def test_prefix_stability_more_samples_keeps_earlier_streams():
    # Sample i must use the same stream regardless of total n_bootstraps, so a
    # SeedSequence spawned for n=10 must share its first 4 children with n=4.
    root_small = resolve_seed_sequence(123)
    root_big = resolve_seed_sequence(123)
    small = spawn_generators(root_small, 4)
    big = spawn_generators(root_big, 10)
    assert [_draws(g) for g in small] == [_draws(g) for g in big[:4]]


def test_generator_input_is_consumed_once_and_reproducible():
    root, info = resolve_and_describe(np.random.default_rng(7))
    assert info.kind == "generator"
    assert info.entropy is not None
    g_from_root = spawn_generators(root, 3)
    g_from_recorded = spawn_generators(np.random.SeedSequence(info.entropy), 3)
    assert [_draws(g) for g in g_from_root] == [_draws(g) for g in g_from_recorded]


def test_describe_kinds():
    assert resolve_and_describe(5)[1].kind == "int"
    assert resolve_and_describe(None)[1].kind == "none"
    assert resolve_and_describe(np.random.SeedSequence(1))[1].kind == "seed_sequence"


def test_negative_seed_rejected():
    with pytest.raises(RNGContractError):
        resolve_seed_sequence(-1)


def test_unsupported_random_state_rejected():
    with pytest.raises(RNGContractError):
        resolve_seed_sequence("not-a-seed")  # type: ignore[arg-type]


def test_single_threaded_blas_is_a_noop_safe_context():
    with single_threaded_blas():
        _ = np.linalg.svd(np.eye(3))  # must run without error


def test_warmup_runs_each_hook_once():
    calls = []
    register_warmup(lambda: calls.append(1))
    warmup_kernels()
    warmup_kernels()  # idempotent
    assert calls.count(1) == 1
