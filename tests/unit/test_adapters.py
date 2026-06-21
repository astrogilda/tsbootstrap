"""Tests for the sktime/skbase bootstrap adapters."""

from __future__ import annotations

import numpy as np
import pytest

from tsbootstrap.adapters import (
    ARResidualBootstrap,
    CircularBlockBootstrap,
    IIDBootstrap,
    MovingBlockBootstrap,
    NonOverlappingBlockBootstrap,
    SieveBootstrap,
    StationaryBlockBootstrap,
    TaperedBlockBootstrap,
    VARResidualBootstrap,
)

BLOCK_ADAPTERS = [
    MovingBlockBootstrap,
    CircularBlockBootstrap,
    NonOverlappingBlockBootstrap,
    TaperedBlockBootstrap,
]


def _ar1(phi: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def test_get_params_set_params_clone_roundtrip():
    est = MovingBlockBootstrap(block_length=5, n_bootstraps=10, random_state=0)
    params = est.get_params()
    assert params["block_length"] == 5
    assert params["n_bootstraps"] == 10
    assert params["random_state"] == 0
    clone = est.clone()
    assert clone.get_params() == params
    est.set_params(block_length=7)
    assert est.get_params()["block_length"] == 7


def test_class_tags():
    assert MovingBlockBootstrap.get_class_tag("object_type") == "bootstrap"
    assert IIDBootstrap.get_class_tag("capability:multivariate") is True
    assert ARResidualBootstrap.get_class_tag("capability:multivariate") is False


@pytest.mark.parametrize("cls", BLOCK_ADAPTERS)
def test_block_adapter_yields_samples(cls):
    x = _ar1(0.5, 100, 0)
    est = cls(n_bootstraps=8, random_state=0)
    samples = list(est.bootstrap(x))
    assert est.get_n_bootstraps() == 8
    assert len(samples) == 8
    assert all(s.shape == (100,) for s in samples)


def test_iid_adapter():
    x = _ar1(0.0, 80, 1)
    samples = list(IIDBootstrap(n_bootstraps=6, random_state=0).bootstrap(x))
    assert len(samples) == 6 and all(s.shape == (80,) for s in samples)


def test_return_indices():
    x = _ar1(0.5, 60, 1)
    out = list(StationaryBlockBootstrap(avg_block_length=5, n_bootstraps=4, random_state=0).bootstrap(x, return_indices=True))
    assert len(out) == 4
    values, indices = out[0]
    assert values.shape == (60,)
    assert indices.shape == (60,)


def test_model_adapters_yield_samples():
    x = _ar1(0.6, 200, 2)
    for est in (ARResidualBootstrap(order=1, n_bootstraps=5, random_state=0), SieveBootstrap(n_bootstraps=5, random_state=0)):
        samples = list(est.bootstrap(x))
        assert len(samples) == 5
        assert all(s.shape == (200,) for s in samples)
        assert all(np.isfinite(s).all() for s in samples)


def test_var_adapter_multivariate():
    rng = np.random.default_rng(0)
    x = np.zeros((200, 2))
    a = np.array([[0.5, 0.1], [0.2, 0.4]])
    for t in range(1, 200):
        x[t] = a @ x[t - 1] + rng.standard_normal(2)
    samples = list(VARResidualBootstrap(order=1, n_bootstraps=4, random_state=0).bootstrap(x))
    assert len(samples) == 4
    assert all(s.shape == (200, 2) for s in samples)


def test_determinism_through_adapter():
    x = _ar1(0.5, 80, 3)
    a = list(MovingBlockBootstrap(block_length=5, n_bootstraps=6, random_state=7).bootstrap(x))
    b = list(MovingBlockBootstrap(block_length=5, n_bootstraps=6, random_state=7).bootstrap(x))
    for sa, sb in zip(a, b):
        np.testing.assert_array_equal(sa, sb)
