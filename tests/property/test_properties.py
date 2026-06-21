"""Property-based invariants for the public bootstrap API (hypothesis).

These assert contracts that must hold for ALL inputs and configurations, and
complement the example-based unit tests. They are part of the release gate.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from tsbootstrap import (
    AR,
    IID,
    CircularBlock,
    MovingBlock,
    NonOverlappingBlock,
    ResidualBootstrap,
    SieveAR,
    StationaryBlock,
    TaperedBlock,
    bootstrap,
)

_SETTINGS = settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
_MODEL_SETTINGS = settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])

OBS_METHODS = [
    IID(),
    MovingBlock(block_length=4),
    CircularBlock(block_length=4),
    StationaryBlock(avg_block_length=4),
    NonOverlappingBlock(block_length=4),
    TaperedBlock(block_length=4),
]


@st.composite
def finite_series(draw, min_n: int = 20, max_n: int = 70) -> np.ndarray:
    n = draw(st.integers(min_n, max_n))
    data = draw(
        st.lists(
            st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    x = np.asarray(data, dtype=np.float64)
    assume(x.std() > 1e-6)  # reject degenerate constant series
    return x


@st.composite
def ar1_series(draw) -> np.ndarray:
    n = draw(st.integers(80, 200))
    phi = draw(st.floats(-0.7, 0.7))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


@given(x=finite_series(), method=st.sampled_from(OBS_METHODS), seed=st.integers(0, 2**32 - 1))
@_SETTINGS
def test_determinism(x, method, seed):
    a = bootstrap(x, method=method, n_bootstraps=5, random_state=seed).values()
    b = bootstrap(x, method=method, n_bootstraps=5, random_state=seed).values()
    np.testing.assert_array_equal(a, b)


@given(x=finite_series(), method=st.sampled_from(OBS_METHODS))
@_SETTINGS
def test_shape_and_finiteness(x, method):
    res = bootstrap(x, method=method, n_bootstraps=7, random_state=0)
    assert res.values().shape == (7, len(x))
    assert np.isfinite(res.values()).all()


@given(x=finite_series(), method=st.sampled_from(OBS_METHODS))
@_SETTINGS
def test_observation_indices_are_valid(x, method):
    res = bootstrap(x, method=method, n_bootstraps=5, random_state=1)
    idx = res.indices()
    assert idx is not None
    assert idx.min() >= 0
    assert idx.max() < len(x)
    assert (res.inbag_counts().sum(axis=1) == len(x)).all()


@given(x=finite_series(), method=st.sampled_from(OBS_METHODS))
@_SETTINGS
def test_resampled_values_come_from_the_original(x, method):
    # Index-resampling methods only reuse original observations (tapering rescales).
    if isinstance(method, TaperedBlock):
        return
    res = bootstrap(x, method=method, n_bootstraps=5, random_state=2)
    original = set(np.round(x, 9).tolist())
    for sample in res.values():
        assert set(np.round(sample, 9).tolist()).issubset(original)


@given(x=ar1_series(), seed=st.integers(0, 2**32 - 1))
@_MODEL_SETTINGS
def test_model_methods_are_deterministic_and_finite(x, seed):
    for method in (ResidualBootstrap(model=AR(order=1)), SieveAR()):
        a = bootstrap(x, method=method, n_bootstraps=4, random_state=seed)
        b = bootstrap(x, method=method, n_bootstraps=4, random_state=seed)
        np.testing.assert_array_equal(a.values(), b.values())
        assert a.values().shape == (4, len(x))
        assert np.isfinite(a.values()).all()
        assert a.indices() is None  # recursive methods fabricate no indices
