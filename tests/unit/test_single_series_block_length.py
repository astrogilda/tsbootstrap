"""Block lengths that leave every series degenerate raise; real-valued stationary means.

One rule for single series and panels (resolve_panel_block_lengths): an explicit length
that makes EVERY series degenerate raises BLOCK_LENGTH_GT_N, because every replicate
would repeat the series and every interval would have zero width; one that leaves at
least one panel series non-degenerate shortens the others with one warning.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tsbootstrap import CircularBlock, MovingBlock, NonOverlappingBlock, StationaryBlock, bootstrap
from tsbootstrap.block.pwsd import resolve_block_length, resolve_panel_block_lengths
from tsbootstrap.errors import Codes, DegenerateBlockBootstrapWarning, MethodConfigError

_X = np.random.default_rng(0).standard_normal(30)
_BACKENDS = [
    "numpy",
    pytest.param(
        "compiled",
        marks=pytest.mark.skipif(
            __import__("importlib").util.find_spec("numba") is None, reason="needs numba"
        ),
    ),
]


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize(
    ("method", "n_ok"),
    [
        (MovingBlock(block_length=30), False),
        (NonOverlappingBlock(block_length=30), False),
        (CircularBlock(block_length=30), False),
        (MovingBlock(block_length=45), False),
        (StationaryBlock(avg_block_length=45), False),
        (StationaryBlock(avg_block_length=30), True),  # mean n still restarts
        (MovingBlock(block_length=29), True),
    ],
)
def test_single_series_degenerate_length_raises(method, n_ok, backend):
    if n_ok:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DegenerateBlockBootstrapWarning)
            bootstrap(_X, method=method, n_bootstraps=4, random_state=3, backend=backend)
        return
    with pytest.raises(MethodConfigError) as info:
        bootstrap(_X, method=method, n_bootstraps=4, random_state=3, backend=backend)
    assert info.value.code == Codes.BLOCK_LENGTH_GT_N


def test_panel_with_one_non_degenerate_series_shortens_the_rest_with_one_warning():
    flat = np.random.default_rng(1).standard_normal(18)
    indptr = np.array([0, 10, 18])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lengths = resolve_panel_block_lengths(9, flat, indptr, kind="circular")
    np.testing.assert_array_equal(lengths, [9, 8])
    assert sum(issubclass(w.category, DegenerateBlockBootstrapWarning) for w in caught) == 1


def test_panel_where_every_series_is_degenerate_raises():
    flat = np.random.default_rng(1).standard_normal(18)
    with pytest.raises(MethodConfigError) as info:
        resolve_panel_block_lengths(10, flat, np.array([0, 10, 18]), kind="circular")
    assert info.value.code == Codes.BLOCK_LENGTH_GT_N


def test_warning_rebuilt_from_its_text_keeps_one_prefix():
    w = DegenerateBlockBootstrapWarning("x")
    again = DegenerateBlockBootstrapWarning(str(w))
    assert str(again) == str(w)


@pytest.mark.parametrize("bad", [0, 0.5])
def test_length_below_one_raises(bad):
    with pytest.raises(MethodConfigError):
        resolve_block_length(bad, _X, kind="stationary")


def test_fixed_block_length_must_be_integral():
    with pytest.raises(MethodConfigError):
        resolve_block_length(2.5, _X, kind="circular")


def test_stationary_mean_length_is_real_valued():
    assert StationaryBlock(avg_block_length=5.5).avg_block_length == 5.5
    assert resolve_block_length(5.5, _X, kind="stationary") == 5.5
    with pytest.raises(ValueError):
        StationaryBlock(avg_block_length=0.9)
    with pytest.raises(ValueError):
        StationaryBlock(avg_block_length=float("inf"))
