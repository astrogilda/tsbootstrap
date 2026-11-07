"""An explicit block length longer than a single series: shortened with one warning.

The same rule as the panel backends (resolve_panel_block_lengths): the length is
shortened to the series length and one DegenerateBlockBootstrapWarning is emitted.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tsbootstrap import CircularBlock, MovingBlock, NonOverlappingBlock, StationaryBlock, bootstrap
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


def _run(method, backend):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = bootstrap(_X, method=method, n_bootstraps=6, random_state=3, backend=backend)
    degenerate = [w for w in caught if issubclass(w.category, DegenerateBlockBootstrapWarning)]
    return res, degenerate


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize(
    ("cls", "field"),
    [
        (MovingBlock, "block_length"),
        (CircularBlock, "block_length"),
        (NonOverlappingBlock, "block_length"),
        (StationaryBlock, "avg_block_length"),
    ],
)
def test_over_long_length_is_shortened_with_one_warning(cls, field, backend):
    res, warned = _run(cls(**{field: 45}), backend)
    assert len(warned) == 1
    msg = str(warned[0].message)
    assert msg.count(f"[{Codes.DEGENERATE_BLOCK_BOOTSTRAP}]") == 1
    assert warned[0].message.context == {"block_length": 45, "n": 30}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DegenerateBlockBootstrapWarning)
        ref = bootstrap(
            _X, method=cls(**{field: 30}), n_bootstraps=6, random_state=3, backend=backend
        )
    np.testing.assert_array_equal(res.values(), ref.values())


@pytest.mark.parametrize("backend", _BACKENDS)
def test_stationary_mean_equal_to_n_is_not_shortened(backend):
    _res, warned = _run(StationaryBlock(avg_block_length=30), backend)
    assert warned == []


def test_fixed_block_length_equal_to_n_warns_once():
    _res, warned = _run(MovingBlock(block_length=30), "numpy")
    assert len(warned) == 1


def test_warning_rebuilt_from_its_text_keeps_one_prefix():
    w = DegenerateBlockBootstrapWarning("x")
    again = DegenerateBlockBootstrapWarning(str(w))
    assert str(again) == str(w)


def test_non_positive_length_raises():
    from tsbootstrap.block.pwsd import resolve_block_length

    with pytest.raises(MethodConfigError):
        resolve_block_length(0, _X, kind="circular")
