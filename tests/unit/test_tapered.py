"""Tests for the tapered block bootstrap (tsbootstrap.block.tapered).

The variance-preservation test is the gate: if energy normalization is wrong,
the tapered bootstrap distorts the variance and the method must be cut.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsbootstrap.api import bootstrap
from tsbootstrap.block.tapered import make_taper_window
from tsbootstrap.methods import TaperedBlock

WINDOWS = ["bartlett", "blackman", "hamming", "hann", "tukey"]


def _ar1(phi: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


class TestTaperWindow:
    @pytest.mark.parametrize("name", WINDOWS)
    def test_window_is_energy_normalized(self, name):
        w = make_taper_window(name, 32, alpha=0.5)
        assert np.isclose(np.mean(w**2), 1.0, atol=1e-12)

    def test_window_is_not_floored(self):
        # A correct hann taper dips well below 0.1 (no flooring of the window weights).
        w = make_taper_window("hann", 40)
        assert w.min() < 0.1
        assert not np.any(np.isclose(w, 0.1))


class TestTaperedBlock:
    @pytest.mark.parametrize("name", WINDOWS)
    def test_runs_and_shape(self, name):
        x = _ar1(0.6, 200, 0)
        res = bootstrap(x, method=TaperedBlock(window=name, block_length=10), n_bootstraps=5, random_state=1)
        assert res.values().shape == (5, 200)

    def test_determinism(self):
        x = _ar1(0.5, 150, 2)
        a = bootstrap(x, method=TaperedBlock(window="hann", block_length=8), n_bootstraps=6, random_state=3)
        b = bootstrap(x, method=TaperedBlock(window="hann", block_length=8), n_bootstraps=6, random_state=3)
        np.testing.assert_array_equal(a.values(), b.values())

    @pytest.mark.parametrize("name", WINDOWS)
    def test_variance_preservation_gate(self, name):
        x = _ar1(0.5, 400, 4)
        res = bootstrap(x, method=TaperedBlock(window=name, block_length=20), n_bootstraps=400, random_state=5)
        boot_var = res.values().var(axis=1).mean()
        ratio = boot_var / x.var()
        assert 0.75 <= ratio <= 1.3, f"{name}: variance ratio {ratio:.3f} outside band"

    def test_mean_preservation(self):
        x = _ar1(0.6, 400, 6)
        res = bootstrap(x, method=TaperedBlock(window="hann", block_length=20), n_bootstraps=400, random_state=7)
        boot_means = res.values().mean(axis=1)
        assert abs(boot_means.mean() - x.mean()) < 0.2 * x.std()
