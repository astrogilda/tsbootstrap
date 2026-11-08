"""The stationary bootstrap's block lengths are geometric with the requested real mean.

Both backends. A position t > 0 starts a new block with probability p = 1 / L; the
observed break frequency (a new block whose start is not the next index) is
p * (1 - 1 / n). The gap from the first break after t = 0 to the next one is exactly
Geometric(p * (1 - 1 / n)) (the first break is a stopping time), so a chi-square test on
it checks the whole distribution, not only the mean. A run is not counted from t = 0 or
up to t = n - 1, so the window does not truncate it.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

import tsbootstrap as tsb
from tsbootstrap.block._compiled import restart_threshold
from tsbootstrap.methods import StationaryBlock

_BACKENDS = [
    "numpy",
    pytest.param(
        "compiled",
        marks=pytest.mark.skipif(
            __import__("importlib").util.find_spec("numba") is None, reason="needs numba"
        ),
    ),
]


def _breaks(backend: str, mean: float, n: int, b: int, seed: int) -> np.ndarray:
    x = np.random.default_rng(0).standard_normal(n)
    idx = np.asarray(
        tsb.bootstrap(
            x, method=StationaryBlock(avg_block_length=mean), n_bootstraps=b,
            random_state=seed, backend=backend,
        ).indices()
    )
    return idx[:, 1:] != (idx[:, :-1] + 1) % n


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("mean", [5.0, 5.5, 1.75])
def test_break_frequency_and_gap_distribution(backend, mean):
    n, b = 400, 4000
    brk = _breaks(backend, mean, n, b, seed=11)
    p = (1.0 / mean) * (1.0 - 1.0 / n)
    freq = brk.mean()
    se = np.sqrt(p * (1 - p) / brk.size)
    assert abs(freq - p) < 4 * se, (freq, p, se)
    gaps = []
    for row in brk:
        pos = np.flatnonzero(row)
        if pos.size >= 2:
            gaps.append(pos[1] - pos[0])
    gaps = np.asarray(gaps)
    k_max = 12
    obs = np.bincount(np.minimum(gaps, k_max), minlength=k_max + 1)[1:]
    pmf = stats.geom(p).pmf(np.arange(1, k_max))
    exp = np.r_[pmf, 1.0 - pmf.sum()] * gaps.size
    assert stats.chisquare(obs, exp).pvalue > 1e-4


def test_restart_threshold_is_exact():
    assert restart_threshold(1.0) == 2**64 - 1
    assert restart_threshold(2.0) == 2**63 - 1
    assert restart_threshold(5.5) == (2**64 * 2) // 11 - 1
    with pytest.raises(Exception):
        restart_threshold(0.5)
