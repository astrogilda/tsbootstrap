"""Reference cross-checks against an independent implementation (arch).

The block bootstrap's estimate of the standard error of the mean must agree with
arch's implementation of the same procedure on the same data. Exact equality is
not expected (different RNG streams), but the estimates must agree to within a
small factor.
"""

from __future__ import annotations

import pytest

from tests._helpers.dgp import ar1_stationary
from tsbootstrap import StationaryBlock, bootstrap


@pytest.mark.slow
def test_stationary_bootstrap_se_matches_arch():
    arch_bootstrap = pytest.importorskip("arch.bootstrap")
    x = ar1_stationary(500, 0.5, seed=0)
    block_len = 20
    n_boot = 2000

    mine = bootstrap(
        x, method=StationaryBlock(avg_block_length=block_len), n_bootstraps=n_boot, random_state=1
    )
    mine_se = float(mine.values().mean(axis=1).std())

    ref = arch_bootstrap.StationaryBootstrap(block_len, x, seed=12345)
    arch_means = ref.apply(lambda d: d.mean(), n_boot).ravel()
    arch_se = float(arch_means.std())

    ratio = mine_se / arch_se
    assert 0.8 <= ratio <= 1.25, (
        f"SE-of-mean ratio {ratio:.3f} (mine={mine_se:.4f}, arch={arch_se:.4f})"
    )
