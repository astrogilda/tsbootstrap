"""Tests for exogenous-covariate support (ARX residual bootstrap)."""

from __future__ import annotations

import numpy as np
import pytest

from tsbootstrap import AR, ARIMA, MovingBlock, ResidualBootstrap, bootstrap
from tsbootstrap.errors import MethodConfigError, TSBootstrapError
from tsbootstrap.model.fit import fit_ar


def _arx(n: int, phi: float, beta: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    exog = rng.standard_normal((n, 1))
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + beta * exog[t, 0] + e[t]
    return x, exog


def test_arx_recovers_the_exog_effect():
    x, exog = _arx(500, 0.5, 2.0, 0)
    spec = ResidualBootstrap(model=AR(order=1))
    res = bootstrap(x, method=spec, n_bootstraps=200, random_state=0, exog=exog)
    fitted_beta = fit_ar(x, 1, exog).exog_coefs[0]
    boot_betas = np.array([fit_ar(s, 1, exog).exog_coefs[0] for s in res.values()])
    assert abs(boot_betas.mean() - fitted_beta) < 0.15


def test_arx_shape_and_determinism():
    x, exog = _arx(300, 0.4, 1.0, 1)
    spec = ResidualBootstrap(model=AR(order=1))
    a = bootstrap(x, method=spec, n_bootstraps=8, random_state=7, exog=exog)
    b = bootstrap(x, method=spec, n_bootstraps=8, random_state=7, exog=exog)
    assert a.values().shape == (8, 300)
    np.testing.assert_array_equal(a.values(), b.values())


def test_exog_rejected_for_block_method():
    x, exog = _arx(100, 0.5, 1.0, 2)
    with pytest.raises(MethodConfigError):
        bootstrap(x, method=MovingBlock(block_length=5), n_bootstraps=2, exog=exog)


def test_exog_not_yet_for_arima():
    x, exog = _arx(120, 0.5, 1.0, 3)
    with pytest.raises(MethodConfigError):
        bootstrap(x, method=ResidualBootstrap(model=ARIMA(order=(1, 1, 1))), n_bootstraps=2, exog=exog)


def test_exog_length_mismatch_raises():
    x, exog = _arx(120, 0.5, 1.0, 4)
    with pytest.raises(TSBootstrapError):
        bootstrap(x, method=ResidualBootstrap(model=AR(order=1)), n_bootstraps=2, exog=exog[:50])
