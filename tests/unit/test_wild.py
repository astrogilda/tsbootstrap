"""Tests for the wild and block-wild bootstrap innovations.

The exact-magnitude test is the gate: a Rademacher multiplier flips signs and
nothing else, so ``|e*_t| == |e_hat_t|`` must hold to the last bit. Any
centering, scaling, or alignment defect in the wild path breaks that identity
immediately, which is why it anchors this suite alongside the statistical
moment checks on the multiplier distributions themselves.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._helpers.dgp import ar1
from tsbootstrap.api import bootstrap
from tsbootstrap.errors import Codes, DegenerateBlockBootstrapWarning, MethodConfigError
from tsbootstrap.methods import (
    AR,
    ARIMA,
    VAR,
    BlockWild,
    MovingBlock,
    ResidualBootstrap,
    SieveAR,
    Wild,
)
from tsbootstrap.model.recursive import (
    _draw_innovations_and_inits,
    _draw_multipliers,
    _prepare_residual,
)

DISTRIBUTIONS = ["rademacher", "gaussian", "mammen"]


def _ar_wild_context(x, innovation, exog=None):
    ctx = _prepare_residual(
        x.reshape(-1, 1), ResidualBootstrap(model=AR(order=1), innovation=innovation), exog
    )
    assert not hasattr(ctx, "reason"), "preparation unexpectedly failed"
    return ctx


class TestMultiplierDistributions:
    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    def test_mean_zero_unit_variance(self, dist):
        v = _draw_multipliers(np.random.default_rng(0), dist, 200_000)
        assert abs(v.mean()) < 0.01
        assert abs(v.var() - 1.0) < 0.01

    def test_mammen_third_moment_is_one(self):
        v = _draw_multipliers(np.random.default_rng(0), "mammen", 200_000)
        assert abs((v**3).mean() - 1.0) < 0.02

    def test_rademacher_support_is_exactly_plus_minus_one(self):
        v = _draw_multipliers(np.random.default_rng(0), "rademacher", 10_000)
        assert set(np.unique(v)) == {-1.0, 1.0}


class TestWildDrawIdentities:
    def test_ar_rademacher_magnitude_identity(self):
        # The wild signature: multipliers change signs only, so every replicate's
        # innovations match the centered residuals in absolute value, exactly.
        x = ar1(0.5, 200, 0)
        ctx = _ar_wild_context(x, Wild())
        gens = [np.random.default_rng(s) for s in range(8)]
        e_star, _ = _draw_innovations_and_inits(ctx, gens, x.shape[0] - 1)
        eps = ctx.resampling_innovations
        for i in range(len(gens)):
            np.testing.assert_array_equal(np.abs(e_star[i]), np.abs(eps))

    def test_var_one_multiplier_per_time_step(self):
        # A single scalar multiplier per time step, broadcast across components,
        # preserves the cross-sectional residual covariance: every row of
        # e_star[i] / eps is a constant vector (the shared sign).
        rng = np.random.default_rng(1)
        x = np.cumsum(rng.standard_normal((300, 3)), axis=0) * 0.1 + rng.standard_normal((300, 3))
        ctx = _prepare_residual(
            x, ResidualBootstrap(model=VAR(order=1), innovation=Wild()), None
        )
        gens = [np.random.default_rng(s) for s in range(4)]
        e_star, _ = _draw_innovations_and_inits(ctx, gens, x.shape[0] - 1)
        eps = ctx.resampling_innovations
        for i in range(len(gens)):
            ratio = e_star[i] / eps
            # each time step's d ratios are identical (one shared multiplier)
            np.testing.assert_allclose(
                ratio, np.broadcast_to(ratio[:, [0]], ratio.shape), rtol=0, atol=1e-15
            )
            np.testing.assert_array_equal(np.abs(e_star[i]), np.abs(eps))

    def test_block_wild_multipliers_constant_within_blocks(self):
        x = ar1(0.4, 51, 3)
        ctx = _ar_wild_context(x, BlockWild(block_length=7))
        gens = [np.random.default_rng(11)]
        n_steps = x.shape[0] - 1  # 50 residuals
        e_star, _ = _draw_innovations_and_inits(ctx, gens, n_steps)
        eps = ctx.resampling_innovations
        v = e_star[0] / eps
        # constant on [0,7), [7,14), ..., [49,50)
        for start in range(0, n_steps, 7):
            block = v[start : start + 7]
            np.testing.assert_allclose(block, block[0], rtol=0, atol=1e-15)
        # and the block values are the generator's first ceil(50/7)=8 draws, in order
        expected = _draw_multipliers(np.random.default_rng(11), "rademacher", 8)
        np.testing.assert_allclose(v[::7], expected[: len(v[::7])], rtol=0, atol=1e-15)

    def test_arx_exog_context_does_not_touch_the_multiplier_step(self):
        # With exogenous regressors in the context, the draw step still satisfies
        # the exact magnitude identity: the deterministic exog forcing is added
        # after the multiplier step, never scaled by it.
        rng = np.random.default_rng(7)
        exog = rng.standard_normal((200, 2))
        x = ar1(0.5, 200, 5) + exog @ np.array([0.8, -0.3])
        ctx = _ar_wild_context(x, Wild(), exog=exog)
        assert ctx.exog_state is not None
        gens = [np.random.default_rng(s) for s in range(4)]
        e_star, _ = _draw_innovations_and_inits(ctx, gens, x.shape[0] - 1)
        eps = ctx.resampling_innovations
        for i in range(len(gens)):
            np.testing.assert_array_equal(np.abs(e_star[i]), np.abs(eps))


class TestWildEndToEnd:
    @pytest.mark.parametrize("dist", DISTRIBUTIONS)
    def test_ar_runs_deterministic_and_shaped(self, dist):
        x = ar1(0.5, 150, 2)
        spec = ResidualBootstrap(model=AR(order=1), innovation=Wild(distribution=dist))
        a = bootstrap(x, method=spec, n_bootstraps=6, random_state=3)
        b = bootstrap(x, method=spec, n_bootstraps=6, random_state=3)
        np.testing.assert_array_equal(a.values(), b.values())
        assert a.values().shape == (6, 150)
        assert np.all(np.isfinite(a.values()))

    def test_var_and_arima_and_sieve_run(self):
        rng = np.random.default_rng(4)
        xm = rng.standard_normal((120, 2))
        cases = [
            (ResidualBootstrap(model=VAR(order=1), innovation=BlockWild(block_length=5)), xm),
            (ResidualBootstrap(model=ARIMA(order=(1, 0, 0)), innovation=Wild()), xm[:, 0]),
            (SieveAR(innovation=Wild(distribution="mammen")), xm[:, 0]),
        ]
        for spec, data in cases:
            res = bootstrap(data, method=spec, n_bootstraps=4, random_state=9)
            vals = res.values()
            assert vals.shape[0] == 4
            assert np.all(np.isfinite(vals))

    def test_arima_wild_deterministic(self):
        x = ar1(0.6, 180, 8)
        spec = ResidualBootstrap(model=ARIMA(order=(1, 0, 1)), innovation=Wild())
        a = bootstrap(x, method=spec, n_bootstraps=5, random_state=13)
        b = bootstrap(x, method=spec, n_bootstraps=5, random_state=13)
        np.testing.assert_array_equal(a.values(), b.values())

    def test_arx_wild_end_to_end(self):
        rng = np.random.default_rng(21)
        exog = rng.standard_normal((160, 1))
        x = ar1(0.4, 160, 22) + 0.9 * exog[:, 0]
        spec = ResidualBootstrap(model=AR(order=1), innovation=Wild())
        a = bootstrap(x, method=spec, n_bootstraps=5, random_state=6, exog=exog)
        b = bootstrap(x, method=spec, n_bootstraps=5, random_state=6, exog=exog)
        np.testing.assert_array_equal(a.values(), b.values())
        assert np.all(np.isfinite(a.values()))


class TestWildErrorPaths:
    def test_wild_requires_burn_in_zero(self):
        x = ar1(0.5, 100, 1)
        spec = ResidualBootstrap(model=AR(order=1, burn_in=5), innovation=Wild())
        with pytest.raises(MethodConfigError) as err:
            bootstrap(x, method=spec, n_bootstraps=2, random_state=0)
        assert err.value.code == Codes.UNSUPPORTED_MODEL_FEATURE
        assert "burn_in" in str(err.value)

    def test_wild_requires_fixed_initial(self):
        x = ar1(0.5, 100, 1)
        spec = ResidualBootstrap(model=AR(order=1, initial="random_block"), innovation=Wild())
        with pytest.raises(MethodConfigError) as err:
            bootstrap(x, method=spec, n_bootstraps=2, random_state=0)
        assert err.value.code == Codes.UNSUPPORTED_MODEL_FEATURE
        assert "initial" in str(err.value)

    def test_block_wild_length_exceeding_residuals_raises(self):
        x = ar1(0.5, 100, 1)  # 99 residuals for AR(1)
        spec = ResidualBootstrap(model=AR(order=1), innovation=BlockWild(block_length=200))
        with pytest.raises(MethodConfigError) as err:
            bootstrap(x, method=spec, n_bootstraps=2, random_state=0)
        assert err.value.code == Codes.BLOCK_LENGTH_GT_RESIDUALS

    def test_block_wild_degenerate_length_warns_and_runs(self):
        x = ar1(0.5, 100, 1)  # 99 residuals; L == 99 is one block over the whole path
        spec = ResidualBootstrap(model=AR(order=1), innovation=BlockWild(block_length=99))
        with pytest.warns(DegenerateBlockBootstrapWarning):
            res = bootstrap(x, method=spec, n_bootstraps=2, random_state=0)
        assert np.all(np.isfinite(res.values()))

    def test_block_innovations_still_rejected(self):
        # Pre-existing contract: block innovation specs are constructible but not
        # executable; the wild work must not silently enable them.
        x = ar1(0.5, 100, 1)
        spec = ResidualBootstrap(model=AR(order=1), innovation=MovingBlock(block_length=5))
        with pytest.raises(MethodConfigError) as err:
            bootstrap(x, method=spec, n_bootstraps=2, random_state=0)
        assert err.value.code == Codes.UNSUPPORTED_MODEL_FEATURE

    def test_sieve_wild_requires_burn_in_zero(self):
        x = ar1(0.5, 120, 2)
        with pytest.raises(MethodConfigError):
            bootstrap(x, method=SieveAR(burn_in=3, innovation=Wild()), n_bootstraps=2, random_state=0)


class TestCompiledBackendGuard:
    def test_compiled_supports_rejects_wild_innovations(self):
        from tsbootstrap.block._compiled import compiled_supports, unsupported_method_error
        from tsbootstrap.methods import IID

        wild_spec = ResidualBootstrap(model=AR(order=1), innovation=Wild())
        iid_spec = ResidualBootstrap(model=AR(order=1), innovation=IID())
        assert compiled_supports(wild_spec) is False
        assert compiled_supports(iid_spec) is True
        err = unsupported_method_error(wild_spec)
        assert "Wild" in str(err)
        assert "ResidualBootstrap" not in str(err).split(";")[0]  # names the innovation, not the method

    def test_compiled_reduce_refuses_wild(self):
        pytest.importorskip("numba")
        from tsbootstrap.api import bootstrap_reduce

        x = ar1(0.5, 100, 1)
        spec = ResidualBootstrap(model=AR(order=1), innovation=BlockWild(block_length=5))
        with pytest.raises(MethodConfigError) as err:
            bootstrap_reduce(x, method=spec, statistic="mean", n_bootstraps=4, backend="compiled")
        assert "BlockWild" in str(err.value)
