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
        ctx = _prepare_residual(x, ResidualBootstrap(model=VAR(order=1), innovation=Wild()), None)
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
            bootstrap(
                x, method=SieveAR(burn_in=3, innovation=Wild()), n_bootstraps=2, random_state=0
            )


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
        assert (
            "ResidualBootstrap" not in str(err).split(";")[0]
        )  # names the innovation, not the method

    def test_compiled_reduce_refuses_wild(self):
        pytest.importorskip("numba")
        from tsbootstrap.api import bootstrap_reduce

        x = ar1(0.5, 100, 1)
        spec = ResidualBootstrap(model=AR(order=1), innovation=BlockWild(block_length=5))
        with pytest.raises(MethodConfigError) as err:
            bootstrap_reduce(x, method=spec, statistic="mean", n_bootstraps=4, backend="compiled")
        assert "BlockWild" in str(err.value)


class TestWildSurvivorKillers:
    """Deterministic gates added after the first full mutation run of the wild code.

    Each test names the surviving-mutant cluster it kills; together with the
    equivalents registry they account for every new survivor of that run.
    """

    def test_arima_wild_tail_identity_and_mammen_product(self):
        # Kills: _arima_batched plan=None / distribution=None / eps division,
        # _prepare_arima wild=None / dropped kwarg / _wild_plan(None, ...),
        # _draw_innovations_and_inits eps division (AR side, mammen product).
        from tsbootstrap.model.recursive import _arima_batched, _prepare_arima

        x = ar1(0.6, 160, 31)
        # Rademacher: continuation magnitudes match the centered residual tail exactly.
        ctx = _prepare_arima(x, ARIMA(order=(1, 0, 1)), None, Wild())
        gens = [np.random.default_rng(s) for s in range(3)]
        sims = _arima_batched(ctx, x.shape[0], gens, np.dtype(np.float64))
        assert sims.shape[0] == 3
        eps = ctx.resampling_innovations
        k = ctx.arma.init_w.shape[0]
        m_tail = eps.shape[0] - k
        # Reconstruct the innovations the simulation consumed: identical draws.
        for i in range(3):
            gen = np.random.default_rng(i)
            v = _draw_multipliers(gen, "rademacher", m_tail)
            np.testing.assert_array_equal(np.abs(eps[k:] * v), np.abs(eps[k:]))
        # A wild plan must be present and the tail multipliers non-constant (kills the
        # collapsed-to-scalar n_draw mutant: one shared multiplier would be constant).
        assert ctx.wild is not None
        v0 = _draw_multipliers(np.random.default_rng(0), "rademacher", m_tail)
        assert np.unique(v0).size > 1
        # Mirror reconstruction: rebuild the expected continuation from the same
        # fitted context, drawing the multipliers independently. A dropped wild
        # plan (IID index resampling), a wrong distribution branch, or division
        # instead of multiplication diverges from this reconstruction, while
        # platform floating-point differences affect both sides identically.
        from tsbootstrap.engines.arma_scipy import simulate_arma_batched
        from tsbootstrap.model.arima import arma_initial_state, integrate_batched

        def _expected_tail(ctx_e, dist):
            arma = ctx_e.arma
            eps_e = ctx_e.resampling_innovations
            k_e = arma.init_w.shape[0]
            m_e = eps_e.shape[0] - k_e
            v_e = _draw_multipliers(np.random.default_rng(7), dist, m_e)
            e_star_e = (eps_e[k_e:] * v_e)[None, :]
            init_state = arma_initial_state(
                arma.ar_coefs, arma.ma_coefs, arma.init_w, arma.residuals[:k_e]
            )
            w_centered = simulate_arma_batched(
                arma.ar_coefs,
                arma.ma_coefs,
                e_star_e,
                init_state=init_state,
                init_values=arma.init_w,
            )
            return integrate_batched(w_centered + arma.mean, ctx_e.levels)

        sims_r = _arima_batched(ctx, x.shape[0], [np.random.default_rng(7)], np.dtype(np.float64))
        np.testing.assert_array_equal(sims_r[0, :, 0], _expected_tail(ctx, "rademacher")[0])
        ctx_m = _prepare_arima(x, ARIMA(order=(1, 0, 1)), None, Wild(distribution="mammen"))
        assert ctx_m.wild is not None and ctx_m.wild.distribution == "mammen"
        sims_m = _arima_batched(ctx_m, x.shape[0], [np.random.default_rng(7)], np.dtype(np.float64))
        np.testing.assert_array_equal(sims_m[0, :, 0], _expected_tail(ctx_m, "mammen")[0])

    def test_ar_mammen_innovations_are_the_exact_product(self):
        # Kills: _draw_innovations_and_inits eps / v (division equals multiplication
        # for Rademacher but not for Mammen).
        x = ar1(0.5, 150, 33)
        ctx = _ar_wild_context(x, Wild(distribution="mammen"))
        gens = [np.random.default_rng(5)]
        e_star, _ = _draw_innovations_and_inits(ctx, gens, x.shape[0] - 1)
        v = _draw_multipliers(np.random.default_rng(5), "mammen", x.shape[0] - 1)
        np.testing.assert_array_equal(e_star[0], ctx.resampling_innovations * v)

    def test_arima_block_wild_non_dividing_length_runs_exactly(self):
        # Kills: the _arima_batched n_draw ceil-division mutants and every
        # np.repeat argument mutant (each raises or misshapes on a non-dividing L).
        x = ar1(0.6, 163, 35)
        spec = ResidualBootstrap(model=ARIMA(order=(1, 0, 0)), innovation=BlockWild(block_length=7))
        res = bootstrap(x, method=spec, n_bootstraps=3, random_state=11)
        vals = res.values()
        assert vals.shape == (3, 163)
        assert np.all(np.isfinite(vals))
        a = bootstrap(x, method=spec, n_bootstraps=3, random_state=11)
        np.testing.assert_array_equal(vals, a.values())

    def test_wild_compatibility_error_messages_exact(self):
        # Kills: every _check_wild_compatible message-string mutant.
        x = ar1(0.5, 100, 37)
        with pytest.raises(MethodConfigError) as err_init:
            bootstrap(
                x,
                method=ResidualBootstrap(
                    model=AR(order=1, initial="random_block"), innovation=Wild()
                ),
                n_bootstraps=2,
                random_state=0,
            )
        assert (
            "wild innovations require initial='fixed' (the multiplier stream is aligned "
            "one-to-one with the residuals, conditional on the observed initial values)"
        ) in str(err_init.value)
        with pytest.raises(MethodConfigError) as err_burn:
            bootstrap(
                x,
                method=ResidualBootstrap(model=AR(order=1, burn_in=3), innovation=Wild()),
                n_bootstraps=2,
                random_state=0,
            )
        assert (
            "wild innovations require burn_in=0 (there is one multiplier per residual; "
            "burn-in steps would have no residual to multiply)"
        ) in str(err_burn.value)

    def test_block_wild_auto_resolves_and_is_deterministic(self):
        # Kills: the _wild_plan "auto" comparison mutants, the arr2d reshape mutants
        # reachable on the 1-D path, and the optimal_block_length argument mutants.
        x = ar1(0.7, 240, 39)
        spec = ResidualBootstrap(model=AR(order=1), innovation=BlockWild(block_length="auto"))
        a = bootstrap(x, method=spec, n_bootstraps=4, random_state=3)
        b = bootstrap(x, method=spec, n_bootstraps=4, random_state=3)
        np.testing.assert_array_equal(a.values(), b.values())
        assert np.all(np.isfinite(a.values()))
        # VAR: the 2-D residual matrix must reach the block-length rule per column,
        # not flattened (a flatten changes the resolved length and the whole output).
        rng = np.random.default_rng(41)
        xm = rng.standard_normal((180, 2))
        specv = ResidualBootstrap(model=VAR(order=1), innovation=BlockWild(block_length="auto"))
        va = bootstrap(xm, method=specv, n_bootstraps=3, random_state=5)
        vb = bootstrap(xm, method=specv, n_bootstraps=3, random_state=5)
        np.testing.assert_array_equal(va.values(), vb.values())
        assert np.all(np.isfinite(va.values()))

    def test_block_wild_auto_uses_per_column_lengths_for_var(self):
        # Kills: the _wild_plan arr2d conditional mutants that flatten a 2-D
        # (VAR) residual matrix before the block-length rule. Column 0 carries
        # strong dependence, column 1 is white noise: the per-column maximum and
        # the flattened estimate provably differ (28 vs 40 on this fixture).
        from tsbootstrap.block.pwsd import optimal_block_length
        from tsbootstrap.model.recursive import _wild_plan

        rng = np.random.default_rng(2)
        n = 300
        e0 = np.empty(n)
        e0[0] = rng.standard_normal()
        for t in range(1, n):
            e0[t] = 0.9 * e0[t - 1] + rng.standard_normal()
        centered = np.column_stack([e0, rng.standard_normal(n)])
        plan = _wild_plan(BlockWild(block_length="auto"), centered)
        assert plan is not None
        expected = optimal_block_length(centered, kind="circular")
        flattened = optimal_block_length(centered.reshape(-1, 1), kind="circular")
        assert expected != flattened  # the fixture discriminates, or this test is vacuous
        assert plan.block_length == expected

    def test_block_length_gt_residuals_error_payload_exact(self):
        # Kills: the BLOCK_LENGTH_GT_RESIDUALS message and context mutants.
        x = ar1(0.5, 100, 43)  # AR(1): 99 residuals
        spec = ResidualBootstrap(model=AR(order=1), innovation=BlockWild(block_length=200))
        with pytest.raises(MethodConfigError) as err:
            bootstrap(x, method=spec, n_bootstraps=2, random_state=0)
        assert "block_length 200 exceeds the number of residuals 99" in str(err.value)
        assert err.value.context == {"block_length": 200, "n_residuals": 99}

    def test_degenerate_block_wild_warns_exactly_and_uses_one_block(self):
        # Kills: the degenerate-branch warning message/context mutants and the
        # dropped clamp (length=None would silently degrade to plain wild).
        from tsbootstrap.model.recursive import _wild_plan

        x = ar1(0.5, 100, 45)
        ctx_probe = _ar_wild_context(x, Wild())
        m = ctx_probe.resampling_innovations.shape[0]
        with pytest.warns(
            DegenerateBlockBootstrapWarning,
            match=rf"block-wild block length {m} >= residual count {m}; one multiplier covers the whole path",
        ) as rec:
            plan = _wild_plan(BlockWild(block_length=m), ctx_probe.resampling_innovations)
        assert plan is not None and plan.block_length == m  # clamped, NOT None
        assert rec[0].message.context == {"block_length": m, "n_residuals": m}
        # End to end: the whole path shares one multiplier (constant ratio).
        spec = ResidualBootstrap(model=AR(order=1), innovation=BlockWild(block_length=m))
        with pytest.warns(DegenerateBlockBootstrapWarning):
            ctx = _prepare_residual(x.reshape(-1, 1), spec, None)
        gens = [np.random.default_rng(9)]
        e_star, _ = _draw_innovations_and_inits(ctx, gens, m)
        ratio = e_star[0] / ctx.resampling_innovations
        np.testing.assert_allclose(ratio, ratio[0], rtol=0, atol=1e-15)
