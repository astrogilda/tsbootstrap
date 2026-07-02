"""Tests for the classical confidence-interval layer (tsbootstrap.uq.classical)."""

from __future__ import annotations

import numpy as np
import pytest

from tsbootstrap.errors import Codes, MethodConfigError
from tsbootstrap.uq import (
    basic_interval,
    bca_interval,
    block_jackknife_se,
    jackknife_acceleration,
    jackknife_statistics,
    percentile_interval,
    studentized_interval,
)


def _mean_stat(values, _indices):
    return float(np.mean(values))


class TestPercentileBasic:
    def test_percentile_matches_numpy_quantile_1d(self):
        rng = np.random.default_rng(0)
        stats = rng.standard_normal(500)
        lo, hi = percentile_interval(stats, alpha=0.1)
        np.testing.assert_array_equal(lo, np.quantile(stats, 0.05))
        np.testing.assert_array_equal(hi, np.quantile(stats, 0.95))

    def test_percentile_matches_numpy_quantile_2d(self):
        rng = np.random.default_rng(1)
        stats = rng.standard_normal((500, 3))
        lo, hi = percentile_interval(stats, alpha=0.1)
        np.testing.assert_array_equal(lo, np.quantile(stats, 0.05, axis=0))
        np.testing.assert_array_equal(hi, np.quantile(stats, 0.95, axis=0))
        assert lo.shape == (3,)
        assert hi.shape == (3,)

    def test_basic_reflection_hand_pin(self):
        # Sorted [0, 1, 2, 3, 10]; alpha=0.5 -> q_lo = quantile(.25) = 1.0,
        # q_hi = quantile(.75) = 3.0. theta_hat = 5.0, so the reflected bounds are
        # lower = 2*5 - 3 = 7.0 and upper = 2*5 - 1 = 9.0 (asymmetric about theta_hat,
        # so a percentile interval (1.0, 3.0) would fail this pin).
        stats = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
        lo, hi = basic_interval(stats, 5.0, alpha=0.5)
        np.testing.assert_array_equal(lo, 7.0)
        np.testing.assert_array_equal(hi, 9.0)

    @pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.5])
    def test_percentile_rejects_alpha_out_of_range(self, bad_alpha):
        stats = np.arange(10.0)
        with pytest.raises(MethodConfigError) as exc:
            percentile_interval(stats, alpha=bad_alpha)
        assert exc.value.code == Codes.INVALID_PARAMETER

    @pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.5])
    def test_basic_rejects_alpha_out_of_range(self, bad_alpha):
        stats = np.arange(10.0)
        with pytest.raises(MethodConfigError) as exc:
            basic_interval(stats, 4.5, alpha=bad_alpha)
        assert exc.value.code == Codes.INVALID_PARAMETER

    def test_percentile_rejects_empty(self):
        with pytest.raises(ValueError):
            percentile_interval(np.empty(0))


class TestJackknife:
    def test_jackknife_statistics_leave_one_out_means(self):
        x = np.array([2.0, 4.0, 6.0, 8.0])
        got = jackknife_statistics(x, _mean_stat)
        # Leave-one-out means: total = 20, theta_(i) = (20 - x_i) / 3.
        expected = np.array([(20.0 - xi) / 3.0 for xi in x])
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_block_length_one_equals_delete_one_closed_form(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal(40)
        se = block_jackknife_se(x, _mean_stat, block_length=1)
        expected = float(np.std(x, ddof=1) / np.sqrt(x.shape[0]))
        np.testing.assert_allclose(se, expected, rtol=1e-12)

    def test_block_jackknife_g_minus_1_over_g_hand_pin(self):
        # x=[1..6], L=2 -> g=3 blocks. Deleting each block leaves means 4.5, 3.5, 2.5;
        # their mean is 3.5 so sum of squared deviations is 1 + 0 + 1 = 2, and
        # se^2 = (g-1)/g * 2 = (2/3)*2 = 4/3.
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        se = block_jackknife_se(x, _mean_stat, block_length=2)
        np.testing.assert_allclose(se, np.sqrt(4.0 / 3.0), rtol=1e-12)

    def test_block_se_exceeds_delete_one_under_dependence(self):
        # Positively autocorrelated AR(1); the block jackknife (which respects the
        # dependence) must inflate the SE relative to the iid delete-one estimate.
        rng = np.random.default_rng(0)
        phi = 0.8
        n = 400
        x = np.empty(n)
        x[0] = rng.standard_normal()
        for t in range(1, n):
            x[t] = phi * x[t - 1] + rng.standard_normal()
        block_se = block_jackknife_se(x, _mean_stat, block_length=20)
        delete_one = block_jackknife_se(x, _mean_stat, block_length=1)
        assert block_se > delete_one

    def test_too_few_groups_raises(self):
        x = np.array([1.0, 2.0, 3.0])
        with pytest.raises(MethodConfigError) as exc:
            block_jackknife_se(x, _mean_stat, block_length=2)
        assert exc.value.code == Codes.INVALID_PARAMETER


class TestStudentized:
    def test_pivot_orientation_sign_pin(self):
        # se_b == se_hat == 1, theta_hat == 0, alpha=0.5 -> t == statistics, and
        # t_lo = quantile(.25) = 1.0, t_hi = quantile(.75) = 3.0. The correct minus
        # sign puts the UPPER pivot quantile on the LOWER bound:
        #   lower = 0 - 3.0 = -3.0, upper = 0 - 1.0 = -1.0.
        # A '+' mutant would return (3.0, 1.0); a quantile swap would return (-1, -3).
        stats = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
        ses = np.ones_like(stats)
        lo, hi = studentized_interval(stats, ses, 0.0, 1.0, alpha=0.5)
        np.testing.assert_array_equal(lo, -3.0)
        np.testing.assert_array_equal(hi, -1.0)
        assert lo < hi

    def test_zero_replicate_se_raises(self):
        stats = np.array([1.0, 2.0, 3.0])
        ses = np.array([1.0, 0.0, 1.0])
        with pytest.raises(ValueError):
            studentized_interval(stats, ses, 2.0, 1.0)

    def test_zero_point_se_raises(self):
        stats = np.array([1.0, 2.0, 3.0])
        ses = np.array([1.0, 1.0, 1.0])
        with pytest.raises(ValueError):
            studentized_interval(stats, ses, 2.0, 0.0)


class TestBCa:
    def test_acceleration_hand_value(self):
        # x=[1,2,10], mean. Leave-one-out means are [6, 5.5, 1.5]; theta_dot = 13/3.
        # d = theta_dot - jack = [-5/3, -7/6, 17/6]. sum d^3 = 3570/216, sum d^2 = 438/36,
        # so a = (3570/216) / (6 * (438/36)^1.5) = 0.06490913176430597.
        x = np.array([1.0, 2.0, 10.0])
        a = jackknife_acceleration(x, _mean_stat)
        np.testing.assert_allclose(a, 0.06490913176430597, rtol=1e-12)

    def test_acceleration_zero_for_symmetric_jackknife(self):
        # Symmetric sample -> symmetric leave-one-out means -> sum(d^3) = 0 -> a = 0.
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        a = jackknife_acceleration(x, _mean_stat)
        np.testing.assert_array_equal(a, 0.0)

    def test_bca_equals_percentile_when_z0_and_accel_zero(self):
        # Symmetric replicates about theta_hat=0 give p0=0.5 (z0=0); with acceleration=0
        # the adjusted levels collapse to alpha/2 and 1-alpha/2, i.e. the percentile
        # interval (up to the ~1e-16 ndtr(ndtri(.)) roundtrip at non-half levels).
        stats = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
        lo_b, hi_b = bca_interval(stats, 0.0, 0.0, alpha=0.2)
        lo_p, hi_p = percentile_interval(stats, alpha=0.2)
        np.testing.assert_allclose(lo_b, lo_p, rtol=1e-12)
        np.testing.assert_allclose(hi_b, hi_p, rtol=1e-12)

    def test_bca_tie_adjustment_pins_half_weight(self):
        # stats=[-1,0,0,1], theta_hat=0: #{<0}=1, #{==0}=2, so the tie-adjusted
        # p0 = (1 + 0.5*2)/4 = 0.5 -> z0=0, and with acceleration=0 BCa reduces to the
        # percentile interval. Dropping the 0.5*ties term (p0=0.25) or counting ties as
        # strictly-below (p0=0.75) would move z0 off zero and break this equality.
        stats = np.array([-1.0, 0.0, 0.0, 1.0])
        lo_b, hi_b = bca_interval(stats, 0.0, 0.0, alpha=0.5)
        lo_p, hi_p = percentile_interval(stats, alpha=0.5)
        np.testing.assert_array_equal(lo_b, lo_p)
        np.testing.assert_array_equal(hi_b, hi_p)

    def test_degenerate_p0_raises(self):
        # Every replicate above theta_hat -> p0 = 0 -> z0 = -inf -> degenerate.
        stats = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            bca_interval(stats, 0.0, 0.0)


class TestTopLevelExports:
    """The classical CI surface is re-exported at the top level and from the uq package."""

    _CLASSICAL_NAMES = (
        "percentile_interval",
        "basic_interval",
        "jackknife_statistics",
        "block_jackknife_se",
        "studentized_interval",
        "jackknife_acceleration",
        "bca_interval",
    )

    def test_names_exported_and_in_all(self):
        import tsbootstrap
        import tsbootstrap.uq as uq

        for name in self._CLASSICAL_NAMES:
            assert hasattr(tsbootstrap, name), name
            assert name in tsbootstrap.__all__, name
            assert name in uq.__all__, name

    def test_flat_import_matches_submodule(self):
        import tsbootstrap
        import tsbootstrap.uq as uq

        for name in self._CLASSICAL_NAMES:
            assert getattr(tsbootstrap, name) is getattr(uq, name)


class TestConfInt:
    def test_percentile_smoke_and_determinism(self):
        from tests._helpers.dgp import ar1
        from tsbootstrap import MovingBlock, conf_int

        x = ar1(0.5, 200, 0)
        a = conf_int(
            x, "mean", method=MovingBlock(block_length=10), n_bootstraps=200, random_state=1
        )
        b = conf_int(
            x, "mean", method=MovingBlock(block_length=10), n_bootstraps=200, random_state=1
        )
        for u, v in zip(a, b, strict=True):
            np.testing.assert_array_equal(u, v)
        lo, hi, point = a
        assert float(lo) <= float(point) <= float(hi)

    def test_string_and_callable_statistics_agree(self):
        from tests._helpers.dgp import ar1
        from tsbootstrap import MovingBlock, conf_int

        x = ar1(0.4, 150, 2)
        a = conf_int(
            x, "mean", method=MovingBlock(block_length=8), n_bootstraps=100, random_state=3
        )
        b = conf_int(
            x,
            lambda values, indices: values.mean(axis=0),
            method=MovingBlock(block_length=8),
            n_bootstraps=100,
            random_state=3,
        )
        for u, v in zip(a, b, strict=True):
            np.testing.assert_allclose(u, v, rtol=0, atol=1e-12)

    def test_studentized_deterministic_across_chunk_boundary(self):
        # B=3000 crosses the fixed 2048-chunk boundary; byte-equality proves the
        # composite theta+se reducer inherits the chunking determinism contract.
        from tests._helpers.dgp import ar1
        from tsbootstrap import MovingBlock, conf_int

        x = ar1(0.5, 120, 4)
        kwargs = {
            "method": MovingBlock(block_length=8),
            "kind": "studentized",
            "n_bootstraps": 3000,
            "random_state": 7,
        }
        a = conf_int(x, "mean", **kwargs)
        b = conf_int(x, "mean", **kwargs)
        for u, v in zip(a, b, strict=True):
            np.testing.assert_array_equal(u, v)
        lo, hi, point = a
        assert float(lo) < float(point) < float(hi)

    def test_bca_accepts_iid(self):
        from tsbootstrap import IID, conf_int

        rng = np.random.default_rng(5)
        x = rng.exponential(1.0, size=80)
        lo, hi, point = conf_int(
            x, "mean", method=IID(), kind="bca", n_bootstraps=400, random_state=6
        )
        assert float(lo) < float(point) < float(hi)

    def test_bca_refusal_matrix(self):
        from tsbootstrap import (
            AR,
            CircularBlock,
            MovingBlock,
            NonOverlappingBlock,
            ResidualBootstrap,
            SieveAR,
            StationaryBlock,
            TaperedBlock,
            conf_int,
        )

        rng = np.random.default_rng(8)
        x = rng.standard_normal(100)
        methods = [
            MovingBlock(block_length=5),
            CircularBlock(block_length=5),
            StationaryBlock(avg_block_length=5),
            NonOverlappingBlock(block_length=5),
            TaperedBlock(block_length=5),
            ResidualBootstrap(model=AR(order=1)),
            SieveAR(),
        ]
        for method in methods:
            with pytest.raises(MethodConfigError) as err:
                conf_int(x, "mean", method=method, kind="bca", n_bootstraps=50, random_state=0)
            assert err.value.code == Codes.UNSUPPORTED_MODEL_FEATURE
            assert "studentized" in str(err.value)

    def test_compiled_backend_refuses_studentized_and_bca(self):
        from tsbootstrap import IID, MovingBlock, conf_int

        rng = np.random.default_rng(9)
        x = rng.standard_normal(80)
        for kind, method in (("studentized", MovingBlock(block_length=5)), ("bca", IID())):
            with pytest.raises(MethodConfigError) as err:
                conf_int(
                    x,
                    "mean",
                    method=method,
                    kind=kind,
                    backend="compiled",
                    n_bootstraps=50,
                    random_state=0,
                )
            assert err.value.code == Codes.INVALID_PARAMETER
            assert "compiled" in str(err.value)

    def test_unknown_kind_rejected(self):
        from tsbootstrap import MovingBlock, conf_int

        with pytest.raises(MethodConfigError) as err:
            conf_int(np.arange(50.0), "mean", method=MovingBlock(block_length=5), kind="pivot")
        assert err.value.code == Codes.INVALID_PARAMETER

    def test_panel_shaped_input_fails_loudly(self):
        from tsbootstrap import MovingBlock, conf_int

        rng = np.random.default_rng(10)
        with pytest.raises(MethodConfigError) as err:
            conf_int(
                [rng.standard_normal(50), rng.standard_normal(60)],
                "mean",
                method=MovingBlock(block_length=5),
            )
        assert "conf_int_panel" in str(err.value)
        with pytest.raises(MethodConfigError):
            conf_int(rng.standard_normal((4, 50, 1)), "mean", method=MovingBlock(block_length=5))

    def test_studentized_residual_method_runs(self):
        # Studentized is available for ALL specs, recursive included: the block
        # length falls back to the Politis-White rule on the original series.
        from tests._helpers.dgp import ar1
        from tsbootstrap import AR, ResidualBootstrap, conf_int

        x = ar1(0.6, 150, 11)
        lo, hi, point = conf_int(
            x,
            "mean",
            method=ResidualBootstrap(model=AR(order=1)),
            kind="studentized",
            n_bootstraps=200,
            random_state=12,
        )
        assert float(lo) < float(point) < float(hi)


class TestConfIntPanel:
    def _panel(self):
        rng = np.random.default_rng(20)
        return [rng.standard_normal(80 + 10 * i) for i in range(4)]

    def test_percentile_shapes_and_determinism(self):
        from tsbootstrap import MovingBlock, conf_int_panel

        panel = self._panel()
        a = conf_int_panel(
            panel, "mean", method=MovingBlock(block_length=5), n_bootstraps=100, random_state=1
        )
        b = conf_int_panel(
            panel, "mean", method=MovingBlock(block_length=5), n_bootstraps=100, random_state=1
        )
        for u, v in zip(a, b, strict=True):
            np.testing.assert_array_equal(u, v)
        lo, hi, point = a
        assert lo.shape == hi.shape == point.shape == (4,)
        assert np.all(lo <= point) and np.all(point <= hi)

    def test_flat_plus_indptr_matches_list_form(self):
        from tsbootstrap import MovingBlock, conf_int_panel

        panel = self._panel()
        flat = np.concatenate(panel)
        indptr = np.cumsum([0] + [len(s) for s in panel])
        a = conf_int_panel(
            panel, "mean", method=MovingBlock(block_length=5), n_bootstraps=100, random_state=2
        )
        b = conf_int_panel(
            flat,
            "mean",
            method=MovingBlock(block_length=5),
            indptr=indptr,
            n_bootstraps=100,
            random_state=2,
        )
        for u, v in zip(a, b, strict=True):
            np.testing.assert_array_equal(u, v)

    def test_studentized_requires_explicit_block_length(self):
        from tsbootstrap import MovingBlock, conf_int_panel

        panel = self._panel()
        with pytest.raises(MethodConfigError) as err:
            conf_int_panel(
                panel,
                "mean",
                method=MovingBlock(block_length="auto"),
                kind="studentized",
                n_bootstraps=50,
                random_state=0,
            )
        assert "se_block_length" in str(err.value)
        lo, hi, point = conf_int_panel(
            panel,
            "mean",
            method=MovingBlock(block_length="auto"),
            kind="studentized",
            se_block_length=8,
            n_bootstraps=100,
            random_state=3,
        )
        assert lo.shape == (4,)
        assert np.all(lo < point) and np.all(point < hi)

    def test_bca_rejected_for_panels(self):
        from tsbootstrap import IID, conf_int_panel

        with pytest.raises(MethodConfigError) as err:
            conf_int_panel(self._panel(), "mean", method=IID(), kind="bca")
        assert err.value.code == Codes.INVALID_PARAMETER
