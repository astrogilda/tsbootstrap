"""Behavioral contract goldens that LOCK the current public output (G1, G3, G4, G6).

These pin the literal output, dtypes, no-aliasing guarantee, and VAR tolerance band of
the current engines so future optimization work (e.g. the int32 index change) cannot
silently alter observable behavior. Every reference array here was captured from the
current code; a diff is a behavior change that must be made on purpose.

The companion stream-routing invariants (G2) live in ``test_rng_contract.py`` next to the
rest of the RNG contract; the int32-overflow guard (G5) lands alongside the int32 index change.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._helpers.dgp import ar1
from tsbootstrap import (
    AR,
    ARIMA,
    IID,
    VAR,
    CircularBlock,
    MovingBlock,
    NonOverlappingBlock,
    ResidualBootstrap,
    SieveAR,
    StationaryBlock,
    TaperedBlock,
    bootstrap,
    bootstrap_reduce,
)
from tsbootstrap.errors import MethodConfigError


def _var1(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = np.array([[0.5, 0.1], [0.2, 0.4]])
    x = np.zeros((n, 2))
    for t in range(1, n):
        x[t] = a @ x[t - 1] + rng.standard_normal(2)
    return x


def _arx(n: int, phi: float, beta: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    exog = rng.standard_normal((n, 1))
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + beta * exog[t, 0] + e[t]
    return x, exog


def _varx(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    exog = rng.standard_normal((n, 1))
    a = np.array([[0.4, 0.1], [0.05, 0.3]])
    b = np.array([[1.2], [-1.5]])  # (d, k)
    x = np.zeros((n, 2))
    for t in range(1, n):
        x[t] = a @ x[t - 1] + (b @ exog[t]) + rng.standard_normal(2)
    return x, exog


# --------------------------------------------------------------------------- #
# G1: per-method LITERAL goldens for every observation-resampling family.
# Fixed seed + small fixed input; .values() and .indices() pinned to a captured
# reference. Today only AR has a golden (test_batched_engine.py); this covers the rest.
# --------------------------------------------------------------------------- #
_GOLDEN_X = ar1(0.6, 40, 0)  # the shared 1-D fixture for the block/IID goldens

# Each entry: spec, expected values[0, :5], expected indices[0, :8].
_BLOCK_GOLDENS = {
    "IID": (
        IID(),
        [
            0.7791226231137061,
            -0.34283569625763266,
            0.5175309201585634,
            -0.34283569625763266,
            0.3790854259143617,
        ],
        [19, 36, 23, 36, 8, 35, 13, 12],
    ),
    "MovingBlock": (
        MovingBlock(block_length=5),
        [
            -1.4176918559818088,
            -0.43898457721495243,
            0.7791226231137061,
            0.3389389109241894,
            1.5698268171041996,
        ],
        [17, 18, 19, 20, 21, 33, 34, 35],
    ),
    "CircularBlock": (
        CircularBlock(block_length=5),
        [
            0.7791226231137061,
            0.3389389109241894,
            1.5698268171041996,
            0.2767014167759062,
            0.5175309201585634,
        ],
        [19, 20, 21, 22, 23, 36, 37, 38],
    ),
    "NonOverlappingBlock": (
        NonOverlappingBlock(block_length=5),
        [
            -2.1523230830507454,
            -1.835652832687757,
            -1.4176918559818088,
            -0.43898457721495243,
            0.7791226231137061,
        ],
        [15, 16, 17, 18, 19, 35, 36, 37],
    ),
    "StationaryBlock": (
        StationaryBlock(avg_block_length=5),
        [
            1.8431030087365128,
            0.1257302210933933,
            -0.05666673063526591,
            0.6064226120621226,
            0.46875368439031323,
        ],
        [39, 0, 1, 2, 3, 24, 25, 26],
    ),
    "TaperedBlock": (
        TaperedBlock(block_length=5),
        [
            -0.2202271111201936,
            -0.4199244424581452,
            1.6043275298021384,
            0.2902192949867057,
            -0.2202271111201936,
        ],
        [17, 18, 19, 20, 21, 33, 34, 35],
    ),
}


class TestG1BlockFamilyGoldens:
    @pytest.mark.parametrize("name", list(_BLOCK_GOLDENS))
    def test_values_and_indices_golden(self, name):
        spec, exp_vals, exp_idx = _BLOCK_GOLDENS[name]
        res = bootstrap(_GOLDEN_X, method=spec, n_bootstraps=3, random_state=42)
        np.testing.assert_allclose(res.values()[0, :5], exp_vals, atol=1e-12)
        np.testing.assert_array_equal(res.indices()[0, :8], exp_idx)


class TestG1RecursiveGoldens:
    """Literal .values() goldens for the recursive families (no observation indices)."""

    def test_ar_residual_golden(self):
        v = bootstrap(
            _GOLDEN_X,
            method=ResidualBootstrap(model=AR(order=1)),
            n_bootstraps=3,
            random_state=42,
        ).values()
        np.testing.assert_allclose(
            v[0, :5],
            [
                0.1257302210933933,
                -0.15659536806713267,
                -0.854694195731939,
                -0.34052207734260864,
                -0.9941846290247891,
            ],
            atol=1e-12,
        )

    def test_arima_residual_golden(self):
        y = np.cumsum(ar1(0.5, 41, 1))[:40]  # an integrated series for ARIMA(1,1,1)
        v = bootstrap(
            y,
            method=ResidualBootstrap(model=ARIMA(order=(1, 1, 1))),
            n_bootstraps=3,
            random_state=42,
        ).values()
        np.testing.assert_allclose(
            v[0, :5],
            [
                0.345584192064786,
                1.3399944315983374,
                1.6322306432743918,
                0.9870484879738917,
                1.5632683881209846,
            ],
            atol=1e-12,
        )

    def test_var_residual_golden(self):
        v = bootstrap(
            _var1(40, 0),
            method=ResidualBootstrap(model=VAR(order=1)),
            n_bootstraps=3,
            random_state=42,
        ).values()
        np.testing.assert_allclose(
            v[0, :3],
            [
                [0.0, 0.0],
                [0.8857745892917128, 1.4468454831411872],
                [0.5916210188810951, 1.4351448872853465],
            ],
            atol=1e-12,
        )

    def test_sieve_ar_golden(self):
        v = bootstrap(
            _GOLDEN_X,
            method=SieveAR(min_lag=1, max_lag=3),
            n_bootstraps=3,
            random_state=42,
        ).values()
        np.testing.assert_allclose(
            v[0, :5],
            [
                0.1257302210933933,
                -0.15659536806713267,
                -0.854694195731939,
                -0.34052207734260864,
                -0.9941846290247891,
            ],
            atol=1e-12,
        )


class TestG1ExogGoldens:
    """Literal .values() goldens for the exogenous-regressor ARX/VARX paths."""

    def test_arx_golden(self):
        x, exog = _arx(40, 0.5, 2.0, 0)
        v = bootstrap(
            x,
            method=ResidualBootstrap(model=AR(order=1)),
            n_bootstraps=3,
            random_state=42,
            exog=exog,
        ).values()
        np.testing.assert_allclose(
            v[0, :5],
            [
                -1.2590655321041202,
                -1.5971947009652947,
                -0.6212029906974876,
                -0.7766418885767343,
                -2.3083991888676922,
            ],
            atol=1e-12,
        )

    def test_varx_golden(self):
        x, exog = _varx(40, 0)
        v = bootstrap(
            x,
            method=ResidualBootstrap(model=VAR(order=1)),
            n_bootstraps=3,
            random_state=42,
            exog=exog,
        ).values()
        np.testing.assert_allclose(
            v[0, :3],
            [
                [0.0, 0.0],
                [-0.00015381519504510666, 2.009840764531376],
                [0.7751689164856954, -0.7901138586477876],
            ],
            atol=1e-12,
        )


# --------------------------------------------------------------------------- #
# G3: dtype pins. .values() is float64; .indices() is int32 (the int32 index change flipped
# this from the platform np.intp to a fixed int32, halving the index memory while the
# producer guard, G5, refuses any series too long to address in int32).
# --------------------------------------------------------------------------- #
class TestG3DtypePins:
    def test_values_dtype_is_float64(self):
        res = bootstrap(
            _GOLDEN_X, method=MovingBlock(block_length=5), n_bootstraps=3, random_state=0
        )
        assert res.values().dtype == np.float64

    def test_recursive_values_dtype_is_float64(self):
        res = bootstrap(
            _var1(40, 0),
            method=ResidualBootstrap(model=VAR(order=1)),
            n_bootstraps=3,
            random_state=0,
        )
        assert res.values().dtype == np.float64

    def test_indices_dtype_is_int32(self):
        # The int32 index change pins the index dtype to a fixed int32 (was platform np.intp).
        # The values are bit-identical; only the storage width changed.
        res = bootstrap(
            _GOLDEN_X, method=MovingBlock(block_length=5), n_bootstraps=3, random_state=0
        )
        assert res.indices().dtype == np.dtype(np.int32)

    def test_iid_indices_dtype_is_int32(self):
        res = bootstrap(_GOLDEN_X, method=IID(), n_bootstraps=3, random_state=0)
        assert res.indices().dtype == np.dtype(np.int32)


# --------------------------------------------------------------------------- #
# G5: int32 producer guard. A series at or above 2**31 observations cannot be
# addressed by an int32 index, so the producer must raise loudly (never silently
# wrap to a negative index). Tested via a stub that reports a huge shape, so no
# 2**31-element array is ever allocated.
# --------------------------------------------------------------------------- #
class TestG5ProducerGuard:
    class _HugeArr:
        """A stand-in for the coerced observation array that only reports a shape."""

        def __init__(self, n_obs: int) -> None:
            self.shape = (n_obs, 1)

    def test_producer_raises_at_int32_limit(self, monkeypatch):
        import tsbootstrap.api as api

        # Report n_obs == 2**31 without allocating it; the guard reads only arr.shape.
        monkeypatch.setattr(
            api, "coerce_observations", lambda X: (self._HugeArr(api._MAX_N_OBS), True)
        )
        with pytest.raises(ValueError, match="int32 index limit"):
            api.bootstrap(object(), method=IID(), n_bootstraps=2, random_state=0)

    def test_producer_accepts_at_largest_legal_real_array(self):
        # A real (small) series sits far below the guard, so no ValueError is raised
        # and the produced indices are int32. This pins that the guard does not fire
        # on legal input (the >= 2**31 case is covered above with a stub).
        res = bootstrap(_GOLDEN_X, method=IID(), n_bootstraps=2, random_state=0)
        assert res.indices().dtype == np.dtype(np.int32)


# --------------------------------------------------------------------------- #
# G4: .values() no-aliasing / no-mutation. Mutating the returned array must not
# corrupt the result object or a second .values() call (.values() returns a copy).
# --------------------------------------------------------------------------- #
class TestG4NoAliasing:
    def test_mutating_values_does_not_corrupt_second_call(self):
        res = bootstrap(
            _GOLDEN_X, method=MovingBlock(block_length=5), n_bootstraps=3, random_state=0
        )
        first = res.values()
        snapshot = first.copy()
        first[:] = -999.0  # caller scribbles all over the returned array
        second = res.values()
        np.testing.assert_array_equal(second, snapshot)

    def test_values_does_not_alias_per_sample_buffers(self):
        res = bootstrap(
            _GOLDEN_X, method=MovingBlock(block_length=5), n_bootstraps=3, random_state=0
        )
        stacked = res.values()
        assert not np.shares_memory(stacked, res[0].values)

    def test_mutating_values_does_not_corrupt_per_sample(self):
        res = bootstrap(
            _GOLDEN_X, method=MovingBlock(block_length=5), n_bootstraps=3, random_state=0
        )
        per_sample_before = res[0].values.copy()
        stacked = res.values()
        stacked[0, :] = -1234.5
        np.testing.assert_array_equal(res[0].values, per_sample_before)


# --------------------------------------------------------------------------- #
# G6: VAR tolerance-band value golden captured at the PRODUCTION dispatch._CHUNK_SIZE, so
# chunk-boundary behavior is pinned. VAR's batched matmul is BLAS-shape-sensitive,
# hence a tolerance band rather than a bit-exact literal.
# --------------------------------------------------------------------------- #
class TestG6VARToleranceBandGolden:
    def test_var_golden_at_production_chunk_size(self):
        # No monkeypatch: this captures the value at the real dispatch._CHUNK_SIZE so the
        # production chunk-boundary behavior is what gets pinned.
        v = bootstrap(
            _var1(60, 1),
            method=ResidualBootstrap(model=VAR(order=1)),
            n_bootstraps=5,
            random_state=0,
        ).values()
        np.testing.assert_allclose(
            v[0, 5, :], [0.3437596652667745, 0.18398491148580223], rtol=1e-9, atol=1e-9
        )
        np.testing.assert_allclose(
            v[2, 10, :], [-0.6240955154581712, -0.44369904153463746], rtol=1e-9, atol=1e-9
        )


# --------------------------------------------------------------------------- #
# G7: float32 simulation-dtype context. The values array is cast to the requested
# dtype at the executor boundary, while the model fit and every reduction stay
# float64. References here were captured fresh in float32 (NOT the float64 goldens
# +- eps), so they are compared at a float32 ULP-scaled tolerance, not the 1e-12
# the float64 goldens use. A non-{float64, float32} dtype (e.g. bfloat16) is rejected.
# --------------------------------------------------------------------------- #
_F32_OBS_VALS = [  # MovingBlock(block_length=5), n_bootstraps=3, random_state=42, dtype="float32"
    -1.4176918268203735,
    -0.43898457288742065,
    0.7791226506233215,
    0.3389389216899872,
    1.5698268413543701,
]
_F32_RECURSIVE_VALS = [  # ResidualBootstrap(AR(order=1)), n_bootstraps=3, random_state=42, f32
    0.1257302165031433,
    -0.1565953642129898,
    -0.8546941876411438,
    -0.34052208065986633,
    -0.9941846132278442,
]


class TestG7Float32DtypeContext:
    def test_observation_values_are_float32(self):
        res = bootstrap(
            _GOLDEN_X,
            method=MovingBlock(block_length=5),
            n_bootstraps=3,
            random_state=42,
            dtype="float32",
        )
        assert res.values().dtype == np.dtype(np.float32)
        # The indices stay int32 regardless of the value dtype.
        assert res.indices().dtype == np.dtype(np.int32)
        np.testing.assert_allclose(res.values()[0, :5], _F32_OBS_VALS, rtol=1e-5, atol=1e-4)

    def test_recursive_values_are_float32(self):
        res = bootstrap(
            _GOLDEN_X,
            method=ResidualBootstrap(model=AR(order=1)),
            n_bootstraps=3,
            random_state=42,
            dtype="float32",
        )
        assert res.values().dtype == np.dtype(np.float32)
        np.testing.assert_allclose(res.values()[0, :5], _F32_RECURSIVE_VALS, rtol=1e-5, atol=1e-4)

    def test_default_dtype_is_float64(self):
        res = bootstrap(_GOLDEN_X, method=IID(), n_bootstraps=2, random_state=0)
        assert res.values().dtype == np.dtype(np.float64)
        assert res.metadata.dtype == "float64"

    def test_metadata_records_dtype(self):
        res = bootstrap(
            _GOLDEN_X,
            method=MovingBlock(block_length=5),
            n_bootstraps=2,
            random_state=0,
            dtype="float32",
        )
        assert res.metadata.dtype == "float32"

    def test_reduction_stays_float64_on_float32_path(self):
        # The values handed to the statistic are float32, but the statistic output and the
        # stored statistics array stay float64: only the path tensor is down-cast, the
        # reduction is not.
        seen: dict[str, object] = {}

        def stat(v, idx):
            seen["values_dtype"] = v.dtype
            return float(np.mean(v))

        red = bootstrap_reduce(
            _GOLDEN_X,
            method=MovingBlock(block_length=5),
            statistic=stat,
            n_bootstraps=3,
            random_state=0,
            dtype="float32",
        )
        assert seen["values_dtype"] == np.dtype(np.float32)
        assert red.statistics.dtype == np.dtype(np.float64)

    @pytest.mark.parametrize("bad", ["bfloat16", "bf16", "float16", "int32"])
    def test_unsupported_dtype_rejected(self, bad):
        with pytest.raises(MethodConfigError, match="dtype must be one of"):
            bootstrap(_GOLDEN_X, method=IID(), n_bootstraps=2, random_state=0, dtype=bad)
