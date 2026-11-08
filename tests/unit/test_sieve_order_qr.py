"""SieveAR order selection by one QR: the same order as one SVD solve per candidate."""

from __future__ import annotations

import numpy as np
import pytest

from tsbootstrap.errors import InputDataError
from tsbootstrap.model import fit as fit_mod
from tsbootstrap.model.fit import select_ar_order


def _reference(x, *, min_lag=1, max_lag=None, criterion="bic", return_ics=False):
    """The per-order SVD selection this module used before the QR path."""
    series = np.ascontiguousarray(np.asarray(x, dtype=np.float64).ravel())
    n = series.shape[0]
    upper = max_lag if max_lag is not None else int(np.ceil(10 * np.log10(n)))
    upper = min(upper, (n - 2) // 3)
    target = series[upper:]
    n_eff = target.shape[0]
    penalty = {"aic": 2.0, "hqic": 2.0 * float(np.log(np.log(n_eff)))}.get(
        criterion, float(np.log(n_eff))
    )
    full = np.column_stack(
        [np.ones(n_eff), *(series[upper - j : n - j] for j in range(1, upper + 1))]
    )
    best_ic, best_order, ics = np.inf, min_lag, {}
    for k in range(min_lag, upper + 1):
        design = full[:, : k + 1]
        beta, _r, rank, _s = np.linalg.lstsq(design, target, rcond=None)
        if rank < design.shape[1]:
            raise InputDataError("rank-deficient")
        resid = target - design @ beta
        ic = n_eff * float(np.log(float(resid @ resid) / n_eff)) + penalty * (k + 1)
        ics[k] = ic
        if ic < best_ic:
            best_ic, best_order = ic, k
    return (best_order, ics) if return_ics else best_order


def _ar(n, seed, phis, scale=1.0, offset=0.0):
    rng = np.random.default_rng(seed)
    p = len(phis)
    x = np.zeros(n + 200)
    e = rng.standard_normal(n + 200)
    for t in range(p, n + 200):
        x[t] = sum(phis[j] * x[t - 1 - j] for j in range(p)) + e[t]
    return offset + scale * x[200:]


@pytest.mark.parametrize("criterion", ["aic", "bic", "hqic"])
@pytest.mark.parametrize("seed", range(40))
def test_same_order_as_per_order_svd_on_random_ar(seed, criterion):
    rng = np.random.default_rng(1000 + seed)
    p = int(rng.integers(1, 5))
    phis = list(rng.uniform(-0.4, 0.4, size=p))
    n = int(rng.integers(60, 600))
    x = _ar(n, seed, phis)
    max_lag = None if seed % 2 else int(rng.integers(2, 15))
    assert select_ar_order(x, max_lag=max_lag, criterion=criterion) == _reference(
        x, max_lag=max_lag, criterion=criterion
    )


@pytest.mark.parametrize(
    ("phis", "scale", "offset"),
    [
        ([0.999], 1.0, 0.0),  # near unit root: lag columns nearly collinear
        ([1.6, -0.6005], 1.0, 0.0),  # near-unit-root AR(2)
        ([0.5], 1e-3, 1e4),  # large level, small variation: intercept nearly collinear
        ([0.5], 1e-6, 1e6),  # rank-deficient to lstsq: both paths raise the same error
        ([0.3, 0.2], 1e8, 0.0),  # large scale
    ],
)
@pytest.mark.parametrize("criterion", ["aic", "bic", "hqic"])
def test_same_order_on_ill_conditioned_designs(phis, scale, offset, criterion):
    x = _ar(800, 7, phis, scale=scale, offset=offset)

    def outcome(fn):
        try:
            return fn(x, max_lag=12, criterion=criterion)
        except InputDataError as exc:
            return type(exc)

    assert outcome(select_ar_order) == outcome(_reference)


def test_constant_series_raises_the_rank_error():
    with pytest.raises(InputDataError):
        select_ar_order(np.full(100, 3.0), max_lag=5)


def _near_tie_series():
    """Bisect a mixing weight until the AIC of orders 2 and 3 agree to rounding.

    x(w) = base + w * bump moves the gap between the two criteria continuously, so a
    bisection on w finds a series where a last-bit change in any residual sum could flip
    the decision. That is the series a platform-dependent QR or log would decide two ways.
    """
    base = _ar(300, 11, [0.5, 0.25])
    bump = np.sin(np.arange(300) * 0.7)

    def gap(w):
        _o, ics = _reference(base + w * bump, max_lag=6, criterion="aic", return_ics=True)
        return ics[3] - ics[2]

    lo, hi = 0.0, 5.0
    assert np.sign(gap(lo)) != np.sign(gap(hi))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if np.sign(gap(mid)) == np.sign(gap(lo)):
            lo = mid
        else:
            hi = mid
    assert abs(gap(lo)) < 1e-9
    return base + lo * bump


_PROBE = """
import sys, hashlib, numpy as np
from tsbootstrap.model import fit
x = np.frombuffer(bytes.fromhex(sys.argv[1]), dtype=np.float64)
upper = 6
target = x[upper:]
lags = np.lib.stride_tricks.sliding_window_view(x, upper)[: x.size - upper, ::-1]
design = np.column_stack([np.ones(target.size), lags])
r, z = fit._householder_rotate(design, target)
digest = hashlib.sha256(r.tobytes() + z.tobytes()).hexdigest()
print(fit.select_ar_order(x, max_lag=upper, criterion="aic"), digest)
"""


@pytest.mark.parametrize(
    "env",
    [
        {"OPENBLAS_CORETYPE": "Haswell"},
        {"OPENBLAS_CORETYPE": "Prescott"},
        {"OPENBLAS_CORETYPE": "SandyBridge"},
        {"NPY_DISABLE_CPU_FEATURES": "AVX512F AVX512_SKX AVX2 FMA3"},
    ],
    ids=["haswell", "prescott", "sandybridge", "no-avx"],
)
def test_near_tie_is_decided_the_same_under_every_cpu_kernel(env):
    """Same order and byte-equal R and Q.T y whatever BLAS kernel or SIMD level runs."""
    import os
    import subprocess
    import sys

    x = _near_tie_series()
    arg = x.tobytes().hex()

    def run(extra):
        out = subprocess.run(
            [sys.executable, "-c", _PROBE, arg],
            env={**os.environ, **extra},
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.split()

    assert run(env) == run({})


def test_qr_factor_matches_lapack_to_rounding():
    rng = np.random.default_rng(5)
    design = np.column_stack([np.ones(400), rng.standard_normal((400, 9))])
    target = rng.standard_normal(400)
    r, z = fit_mod._householder_rotate(design, target)
    q_ref, r_ref = np.linalg.qr(design)
    np.testing.assert_allclose(np.abs(r), np.abs(r_ref), atol=1e-12)
    rss_ref = float(np.sum((target - q_ref @ (q_ref.T @ target)) ** 2))
    assert float(np.sum(z[10:] ** 2)) == pytest.approx(rss_ref, rel=1e-12)


def test_default_upper_is_integer_ceil_of_ten_log10():
    for n in (2, 9, 10, 11, 99, 100, 101, 1000, 1001, 4641, 4642, 10**5):
        assert fit_mod._default_upper(n) == int(np.ceil(10 * np.log10(n)))


def test_well_conditioned_design_takes_the_qr_path(monkeypatch):
    calls = []
    real = fit_mod._ic_by_ols

    def spy(*args, **kwargs):
        calls.append(args[2])
        return real(*args, **kwargs)

    monkeypatch.setattr(fit_mod, "_ic_by_ols", spy)
    x = _ar(2000, 3, [0.5])
    select_ar_order(x, max_lag=30, criterion="bic")
    assert calls == []


def test_order_cap_keeps_white_noise_fits_stationary():
    """The cap n - upper >= 2 (upper + 1) stops saturated fits on short white noise."""
    from tsbootstrap.model.fit import fit_ar

    for n in (20, 30, 40):
        bad = 0
        for seed in range(200):
            x = np.random.default_rng(seed).standard_normal(n)
            p = select_ar_order(x)
            phi = fit_ar(x, p).ar_coefs
            roots = np.abs(np.roots(np.r_[1.0, -phi])) if p else np.zeros(1)
            bad += bool(roots.size and roots.max() >= 1.0)
        assert bad / 200 <= 0.08, (n, bad)


def test_min_lag_beyond_the_cap_raises_order_too_large():
    from tsbootstrap.errors import Codes, MethodConfigError

    x = np.random.default_rng(0).standard_normal(40)
    with pytest.raises(MethodConfigError) as info:
        select_ar_order(x, min_lag=19)
    assert info.value.code == Codes.ORDER_TOO_LARGE


@pytest.mark.parametrize("criterion", ["AIC", "bic ", "aicc", ""])
def test_unknown_criterion_raises(criterion):
    from tsbootstrap.errors import Codes, MethodConfigError

    x = np.random.default_rng(1).standard_normal(200)
    with pytest.raises(MethodConfigError) as info:
        select_ar_order(x, criterion=criterion)
    assert info.value.code == Codes.INVALID_PARAMETER
