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
    upper = max(min_lag, min(upper, n // 2 - 1))
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


def test_exact_near_tie_is_decided_like_the_per_order_solve():
    """Bisect a mixing weight until two candidate orders tie to rounding, then compare.

    x(w) = base + w * bump moves the AIC gap between orders k and k+1 continuously, so a
    bisection on w finds a series where the two criteria agree to rounding. The QR and SVD
    residual sums differ by a few ulps there, which is the case the tie band settles by
    re-scoring with the per-order solve.
    """
    base = _ar(300, 11, [0.5, 0.25])
    bump = np.sin(np.arange(300) * 0.7)
    k = 2

    def gap(w):
        _o, ics = _reference(base + w * bump, max_lag=6, criterion="aic", return_ics=True)
        return ics[k + 1] - ics[k]

    lo, hi = 0.0, 5.0
    if np.sign(gap(lo)) == np.sign(gap(hi)):
        pytest.fail(f"no sign change on [0, 5]: gaps {gap(lo)} and {gap(hi)}")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if np.sign(gap(mid)) == np.sign(gap(lo)):
            lo = mid
        else:
            hi = mid
    for w in (lo, hi):
        x = base + w * bump
        assert abs(gap(w)) < 1e-9
        assert select_ar_order(x, max_lag=6, criterion="aic") == _reference(
            x, max_lag=6, criterion="aic"
        )


def test_wide_tie_band_still_matches(monkeypatch):
    """With every candidate inside the band, the decision is the per-order solve's."""
    monkeypatch.setattr(fit_mod, "_IC_TIE_RTOL", 1.0)
    for seed in range(5):
        x = _ar(400, seed, [0.4, 0.2])
        assert select_ar_order(x, max_lag=10, criterion="aic") == _reference(
            x, max_lag=10, criterion="aic"
        )


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
