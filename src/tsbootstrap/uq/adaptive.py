"""Adaptive and nonexchangeable conformal calibration for distribution shift.

Base EnbPI (and any fixed-calibration conformal method) assumes the calibration
residuals are representative of the test residuals. Under distribution shift or
volatility clustering that fails, and intervals silently under- or over-cover. These
two methods adapt the calibration to recent behaviour and compose on the per-replicate
nonconformity scores produced by the bootstrap (e.g. via :func:`bootstrap_reduce`):

- :func:`aci_halfwidths`, Adaptive Conformal Inference (Gibbs & Candès 2021): adapt the
  quantile *level* online from realized coverage errors, so long-run coverage tracks the
  target even when the score distribution drifts.
- :func:`nexcp_quantile`, Nonexchangeable Conformal Prediction (Barber et al. 2023): a
  recency-weighted quantile of the scores, so recent residuals dominate the interval.
- :func:`agaci_bounds`, Aggregated Adaptive Conformal Inference (Zaffran et al. 2022):
  run a grid of ACI experts (one per step size) and aggregate their lower/upper interval
  endpoints online with Bernstein Online Aggregation (Wintenberger 2017), so the target
  step size need not be chosen. It carries a regret/efficiency guarantee but no coverage
  certificate (see the function docstring).

Coverage is approximate / long-run under temporal dependence, not finite-sample
distribution-free, consistent with the rest of the UQ layer.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

# The K=30 step-size grid Zaffran et al. (2022) use for AgACI, transcribed verbatim from
# their reference implementation (AgACI/Script/acp_gamma.R:14-19). gamma=0 is the static,
# never-adapting conformal anchor expert; the grid is denser at small gamma (log-ish
# spacing 5e-6 .. 9e-2) because ACI is most sensitive there. An immutable tuple is a safe
# keyword default.
DEFAULT_AGACI_GAMMAS: tuple[float, ...] = (
    0.0,
    5e-6,
    5e-5,
    1e-4,
    2e-4,
    3e-4,
    4e-4,
    5e-4,
    6e-4,
    7e-4,
    8e-4,
    9e-4,
    1e-3,
    2e-3,
    3e-3,
    4e-3,
    5e-3,
    6e-3,
    7e-3,
    8e-3,
    9e-3,
    1e-2,
    2e-2,
    3e-2,
    4e-2,
    5e-2,
    6e-2,
    7e-2,
    8e-2,
    9e-2,
)


class AgACIBounds(NamedTuple):
    """The two load-bearing asymmetric half-widths AgACI produces per step.

    Unlike :func:`aci_halfwidths`, whose second return element is a discardable
    diagnostic, BOTH fields here are load-bearing: the interval at step ``t`` is
    ``[prediction_t - lower[t], prediction_t + upper[t]]``. Destructuring away either
    field (``hw, _ = agaci_bounds(...)``) is a bug, not an idiom.
    """

    lower: NDArray[np.float64]
    upper: NDArray[np.float64]


def aci_halfwidths(
    calibration_scores: object,
    test_scores: object,
    *,
    alpha: float = 0.1,
    gamma: float = 0.05,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Adaptive Conformal Inference: online-adapted interval half-widths.

    Parameters
    ----------
    calibration_scores : array-like, shape (m,)
        Nonconformity scores (e.g. ``|residual|``) from the bootstrap calibration set.
    test_scores : array-like, shape (T,)
        Realized scores ``|y_t - prediction_t|`` over the test sequence, in time order.
    alpha : float
        Target miscoverage (interval target coverage is ``1 - alpha``).
    gamma : float
        Adaptation step size. ``gamma = 0`` recovers static conformal.

    Returns
    -------
    halfwidths : ndarray, shape (T,)
        Interval half-width ``q_t`` to use at each step (``prediction_t ± q_t``).
    alphas : ndarray, shape (T,)
        The adapted miscoverage level used at each step.

    The update is ``alpha_{t+1} = alpha_t + gamma * (alpha - err_t)`` with
    ``err_t = 1`` when step ``t`` is miscovered: a miss shrinks the level and widens the
    next interval. Coverage converges to ``1 - alpha`` regardless of how the scores drift.
    """
    cal = np.asarray(calibration_scores, dtype=np.float64).ravel()  # ravel yields contiguous 1-D
    test = np.asarray(test_scores, dtype=np.float64).ravel()
    if cal.size == 0:
        raise ValueError("calibration_scores must be non-empty")

    a_t = float(alpha)
    halfwidths = np.empty(test.shape[0], dtype=np.float64)
    alphas = np.empty(test.shape[0], dtype=np.float64)
    # Sort the calibration scores once; each per-step quantile is then a constant-time
    # linear interpolation over the sorted array. This reproduces numpy's
    # ``np.quantile(cal, p, method="linear")`` bit-for-bit, including its two-branch
    # lerp (``a + (b - a) * t`` for ``t < 0.5``; ``b - (b - a) * (1 - t)`` otherwise).
    cs = np.sort(cal)
    last = cs.shape[0] - 1
    cs_last = float(cs[last])
    for t in range(test.shape[0]):
        a_clip = min(max(a_t, 0.0), 1.0)
        if a_clip <= 0.0:
            q = np.inf  # cover everything
        elif a_clip >= 1.0:
            q = 0.0
        else:
            h = (1.0 - a_clip) * last
            lo = int(np.floor(h))
            if lo >= last:
                q = cs_last
            else:
                frac = h - lo
                a = float(cs[lo])
                b = float(cs[lo + 1])
                diff = b - a
                q = b - diff * (1.0 - frac) if frac >= 0.5 else a + diff * frac
        halfwidths[t] = q
        alphas[t] = a_clip
        err = 1.0 if test[t] > q else 0.0
        a_t = a_t + gamma * (alpha - err)
    return halfwidths, alphas


def nexcp_quantile(scores: object, *, alpha: float = 0.1, decay: float = 0.99) -> float:
    """Recency-weighted (nonexchangeable) conformal quantile of the scores.

    Weights score ``i`` (0 = oldest, last = most recent) by ``decay ** (n - 1 - i)`` and
    returns the smallest score whose normalized weighted CDF reaches ``1 - alpha``. With
    ``decay = 1`` this is the ordinary empirical quantile; smaller ``decay`` puts more
    weight on recent residuals, widening the interval when recent volatility rises.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()  # ravel yields a contiguous 1-D
    n = s.shape[0]
    if n == 0:
        raise ValueError("scores must be non-empty")
    if not 0.0 < decay <= 1.0:
        raise ValueError("decay must be in (0, 1]")
    weights = decay ** np.arange(n - 1, -1, -1, dtype=np.float64)
    order = np.argsort(s, kind="stable")
    s_sorted = s[order]
    cdf = np.cumsum(weights[order])
    # Compare the unnormalized weighted CDF against (1 - alpha) * total rather than
    # normalizing the weights first: the searchsorted boundary is the same and one pass
    # over the array is saved.
    target = (1.0 - alpha) * cdf[n - 1]
    idx = int(np.searchsorted(cdf, target, side="left"))
    return float(s_sorted[min(idx, n - 1)])


def _boa_aggregate(
    experts: NDArray[np.float64],
    targets: NDArray[np.float64],
    *,
    tau: float,
    regret_constant: float = 2.2,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Bernstein Online Aggregation of expert endpoints under linearized pinball loss.

    A pure-numpy transliteration of ``opera``'s ``BOA.R`` (Wintenberger 2017), the
    aggregator Zaffran et al. (2022) invoke via ``model='BOA', loss.gradient=TRUE``. It
    runs one online expert-aggregation problem: at each step it forms a convex
    combination of the ``K`` expert endpoints, predicts, then updates each expert's
    cumulative regularized regret and per-expert adaptive learning rate.

    Parameters
    ----------
    experts : ndarray, shape (T, K)
        Expert endpoint offsets over the test stream, one column per expert. Must be
        finite (infinite ACI endpoints are clipped to a sentinel by the caller).
    targets : ndarray, shape (T,)
        Realized signed residuals ``s_t = y_t - prediction_t`` in time order. The pinball
        subgradient ``1{s_t < aggregated_offset} - tau`` needs the sign of each miss.
    tau : float
        Pinball quantile level (``alpha / 2`` for the lower bound, ``1 - alpha / 2`` for
        the upper bound).
    regret_constant : float
        The fixed Bernstein constant in ``eta_inv2 += regret_constant * r ** 2`` (2.2 in
        ``opera``).

    Returns
    -------
    prediction : ndarray, shape (T,)
        The aggregated endpoint offset at each step.
    weights : ndarray, shape (T, K)
        The convex expert weights used at each step.

    Notes
    -----
    State per expert: cumulative regularized regret ``R_reg`` (init 0), inverse-squared
    learning-rate accumulator ``eta_inv2`` (init 0), fixed prior ``w0 = 1``. Round 1 has
    ``eta_inv2`` all zero, so the weights fall back to the uniform prior (the first
    aggregated endpoint is the mean of the experts). The ``-log(eta_inv2) / 2`` term makes
    the weight proportional to the per-expert adaptive rate ``1 / sqrt(eta_inv2)``; the
    softmax is stabilized by subtracting ``max(Raux)``. Zero-regret experts (``eta_inv2``
    stays 0) are masked out of the softmax and retain their prior mass via the
    ``w0[active].sum()`` scaling, matching ``opera``'s awake/latch mechanism. O(T*K) time,
    O(K) state.
    """
    E = np.asarray(experts, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64).ravel()
    T, K = E.shape
    w0 = np.ones(K, dtype=np.float64)
    R_reg = np.zeros(K, dtype=np.float64)
    eta_inv2 = np.zeros(K, dtype=np.float64)
    prediction = np.empty(T, dtype=np.float64)
    weights = np.empty((T, K), dtype=np.float64)
    for t in range(T):
        nz = eta_inv2 > 0.0
        w = w0.copy()
        if nz.any():
            # Stabilized softmax over the active (nonzero-regret) experts, scaled by the
            # active prior mass; zero-regret experts keep their prior weight (w already
            # equals w0 there).
            raux = -0.5 * np.log(eta_inv2[nz]) + np.log(w0[nz]) + R_reg[nz] / np.sqrt(eta_inv2[nz])
            ex = np.exp(raux - raux.max())
            w[nz] = w0[nz].sum() * ex / ex.sum()
        p = w / w.sum()
        pred_off = float(E[t] @ p)
        weights[t] = p
        prediction[t] = pred_off
        grad = (1.0 if y[t] < pred_off else 0.0) - tau
        r = grad * (pred_off - E[t])
        eta_inv2 = eta_inv2 + regret_constant * r * r
        safe = eta_inv2 > 0.0
        r_reg = np.zeros(K, dtype=np.float64)
        r_reg[safe] = r[safe] - r[safe] ** 2 / np.sqrt(eta_inv2[safe])
        R_reg = R_reg + r_reg
    return prediction, weights


def agaci_bounds(
    calibration_scores: object,
    test_residuals: object,
    *,
    alpha: float = 0.1,
    gammas: object | None = DEFAULT_AGACI_GAMMAS,
    boa_regret_constant: float = 2.2,
    infinite_sentinel: float | None = None,
    require_signed: bool = True,
) -> AgACIBounds:
    """Aggregated Adaptive Conformal Inference (AgACI): asymmetric adaptive half-widths.

    Run a grid of ACI experts (one :func:`aci_halfwidths` pass per step size ``gamma``)
    and aggregate their lower and upper interval endpoints with two independent Bernstein
    Online Aggregations under the pinball loss at ``tau = alpha / 2`` and
    ``tau = 1 - alpha / 2``. The online weights track whichever step size is currently
    best in pinball loss, so no single ``gamma`` has to be chosen and, unlike a large-
    ``gamma`` ACI expert, the aggregated interval is always finite. Section 3 of Zaffran
    et al. (2022) constructs AgACI exactly this way.

    Parameters
    ----------
    calibration_scores : array-like, shape (m,)
        The absolute out-of-bag residual buffer (e.g.
        :attr:`EnbPIEnsemble.oob_residuals`). Same role and coercion as
        :func:`aci_halfwidths`' ``calibration_scores``.
    test_residuals : array-like, shape (T,)
        The SIGNED realized residuals ``s_t = y_t - prediction_t``, time-ordered. This
        deliberately diverges from :func:`aci_halfwidths` (which takes absolute scores):
        AgACI's two-sided pinball gradient needs the sign of each miss to load it onto the
        lower versus the upper bound. Must be finite; ``T`` is driven entirely by its
        length. The per-expert ACI pass is driven by ``abs(test_residuals)`` internally.
    alpha : float
        Target miscoverage. Split into ``tau_lower = alpha / 2`` and
        ``tau_upper = 1 - alpha / 2``. Must be in ``(0, 1)``.
    gammas : array-like or None
        The ACI step-size grid; each entry is one ACI expert (``K = len(gammas)``). All
        entries must be finite and non-negative. ``None`` selects
        :data:`DEFAULT_AGACI_GAMMAS` (the K=30 grid of Zaffran et al.).
    boa_regret_constant : float
        ``opera``'s fixed Bernstein constant in the BOA learning-rate accumulation. Must
        be positive; exposed for faithfulness and pinning.
    infinite_sentinel : float or None
        Finite clip for ``+inf`` expert half-widths (emitted when a large-``gamma`` expert
        drives its level below 0). ``None`` selects a deterministic data-adaptive default
        ``min(max(1.0, 10.0 * range_ref), 1e6)`` where ``range_ref`` is the max of the
        finite expert half-widths, ``max(abs(test_residuals))``, and ``1.0``. The floor
        stops degenerate all-zero data collapsing the clip to a no-op; the cap stops the
        BOA learning-rate accumulator overflowing on large-magnitude data. This default is
        deliberately not bit-comparable to ``opera``'s fixed +/-1000.
    require_signed : bool
        When ``True`` (default), ``test_residuals`` with zero strictly-negative entries on
        a stream of length >= 8 raises :class:`ValueError`: an all-non-negative stream is
        the near-certain signature of a caller passing absolute scores by ACI habit, which
        makes the lower-bound indicator constant and biases that bound with no error. Set
        ``False`` for genuinely one-sided residual data.

    Returns
    -------
    AgACIBounds
        Named tuple ``(lower, upper)``, each shape ``(T,)``, float64, non-negative, and
        finite. The interval at step ``t`` is
        ``[prediction_t - lower[t], prediction_t + upper[t]]``. BOTH fields are
        load-bearing, unlike :func:`aci_halfwidths`' diagnostic second element.

    Notes
    -----
    Coverage: AgACI carries a regret/efficiency guarantee (via BOA it is asymptotically no
    worse in pinball loss than the best fixed-``gamma`` expert in the grid, with
    ``O(sqrt(T log K))`` regret) but NO finite-T or asymptotic coverage certificate:
    aggregating the two endpoints breaks the bounded-level-excursion argument that gives
    single ACI its long-run coverage guarantee. Empirically it keeps marginal coverage
    close to ``1 - alpha`` while producing shorter, always-finite intervals. Do not read
    this as a guarantee of ``1 - alpha`` coverage.

    Interval crossing is structurally impossible: every lower expert offset ``-q_k <= 0``
    and every upper offset ``+q_k >= 0``, and a convex BOA combination preserves the sign,
    so ``lower`` and ``upper`` are both non-negative and the interval never crosses.

    A deliberate fidelity divergence from the R source: when ``alpha_t >= 1`` an ACI
    expert returns ``q = 0`` (a finite confident expert) where the R source emits the
    absolute empty interval ``(0, 0)``; here that feeds BOA as a finite expert, not the
    ``+inf`` sentinel path. Non-bit-comparable to ``opera`` in that rare regime.

    References
    ----------
    Zaffran, M., Feron, O., Goude, Y., Josse, J., and Dieuleveut, A. (2022). Adaptive
    Conformal Predictions for Time Series. Proceedings of the 39th International Conference
    on Machine Learning (ICML), PMLR 162, pp. 25834-25866.

    Wintenberger, O. (2017). Optimal learning with Bernstein Online Aggregation. Machine
    Learning 106(1), pp. 119-141.

    Gibbs, I. and Candes, E. (2021). Adaptive Conformal Inference Under Distribution Shift.
    Advances in Neural Information Processing Systems 34.
    """
    cal = np.asarray(calibration_scores, dtype=np.float64).ravel()  # ravel yields contiguous 1-D
    s = np.asarray(test_residuals, dtype=np.float64).ravel()
    grid_src = DEFAULT_AGACI_GAMMAS if gammas is None else gammas
    grid = np.asarray(grid_src, dtype=np.float64).ravel()

    if cal.size == 0:
        raise ValueError("calibration_scores must be non-empty")
    if s.size == 0:
        raise ValueError("test_residuals must be non-empty")
    if not np.all(np.isfinite(s)):
        raise ValueError(
            "test_residuals must be finite; a non-finite target silently corrupts the BOA "
            "aggregation state for every subsequent step"
        )
    if grid.size == 0:
        raise ValueError("gammas must be non-empty")
    if not np.all(np.isfinite(grid)):
        raise ValueError("gammas must all be finite")
    if not np.all(grid >= 0.0):
        raise ValueError("gammas must all be non-negative")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if boa_regret_constant <= 0.0:
        raise ValueError("boa_regret_constant must be positive")
    if require_signed and s.size >= 8 and int((s < 0.0).sum()) == 0:
        raise ValueError(
            "test_residuals must be SIGNED realized residuals (y_t - prediction_t); an "
            "all-non-negative stream of length >= 8 is the signature of absolute scores "
            "passed by ACI habit, which biases the lower bound. Pass require_signed=False "
            "for genuinely one-sided residual data."
        )

    test_abs = np.abs(s)
    # One ACI pass per gamma builds the (T, K) expert half-width matrix, columns in grid
    # order. The sorted-calibration per-step quantile and the level recursion come straight
    # from aci_halfwidths; they are not reimplemented here.
    Q = np.empty((s.shape[0], grid.shape[0]), dtype=np.float64)
    for k in range(grid.shape[0]):
        q_k, _ = aci_halfwidths(cal, test_abs, alpha=alpha, gamma=float(grid[k]))
        Q[:, k] = q_k

    # Clip +inf experts (only +inf is possible: half-widths are non-negative) to a finite,
    # deterministic, floored-and-capped sentinel so the pinball loss stays finite for BOA.
    finite_Q = Q[np.isfinite(Q)]
    range_ref = max(
        float(finite_Q.max()) if finite_Q.size else 0.0,
        float(test_abs.max()),
        1.0,
    )
    sentinel = (
        float(infinite_sentinel)
        if infinite_sentinel is not None
        else min(max(1.0, 10.0 * range_ref), 1e6)
    )
    Q = np.where(np.isfinite(Q), Q, sentinel)

    # Two independent BOAs on the endpoint offsets relative to the common point
    # prediction: lower offsets -q_k <= 0, upper offsets +q_k >= 0. The point prediction
    # cancels in the regret and inside 1{y < point + off} = 1{s < off}, so aggregating the
    # offsets is exactly equivalent to aggregating the y-space endpoints.
    low_off, _ = _boa_aggregate(-Q, s, tau=alpha / 2.0, regret_constant=boa_regret_constant)
    high_off, _ = _boa_aggregate(Q, s, tau=1.0 - alpha / 2.0, regret_constant=boa_regret_constant)

    return AgACIBounds(lower=-low_off, upper=high_off)


__all__ = [
    "aci_halfwidths",
    "nexcp_quantile",
    "agaci_bounds",
    "AgACIBounds",
    "DEFAULT_AGACI_GAMMAS",
]
