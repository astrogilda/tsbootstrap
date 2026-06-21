# Plan — Exogenous covariates for ARIMA (ARIMAX) and VAR (VARX)

Status: **implemented** (VARX commit 192ee77, ARIMAX commit b3df181). AR/VAR/ARIMA all
accept `exog`. This plan documents the design that shipped; it superseded the debate-era
assumption that "VARX needs the VARMAX fitting path" — false for pure VARX (see VARX section).

## 0. What ships today (the pattern to mirror)

ARX is `ResidualBootstrap(model=AR(...))` + `exog`. The exog enters the AR equation
directly and is **held fixed** during regeneration:

```
X*_t = c + sum_j phi_j X*_{t-j} + beta . z_t + e*_t
```

Implementation already in place (mirror it):
- `fit_ar(x, order, exog)` adds the exog columns to the OLS design; `ARFit.exog_coefs`
  carries `beta`.
- `_ar_batched` adds the deterministic `exog[p:p+m] @ beta` contribution to the per-step
  innovations before `lfilter` runs.
- Guard: exog requires `initial="fixed"` and `burn_in=0` so the exog time-alignment holds
  (`_check_exog_compatible`).
- `bootstrap(X, ..., exog=...)` validates exog (length, finiteness) and routes it to the
  preparer; observation-resampling methods reject exog.

## 1. VARX — VAR with exogenous regressors (EASY: still OLS)

**Statistics.** VARX is a VAR with extra regressors, no moving-average term:

```
X_t = c + sum_j A_j X_{t-1-j} + B z_t + e_t        (X in R^d, z in R^k)
```

This is a linear model in `(c, A_1..A_p, B)` — estimated by the same multivariate OLS the
VAR fit already uses, just with `k` extra design columns. **No VARMAX / MLE is required**
(VARMAX is only for VARMA-X, i.e. with MA terms). This is the exact ARX pattern lifted to
the vector case.

**Implementation.**
1. `fit_var(data, order, exog=None)`: append `exog[p:]` columns to the design matrix
   `[1, lag1_block, ..., lagp_block, exog_block]`; `VARFit` gains `exog_coefs: (k, d)`
   (`B^T`). Reuse the existing `_ols` (rank guard included). Verify against statsmodels
   `VARMAX(order=(p,0), exog=...)` coefficients to ~1e-10 on a fixture (one-off check; the
   shipped fit stays pure-numpy OLS).
2. Recursion forcing: in BOTH `_var_recurrence_numpy` and the numba kernel, add the
   deterministic `z_t @ B` term to each step. Pass `exog_contrib = exog[p:p+m] @ exog_coefs`
   (shape `(m, d)`) into the kernel and add `exog_contrib[t-p, i]` inside the inner loop;
   the kernel signature gains one array argument (default zeros when no exog).
3. `_prepare_var(data, model, exog)`: thread exog through; apply the same
   `initial="fixed"`, `burn_in=0` guard as ARX. Store exog + `exog_coefs` on `_VARContext`.
4. Remove the `exog is not None -> raise` branch for `VAR` in `_prepare_residual`.

**Tests.** Recover the exog effect (refit VARX on the samples, `B_hat` centers on the
fitted `B`); determinism; shape; the `initial`/`burn_in` guards; numba/numpy agreement
with exog present.

**Effort.** Small-to-moderate; the only non-trivial piece is adding the exog term to the
two kernels symmetrically. No new dependency.

## 2. ARIMAX — regression with ARIMA errors (MODERATE: keep statsmodels)

**Statistics.** The standard ARIMAX is *regression with ARIMA errors* (what statsmodels
`ARIMA(y, order, exog)` fits), not exog added to the differenced equation:

```
y_t = beta . z_t + eta_t,      eta_t ~ ARIMA(p, d, q)
```

The exog enters the **level**; the ARIMA part models the regression residual `eta_t`. So
the bootstrap factorises cleanly:

1. **Fit** `ARIMA(y, order=(p,d,q), exog=z)` (statsmodels): extract `beta`, the ARMA
   parameters `(phi, theta)`, and the in-sample innovations `e_t`. Compute the regression
   residual series `eta = y - z @ beta` and keep its first `d` levels for inverse-differencing.
2. **Bootstrap a replicate** (reuse the existing ARIMA machinery):
   - resample centered innovations `e*`;
   - simulate the ARMA process on the differenced scale: `w* = ARMA_sim(phi, theta, e*)`;
   - inverse-difference `w*` using `eta`'s initial `d` levels -> `eta*`;
   - add the held-fixed exog level back: `y*_t = eta*_t + beta . z_t`.

This is the existing `_arima_batched` path plus (a) subtracting `z @ beta` before fitting
and (b) adding it back after inverse-differencing. The exog is exogenous, held fixed.

**Implementation.**
1. `model/arima.py`: `fit_arima(y, order, exog=None)` -> extend `ARMAFit`/context with
   `exog_coefs` (`beta`) and store `eta`'s initial levels (already stored for the
   no-exog inverse-difference; compute them on `eta` instead of `y` when exog is present).
   Extraction from statsmodels `ARIMAResults`: `beta = params[exog slots]`, `resid` =
   innovations, `arparams`/`maparams` for the ARMA part.
2. `_arima_batched`: after inverse-differencing to `eta*`, add `exog @ beta` (the
   `(m, ...)` fixed contribution) to produce `y*`. Single deterministic add, like ARX.
3. `_prepare_arima(series, model, exog)`: thread exog; compute `eta = y - z@beta`
   (requires fitting `beta` first — let statsmodels do it jointly), difference `eta`, fit
   ARMA on it. Same `initial`/`burn_in` exog guard.
4. Remove the `exog is not None -> raise` branch for `ARIMA` in `_prepare_residual`.

**Subtlety to verify.** statsmodels reports innovations on the (differenced, whitened)
scale; confirm the simulate -> inverse-difference -> add-exog round-trip reproduces the
fitted `y` when `e* = e` (a falsification test: zero-noise reconstruction equals the
fitted conditional mean). Keep statsmodels for this path (genuine MLE/Kalman); do not
attempt an OLS shortcut (ARMA has MA terms).

**Tests.** Exog-effect recovery; zero-innovation reconstruction; determinism; the `d`-level
inverse-difference correctness with exog; guards.

**Effort.** Moderate; the statsmodels extraction + the level round-trip are the careful parts.

## 3. Shared API / validation

- `bootstrap(..., exog=...)` already coerces and length-checks exog and restricts it to
  model-based methods; ARIMAX/VARX need no entry-point change beyond removing the two
  per-model `raise`s.
- The exog `initial="fixed"`, `burn_in=0` constraint is uniform across ARX/ARIMAX/VARX.
- `bootstrap_reduce` composes unchanged (exog only affects path generation).

## 4. Sequence and gating

1. VARX first (pure OLS, mirrors ARX, no statsmodels) — lower risk, ships the vector case.
2. ARIMAX second (statsmodels extraction + level round-trip) — needs the reconstruction
   falsification test green before landing.
3. Each lands as its own commit with exog-effect-recovery + determinism + guard tests, and
   the full statistical gate must stay green.

## 5. Scope decision and future extension

v1 supports **static contemporaneous exog** (`beta . z_t`), matching ARX and statsmodels
`ARIMA(exog=)`. This is the standard, most-requested formulation.

**Future extension (tracked in TODO "Future capabilities"):** distributed-lag / transfer-
function dynamic regression (`beta(L) . z_t = sum_l beta_l z_{t-l}`), where a covariate
affects the series over several lags. It is a strict superset of static exog (a degenerate
0-lag transfer function), so the static path here is forward-compatible: the design matrix
would gain lagged exog columns and the recursion forcing would sum over the lag window. Not
scheduled; build when a delayed-covariate use case appears.
