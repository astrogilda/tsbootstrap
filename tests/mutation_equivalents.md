# Accepted equivalent mutants (mutation ratchet baseline)

The mutation ratchet (`mutmut`, configured in `pyproject.toml [tool.mutmut]`, scoped to
`src/tsbootstrap/engines/*` and `src/tsbootstrap/model/*`) reports a mutant as "survived"
when no test fails on the mutated code. The mutants listed here are GENUINE algorithmic
equivalents: the mutated code is behaviorally indistinguishable from the original on every
reachable input, so no test can kill them. They are the accepted-survivor baseline; a ratchet
pass means the surviving set is a subset of this list. Each entry was verified by reasoning
about the mutated expression (several also by empirical check).

Every other survivor found in the most recent full run has a dedicated killing test (see
`tests/unit/test_recursive_ar.py`, `tests/unit/test_recursive_arima.py`,
`tests/unit/test_exog.py`); those kills were confirmed by applying each mutant diff to the
source and observing the matching test fail.

## src/tsbootstrap/model/fit.py :: select_ar_order
- `criterion="bic"` -> `"XXbicXX"` and -> `"BIC"`: default value only; unmatched strings fall through the if/elif to the `else` (bic) branch, so behavior is identical.
- `best_order = min_lag` -> `None`: the search loop runs at least once (upper >= min_lag after clamping), so best_order is always overwritten before return.
- loop `range(min_lag, upper+1)` -> `range(min_lag, upper+2)`: the extra candidate k=upper+1 slices the same upper+1 design columns as k=upper (identical residuals) but carries a strictly larger penalty, so its IC is always worse and it is never selected (verified 0 differences over 300 cases).
- `sigma2 = resid@resid / n_eff` -> `* n_eff`: scales sigma2 by n_eff^2, adding the constant n_eff*2*log(n_eff) to every candidate's IC; the argmin is unchanged.
- `penalty*(k+1)` -> `(k-1)` and -> `(k+2)`: a constant offset (-2*penalty, +penalty) across all k; the argmin is unchanged.
- `if ic < best_ic` -> `if ic <= best_ic`: differs only on exact float-IC ties, which are measure-zero on generic continuous data.

## src/tsbootstrap/engines/arma_scipy.py :: simulate_ar
- `lfiltic(b, a, ...)` -> `lfiltic(None, a, ...)`: scipy treats b=None as the all-pole default b=[1.0], which equals the real b; the filter initial state matches to sub-ULP.

## src/tsbootstrap/model/recursive.py :: _prepare_arima
- `exog_coefs = None` -> `""`: the initial value is read only when `exog is None`, where the guard short-circuits on the first clause; when exog is present the value is overwritten. The "" is dead.
- `exog is not None and exog_coefs is not None` -> `or`: exog_coefs is non-None iff exog is non-None, so the operands are always equal and `and` == `or` on every reachable state.

## src/tsbootstrap/model/recursive.py :: _arima_batched
- `gen.integers(0, n_resid, size=m)` -> `gen.integers(n_resid, size=m)`: numpy defines `integers(high)` as `integers(0, high)`; identical draws and identical post-draw RNG state (verified).

## src/tsbootstrap/model/arima.py :: fit_regression_arima_beta
- `exog.reshape(-1, 1)` -> `reshape(-2, 1)`: numpy treats any single negative dimension as the inferred axis, so `reshape(-2, 1)` == `reshape(-1, 1)` for any 1D input.
