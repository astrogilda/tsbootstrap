# Accepted equivalent mutants (mutation ratchet baseline)

The mutation ratchet (`mutmut`, configured in `pyproject.toml [tool.mutmut]`, scoped to
`src/tsbootstrap/engines/*`, `src/tsbootstrap/model/*`, `src/tsbootstrap/uq/adaptive.py`, and
`src/tsbootstrap/prng_keys.py`) reports a mutant as "survived"
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

## src/tsbootstrap/model/recursive.py :: _prepare_arima
- `exog_coefs = None` -> `""`: the initial value is read only when `exog is None`, where the guard short-circuits on the first clause; when exog is present the value is overwritten. The "" is dead.
- `exog is not None and exog_coefs is not None` -> `or`: exog_coefs is non-None iff exog is non-None, so the operands are always equal and `and` == `or` on every reachable state.

## src/tsbootstrap/model/recursive.py :: _arima_batched
- `gen.integers(0, n_resid, size=m)` -> `gen.integers(n_resid, size=m)`: numpy defines `integers(high)` as `integers(0, high)`; identical draws and identical post-draw RNG state (verified).

## src/tsbootstrap/model/arima.py :: fit_regression_arima_beta
- `exog.reshape(-1, 1)` -> `reshape(-2, 1)`: numpy treats any single negative dimension as the inferred axis, so `reshape(-2, 1)` == `reshape(-1, 1)` for any 1D input.

## src/tsbootstrap/model/arima.py :: arma_initial_state
- `ValueError(...)` message arg -> `None`: raised only on the init-length-mismatch path, which the covering tests assert via bare `pytest.raises(ValueError)`; the message content is never inspected and `ValueError(None)` is still a `ValueError`. Dead message content.
- `a = concatenate([[1.0], ...])` -> `[[2.0], ...]`: `a` is consumed only by `lfiltic(b, a, ...)`, whose loops read `a[m+1:]` and never `a[0]`; `K = max(M, N)` is independent of `a[0]`'s value. `a[0]` is structurally dead in lfiltic (verified bit-identical).
- `b = concatenate([[1.0], ...])` -> `[[2.0], ...]`: same proof; lfiltic reads `b[m+1:]` and never `b[0]`.

## src/tsbootstrap/engines/arma_scipy.py :: simulate_ar_batched
- `a[m+1:p+1]` -> `a[m+1:p+2]`: `a` has exactly `p+1` elements (indices 0..p); the out-of-range upper bound is clipped to the array end by numpy, so `a[m+1:p+2]` == `a[m+1:p+1]` for every `m` in `range(p)`. Reduction bit-identical.
- `lfilter(..., axis=1)` -> default axis: `forcing` is always 2-D `(n_paths, m)`; scipy `lfilter`'s default `axis=-1` equals `axis=1` for 2-D input, and the `zi` shape is consistent with both.

## src/tsbootstrap/engines/arma_scipy.py :: simulate_arma_batched
- `lfilter(..., axis=1)` -> default axis (zero-state path): `e` is always 2-D `(B, m)`; default `axis=-1` == `axis=1`.
- `lfilter(..., axis=1)` -> default axis (conditional path): same; `zi` broadcast over the B axis is unaffected.

## src/tsbootstrap/model/fit.py :: _ols
- `lstsq(design, y, rcond=None)` -> drop `rcond=None`: numpy 2.x defines the omitted-`rcond` default AS `rcond=None` (machine-precision cutoff); rank and beta are bit-identical on full, rank-deficient, and ill-conditioned designs (no deprecation path in numpy 2.x).
- `cast(..., beta)` -> `cast(None, beta)`: `typing.cast` is a runtime no-op that returns its second argument unchanged; only the (type-erased) annotation differs.

## src/tsbootstrap/model/fit.py :: fit_ar
- `exog_arr.reshape(-1, 1)` -> `reshape(-2, 1)`: single negative dimension is the inferred axis, so `reshape(-2, 1)` == `reshape(-1, 1)` for the 1D-exog branch (same rule as the `fit_regression_arima_beta` entry); the 2-D branch is guarded by `ndim == 1` and unaffected.

## src/tsbootstrap/model/fit.py :: _require_statsmodels
- `code=Codes.BACKEND_NOT_INSTALLED` -> `code=None`, and dropping the `code=` kwarg: `BackendError`'s class default IS `Codes.BACKEND_NOT_INSTALLED`, and `TSBootstrapError.__init__` resolves `self.code = code or type(self).code`, so `None` falls back to the identical code. (Doubly safe: the `except ImportError` body is also dead while statsmodels is installed; the message/hint mutants on that branch are killed by `test_require_statsmodels_raises_full_contract_when_absent`, which monkeypatches the import.)

## src/tsbootstrap/model/fit.py :: fit_var
- `exog_arr.reshape(-1, 1)` -> `reshape(-1,)`: reachable only for 1-D exog; `np.column_stack` promotes a 1-D array to the identical single column as `reshape(-1, 1)`.
- `exog_arr.reshape(-1, 1)` -> `reshape(-2, 1)`: single negative dimension inferred; equal for any 1-D input (same rule as above).

## src/tsbootstrap/model/stability.py :: ar_spectral_radius / var_spectral_radius
- `if p > 1` -> `if p >= 1`: for `p == 0` the function returns early (`0.0`); for `p == 1` the guarded body assigns `companion[1:, :p-1] = np.eye(p-1)`, i.e. a shape-`(0, 0)` write from a shape-`(0, 0)` array (a verified no-op); for `p > 1` both predicates hold. Identical on every reachable input. (Applies to the AR and VAR twins.)

## src/tsbootstrap/model/stability.py :: check_ar_stability / check_var_stability
- `code=Codes.UNSTABLE_MODEL` -> `code=None`, and dropping the `code=` kwarg: `ModelStabilityError.code` is a class attribute and `__init__` resolves `code or type(self).code`, so `None` falls back to the same `Codes.UNSTABLE_MODEL`; the existing `err.code == Codes.UNSTABLE_MODEL` assertions still pass. (Applies to AR and VAR twins.)
- `stacklevel=3` -> dropped, and `stacklevel=3` -> `4`: `stacklevel` only changes the source file/line a warning is ATTRIBUTED to; the warning category, message, and context are identical and no test asserts `w.filename`/`w.lineno`. Cosmetic attribution only. (Applies to AR and VAR twins.) NOTE: if the class-attribute `code` default is ever removed, the `code` mutants above become real gaps.

## src/tsbootstrap/model/recursive.py :: _draw_innovations_and_inits
- `gen.integers(0, n_resid, size=...)` -> `gen.integers(n_resid, size=...)`: numpy defines `integers(high)` as `integers(0, high)`; identical draws and identical post-draw RNG state.
- `gen.integers(0, n_series - p + 1)` -> `gen.integers(n_series - p + 1)`: same identity; `start` is drawn from `[0, n_series - p + 1)` either way.

## src/tsbootstrap/model/recursive.py :: _build_ar_context
- `exog is not None and exog_coefs is not None` -> `or`: `fit_ar` sets `exog_coefs` non-None iff `exog` is non-None, so the two operands are always equal on every reachable state and `and` == `or`.

## src/tsbootstrap/model/recursive.py :: _prepare_var
- `exog is not None and exog_coefs is not None` -> `or`: `fit_var` sets `exog_coefs` non-None iff `exog` is non-None, so the operands are always equal and `and` == `or` (same proof shape as the `_prepare_arima` / `_build_ar_context` entries).

## Wild innovation code (first full run after the 0.4.0 wild/block-wild landing)

### model/recursive.py :: _arima_batched / _draw_innovations_and_inits (dead-store initializers)

- `x__arima_batched__mutmut_22` / `_23` and `x__draw_innovations_and_inits__mutmut_15` / `_16`
  (`n_draw = 0` -> `None` / `1`): the initializer is a dead store. When the wild plan is None the
  variable is never read; when a plan exists the conditional assignment overwrites it before any
  read. No reachable input observes the initial value.

### model/recursive.py :: _draw_multipliers

- `x__draw_multipliers__mutmut_6` (`astype(np.float64)` -> `astype(None)`): numpy defines
  `dtype(None)` as float64, so the cast is byte-identical (verified empirically).
- `x__draw_multipliers__mutmut_10` (`integers(0, 2, size)` -> `integers(2, size)`): when `high`
  is omitted, numpy treats the single argument as the exclusive upper bound with `low=0`; the two
  calls draw identical values from the identical stream (verified empirically).
- `x__draw_multipliers__mutmut_29` (Mammen threshold `u < P` -> `u <= P`): `u` is a continuous
  float64 draw; equality with the irrational-valued threshold has measure zero (no representable
  seed-reachable draw equals it), so the two comparisons partition every reachable draw
  identically.

### model/recursive.py :: _wild_plan

- `x__wild_plan__mutmut_17` (`reshape(-1, 1)` -> `reshape(-1,)`): only the 1-D branch reaches
  this expression, and `optimal_block_length` reshapes 1-D input to a column itself, so both
  shapes produce the identical estimate (verified empirically).
- `x__wild_plan__mutmut_23` / `_25` / `_26` / `_27` (`kind="circular"` -> `None` / dropped /
  `"XXcircularXX"` / `"CIRCULAR"`): `optimal_block_length` only compares `kind` against the exact
  string `"stationary"`; every other value selects the circular constant, so all four mutants
  compute the identical length (verified empirically for `None`).
- `x__wild_plan__mutmut_41` / `_48` (`stacklevel=4` dropped / -> `5`): stacklevel controls only
  the file/line attribution of the emitted warning, not whether it is raised, its category, its
  message, or its context payload; no behavioral contract observes it.
- `x__wild_plan__mutmut_19` (`reshape(-1, 1)` -> `reshape(-2, 1)`): numpy treats any negative
  reshape dimension as the single unknown to infer, so `-2` behaves exactly like `-1`
  (verified empirically); the resulting array is identical.

## AgACI / ACI / NexCP calibrators (uq/adaptive.py)

First full run after the 0.5.0 AgACI landing. Every other survivor has a dedicated killing
test in `tests/unit/test_mutation_kills.py` (exact-message, quantile-bit-exactness, over-coverage,
default-argument, BOA-forwarding, and two deterministic golden tests, plus the metamorphic
versions in `tests/property/test_conformal_rng_invariants.py`); each kill was confirmed by the
390-mutant remote run dropping the survivor set from 104 to the 15 genuine equivalents below.

### _aci_recursion_presorted (the presorted ACI quantile recursion)
- `a_clip = min(max(a_t, 0.0), 1.0)` -> `2.0` (widen the upper clamp): whenever `a_t > 1` the very
  next branch `elif a_clip >= 1.0: q = 0.0` fires identically for any `a_clip` in `(1, 2]`, so the
  wider clamp never changes `q`. Verified: the over-coverage test still passes.
- `frac >= 0.5` -> `frac > 0.5`: differs only at exactly `frac == 0.5`, where the two lerp branches
  `b - diff*(1-frac)` and `a + diff*frac` are algebraically identical (both equal `0.5*(a+b)`), so
  the branch taken does not change `q`.
- `frac >= 0.5` -> `frac >= 1.5`: `frac` is in `[0, 1)` so the guard is never true (always the else
  branch); but `b - diff*(1-frac) == a + diff*frac` identically, so forcing the else branch returns
  the same `q` (the two-branch form only exists to match numpy's quantile evaluation order).
- `test[t] > q` -> `test[t] >= q`: differs only at the measure-zero point `test[t] == q`; no
  representable seed-reachable stream lands a residual exactly on its own quantile, so the realized
  coverage indicator is identical.

### _boa_aggregate (the Bernstein Online Aggregation kernel)
- `return_weights: bool = False` -> `True` (default only): `agaci_bounds` calls
  `pred, _ = _boa_aggregate(...)` and discards the weights, so the extra allocation is unobservable;
  the golden tests still pass bit-for-bit.
- `raux = ... + np.log(w0[nz])` -> `... - np.log(w0[nz])`: `w0` is the uniform prior weight (the same
  value for every expert), so `+/- log(w0)` is a per-expert-constant shift and the max-stabilized
  softmax `exp(raux - raux.max()) / sum` is invariant to a constant added to every `raux`. Verified:
  the bit-exact agaci golden does not distinguish it.
- `exp(raux - raux.max())` -> `exp(raux + raux.max())`: softmax shift-invariance again -- adding vs
  subtracting the constant `raux.max()` from every element leaves the normalized weights identical;
  the `- max` form exists only for float overflow safety.

### agaci_bounds (the two-sided AgACI driver)
The scale-equivariant sentinel is `min(10.0 * range_ref, cap)` with
`data_scale = max(finite_Q.max() if finite_Q.size else 0.0, test_abs_max)` and
`range_ref = data_scale if data_scale > 0.0 else 1.0`. Every ACI expert's step-0 half-width is a
finite quantile (the level starts at `alpha > 0`), so `finite_Q` is NEVER empty and `data_scale` is
always `finite_Q.max() > 0`; the two degenerate fallbacks below are therefore dead code.
- `finite_Q.max() if finite_Q.size else 0.0` -> `else 1.0`: `finite_Q.size` is always true (the
  step-0 half-width is finite), so this `else` branch is never taken.
- `data_scale > 0.0` -> `data_scale >= 0.0`: differs only at `data_scale == 0`, which needs all-zero
  residuals -- but all-zero residuals cause no misses, hence a finite expert, hence `data_scale > 0`.
  Unreachable.
- `range_ref = data_scale if data_scale > 0.0 else 1.0` -> `else 2.0`: the `data_scale <= 0` branch is
  dead by the same argument, so the fallback constant is never read.

### nexcp_quantile (the decay-weighted NexCP quantile)
- `np.argsort(s, kind="stable")` -> `kind=None` and -> argument dropped: `None` selects numpy's
  default quicksort; the returned order statistic is identical (any ties carry equal values, so the
  pick is unchanged).
- `kind="stable"` -> `kind="STABLE"`: numpy normalizes the sort-kind string case-insensitively
  (verified: `np.argsort(kind="STABLE")` succeeds and matches stable), so the sort is identical.
- `np.searchsorted(cdf, target, side="left")` -> `side` dropped: numpy's default `side` is `"left"`,
  so the index is identical.
- `s_sorted[min(idx, n - 1)]` -> `min(idx, n + 1)`: `idx = searchsorted(cdf, target, "left")` with
  `target = (1-alpha) * cdf[-1] <= cdf[-1]`, so `idx <= n - 1` always and the clamp never engages;
  `min(idx, n-1) == min(idx, n+1) == idx`.
