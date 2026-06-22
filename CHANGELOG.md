# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow semantic versioning.

## [0.2.0] - 2026-06-22

v0.2.0 is a correctness-first rewrite. The public surface is one function, `bootstrap(X, *, method=...)`,
configured with a typed, frozen method spec. The 0.1.x class-based API is gone; this is a breaking release.

### Added
- One entry point, `bootstrap`, plus `bootstrap_reduce`. The latter evaluates a per-replicate statistic
  inside the generation loop and returns only the reduced results, which bounds peak memory regardless of
  replicate count.
- Typed method specs (pydantic, frozen, `extra="forbid"`): `IID`; the block family `MovingBlock`,
  `CircularBlock`, `StationaryBlock`, `NonOverlappingBlock`, `TaperedBlock`; and the model-based
  `ResidualBootstrap` (with an `AR`, `ARIMA`, or `VAR` model) and `SieveAR`.
- Automatic block length: `block_length="auto"` uses the Politis-White (2004) / Patton-Politis-White
  (2009) selector instead of a fixed `sqrt(n)`.
- Exogenous regressors for the model-based methods: ARX, VARX, and ARIMAX, held fixed during regeneration.
- Uncertainty quantification (`tsbootstrap.uq`): `EnbPIEnsemble`, a fit/predict object that retains the
  bootstrap ensemble and produces in-sample and out-of-sample prediction intervals with a choice of
  calibrator (static, sliding-window, ACI, or NexCP); adaptive conformal calibrators `aci_halfwidths`
  (Gibbs-Candes 2021) and `nexcp_quantile` (Barber 2023); and `forecast_intervals` for forward AR simulation.
- DataFrame input through Narwhals: pass pandas, Polars, or PyArrow frames and series.
- sktime / skbase estimator classes under `tsbootstrap.adapters`.
- An optional compiled VAR kernel (`pip install "tsbootstrap[accel]"`, numba), auto-selected when present.
- A deterministic per-replicate RNG contract: replicate i is bound to its own stream, so results are
  reproducible for a given seed and environment regardless of worker count or chunking.
- `diagnose(x)` to suggest methods for a series.
- Python 3.13 support (CI matrix and packaging metadata).

### Changed
- Recursive residual bootstraps now regenerate replicates from the fitted dynamics and resampled,
  centered innovations (replacing the previous `fitted + residuals` reconstruction).
- ARIMA replicates are conditioned on the observed initial state with lfilter-consistent residuals.
  Re-injecting a replicate's own residuals now reconstructs the observed series exactly.
- Non-stationary model fits are refused, or skipped via `stability_policy`, to prevent explosive paths.

### Removed (breaking)
- The 0.1.x class API, `TSFit`, `n_jobs`/joblib parallelism, the async layer, the feature-flag system,
  and the statistic-preserving method.
- `burn_in` and `initial` on the `ARIMA` spec: these are now rejected at construction, because ARIMA
  conditions on the observed initial state. They remain available on `AR`, `VAR`, and `SieveAR`.

### Quality
- Blocking CI gates: mypy (strict-minus) and pyright (strict) at zero errors, ruff lint and format, and
  the test suite (including a property-based invariant layer) across the Python 3.10 to 3.13 matrix.
  Coverage is measured and uploaded to Codecov.

[0.2.0]: https://github.com/astrogilda/tsbootstrap/releases/tag/v0.2.0
