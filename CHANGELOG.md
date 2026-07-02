# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow semantic versioning.

## [0.4.0](https://github.com/astrogilda/tsbootstrap/compare/tsbootstrap-v0.3.1...tsbootstrap-v0.4.0) (2026-07-02)


### Features

* classical confidence-interval layer with one-call conf_int ([#223](https://github.com/astrogilda/tsbootstrap/issues/223)) ([9f7c4d2](https://github.com/astrogilda/tsbootstrap/commit/9f7c4d2dc44daca0d14de2f1ff53b6d352a2cbcf))
* wild and block-wild bootstrap innovations ([#222](https://github.com/astrogilda/tsbootstrap/issues/222)) ([865543d](https://github.com/astrogilda/tsbootstrap/commit/865543d0489372a1a0125bcfbbef2da284b57bba))


### Documentation

* complete the citation metadata file ([#220](https://github.com/astrogilda/tsbootstrap/issues/220)) ([30d61a8](https://github.com/astrogilda/tsbootstrap/commit/30d61a8b72d33a28cf1131d34eb826fc7a2b1bc6))

## [0.3.1](https://github.com/astrogilda/tsbootstrap/compare/tsbootstrap-v0.3.0...tsbootstrap-v0.3.1) (2026-06-24)


### Documentation

* refresh tutorial notebooks for the 0.3.0 APIs ([#213](https://github.com/astrogilda/tsbootstrap/issues/213)) ([f5ef2e1](https://github.com/astrogilda/tsbootstrap/commit/f5ef2e1a689c8ad8bec22194b1ff170fd7cd3342))

## [0.3.0](https://github.com/astrogilda/tsbootstrap/compare/tsbootstrap-v0.2.0...tsbootstrap-v0.3.0) (2026-06-23)


### Features

* `scikit-base` registry and testing framework ([#26](https://github.com/astrogilda/tsbootstrap/issues/26)) ([4fc838e](https://github.com/astrogilda/tsbootstrap/commit/4fc838ea9fa2db65940a5a8b3d348642ec79194e))
* `skbase` inheritance for config objects ([#31](https://github.com/astrogilda/tsbootstrap/issues/31)) ([24ced3c](https://github.com/astrogilda/tsbootstrap/commit/24ced3c383817694d29f9e203fee654a5e3ace86))
* `skbase` inheritance in central base classes ([#10](https://github.com/astrogilda/tsbootstrap/issues/10)) ([3d650b8](https://github.com/astrogilda/tsbootstrap/commit/3d650b8e58776d90c0281419cfbd8c82a1f11865))
* 44 sklearn compatible interface (with fkiraly) ([#66](https://github.com/astrogilda/tsbootstrap/issues/66)) ([1b521ca](https://github.com/astrogilda/tsbootstrap/commit/1b521caa0e55735202fb87d9a21c897c5a9eb9ed))
* accept pandas/Polars DataFrames via the narwhals boundary ([c4fd180](https://github.com/astrogilda/tsbootstrap/commit/c4fd1806c1b80eca26c0ac098e3a51133af2eb00))
* adaptive and nonexchangeable conformal calibration ([28396c7](https://github.com/astrogilda/tsbootstrap/commit/28396c7b854abb925bebd7fbf2e61fc8cd8903b0))
* add compiled acceleration backend, named reducers, float32 dtype, and panel reduce ([077a258](https://github.com/astrogilda/tsbootstrap/commit/077a2582f8395c9d237df3c28bf5fb1ca51c6509))
* added dunder methods ([8d98867](https://github.com/astrogilda/tsbootstrap/commit/8d988670237c41c985dcb8ba1f807422d533c1c7))
* ARIMAX exogenous-covariate support (regression with ARIMA errors) ([5fb96e1](https://github.com/astrogilda/tsbootstrap/commit/5fb96e13af6147b4f141b0058672bcb86a497f5d))
* block index kernels and automatic block length ([59c169f](https://github.com/astrogilda/tsbootstrap/commit/59c169f57651c859640c65687db8cf11a35b7409))
* bootstrap_reduce for streaming per-replicate statistics ([64f30d8](https://github.com/astrogilda/tsbootstrap/commit/64f30d8ec1db9e6c37054caa4d010ecf955fdbad))
* change `bootstrap` `test_size` argument default to 0.0 ([#21](https://github.com/astrogilda/tsbootstrap/issues/21)) ([152f0fd](https://github.com/astrogilda/tsbootstrap/commit/152f0fd7b1d8efe6bca0f596e4777611f8302f1f))
* created separate config module ([723b3c8](https://github.com/astrogilda/tsbootstrap/commit/723b3c8ea6bfcfba70e8878dee3553e5042ae62f))
* diagnose() method advisor ([8928be9](https://github.com/astrogilda/tsbootstrap/commit/8928be96472b01a41f0770d682595e672b1d0566))
* enable backends by default with 7.66x performance improvement ([#196](https://github.com/astrogilda/tsbootstrap/issues/196)) ([8b5e410](https://github.com/astrogilda/tsbootstrap/commit/8b5e41070d22edc57484ff7ddc0e03133257921c)), closes [#194](https://github.com/astrogilda/tsbootstrap/issues/194)
* ensure bootstrap tests also run via `check_estimator` ([#84](https://github.com/astrogilda/tsbootstrap/issues/84)) ([a97ee34](https://github.com/astrogilda/tsbootstrap/commit/a97ee34641e16c949ef6a9881522cf19e25be2a5))
* exogenous-covariate support for the AR residual bootstrap ([e44328c](https://github.com/astrogilda/tsbootstrap/commit/e44328c3c7d130ff24b84a4b56ca89ba80bdf6d3))
* extend suite test coverage to further `bootstrap` contract elements - equal shape as origial series, `test_ratio`, number of bootstraps ([#83](https://github.com/astrogilda/tsbootstrap/issues/83)) ([0c15713](https://github.com/astrogilda/tsbootstrap/commit/0c1571381fef8ce1033a7de9e620c5e23d2e3806))
* isolate `arch` and `statsmodels` as soft dependencies, new `all_extras` soft dependency set ([#39](https://github.com/astrogilda/tsbootstrap/issues/39)) ([a372c1a](https://github.com/astrogilda/tsbootstrap/commit/a372c1accff62d08744cf5b8e17e2536bd291775))
* isolate `scikit-learn-extra` ([#52](https://github.com/astrogilda/tsbootstrap/issues/52)) ([5e4cce3](https://github.com/astrogilda/tsbootstrap/commit/5e4cce3125322d0bfbc99a3acfacea0157c36cab))
* **mcp:** add a read-only MCP server with two bootstrap tools ([07bb9cd](https://github.com/astrogilda/tsbootstrap/commit/07bb9cd7173ed5c9e3fed219d20d270c8fe0d5ff))
* re-export the uncertainty-quantification surface at the top level ([7b2f623](https://github.com/astrogilda/tsbootstrap/commit/7b2f62350772f1217b696ee80afa71366860994a))
* recursive AR and sieve bootstrap engine ([55d9e81](https://github.com/astrogilda/tsbootstrap/commit/55d9e81713ffec8591f95e4a1292b592df094fdd))
* recursive ARIMA bootstrap on the differenced scale ([7b8fcec](https://github.com/astrogilda/tsbootstrap/commit/7b8fcece204d5f1df3efdc1e43d89655ca90c83c))
* recursive VAR bootstrap (multivariate) ([38277a7](https://github.com/astrogilda/tsbootstrap/commit/38277a71bc50f7611628c97d647c5d2de3ee13f7))
* remove `numba` ([#24](https://github.com/astrogilda/tsbootstrap/issues/24)) ([5a8d595](https://github.com/astrogilda/tsbootstrap/commit/5a8d59533285b6f480ce7ca593187571153abd66))
* removes stray todo comments from `base_bootstrap` ([#101](https://github.com/astrogilda/tsbootstrap/issues/101)) ([4da06ec](https://github.com/astrogilda/tsbootstrap/commit/4da06ec6444139bc9ed85fd9bd7e88ed8148b413))
* separated ranklags into a module ([27ee6ea](https://github.com/astrogilda/tsbootstrap/commit/27ee6ea1fe14d12788c622b3d8049543e301565a))
* skbase/sktime bootstrap adapters (concrete over shared base) ([818925e](https://github.com/astrogilda/tsbootstrap/commit/818925e25c4d18632effdbc6de2a53e8d23f3e0f))
* stability_policy=skip for pipeline-survivable failed fits ([fa2d5e8](https://github.com/astrogilda/tsbootstrap/commit/fa2d5e87d036ab71392512ba49a807da28ee33fe))
* structured errors, deterministic RNG contract, input validation ([4741a6b](https://github.com/astrogilda/tsbootstrap/commit/4741a6bc584f3433b702fb8ca27753528d4bb19c))
* tapered block bootstrap with energy-normalized windows ([dd23d6d](https://github.com/astrogilda/tsbootstrap/commit/dd23d6dfd3fd9d5483b65245ee7faf68ed663e5a))
* typed method API, structured results, i.i.d. baseline ([d5a31fc](https://github.com/astrogilda/tsbootstrap/commit/d5a31fcc75815d90f59624cba9e9e293547166fa))
* UQ layer with EnbPI and bootstrap forecast intervals ([175724d](https://github.com/astrogilda/tsbootstrap/commit/175724dfc1195b2a3c9d0396af694027af30729b))
* **uq:** EnbPIEnsemble fit/predict object with calibrator family and sliding-window EnbPI ([965074a](https://github.com/astrogilda/tsbootstrap/commit/965074a1e2daa104fa388ec73dd30949ca1b2b8e))
* VARX exogenous-covariate support for the VAR residual bootstrap ([5ba9c43](https://github.com/astrogilda/tsbootstrap/commit/5ba9c4380e2e6acae139ccbf0eb338821160891a))


### Bug Fixes

* added tsbs logo ([289087f](https://github.com/astrogilda/tsbootstrap/commit/289087fbf848464a0f7be4d28a4a30204b429616))
* ARIMA residuals consistent with the lfilter simulation engine ([070faab](https://github.com/astrogilda/tsbootstrap/commit/070faab59173b2e3f95da80f4d91e1689af8d7d2))
* avoid float-equality checks flagged by SonarCloud ([c0b2a6e](https://github.com/astrogilda/tsbootstrap/commit/c0b2a6e6cde8213ae64720bd5a388772b03e46d0))
* change `exog` to `y` in methods of `BaseTimeSeriesBootstrap` ([#30](https://github.com/astrogilda/tsbootstrap/issues/30)) ([#32](https://github.com/astrogilda/tsbootstrap/issues/32)) ([b3f6053](https://github.com/astrogilda/tsbootstrap/commit/b3f605321eb2f9fc9ec8e9ebc5d939bda8fe1cd6))
* change `exog` to `y` in methods of `BaseTimeSeriesBootstrap` ([#30](https://github.com/astrogilda/tsbootstrap/issues/30)) ([#33](https://github.com/astrogilda/tsbootstrap/issues/33)) ([a5aba98](https://github.com/astrogilda/tsbootstrap/commit/a5aba988ad81cb91d91dee7440d98c119eac1090))
* condition the ARIMA bootstrap on the observed initial state ([4fd51c1](https://github.com/astrogilda/tsbootstrap/commit/4fd51c14252be4bdab04582d5715b620fcd40473))
* ensure `check_estimator` runs bootstrap tests for bootstraps ([#87](https://github.com/astrogilda/tsbootstrap/issues/87)) ([ba18672](https://github.com/astrogilda/tsbootstrap/commit/ba186723e100b4e87f7721ed8d1236d5619186e6))
* failing tests ([202f93f](https://github.com/astrogilda/tsbootstrap/commit/202f93fa9e23bfffd5c304ad34ba4fb3154f10c1))
* PR CI trigger condition ([#11](https://github.com/astrogilda/tsbootstrap/issues/11)) ([d2862fb](https://github.com/astrogilda/tsbootstrap/commit/d2862fb5c317183f77f4b3586676579163ae87de))
* **quality:** clear SonarCloud findings and scope analysis to src ([517cfc2](https://github.com/astrogilda/tsbootstrap/commit/517cfc2172f797192ed14ae9b5436bf357b64e20))
* remove fail-fast condition for matrix CI ([#19](https://github.com/astrogilda/tsbootstrap/issues/19)) ([2a12437](https://github.com/astrogilda/tsbootstrap/commit/2a124375ea1499ce11c089a524d1329b4f69eb62))
* removed erraneously comments out test code ([6cea59b](https://github.com/astrogilda/tsbootstrap/commit/6cea59ba007ff12ad1956a208f8cfb8bc2b45f72))
* run each warm-up hook once regardless of registration order ([3349f13](https://github.com/astrogilda/tsbootstrap/commit/3349f139dd9d97837e7f4ad981095d8337d1287f))
* temp commented out residual testing ([f8889cb](https://github.com/astrogilda/tsbootstrap/commit/f8889cbb551b5c87f453a15f2fca02821b3a6c7f))
* temporarily not publishing coverage reports ([2037b71](https://github.com/astrogilda/tsbootstrap/commit/2037b71151464f08e20c90f553803f6c970935fa))
* temporarily skip sporadically failing tests due to LU decomposition ([#28](https://github.com/astrogilda/tsbootstrap/issues/28)) ([7f7d357](https://github.com/astrogilda/tsbootstrap/commit/7f7d35717dbd64a1774a30ca2940be7e07577290))


### Performance Improvements

* add a profiling harness for the hot paths ([71da5bc](https://github.com/astrogilda/tsbootstrap/commit/71da5bc87cd11b8a42f7328533df7a7ddc38c5ab))
* add asv benchmark suite for the bootstrap engines ([347e3b5](https://github.com/astrogilda/tsbootstrap/commit/347e3b5f36a94f15351eb7b6e63371816f009691))
* compiled replicate-parallel VAR kernel behind the accel extra ([67f5b8a](https://github.com/astrogilda/tsbootstrap/commit/67f5b8affcfd098e4820dbaf63bb3ce1415931ce))
* fit AR/VAR/sieve by direct OLS instead of statsmodels ([f5962fa](https://github.com/astrogilda/tsbootstrap/commit/f5962faf3b067a9a49bb76f7d87358f9a3abead6))
* true-vectorize the recursive engines across replicates ([9c46c03](https://github.com/astrogilda/tsbootstrap/commit/9c46c038b1a2e3298799d1c9c3eb46bf955791c9))
* vectorize the AR initial-state across paths ([7d803e5](https://github.com/astrogilda/tsbootstrap/commit/7d803e51b0cf5b6891931e10a8def23539d42a1f))


### Documentation

* add a minimal AGENTS.md for coding agents ([3f18347](https://github.com/astrogilda/tsbootstrap/commit/3f1834732756846482606874662f9099abc87d47))
* add an executable notebook gallery with the quickstart tutorial ([4624085](https://github.com/astrogilda/tsbootstrap/commit/4624085836009d94ee5843cb440af387027bb33f))
* add Context7 and DeepWiki indexing ([95f9df5](https://github.com/astrogilda/tsbootstrap/commit/95f9df5232312bd5b9212ccc68ccf5bce4d29d1c))
* add core-methods tutorial notebooks ([312834c](https://github.com/astrogilda/tsbootstrap/commit/312834c397a44ed5fb49eeb6eee59e70d5f9aa02))
* add decision log and a detailed ARIMAX/VARX plan ([9186627](https://github.com/astrogilda/tsbootstrap/commit/9186627c43e8c80938b7d87f4646a45676701024))
* add getting-started and decision-guide tutorial notebooks ([928c840](https://github.com/astrogilda/tsbootstrap/commit/928c84067fec08ed6793a161d9d8276f136c7d48))
* add the uncertainty-quantification guide and API reference ([3d6492f](https://github.com/astrogilda/tsbootstrap/commit/3d6492f02dbe72216ac3b9c846e3565d8099382e))
* add uncertainty-quantification and integration tutorial notebooks ([c7ed2e6](https://github.com/astrogilda/tsbootstrap/commit/c7ed2e63ebe006f64869d29656d73ca2be6506d7))
* add v0.2.0 plan and development backlog ([45a58d9](https://github.com/astrogilda/tsbootstrap/commit/45a58d9cb7f92cfa1a75fee959275333eeca5e23))
* added readthedocsyaml ([8bb0001](https://github.com/astrogilda/tsbootstrap/commit/8bb0001066e5952221c68908ca3f4e01dd3184cb))
* bump CITATION.cff version to 0.2.0 ([3bbdbef](https://github.com/astrogilda/tsbootstrap/commit/3bbdbefaa126c17e8c3a58ad99d21c719444ff98))
* capture future-capability roadmap and the exog scope decision ([3285ad0](https://github.com/astrogilda/tsbootstrap/commit/3285ad0beed39edf68ba32bce3a4d91b67d19c50))
* correct stale guides and changelog against the v0.2.0 code ([afa60af](https://github.com/astrogilda/tsbootstrap/commit/afa60af1dba72f04fb5e8d1a26809eac5fb567bb))
* docstring for `bootstrap` method in base class ([#20](https://github.com/astrogilda/tsbootstrap/issues/20)) ([2a50e20](https://github.com/astrogilda/tsbootstrap/commit/2a50e20bb4873cfb311e4be53e049bb932e87e7d))
* fixed error with tukeyalpha ([46f3ccd](https://github.com/astrogilda/tsbootstrap/commit/46f3ccdebb6e1952cfeea6dca756c11535d9798c))
* for new src files ([3ee4f1b](https://github.com/astrogilda/tsbootstrap/commit/3ee4f1b19033a0c32bc56229824966a15d1410f5))
* humanize README and CONTRIBUTING prose ([e64367b](https://github.com/astrogilda/tsbootstrap/commit/e64367b21e7d1dbab6b665de72a8931c86229b91))
* installed tsbootstrap in readthedocs, enabled build pull requests ([#72](https://github.com/astrogilda/tsbootstrap/issues/72)) ([16db5c1](https://github.com/astrogilda/tsbootstrap/commit/16db5c1c2e3e611b4e7e0be01a81332b60af13a4))
* mark exogenous support (ARX/VARX/ARIMAX) complete ([0b38699](https://github.com/astrogilda/tsbootstrap/commit/0b3869995cab690d8c96958c75aa63cd94859ca9))
* **mcp:** add the registry manifest, README section, and publish workflow ([44ede19](https://github.com/astrogilda/tsbootstrap/commit/44ede1973a041cd2b0821c81e9fb1d55491a110a))
* remove the Code Climate maintainability badge ([b793157](https://github.com/astrogilda/tsbootstrap/commit/b7931576ee3b313e5251eedc05821374916cf417))
* replace em dashes with plain punctuation across docs, docstrings, and comments ([8641a32](https://github.com/astrogilda/tsbootstrap/commit/8641a32c13e6ff4beac9732f7a1424241888d300))
* rewrite README for the typed v0.2.0 API ([14af6a0](https://github.com/astrogilda/tsbootstrap/commit/14af6a01e79cc7ac0ebe8954f5cfac3ae8440b02))
* rewrite the Sphinx documentation for the new API ([511ed4d](https://github.com/astrogilda/tsbootstrap/commit/511ed4db236b36684f40f86d6d10f89f3ed2c375))
* state behaviour directly in docstrings and notes ([3b866e6](https://github.com/astrogilda/tsbootstrap/commit/3b866e67455a77628e851267e371c4d019247dcf))
* tighten the contributor and migration guides and align the roadmap ([c3db0f6](https://github.com/astrogilda/tsbootstrap/commit/c3db0f62a84a1f2fc7e2214b5bd12d25438939be))
* updated contributing guide ([#78](https://github.com/astrogilda/tsbootstrap/issues/78)) ([5edadc7](https://github.com/astrogilda/tsbootstrap/commit/5edadc743104ddb2d6ada68054a57695af7774b8))
* updated docstrings ([9e9df1f](https://github.com/astrogilda/tsbootstrap/commit/9e9df1fe85b6df18b86edc8ad8b4328e921fe0c1))
* updated pyproject.toml dependencies and .readthedocs.yaml ([653abb9](https://github.com/astrogilda/tsbootstrap/commit/653abb98d4aa7a039f465c6913a8de81c65a008f))
* wire the tutorial gallery and refresh the README for v0.2.0 ([7f7ea62](https://github.com/astrogilda/tsbootstrap/commit/7f7ea628253ff5fcd3de257269d01927179009ca))

## [Unreleased]

### Changed (breaking)
- `BootstrapResult.indices()` (and the per-sample `BootstrapSample.indices`) now returns an `int32`
  array instead of the platform-native `intp` (`int64` on 64-bit builds), which halves the index
  memory. The index values are unchanged and `.values()` is bit-identical. A producer guard refuses a
  series of `2**31` or more observations with a `ValueError` rather than letting an index silently wrap.

## [0.2.0] - 2026-06-22

v0.2.0 rewrites the core for correctness. The public surface is one function, `bootstrap(X, *, method=...)`,
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
