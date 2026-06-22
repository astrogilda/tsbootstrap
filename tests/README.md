# Test Suite Organization

This directory holds the test suite for tsbootstrap. It is split into a
capability-class unit layer and a Hypothesis-driven property layer, both sharing
a small set of synthetic data-generating processes.

## Structure

```
tests/
├── unit/                          # Capability-class unit tests
│   ├── test_adapters.py           # sktime/skbase bootstrap adapters
│   ├── test_adaptive.py           # adaptive/nonexchangeable conformal calibration (ACI, NexCP)
│   ├── test_batched_engine.py     # golden-master + chunking-determinism for the batched engines
│   ├── test_block_indices.py      # block index generation (moving/circular/non-overlapping/stationary)
│   ├── test_bootstrap_api.py      # public bootstrap() entry point (IID baseline)
│   ├── test_diagnostics.py        # diagnose()
│   ├── test_enbpi_ensemble.py     # EnbPIEnsemble fit/predict object and the calibrator family
│   ├── test_errors.py             # structured error/warning taxonomy (tsbootstrap.errors)
│   ├── test_exog.py               # exogenous-covariate support (ARX and VARX residual bootstraps)
│   ├── test_method_specs.py       # typed method specifications (tsbootstrap.methods)
│   ├── test_narwhals_boundary.py  # narwhals DataFrame/Series input boundary (pandas + Polars)
│   ├── test_pwsd.py               # automatic block-length selection (tsbootstrap.block.pwsd)
│   ├── test_recursive_arima.py    # recursive ARIMA bootstrap (differenced-scale simulation)
│   ├── test_recursive_ar.py       # recursive AR and sieve bootstrap
│   ├── test_recursive_var.py      # recursive VAR bootstrap (multivariate)
│   ├── test_reduce.py             # bootstrap_reduce streaming per-replicate statistic API
│   ├── test_rng_contract.py       # deterministic, parallel-safe RNG contract (tsbootstrap.rng)
│   ├── test_tapered.py            # tapered block bootstrap (tsbootstrap.block.tapered)
│   ├── test_uq.py                 # UQ layer: EnbPI (regression) and forecast intervals
│   └── test_validation_contract.py # input-validation contract (tsbootstrap.validation)
│
├── property/                      # Hypothesis property-based tests
│   ├── test_coverage.py           # statistical coverage gate
│   ├── test_invariants.py         # algebraic and metamorphic invariants for engines + public API
│   ├── test_properties.py         # property-based invariants for the public bootstrap API
│   ├── test_reference.py          # reference cross-checks against an independent implementation (arch)
│   └── test_symbolic.py           # symbolic-execution checks via Hypothesis's CrossHair backend
│
├── _helpers/                      # Shared test utilities
│   └── dgp.py                     # synthetic data-generating processes used across the suite
│
└── conftest.py                    # Pytest + Hypothesis profile configuration
```

The repo root is placed on `sys.path` (via `pythonpath = ["."]` in
`pyproject.toml`) so tests can import the shared `tests._helpers` package.

## Test layers

### Unit tests (`tests/unit/`)

One file per capability class: the engines, the public `bootstrap()` entry
point, the block-index generators, the UQ layer, the conformal calibrators, the
input boundaries (narwhals, exogenous regressors), the typed method specs, the
RNG contract, and the error taxonomy. Each file exercises a single component in
isolation, including its edge cases, parameter validation, and determinism
guarantees.

### Property tests (`tests/property/`)

Hypothesis-driven checks that assert behavior across generated inputs rather than
fixed cases: algebraic and metamorphic invariants of the engines and public API,
a statistical coverage gate, reference cross-checks against the `arch` package,
and optional symbolic-execution checks via the CrossHair backend.

## Running the suite

```bash
# Full suite (xdist runs it in parallel by default; see addopts in pyproject.toml)
uv run pytest tests/

# A single layer
uv run pytest tests/unit/
uv run pytest tests/property/

# A single file or a single test
uv run pytest tests/unit/test_uq.py
uv run pytest tests/unit/test_uq.py::test_enbpi_coverage

# By marker
uv run pytest tests/ -m smoke
```

`addopts` in `[tool.pytest.ini_options]` already enables parallel execution
(`-n auto --dist loadscope`), reports the 20 slowest tests, and uses quiet
output. Registered markers include `smoke`, `slow`, `performance`,
`integration`, `network`, `cloud`, and `gpu`.

## Coverage

Coverage is configured in `[tool.coverage.run]` to measure `src/` only:

```bash
uv run pytest tests/ --cov=src --cov-report=term-missing
uv run pytest tests/ --cov=src --cov-report=html   # writes htmlcov/
```

## Hypothesis profiles

The property layer selects a Hypothesis profile via the `HYPOTHESIS_PROFILE`
environment variable (default `dev`). Profiles are registered in `conftest.py`:

| Profile    | Examples | Purpose                                                        |
|------------|----------|---------------------------------------------------------------|
| `dev`      | 25       | Fast local feedback (default)                                  |
| `ci`       | 100      | More examples, slow-health-check relaxed                       |
| `thorough` | 1000     | Nightly deep search                                           |
| `mutmut`   | 40       | Deterministic (derandomized) profile for mutation testing     |
| `symbolic` | 50       | CrossHair concolic backend; registered only when installed    |

```bash
HYPOTHESIS_PROFILE=ci uv run pytest tests/property/
```

The `symbolic` profile needs the optional `hypothesis-crosshair` backend; it is
registered only when that backend is importable, and selecting it without the
backend falls back to `dev` rather than erroring mid-run.
