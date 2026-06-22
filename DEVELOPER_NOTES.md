# Developer Notes

Developer-facing guidance for working on tsbootstrap v0.2.0: optional-dependency
extras, how to run the quality gates, project conventions, and known gotchas.
The test-suite layout (what lives where under `tests/`) is documented separately
in `tests/README.md`; this file does not duplicate the test tree.

## Optional-dependency extras

The core install (`numpy`, `scipy`, `pydantic`, `scikit-base`, `narwhals`) keeps
the dependency surface small. Everything else is an opt-in extra declared under
`[project.optional-dependencies]` in `pyproject.toml`:

| Extra | Pulls in | What it enables |
|-------|----------|-----------------|
| `models` | `statsmodels` | ARIMA bootstraps (its MA/MLE path). AR / VAR / sieve are pure-numpy OLS and need nothing extra. |
| `uq` | `scikit-learn` | Out-of-bag uncertainty quantification (EnbPI), which needs an estimator interface. |
| `accel` | `numba` | Compiled VAR-recurrence kernel (replicate-parallel). Pure-numpy is the default; installing this auto-selects the faster kernel. |
| `examples` | `models`, `uq`, `matplotlib`, `pandas`, `sktime`, `jupyter` | Everything needed to run the tutorial notebooks locally or in CI. |
| `profile` | `scalene`, `line_profiler`, `memory_profiler`, `py-spy`, `snakeviz` | Profiling harness for finding hot paths (see `profiling/README.md`). |
| `docs` | Sphinx + theme/extension stack | Building the documentation site. |
| `dev` | test, lint, type-check, mutation, and benchmark tooling | Full development environment (see Quality gates below). |

Install with `uv` (preferred, reproducible via `uv.lock`):

```bash
uv sync                       # core only
uv sync --extra dev           # development environment
uv sync --all-extras          # everything
```

## Running tests

The suite runs under the project's own interpreter and uses pytest-xdist for
parallelism (configured in `[tool.pytest.ini_options]`):

```bash
uv run pytest tests/          # full suite
uv run pytest tests/unit      # unit tests
uv run pytest tests/property  # property-based (Hypothesis) tests
```

Forcing the project interpreter with `uv run python -m pytest tests/` avoids the
`uv run pytest` PATH fallback to a system pytest that lacks the project
dependencies. Run `uv sync --extra dev` first either way.

## Quality gates

All tooling is in the `dev` extra and runs through `uv run` so it uses the
project's pinned versions:

```bash
uv run mypy src/tsbootstrap      # strict-minus typing gate (config: [tool.mypy])
uv run pyright                   # strict type checking (config: [tool.pyright])
uv run ruff check src/ tests/    # lint (config: [tool.ruff])
uv run ruff format src/ tests/   # formatting (line length 100)
```

### Coverage

Coverage is configured under `[tool.coverage.run]` (source = `src/`, tests
omitted):

```bash
uv run pytest tests/ --cov=src/tsbootstrap --cov-report=html
```

### Mutation testing

A mutation-score ratchet runs over the engine + model core via `mutmut` (3.x).
The scope and reproducible invocation are pinned in `[tool.mutmut]`:

```bash
NUMBA_DISABLE_JIT=1 HYPOTHESIS_PROFILE=mutmut uv run mutmut run
```

- `NUMBA_DISABLE_JIT=1` is required: the `@njit` VAR kernel is otherwise
  cache-masked, so source mutations to it never execute. Interpreted execution
  lets the mutations take effect. (Compiled correctness is covered separately by
  the numba-vs-numpy agreement test in the normal JIT-on suite.)
- `HYPOTHESIS_PROFILE=mutmut` selects the derandomized profile so the
  property-test baseline is reproducible.
- `only_mutate` restricts mutation to `src/tsbootstrap/engines/*` and
  `src/tsbootstrap/model/*`.

## Warning suppression

Test-warning filters live in `[tool.pytest.ini_options].filterwarnings` in
`pyproject.toml` and apply everywhere pytest runs (local and CI alike); there is
no CI-only special case for this.

Note: the current `filterwarnings` comment attributes the `pkg_resources`
deprecation to `statsforecast -> fugue -> triad -> fs`. That attribution is
stale. Neither `statsforecast` nor `fugue` is a dependency of tsbootstrap (they
appear nowhere in `pyproject.toml`). The remaining `pkg_resources` filters are
harmless but the cited dependency chain no longer reflects the project; treat the
comment as a cleanup candidate.

## Conventions

- Package management uses `uv` with the committed `uv.lock` for reproducibility.
- Ruff is the single source for lint and formatting (line length 100,
  docstring convention numpy); see `[tool.ruff]`.
- Typing is gated by both mypy (strict-minus) and pyright (strict). The
  relaxations and their rationales are documented inline in `[tool.mypy]` and
  `[tool.pyright]`; read those before loosening a setting.
- Engine executors and preparers register themselves through the spec-type
  dispatch registry (`@register_executor` / `@register_preparer`). Engine
  submodules are imported purely for that registration side effect, which is why
  pyright's unused-import / unused-function reports are disabled for them.
