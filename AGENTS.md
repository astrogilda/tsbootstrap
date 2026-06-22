# AGENTS.md

Minimal agent context for tsbootstrap. Full docs: README.md and https://tsbootstrap.readthedocs.io.

## What it is
Time series bootstrapping. One entry point: `bootstrap(X, *, method=...)` imported from `tsbootstrap`. Typed method specs live in `tsbootstrap.methods`. `diagnose(X).recommended_methods` suggests methods for a series.

## Setup and checks
- Install: `uv sync --extra dev` (add `--extra examples --extra docs` for notebooks and docs).
- Tests: `uv run pytest tests/`
- Lint and format: `uv run ruff check .` and `uv run ruff format .` (both gated in CI; run `uv run pre-commit install` once so commits gate locally).
- Types: `uv run mypy src/tsbootstrap` and `uv run pyright src/tsbootstrap`.

## Conventions
- Python 3.10+. Model-based methods (AR/ARIMA/VAR residual, sieve) need the `models` extra; the uncertainty-quantification layer `tsbootstrap.uq` needs the `uq` extra.
- No em or en dashes in prose. Conventional-commit messages, imperative mood.
