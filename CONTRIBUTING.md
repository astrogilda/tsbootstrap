# Contributing to tsbootstrap

Contributions are welcome: bug fixes, new features, and documentation improvements.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
   - [Environment Setup](#environment-setup)
   - [Finding Your First Issue](#finding-your-first-issue)
3. [Issue Creation Guidelines](#issue-creation-guidelines)
   - [Reporting Bugs](#reporting-bugs)
   - [Suggesting Enhancements](#suggesting-enhancements)
   - [Asking Questions](#asking-questions)
4. [Making Contributions](#making-contributions)
   - [Your First Code Contribution](#your-first-code-contribution)
   - [Pull Request Process](#pull-request-process)
5. [Improving Documentation](#improving-documentation)
6. [Style Guides](#style-guides)
   - [Code Style](#code-style)
   - [Commit Messages](#commit-messages)
   - [Documentation Style](#documentation-style)
7. [Community and Communication](#community-and-communication)
8. [Joining The Project Team](#joining-the-project-team)

## Code of Conduct

Contributors must follow the [Code of Conduct](https://github.com/astrogilda/tsbootstrap/blob/main/CODE_OF_CONDUCT.md).

## Getting Started

### Environment Setup

tsbootstrap uses [uv](https://docs.astral.sh/uv/) for development. Requires Python 3.10 or higher.

1. Fork and clone the repository, then change into the project root.

2. Sync the locked development environment:
```sh
uv sync --extra dev
```
uv sync builds the locked virtual environment with an editable install, so your edits take effect immediately. Invoke tools with `uv run`, e.g. `uv run pytest`.

3. Install the hooks:
```sh
uv run pre-commit install
scripts/install-hooks.sh
```
`pre-commit install` runs Ruff (lint and format) on each commit. `scripts/install-hooks.sh`
points the clone at `.githooks/`, enabling the version-controlled pre-push gate that runs
Ruff, mypy, and pyright with the same `--extra dev --extra mcp` extras CI uses, so a type
error involving an optional dependency is caught locally instead of on the pull request.
Skip the pre-push gate in an emergency with `PRE_PUSH_SKIP=1 git push`.

To reproduce the SonarCloud scan locally before pushing, set `SONAR_TOKEN` and run
`scripts/sonar-local.sh` (optionally passing a coverage XML path).

4. Verify the installation:
```sh
uv run python -c "import tsbootstrap; print(tsbootstrap.__version__)"
```
This prints the installed version.

5. Run the test suite:
```sh
uv run pytest tests/
```

### Dependency Policy (cooldown and pinned uv)

The project enforces a 7-day dependency cooldown: `exclude-newer = "7 days"` in
`[tool.uv]` makes every `uv lock` refuse distributions published within the last week,
so a compromised release must survive a week of public scrutiny before it can enter the
lockfile. Two practical consequences:

- `uv lock --upgrade` will select the newest release that is at least 7 days old, not
  the absolute newest. This is intentional.
- To adopt an urgent security fix younger than the window, do not lift the cooldown.
  Add a scoped override and remove it once the release ages past the window:
  ```toml
  [tool.uv]
  exclude-newer-package = { somepackage = "2026-07-10T00:00:00Z" }
  ```

The uv version itself is pinned via `required-version` in `[tool.uv]`; CI's setup-uv
reads it instead of installing the latest release. If your local uv is older, upgrade to
the pinned version (`uv self update <version>`, or your package manager's equivalent).
Bumps to the pin belong in their own commit with a green `uv lock --check`.

### Finding Your First Issue

New contributors can pick up an issue labeled `good first issue` or `help wanted`.

## Issue Creation Guidelines

### Reporting Bugs

When reporting a new bug, first check it is not already reported, then open an issue with:

- A clear title and description.
- Steps to reproduce.
- Expected behavior.
- Actual behavior.
- Screenshots or code snippets, if applicable.

### Suggesting Enhancements

When suggesting an enhancement, include:

- A clear title and detailed description.
- Why this enhancement would be beneficial.
- Any potential implementation details or challenges.

### Asking Questions

Ask in [GitHub Discussions](https://github.com/astrogilda/tsbootstrap/discussions) or open an [issue](https://github.com/astrogilda/tsbootstrap/issues), with as much context as you can give.

## Making Contributions

### Your First Code Contribution

The [Contributor's Guide](https://github.com/astrogilda/tsbootstrap/wiki/Contributor's-Guide) has step-by-step instructions for a first contribution.

### Pull Request Process

1. Fork the repository and create your branch from `main`.
2. If you've added code, add tests.
3. Ensure the test suite passes.
4. Update the documentation if necessary.
5. Submit a pull request.

## Improving Documentation

To update the documentation:

- Update, improve, or correct documentation.
- Submit pull requests with your changes.
- Follow our [Documentation Style Guide](https://github.com/astrogilda/tsbootstrap/wiki/Documentation-Style-Guide).

## Style Guides

### Code Style

[Ruff](https://docs.astral.sh/ruff/) handles linting and formatting, via pre-commit hooks and in CI.

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/) for clear, structured commit messages.

### Documentation Style

Write documentation in plain English. Use markdown for formatting.

## Community and Communication

Use [GitHub Discussions](https://github.com/astrogilda/tsbootstrap/discussions) for questions and [GitHub Issues](https://github.com/astrogilda/tsbootstrap/issues) for bugs and features.

## Joining The Project Team

To join the core team, email <sankalp.gilda@gmail.com> with your contributions.
