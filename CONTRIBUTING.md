# Contributing to tsbootstrap

Thank you for considering contributing to tsbootstrap. The project relies on community contributions: bug fixes, new features, documentation improvements, and ideas.

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

Before contributing, please read our [Code of Conduct](https://github.com/astrogilda/tsbootstrap/blob/main/CODE_OF_CONDUCT.md). We are committed to providing a welcoming and inclusive environment. All contributors are expected to adhere to this code.

## Getting Started

### Environment Setup

tsbootstrap uses [uv](https://docs.astral.sh/uv/) for development. Requires Python 3.10 or higher.

1. Fork and clone the repository, then change into the project root.

2. Sync the locked development environment:
```sh
uv sync --extra dev
```
uv creates an isolated virtual environment from `uv.lock` and editable-installs the package, so your changes are picked up automatically. Run tools through the environment with `uv run` (for example `uv run pytest`).

3. Install the pre-commit hooks:
```sh
uv run pre-commit install
```
The hooks run Ruff (lint and format) and the other code-quality checks on each commit.

4. Verify the installation:
```sh
uv run python -c "import tsbootstrap; print(tsbootstrap.__version__)"
```
This should print the version number without errors.

5. Run the test suite:
```sh
uv run pytest tests/
```

### Finding Your First Issue

Looking for a place to start? Check out issues labeled `good first issue` or `help wanted`. These are great for first-timers.

## Issue Creation Guidelines

### Reporting Bugs

Before reporting a bug, ensure it hasn't been reported already. If you find a new bug, create an issue providing:

- A clear title and description.
- Steps to reproduce.
- Expected behavior.
- Actual behavior.
- Screenshots or code snippets, if applicable.

### Suggesting Enhancements

Before suggesting an enhancement, please check if it has already been suggested. When creating an enhancement issue, include:

- A clear title and detailed description.
- Why this enhancement would be beneficial.
- Any potential implementation details or challenges.

### Asking Questions

First, check [GitHub Discussions](https://github.com/astrogilda/tsbootstrap/discussions) and past [issues](https://github.com/astrogilda/tsbootstrap/issues). If you cannot find an answer, open a discussion or an issue and provide as much context as possible.

## Making Contributions

### Your First Code Contribution

Unsure where to begin? Our [Contributor's Guide](https://github.com/astrogilda/tsbootstrap/wiki/Contributor's-Guide) provides step-by-step instructions on how to make your first contribution.

### Pull Request Process

1. Fork the repository and create your branch from `main`.
2. If you've added code, add tests.
3. Ensure the test suite passes.
4. Update the documentation if necessary.
5. Submit a pull request.

## Improving Documentation

Good documentation matters. To contribute:

- Update, improve, or correct documentation.
- Submit pull requests with your changes.
- Follow our [Documentation Style Guide](https://github.com/astrogilda/tsbootstrap/wiki/Documentation-Style-Guide).

## Style Guides

### Code Style

We use [Ruff](https://docs.astral.sh/ruff/) as both the linter and the code formatter to ensure consistency. This runs through the pre-commit hooks and is also checked automatically in CI when pushing code.

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/) for clear, structured commit messages.

### Documentation Style

Documentation should be clear, concise, and written in simple English. Use markdown for formatting.

## Community and Communication

Connect with other contributors and the core team through [GitHub Discussions](https://github.com/astrogilda/tsbootstrap/discussions) for questions and ideas, or [GitHub Issues](https://github.com/astrogilda/tsbootstrap/issues) for bugs and feature requests.

## Joining The Project Team

Interested in joining the core team? Email us at <sankalp.gilda@gmail.com> with your contributions and why you're interested in joining.
