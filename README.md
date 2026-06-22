<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/github/all-contributors/astrogilda/tsbootstrap?color=ee8449&style=flat-square)](#contributors)
<!-- ALL-CONTRIBUTORS-BADGE:END -->


<div align="center">
    <div style="float: left; margin-right: 20px;">
        <img src="https://github.com/astrogilda/tsbootstrap/blob/main/tsbootstrap_logo.png" width="120" />
    </div>
    <h3>Generate bootstrapped samples from time-series data. The full documentation is available <a href="https://tsbootstrap.readthedocs.io/en/latest/">here</a>.</h3>
    <div style="clear: both;"></div>
    <br>
    <p align="center">
        <img src="https://img.shields.io/badge/Markdown-000000.svg?stylee&logo=Markdown&logoColor=white" alt="Markdown" />
        <img src="https://img.shields.io/badge/Python-3776AB.svg?stylee&logo=Python&logoColor=white" alt="Python" />
        <img src="https://img.shields.io/badge/Pytest-0A9EDC.svg?stylee&logo=Pytest&logoColor=white" alt="pytest" />
        <img src="https://img.shields.io/badge/GitHub%20Actions-2088FF.svg?style&logo=GitHub-Actions&logoColor=white" alt="actions" />
    </p>
    <a href="https://arxiv.org/abs/2404.15227"><img src="https://img.shields.io/static/v1?label=arXiv&message=2404.15227&color=B31B1B&logo=arXiv" alt="preprint">
    </a>
    <a href="https://pypi.org/project/tsbootstrap/">
        <img src="https://img.shields.io/pypi/v/tsbootstrap?color=5D6D7E&logo=pypi" alt="pypi-version" />
    </a>
    <a href="https://pypi.org/project/tsbootstrap/">
        <img src="https://img.shields.io/pypi/pyversions/tsbootstrap?color=5D6D7E&logo=python" alt="pypi-python-version" />
    </a>
    <a href="https://pepy.tech/project/tsbootstrap">
        <img src="https://static.pepy.tech/badge/tsbootstrap" alt="Downloads"/>
    </a>
    <img src="https://img.shields.io/github/license/eli64s/readme-ai?color=5D6D7E" alt="github-license" />
    </a>
    <img src="https://github.com/astrogilda/tsbootstrap/workflows/CI/badge.svg" alt="Build Status"/>
    <a href="https://codecov.io/gh/astrogilda/tsbootstrap"><img src="https://codecov.io/gh/astrogilda/tsbootstrap/branch/main/graph/badge.svg" alt="codecov"/></a>
    <a href="https://doi.org/10.5281/zenodo.8226495"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.8226495.svg" alt="DOI"/></a>
    <a href="https://mybinder.org/v2/gh/astrogilda/tsbootstrap/HEAD?labpath=docs/source/tutorials/quickstart.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Launch tutorials on Binder"/></a>
    <a href="https://codeclimate.com/github/astrogilda/tsbootstrap/maintainability"><img src="https://api.codeclimate.com/v1/badges/d80a0615a8c00f31565c/maintainability" alt="Code Quality"/></a>
    <img src="https://img.shields.io/github/last-commit/astrogilda/tsbootstrap" alt="Last Commit"/>
    <img src="https://img.shields.io/github/issues/astrogilda/tsbootstrap" alt="Issues"/>
    <img src="https://img.shields.io/github/issues-pr/astrogilda/tsbootstrap" alt="Pull Requests"/>
    <img src="https://img.shields.io/github/v/tag/astrogilda/tsbootstrap" alt="Tag"/>
</div>



## 📒 Table of Contents

1. [🚀 Getting Started](#-getting-started)
2. [🧩 Modules](#-modules)
3. [🗺 Roadmap](#-roadmap)
4. [🤝 Contributing](#-contributing)
5. [📄 License](#-license)
6. [📍 Time Series Bootstrapping Methods intro](#time-series-bootstrapping)
7. [👏 Contributors](#-contributors)



---

## 🚀 Getting Started

### 🎮 Using tsbootstrap

`tsbootstrap` exposes one typed entry point, `bootstrap`, configured with a method
specification. The same call works for every method.

```python
import numpy as np
from tsbootstrap import bootstrap, MovingBlock

x = np.random.default_rng(0).standard_normal(200)

result = bootstrap(x, method=MovingBlock(block_length="auto"), n_bootstraps=999, random_state=0)

samples = result.values()      # (n_bootstraps, n) resampled series
oob = result.get_oob_mask()    # (n_bootstraps, n) out-of-bag mask
```

Choose a method spec for the structure you need (block lengths default to the
automatic Politis-White selection):

```python
from tsbootstrap import StationaryBlock, ResidualBootstrap, SieveAR, AR, ARIMA, diagnose

bootstrap(x, method=StationaryBlock(avg_block_length="auto"))

# recursive model-based bootstraps (need the model extra: uv add "tsbootstrap[models]")
bootstrap(x, method=ResidualBootstrap(model=AR(order=2)))
bootstrap(x, method=ResidualBootstrap(model=ARIMA(order=(1, 1, 1))))
bootstrap(x, method=SieveAR())

# not sure which fits? ask:
print(diagnose(x).recommended_methods)
```

Inputs can be NumPy arrays, lists, or pandas / Polars DataFrames and Series. The
result is a `BootstrapResult` carrying the samples, provenance metadata, and
out-of-bag / in-bag primitives. For the sktime ecosystem, the same methods are
also available as estimator classes (`MovingBlockBootstrap`, `ResidualBootstrap`,
…) under `tsbootstrap.adapters`.

### 📦 Installation

Requires Python 3.10 or higher.

```sh
# with uv (recommended):
uv add tsbootstrap                   # core: i.i.d. and block methods
uv add "tsbootstrap[models]"         # adds AR / ARIMA / VAR / sieve (statsmodels)

# with pip:
pip install tsbootstrap
pip install "tsbootstrap[models]"
```

The model-based methods import statsmodels lazily and raise a clear install hint if
the `models` extra is missing.

## 🧩 Modules

The package is small and layered around the functional core:

| Area | Module(s) | Role |
| --- | --- | --- |
| Public API | `api.py`, `methods.py`, `results.py`, `errors.py`, `diagnostics.py` | the `bootstrap()` entry point, typed method specs, structured results, error taxonomy, and `diagnose()` |
| Infrastructure | `rng.py`, `validation.py`, `dispatch.py`, `metadata.py` | deterministic RNG contract, input coercion (incl. the narwhals DataFrame boundary), spec → executor dispatch, method metadata |
| Block methods | `block/` | vectorized index kernels, true Politis-Romano stationary, energy-normalized tapering, PWSD block length, OOB primitives |
| Model methods | `model/`, `engines/` | model fitting, stability guards, and recursive AR/ARMA/VAR simulation |
| Ecosystem | `adapters/` | skbase / sktime estimator classes over the functional core |


## 🗺 Roadmap

This is an abridged version; for the complete and evolving list of plans and improvements, see [Issue #144](https://github.com/astrogilda/tsbootstrap/issues/144).

- **Performance and Scaling**: handling large datasets, distributed backend integration (`Dask`, `Spark`, `Ray`), profiling/optimization
- **Tuning and AutoML**: adaptive block length, adaptive resampling, evaluation based parameter selection
- **Real-time and Stream Data**: stream bootstraps, data update interface
- **Stage 2 `sktime` Integration**: evaluation module, datasets, benchmarks, sktime forecasters in bootstraps
- **API and Capability Extension**: panel/hierarchical data, exogenous data, update/stream, model state management
- **Scope Extension (TBD)**: time series augmentation, fully probabilistic models

## 🤝 Contributing

We welcome contributions.

See our [good first issues ](https://github.com/astrogilda/tsbootstrap/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
for getting started.

Below is a quick start guide to contributing.

### Developer setup

1. Fork the tsbootstrap repository

2. Clone the fork to local:
```sh
git clone https://github.com/astrogilda/tsbootstrap
```

3. In the local repository root, sync the locked development environment with uv:
```sh
uv sync --extra dev
```

4. uv creates an isolated virtual environment from `uv.lock` and editable-installs the
package, so changes to the package are reflected in your environment automatically. Run
tools through the environment with `uv run` (for example `uv run pytest`).

5. Set up git hooks and pre-commit:
```sh
# Install pre-commit hooks
pre-commit install

# Configure git to use the project's hooks
git config core.hooksPath .githooks
```

This ensures that docs requirements stay in sync with `pyproject.toml` and
other code quality checks run automatically.

### Verifying the Installation

After installation, you can verify that tsbootstrap has been installed correctly by checking its version or by trying to import it in Python:
```
python -c "import tsbootstrap; print(tsbootstrap.__version__)"
```

This command should output the version number of tsbootstrap without any errors, indicating that the installation was successful.

### Contribution workflow

Please follow these steps:

3. Create a new branch with a descriptive name (e.g., `new-feature-branch` or `bugfix-issue-123`).
```sh
git checkout -b new-feature-branch
```
4. Make changes to the project's codebase.
5. Commit your changes to your local branch with a clear commit message that explains the changes you've made.
```sh
git commit -m 'Implemented new feature.'
```
6. Push your changes to your forked repository on GitHub using the following command
```sh
git push origin new-feature-branch
```
7. Create a new pull request to the original project repository. In the pull request, describe the changes you've made and why they're necessary.
The project maintainers will review your changes and provide feedback or merge them into the main branch.

### 🧪 Running Tests

To run all tests, in your developer environment, run:

```sh
pytest tests/
```

The sktime adapter classes can be validated with sktime's estimator checks:

```python
from sktime.utils import check_estimator
from tsbootstrap.adapters import MovingBlockBootstrap

check_estimator(MovingBlockBootstrap)
```

### Contribution guide

For more detailed information on how to contribute, please refer to our [CONTRIBUTING.md](https://github.com/astrogilda/tsbootstrap/blob/main/CONTRIBUTING.md)  guide.
---

## 📄 License

This project is licensed under the `ℹ️  MIT` License. See the [LICENSE](https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/adding-a-license-to-a-repository) file for additional info.

---
## 👏 Contributors

Thanks goes to these wonderful people:

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!


---


## 📍 Time Series Bootstrapping
`tsbootstrap` implements bootstrapping methods for time series data. It generates resampled copies of univariate and multivariate series that preserve their chronological order and dependence structure.

### Overview
Traditional bootstrap methods resample observations independently, which breaks the dependence in a time series: each observation usually depends on the ones before it. Time series bootstraps resample while preserving chronological order and correlation, so the resulting uncertainty estimates stay valid under that dependence.

### Bootstrapping methodology
`tsbootstrap` resamples either the observations directly (i.i.d. and block methods) or
the innovations of a fitted model (residual and sieve methods), respecting the
chronological order and dependence structure of the data.

### Block bootstrap
Block methods resample blocks of consecutive observations to preserve short-range
dependence. The block length defaults to the automatic Politis-White (2004) selection.

- **Moving block** (`MovingBlock`): overlapping fixed-length blocks (Kunsch 1989).
- **Circular block** (`CircularBlock`): blocks wrap around the series end (Politis-Romano 1992).
- **Stationary block** (`StationaryBlock`): geometric block lengths with independent uniform
  restart points (Politis-Romano 1994).
- **Non-overlapping block** (`NonOverlappingBlock`): disjoint blocks (Carlstein 1986).
- **Tapered block** (`TaperedBlock(window=...)`): blocks weighted by an energy-normalized
  window (Bartlett, Blackman, Hamming, Hann, or Tukey; Paparoditis-Politis 2001).

### Residual bootstrap
For dependent data with a good model fit, `ResidualBootstrap(model=...)` regenerates the
series **recursively** from the fitted dynamics and resampled, centered innovations (not
`fitted + residuals`). Supported models: `AR`, `ARIMA`, and `VAR` (multivariate). A
non-stationary fit is refused (or skipped, per `stability_policy`) rather than producing
explosive paths.

### Sieve bootstrap
`SieveAR` selects an autoregressive order on the original series, then runs the AR recursion;
suited to data with autoregressive structure.

### Deferred to a later release
Markov resampling, the distribution bootstrap, GARCH/volatility models, and
frequency-domain / seasonal block methods are planned for a future version. The
statistic-preserving method has been removed.
