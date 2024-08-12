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

`tsbootstrap` provides a unified, `sklearn`-like interface to all bootstrap methods.

Example using a `MovingBlockBootstrap` - all bootstrap algorithms follow
the same interface!

```python
from tsbootstrap import MovingBlockBootstrap
import numpy as np

# Create custom time series data. While below is for univariate time series, the bootstraps can handle multivariate time series as well.
n_samples = 10
X = np.arange(n_samples)

# Instantiate the bootstrap object
n_bootstraps = 3
block_length = 3
rng = 42
mbb = MovingBlockBootstrap(
    n_bootstraps=n_bootstraps, rng=rng, block_length=block_length
)

# Generate bootstrapped samples
return_indices = False
bootstrapped_samples = mbb.bootstrap(X, return_indices=return_indices)

# Collect bootstrap samples
X_bootstrapped = []
for data in bootstrapped_samples:
    X_bootstrapped.append(data)

X_bootstrapped = np.array(X_bootstrapped)
```

### 📦 Installation and Setup

``tsbootstrap`` is installed via ``pip``, either from PyPI or locally.

#### ✔️ Prerequisites

- Python (3.9 or higher)
- `pip` (latest version recommended), plus suitable environment manager (`venv`, `conda`)

You can also consider using ``uv`` to speed up environment setu.

#### Installing from PyPI

To install the latest release of `tsbootstrap` directly from PyPI, run:

```sh
pip install tsbootstrap
```

To install with all optional dependencies:

```
pip install "tsbootstrap[all_extras]"
```
---

Bootstrap algorithms manage their own dependencies - if an extra is needed but not
present, the object will raise this at construction.

## 🧩 Modules
The `tsbootstrap` package contains various modules that handle tasks such as bootstrapping, time series simulation, and utility functions. This modular approach ensures flexibility, extensibility, and ease of maintenance.


<details closed><summary>root</summary>

| File                                                                                       | Summary                   |
| ---                                                                                        | ---                       |
| [setup.sh](https://github.com/astrogilda/tsbootstrap/blob/main/setup.sh)                         | Shell script for initial setup and environment configuration. |
| [commitlint.config.js](https://github.com/astrogilda/tsbootstrap/blob/main/commitlint.config.js) | Configuration for enforcing conventional commit messages. |
| [CITATION.cff](https://github.com/astrogilda/tsbootstrap/blob/main/CITATION.cff)                 | Citation metadata for the project. |
| [CODE_OF_CONDUCT.md](https://github.com/astrogilda/tsbootstrap/blob/main/CODE_OF_CONDUCT.md)                 | Guidelines for community conduct and interactions. |
| [CONTRIBUTING.md](https://github.com/astrogilda/tsbootstrap/blob/main/CONTRIBUTING.md)                 | Instructions for contributing to the project. |
| [.codeclimate.yml](https://github.com/astrogilda/tsbootstrap/blob/main/.codeclimate.yml)                 | Configuration for Code Climate quality checks. |
| [.gitignore](https://github.com/astrogilda/tsbootstrap/blob/main/.gitignore)                 | Specifies files and folders to be ignored by Git. |
| [.pre-commit-config.yaml](https://github.com/astrogilda/tsbootstrap/blob/main/.pre-commit-config.yaml)                 | Configuration for pre-commit hooks. |
| [poetry.toml](https://github.com/astrogilda/tsbootstrap/blob/main/poetry.toml)                 | Configuration file for Poetry package management. |
| [tsbootstrap_logo.png](https://github.com/astrogilda/tsbootstrap/blob/main/tsbootstrap_logo.png)                 | Project logo image. |

</details>



</details>

<details closed><summary>tsbootstrap</summary>

| File                                                                                                         | Summary                               |
| ---                                                                                                          | ---                                   |
| [block_generator.py](https://github.com/astrogilda/tsbootstrap/blob/main/src/tsbootstrap/block_generator.py)             | Generates blocks for bootstrapping.             |
| [markov_sampler.py](https://github.com/astrogilda/tsbootstrap/blob/main/src/tsbootstrap/markov_sampler.py)               | Implements sampling methods based on Markov models.             |
| [time_series_model.py](https://github.com/astrogilda/tsbootstrap/blob/main/src/tsbootstrap/time_series_model.py)         | Defines base and specific time series models.             |
| [block_length_sampler.py](https://github.com/astrogilda/tsbootstrap/blob/main/src/tsbootstrap/block_length_sampler.py)   | Samples block lengths for block bootstrapping methods.             |
| [base_bootstrap.py](https://github.com/astrogilda/tsbootstrap/blob/main/src/tsbootstrap/bootstrap.py)                         | Contains the implementation for different types of base, abstract bootstrapping classes for time series data. |
| [base_bootstrap_configs.py](https://github.com/astrogilda/tsbootstrap/blob/main/src/tsbootstrap/bootstrap_configs.py)                         | Provides configuration classes for different base, abstract bootstrapping classes. |
| [block_bootstrap.py](https://github.com/astrogilda/tsbootstrap/blob/main/src/tsbootstrap/bootstrap.py)                         | Contains the implementation for different types of block bootstrapping methods for time series data. |
| [block_bootstrap_configs.py](https://github.com/astrogilda/tsbootstrap/blob/main/src/tsbootstrap/bootstrap_configs.py)                         | Provides configuration classes for different block bootstrapping methods. |
| [bootstrap.py](https://github.com/astrogilda/tsbootstrap/blob/main/src/tsbootstrap/bootstrap.py)                         | Contains the implementation for different types of bootstrapping methods for time series data, including residual, distribution, markov, statistic-preserving, and sieve. |
| [time_series_simulator.py](https://github.com/astrogilda/tsbootstrap/blob/main/src/tsbootstrap/time_series_simulator.py) | Simulates time series data based on various models.             |
| [block_resampler.py](https://github.com/astrogilda/tsbootstrap/blob/main/src/tsbootstrap/block_resampler.py)             | Implements methods for block resampling in time series.             |
| [tsfit.py](https://github.com/astrogilda/tsbootstrap/blob/main/src/tsbootstrap/tsfit.py)                                 | Fits time series models to data.             |
| [ranklags.py](https://github.com/astrogilda/tsbootstrap/blob/main/src/tsbootstrap/ranklags.py)                                 | Provides functionalities to rank lags in a time series.             |
</details>

<details closed><summary>utils</summary>

| File                                                                                               | Summary                   |
| ---                                                                                                | ---                       |
| [types.py](https://github.com/astrogilda/tsbootstrap/blob/main/src/tsbootstrap/utils/types.py)                 | Defines custom types used across the project. |
| [validate.py](https://github.com/astrogilda/tsbootstrap/blob/main/src/tsbootstrap/utils/validate.py)           | Contains validation utilities. |
| [odds_and_ends.py](https://github.com/astrogilda/tsbootstrap/blob/main/src/tsbootstrap/utils/odds_and_ends.py) | Contains miscellaneous utility functions. |

</details>


## 🗺 Roadmap

This is an abridged version; for the complete and evolving list of plans and improvements, see [Issue #144](https://github.com/astrogilda/tsbootstrap/issues/144).

- **Performance and Scaling**: handling large datasets, distributed backend integration (`Dask`, `Spark`, `Ray`), profiling/optimization
- **Tuning and AutoML**: adaptive block length, adaptive resampling, evaluation based parameter selection
- **Real-time and Stream Data**: stream bootstraps, data update interface
- **Stage 2 `sktime` Integration**: evaluation module, datasets, benchmarks, sktime forecasters in bootstraps
- **API and Capability Extension**: panel/hierarchical data, exogenous data, update/stream, model state management
- **Scope Extension (TBD)**: time series augmentation, fully probabilistic models

## 🤝 Contributing

Contributions are always welcome!

See our [good first issues ](https://github.com/astrogilda/tsbootstrap/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
for getting started.

Below is a quick start guide to contributing.

### Developer setup

1. Fork the tsbootstrap repository

2. Clone the fork to local:
```sh
git clone https://github.com/astrogilda/tsbootstrap
```

3. In the local repository root, set up a python environment, e.g., `venv` or `conda`.


4. Editable install via `pip`, including developer dependencies:
```
pip install -e ".[dev]"
```

The editable install ensures that changes to the package are reflected in
your environment.

### Verifying the Installation

After installation, you can verify that tsbootstrap has been installed correctly by checking its version or by trying to import it in Python:
```
python -c "import tsbootstrap; print(tsbootstrap.__version__)"
```

This command should output the version number of tsbootstrap without any errors, indicating that the installation was successful.

That's it! You are now set up and ready to go. You can start using tsbootstrap for your time series bootstrapping needs.

### Contribution workflow

Contributions are always welcome! Please follow these steps:

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

Individual bootstrap algorithms can be tested as follows:

```python
from tsbootstrap.utils import check_estimator

check_estimator(my_bootstrap_algo)
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
`tsbootstrap` is a comprehensive project designed to implement an array of bootstrapping techniques specifically tailored for time series data. This project is targeted towards data scientists, statisticians, economists, and other professionals or researchers who regularly work with time series data and require robust methods for generating bootstrapped copies of univariate and multivariate time series data.

### Overview
Time series bootstrapping is a nuanced resampling method that is applied to time-dependent data. Traditional bootstrapping methods often assume independence between data points, which is an assumption that does not hold true for time series data where a data point is often dependent on previous data points. Time series bootstrapping techniques respect the chronological order and correlations of the data, providing more accurate estimates of uncertainty or variability.

### Bootstrapping Methodology
The `tsbootstrap` project offers a diverse set of bootstrapping techniques that can be applied to either the entire input time series (classes prefixed with `Whole`), or after partitioning the data into blocks (classes prefixed with `Block`). These methodologies can be applied directly to the raw input data or to the residuals obtained after fitting one of the five statistical models defined in `time_series_model.py` (classes with `Residual` in their names).

### Block Bootstrap
Block Bootstrap is a prevalent approach in time series bootstrapping. It involves resampling blocks of consecutive data points, thus respecting the internal structures of the data. There are several techniques under Block Bootstrap, each with its unique approach. `tsbootstrap` provides highly flexible block bootstrapping, allowing the user to specify the block length sampling, block generation, and block resampling strategies. For additional details, refer to `block_length_sampler.py`, `block_generator.py`, and `block_resampler.py`.

The Moving Block Bootstrap, Circular Block Bootstrap, Stationary Block Bootstrap, and NonOverlapping Block Bootstrap methods are all variations of the Block Bootstrap that use different methods to sample the data, maintaining various types of dependencies.

Bartlett's, Blackman's, Hamming's, Hanning's, and Tukey's Bootstrap methods are specific implementations of the Block Bootstrap that use different window shapes to taper the data, reducing the influence of data points far from the center. In `tsbootstrap`, these methods inherit from `MovingBlockBootstrap`, but can easily be modified to inherit from any of the other three base block bootstrapping classes.

Each method comes with its distinct strengths and weaknesses. The choice of method should be based on the characteristics of the data and the specific requirements of the analysis.

#### (i) Moving Block Bootstrap
This method is implemented in `MovingBlockBootstrap` and is used for time series data where blocks of data are resampled to maintain the dependency structure within the blocks. It's useful when the data has dependencies that need to be preserved. It's not recommended when the data does not have any significant dependencies.

#### (ii) Circular Block Bootstrap
This method is implemented in `CircularBlockBootstrap` and treats the data as if it is circular (the end of the data is next to the beginning of the data). It's useful when the data is cyclical or seasonal in nature. It's not recommended when the data does not have a cyclical or seasonal component.

#### (iii) Stationary Block Bootstrap
This method is implemented in `StationaryBlockBootstrap` and randomly resamples blocks of data with block lengths that follow a geometric distribution. It's useful for time series data where the degree of dependency needs to be preserved, and it doesn't require strict stationarity of the underlying process. It's not recommended when the data has strong seasonality or trend components which violate the weak dependence assumption.

#### (iv) NonOverlapping Block Bootstrap
 This method is implemented in `NonOverlappingBlockBootstrap` and resamples blocks of data without overlap. It's useful when the data has dependencies that need to be preserved and when overfitting is a concern. It's not recommended when the data does not have any significant dependencies or when the introduction of bias due to non-overlapping selection is a concern.

#### (v) Bartlett's Bootstrap
 Bartlett's method is a time series bootstrap method that uses a window or filter that tapers off as you move away from the center of the window. It's useful when you have a large amount of data and you want to reduce the influence of the data points far away from the center. This method is not advised when the tapering of data points is not desired or when the dataset is small as the tapered data points might contain valuable information. It is implemented in `BartlettsBootstrap`.

#### (vi) Blackman Bootstrap
Similar to Bartlett's method, Blackman's method uses a window that tapers off as you move away from the center of the window. The key difference is the shape of the window (Blackman window has a different shape than Bartlett). It's useful when you want to reduce the influence of the data points far from the center with a different window shape. It's not recommended when the dataset is small or tapering of data points is not desired. It is implemented in `BlackmanBootstrap`.

#### (vii) Hamming Bootstrap
 Similar to the Bartlett and Blackman methods, the Hamming method uses a specific type of window function. It's useful when you want to reduce the influence of the data points far from the center with the Hamming window shape. It's not recommended for small datasets or when tapering of data points is not desired. It is implemented in `HammingBootstrap`.

#### (viii) Hanning Bootstrap
This method also uses a specific type of window function. It's useful when you want to reduce the influence of the data points far from the center with the Hanning window shape. It's not recommended for small datasets or when tapering of data points is not desired. It is implemented in `HanningBootstrap`.

#### (ix) Tukey Bootstrap
Similar to the Bartlett, Blackman, Hamming, and Hanning methods, the Tukey method uses a specific type of window function. It's useful when you want to reduce the influence of the data points far from the center with the Tukey window shape. It's not recommended for small datasets or when tapering of data points is not desired. It is implemented in `TukeyBootstrap`.

### Residual Bootstrap
Residual Bootstrap is a method designed for time series data where a model is fit to the data, and the residuals (the difference between the observed and predicted data) are bootstrapped. It's particularly useful when a good model fit is available for the data. However, it's not recommended when a model fit is not available or is poor. `tsbootstrap` provides four time series models to fit to the input data -- `AutoReg`, `ARIMA`, `SARIMA`, and `VAR` (for multivariate input time series data). For more details, refer to `time_series_model.py` and `tsfit.py`.

### Statistic-Preserving Bootstrap
Statistic-Preserving Bootstrap is a unique method designed to generate bootstrapped time series data while preserving a specific statistic of the original data. This method can be beneficial in scenarios where it's important to maintain the original data's characteristics in the bootstrapped samples. It is implemented in `StatisticPreservingBootstrap`.

### Distribution Bootstrap
Distribution Bootstrap generates bootstrapped samples by fitting a distribution to the residuals and then generating new residuals from the fitted distribution. The new residuals are then added to the fitted values to create the bootstrapped samples. This method is based on the assumption that the residuals follow a specific distribution (like Gaussian, Poisson, etc). It's not recommended when the distribution of residuals is unknown or hard to determine. It is implemented in `DistributionBootstrap`.

### Markov Bootstrap
Markov Bootstrap is used for bootstrapping time series data where the residuals of the data are presumed to follow a Markov process. This method is especially useful in scenarios where the current residual primarily depends on the previous one, with little to no dependency on residuals from further in the past. Markov Bootstrap technique is designed to preserve this dependency structure in the bootstrapped samples, making it particularly valuable for time series data that exhibits Markov properties. However, it's not advisable when the residuals of the time series data exhibit long-range dependencies, as the Markov assumption of limited dependency may not hold true. It is implemented in `MarkovBootstrap`. See `markov_sampler.py` for implementation details.

### Sieve Bootstrap
Sieve Bootstrap is designed for handling dependent data, where the residuals of the time series data follow an autoregressive process. This method aims to preserve and simulate the dependencies inherent in the original data within the bootstrapped samples. It operates by approximating the autoregressive process ofthe residuals using a finite order autoregressive model. The order of the model is determined based on the data, and the residuals are then bootstrapped. The Sieve Bootstrap technique is particularly valuable for time series data that exhibits autoregressive properties. However, it's not advisable when the residuals of the time series data do not follow an autoregressive process. It is implemented in `SieveBootstrap`. See `time_series_simulator.py` for implementations details.
