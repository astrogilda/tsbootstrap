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

# Create custom time series data

n_samples = 1000

X = np.random.normal(0, 1, (n_samples, 1))

# Instantiate the bootstrap object
mbb = MovingBlockBootstrap(n_bootstraps=100, rng=42, block_length=10)

# Generate the generator for 1000 bootstrapped samples
bootstrapped_samples = mbb.bootstrap(X)
# this is a generator, yielding np.arrays of the same shape as X
# assumed bootstrapped from the same generative distribution

# some bootstraps can use exogeneous data
# (all take the argument for API uniformity, but only some use it)
from tsbootstrap import BlockResidualBootstrap

resid_bootstrap = BlockResidualBootstrap(n_bootstraps=42)
y = np.random.normal(0, 1, (n_samples, 10))
bootstrapped_samples_with_exog = mbb.bootstrap(X, y=y)
```

### 📦 Installation and Setup

``tsbootstrap`` is installed via ``pip``, either from PyPI or locally.

#### ✔️ Prerequisites

- Python (3.8 or higher)
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

> - [ ] `ℹ️  Task 1: in distributionbootstrap, allow mixture of distributions`
> - [ ] `ℹ️  Task 2: allow fractional block_length`
> - [ ] `ℹ️  Task 3: enable multi-processing`
> - [ ] `ℹ️  Task 4: test -- for biascorrectblockbootstrap, see if the statistic on the bootstrapped sample is close to the statistic on the original sample`


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
pip install -e .[dev]
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
<<<<<<< HEAD
=======

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

---


## 🚀 Getting Started

### ✔️ Prerequisites

Before you begin, ensure that you have the following prerequisites installed:
- Python (3.8 or higher)

One of the two sets below; the former is conventional and more widely used, while the latter provides significant speed benefits:
- pip (latest version recommended)
- virtualenv (recommended for local installation)
OR
- uv (see https://github.com/astral-sh/uv for installation instructions)

### 📦 Installation and Setup

This project uses `pyproject.toml` for managing dependencies and package settings. You can install the package and its dependencies directly using pip, either from PyPI or locally.

#### Installing from PyPI

All the below installations can be **significantly** sped up by using `uv` instead of `pip`. The substitution is simple -- instead of `pip install tsbootstrap`, simply run `uv pip install tsbootstrap`.

To install the latest release of `tsbootstrap` directly from PyPI, run:

```sh
pip install tsbootstrap
```

To include optional dependencies, you can use:

```
pip install "tsbootstrap[all_extras]"
```

To include dev dependencies, you can use:

```
pip install "tsbootstrap[dev]"
```

To include docs dependencies, you can use:

```
pip install "tsbootstrap[docs]"
```

To include **all** dependencies, you can use:

```
pip install "tsbootstrap[all_extras,dev,docs]"
```

#### Installing locally

1. Clone the tsbootstrap repository:
```sh
git clone https://github.com/astrogilda/tsbootstrap
```

2. Change to the project directory:
```sh
cd tsbootstrap
```

3.1 Create a virtual environment
```
python -m venv venv
```
or
```
uv venv venv
```

3.2 Activate the virtual environment
- On Windows
```
venv\Scripts\activate
```
- On Unix or MacOS
```
source venv/bin/activate
```

4. Install via `pip` or `uv`
```
pip install .
```
or
```
uv pip install .
```

Similarly, to include optional dependencies during local installation:
```
pip install .[all_extras]
```
or
```
uv pip install .[all_extras]
```

#### `uv` vs `pip`
`uv` is significantly faster than `pip`, both when creating the virtual environment, and installing packages. See the below figure, which demonstrates gains on the order of 10.
![Significant differences in installation times](uv_vs_pip.jpg)

#### Verifying the Installation
After installation, you can verify that tsbootstrap has been installed correctly by checking its version or by trying to import it in Python:
```
python -c "import tsbootstrap; print(tsbootstrap.__version__)"
```

This command should output the version number of tsbootstrap without any errors, indicating that the installation was successful.

That's it! You are now set up and ready to go. You can start using tsbootstrap for your time series bootstrapping needs.

### 🎮 Using tsbootstrap

Here's a basic example using the Moving Block Bootstrap method:

```python
from tsbootstrap import MovingBlockBootstrap
import numpy as np

np.random.seed(0)

# Create custom time series data

n_samples = 1000

y = np.random.normal(0, 1, n_samples).cumsum()

x1 = np.arange(1, n_samples + 1).reshape(-1, 1)
x2 = np.random.normal(0, 1, (n_samples, 1))
exog = np.concatenate([x1, x2], axis=1)

# Instantiate the bootstrap object
mbb_config = MovingBlockBootstrapConfig(
    n_bootstraps=1000, rng=42, block_length=10
)
mbb = MovingBlockBootstrap(n_bootstraps=1000, rng=42, block_length=10)

# Generate the generator for 1000 bootstrapped samples
bootstrapped_samples = bootstrap.bootstrap(n=1000)
```

### 🧪 Running Tests
```sh
pytest tests/
```

---


## 🗺 Roadmap

### Performance and Scaling
- **Memory Optimization:** Use `numpy.memmap` for handling large datasets within simulation methods, allowing parts of the data to be loaded on demand, reducing memory overhead. Opt for in-place operations `(+=, *=)` in numerical computations to avoid unnecessary data duplication and to minimize peak memory usage.
- **Profiling for Optimization:** Utilize Python profiling tools such as `cProfile` and `memray` to identify performance bottlenecks. Analyze time complexity of critical functions and optimize by either improving algorithmic approaches or by utilizing more efficient data structures.
- **Big Data Integration:** Integrate with distributed computing frameworks like Apache Spark or Dask by adapting the `time_series_simulator.py` module to partition data processing across multiple nodes.

### Tuning and Automation
- **Adaptive Block Length:** Develop algorithms in `block_resampler.py` that adjust block sizes dynamically based on the autocorrelation properties of the input data, optimizing the balance between bias and variance in bootstrap samples.
- **Fractional Block Length:** Modify the block length handling logic to accept and correctly process fractional lengths, providing finer granularity in block resampling.
- **Adaptive Resampling:** Implement adaptive resampling methods that modify the sampling technique based on real-time analysis of the dataset’s variance and skewness to improve the representativeness of bootstrap samples.
- **Feedback-Driven Accuracy:** Establish feedback loops in `bootstrap.py` that compare statistical properties of the original and bootstrapped datasets and iteratively refine the resampling process to minimize errors.

### Real-Time and Stream Data
- **Real-Time Bootstrapping:** Enable `bootstrap.py` to process data in real-time by incorporating event-driven programming or reactive frameworks that handle data streams efficiently.

### Enhanced Composability with `sktime`
- **Evaluation and Comparison Tools:** Develop a standardized evaluation module within `tsbootstrap` to leverage `sktime`'s comparison metrics (MASE, MAP, etc.), enabling detailed performance analytics between bootstrapped and original time series data.
- **Shared Datasets and Benchmarks:** Establish a shared repository of time series datasets commonly used in both `tsbootstrap` and `sktime`. Then, create a suite of benchmark tests that automatically apply both resampling methods from `tsbootstrap` and forecasters from `sktime` to these datasets, allowing users to directly compare methodologies under identical conditions.
- **Documentation and Examples:** Create comprehensive documentation and tutorials that illustrate how `tsbootstrap` can be integrated with `sktime`, offering practical examples and best practices in leveraging the combined strengths of both libraries.
- **Integration with Arbitrary `sktime` Forecasters:** Enable the use of any `sktime` forecaster in forecaster-based bootstraps within `tsbootstrap`.
- **Distribution and Sampler-like Object:** Use `tsbootstrap` bootstraps to create a distribution or sampler-like object, enhancing the probabilistic forecasting capabilities.

### API Extension
- **DataFrame Support:** Adapt core functionalities to accept `pd.DataFrame` inputs, ensuring outputs maintain the original index and columns to seamlessly integrate with pandas workflows.
- **Handling Panels and Hierarchical Data:** Extend API to support panel data and hierarchical time series, broadening the applicability of the library.
- **Exogenous Data Integration:** Enhance handling of exogenous variables within bootstraps to support complex forecasting models.
- **Update and Streaming Capabilities:** Develop methods to update and stream data through the bootstrapping process, facilitating real-time data analysis.
- **Model State Management:** Differentiate between fittable or pretrained models within the API, providing users with flexible model deployment options.

### Miscellaneous
- **Time Series Augmentation:** Explore and implement time series augmentation techniques to enrich training datasets and improve model robustness.
- **Full Proba Models:** Develop full probabilistic models that can be sampled from, expanding the predictive capabilities of `tsbootstrap`.

## 🤝 Contributing

Contributions are always welcome! Please follow these steps:
1. Fork the project repository. This creates a copy of the project on your account that you can modify without affecting the original project.
2. Clone the forked repository to your local machine using a Git client like Git or GitHub Desktop.
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
>>>>>>> d3d9a15 (added to roadmap based on franzs recs)
