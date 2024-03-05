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
    <a href="https://doi.org/10.5281/zenodo.8226496"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.8226496.svg" alt="DOI"/></a>
    <a href="https://codeclimate.com/github/astrogilda/tsbootstrap/maintainability"><img src="https://api.codeclimate.com/v1/badges/d80a0615a8c00f31565c/maintainability" alt="Code Quality"/></a>
    <img src="https://img.shields.io/github/last-commit/astrogilda/tsbootstrap" alt="Last Commit"/>
    <img src="https://img.shields.io/github/issues/astrogilda/tsbootstrap" alt="Issues"/>
    <img src="https://img.shields.io/github/issues-pr/astrogilda/tsbootstrap" alt="Pull Requests"/>
    <img src="https://img.shields.io/github/v/tag/astrogilda/tsbootstrap" alt="Tag"/>
</div>



## 📒 Table of Contents
1. [📍 Time Series Bootstrapping](#time-series-bootstrapping)
    - [Overview](#overview)
    - [Bootstrapping Methodology](#bootstrapping-methodology)
    - [Block Bootstrap](#block-bootstrap)
        - [Moving Block Bootstrap](#moving-block-bootstrap)
        - [Circular Block Bootstrap](#circular-block-bootstrap)
        - [Stationary Block Bootstrap](#stationary-block-bootstrap)
        - [NonOverlapping Block Bootstrap](#nonoverlapping-block-bootstrap)
        - [Bartletts Bootstrap](#bartletts-bootstrap)
        - [Blackman Bootstrap](#blackman-bootstrap)
        - [Hamming Bootstrap](#hamming-bootstrap)
        - [Hanning Bootstrap](#hanning-bootstrap)
        - [Tukey Bootstrap](#tukey-bootstrap)
    - [Residual Bootstrap](#residual-bootstrap)
    - [Bias Corrected Bootstrap](#bias-corrected-bootstrap)
    - [Distribution Bootstrap](#distribution-bootstrap)
    - [Markov Bootstrap](#markov-bootstrap)
    - [Sieve Bootstrap](#sieve-bootstrap)
3. [🧩 Modules](#-modules)
4. [🚀 Getting Started](#-getting-started)
5. [🗺 Roadmap](#-roadmap)
6. [🤝 Contributing](#-contributing)
7. [📄 License](#-license)
8. [👏 Acknowledgments](#-acknowledgments)


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

## 🧩 Modules
The `tsbootstrap` package contains various modules that handle tasks such as bootstrapping, time series simulation, and utility functions. This modular approach ensures flexibility, extensibility, and ease of maintenance.


<details closed><summary>root</summary>

| File                                                                                       | Summary                   |
| ---                                                                                        | ---                       |
| [setup.sh](https://github.com/astrogilda/tsbootstrap/blob/main/setup.sh)                         | Shell script for initial setup and environment configuration. |
| [commitlint.config.js](https://github.com/astrogilda/tsbootstrap/blob/main/commitlint.config.js) | Configuration for enforcing conventional commit messages. |
| [CITATION.cff](https://github.com/astrogilda/tsbootstrap/blob/main/CITATION.cff)                 | Citation metadata for the project. |
| [CODE_OF_CONDUCT.md](https://github.com/astrogilda/tsbootstrap/blob/main/CODE_OF_CONDUCT.md)                 | Guidelines for community conduct and interactions. |
| [CONTRIBUTING.md](https://github.com/astrogilda/tsbootstrap/blob/main/CITATION.md)                 | Instructions for contributing to the project. |
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
> - `ℹ️ Requirement 1`
> - `ℹ️ Requirement 2`
> - `ℹ️ ...`

### 📦 Installation and Setup

This project comes with a `setup.sh` script to ease the setup process. The script will create a new Python virtual environment, install the necessary dependencies, and handle some version-specific installations.

Here are the steps to follow:

1. Ensure that you have Python, Poetry, and Bash installed on your system. If not, you can install them using the links below:
    - [Python](https://www.python.org/downloads/)
    - [Poetry](https://python-poetry.org/docs/#installation)
    - [Bash](https://www.gnu.org/software/bash/)

2. Clone the tsbootstrap repository:
```sh
git clone https://github.com/astrogilda/tsbootstrap
```

3. Change to the project directory:
```sh
cd tsbootstrap
```

4. Make the `setup.sh` script executable:
```sh
chmod +x setup.sh
```

5. Run the `setup.sh` script:
```sh
./setup.sh
```

The `setup.sh` script sets up a Python environment using Poetry, locks and installs the necessary dependencies, and installs `dtaidistance` if the Python version is 3.9 or lower.

6. Activate the python shell:
```sh
poetry shell
```

That's it! You are now set up and ready to go.

### 🎮 Using tsbootstrap

Here's a basic example using the Moving Block Bootstrap method:

```python
from tsbootstrap import MovingBlockBootstrap, MovingBlockBootstrapConfig
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
mbb = MovingBlockBootstrap(config=mbb_config)

# Generate the generator for 1000 bootstrapped samples
bootstrapped_samples = bootstrap.bootstrap(n=1000)
```

### 🧪 Running Tests
```sh
pytest tests/
```

---


## 🗺 Roadmap

> - [ ] `ℹ️  Task 1: in distributionbootstrap, allow mixture of distributions`
> - [ ] `ℹ️  Task 2: allow fractional block_length`
> - [ ] `ℹ️  Task 3: enable multi-processing`
> - [ ] `ℹ️  Task 4: test -- for biascorrectblockbootstrap, see if the statistic on the bootstrapped sample is close to the statistic on the original sample`


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
