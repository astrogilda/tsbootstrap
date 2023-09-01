<div align="center">
    <h1 align="center">
        <img src="https://github.com/astrogilda/ts_bs/blob/main/ts_bs_logo.png" width="80" />
        <!--<img src="https://img.icons8.com/?size=512&id=kTuxVYRKeKEY&format=png" width="80" />-->
        <br> ts-bs
    </h1>
    <h3>◦ Generate bootstrapped samples from time-series data.</h3>
    <br>
    <p align="center">
        <img src="https://img.shields.io/badge/Markdown-000000.svg?stylee&logo=Markdown&logoColor=white" alt="Markdown" />
        <img src="https://img.shields.io/badge/Python-3776AB.svg?stylee&logo=Python&logoColor=white" alt="Python" />
        <img src="https://img.shields.io/badge/Pytest-0A9EDC.svg?stylee&logo=Pytest&logoColor=white" alt="pytest" />
        <img src="https://img.shields.io/badge/GitHub%20Actions-2088FF.svg?style&logo=GitHub-Actions&logoColor=white" alt="actions" />
        <!--
        <img src="https://img.shields.io/badge/GNU%20Bash-4EAA25.svg?style&logo=GNU-Bash&logoColor=white" alt="GNU Bash" />
        <img src="https://img.shields.io/badge/JavaScript-F7DF1E.svg?style&logo=JavaScript&logoColor=black" alt="JavaScript" />
        -->
    </p>
    <!--
    <a href="https://pypi.org/project/ts_bs/">
        <img src="https://img.shields.io/pypi/v/ts_bs?color=5D6D7E&logo=pypi" alt="pypi-version" />
    </a>
    <a href="https://pypi.org/project/ts_bs/">
        <img src="https://img.shields.io/pypi/pyversions/ts_bs?color=5D6D7E&logo=python" alt="pypi-python-version" />
    </a>
    <a href="https://pypi.org/project/ts_bs/">
        <img src="https://img.shields.io/pypi/dm/ts_bs?color=5D6D7E" alt="pypi-downloads" />
    </a>
    -->
    <img src="https://img.shields.io/github/license/eli64s/readme-ai?color=5D6D7E" alt="github-license" />
    </a>
    <img src="https://github.com/astrogilda/ts_bs/workflows/CI/badge.svg" alt="Build Status"/>
    <a href="https://codecov.io/gh/astrogilda/ts_bs"><img src="https://codecov.io/gh/astrogilda/ts_bs/branch/main/graph/badge.svg" alt="codecov"/></a>
    <a href="https://doi.org/10.5281/zenodo.8226496"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.8226496.svg" alt="DOI"/></a>
    <img src="https://img.shields.io/codeclimate/maintainability/astrogilda/ts_bs" alt="Code Quality"/>
    <img src="https://img.shields.io/github/last-commit/astrogilda/ts_bs" alt="Last Commit"/>
    <img src="https://img.shields.io/github/issues/astrogilda/ts_bs" alt="Issues"/>
    <img src="https://img.shields.io/github/issues-pr/astrogilda/ts_bs" alt="Pull Requests"/>
    <img src="https://img.shields.io/github/v/tag/astrogilda/ts_bs" alt="Tag"/>
</div>



## 📒 Table of Contents
- [📒 Table of Contents](#-table-of-contents)
- [📍 Time Series Bootstrapping](#time-series-bootstrapping)
  - [Overview](#overview)
  - [Modular Design](#modular-design)
  - [Bootstrapping Methodology](#bootstrapping-methodology)
- [📂 Project Structure](#-project-structure)
- [🧩 Modules](#-modules)
- [🚀 Getting Started](#-getting-started)
- [🗺 Roadmap](#-roadmap)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [👏 Acknowledgments](#-acknowledgments)

---


## 📍 Time Series Bootstrapping

ts_bs is a project designed to [provide a brief description of the project's purpose and functionality]. It's aimed at [target audience or use cases].


---


## 📂 Project Structure


The `ts_bs` package contains various modules that handle tasks such as bootstrapping, time series simulation, and utility functions. This modular approach ensures flexibility, extensibility, and ease of maintenance.

### `bootstrap.py`

#### `BaseTimeSeriesBootstrap` Class
- **Constructor**: Initializes the base bootstrap class with a configuration object.
- **`_fit_model` Method**: Abstract method for fitting a time series model.

#### `BlockBootstrap` Class
- **Constructor**: Inherits from `BaseBootstrap` and initializes block-specific attributes.

#### `MovingBlockBootstrap` Class

#### `StationaryBlockBootstrap` Class

#### `StationaryBlockBootstrap` Class

#### `NonOverlappingBlockBootstrap` Class

#### `BaseBlockBootstrap` Class

#### `BartlettsBootstrap` Class

#### `HammingBootstrap` Class

#### `HanningBootstrap` Class

#### `BlackmanBootstrap` Class

#### `TukeyBootstrap` Class

#### `BaseResidualBootstrap` Class
- **Constructor**: Initializes with a configuration object and sets up random number generation.
- **`_generate_samples_single_bootstrap` Method**: Abstract method for generating a single bootstrap sample.

#### `WholeResidualBootstrap` Class
- **Constructor**: Initializes with a configuration object and sets up random number generation.
- **`_generate_samples_single_bootstrap` Method**: Abstract method for generating a single bootstrap sample.

#### `BlockResidualBootstrap` Class
- **Constructor**: Initializes with a configuration object and sets up random number generation.
- **`_generate_samples_single_bootstrap` Method**: Abstract method for generating a single bootstrap sample.

#### `BaseMarkovBootstrap` Class
- **Constructor**: Inherits from `BaseResidualBootstrap` and initializes distribution-specific attributes.
- **`fit_distribution` Method**: Fits a specific distribution to the residuals.

#### `WholeMarkovBootstrap` Class
- **`_generate_samples_single_bootstrap` Method**: Generate a single bootstrap sample for the whole distribution bootstrap.

#### `BlockMarkovBootstrap` Class
- **Constructor**: Inherits from `BaseDistributionBootstrap` and `BaseBlockBootstrap`.
- **`_generate_samples_single_bootstrap` Method**: Generate a single bootstrap sample for the block distribution bootstrap.

#### `BaseBiasCorrectedBootstrap` Class
- **Constructor**: Inherits from `BaseResidualBootstrap` and initializes distribution-specific attributes.
- **`fit_distribution` Method**: Fits a specific distribution to the residuals.

#### `WholeBiasCorrectedBootstrap` Class
- **`_generate_samples_single_bootstrap` Method**: Generate a single bootstrap sample for the whole distribution bootstrap.

#### `BlockBiasCorrectedBootstrap` Class
- **Constructor**: Inherits from `BaseDistributionBootstrap` and `BaseBlockBootstrap`.
- **`_generate_samples_single_bootstrap` Method**: Generate a single bootstrap sample for the block distribution bootstrap.

#### `BaseDistributionBootstrap` Class
- **Constructor**: Inherits from `BaseResidualBootstrap` and initializes distribution-specific attributes.
- **`fit_distribution` Method**: Fits a specific distribution to the residuals.

#### `WholeDistributionBootstrap` Class
- **`_generate_samples_single_bootstrap` Method**: Generate a single bootstrap sample for the whole distribution bootstrap.

#### `BlockDistributionBootstrap` Class
- **Constructor**: Inherits from `BaseDistributionBootstrap` and `BaseBlockBootstrap`.
- **`_generate_samples_single_bootstrap` Method**: Generate a single bootstrap sample for the block distribution bootstrap.

#### `BaseSieveBootstrap` Class
- **Constructor**: Inherits from `BaseResidualBootstrap` and initializes distribution-specific attributes.
- **`fit_distribution` Method**: Fits a specific distribution to the residuals.

#### `WholeSieveBootstrap` Class
- **`_generate_samples_single_bootstrap` Method**: Generate a single bootstrap sample for the whole distribution bootstrap.

#### `BlockSieveBootstrap` Class
- **Constructor**: Inherits from `BaseDistributionBootstrap` and `BaseBlockBootstrap`.
- **`_generate_samples_single_bootstrap` Method**: Generate a single bootstrap sample for the block distribution bootstrap.


### `bootstrap_configs.py`

Contains various configuration classes for bootstrapping methods.

#### `BaseBootstrapConfig` Class
- **Constructor**: Initializes base configuration attributes like `model_type` and `refit`.

#### `BaseResidualBootstrapConfig` Class
- **Constructor**: Initializes residual bootstrap configuration attributes like `residual_method`.

#### `BaseDistributionBootstrapConfig` Class
- **Constructor**: Initializes distribution bootstrap configuration attributes like `distribution`.

### `time_series_simulator.py`

#### `TimeSeriesSimulator` Class
- **Constructor**: Initializes with a model and configuration object.
- **`simulate` Method**: Simulates time series data based on the provided model and configuration.

### `tsfit.py`

Utility functions for time series fitting.

#### `fit_ar_model` Function
- Fits an AR model to the provided time series data.

#### `fit_garch_model` Function
- Fits a GARCH model to the provided time series data.

### `markov_sampler.py`

#### `MarkovSampler` Class
- **Constructor**: Initializes with a transition matrix.
- **`sample` Method**: Samples from the Markov chain.

### `time_series_model.py`

#### `TimeSeriesModel` Class
- **Constructor**: Initializes with time series data.
- **`fit` Method**: Fits the model to the time series data.

### `ranklags.py`

Utility functions for lag ranking in time series data.

#### `rank_lags` Function
- Ranks lags based on their importance in explaining the time series.

### `block_resampler.py`

#### `BlockResampler` Class
- **Constructor**: Initializes with block lengths and data.
- **`resample` Method**: Resamples blocks of data.

### `block_length_sampler.py`

#### `BlockLengthSampler` Class
- **Constructor**: Initializes with block lengths and their probabilities.
- **`sample` Method**: Samples a block length.

### `block_generator.py`

#### `BlockGenerator` Class
- **Constructor**: Initializes with block lengths and data.
- **`generate` Method**: Generates blocks of data.

### `utils` Folder

#### `validate.py`

##### `validate_literal_type` Function
- Validates an input value against a Literal type.

##### `validate_integer` Function
- Validates if an input value is an integer.

#### `types.py`

##### `ArrayLike` Type
- Type alias for array-like structures.

##### `RandomState` Type
- Type alias for random state.

#### `odds_and_ends.py`

##### `extract_array` Function
- Extracts a numpy array from an array-like structure.

##### `extract_random_state` Function
- Extracts a numpy random state from a RandomState type.


## 🧩 Modules

<details closed><summary>root</summary>

| File                                                                                       | Summary                   |
| ---                                                                                        | ---                       |
| [setup.sh](https://github.com/astrogilda/ts_bs/blob/main/setup.sh)                         | HTTPStatus Exception: 429 |
| [commitlint.config.js](https://github.com/astrogilda/ts_bs/blob/main/commitlint.config.js) | HTTPStatus Exception: 429 |
| [CITATION.cff](https://github.com/astrogilda/ts_bs/blob/main/CITATION.cff)                 | HTTPStatus Exception: 429 |
| [CODE_OF_CONDUCT.md](https://github.com/astrogilda/ts_bs/blob/main/CODE_OF_CONDUCT.md)                 | HTTPStatus Exception: 429 |
| [CONTRIBUTING.md](https://github.com/astrogilda/ts_bs/blob/main/CITATION.md)                 | HTTPStatus Exception: 429 |


</details>

<details closed><summary>ts_bs</summary>

| File                                                                                                         | Summary                               |
| ---                                                                                                          | ---                                   |
| [block_generator.py](https://github.com/astrogilda/ts_bs/blob/main/src/ts_bs/block_generator.py)             | Generates blocks for bootstrapping.             |
| [markov_sampler.py](https://github.com/astrogilda/ts_bs/blob/main/src/ts_bs/markov_sampler.py)               | Implements sampling methods based on Markov models.             |
| [time_series_model.py](https://github.com/astrogilda/ts_bs/blob/main/src/ts_bs/time_series_model.py)         | Defines base and specific time series models.             |
| [block_length_sampler.py](https://github.com/astrogilda/ts_bs/blob/main/src/ts_bs/block_length_sampler.py)   | Samples block lengths for block bootstrapping methods.             |
| [bootstrap.py](https://github.com/astrogilda/ts_bs/blob/main/src/ts_bs/bootstrap.py)                         | Contains the implementation for different types of bootstrapping methods for time series data. |
| [bootstrap_configs.py](https://github.com/astrogilda/ts_bs/blob/main/src/ts_bs/bootstrap_configs.py)                         | Provides configuration classes for different bootstrap methods. |
| [time_series_simulator.py](https://github.com/astrogilda/ts_bs/blob/main/src/ts_bs/time_series_simulator.py) | Simulates time series data based on various models.             |
| [block_resampler.py](https://github.com/astrogilda/ts_bs/blob/main/src/ts_bs/block_resampler.py)             | Implements methods for block resampling in time series.             |
| [tsfit.py](https://github.com/astrogilda/ts_bs/blob/main/src/ts_bs/tsfit.py)                                 | Fits time series models to data.             |
| [ranklags.py](https://github.com/astrogilda/ts_bs/blob/main/src/ts_bs/ranklags.py)                                 | Provides functionalities to rank lags in a time series.             |
</details>

<details closed><summary>utils</summary>

| File                                                                                               | Summary                   |
| ---                                                                                                | ---                       |
| [types.py](https://github.com/astrogilda/ts_bs/blob/main/src/ts_bs/utils/types.py)                 | Defines custom types used across the project. |
| [validate.py](https://github.com/astrogilda/ts_bs/blob/main/src/ts_bs/utils/validate.py)           | Contains validation utilities. |
| [odds_and_ends.py](https://github.com/astrogilda/ts_bs/blob/main/src/ts_bs/utils/odds_and_ends.py) | Contains miscellaneous utility functions. |

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

2. Clone the ts_bs repository:
```sh
git clone https://github.com/astrogilda/ts_bs
```

3. Change to the project directory:
```sh
cd ts_bs
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

### 🎮 Using ts_bs

```sh
python main.py
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

## 👏 Acknowledgments

> - `ℹ️  List any resources, contributors, inspiration, etc.`

---
