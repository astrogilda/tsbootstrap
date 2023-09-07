from enum import unique
from typing import Optional, Union, get_args
from unittest.mock import patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import (
    booleans,
    dictionaries,
    floats,
    integers,
    just,
    lists,
    none,
    one_of,
    sampled_from,
    text,
    tuples,
)
from ts_bs.base_bootstrap_configs import (
    BaseResidualBootstrapConfig,
)
from ts_bs.block_bootstrap import BLOCK_BOOTSTRAP_TYPES_DICT
from ts_bs.block_bootstrap_configs import (
    BaseBlockBootstrapConfig,
    BlockBootstrapConfig,
)
from ts_bs.bootstrap import (
    BlockResidualBootstrap,
    WholeResidualBootstrap,
)
from ts_bs.tsfit import TSFitBestLag
from ts_bs.utils.types import ModelTypesWithoutArch

# The shape is a strategy generating tuples (num_rows, num_columns)
X_shape = tuples(
    integers(min_value=20, max_value=100), integers(min_value=1, max_value=10)
)
X_shape_univariate = tuples(
    integers(min_value=20, max_value=100), integers(min_value=1, max_value=1)
)
X_shape_multivariate = tuples(
    integers(min_value=20, max_value=100), integers(min_value=2, max_value=10)
)
X_strategy = arrays(
    dtype=float,
    shape=X_shape,
    elements=floats(min_value=1, max_value=100),
    unique=True,
)
X_strategy_univariate = arrays(
    dtype=float,
    shape=X_shape_univariate,
    elements=floats(min_value=1, max_value=100),
    unique=True,
)
X_strategy_multivariate = arrays(
    dtype=float,
    shape=X_shape_multivariate,
    elements=floats(min_value=1, max_value=100),
    unique=True,
)

# Model strategy
model_strategy = sampled_from(
    [str(arg) for arg in get_args(ModelTypesWithoutArch)]
)
model_strategy_univariate = model_strategy.filter(lambda x: x != "var")
model_strategy_multivariate = just("var")


class TestWholeResidualBootstrap:
    class TestPassingCases:
        @settings(deadline=None, max_examples=10)
        @given(
            model_type=model_strategy_univariate,
            order=integers(min_value=1, max_value=5),
            save_models=booleans(),
            n_bootstraps=integers(min_value=1, max_value=10),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
            X=X_strategy_univariate,
        )
        def test_whole_residual_bootstrap(
            self,
            model_type: str,
            order: int,
            save_models: bool,
            n_bootstraps: int,
            rng: int,
            X: np.ndarray,
        ) -> None:
            """
            Test if the WholeResidualBootstrap class initializes correctly and if the _generate_samples_single_bootstrap method runs without errors.
            """
            config = BaseResidualBootstrapConfig(
                model_type=model_type,
                order=order,
                save_models=save_models,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )
            bootstrap = WholeResidualBootstrap(config=config)

            assert bootstrap.config == config

            # Check that _generate_samples_single_bootstrap method runs without errors
            # if (model_type != "var" and X.shape[1] == 1) or (model_type == "var" and X.shape[1] > 1):
            indices, data = bootstrap._generate_samples_single_bootstrap(
                np.array(X)
            )
            assert isinstance(indices, list)
            assert all(isinstance(i, np.ndarray) for i in indices)
            assert isinstance(data, list)
            assert all(isinstance(d, np.ndarray) for d in data)

    class TestFailingCases:
        @settings(deadline=None, max_examples=10)
        @given(
            model_type=model_strategy_univariate,
            order=integers(min_value=1, max_value=5),
            save_models=booleans(),
            params=dictionaries(
                keys=text(min_size=1, max_size=3),
                values=integers(min_value=1, max_value=10),
            ),
            block_length=integers(min_value=1, max_value=10),
            n_bootstraps=integers(min_value=1, max_value=10),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
            X=X_strategy_univariate,
        )
        def test_invalid_fit_model(
            self,
            model_type: str,
            order: int,
            save_models: bool,
            params: dict[str, int],
            block_length: int,
            n_bootstraps: int,
            rng: int,
            X: np.ndarray,
        ) -> None:
            """
            Test if the WholeResidualBootstrap's _generate_samples_single_bootstrap method raises a ValueError when the fit_model method fails.
            """
            config = BaseResidualBootstrapConfig(
                model_type=model_type,
                order=order,
                save_models=save_models,
                model_params=params,
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )
            bootstrap = WholeResidualBootstrap(config=config)

            # Check that _generate_samples_single_bootstrap method raises a ValueError when the fit_model method fails
            with patch.object(
                TSFitBestLag, "fit", side_effect=ValueError
            ), pytest.raises(ValueError):
                bootstrap._generate_samples_single_bootstrap(np.array(X))


class TestBlockResidualBootstrap:
    class TestPassingCases:
        @settings(deadline=None, max_examples=10)
        @given(
            model_type=model_strategy_univariate,
            order=integers(min_value=1, max_value=10),
            save_models=booleans(),
            params=dictionaries(
                keys=text(min_size=1, max_size=3),
                values=integers(min_value=1, max_value=10),
            ),
            bootstrap_type=sampled_from(list(BLOCK_BOOTSTRAP_TYPES_DICT)),
            block_length=integers(min_value=1, max_value=5),
            n_bootstraps=integers(min_value=1, max_value=10),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
            X=X_strategy_univariate,
        )
        def test_block_residual_bootstrap(
            self,
            model_type: str,
            order: int,
            save_models: bool,
            params: dict[str, int],
            bootstrap_type: str,
            block_length: int,
            n_bootstraps: int,
            rng: int,
            X: list[list[float]],
        ) -> None:
            """
            Test if the BlockResidualBootstrap class initializes correctly and if the _generate_samples_single_bootstrap method runs without errors.
            """
            residual_config = BaseResidualBootstrapConfig(
                model_type=model_type,
                order=order,
                save_models=save_models,
                model_params=params,
            )
            block_config = BaseBlockBootstrapConfig(
                bootstrap_type=bootstrap_type,
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )
            bootstrap = BlockResidualBootstrap(
                residual_config=residual_config, block_config=block_config
            )
            from pprint import pprint

            pprint(dir(bootstrap.config))
            print("")
            assert bootstrap.config == residual_config

            # Check that _generate_samples_single_bootstrap method runs without errors
            indices, data = bootstrap._generate_samples_single_bootstrap(
                np.array(X)
            )
            assert isinstance(indices, list)
            assert all(isinstance(i, np.ndarray) for i in indices)
            assert isinstance(data, list)
            assert all(isinstance(d, np.ndarray) for d in data)

    class TestFailingCases:
        @settings(deadline=None, max_examples=10)
        @given(
            model_type=text(min_size=3, max_size=10),
            order=integers(min_value=1, max_value=10),
            save_models=booleans(),
            params=dictionaries(
                keys=text(min_size=1, max_size=3),
                values=integers(min_value=1, max_value=10),
            ),
            bootstrap_type=sampled_from(list(BLOCK_BOOTSTRAP_TYPES_DICT)),
            block_length=integers(min_value=1, max_value=10),
            n_bootstraps=integers(min_value=1, max_value=10),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
            X=lists(lists(floats()), min_size=2, max_size=10).filter(
                lambda x: len(set(map(len, x))) == 1
            ),
        )
        def test_invalid_fit_model(
            self,
            model_type: str,
            order: int,
            save_models: bool,
            params: dict[str, int],
            bootstrap_type: str,
            block_length: int,
            n_bootstraps: int,
            rng: int,
            X: list[list[float]],
        ) -> None:
            """
            Test if the BlockResidualBootstrap's _generate_samples_single_bootstrap method raises a ValueError when the fit_model method fails.
            """
            residual_config = BaseResidualBootstrapConfig(
                model_type=model_type,
                order=order,
                save_models=save_models,
                model_params=params,
            )
            block_config = BlockBootstrapConfig(
                bootstrap_type=bootstrap_type,
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )
            bootstrap = BlockResidualBootstrap(
                residual_config=residual_config, block_config=block_config
            )

            # Check that _generate_samples_single_bootstrap method raises a ValueError when the fit_model method fails
            with patch.object(
                TSFitBestLag, "fit", side_effect=ValueError
            ), pytest.raises(ValueError):
                bootstrap._generate_samples_single_bootstrap(np.array(X))


# Continue with the similar test cases for the other classes
