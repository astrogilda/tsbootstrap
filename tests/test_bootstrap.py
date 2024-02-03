from collections.abc import Callable
from math import comb
from typing import get_args
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
    none,
    one_of,
    sampled_from,
    text,
    tuples,
)
from numpy.linalg import LinAlgError
from pyexpat import model
from tsbootstrap.base_bootstrap import BaseStatisticPreservingBootstrap
from tsbootstrap.base_bootstrap_configs import (
    BaseDistributionBootstrapConfig,
    BaseMarkovBootstrapConfig,
    BaseResidualBootstrapConfig,
    BaseSieveBootstrapConfig,
    BaseStatisticPreservingBootstrapConfig,
)
from tsbootstrap.block_bootstrap import BLOCK_BOOTSTRAP_TYPES_DICT
from tsbootstrap.block_bootstrap_configs import BaseBlockBootstrapConfig
from tsbootstrap.bootstrap import (
    BlockDistributionBootstrap,
    BlockMarkovBootstrap,
    BlockResidualBootstrap,
    BlockSieveBootstrap,
    BlockStatisticPreservingBootstrap,
    WholeDistributionBootstrap,
    WholeMarkovBootstrap,
    WholeResidualBootstrap,
    WholeSieveBootstrap,
    WholeStatisticPreservingBootstrap,
)
from tsbootstrap.tsfit import TSFitBestLag
from tsbootstrap.utils.types import (
    BlockCompressorTypes,
    ModelTypes,
    ModelTypesWithoutArch,
)

# The shape is a strategy generating tuples (num_rows, num_columns)
# min of 30 elements to enable transition from one state to another, even with two n_states, for HMM
X_shape = tuples(
    integers(min_value=40, max_value=100), integers(min_value=1, max_value=10)
)
X_shape_univariate = tuples(
    integers(min_value=40, max_value=100), integers(min_value=1, max_value=1)
)
X_shape_multivariate = tuples(
    integers(min_value=40, max_value=100), integers(min_value=2, max_value=10)
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

# Resids model strategy for sieve bootstrap
resids_model_strategy = sampled_from(
    [str(arg) for arg in get_args(ModelTypes)]
)
resids_model_strategy_univariate = resids_model_strategy.filter(
    lambda x: x != "var"
)
resids_model_strategy_multivariate = just("var")

# Markov method strategy
markov_method_strategy = sampled_from(
    [str(arg) for arg in get_args(BlockCompressorTypes)]
)


# class TestWholeResidualBootstrap:
#     class TestPassingCases:

#         @settings(deadline=None, max_examples=10)
#         @given(
#             model_type=model_strategy_univariate,
#             order=integers(min_value=1, max_value=5),
#             save_models=booleans(),
#             n_bootstraps=integers(min_value=1, max_value=10),
#             rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
#             X=X_strategy_univariate,
#         )
#         def test_whole_residual_bootstrap(
#             self,
#             model_type: str,
#             order: int,
#             save_models: bool,
#             n_bootstraps: int,
#             rng: int,
#             X: np.ndarray,
#         ) -> None:
#             """
#             Test if the WholeResidualBootstrap class initializes correctly and if the _generate_samples_single_bootstrap method runs without errors.
#             """
#             config = BaseResidualBootstrapConfig(
#                 model_type=model_type,
#                 order=order,
#                 save_models=save_models,
#                 n_bootstraps=n_bootstraps,
#                 rng=rng,
#             )
#             bootstrap = WholeResidualBootstrap(config=config)

#             assert bootstrap.config == config

#             # Check that _generate_samples_single_bootstrap method runs without errors
#             # if (model_type != "var" and X.shape[1] == 1) or (model_type == "var" and X.shape[1] > 1):
#             indices, data = bootstrap._generate_samples_single_bootstrap(
#                 np.array(X)
#             )
#             assert isinstance(indices, list)
#             assert len(indices) == 1
#             assert isinstance(indices[0], np.ndarray)
#             assert len(indices[0]) == X.shape[0]

#             assert isinstance(data, list)
#             assert len(data) == 1
#             assert isinstance(data[0], np.ndarray)
#             assert len(data[0]) == X.shape[0]

#             # Check that _generate_samples method runs without errors
#             bootstrap = WholeResidualBootstrap(config=config)
#             indices_data_gen = bootstrap._generate_samples(
#                 np.array(X), return_indices=True
#             )
#             indices_data_gen_list = list(indices_data_gen)
#             assert isinstance(indices_data_gen_list, list)
#             assert len(indices_data_gen_list) == n_bootstraps
#             # Unpack indices and data
#             indices, data = zip(*indices_data_gen_list)
#             assert isinstance(indices, tuple)
#             assert len(indices) == n_bootstraps
#             assert all(isinstance(i, list) for i in indices)
#             assert all(np.prod(np.shape(i)) == X.shape[0] for i in indices)

#             assert isinstance(data, tuple)
#             assert len(data) == n_bootstraps
#             assert all(isinstance(d, np.ndarray) for d in data)
#             assert all(np.prod(np.shape(d)) == X.shape[0] for d in data)

#             # Check that bootstrap.bootstrap method runs without errors
#             bootstrap = WholeResidualBootstrap(config=config)
#             indices_data_gen = bootstrap.bootstrap(
#                 np.array(X), return_indices=True, test_ratio=0.2
#             )
#             indices_data_gen_list = list(indices_data_gen)
#             assert isinstance(indices_data_gen_list, list)
#             assert len(indices_data_gen_list) == n_bootstraps
#             # Unpack indices and data
#             indices, data = zip(*indices_data_gen_list)
#             assert isinstance(indices, tuple)
#             assert len(indices) == n_bootstraps
#             assert all(isinstance(i, list) for i in indices)
#             assert all(
#                 np.prod(np.shape(i)) == int(X.shape[0] * 0.8) for i in indices
#             )

#             assert isinstance(data, tuple)
#             assert len(data) == n_bootstraps
#             assert all(isinstance(d, np.ndarray) for d in data)
#             assert all(
#                 np.prod(np.shape(d)) == int(X.shape[0] * 0.8) for d in data
#             )

#     class TestFailingCases:
#         @settings(deadline=None, max_examples=10)
#         @given(
#             model_type=model_strategy_univariate,
#             order=integers(min_value=1, max_value=5),
#             save_models=booleans(),
#             params=dictionaries(
#                 keys=text(min_size=1, max_size=3),
#                 values=integers(min_value=1, max_value=10),
#             ),
#             n_bootstraps=integers(min_value=1, max_value=10),
#             rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
#             X=X_strategy_univariate,
#         )
#         def test_invalid_fit_model(
#             self,
#             model_type: str,
#             order: int,
#             save_models: bool,
#             params: dict[str, int],
#             n_bootstraps: int,
#             rng: int,
#             X: np.ndarray,
#         ) -> None:
#             """
#             Test if the WholeResidualBootstrap's _generate_samples_single_bootstrap method raises a ValueError when the fit_model method fails.
#             """
#             config = BaseResidualBootstrapConfig(
#                 model_type=model_type,
#                 order=order,
#                 save_models=save_models,
#                 model_params=params,
#                 n_bootstraps=n_bootstraps,
#                 rng=rng,
#             )
#             bootstrap = WholeResidualBootstrap(config=config)

#             # Check that _generate_samples_single_bootstrap method raises a ValueError when the fit_model method fails
#             with patch.object(
#                 TSFitBestLag, "fit", side_effect=ValueError
#             ), pytest.raises(ValueError):
#                 bootstrap._generate_samples_single_bootstrap(np.array(X))


# class TestBlockResidualBootstrap:
#     class TestPassingCases:
#         @settings(deadline=None, max_examples=10)
#         @given(
#             model_type=model_strategy_univariate,
#             order=integers(min_value=1, max_value=10),
#             save_models=booleans(),
#             bootstrap_type=sampled_from(list(BLOCK_BOOTSTRAP_TYPES_DICT)),
#             block_length=integers(min_value=1, max_value=5),
#             combine_generation_and_sampling_flag=booleans(),
#             n_bootstraps=integers(min_value=1, max_value=10),
#             rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
#             X=X_strategy_univariate,
#         )
#         def test_block_residual_bootstrap(
#             self,
#             model_type: str,
#             order: int,
#             save_models: bool,
#             combine_generation_and_sampling_flag: bool,
#             bootstrap_type: str,
#             block_length: int,
#             n_bootstraps: int,
#             rng: int,
#             X: np.ndarray,
#         ) -> None:
#             """
#             Test if the BlockResidualBootstrap class initializes correctly and if the _generate_samples_single_bootstrap method runs without errors.
#             """
#             residual_config = BaseResidualBootstrapConfig(
#                 model_type=model_type,
#                 order=order,
#                 save_models=save_models,
#                 n_bootstraps=n_bootstraps,
#                 rng=rng,
#             )
#             block_config = BaseBlockBootstrapConfig(
#                 bootstrap_type=bootstrap_type,
#                 block_length=block_length,
#                 combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
#             )
#             bootstrap = BlockResidualBootstrap(
#                 residual_config=residual_config, block_config=block_config
#             )

#             assert bootstrap.config == residual_config

#             # Check that _generate_samples_single_bootstrap method runs without errors
#             indices, data = bootstrap._generate_samples_single_bootstrap(
#                 np.array(X)
#             )
#             assert isinstance(indices, list)
#             assert all(isinstance(i, np.ndarray) for i in indices)
#             assert isinstance(data, list)
#             assert all(isinstance(d, np.ndarray) for d in data)

#             # Check that _generate_samples method runs without errors
#             bootstrap = BlockResidualBootstrap(
#                 residual_config=residual_config, block_config=block_config
#             )
#             indices_data_gen = bootstrap._generate_samples(
#                 np.array(X), return_indices=True
#             )
#             indices_data_gen_list = list(indices_data_gen)
#             assert isinstance(indices_data_gen_list, list)
#             # assert len(indices_data_gen_list) == n_bootstraps
#             # Unpack indices and data
#             indices, data = zip(*indices_data_gen_list)
#             assert isinstance(indices, tuple)
#             assert len(indices) == n_bootstraps
#             assert all(isinstance(i, list) for i in indices)
#             assert all(
#                 sum(len(i_iter) for i_iter in i) == X.shape[0] for i in indices
#             )

#             assert isinstance(data, tuple)
#             assert len(data) == n_bootstraps
#             assert all(isinstance(d, np.ndarray) for d in data)
#             assert all(
#                 sum(len(d_iter) for d_iter in d) == X.shape[0] for d in data
#             )

#             # Check that bootstrap.bootstrap method runs without errors
#             bootstrap = BlockResidualBootstrap(
#                 residual_config=residual_config, block_config=block_config
#             )
#             indices_data_gen = bootstrap.bootstrap(
#                 np.array(X), return_indices=True, test_ratio=0.2
#             )
#             indices_data_gen_list = list(indices_data_gen)
#             assert isinstance(indices_data_gen_list, list)
#             assert len(indices_data_gen_list) == n_bootstraps
#             # Unpack indices and data
#             indices, data = zip(*indices_data_gen_list)
#             assert isinstance(indices, tuple)
#             assert len(indices) == n_bootstraps
#             assert all(isinstance(i, list) for i in indices)
#             assert all(
#                 sum(len(i_iter) for i_iter in i) == int(X.shape[0] * 0.8)
#                 for i in indices
#             )

#             assert isinstance(data, tuple)
#             assert len(data) == n_bootstraps
#             assert all(isinstance(d, np.ndarray) for d in data)
#             assert all(
#                 sum(len(d_iter) for d_iter in d) == int(X.shape[0] * 0.8)
#                 for d in data
#             )

#     class TestFailingCases:
#         @settings(deadline=None, max_examples=10)
#         @given(
#             model_type=model_strategy_univariate,
#             order=integers(min_value=1, max_value=10),
#             save_models=booleans(),
#             params=dictionaries(
#                 keys=text(min_size=1, max_size=3),
#                 values=integers(min_value=1, max_value=10),
#             ),
#             bootstrap_type=sampled_from(list(BLOCK_BOOTSTRAP_TYPES_DICT)),
#             block_length=integers(min_value=1, max_value=10),
#             n_bootstraps=integers(min_value=1, max_value=10),
#             rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
#             X=X_strategy_univariate,
#         )
#         def test_invalid_fit_model(
#             self,
#             model_type: str,
#             order: int,
#             save_models: bool,
#             params: dict[str, int],
#             bootstrap_type: str,
#             block_length: int,
#             n_bootstraps: int,
#             rng: int,
#             X: np.ndarray,
#         ) -> None:
#             """
#             Test if the BlockResidualBootstrap's _generate_samples_single_bootstrap method raises a ValueError when the fit_model method fails.
#             """
#             residual_config = BaseResidualBootstrapConfig(
#                 model_type=model_type,
#                 order=order,
#                 save_models=save_models,
#                 model_params=params,
#                 n_bootstraps=n_bootstraps,
#                 rng=rng,
#             )
#             block_config = BaseBlockBootstrapConfig(
#                 bootstrap_type=bootstrap_type,
#                 block_length=block_length,
#             )
#             bootstrap = BlockResidualBootstrap(
#                 residual_config=residual_config, block_config=block_config
#             )

#             # Check that _generate_samples_single_bootstrap method raises a ValueError when the fit_model method fails
#             with patch.object(
#                 TSFitBestLag, "fit", side_effect=ValueError
#             ), pytest.raises(ValueError):
#                 bootstrap._generate_samples_single_bootstrap(np.array(X))


class TestWholeMarkovBootstrap:
    class TestPassingCases:
        @settings(deadline=None, max_examples=10)
        @given(
            model_type=model_strategy_univariate,
            order=integers(min_value=1, max_value=5),
            save_models=booleans(),
            n_bootstraps=integers(min_value=1, max_value=10),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
            apply_pca_flag=booleans(),
            blocks_as_hidden_states_flag=booleans(),
            method=markov_method_strategy,
            n_states=just(2),
        )
        def test_whole_markov_bootstrap(
            self,
            model_type: str,
            order: int,
            save_models: bool,
            n_bootstraps: int,
            rng: int,
            apply_pca_flag: bool,
            blocks_as_hidden_states_flag: bool,
            method: str,
            n_states: int,
        ) -> None:
            """
            Test if the WholeMarkovBootstrap class initializes correctly and if the _generate_samples_single_bootstrap method runs without errors.
            """
            X = np.random.rand(20, 1)
            config = BaseMarkovBootstrapConfig(
                model_type=model_type,
                order=order,
                save_models=save_models,
                n_bootstraps=n_bootstraps,
                rng=rng,
                apply_pca_flag=apply_pca_flag,
                blocks_as_hidden_states_flag=blocks_as_hidden_states_flag,
                method=method,
                n_states=n_states,
            )
            bootstrap = WholeMarkovBootstrap(config=config)

            assert bootstrap.config == config

            # Check that _generate_samples_single_bootstrap method runs without errors
            indices, data = bootstrap._generate_samples_single_bootstrap(
                np.array(X)
            )
            assert isinstance(indices, list)
            assert len(indices) == 1
            assert isinstance(indices[0], np.ndarray)
            assert len(indices[0]) == X.shape[0]

            assert isinstance(data, list)
            assert len(data) == 1
            assert isinstance(data[0], np.ndarray)
            assert len(data[0]) == X.shape[0]

            # Check that _generate_samples method runs without errors
            bootstrap = WholeMarkovBootstrap(config=config)
            indices_data_gen = bootstrap._generate_samples(
                np.array(X), return_indices=True
            )
            indices_data_gen_list = list(indices_data_gen)
            assert isinstance(indices_data_gen_list, list)
            assert len(indices_data_gen_list) == n_bootstraps
            # Unpack indices and data
            indices, data = zip(*indices_data_gen_list)
            assert isinstance(indices, tuple)
            assert len(indices) == n_bootstraps
            assert all(isinstance(i, list) for i in indices)
            assert all(np.prod(np.shape(i)) == X.shape[0] for i in indices)

            assert isinstance(data, tuple)
            assert len(data) == n_bootstraps
            assert all(isinstance(d, np.ndarray) for d in data)
            assert all(np.prod(np.shape(d)) == X.shape[0] for d in data)

            # Check that bootstrap.bootstrap method runs without errors
            bootstrap = WholeMarkovBootstrap(config=config)
            indices_data_gen = bootstrap.bootstrap(
                np.array(X), return_indices=True, test_ratio=0.2
            )
            indices_data_gen_list = list(indices_data_gen)
            assert isinstance(indices_data_gen_list, list)
            assert len(indices_data_gen_list) == n_bootstraps
            # Unpack indices and data
            indices, data = zip(*indices_data_gen_list)
            assert isinstance(indices, tuple)
            assert len(indices) == n_bootstraps
            assert all(isinstance(i, list) for i in indices)
            assert all(
                np.prod(np.shape(i)) == int(X.shape[0] * 0.8) for i in indices
            )

            assert isinstance(data, tuple)
            assert len(data) == n_bootstraps
            assert all(isinstance(d, np.ndarray) for d in data)
            assert all(
                np.prod(np.shape(d)) == int(X.shape[0] * 0.8) for d in data
            )

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
            n_bootstraps=integers(min_value=1, max_value=10),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
            apply_pca_flag=booleans(),
            blocks_as_hidden_states_flag=booleans(),
            method=markov_method_strategy,
            n_states=just(2),
        )
        def test_invalid_fit_model(
            self,
            model_type: str,
            order: int,
            save_models: bool,
            params: dict[str, int],
            n_bootstraps: int,
            rng: int,
            apply_pca_flag: bool,
            blocks_as_hidden_states_flag: bool,
            method: str,
            n_states: int,
        ) -> None:
            """
            Test if the WholeMarkovBootstrap's _generate_samples_single_bootstrap method raises a ValueError when the fit_model method fails.
            """
            X = np.random.rand(20, 1)
            config = BaseMarkovBootstrapConfig(
                model_type=model_type,
                order=order,
                save_models=save_models,
                model_params=params,
                n_bootstraps=n_bootstraps,
                rng=rng,
                apply_pca_flag=apply_pca_flag,
                blocks_as_hidden_states_flag=blocks_as_hidden_states_flag,
                method=method,
                n_states=n_states,
            )
            bootstrap = WholeMarkovBootstrap(config=config)

            # Check that _generate_samples_single_bootstrap method raises a ValueError when the fit_model method fails
            with patch.object(
                TSFitBestLag, "fit", side_effect=ValueError
            ), pytest.raises(ValueError):
                bootstrap._generate_samples_single_bootstrap(np.array(X))


class TestBlockMarkovBootstrap:
    class TestPassingCases:
        @settings(deadline=None, max_examples=100)
        @given(
            model_type=model_strategy_univariate,
            order=integers(min_value=1, max_value=5),
            save_models=booleans(),
            bootstrap_type=sampled_from(list(BLOCK_BOOTSTRAP_TYPES_DICT)),
            block_length=integers(min_value=1, max_value=5),
            combine_generation_and_sampling_flag=booleans(),
            apply_pca_flag=booleans(),
            blocks_as_hidden_states_flag=booleans(),
            method=markov_method_strategy,
            n_states=just(2),
            n_bootstraps=integers(min_value=1, max_value=10),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
        )
        def test_block_markov_bootstrap(
            self,
            model_type: str,
            order: int,
            save_models: bool,
            combine_generation_and_sampling_flag: bool,
            bootstrap_type: str,
            block_length: int,
            apply_pca_flag: bool,
            blocks_as_hidden_states_flag: bool,
            method: str,
            n_states: int,
            n_bootstraps: int,
            rng: int,
        ) -> None:
            """
            Test if the BlockMarkovBootstrap class initializes correctly and if the _generate_samples_single_bootstrap method runs without errors.
            """
            X = np.random.rand(20, 1)
            markov_config = BaseMarkovBootstrapConfig(
                model_type=model_type,
                order=order,
                save_models=save_models,
                n_bootstraps=n_bootstraps,
                rng=rng,
                apply_pca_flag=apply_pca_flag,
                blocks_as_hidden_states_flag=blocks_as_hidden_states_flag,
                method=method,
                n_states=n_states,
            )
            block_config = BaseBlockBootstrapConfig(
                bootstrap_type=bootstrap_type,
                block_length=block_length,
                combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            )
            bootstrap = BlockMarkovBootstrap(
                markov_config=markov_config, block_config=block_config
            )

            assert bootstrap.config == markov_config

            # Check that _generate_samples_single_bootstrap method runs without errors
            try:
                indices, data = bootstrap._generate_samples_single_bootstrap(
                    np.array(X)
                )
            except LinAlgError as e:
                print(e)
            else:
                assert isinstance(indices, list)
                assert all(isinstance(i, np.ndarray) for i in indices)
                assert isinstance(data, list)
                assert all(isinstance(d, np.ndarray) for d in data)

            # Check that _generate_samples method runs without errors
            bootstrap = BlockMarkovBootstrap(
                markov_config=markov_config, block_config=block_config
            )
            indices_data_gen = bootstrap._generate_samples(
                np.array(X), return_indices=True
            )
            indices_data_gen_list = list(indices_data_gen)
            assert isinstance(indices_data_gen_list, list)
            # assert len(indices_data_gen_list) == n_bootstraps
            # Unpack indices and data
            indices, data = zip(*indices_data_gen_list)
            assert isinstance(indices, tuple)
            assert len(indices) == n_bootstraps
            assert all(isinstance(i, list) for i in indices)
            assert all(
                sum(len(i_iter) for i_iter in i) == X.shape[0] for i in indices
            )

            assert isinstance(data, tuple)
            assert len(data) == n_bootstraps
            assert all(isinstance(d, np.ndarray) for d in data)
            assert all(
                sum(len(d_iter) for d_iter in d) == X.shape[0] for d in data
            )

            # Check that bootstrap.bootstrap method runs without errors
            bootstrap = BlockMarkovBootstrap(
                markov_config=markov_config, block_config=block_config
            )
            indices_data_gen = bootstrap.bootstrap(
                np.array(X), return_indices=True, test_ratio=0.2
            )
            indices_data_gen_list = list(indices_data_gen)
            assert isinstance(indices_data_gen_list, list)
            assert len(indices_data_gen_list) == n_bootstraps
            # Unpack indices and data
            indices, data = zip(*indices_data_gen_list)
            assert isinstance(indices, tuple)
            assert len(indices) == n_bootstraps
            assert all(isinstance(i, list) for i in indices)
            assert all(
                sum(len(i_iter) for i_iter in i) == int(X.shape[0] * 0.8)
                for i in indices
            )

            assert isinstance(data, tuple)
            assert len(data) == n_bootstraps
            assert all(isinstance(d, np.ndarray) for d in data)
            assert all(
                sum(len(d_iter) for d_iter in d) == int(X.shape[0] * 0.8)
                for d in data
            )

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
            bootstrap_type=sampled_from(list(BLOCK_BOOTSTRAP_TYPES_DICT)),
            block_length=integers(min_value=1, max_value=10),
            n_bootstraps=integers(min_value=1, max_value=10),
            apply_pca_flag=booleans(),
            blocks_as_hidden_states_flag=booleans(),
            method=markov_method_strategy,
            n_states=just(2),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
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
            apply_pca_flag: bool,
            blocks_as_hidden_states_flag: bool,
            method: str,
            n_states: int,
            rng: int,
        ) -> None:
            """
            Test if the BlockMarkovBootstrap's _generate_samples_single_bootstrap method raises a ValueError when the fit_model method fails.
            """
            X = np.random.rand(20, 1)
            markov_config = BaseMarkovBootstrapConfig(
                model_type=model_type,
                order=order,
                save_models=save_models,
                model_params=params,
                n_bootstraps=n_bootstraps,
                rng=rng,
                apply_pca_flag=apply_pca_flag,
                blocks_as_hidden_states_flag=blocks_as_hidden_states_flag,
                method=method,
                n_states=n_states,
            )
            block_config = BaseBlockBootstrapConfig(
                bootstrap_type=bootstrap_type,
                block_length=block_length,
            )
            bootstrap = BlockMarkovBootstrap(
                markov_config=markov_config, block_config=block_config
            )

            # Check that _generate_samples_single_bootstrap method raises a ValueError when the fit_model method fails
            with patch.object(
                TSFitBestLag, "fit", side_effect=ValueError
            ), pytest.raises(ValueError):
                bootstrap._generate_samples_single_bootstrap(np.array(X))


class TestWholeStatisticPreservingBootstrap:
    class TestPassingCases:
        @settings(deadline=None, max_examples=10)
        @given(
            statistic=sampled_from([np.mean, np.median, np.std]),
            statistic_axis=sampled_from([0, 1]),
            statistic_keepdims=just(True),
            n_bootstraps=integers(min_value=1, max_value=10),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
            X=X_strategy_univariate,
        )
        def test_whole_statistic_preserving_bootstrap(
            self,
            statistic: Callable[[np.ndarray], float],
            statistic_axis: int,
            statistic_keepdims: bool,
            n_bootstraps: int,
            rng: int,
            X: np.ndarray,
        ) -> None:
            """
            Test if the WholeStatisticPreservingBootstrap class initializes correctly and if the _generate_samples_single_bootstrap method runs without errors.
            """
            config = BaseStatisticPreservingBootstrapConfig(
                n_bootstraps=n_bootstraps,
                rng=rng,
                statistic=statistic,
                statistic_axis=statistic_axis,
                statistic_keepdims=statistic_keepdims,
            )
            bootstrap = WholeStatisticPreservingBootstrap(config=config)

            assert bootstrap.config == config

            # Check that _generate_samples_single_bootstrap method runs without errors
            indices, data = bootstrap._generate_samples_single_bootstrap(
                np.array(X)
            )
            assert isinstance(indices, list)
            assert len(indices) == 1
            assert isinstance(indices[0], np.ndarray)
            assert len(indices[0]) == X.shape[0]

            assert isinstance(data, list)
            assert len(data) == 1
            assert isinstance(data[0], np.ndarray)
            assert len(data[0]) == X.shape[0]

            # Check that _generate_samples method runs without errors
            bootstrap = WholeStatisticPreservingBootstrap(config=config)
            indices_data_gen = bootstrap._generate_samples(
                np.array(X), return_indices=True
            )
            indices_data_gen_list = list(indices_data_gen)
            assert isinstance(indices_data_gen_list, list)
            assert len(indices_data_gen_list) == n_bootstraps
            # Unpack indices and data
            indices, data = zip(*indices_data_gen_list)
            assert isinstance(indices, tuple)
            assert len(indices) == n_bootstraps
            assert all(isinstance(i, list) for i in indices)
            assert all(np.prod(np.shape(i)) == X.shape[0] for i in indices)

            assert isinstance(data, tuple)
            assert len(data) == n_bootstraps
            assert all(isinstance(d, np.ndarray) for d in data)
            assert all(np.prod(np.shape(d)) == X.shape[0] for d in data)

            # Check that bootstrap.bootstrap method runs without errors
            bootstrap = WholeStatisticPreservingBootstrap(config=config)
            indices_data_gen = bootstrap.bootstrap(
                np.array(X), return_indices=True, test_ratio=0.2
            )
            indices_data_gen_list = list(indices_data_gen)
            assert isinstance(indices_data_gen_list, list)
            assert len(indices_data_gen_list) == n_bootstraps
            # Unpack indices and data
            indices, data = zip(*indices_data_gen_list)
            assert isinstance(indices, tuple)
            assert len(indices) == n_bootstraps
            assert all(isinstance(i, list) for i in indices)
            assert all(
                np.prod(np.shape(i)) == int(X.shape[0] * 0.8) for i in indices
            )

            assert isinstance(data, tuple)
            assert len(data) == n_bootstraps
            assert all(isinstance(d, np.ndarray) for d in data)
            assert all(
                np.prod(np.shape(d)) == int(X.shape[0] * 0.8) for d in data
            )


class TestBlockStatisticPreservingBootstrap:
    class TestPassingCases:
        @pytest.mark.skip(reason="known LU decomposition issue, see #41")
        @settings(deadline=None, max_examples=10)
        @given(
            statistic=sampled_from([np.mean, np.median, np.std]),
            statistic_axis=sampled_from([0, 1]),
            statistic_keepdims=just(True),
            bootstrap_type=sampled_from(list(BLOCK_BOOTSTRAP_TYPES_DICT)),
            block_length=integers(min_value=1, max_value=5),
            combine_generation_and_sampling_flag=booleans(),
            n_bootstraps=integers(min_value=1, max_value=10),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
            X=X_strategy_univariate,
        )
        def test_block_statistic_preserving_bootstrap(
            self,
            statistic: Callable[[np.ndarray], float],
            statistic_axis: int,
            statistic_keepdims: bool,
            combine_generation_and_sampling_flag: bool,
            bootstrap_type: str,
            block_length: int,
            n_bootstraps: int,
            rng: int,
            X: np.ndarray,
        ) -> None:
            """
            Test if the BlockStatisticPreservingBootstrap class initializes correctly and if the _generate_samples_single_bootstrap method runs without errors.
            """
            statistic_config = BaseStatisticPreservingBootstrapConfig(
                n_bootstraps=n_bootstraps,
                rng=rng,
                statistic=statistic,
                statistic_axis=statistic_axis,
                statistic_keepdims=statistic_keepdims,
            )
            block_config = BaseBlockBootstrapConfig(
                bootstrap_type=bootstrap_type,
                block_length=block_length,
                combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            )
            bootstrap = BlockStatisticPreservingBootstrap(
                statistic_config=statistic_config, block_config=block_config
            )

            assert bootstrap.config == statistic_config

            # Check that _generate_samples_single_bootstrap method runs without errors
            indices, data = bootstrap._generate_samples_single_bootstrap(
                np.array(X)
            )
            assert isinstance(indices, list)
            assert all(isinstance(i, np.ndarray) for i in indices)
            assert isinstance(data, list)
            assert all(isinstance(d, np.ndarray) for d in data)

            # Check that _generate_samples method runs without errors
            bootstrap = BlockStatisticPreservingBootstrap(
                statistic_config=statistic_config, block_config=block_config
            )
            indices_data_gen = bootstrap._generate_samples(
                np.array(X), return_indices=True
            )
            indices_data_gen_list = list(indices_data_gen)
            assert isinstance(indices_data_gen_list, list)
            assert len(indices_data_gen_list) == n_bootstraps
            # Unpack indices and data
            indices, data = zip(*indices_data_gen_list)
            assert isinstance(indices, tuple)
            assert len(indices) == n_bootstraps
            assert all(isinstance(i, list) for i in indices)
            assert all(
                sum(len(i_iter) for i_iter in i) == X.shape[0] for i in indices
            )

            assert isinstance(data, tuple)
            assert len(data) == n_bootstraps
            assert all(isinstance(d, np.ndarray) for d in data)
            assert all(
                sum(len(d_iter) for d_iter in d) == X.shape[0] for d in data
            )

            # Check that bootstrap.bootstrap method runs without errors
            bootstrap = BlockStatisticPreservingBootstrap(
                statistic_config=statistic_config, block_config=block_config
            )
            indices_data_gen = bootstrap.bootstrap(
                np.array(X), return_indices=True, test_ratio=0.2
            )
            indices_data_gen_list = list(indices_data_gen)
            assert isinstance(indices_data_gen_list, list)
            assert len(indices_data_gen_list) == n_bootstraps
            # Unpack indices and data
            indices, data = zip(*indices_data_gen_list)
            assert isinstance(indices, tuple)
            assert len(indices) == n_bootstraps
            assert all(isinstance(i, list) for i in indices)
            assert all(
                sum(len(i_iter) for i_iter in i) == int(X.shape[0] * 0.8)
                for i in indices
            )

            assert isinstance(data, tuple)
            assert len(data) == n_bootstraps
            assert all(isinstance(d, np.ndarray) for d in data)
            assert all(
                sum(len(d_iter) for d_iter in d) == int(X.shape[0] * 0.8)
                for d in data
            )

    class TestFailingCases:
        @settings(deadline=None, max_examples=10)
        @given(
            statistic=sampled_from([np.mean, np.median, np.std]),
            statistic_axis=sampled_from([0, 1]),
            statistic_keepdims=just(True),
            bootstrap_type=sampled_from(list(BLOCK_BOOTSTRAP_TYPES_DICT)),
            block_length=integers(min_value=1, max_value=10),
            n_bootstraps=integers(min_value=1, max_value=10),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
            X=X_strategy_univariate,
        )
        def test_invalid_fit_model(
            self,
            statistic: Callable[[np.ndarray], float],
            statistic_axis: int,
            statistic_keepdims: bool,
            bootstrap_type: str,
            block_length: int,
            n_bootstraps: int,
            rng: int,
            X: np.ndarray,
        ) -> None:
            """
            Test if the BlockStatisticPreservingBootstrap's _generate_samples_single_bootstrap method raises a ValueError when the fit_model method fails.
            """
            statistic_config = BaseStatisticPreservingBootstrapConfig(
                statistic=statistic,
                statistic_axis=statistic_axis,
                statistic_keepdims=statistic_keepdims,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )
            block_config = BaseBlockBootstrapConfig(
                bootstrap_type=bootstrap_type,
                block_length=block_length,
            )
            bootstrap = BlockStatisticPreservingBootstrap(
                statistic_config=statistic_config, block_config=block_config
            )

            # Check that _generate_samples_single_bootstrap method raises a ValueError when the fit_model method fails
            with patch.object(
                BaseStatisticPreservingBootstrap,
                "_calculate_statistic",
                side_effect=ValueError,
            ), pytest.raises(ValueError):
                bootstrap._generate_samples_single_bootstrap(np.array(X))


class TestWholeDistributionBootstrap:
    class TestPassingCases:
        @pytest.mark.skip(reason="known LU decomposition issue, see #41")
        @settings(deadline=None, max_examples=10)
        @given(
            model_type=model_strategy_univariate,
            order=integers(min_value=1, max_value=5),
            save_models=booleans(),
            n_bootstraps=integers(min_value=1, max_value=10),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
            X=X_strategy_univariate,
            distribution=sampled_from(["normal", "uniform"]),
            refit=booleans(),
        )
        def test_whole_distribution_bootstrap(
            self,
            model_type: str,
            order: int,
            save_models: bool,
            n_bootstraps: int,
            rng: int,
            X: np.ndarray,
            distribution: str,
            refit: bool,
        ) -> None:
            """
            Test if the WholeStatisticPreservingBootstrap class initializes correctly and if the _generate_samples_single_bootstrap method runs without errors.
            """
            config = BaseDistributionBootstrapConfig(
                model_type=model_type,
                order=order,
                save_models=save_models,
                n_bootstraps=n_bootstraps,
                rng=rng,
                distribution=distribution,
                refit=refit,
            )
            bootstrap = WholeDistributionBootstrap(config=config)

            assert bootstrap.config == config

            # Check that _generate_samples_single_bootstrap method runs without errors
            indices, data = bootstrap._generate_samples_single_bootstrap(
                np.array(X)
            )
            assert isinstance(indices, list)
            assert len(indices) == 1
            assert isinstance(indices[0], np.ndarray)
            assert len(indices[0]) == X.shape[0]

            assert isinstance(data, list)
            assert len(data) == 1
            assert isinstance(data[0], np.ndarray)
            assert len(data[0]) == X.shape[0]

            # Check that _generate_samples method runs without errors
            bootstrap = WholeDistributionBootstrap(config=config)
            indices_data_gen = bootstrap._generate_samples(
                np.array(X), return_indices=True
            )
            indices_data_gen_list = list(indices_data_gen)
            assert isinstance(indices_data_gen_list, list)
            assert len(indices_data_gen_list) == n_bootstraps
            # Unpack indices and data
            indices, data = zip(*indices_data_gen_list)
            assert isinstance(indices, tuple)
            assert len(indices) == n_bootstraps
            assert all(isinstance(i, list) for i in indices)
            assert all(np.prod(np.shape(i)) == X.shape[0] for i in indices)

            assert isinstance(data, tuple)
            assert len(data) == n_bootstraps
            assert all(isinstance(d, np.ndarray) for d in data)
            assert all(np.prod(np.shape(d)) == X.shape[0] for d in data)

            # Check that bootstrap.bootstrap method runs without errors
            bootstrap = WholeDistributionBootstrap(config=config)
            indices_data_gen = bootstrap.bootstrap(
                np.array(X), return_indices=True, test_ratio=0.2
            )
            indices_data_gen_list = list(indices_data_gen)
            assert isinstance(indices_data_gen_list, list)
            assert len(indices_data_gen_list) == n_bootstraps
            # Unpack indices and data
            indices, data = zip(*indices_data_gen_list)
            assert isinstance(indices, tuple)
            assert len(indices) == n_bootstraps
            assert all(isinstance(i, list) for i in indices)
            assert all(
                np.prod(np.shape(i)) == int(X.shape[0] * 0.8) for i in indices
            )

            assert isinstance(data, tuple)
            assert len(data) == n_bootstraps
            assert all(isinstance(d, np.ndarray) for d in data)
            assert all(
                np.prod(np.shape(d)) == int(X.shape[0] * 0.8) for d in data
            )

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
            n_bootstraps=integers(min_value=1, max_value=10),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
            X=X_strategy_univariate,
            distribution=sampled_from(["normal", "uniform"]),
            refit=booleans(),
        )
        def test_invalid_fit_model(
            self,
            model_type: str,
            order: int,
            save_models: bool,
            params: dict[str, int],
            n_bootstraps: int,
            rng: int,
            X: np.ndarray,
            distribution: str,
            refit: bool,
        ) -> None:
            """
            Test if the WholeDistributionBootstrap's _generate_samples_single_bootstrap method raises a ValueError when the fit_model method fails.
            """
            config = BaseDistributionBootstrapConfig(
                model_type=model_type,
                order=order,
                save_models=save_models,
                model_params=params,
                n_bootstraps=n_bootstraps,
                rng=rng,
                distribution=distribution,
                refit=refit,
            )
            bootstrap = WholeDistributionBootstrap(config=config)

            # Check that _generate_samples_single_bootstrap method raises a ValueError when the fit_model method fails
            with patch.object(
                TSFitBestLag, "fit", side_effect=ValueError
            ), pytest.raises(ValueError):
                bootstrap._generate_samples_single_bootstrap(np.array(X))


class TestBlockDistributionBootstrap:
    class TestPassingCases:
        @pytest.mark.skip(reason="known LU decomposition issue, see #41")
        @settings(deadline=None, max_examples=10)
        @given(
            model_type=model_strategy_univariate,
            order=integers(min_value=1, max_value=5),
            save_models=booleans(),
            bootstrap_type=sampled_from(list(BLOCK_BOOTSTRAP_TYPES_DICT)),
            block_length=integers(min_value=1, max_value=5),
            combine_generation_and_sampling_flag=booleans(),
            n_bootstraps=integers(min_value=1, max_value=10),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
            X=X_strategy_univariate,
            distribution=sampled_from(["normal", "uniform"]),
            refit=booleans(),
        )
        def test_block_statistic_preserving_bootstrap(
            self,
            model_type: str,
            order: int,
            save_models: bool,
            combine_generation_and_sampling_flag: bool,
            bootstrap_type: str,
            block_length: int,
            n_bootstraps: int,
            rng: int,
            X: np.ndarray,
            distribution: str,
            refit: bool,
        ) -> None:
            """
            Test if the BlockStatisticPreservingBootstrap class initializes correctly and if the _generate_samples_single_bootstrap method runs without errors.
            """
            distribution_config = BaseDistributionBootstrapConfig(
                model_type=model_type,
                order=order,
                save_models=save_models,
                n_bootstraps=n_bootstraps,
                rng=rng,
                distribution=distribution,
                refit=refit,
            )
            block_config = BaseBlockBootstrapConfig(
                bootstrap_type=bootstrap_type,
                block_length=block_length,
                combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            )
            bootstrap = BlockDistributionBootstrap(
                distribution_config=distribution_config,
                block_config=block_config,
            )

            assert bootstrap.config == distribution_config

            # Check that _generate_samples_single_bootstrap method runs without errors
            indices, data = bootstrap._generate_samples_single_bootstrap(
                np.array(X)
            )
            assert isinstance(indices, list)
            assert all(isinstance(i, np.ndarray) for i in indices)
            assert isinstance(data, list)
            assert all(isinstance(d, np.ndarray) for d in data)

            # Check that _generate_samples method runs without errors
            bootstrap = BlockDistributionBootstrap(
                distribution_config=distribution_config,
                block_config=block_config,
            )
            indices_data_gen = bootstrap._generate_samples(
                np.array(X), return_indices=True
            )
            indices_data_gen_list = list(indices_data_gen)
            assert isinstance(indices_data_gen_list, list)
            assert len(indices_data_gen_list) == n_bootstraps
            # Unpack indices and data
            indices, data = zip(*indices_data_gen_list)
            assert isinstance(indices, tuple)
            assert len(indices) == n_bootstraps
            assert all(isinstance(i, list) for i in indices)
            assert all(
                sum(len(i_iter) for i_iter in i) == X.shape[0] for i in indices
            )

            assert isinstance(data, tuple)
            assert len(data) == n_bootstraps
            assert all(isinstance(d, np.ndarray) for d in data)
            assert all(
                sum(len(d_iter) for d_iter in d) == X.shape[0] for d in data
            )

            # Check that bootstrap.bootstrap method runs without errors
            bootstrap = BlockDistributionBootstrap(
                distribution_config=distribution_config,
                block_config=block_config,
            )
            indices_data_gen = bootstrap.bootstrap(
                np.array(X), return_indices=True, test_ratio=0.2
            )
            indices_data_gen_list = list(indices_data_gen)
            assert isinstance(indices_data_gen_list, list)
            assert len(indices_data_gen_list) == n_bootstraps
            # Unpack indices and data
            indices, data = zip(*indices_data_gen_list)
            assert isinstance(indices, tuple)
            assert len(indices) == n_bootstraps
            assert all(isinstance(i, list) for i in indices)
            assert all(
                sum(len(i_iter) for i_iter in i) == int(X.shape[0] * 0.8)
                for i in indices
            )

            assert isinstance(data, tuple)
            assert len(data) == n_bootstraps
            assert all(isinstance(d, np.ndarray) for d in data)
            assert all(
                sum(len(d_iter) for d_iter in d) == int(X.shape[0] * 0.8)
                for d in data
            )

    class TestFailingCases:
        @settings(deadline=None, max_examples=10)
        @given(
            model_type=model_strategy_univariate,
            order=integers(min_value=1, max_value=5),
            save_models=booleans(),
            bootstrap_type=sampled_from(list(BLOCK_BOOTSTRAP_TYPES_DICT)),
            block_length=integers(min_value=1, max_value=5),
            combine_generation_and_sampling_flag=booleans(),
            n_bootstraps=integers(min_value=1, max_value=10),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
            X=X_strategy_univariate,
            distribution=sampled_from(["normal", "uniform"]),
            refit=booleans(),
        )
        def test_invalid_fit_model(
            self,
            model_type: str,
            order: int,
            save_models: bool,
            bootstrap_type: str,
            block_length: int,
            combine_generation_and_sampling_flag: bool,
            n_bootstraps: int,
            rng: int,
            X: np.ndarray,
            distribution: str,
            refit: bool,
        ) -> None:
            """
            Test if the BlockStatisticPreservingBootstrap's _generate_samples_single_bootstrap method raises a ValueError when the fit_model method fails.
            """
            distribution_config = BaseDistributionBootstrapConfig(
                model_type=model_type,
                order=order,
                save_models=save_models,
                n_bootstraps=n_bootstraps,
                rng=rng,
                distribution=distribution,
                refit=refit,
            )
            block_config = BaseBlockBootstrapConfig(
                bootstrap_type=bootstrap_type,
                block_length=block_length,
                combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            )
            bootstrap = BlockDistributionBootstrap(
                distribution_config=distribution_config,
                block_config=block_config,
            )

            # Check that _generate_samples_single_bootstrap method raises a ValueError when the fit_model method fails
            with patch.object(
                TSFitBestLag, "fit", side_effect=ValueError
            ), pytest.raises(ValueError):
                bootstrap._generate_samples_single_bootstrap(np.array(X))


class TestWholeSieveBootstrap:
    class TestPassingCases:
        @pytest.mark.skip(reason="known LU decomposition issue, see #41")
        @settings(deadline=None, max_examples=10)
        @given(
            model_type=model_strategy_univariate,
            order=integers(min_value=1, max_value=4),
            save_models=booleans(),
            n_bootstraps=integers(min_value=1, max_value=10),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
            X=X_strategy_univariate,
            resids_model_type=resids_model_strategy_univariate,
            resids_order=integers(min_value=1, max_value=5),
            save_resids_models=booleans(),
        )
        def test_whole_sieve_bootstrap(
            self,
            model_type: str,
            order: int,
            save_models: bool,
            n_bootstraps: int,
            rng: int,
            X: np.ndarray,
            resids_model_type: str,
            resids_order: int,
            save_resids_models: bool,
        ) -> None:
            """
            Test if the WholeSieveBootstrap class initializes correctly and if the _generate_samples_single_bootstrap method runs without errors.
            """
            config = BaseSieveBootstrapConfig(
                model_type=model_type,
                order=order,
                save_models=save_models,
                n_bootstraps=n_bootstraps,
                rng=rng,
                resids_model_type=resids_model_type,
                resids_order=resids_order,
                save_resids_models=save_resids_models,
            )
            bootstrap = WholeSieveBootstrap(config=config)

            assert bootstrap.config == config

            # Check that _generate_samples_single_bootstrap method runs without errors
            indices, data = bootstrap._generate_samples_single_bootstrap(
                np.array(X)
            )
            assert isinstance(indices, list)
            assert len(indices) == 1
            assert isinstance(indices[0], np.ndarray)
            assert len(indices[0]) == X.shape[0]

            assert isinstance(data, list)
            assert len(data) == 1
            assert isinstance(data[0], np.ndarray)
            assert len(data[0]) == X.shape[0]

            # Check that _generate_samples method runs without errors
            bootstrap = WholeSieveBootstrap(config=config)
            indices_data_gen = bootstrap._generate_samples(
                np.array(X), return_indices=True
            )
            indices_data_gen_list = list(indices_data_gen)
            assert isinstance(indices_data_gen_list, list)
            assert len(indices_data_gen_list) == n_bootstraps
            # Unpack indices and data
            indices, data = zip(*indices_data_gen_list)
            assert isinstance(indices, tuple)
            assert len(indices)
            assert all(isinstance(i, list) for i in indices)
            assert all(np.prod(np.shape(i)) == X.shape[0] for i in indices)

            assert isinstance(data, tuple)
            assert len(data)
            assert all(isinstance(d, np.ndarray) for d in data)
            assert all(np.prod(np.shape(d)) == X.shape[0] for d in data)

            # Check that bootstrap.bootstrap method runs without errors
            bootstrap = WholeSieveBootstrap(config=config)
            indices_data_gen = bootstrap.bootstrap(
                np.array(X), return_indices=True, test_ratio=0.2
            )
            indices_data_gen_list = list(indices_data_gen)
            assert isinstance(indices_data_gen_list, list)
            assert len(indices_data_gen_list)
            # Unpack indices and data
            indices, data = zip(*indices_data_gen_list)
            assert isinstance(indices, tuple)
            assert len(indices)
            assert all(isinstance(i, list) for i in indices)

            assert isinstance(data, tuple)
            assert len(data)
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
            n_bootstraps=integers(min_value=1, max_value=10),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
            X=X_strategy_univariate,
            resids_model_type=resids_model_strategy_univariate,
            resids_order=integers(min_value=1, max_value=5),
            save_resids_models=booleans(),
        )
        def test_invalid_fit_model(
            self,
            model_type: str,
            order: int,
            save_models: bool,
            params: dict[str, int],
            n_bootstraps: int,
            rng: int,
            X: np.ndarray,
            resids_model_type: str,
            resids_order: int,
            save_resids_models: bool,
        ) -> None:
            """
            Test if the WholeSieveBootstrap's _generate_samples_single_bootstrap method raises a ValueError when the fit_model method fails.
            """
            config = BaseSieveBootstrapConfig(
                model_type=model_type,
                order=order,
                save_models=save_models,
                model_params=params,
                n_bootstraps=n_bootstraps,
                rng=rng,
                resids_model_type=resids_model_type,
                resids_order=resids_order,
                save_resids_models=save_resids_models,
            )
            bootstrap = WholeSieveBootstrap(config=config)

            # Check that _generate_samples_single_bootstrap method raises a ValueError when the fit_model method fails
            with patch.object(
                TSFitBestLag, "fit", side_effect=ValueError
            ), pytest.raises(ValueError):
                bootstrap._generate_samples_single_bootstrap(np.array(X))


class TestBlockSieveBootstrap:
    class TestPassingCases:
        @pytest.mark.skip(reason="known LU decomposition issue, see #41")
        @settings(deadline=None, max_examples=10)
        @given(
            model_type=model_strategy_univariate,
            order=integers(min_value=1, max_value=5),
            save_models=booleans(),
            bootstrap_type=sampled_from(list(BLOCK_BOOTSTRAP_TYPES_DICT)),
            block_length=integers(min_value=1, max_value=5),
            combine_generation_and_sampling_flag=booleans(),
            n_bootstraps=integers(min_value=1, max_value=10),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
            X=X_strategy_univariate,
            resids_model_type=resids_model_strategy_univariate,
            resids_order=integers(min_value=1, max_value=5),
            save_resids_models=booleans(),
        )
        def test_block_statistic_preserving_bootstrap(
            self,
            model_type: str,
            order: int,
            save_models: bool,
            combine_generation_and_sampling_flag: bool,
            bootstrap_type: str,
            block_length: int,
            n_bootstraps: int,
            rng: int,
            X: np.ndarray,
            resids_model_type: str,
            resids_order: int,
            save_resids_models: bool,
        ) -> None:
            """
            Test if the BlockStatisticPreservingBootstrap class initializes correctly and if the _generate_samples_single_bootstrap method runs without errors.
            """
            sieve_config = BaseSieveBootstrapConfig(
                model_type=model_type,
                order=order,
                save_models=save_models,
                n_bootstraps=n_bootstraps,
                rng=rng,
                resids_model_type=resids_model_type,
                resids_order=resids_order,
                save_resids_models=save_resids_models,
            )
            block_config = BaseBlockBootstrapConfig(
                bootstrap_type=bootstrap_type,
                block_length=block_length,
                combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            )
            bootstrap = BlockSieveBootstrap(
                sieve_config=sieve_config, block_config=block_config
            )

            assert bootstrap.config == sieve_config

            # Check that _generate_samples_single_bootstrap method runs without errors
            indices, data = bootstrap._generate_samples_single_bootstrap(
                np.array(X)
            )
            assert isinstance(indices, list)
            assert all(isinstance(i, np.ndarray) for i in indices)
            assert isinstance(data, list)
            assert all(isinstance(d, np.ndarray) for d in data)

            # Check that _generate_samples method runs without errors
            bootstrap = BlockSieveBootstrap(
                sieve_config=sieve_config, block_config=block_config
            )
            indices_data_gen = bootstrap._generate_samples(
                np.array(X), return_indices=True
            )
            indices_data_gen_list = list(indices_data_gen)
            assert isinstance(indices_data_gen_list, list)
            assert len(indices_data_gen_list) == n_bootstraps
            # Unpack indices and data
            indices, data = zip(*indices_data_gen_list)
            assert isinstance(indices, tuple)
            assert len(indices) == n_bootstraps
            assert all(isinstance(i, list) for i in indices)
            assert all(
                sum(len(i_iter) for i_iter in i) == X.shape[0] for i in indices
            )

            assert isinstance(data, tuple)
            assert len(data) == n_bootstraps
            assert all(isinstance(d, np.ndarray) for d in data)
            assert all(
                sum(len(d_iter) for d_iter in d) == X.shape[0] for d in data
            )

            # Check that bootstrap.bootstrap method runs without errors
            bootstrap = BlockSieveBootstrap(
                sieve_config=sieve_config, block_config=block_config
            )
            indices_data_gen = bootstrap.bootstrap(
                np.array(X), return_indices=True, test_ratio=0.2
            )
            indices_data_gen_list = list(indices_data_gen)
            assert isinstance(indices_data_gen_list, list)
            assert len(indices_data_gen_list) == n_bootstraps
            # Unpack indices and data
            indices, data = zip(*indices_data_gen_list)
            assert isinstance(indices, tuple)
            assert len(indices) == n_bootstraps
            assert all(isinstance(i, list) for i in indices)
            assert all(
                sum(len(i_iter) for i_iter in i) == int(X.shape[0] * 0.8)
                for i in indices
            )

            assert isinstance(data, tuple)
            assert len(data) == n_bootstraps
            assert all(isinstance(d, np.ndarray) for d in data)
            assert all(
                sum(len(d_iter) for d_iter in d) == int(X.shape[0] * 0.8)
                for d in data
            )

    class TestFailingCases:
        @settings(deadline=None, max_examples=10)
        @given(
            model_type=model_strategy_univariate,
            order=integers(min_value=1, max_value=5),
            save_models=booleans(),
            bootstrap_type=sampled_from(list(BLOCK_BOOTSTRAP_TYPES_DICT)),
            block_length=integers(min_value=1, max_value=5),
            combine_generation_and_sampling_flag=booleans(),
            n_bootstraps=integers(min_value=1, max_value=10),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
            X=X_strategy_univariate,
            resids_model_type=resids_model_strategy_univariate,
            resids_order=integers(min_value=1, max_value=5),
            save_resids_models=booleans(),
        )
        def test_invalid_fit_model(
            self,
            model_type: str,
            order: int,
            save_models: bool,
            bootstrap_type: str,
            block_length: int,
            combine_generation_and_sampling_flag: bool,
            n_bootstraps: int,
            rng: int,
            X: np.ndarray,
            resids_model_type: str,
            resids_order: int,
            save_resids_models: bool,
        ) -> None:
            """
            Test if the BlockStatisticPreservingBootstrap's _generate_samples_single_bootstrap method raises a ValueError when the fit_model method fails.
            """
            sieve_config = BaseSieveBootstrapConfig(
                model_type=model_type,
                order=order,
                save_models=save_models,
                n_bootstraps=n_bootstraps,
                rng=rng,
                resids_model_type=resids_model_type,
                resids_order=resids_order,
                save_resids_models=save_resids_models,
            )
            block_config = BaseBlockBootstrapConfig(
                bootstrap_type=bootstrap_type,
                block_length=block_length,
                combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            )
            bootstrap = BlockSieveBootstrap(
                sieve_config=sieve_config, block_config=block_config
            )

            # Check that _generate_samples_single_bootstrap method raises a ValueError when the fit_model method fails
            with patch.object(
                TSFitBestLag, "fit", side_effect=ValueError
            ), pytest.raises(ValueError):
                bootstrap._generate_samples_single_bootstrap(np.array(X))
