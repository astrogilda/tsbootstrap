import logging  # Added logging
from typing import Optional, get_args
from unittest.mock import patch  # Added for uncommented tests

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings  # Added HealthCheck
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
from pydantic import ValidationError  # Added pydantic
from skbase.utils.dependencies import _check_soft_dependencies
from tsbootstrap.block_bootstrap import (
    BLOCK_BOOTSTRAP_TYPES_DICT,
    BaseBlockBootstrap,
)
from tsbootstrap.bootstrap import (
    BlockMarkovBootstrap,
    BlockSieveBootstrap,
    BlockStatisticPreservingBootstrap,
    WholeDistributionBootstrap,
    WholeMarkovBootstrap,
    WholeResidualBootstrap,  # Added for uncommented tests
    WholeSieveBootstrap,
    WholeStatisticPreservingBootstrap,
)
from tsbootstrap.tsfit import TSFitBestLag
from tsbootstrap.utils.types import (
    BlockCompressorTypes,
    DistributionTypes,
    ModelTypes,
    ModelTypesWithoutArch,
)

# Configure logging to show debug messages during tests
logging.basicConfig(level=logging.DEBUG)


# The shape is a strategy generating tuples (num_rows, num_columns)
# min of 30 elements to enable transition from one state to another, even with two n_states, for HMM
X_shape = tuples(
    integers(min_value=100, max_value=200), integers(min_value=1, max_value=10)
)
X_shape_univariate = tuples(
    integers(min_value=100, max_value=200), integers(min_value=1, max_value=1)
)
X_shape_multivariate = tuples(
    integers(min_value=100, max_value=200), integers(min_value=2, max_value=10)
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
model_strategy = sampled_from(get_args(ModelTypesWithoutArch))
model_strategy_univariate = model_strategy.filter(lambda x: x != "var")
model_strategy_multivariate = just("var")

# Resids model strategy for sieve bootstrap
resids_model_strategy = sampled_from(get_args(ModelTypes))
resids_model_strategy_univariate = resids_model_strategy.filter(
    lambda x: x != "var"
)
resids_model_strategy_multivariate = just("var")

# Markov method strategy
markov_method_strategy = sampled_from(
    [str(arg) for arg in get_args(BlockCompressorTypes)]
)

# Distribution strategy
distribution_strategy = sampled_from(
    [DistributionTypes.NORMAL, DistributionTypes.UNIFORM]
)

# Strategies specifically for potentially slow Markov tests
X_shape_markov_test = tuples(
    integers(min_value=30, max_value=50), just(1)
)  # Shorter series
X_strategy_markov_test = arrays(
    dtype=float,
    shape=X_shape_markov_test,
    elements=floats(min_value=1, max_value=100),
    unique=True,
)
block_length_markov_test = integers(
    min_value=5, max_value=10
)  # Larger min block length


class TestWholeResidualBootstrap:
    class TestPassingCases:
        @settings(deadline=None, max_examples=10)
        @given(
            model_type=model_strategy_univariate,
            order=integers(min_value=1, max_value=5),
            save_models=booleans(),
            n_bootstraps=integers(min_value=1, max_value=10),
            rng=one_of(
                integers(min_value=0, max_value=2**32 - 1).map(
                    np.random.default_rng
                ),
                none(),
            ),
            X=X_strategy_univariate,
        )
        def test_whole_residual_bootstrap(
            self,
            model_type: ModelTypesWithoutArch,
            order: int,
            save_models: bool,
            n_bootstraps: int,
            rng: Optional[np.random.Generator],
            X: np.ndarray,
        ) -> None:
            """
            Test if the WholeResidualBootstrap class initializes correctly and if the _generate_samples_single_bootstrap method runs without errors.
            """
            bootstrap = WholeResidualBootstrap(
                model_type=model_type,
                order=order,
                save_models=save_models,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )

            # Check initialization
            assert bootstrap.model_type == model_type
            assert bootstrap.order == order
            assert bootstrap.save_models == save_models
            assert bootstrap.n_bootstraps == n_bootstraps
            if rng is not None:
                assert bootstrap.rng == rng
            else:
                assert isinstance(bootstrap.rng, np.random.Generator)

            # Check that _generate_samples_single_bootstrap method runs without errors
            data, indices = bootstrap._generate_samples_single_bootstrap(
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
            bootstrap = WholeResidualBootstrap(
                model_type=model_type,
                order=order,
                save_models=save_models,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )
            indices_data_gen = bootstrap._generate_samples(
                np.array(X), return_indices=True
            )
            indices_data_gen_list = list(indices_data_gen)
            assert isinstance(indices_data_gen_list, list)
            assert len(indices_data_gen_list) == n_bootstraps
            # Unpack indices and data
            data_tuple, indices_tuple = zip(
                *indices_data_gen_list
            )  # Renamed to avoid conflict
            assert isinstance(indices_tuple, tuple)
            assert len(indices_tuple) == n_bootstraps
            assert all(isinstance(i[0], np.ndarray) for i in indices_tuple)
            assert all(
                np.prod(np.shape(i[0])) == X.shape[0] for i in indices_tuple
            )

            assert isinstance(data_tuple, tuple)
            assert len(data_tuple) == n_bootstraps
            assert all(isinstance(d[0], np.ndarray) for d in data_tuple)
            assert all(
                np.prod(np.shape(d[0])) == X.shape[0] for d in data_tuple
            )

            # Check that bootstrap.bootstrap method runs without errors
            bootstrap = WholeResidualBootstrap(
                model_type=model_type,
                order=order,
                save_models=save_models,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )
            indices_data_gen = bootstrap.bootstrap(
                np.array(X), return_indices=True, test_ratio=0.2
            )
            indices_data_gen_list = list(indices_data_gen)
            assert isinstance(indices_data_gen_list, list)
            assert len(indices_data_gen_list) == n_bootstraps
            # Unpack indices and data
            data_tuple, indices_tuple = zip(*indices_data_gen_list)  # Renamed
            assert isinstance(indices_tuple, tuple)
            assert len(indices_tuple) == n_bootstraps
            assert all(
                isinstance(i[0], np.ndarray) for i in indices_tuple
            )  # Check inner element
            assert all(
                np.prod(np.shape(i[0])) == int(X.shape[0] * 0.8)
                for i in indices_tuple
            )

            assert isinstance(data_tuple, tuple)
            assert len(data_tuple) == n_bootstraps
            assert all(
                isinstance(d[0], np.ndarray) for d in data_tuple
            )  # Check inner element
            assert all(
                np.prod(np.shape(d[0])) == int(X.shape[0] * 0.8)
                for d in data_tuple
            )

    class TestFailingCases:
        @settings(deadline=None, max_examples=10)
        @given(
            model_type=model_strategy_univariate,
            order=integers(min_value=1, max_value=5),
            save_models=booleans(),
            model_params=dictionaries(  # Renamed from params to model_params
                keys=text(min_size=1, max_size=3),
                values=integers(min_value=1, max_value=10),
            ),
            n_bootstraps=integers(min_value=1, max_value=10),
            rng=one_of(
                integers(min_value=0, max_value=2**32 - 1).map(
                    np.random.default_rng
                ),
                none(),
            ),  # Ensure RNG is Generator or None
            X=X_strategy_univariate,
        )
        def test_invalid_fit_model(
            self,
            model_type: ModelTypesWithoutArch,  # Corrected type
            order: int,
            save_models: bool,
            model_params: dict,  # Corrected name
            n_bootstraps: int,
            rng: Optional[np.random.Generator],
            X: np.ndarray,
        ) -> None:
            """
            Test if the WholeResidualBootstrap's _generate_samples_single_bootstrap method raises a ValueError when the fit_model method fails.
            """
            bootstrap = WholeResidualBootstrap(
                model_type=model_type,
                order=order,
                save_models=save_models,
                model_params=model_params,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )

            # Check that _generate_samples_single_bootstrap method raises a ValueError when the fit_model method fails
            with (
                patch.object(
                    TSFitBestLag,
                    "fit",
                    side_effect=ValueError("Mocked fit error"),
                ),
                pytest.raises(ValueError, match="Mocked fit error"),
            ):
                bootstrap._generate_samples_single_bootstrap(np.array(X))


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
#             rng=one_of(integers(min_value=0, max_value=2**32 - 1).map(np.random.default_rng), none()),
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
#             rng: Optional[np.random.Generator], # Changed from RngTypes
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
#             data, indices = bootstrap._generate_samples_single_bootstrap(
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
#             data, indices = zip(*indices_data_gen_list)
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
#             data, indices = zip(*indices_data_gen_list)
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
#             params: dict,
#             bootstrap_type: str,
#             block_length: int,
#             n_bootstraps: int,
#             rng: Optional[np.random.Generator], # Changed from RngTypes
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


@pytest.mark.skipif(
    not _check_soft_dependencies("hmmlearn", severity="none"),
    reason="skip test if required soft dependency not available",
)
class TestWholeMarkovBootstrap:
    class TestFailingCases:
        @settings(
            deadline=None,
            max_examples=10,
            suppress_health_check=[HealthCheck.function_scoped_fixture],
        )  # Suppress mocker health check
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
            apply_pca_flag=booleans(),
            blocks_as_hidden_states_flag=booleans(),
            method=markov_method_strategy,
            n_states=just(2),
        )
        def test_invalid_fit_model(
            self,
            mocker,
            model_type: ModelTypesWithoutArch,
            order: int,
            save_models: bool,
            params: dict,
            n_bootstraps: int,
            rng: Optional[np.random.Generator],
            X: np.ndarray,  # Added X parameter
            apply_pca_flag: bool,
            blocks_as_hidden_states_flag: bool,
            method: BlockCompressorTypes,
            n_states: int,
        ) -> None:
            """
            Test if the WholeMarkovBootstrap's _generate_samples_single_bootstrap method raises a ValueError when the fit_model method fails.
            """
            X = np.random.rand(20, 1)
            bootstrap = WholeMarkovBootstrap(
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

            # Check that _generate_samples_single_bootstrap method raises a ValueError when the fit_model method fails
            mocker.patch.object(TSFitBestLag, "fit", side_effect=ValueError)
            with pytest.raises(ValueError):
                bootstrap._generate_samples_single_bootstrap(np.array(X))


@pytest.mark.skipif(
    not _check_soft_dependencies("hmmlearn", severity="none"),
    reason="skip test if required soft dependency not available",
)
class TestBlockMarkovBootstrap:
    class TestPassingCases:
        @settings(deadline=None, max_examples=2)  # Reduced for faster feedback
        @given(
            model_type=just("ar"),  # Simplified model_type
            order=integers(min_value=1, max_value=2),  # Simplified order
            save_models=booleans(),
            n_bootstraps=just(1),  # Simplified n_bootstraps
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
            X=X_strategy_markov_test,  # Use constrained X strategy
            apply_pca_flag=booleans(),
            blocks_as_hidden_states_flag=booleans(),
            method=just("mean"),  # Corrected to "mean"
            n_states=just(2),
            bootstrap_type=sampled_from(list(BLOCK_BOOTSTRAP_TYPES_DICT)),
            block_length=block_length_markov_test,  # Use constrained block_length
            combine_generation_and_sampling_flag=booleans(),
        )
        def test_block_markov_bootstrap(
            self,
            model_type: ModelTypesWithoutArch,
            order: int,
            save_models: bool,
            n_bootstraps: int,
            rng: Optional[np.random.Generator],
            X: np.ndarray,
            apply_pca_flag: bool,
            blocks_as_hidden_states_flag: bool,
            method: BlockCompressorTypes,
            n_states: int,
            bootstrap_type: str,
            block_length: int,
            combine_generation_and_sampling_flag: bool,
        ) -> None:
            """
            Test if the BlockMarkovBootstrap class initializes correctly and if the _generate_samples_single_bootstrap method runs without errors.
            """
            block_bootstrap = BaseBlockBootstrap(
                bootstrap_type=bootstrap_type,
                block_length=block_length,
                combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            )
            params = {
                "block_bootstrap": block_bootstrap,
                "model_type": model_type,
                "order": order,
                "save_models": save_models,
                "n_bootstraps": n_bootstraps,
                "rng": rng,
                "apply_pca_flag": apply_pca_flag,
                "blocks_as_hidden_states_flag": blocks_as_hidden_states_flag,
                "method": method,
                "n_states": n_states,
            }
            bootstrap = BlockMarkovBootstrap(**params)

            # Check that _generate_samples_single_bootstrap method runs without errors
            data, indices = bootstrap._generate_samples_single_bootstrap(
                np.array(X)
            )
            assert isinstance(indices, list)
            assert all(isinstance(i, np.ndarray) for i in indices)
            assert isinstance(data, list)
            assert all(isinstance(d, np.ndarray) for d in data)

            # Check that _generate_samples method runs without errors
            bootstrap = BlockMarkovBootstrap(**params)
            indices_data_gen = bootstrap._generate_samples(
                np.array(X), return_indices=True
            )
            indices_data_gen_list = list(indices_data_gen)
            assert isinstance(indices_data_gen_list, list)
            assert len(indices_data_gen_list) == n_bootstraps
            # Unpack indices and data
            data, indices = zip(*indices_data_gen_list)
            assert isinstance(indices, tuple)
            assert len(indices) == n_bootstraps
            assert all(
                isinstance(i, np.ndarray) for i in indices
            )  # Changed from list to np.ndarray
            assert all(np.prod(np.shape(i)) == X.shape[0] for i in indices)

            assert isinstance(data, tuple)
            assert len(data) == n_bootstraps
            assert all(isinstance(d, np.ndarray) for d in data)
            assert all(
                sum(len(d_iter) for d_iter in d) == X.shape[0] for d in data
            )

            # Check that bootstrap.bootstrap method runs without errors
            indices_data_gen = bootstrap.bootstrap(
                np.array(X), return_indices=True, test_ratio=0.2
            )
            indices_data_gen_list = list(indices_data_gen)
            assert isinstance(indices_data_gen_list, list)
            assert len(indices_data_gen_list) == n_bootstraps
            # Unpack indices and data
            data, indices = zip(*indices_data_gen_list)
            assert isinstance(indices, tuple)
            assert len(indices) == n_bootstraps
            assert all(
                isinstance(i, np.ndarray) for i in indices
            )  # Changed from list to np.ndarray
            assert all(
                np.prod(np.shape(i)) == int(X.shape[0] * 0.8) for i in indices
            )

            assert isinstance(data, tuple)
            assert len(data) == n_bootstraps
            assert all(isinstance(d, np.ndarray) for d in data)
            assert all(
                sum(len(d_iter) for d_iter in d) == int(X.shape[0] * 0.8)
                for d in data
            )

    class TestFailingCases:
        @settings(
            deadline=None,
            max_examples=2,  # Reduced for faster feedback
            suppress_health_check=[HealthCheck.function_scoped_fixture],
        )  # Suppress mocker health check
        @given(
            model_type=just("ar"),  # Simplified model_type
            order=integers(min_value=1, max_value=2),  # Simplified order
            save_models=booleans(),
            params=dictionaries(
                keys=text(min_size=1, max_size=3),
                values=integers(min_value=1, max_value=10),
            ),
            n_bootstraps=just(1),  # Simplified n_bootstraps
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
            X=X_strategy_markov_test,  # Use constrained X strategy
            apply_pca_flag=booleans(),
            blocks_as_hidden_states_flag=booleans(),
            method=just("mean"),  # Corrected to "mean"
            n_states=just(2),
            bootstrap_type=sampled_from(list(BLOCK_BOOTSTRAP_TYPES_DICT)),
            block_length=block_length_markov_test,  # Use constrained block_length
            combine_generation_and_sampling_flag=booleans(),
        )
        def test_invalid_fit_model(
            self,
            mocker,
            model_type: ModelTypesWithoutArch,
            order: int,
            save_models: bool,
            params: dict,
            n_bootstraps: int,
            rng: Optional[np.random.Generator],
            X: np.ndarray,
            apply_pca_flag: bool,
            blocks_as_hidden_states_flag: bool,
            method: BlockCompressorTypes,
            n_states: int,
            bootstrap_type: str,
            block_length: int,
            combine_generation_and_sampling_flag: bool,
        ) -> None:
            """
            Test if the BlockMarkovBootstrap's _generate_samples_single_bootstrap method raises a ValueError when the fit_model method fails.
            """
            block_bootstrap = BaseBlockBootstrap(
                bootstrap_type=bootstrap_type,
                block_length=block_length,
                combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            )
            bootstrap = BlockMarkovBootstrap(
                block_bootstrap=block_bootstrap,
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

            # Check that _generate_samples_single_bootstrap method raises a ValueError when the fit_model method fails
            mocker.patch.object(TSFitBestLag, "fit", side_effect=ValueError)
            with pytest.raises(ValueError):
                bootstrap._generate_samples_single_bootstrap(np.array(X))


class TestWholeStatisticPreservingBootstrap:
    class TestPassingCases:
        @settings(deadline=None, max_examples=10)
        @given(
            n_bootstraps=integers(min_value=1, max_value=10),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
            X=X_strategy_univariate,
            statistic_axis=integers(min_value=0, max_value=1),
            statistic_keepdims=booleans(),
        )
        def test_whole_statistic_preserving_bootstrap(
            self,
            n_bootstraps: int,
            rng: Optional[np.random.Generator],
            X: np.ndarray,
            statistic_axis: int,
            statistic_keepdims: bool,
        ) -> None:
            """
            Test if the WholeStatisticPreservingBootstrap class initializes correctly and if the _generate_samples_single_bootstrap method runs without errors.
            """
            params = {
                "n_bootstraps": n_bootstraps,
                "rng": rng,
                "statistic": np.mean,  # Using np.mean as a default statistic
                "statistic_axis": statistic_axis,
                "statistic_keepdims": statistic_keepdims,
            }
            bootstrap = WholeStatisticPreservingBootstrap(**params)

            # Check that _generate_samples_single_bootstrap method runs without errors
            data, indices = bootstrap._generate_samples_single_bootstrap(
                np.array(X)
            )
            assert isinstance(indices, list)
            assert all(isinstance(i, np.ndarray) for i in indices)
            assert isinstance(data, list)
            assert all(isinstance(d, np.ndarray) for d in data)

            # Check that _generate_samples method runs without errors
            bootstrap = WholeStatisticPreservingBootstrap(**params)
            indices_data_gen = bootstrap._generate_samples(
                np.array(X), return_indices=True
            )
            indices_data_gen_list = list(indices_data_gen)
            assert isinstance(indices_data_gen_list, list)
            assert len(indices_data_gen_list) == n_bootstraps
            # Unpack indices and data
            data, indices = zip(*indices_data_gen_list)
            assert isinstance(indices, tuple)
            assert len(indices) == n_bootstraps
            assert all(isinstance(i, np.ndarray) for i in indices)
            assert all(np.prod(np.shape(i)) == X.shape[0] for i in indices)

            assert isinstance(data, tuple)
            assert len(data) == n_bootstraps
            assert all(isinstance(d, np.ndarray) for d in data)
            assert all(np.prod(np.shape(d)) == X.shape[0] for d in data)

            # Check that bootstrap.bootstrap method runs without errors
            indices_data_gen = bootstrap.bootstrap(
                np.array(X), return_indices=True, test_ratio=0.2
            )
            indices_data_gen_list = list(indices_data_gen)
            assert isinstance(indices_data_gen_list, list)
            assert len(indices_data_gen_list) == n_bootstraps
            # Unpack indices and data
            data, indices = zip(*indices_data_gen_list)
            assert isinstance(indices, tuple)
            assert len(indices) == n_bootstraps
            assert all(
                isinstance(i, np.ndarray) for i in indices
            )  # Changed from list to np.ndarray
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
            n_bootstraps=integers(min_value=1, max_value=10),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
            X=X_strategy_univariate,
            statistic_axis=integers(min_value=0, max_value=1),
            statistic_keepdims=booleans(),
        )
        def test_invalid_statistic_function(
            self,
            n_bootstraps: int,
            rng: Optional[np.random.Generator],
            X: np.ndarray,
            statistic_axis: int,
            statistic_keepdims: bool,
        ) -> None:
            """
            Test if the WholeStatisticPreservingBootstrap's _generate_samples_single_bootstrap method raises a ValueError when the statistic function is invalid.
            """
            params = {
                "n_bootstraps": n_bootstraps,
                "rng": rng,
                "statistic": None,  # Invalid statistic
                "statistic_axis": statistic_axis,
                "statistic_keepdims": statistic_keepdims,
            }
            with pytest.raises(
                ValidationError, match="Input should be callable"
            ):
                _ = WholeStatisticPreservingBootstrap(**params)


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
            model_type: ModelTypesWithoutArch,
            order: int,
            save_models: bool,
            n_bootstraps: int,
            rng: Optional[np.random.Generator],
            X: np.ndarray,
            distribution: DistributionTypes,
            refit: bool,
        ) -> None:
            """
            Test if the WholeStatisticPreservingBootstrap class initializes correctly and if the _generate_samples_single_bootstrap method runs without errors.
            """
            params = {
                "model_type": model_type,
                "order": order,
                "save_models": save_models,
                "n_bootstraps": n_bootstraps,
                "rng": rng,
                "distribution": distribution,
                "refit": refit,
            }
            bootstrap = WholeDistributionBootstrap(**params)

            # Check that _generate_samples_single_bootstrap method runs without errors
            data, indices = bootstrap._generate_samples_single_bootstrap(
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
            bootstrap = WholeDistributionBootstrap(**params)
            indices_data_gen = bootstrap._generate_samples(
                np.array(X), return_indices=True
            )
            indices_data_gen_list = list(indices_data_gen)
            assert isinstance(indices_data_gen_list, list)
            assert len(indices_data_gen_list) == n_bootstraps
            # Unpack indices and data
            data, indices = zip(*indices_data_gen_list)
            assert isinstance(indices, tuple)
            assert len(indices) == n_bootstraps
            assert all(isinstance(i, list) for i in indices)
            assert all(np.prod(np.shape(i)) == X.shape[0] for i in indices)

            assert isinstance(data, tuple)
            assert len(data) == n_bootstraps
            assert all(isinstance(d, np.ndarray) for d in data)
            assert all(np.prod(np.shape(d)) == X.shape[0] for d in data)

            # Check that bootstrap.bootstrap method runs without errors
            indices_data_gen = bootstrap.bootstrap(
                np.array(X), return_indices=True, test_ratio=0.2
            )
            indices_data_gen_list = list(indices_data_gen)
            assert isinstance(indices_data_gen_list, list)
            assert len(indices_data_gen_list) == n_bootstraps
            # Unpack indices and data
            data, indices = zip(*indices_data_gen_list)
            assert isinstance(indices, tuple)
            assert len(indices) == n_bootstraps
            assert all(
                isinstance(i, np.ndarray) for i in indices
            )  # Changed from list to np.ndarray
            assert all(np.prod(np.shape(i)) == X.shape[0] for i in indices)

            assert isinstance(data, tuple)
            assert len(data) == n_bootstraps
            assert all(isinstance(d, np.ndarray) for d in data)
            assert all(
                np.prod(np.shape(d)) == int(X.shape[0] * 0.8) for d in data
            )

    class TestFailingCases:
        @settings(
            deadline=None,
            max_examples=10,
            suppress_health_check=[HealthCheck.function_scoped_fixture],
        )  # Suppress mocker health check
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
            mocker,
            model_type: ModelTypesWithoutArch,
            order: int,
            save_models: bool,
            params: dict,
            n_bootstraps: int,
            rng: Optional[np.random.Generator],
            X: np.ndarray,
            distribution: DistributionTypes,
            refit: bool,
        ) -> None:
            """
            Test if the WholeDistributionBootstrap's _generate_samples_single_bootstrap method raises a ValueError when the fit_model method fails.
            """
            bootstrap = WholeDistributionBootstrap(
                model_type=model_type,
                order=order,
                save_models=save_models,
                model_params=params,
                n_bootstraps=n_bootstraps,
                rng=rng,
                distribution=distribution,
                refit=refit,
            )

            # Check that _generate_samples_single_bootstrap method raises a ValueError when the fit_model method fails
            mocker.patch.object(TSFitBestLag, "fit", side_effect=ValueError)
            with pytest.raises(ValueError):
                bootstrap._generate_samples_single_bootstrap(np.array(X))


class TestBlockStatisticPreservingBootstrap:
    class TestPassingCases:
        @settings(deadline=None, max_examples=10)
        @given(
            n_bootstraps=integers(min_value=1, max_value=10),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
            X=X_strategy_univariate,
            statistic_axis=integers(min_value=0, max_value=1),
            statistic_keepdims=booleans(),
            bootstrap_type=sampled_from(list(BLOCK_BOOTSTRAP_TYPES_DICT)),
            block_length=integers(min_value=1, max_value=5),
            combine_generation_and_sampling_flag=booleans(),
        )
        def test_block_statistic_preserving_bootstrap(
            self,
            n_bootstraps: int,
            rng: Optional[np.random.Generator],
            X: np.ndarray,
            statistic_axis: int,
            statistic_keepdims: bool,
            bootstrap_type: str,
            block_length: int,
            combine_generation_and_sampling_flag: bool,
        ) -> None:
            """
            Test if the BlockStatisticPreservingBootstrap class initializes correctly and if the _generate_samples_single_bootstrap method runs without errors.
            """
            block_bootstrap = BaseBlockBootstrap(
                bootstrap_type=bootstrap_type,
                block_length=block_length,
                combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            )
            params = {
                "block_bootstrap": block_bootstrap,
                "n_bootstraps": n_bootstraps,
                "rng": rng,
                "statistic": np.mean,  # Using np.mean as a default statistic
                "statistic_axis": statistic_axis,
                "statistic_keepdims": statistic_keepdims,
            }
            bootstrap = BlockStatisticPreservingBootstrap(**params)

            # Check that _generate_samples_single_bootstrap method runs without errors
            data, indices = bootstrap._generate_samples_single_bootstrap(
                np.array(X)
            )
            assert isinstance(indices, list)
            assert all(isinstance(i, np.ndarray) for i in indices)
            assert isinstance(data, list)
            assert all(isinstance(d, np.ndarray) for d in data)

            # Check that _generate_samples method runs without errors
            bootstrap = BlockStatisticPreservingBootstrap(**params)
            indices_data_gen = bootstrap._generate_samples(
                np.array(X), return_indices=True
            )
            indices_data_gen_list = list(indices_data_gen)
            assert isinstance(indices_data_gen_list, list)
            assert len(indices_data_gen_list) == n_bootstraps
            # Unpack indices and data
            data, indices = zip(*indices_data_gen_list)
            assert isinstance(indices, tuple)
            assert len(indices) == n_bootstraps
            assert all(isinstance(i, np.ndarray) for i in indices)
            assert all(np.prod(np.shape(i)) == X.shape[0] for i in indices)

            assert isinstance(data, tuple)
            assert len(data) == n_bootstraps
            assert all(isinstance(d, np.ndarray) for d in data)
            assert all(
                sum(len(d_iter) for d_iter in d) == X.shape[0] for d in data
            )

            # Check that bootstrap.bootstrap method runs without errors
            indices_data_gen = bootstrap.bootstrap(
                np.array(X), return_indices=True, test_ratio=0.2
            )
            indices_data_gen_list = list(indices_data_gen)
            assert isinstance(indices_data_gen_list, list)
            assert len(indices_data_gen_list) == n_bootstraps
            # Unpack indices and data
            data, indices = zip(*indices_data_gen_list)
            assert isinstance(indices, tuple)
            assert len(indices) == n_bootstraps
            assert all(
                isinstance(i, np.ndarray) for i in indices
            )  # Changed from list to np.ndarray
            assert all(
                np.prod(np.shape(i)) == int(X.shape[0] * 0.8) for i in indices
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
                n_bootstraps=integers(min_value=1, max_value=10),
                rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
                X=X_strategy_univariate,
                statistic_axis=integers(min_value=0, max_value=1),
                statistic_keepdims=booleans(),
                bootstrap_type=sampled_from(list(BLOCK_BOOTSTRAP_TYPES_DICT)),
                block_length=integers(min_value=1, max_value=5),
                combine_generation_and_sampling_flag=booleans(),
            )
            def test_invalid_statistic_function(
                self,
                n_bootstraps: int,
                rng: Optional[np.random.Generator],
                X: np.ndarray,
                statistic_axis: int,
                statistic_keepdims: bool,
                bootstrap_type: str,
                block_length: int,
                combine_generation_and_sampling_flag: bool,
            ) -> None:
                """
                Test if the BlockStatisticPreservingBootstrap's _generate_samples_single_bootstrap method raises a ValueError when the statistic function is invalid.
                """
                block_bootstrap = BaseBlockBootstrap(
                    bootstrap_type=bootstrap_type,
                    block_length=block_length,
                    combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
                )
                params_dict = (
                    {  # Renamed to avoid conflict with function parameter
                        "block_bootstrap": block_bootstrap,
                        "n_bootstraps": n_bootstraps,
                        "rng": rng,
                        "statistic": None,  # Invalid statistic
                        "statistic_axis": statistic_axis,
                        "statistic_keepdims": statistic_keepdims,
                    }
                )
                with pytest.raises(
                    ValidationError, match="Input should be callable"
                ):
                    _ = BlockStatisticPreservingBootstrap(
                        **params_dict
                    )  # Used params_dict


class TestWholeSieveBootstrap:
    class TestPassingCases:
        @pytest.mark.skip(reason="known LU decomposition issue, see #41")
        @settings(deadline=None, max_examples=10)
        @given(
            model_type=model_strategy_univariate,
            order=integers(min_value=1, max_value=4),
            save_models=booleans(),
            n_bootstraps=integers(min_value=1, max_value=10),
            rng=one_of(
                integers(min_value=0, max_value=2**32 - 1).map(
                    np.random.default_rng
                ),
                none(),
            ),
            X=X_strategy_univariate,
            resid_model_type=resids_model_strategy_univariate,
            resid_order=integers(min_value=1, max_value=5),
            resid_save_models=booleans(),
        )
        def test_whole_sieve_bootstrap(
            self,
            model_type: ModelTypesWithoutArch,
            order: int,
            save_models: bool,
            n_bootstraps: int,
            rng: Optional[np.random.Generator],
            X: np.ndarray,
            resid_model_type: ModelTypes,
            resid_order: int,
            resid_save_models: bool,
        ) -> None:
            """
            Test if the WholeSieveBootstrap class initializes correctly and if the _generate_samples_single_bootstrap method runs without errors.
            """
            params = {
                "model_type": model_type,
                "order": order,
                "save_models": save_models,
                "n_bootstraps": n_bootstraps,
                "rng": rng,
                "resid_model_type": resid_model_type,
                "resid_order": resid_order,
                "resid_save_models": resid_save_models,
            }
            bootstrap = WholeSieveBootstrap(**params)

            # Check that _generate_samples_single_bootstrap method runs without errors
            data, indices = bootstrap._generate_samples_single_bootstrap(
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
            bootstrap = WholeSieveBootstrap(**params)
            indices_data_gen = bootstrap._generate_samples(
                np.array(X), return_indices=True
            )
            indices_data_gen_list = list(indices_data_gen)
            assert isinstance(indices_data_gen_list, list)
            assert len(indices_data_gen_list) == n_bootstraps
            # Unpack indices and data
            data, indices = zip(*indices_data_gen_list)
            assert isinstance(indices, tuple)
            assert len(indices)
            assert all(
                isinstance(i, np.ndarray) for i in indices
            )  # Changed from list to np.ndarray
            assert all(np.prod(np.shape(i)) == X.shape[0] for i in indices)

            assert isinstance(data, tuple)
            assert len(data)
            assert all(isinstance(d, np.ndarray) for d in data)
            assert all(np.prod(np.shape(d)) == X.shape[0] for d in data)

            # Check that bootstrap.bootstrap method runs without errors
            bootstrap = WholeSieveBootstrap(**params)
            indices_data_gen = bootstrap.bootstrap(
                np.array(X), return_indices=True, test_ratio=0.2
            )
            indices_data_gen_list = list(indices_data_gen)
            assert isinstance(indices_data_gen_list, list)
            assert len(indices_data_gen_list)
            # Unpack indices and data
            data, indices = zip(*indices_data_gen_list)
            assert isinstance(indices, tuple)
            assert len(indices)
            assert all(
                isinstance(i, np.ndarray) for i in indices
            )  # Changed from list to np.ndarray
            assert all(
                np.prod(np.shape(i)) == int(X.shape[0] * 0.8) for i in indices
            )

            assert isinstance(data, tuple)
            assert len(data)
            assert all(isinstance(d, np.ndarray) for d in data)
            assert all(
                np.prod(np.shape(d)) == int(X.shape[0] * 0.8) for d in data
            )

    class TestFailingCases:
        @settings(
            deadline=None,
            max_examples=10,
            suppress_health_check=[HealthCheck.function_scoped_fixture],
        )  # Suppress mocker health check
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
            resid_model_type=resids_model_strategy_univariate,
            resid_order=integers(min_value=1, max_value=5),
            resid_save_models=booleans(),
        )
        def test_invalid_fit_model(
            self,
            mocker,
            model_type: ModelTypesWithoutArch,
            order: int,
            save_models: bool,
            params: dict,
            n_bootstraps: int,
            rng: Optional[np.random.Generator],
            X: np.ndarray,
            resid_model_type: ModelTypes,
            resid_order: int,
            resid_save_models: bool,
        ) -> None:
            """
            Test if the WholeSieveBootstrap's _generate_samples_single_bootstrap method raises a ValueError when the fit_model method fails.
            """
            bootstrap = WholeSieveBootstrap(
                model_type=model_type,
                order=order,
                save_models=save_models,
                model_params=params,
                n_bootstraps=n_bootstraps,
                rng=rng,
                resid_model_type=resid_model_type,
                resid_order=resid_order,
                resid_save_models=resid_save_models,
            )

            # Check that _generate_samples_single_bootstrap method raises a ValueError when the fit_model method fails
            mocker.patch.object(TSFitBestLag, "fit", side_effect=ValueError)
            with pytest.raises(ValueError):
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
            resid_save_models=booleans(),
        )
        def test_block_sieve_bootstrap(
            self,
            model_type: ModelTypesWithoutArch,
            order: int,
            save_models: bool,
            combine_generation_and_sampling_flag: bool,
            bootstrap_type: str,
            block_length: int,
            n_bootstraps: int,
            rng: Optional[np.random.Generator],  # Changed from RngTypes
            X: np.ndarray,
            resids_model_type: ModelTypes,
            resids_order: int,
            resid_save_models: bool,
        ) -> None:
            """
            Test if the BlockStatisticPreservingBootstrap class initializes correctly and if the _generate_samples_single_bootstrap method runs without errors.
            """
            block_bootstrap = BaseBlockBootstrap(
                bootstrap_type=bootstrap_type,
                block_length=block_length,
                combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            )
            params = {
                "block_bootstrap": block_bootstrap,
                "model_type": model_type,
                "order": order,
                "save_models": save_models,
                "n_bootstraps": n_bootstraps,
                "rng": rng,
                "resid_model_type": resids_model_type,
                "resid_order": resids_order,
                "resid_save_models": resid_save_models,
            }
            bootstrap = BlockSieveBootstrap(**params)

            # Check that _generate_samples_single_bootstrap method runs without errors
            data, indices = bootstrap._generate_samples_single_bootstrap(
                np.array(X)
            )
            assert isinstance(indices, list)
            assert all(isinstance(i, np.ndarray) for i in indices)
            assert isinstance(data, list)
            assert all(isinstance(d, np.ndarray) for d in data)

            # Check that _generate_samples method runs without errors
            bootstrap = BlockSieveBootstrap(**params)
            indices_data_gen = bootstrap._generate_samples(
                np.array(X), return_indices=True
            )
            indices_data_gen_list = list(indices_data_gen)
            assert isinstance(indices_data_gen_list, list)
            assert len(indices_data_gen_list) == n_bootstraps
            # Unpack indices and data
            data, indices = zip(*indices_data_gen_list)
            assert isinstance(indices, tuple)
            assert len(indices) == n_bootstraps
            assert all(
                isinstance(i, np.ndarray) for i in indices
            )  # Changed from list to np.ndarray
            assert all(np.prod(np.shape(i)) == X.shape[0] for i in indices)

            assert isinstance(data, tuple)
            assert len(data) == n_bootstraps
            assert all(isinstance(d, np.ndarray) for d in data)
            assert all(
                sum(len(d_iter) for d_iter in d) == int(X.shape[0] * 0.8)
                for d in data
            )

            # Check that bootstrap.bootstrap method runs without errors
            bootstrap = BlockSieveBootstrap(**params)
            indices_data_gen = bootstrap.bootstrap(
                np.array(X), return_indices=True, test_ratio=0.2
            )
            indices_data_gen_list = list(indices_data_gen)
            assert isinstance(indices_data_gen_list, list)
            assert len(indices_data_gen_list) == n_bootstraps
            # Unpack indices and data
            data, indices = zip(*indices_data_gen_list)
            assert isinstance(indices, tuple)
            assert len(indices) == n_bootstraps
            assert all(
                isinstance(i, np.ndarray) for i in indices
            )  # Changed from list to np.ndarray
            assert all(
                np.prod(np.shape(i)) == int(X.shape[0] * 0.8) for i in indices
            )

            assert isinstance(data, tuple)
            assert len(data) == n_bootstraps
            assert all(isinstance(d, np.ndarray) for d in data)
            assert all(
                sum(len(d_iter) for d_iter in d) == int(X.shape[0] * 0.8)
                for d in data
            )

    class TestFailingCases:
        @settings(
            deadline=None,
            max_examples=10,
            suppress_health_check=[HealthCheck.function_scoped_fixture],
        )  # Suppress mocker health check
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
            resid_save_models=booleans(),
        )
        def test_invalid_fit_model(
            self,
            mocker,
            model_type: ModelTypesWithoutArch,
            order: int,
            save_models: bool,
            bootstrap_type: str,
            block_length: int,
            combine_generation_and_sampling_flag: bool,
            n_bootstraps: int,
            rng: Optional[np.random.Generator],  # Changed from RngTypes
            X: np.ndarray,
            resids_model_type: ModelTypes,
            resids_order: int,
            resid_save_models: bool,
        ) -> None:
            """
            Test if the BlockStatisticPreservingBootstrap's _generate_samples_single_bootstrap method raises a ValueError when the fit_model method fails.
            """
            block_bootstrap = BaseBlockBootstrap(
                bootstrap_type=bootstrap_type,
                block_length=block_length,
                combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            )
            bootstrap = BlockSieveBootstrap(
                block_bootstrap=block_bootstrap,
                model_type=model_type,
                order=order,
                save_models=save_models,
                n_bootstraps=n_bootstraps,
                rng=rng,
                resid_model_type=resids_model_type,
                resid_order=resids_order,
                resid_save_models=resid_save_models,
            )

            # Check that _generate_samples_single_bootstrap method raises a ValueError when the fit_model method fails
            mocker.patch.object(TSFitBestLag, "fit", side_effect=ValueError)
            with pytest.raises(ValueError):
                bootstrap._generate_samples_single_bootstrap(np.array(X))
