from collections.abc import Callable
from typing import get_args

import numpy as np
import pytest
from hypothesis import given
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
)
from tsbootstrap.base_bootstrap_configs import (
    BaseDistributionBootstrapConfig,
    BaseMarkovBootstrapConfig,
    BaseResidualBootstrapConfig,
    BaseSieveBootstrapConfig,
    BaseStatisticPreservingBootstrapConfig,
    BaseTimeSeriesBootstrapConfig,
)
from tsbootstrap.utils.types import (
    BlockCompressorTypes,
    ModelTypes,
    ModelTypesWithoutArch,
)


class TestBaseTimeSeriesBootstrapConfig:
    class TestPassingCases:
        @given(
            n_bootstraps=integers(min_value=1),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
        )
        def test_base_time_series_bootstrap_config(
            self, n_bootstraps: int, rng: int
        ) -> None:
            """
            Test if the BaseTimeSeriesBootstrapConfig initializes correctly with integer n_bootstraps and integer rng.
            """
            config = BaseTimeSeriesBootstrapConfig(
                n_bootstraps=n_bootstraps, rng=rng
            )
            assert isinstance(config.n_bootstraps, int)
            assert config.n_bootstraps == n_bootstraps
            assert isinstance(config.rng, np.random.Generator)

        @given(n_bootstraps=integers(min_value=1))
        def test_no_rng_provided(self, n_bootstraps: int) -> None:
            """
            Test if the BaseTimeSeriesBootstrapConfig initializes correctly when no rng is provided.
            """
            config = BaseTimeSeriesBootstrapConfig(n_bootstraps=n_bootstraps)
            assert isinstance(config.n_bootstraps, int)
            assert config.n_bootstraps == n_bootstraps
            assert isinstance(config.rng, np.random.Generator)

    class TestFailingCases:
        @given(n_bootstraps=one_of(text(), lists(integers()), floats()))
        def test_invalid_n_bootstraps_type(self, n_bootstraps) -> None:
            """
            Test if the BaseTimeSeriesBootstrapConfig raises a TypeError when n_bootstraps is not an integer.
            """
            with pytest.raises(TypeError):
                _ = BaseTimeSeriesBootstrapConfig(n_bootstraps=n_bootstraps)

        @given(n_bootstraps=integers(max_value=0))
        def test_invalid_n_bootstraps_value(self, n_bootstraps: int) -> None:
            """
            Test if the BaseTimeSeriesBootstrapConfig raises a ValueError when n_bootstraps is less than or equal to 0.
            """
            with pytest.raises(ValueError):
                _ = BaseTimeSeriesBootstrapConfig(n_bootstraps=n_bootstraps)

        @given(rng=one_of(text(), lists(integers()), floats()))
        def test_invalid_rng_type(self, rng) -> None:
            """
            Test if the BaseTimeSeriesBootstrapConfig raises a TypeError when rng is not an integer or None.
            """
            with pytest.raises(TypeError):
                _ = BaseTimeSeriesBootstrapConfig(rng=rng)


class TestBaseResidualBootstrapConfig:
    class TestPassingCases:
        @given(
            n_bootstraps=integers(min_value=1),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
            model_type=sampled_from(
                [str(arg) for arg in get_args(ModelTypesWithoutArch)]
            ),
            order=one_of(
                none(),
                integers(min_value=1),
                lists(integers(min_value=1), min_size=1, max_size=4),
            ),
            save_models=booleans(),
            kwargs=dictionaries(
                keys=text(min_size=1),
                values=one_of(text(), integers(), floats(), booleans()),
            ),
        )
        def test_base_residual_bootstrap_config(
            self,
            n_bootstraps: int,
            rng: int,
            model_type: str,
            order,
            save_models: bool,
            kwargs,
        ) -> None:
            """
            Test if the BaseResidualBootstrapConfig initializes correctly with valid inputs.
            """
            config = BaseResidualBootstrapConfig(
                n_bootstraps=n_bootstraps,
                rng=rng,
                model_type=model_type,
                order=order,
                save_models=save_models,
                **kwargs,
            )
            assert config.n_bootstraps == n_bootstraps
            assert isinstance(config.rng, np.random.Generator)
            assert config.model_type == model_type
            assert config.order == order
            assert config.save_models == save_models

    class TestFailingCases:
        @given(
            model_type=text().filter(
                lambda x: x
                not in [str(arg) for arg in get_args(ModelTypesWithoutArch)]
            )
        )
        def test_invalid_model_type(self, model_type: str) -> None:
            """
            Test if the BaseResidualBootstrapConfig raises a ValueError when model_type is not one of the ModelTypesWithoutArch.
            """
            with pytest.raises(ValueError):
                _ = BaseResidualBootstrapConfig(model_type=model_type)

        @given(order=one_of(text(), lists(text(), min_size=1), floats()))
        def test_invalid_order_type(self, order) -> None:
            """
            Test if the BaseResidualBootstrapConfig raises a TypeError when order is not an integer, list of integers, or None.
            """
            print(order)
            with pytest.raises(TypeError):
                _ = BaseResidualBootstrapConfig(order=order)

        @given(order=integers(max_value=0))
        def test_invalid_order_value(self, order) -> None:
            """
            Test if the BaseResidualBootstrapConfig raises a ValueError when order is less than or equal to 0.
            """
            with pytest.raises(ValueError):
                _ = BaseResidualBootstrapConfig(order=order)

        @given(
            save_models=one_of(
                none(), text(), lists(booleans()), integers(), floats()
            )
        )
        def test_invalid_save_models_type(self, save_models) -> None:
            """
            Test if the BaseResidualBootstrapConfig raises a TypeError when save_models is not a boolean.
            """
            with pytest.raises(TypeError):
                _ = BaseResidualBootstrapConfig(save_models=save_models)


class TestBaseMarkovBootstrapConfig:
    class TestPassingCases:
        @given(
            n_bootstraps=integers(min_value=1),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
            model_type=sampled_from(
                [str(arg) for arg in get_args(ModelTypesWithoutArch)]
            ),
            order=one_of(
                none(),
                integers(min_value=1),
                lists(integers(min_value=1), min_size=1, max_size=4),
            ),
            save_models=booleans(),
            kwargs_base_residual=dictionaries(
                keys=text(min_size=1),
                values=one_of(text(), integers(), floats(), booleans()),
            ),
            method=sampled_from(
                [str(arg) for arg in get_args(BlockCompressorTypes)]
            ),
            apply_pca_flag=booleans(),
            pca=none(),
            n_iter_hmm=integers(min_value=10),
            n_fits_hmm=integers(min_value=1),
            blocks_as_hidden_states_flag=booleans(),
            n_states=integers(min_value=2),
        )
        def test_base_markov_bootstrap_config(
            self,
            n_bootstraps: int,
            rng: int,
            model_type: str,
            order,
            save_models: bool,
            kwargs_base_residual,
            method: str,
            apply_pca_flag: bool,
            pca: None,
            n_iter_hmm: int,
            n_fits_hmm: int,
            blocks_as_hidden_states_flag: bool,
            n_states: int,
        ) -> None:
            """
            Test if the BaseMarkovBootstrapConfig initializes correctly with valid inputs.
            """
            config = BaseMarkovBootstrapConfig(
                n_bootstraps=n_bootstraps,
                rng=rng,
                model_type=model_type,
                order=order,
                save_models=save_models,
                method=method,
                apply_pca_flag=apply_pca_flag,
                pca=pca,
                n_iter_hmm=n_iter_hmm,
                n_fits_hmm=n_fits_hmm,
                blocks_as_hidden_states_flag=blocks_as_hidden_states_flag,
                n_states=n_states,
                **kwargs_base_residual,
            )
            assert config.n_bootstraps == n_bootstraps
            assert isinstance(config.rng, np.random.Generator)
            assert config.model_type == model_type
            assert config.order == order
            assert config.save_models == save_models
            assert config.method == method
            assert config.apply_pca_flag == apply_pca_flag
            assert config.pca == pca
            assert config.n_iter_hmm == n_iter_hmm
            assert config.n_fits_hmm == n_fits_hmm
            assert (
                config.blocks_as_hidden_states_flag
                == blocks_as_hidden_states_flag
            )
            assert config.n_states == n_states

    class TestFailingCases:
        @given(
            method=text().filter(
                lambda x: x
                not in [str(arg) for arg in get_args(BlockCompressorTypes)]
            )
        )
        def test_invalid_method(self, method: str) -> None:
            """
            Test if the BaseMarkovBootstrapConfig raises a ValueError when method is not one of the BlockCompressorTypes.
            """
            with pytest.raises(ValueError):
                _ = BaseMarkovBootstrapConfig(method=method)

        @given(
            apply_pca_flag=one_of(
                none(), text(), lists(booleans()), integers(), floats()
            )
        )
        def test_invalid_apply_pca_flag_type(self, apply_pca_flag) -> None:
            """
            Test if the BaseMarkovBootstrapConfig raises a TypeError when apply_pca_flag is not a boolean.
            """
            with pytest.raises(TypeError):
                _ = BaseMarkovBootstrapConfig(apply_pca_flag=apply_pca_flag)

        @given(pca=one_of(text(), integers(), floats(), lists(booleans())))
        def test_invalid_pca_type(self, pca) -> None:
            """
            Test if the BaseMarkovBootstrapConfig raises a TypeError when pca is not None.
            """
            with pytest.raises(TypeError):
                _ = BaseMarkovBootstrapConfig(pca=pca)

        @given(n_iter_hmm=integers(max_value=9))
        def test_invalid_n_iter_hmm_value(self, n_iter_hmm: int) -> None:
            """
            Test if the BaseMarkovBootstrapConfig raises a ValueError when n_iter_hmm is less than 10.
            """
            with pytest.raises(ValueError):
                _ = BaseMarkovBootstrapConfig(n_iter_hmm=n_iter_hmm)

        @given(n_fits_hmm=integers(max_value=0))
        def test_invalid_n_fits_hmm_value(self, n_fits_hmm: int) -> None:
            """
            Test if the BaseMarkovBootstrapConfig raises a ValueError when n_fits_hmm is less than 1.
            """
            with pytest.raises(ValueError):
                _ = BaseMarkovBootstrapConfig(n_fits_hmm=n_fits_hmm)

        @given(
            blocks_as_hidden_states_flag=one_of(
                none(), text(), lists(booleans()), integers(), floats()
            )
        )
        def test_invalid_blocks_as_hidden_states_flag_type(
            self, blocks_as_hidden_states_flag
        ) -> None:
            """
            Test if the BaseMarkovBootstrapConfig raises a TypeError when blocks_as_hidden_states_flag is not a boolean.
            """
            with pytest.raises(TypeError):
                _ = BaseMarkovBootstrapConfig(
                    blocks_as_hidden_states_flag=blocks_as_hidden_states_flag
                )

        @given(n_states=integers(max_value=1))
        def test_invalid_n_states_value(self, n_states: int) -> None:
            """
            Test if the BaseMarkovBootstrapConfig raises a ValueError when n_states is less than 2.
            """
            with pytest.raises(ValueError):
                _ = BaseMarkovBootstrapConfig(n_states=n_states)


class TestBaseStatisticPreservingBootstrapConfig:
    class TestPassingCases:
        @given(
            n_bootstraps=integers(min_value=1),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
            statistic=one_of(
                just(np.mean),
                just(np.median),
                just(np.sum),
                just(np.min),
                just(np.max),
            ),
            statistic_axis=integers(min_value=0),
            statistic_keepdims=booleans(),
        )
        def test_base_statistic_preserving_bootstrap_config(
            self,
            n_bootstraps: int,
            rng: int,
            statistic: Callable,
            statistic_axis: int,
            statistic_keepdims: bool,
        ) -> None:
            """
            Test if the BaseStatisticPreservingBootstrapConfig initializes correctly with valid inputs.
            """
            config = BaseStatisticPreservingBootstrapConfig(
                n_bootstraps=n_bootstraps,
                rng=rng,
                statistic=statistic,
                statistic_axis=statistic_axis,
                statistic_keepdims=statistic_keepdims,
            )
            assert config.n_bootstraps == n_bootstraps
            assert isinstance(config.rng, np.random.Generator)
            assert config.statistic == statistic
            assert config.statistic_axis == statistic_axis
            assert config.statistic_keepdims == statistic_keepdims

    class TestFailingCases:
        @given(
            statistic=one_of(none(), text(), integers(), floats(), booleans())
        )
        def test_invalid_statistic_type(self, statistic) -> None:
            """
            Test if the BaseStatisticPreservingBootstrapConfig raises a TypeError when statistic is not a callable.
            """
            with pytest.raises(TypeError):
                _ = BaseStatisticPreservingBootstrapConfig(statistic=statistic)

        @given(
            statistic_axis=one_of(none(), text(), lists(integers()), floats())
        )
        def test_invalid_statistic_axis_type(self, statistic_axis) -> None:
            """
            Test if the BaseStatisticPreservingBootstrapConfig raises a TypeError when statistic_axis is not an integer.
            """
            with pytest.raises(TypeError):
                _ = BaseStatisticPreservingBootstrapConfig(
                    statistic_axis=statistic_axis
                )

        @given(statistic_axis=integers(max_value=-1))
        def test_invalid_statistic_axis_value(
            self, statistic_axis: int
        ) -> None:
            """
            Test if the BaseStatisticPreservingBootstrapConfig raises a ValueError when statistic_axis is less than 0.
            """
            with pytest.raises(ValueError):
                _ = BaseStatisticPreservingBootstrapConfig(
                    statistic_axis=statistic_axis
                )

        @given(
            statistic_keepdims=one_of(
                none(), text(), lists(booleans()), integers(), floats()
            )
        )
        def test_invalid_statistic_keepdims_type(
            self, statistic_keepdims
        ) -> None:
            """
            Test if the BaseStatisticPreservingBootstrapConfig raises a TypeError when statistic_keepdims is not a boolean.
            """
            with pytest.raises(TypeError):
                _ = BaseStatisticPreservingBootstrapConfig(
                    statistic_keepdims=statistic_keepdims
                )


class TestBaseDistributionBootstrapConfig:
    class TestPassingCases:
        @given(
            n_bootstraps=integers(min_value=1),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
            model_type=sampled_from(
                [str(arg) for arg in get_args(ModelTypesWithoutArch)]
            ),
            order=one_of(
                none(),
                integers(min_value=1),
                lists(integers(min_value=1), min_size=1, max_size=4),
            ),
            save_models=booleans(),
            kwargs_base_residual=dictionaries(
                keys=text(min_size=1),
                values=one_of(text(), integers(), floats(), booleans()),
            ),
            distribution=sampled_from(
                list(BaseDistributionBootstrapConfig.distribution_methods)
            ),
            refit=booleans(),
        )
        def test_base_distribution_bootstrap_config(
            self,
            n_bootstraps: int,
            rng: int,
            model_type: str,
            order,
            save_models: bool,
            kwargs_base_residual,
            distribution: str,
            refit: bool,
        ) -> None:
            """
            Test if the BaseDistributionBootstrapConfig initializes correctly with valid inputs.
            """
            if model_type != "var":
                config = BaseDistributionBootstrapConfig(
                    n_bootstraps=n_bootstraps,
                    rng=rng,
                    model_type=model_type,
                    order=order,
                    save_models=save_models,
                    distribution=distribution,
                    refit=refit,
                    **kwargs_base_residual,
                )
                assert config.n_bootstraps == n_bootstraps
                assert isinstance(config.rng, np.random.Generator)
                assert config.model_type == model_type
                assert config.order == order
                assert config.save_models == save_models
                assert config.distribution == distribution
                assert config.refit == refit

    class TestFailingCases:
        @given(
            distribution=text().filter(
                lambda x: x
                not in BaseDistributionBootstrapConfig.distribution_methods
            )
        )
        def test_invalid_distribution(self, distribution: str) -> None:
            """
            Test if the BaseDistributionBootstrapConfig raises a ValueError when distribution is not one of the distribution_methods.
            """
            with pytest.raises(ValueError):
                _ = BaseDistributionBootstrapConfig(distribution=distribution)

        @given(
            refit=one_of(
                none(), text(), lists(booleans()), integers(), floats()
            )
        )
        def test_invalid_refit_type(self, refit) -> None:
            """
            Test if the BaseDistributionBootstrapConfig raises a TypeError when refit is not a boolean.
            """
            with pytest.raises(TypeError):
                _ = BaseDistributionBootstrapConfig(refit=refit)


class TestBaseSieveBootstrapConfig:
    class TestPassingCases:
        @given(
            n_bootstraps=integers(min_value=1),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
            model_type=sampled_from(
                [str(arg) for arg in get_args(ModelTypesWithoutArch)]
            ),
            order=one_of(
                none(),
                integers(min_value=1),
                lists(integers(min_value=1), min_size=1, max_size=4),
            ),
            save_models=booleans(),
            kwargs_base_residual=dictionaries(
                keys=text(min_size=1),
                values=one_of(text(), integers(), floats(), booleans()),
            ),
            resids_model_type=sampled_from(
                [str(arg) for arg in get_args(ModelTypes)]
            ),
            resids_order=one_of(
                none(),
                integers(min_value=1),
                lists(integers(min_value=1), min_size=1, max_size=4),
            ),
            save_resids_models=booleans(),
            kwargs_base_sieve=dictionaries(
                keys=text(min_size=1),
                values=one_of(text(), integers(), floats(), booleans()),
            ),
        )
        def test_base_sieve_bootstrap_config(
            self,
            n_bootstraps: int,
            rng: int,
            model_type: str,
            order,
            save_models: bool,
            kwargs_base_residual,
            resids_model_type: str,
            resids_order,
            save_resids_models: bool,
            kwargs_base_sieve,
        ) -> None:
            """
            Test if the BaseSieveBootstrapConfig initializes correctly with valid inputs.
            """
            if resids_model_type == "var" and model_type != "var":
                pass
            else:
                config = BaseSieveBootstrapConfig(
                    n_bootstraps=n_bootstraps,
                    rng=rng,
                    model_type=model_type,
                    order=order,
                    save_models=save_models,
                    resids_model_type=resids_model_type,
                    resids_order=resids_order,
                    save_resids_models=save_resids_models,
                    kwargs_base_sieve=kwargs_base_sieve,
                    **kwargs_base_residual,
                )
                assert config.n_bootstraps == n_bootstraps
                assert isinstance(config.rng, np.random.Generator)
                assert config.model_type == model_type
                assert config.order == order
                assert config.save_models == save_models
                if model_type == "var":
                    assert config.resids_model_type == "var"
                else:
                    assert config.resids_model_type == resids_model_type
                assert config.resids_order == resids_order
                assert config.save_resids_models == save_resids_models
                assert config.resids_model_params == kwargs_base_sieve

    class TestFailingCases:
        @given(
            resids_model_type=text().filter(
                lambda x: x not in [str(arg) for arg in get_args(ModelTypes)]
            )
        )
        def test_invalid_resids_model_type(
            self, resids_model_type: str
        ) -> None:
            """
            Test if the BaseSieveBootstrapConfig raises a ValueError when resids_model_type is not one of the ModelTypes.
            """
            with pytest.raises(ValueError):
                _ = BaseSieveBootstrapConfig(
                    resids_model_type=resids_model_type
                )

        @given(
            resids_order=one_of(text(), lists(text(), min_size=1), floats())
        )
        def test_invalid_resids_order_type(self, resids_order) -> None:
            """
            Test if the BaseSieveBootstrapConfig raises a TypeError when resids_order is not an integer, list of integers, or None.
            """
            with pytest.raises(TypeError):
                _ = BaseSieveBootstrapConfig(resids_order=resids_order)

        @given(resids_order=integers(max_value=0))
        def test_invalid_resids_order_value(self, resids_order: int) -> None:
            """
            Test if the BaseSieveBootstrapConfig raises a ValueError when resids_order is less than or equal to 0.
            """
            with pytest.raises(ValueError):
                _ = BaseSieveBootstrapConfig(resids_order=resids_order)

        @given(
            save_resids_models=one_of(
                none(), text(), lists(booleans()), integers(), floats()
            )
        )
        def test_invalid_save_resids_models_type(
            self, save_resids_models
        ) -> None:
            """
            Test if the BaseSieveBootstrapConfig raises a TypeError when save_resids_models is not a boolean.
            """
            with pytest.raises(TypeError):
                _ = BaseSieveBootstrapConfig(
                    save_resids_models=save_resids_models
                )
