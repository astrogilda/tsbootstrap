import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import (
    floats,
    integers,
    lists,
    none,
    one_of,
    sampled_from,
    text,
    tuples,
)
from tsbootstrap.block_bootstrap import (
    BartlettsBootstrap,
    BaseBlockBootstrap,
    BlackmanBootstrap,
    BlockBootstrap,
    CircularBlockBootstrap,
    HammingBootstrap,
    HanningBootstrap,
    MovingBlockBootstrap,
    NonOverlappingBlockBootstrap,
    StationaryBlockBootstrap,
    TukeyBootstrap,
)
from tsbootstrap.block_bootstrap_configs import (
    BLOCK_BOOTSTRAP_TYPES_DICT,
    BartlettsBootstrapConfig,
    BaseBlockBootstrapConfig,
    BlackmanBootstrapConfig,
    BlockBootstrapConfig,
    CircularBlockBootstrapConfig,
    HammingBootstrapConfig,
    HanningBootstrapConfig,
    MovingBlockBootstrapConfig,
    NonOverlappingBlockBootstrapConfig,
    StationaryBlockBootstrapConfig,
    TukeyBootstrapConfig,
)

# The shape is a strategy generating tuples (num_rows, num_columns)
X_shape = tuples(
    integers(min_value=6, max_value=100), integers(min_value=1, max_value=10)
)


class TestBlockBootstrap:
    class TestPassingCases:
        @settings(max_examples=10, deadline=None)
        @given(
            n_bootstraps=integers(min_value=1),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
            X=arrays(dtype=float, shape=X_shape),
        )
        def test_block_bootstrap(
            self,
            n_bootstraps: int,
            rng: int,
            X: np.ndarray,
        ) -> None:
            """
            Test if the BlockBootstrap class initializes correctly and if the bootstrap and _generate_blocks methods run without errors.
            """
            block_length = np.random.randint(1, int(0.8 * X.shape[0]) - 1)
            config = BlockBootstrapConfig(
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )
            bootstrap = BlockBootstrap(config=config)

            assert bootstrap.config == config
            assert bootstrap.blocks is None
            assert bootstrap.block_resampler is None

            # Check that bootstrap method runs without errors
            _ = list(bootstrap.bootstrap(np.array(X)))

            # Check that _generate_blocks method runs without errors and correctly sets blocks and block_resampler
            bootstrap._generate_blocks(np.array(X))
            assert bootstrap.blocks is not None
            assert bootstrap.block_resampler is not None

        @settings(max_examples=10, deadline=None)
        @given(
            X=lists(floats(), min_size=10, max_size=100),
            exog=one_of(none(), lists(floats(), min_size=10, max_size=100)),
        )
        def test__generate_samples_single_bootstrap(
            self, X: list[float], exog: list[float] | None
        ) -> None:
            """
            Test if the BlockBootstrap's _generate_samples_single_bootstrap method runs without errors and returns the correct output.
            """
            config = BlockBootstrapConfig(
                block_length=5,
                n_bootstraps=10,
                rng=42,
            )
            bootstrap = BlockBootstrap(config=config)

            # Generate blocks
            bootstrap._generate_blocks(np.array(X))

            # Check _generate_samples_single_bootstrap method
            indices, data = bootstrap._generate_samples_single_bootstrap(
                np.array(X), exog=exog if exog is None else np.array(exog)
            )

            assert isinstance(indices, list)
            assert all(isinstance(i, np.ndarray) for i in indices)
            assert isinstance(data, list)
            assert all(isinstance(d, np.ndarray) for d in data)

    class TestFailingCases:
        @settings(max_examples=10, deadline=None)
        @given(
            block_length=integers(max_value=0),
            n_bootstraps=integers(min_value=1),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
        )
        def test_invalid_block_length(
            self, block_length: int, n_bootstraps: int, rng: int
        ) -> None:
            """
            Test if the BlockBootstrap's __init__ method raises a ValueError when block_length is less than or equal to 0.
            """
            with pytest.raises(ValueError):
                _ = BlockBootstrapConfig(
                    block_length=block_length,
                    n_bootstraps=n_bootstraps,
                    rng=rng,
                )

        @settings(max_examples=10, deadline=None)
        @given(
            n_bootstraps=integers(min_value=1),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
            X=arrays(dtype=float, shape=X_shape),
        )
        def test_block_length_greater_than_input_size(
            self,
            n_bootstraps: int,
            rng: int,
            X: np.ndarray,
        ) -> None:
            """
            Test if the BlockBootstrap's _generate_blocks method raises a ValueError when block_length is greater than the size of the input array X.
            """
            block_length = np.random.randint(X.shape[0], X.shape[0] + 10)
            config = BlockBootstrapConfig(
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )
            bootstrap = BlockBootstrap(config=config)

            with pytest.raises(ValueError):
                bootstrap._generate_blocks(np.array(X))


class TestBaseBlockBootstrap:
    class TestPassingCases:
        @settings(max_examples=10, deadline=None)
        @given(
            bootstrap_type=sampled_from(list(BLOCK_BOOTSTRAP_TYPES_DICT)),
            n_bootstraps=integers(min_value=1),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
            X=arrays(dtype=float, shape=X_shape),
        )
        def test_base_block_bootstrap(
            self,
            bootstrap_type: str,
            n_bootstraps: int,
            rng: int,
            X: np.ndarray,
        ) -> None:
            """
            Test if the BaseBlockBootstrap class initializes correctly and if the bootstrap and _generate_samples_single_bootstrap methods run without errors.
            """
            block_length = np.random.randint(1, int(0.8 * X.shape[0]) - 1)
            config = BaseBlockBootstrapConfig(
                bootstrap_type=bootstrap_type,
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )
            bootstrap = BaseBlockBootstrap(config=config)
            print(f"bl : {bootstrap.config.block_length}")

            assert bootstrap.config == config
            assert isinstance(
                bootstrap.bootstrap_instance,
                BLOCK_BOOTSTRAP_TYPES_DICT[bootstrap_type],
            )

            # Check that bootstrap method runs without errors
            _ = list(bootstrap.bootstrap(np.array(X)))

            # Check that _generate_samples_single_bootstrap method runs without errors
            _ = bootstrap._generate_samples_single_bootstrap(X=np.array(X))

    class TestFailingCases:
        @settings(max_examples=10, deadline=None)
        @given(
            bootstrap_type=text().filter(
                lambda x: x not in list(BLOCK_BOOTSTRAP_TYPES_DICT)
            ),
            block_length=integers(min_value=1),
            n_bootstraps=integers(min_value=1),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
        )
        def test_invalid_bootstrap_type(
            self,
            bootstrap_type: str,
            block_length: int,
            n_bootstraps: int,
            rng: int,
        ) -> None:
            """
            Test if the BaseBlockBootstrap's __init__ method raises a ValueError when bootstrap_type is not one of the BLOCK_BOOTSTRAP_TYPES_DICT.
            """
            with pytest.raises(ValueError):
                _ = BaseBlockBootstrapConfig(
                    bootstrap_type=bootstrap_type,
                    block_length=block_length,
                    n_bootstraps=n_bootstraps,
                    rng=rng,
                )

        @settings(max_examples=10, deadline=None)
        @given(
            bootstrap_type=sampled_from(list(BLOCK_BOOTSTRAP_TYPES_DICT)),
            block_length=integers(min_value=1),
            n_bootstraps=integers(min_value=1),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
            X=arrays(dtype=float, shape=X_shape),
        )
        def test_not_implemented__generate_samples_single_bootstrap(
            self,
            bootstrap_type: str,
            block_length: int,
            n_bootstraps: int,
            rng: int,
            X: np.ndarray,
        ) -> None:
            """
            Test if the BaseBlockBootstrap's _generate_samples_single_bootstrap method raises a NotImplementedError when the bootstrap_type does not implement '_generate_samples_single_bootstrap' method.
            """
            config = BaseBlockBootstrapConfig(
                bootstrap_type=bootstrap_type,
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )
            bootstrap = BaseBlockBootstrap(config=config)

            # Check if the bootstrap_type implements '_generate_samples_single_bootstrap' method
            if not hasattr(
                bootstrap.bootstrap_instance,
                "_generate_samples_single_bootstrap",
            ):
                with pytest.raises(NotImplementedError):
                    bootstrap._generate_samples_single_bootstrap(np.array(X))


class TestMovingBlockBootstrap:
    class TestPassingCases:
        @given(
            block_length=integers(min_value=1),
            n_bootstraps=integers(min_value=1),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
        )
        def test_moving_block_bootstrap(
            self, block_length: int, n_bootstraps: int, rng: int
        ) -> None:
            """
            Test if the MovingBlockBootstrap class initializes correctly.
            """
            config = MovingBlockBootstrapConfig(
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )
            bootstrap = MovingBlockBootstrap(config=config)

            assert bootstrap.config == config
            assert bootstrap.config._wrap_around_flag is False
            assert bootstrap.config._overlap_flag is True
            assert bootstrap.config._block_length_distribution is None


class TestStationaryBlockBootstrap:
    class TestPassingCases:
        @given(
            block_length=integers(min_value=1),
            n_bootstraps=integers(min_value=1),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
        )
        def test_stationary_block_bootstrap(
            self, block_length: int, n_bootstraps: int, rng: int
        ) -> None:
            """
            Test if the StationaryBlockBootstrap class initializes correctly.
            """
            config = StationaryBlockBootstrapConfig(
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )
            bootstrap = StationaryBlockBootstrap(config=config)

            assert bootstrap.config == config
            assert bootstrap.config._wrap_around_flag is False
            assert bootstrap.config._overlap_flag is True
            assert bootstrap.config._block_length_distribution == "geometric"


class TestCircularBlockBootstrap:
    class TestPassingCases:
        @given(
            block_length=integers(min_value=1),
            n_bootstraps=integers(min_value=1),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
        )
        def test_circular_block_bootstrap(
            self, block_length: int, n_bootstraps: int, rng: int
        ) -> None:
            """
            Test if the CircularBlockBootstrap class initializes correctly.
            """
            config = CircularBlockBootstrapConfig(
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )
            bootstrap = CircularBlockBootstrap(config=config)

            assert bootstrap.config == config
            assert bootstrap.config._wrap_around_flag is True
            assert bootstrap.config._overlap_flag is True
            assert bootstrap.config._block_length_distribution is None


class TestNonOverlappingBlockBootstrap:
    class TestPassingCases:
        @given(
            block_length=integers(min_value=1),
            n_bootstraps=integers(min_value=1),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
        )
        def test_non_overlapping_block_bootstrap(
            self, block_length: int, n_bootstraps: int, rng: int
        ) -> None:
            """
            Test if the NonOverlappingBlockBootstrap class initializes correctly.
            """
            config = NonOverlappingBlockBootstrapConfig(
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )
            bootstrap = NonOverlappingBlockBootstrap(config=config)

            assert bootstrap.config == config
            assert bootstrap.config._wrap_around_flag is False
            assert bootstrap.config._overlap_flag is False
            assert bootstrap.config._block_length_distribution is None


class TestBartlettsBootstrap:
    class TestPassingCases:
        @given(
            block_length=integers(min_value=1),
            n_bootstraps=integers(min_value=1),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
        )
        def test_bartletts_bootstrap(
            self, block_length: int, n_bootstraps: int, rng: int
        ) -> None:
            """
            Test if the BartlettsBootstrap class initializes correctly.
            """
            config = BartlettsBootstrapConfig(
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )
            bootstrap = BartlettsBootstrap(config=config)

            assert bootstrap.config == config
            assert bootstrap.config.bootstrap_type == "moving"
            assert bootstrap.config.tapered_weights == np.bartlett


class TestHammingBootstrap:
    class TestPassingCases:
        @given(
            block_length=integers(min_value=1),
            n_bootstraps=integers(min_value=1),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
        )
        def test_hamming_bootstrap(
            self, block_length: int, n_bootstraps: int, rng: int
        ) -> None:
            """
            Test if the HammingBootstrap class initializes correctly.
            """
            config = HammingBootstrapConfig(
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )
            bootstrap = HammingBootstrap(config=config)

            assert bootstrap.config == config
            assert bootstrap.config.bootstrap_type == "moving"
            assert bootstrap.config.tapered_weights == np.hamming


class TestHanningBootstrap:
    class TestPassingCases:
        @given(
            block_length=integers(min_value=1),
            n_bootstraps=integers(min_value=1),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
        )
        def test_hanning_bootstrap(
            self, block_length: int, n_bootstraps: int, rng: int
        ) -> None:
            """
            Test if the HanningBootstrap class initializes correctly.
            """
            config = HanningBootstrapConfig(
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )
            bootstrap = HanningBootstrap(config=config)

            assert bootstrap.config == config
            assert bootstrap.config.bootstrap_type == "moving"
            assert bootstrap.config.tapered_weights == np.hanning


class TestBlackmanBootstrap:
    class TestPassingCases:
        @given(
            block_length=integers(min_value=1),
            n_bootstraps=integers(min_value=1),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
        )
        def test_blackman_bootstrap(
            self, block_length: int, n_bootstraps: int, rng: int
        ) -> None:
            """
            Test if the BlackmanBootstrap class initializes correctly.
            """
            config = BlackmanBootstrapConfig(
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )
            bootstrap = BlackmanBootstrap(config=config)

            assert bootstrap.config == config
            assert bootstrap.config.bootstrap_type == "moving"
            assert bootstrap.config.tapered_weights == np.blackman


class TestTukeyBootstrap:
    class TestPassingCases:
        @given(
            block_length=integers(min_value=1),
            n_bootstraps=integers(min_value=1),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
        )
        def test_tukey_bootstrap(
            self, block_length: int, n_bootstraps: int, rng: int
        ) -> None:
            """
            Test if the TukeyBootstrap class initializes correctly.
            """
            config = TukeyBootstrapConfig(
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )
            bootstrap = TukeyBootstrap(config=config)

            assert bootstrap.config == config
            assert bootstrap.config.bootstrap_type == "moving"
            assert bootstrap.config.tapered_weights == np.blackman
