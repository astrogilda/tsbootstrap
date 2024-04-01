from functools import partial

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
from scipy.signal.windows import tukey
from tsbootstrap.block_bootstrap import (
    BLOCK_BOOTSTRAP_TYPES_DICT,
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
    BaseBlockBootstrapConfig,
    BlockBootstrapConfig,
)

# The shape is a strategy generating tuples (num_rows, num_columns)
X_shape = tuples(
    integers(min_value=6, max_value=20), integers(min_value=1, max_value=10)
)


class TestBlockBootstrap:
    class TestPassingCases:
        @settings(max_examples=10, deadline=None)
        @given(
            n_bootstraps=integers(min_value=1, max_value=10),
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
            block_length = np.random.randint(1, X.shape[0] - 1)
            bootstrap = BlockBootstrap(
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )

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
            y=one_of(none(), lists(floats(), min_size=10, max_size=100)),
        )
        def test__generate_samples_single_bootstrap(self, X, y) -> None:
            """
            Test if the BlockBootstrap's _generate_samples_single_bootstrap method runs without errors and returns the correct output.
            """
            bootstrap = BlockBootstrap(
                block_length=5,
                n_bootstraps=10,
                rng=42,
            )

            # Generate blocks
            bootstrap._generate_blocks(np.array(X))

            # Check _generate_samples_single_bootstrap method
            data, indices = bootstrap._generate_samples_single_bootstrap(
                np.array(X), y=y if y is None else np.array(y)
            )

            assert isinstance(indices, list)
            assert all(isinstance(i, np.ndarray) for i in indices)
            assert isinstance(data, list)
            assert all(isinstance(d, np.ndarray) for d in data)

    class TestFailingCases:
        @settings(max_examples=10, deadline=None)
        @given(
            block_length=integers(max_value=0),
            n_bootstraps=integers(min_value=1, max_value=10),
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
            n_bootstraps=integers(min_value=1, max_value=10),
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
            bootstrap = BlockBootstrap(
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )

            with pytest.raises(ValueError):
                bootstrap._generate_blocks(np.array(X))


class TestBaseBlockBootstrap:
    class TestPassingCases:
        # @pytest.mark.skip(reason="known block generation bug, see #73")
        @settings(max_examples=10, deadline=None)
        @given(
            bootstrap_type=sampled_from(list(BLOCK_BOOTSTRAP_TYPES_DICT)),
            n_bootstraps=integers(min_value=1, max_value=10),
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
            block_length = np.random.randint(1, X.shape[0] - 1)
            bootstrap = BaseBlockBootstrap(
                bootstrap_type=bootstrap_type,
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )

            assert isinstance(
                bootstrap.bootstrap_instance,
                BLOCK_BOOTSTRAP_TYPES_DICT[bootstrap_type],
            )
            print(f"bootstrap_type from test: {bootstrap_type}\n")
            print(f"block_length from test: {block_length}\n")
            # Check that bootstrap method runs without errors
            _ = list(bootstrap.bootstrap(np.array(X)))

            # Check that _generate_samples_single_bootstrap method runs without errors
            _ = bootstrap._generate_samples_single_bootstrap(X=np.array(X))

    class TestFailingCases:
        @settings(max_examples=10)  # , deadline=None)
        @given(
            bootstrap_type=text().filter(
                lambda x: x not in list(BLOCK_BOOTSTRAP_TYPES_DICT)
            ),
            block_length=integers(min_value=1),
            n_bootstraps=integers(min_value=1, max_value=10),
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
            n_bootstraps=integers(min_value=1, max_value=10),
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
            bootstrap = BaseBlockBootstrap(
                bootstrap_type=bootstrap_type,
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )

            # Check if the bootstrap_type implements '_generate_samples_single_bootstrap' method
            if not hasattr(
                bootstrap.bootstrap_instance,
                "_generate_samples_single_bootstrap",
            ):
                with pytest.raises(NotImplementedError):
                    bootstrap._generate_samples_single_bootstrap(np.array(X))


class TestMovingBlockBootstrap:
    class TestPassingCases:
        @settings(max_examples=10, deadline=None)
        @given(
            block_length=integers(min_value=1),
            n_bootstraps=integers(min_value=1, max_value=10),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
        )
        def test_moving_block_bootstrap(
            self, block_length: int, n_bootstraps: int, rng: int
        ) -> None:
            """
            Test if the MovingBlockBootstrap class initializes correctly.
            """
            bootstrap = MovingBlockBootstrap(
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )

            assert bootstrap.config._wrap_around_flag is False
            assert bootstrap.config._overlap_flag is True
            assert bootstrap.config._block_length_distribution is None


class TestStationaryBlockBootstrap:
    class TestPassingCases:
        @settings(max_examples=10, deadline=None)
        @given(
            block_length=integers(min_value=1),
            n_bootstraps=integers(min_value=1, max_value=10),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
        )
        def test_stationary_block_bootstrap(
            self, block_length: int, n_bootstraps: int, rng: int
        ) -> None:
            """
            Test if the StationaryBlockBootstrap class initializes correctly.
            """
            bootstrap = StationaryBlockBootstrap(
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )

            assert bootstrap.config._wrap_around_flag is False
            assert bootstrap.config._overlap_flag is True
            assert bootstrap.config._block_length_distribution == "geometric"


class TestCircularBlockBootstrap:
    class TestPassingCases:
        @settings(max_examples=10, deadline=None)
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
            bootstrap = CircularBlockBootstrap(
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )

            assert bootstrap.config._wrap_around_flag is True
            assert bootstrap.config._overlap_flag is True
            assert bootstrap.config._block_length_distribution is None


class TestNonOverlappingBlockBootstrap:
    class TestPassingCases:
        @settings(max_examples=10, deadline=None)
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
            bootstrap = NonOverlappingBlockBootstrap(
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )

            assert bootstrap.config._wrap_around_flag is False
            assert bootstrap.config._overlap_flag is False
            assert bootstrap.config._block_length_distribution is None


class TestBartlettsBootstrap:
    class TestPassingCases:
        @settings(max_examples=10, deadline=None)
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
            bootstrap = BartlettsBootstrap(
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )

            assert bootstrap.config.bootstrap_type == "moving"
            assert bootstrap.config.tapered_weights == np.bartlett


class TestHammingBootstrap:
    class TestPassingCases:
        @settings(max_examples=10, deadline=None)
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
            bootstrap = HammingBootstrap(
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )

            assert bootstrap.config.bootstrap_type == "moving"
            assert bootstrap.config.tapered_weights == np.hamming


class TestHanningBootstrap:
    class TestPassingCases:
        @settings(max_examples=10, deadline=None)
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
            bootstrap = HanningBootstrap(
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )

            assert bootstrap.config.bootstrap_type == "moving"
            assert bootstrap.config.tapered_weights == np.hanning


class TestBlackmanBootstrap:
    class TestPassingCases:
        @settings(max_examples=10, deadline=None)
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
            bootstrap = BlackmanBootstrap(
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )

            assert bootstrap.config.bootstrap_type == "moving"
            assert bootstrap.config.tapered_weights == np.blackman


class TestTukeyBootstrap:
    class TestPassingCases:
        @settings(max_examples=10, deadline=None)
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
            bootstrap = TukeyBootstrap(
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )

            assert bootstrap.config.bootstrap_type == "moving"
            assert (
                bootstrap.config.tapered_weights.func.__name__
                == tukey.__name__
            )
