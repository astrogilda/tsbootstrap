import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import (
    booleans,
    floats,
    integers,
    lists,
    none,
    one_of,
    sampled_from,
    text,
)
from tsbootstrap.block_bootstrap import BLOCK_BOOTSTRAP_TYPES_DICT
from tsbootstrap.block_bootstrap_configs import (
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


class TestBlockBootstrapConfig:
    class TestPassingCases:
        @given(
            block_length=integers(min_value=1),
            wrap_around_flag=booleans(),
            overlap_flag=booleans(),
            combine_generation_and_sampling_flag=booleans(),
            overlap_length=integers(min_value=1),
            min_block_length=integers(min_value=1),
            n_bootstraps=integers(min_value=1),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
        )
        def test_block_bootstrap_config(
            self,
            block_length: int,
            wrap_around_flag: bool,
            overlap_flag: bool,
            combine_generation_and_sampling_flag: bool,
            overlap_length: int,
            min_block_length: int,
            n_bootstraps: int,
            rng: int,
        ) -> None:
            """
            Test if the BlockBootstrapConfig initializes correctly with all integer and boolean parameters.
            """
            config = BlockBootstrapConfig(
                block_length=block_length,
                wrap_around_flag=wrap_around_flag,
                overlap_flag=overlap_flag,
                combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
                overlap_length=overlap_length,
                min_block_length=min_block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )
            assert isinstance(config.block_length, int)
            assert config.block_length == block_length
            assert isinstance(config.overlap_length, int)
            assert config.overlap_length == overlap_length
            assert isinstance(config.min_block_length, int)
            assert config.min_block_length == min_block_length
            assert isinstance(config.wrap_around_flag, bool)
            assert config.wrap_around_flag == wrap_around_flag
            assert isinstance(config.overlap_flag, bool)
            assert config.overlap_flag == overlap_flag
            assert isinstance(
                config.combine_generation_and_sampling_flag, bool
            )
            assert (
                config.combine_generation_and_sampling_flag
                == combine_generation_and_sampling_flag
            )
            assert isinstance(config.n_bootstraps, int)
            assert config.n_bootstraps == n_bootstraps
            assert isinstance(config.rng, np.random.Generator)

    class TestFailingCases:
        @given(block_length=one_of(text(), lists(integers()), floats()))
        def test_invalid_block_length_type(self, block_length) -> None:
            """
            Test if the BlockBootstrapConfig raises a TypeError when block_length is not an integer.
            """
            with pytest.raises(TypeError):
                _ = BlockBootstrapConfig(block_length=block_length)

        @given(block_length=integers(max_value=0))
        def test_invalid_block_length_value(self, block_length: int) -> None:
            """
            Test if the BlockBootstrapConfig raises a ValueError when block_length is less than or equal to 0.
            """
            with pytest.raises(ValueError):
                _ = BlockBootstrapConfig(block_length=block_length)

        @given(overlap_length=one_of(text(), lists(integers()), floats()))
        def test_invalid_overlap_length_type(self, overlap_length) -> None:
            """
            Test if the BlockBootstrapConfig raises a TypeError when overlap_length is not an integer.
            """
            with pytest.raises(TypeError):
                _ = BlockBootstrapConfig(overlap_length=overlap_length)

        @given(overlap_length=integers(max_value=0))
        def test_invalid_overlap_length_value(
            self, overlap_length: int
        ) -> None:
            """
            Test if the BlockBootstrapConfig raises a ValueError when overlap_length is less than or equal to 0.
            """
            with pytest.raises(ValueError):
                _ = BlockBootstrapConfig(overlap_length=overlap_length)

        @given(min_block_length=one_of(text(), lists(integers()), floats()))
        def test_invalid_min_block_length_type(self, min_block_length) -> None:
            """
            Test if the BlockBootstrapConfig raises a TypeError when min_block_length is not an integer.
            """
            with pytest.raises(TypeError):
                _ = BlockBootstrapConfig(min_block_length=min_block_length)

        @given(min_block_length=integers(max_value=0))
        def test_invalid_min_block_length_value(
            self, min_block_length: int
        ) -> None:
            """
            Test if the BlockBootstrapConfig raises a ValueError when min_block_length is less than or equal to 0.
            """
            with pytest.raises(ValueError):
                _ = BlockBootstrapConfig(min_block_length=min_block_length)

        @given(n_bootstraps=one_of(text(), lists(integers()), floats()))
        def test_invalid_n_bootstraps_type(self, n_bootstraps) -> None:
            """
            Test if the BlockBootstrapConfig raises a TypeError when n_bootstraps is not an integer.
            """
            with pytest.raises(TypeError):
                _ = BlockBootstrapConfig(n_bootstraps=n_bootstraps)

        @given(n_bootstraps=integers(max_value=0))
        def test_invalid_n_bootstraps_value(self, n_bootstraps: int) -> None:
            """
            Test if the BlockBootstrapConfig raises a ValueError when n_bootstraps is less than or equal to 0.
            """
            with pytest.raises(ValueError):
                _ = BlockBootstrapConfig(n_bootstraps=n_bootstraps)

        @given(rng=one_of(text(), lists(integers()), floats()))
        def test_invalid_rng_type(self, rng) -> None:
            """
            Test if the BlockBootstrapConfig raises a TypeError when rng is not an integer or None.
            """
            with pytest.raises(TypeError):
                _ = BlockBootstrapConfig(rng=rng)


class TestBaseBlockBootstrapConfig:
    class TestPassingCases:
        @given(
            bootstrap_type=sampled_from(list(BLOCK_BOOTSTRAP_TYPES_DICT)),
            block_length=integers(min_value=1),
            wrap_around_flag=booleans(),
            overlap_flag=booleans(),
            combine_generation_and_sampling_flag=booleans(),
            overlap_length=integers(min_value=1),
            min_block_length=integers(min_value=1),
            n_bootstraps=integers(min_value=1),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
        )
        def test_base_block_bootstrap_config(
            self,
            bootstrap_type: str,
            block_length: int,
            wrap_around_flag: bool,
            overlap_flag: bool,
            combine_generation_and_sampling_flag: bool,
            overlap_length: int,
            min_block_length: int,
            n_bootstraps: int,
            rng: int,
        ) -> None:
            """
            Test if the BaseBlockBootstrapConfig initializes correctly with all integer, boolean and string parameters.
            """
            config = BaseBlockBootstrapConfig(
                bootstrap_type=bootstrap_type,
                block_length=block_length,
                wrap_around_flag=wrap_around_flag,
                overlap_flag=overlap_flag,
                combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
                overlap_length=overlap_length,
                min_block_length=min_block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )
            assert config.bootstrap_type == bootstrap_type
            assert config.block_length == block_length
            assert config.wrap_around_flag == wrap_around_flag
            assert config.overlap_flag == overlap_flag
            assert (
                config.combine_generation_and_sampling_flag
                == combine_generation_and_sampling_flag
            )
            assert config.overlap_length == overlap_length
            assert config.min_block_length == min_block_length
            assert config.n_bootstraps == n_bootstraps
            assert isinstance(config.rng, np.random.Generator)

    class TestFailingCases:
        @given(
            bootstrap_type=text().filter(
                lambda x: x not in list(BLOCK_BOOTSTRAP_TYPES_DICT)
            )
        )
        def test_invalid_bootstrap_type(self, bootstrap_type: str) -> None:
            """
            Test if the BaseBlockBootstrapConfig raises a ValueError when bootstrap_type is not one of the BLOCK_BOOTSTRAP_TYPES_DICT.
            """
            with pytest.raises(ValueError):
                _ = BaseBlockBootstrapConfig(bootstrap_type=bootstrap_type)


class TestMovingBlockBootstrapConfig:
    class TestPassingCases:
        @given(
            block_length=integers(min_value=1),
            n_bootstraps=integers(min_value=1),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
        )
        def test_moving_block_bootstrap_config(
            self,
            block_length: int,
            n_bootstraps: int,
            rng: int,
        ) -> None:
            """
            Test if the MovingBlockBootstrapConfig initializes correctly with all integer parameters.
            """
            config = MovingBlockBootstrapConfig(
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )
            assert config.block_length == block_length
            assert config.n_bootstraps == n_bootstraps
            assert isinstance(config.rng, np.random.Generator)
            assert config.wrap_around_flag is False
            assert config.overlap_flag
            assert config.block_length_distribution is None


class TestCircularBlockBootstrapConfig:
    class TestPassingCases:
        @given(
            block_length=integers(min_value=1),
            n_bootstraps=integers(min_value=1),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
        )
        def test_circular_block_bootstrap_config(
            self,
            block_length: int,
            n_bootstraps: int,
            rng: int,
        ) -> None:
            """
            Test if the CircularBlockBootstrapConfig initializes correctly with all integer parameters.
            """
            config = CircularBlockBootstrapConfig(
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )
            assert config.block_length == block_length
            assert config.n_bootstraps == n_bootstraps
            assert isinstance(config.rng, np.random.Generator)
            assert config.wrap_around_flag
            assert config.overlap_flag
            assert config.block_length_distribution is None


class TestStationaryBlockBootstrapConfig:
    class TestPassingCases:
        @given(
            block_length=integers(min_value=1),
            n_bootstraps=integers(min_value=1),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
        )
        def test_stationary_block_bootstrap_config(
            self,
            block_length: int,
            n_bootstraps: int,
            rng: int,
        ) -> None:
            """
            Test if the StationaryBlockBootstrapConfig initializes correctly with all integer parameters.
            """
            config = StationaryBlockBootstrapConfig(
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )
            assert config.block_length == block_length
            assert config.n_bootstraps == n_bootstraps
            assert isinstance(config.rng, np.random.Generator)
            assert not config.wrap_around_flag
            assert config.overlap_flag
            assert config.block_length_distribution == "geometric"


class TestNonOverlappingBlockBootstrapConfig:
    class TestPassingCases:
        @given(
            block_length=integers(min_value=1),
            n_bootstraps=integers(min_value=1),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
        )
        def test_non_overlapping_block_bootstrap_config(
            self,
            block_length: int,
            n_bootstraps: int,
            rng: int,
        ) -> None:
            """
            Test if the NonOverlappingBlockBootstrapConfig initializes correctly with all integer parameters.
            """
            config = NonOverlappingBlockBootstrapConfig(
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )
            assert config.block_length == block_length
            assert config.n_bootstraps == n_bootstraps
            assert isinstance(config.rng, np.random.Generator)
            assert not config.wrap_around_flag
            assert not config.overlap_flag
            assert config.block_length_distribution is None


class TestBartlettsBootstrapConfig:
    class TestPassingCases:
        @given(
            block_length=integers(min_value=1),
            n_bootstraps=integers(min_value=1),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
        )
        def test_bartletts_bootstrap_config(
            self,
            block_length: int,
            n_bootstraps: int,
            rng: int,
        ) -> None:
            """
            Test if the BartlettsBootstrapConfig initializes correctly with all integer parameters.
            """
            config = BartlettsBootstrapConfig(
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )
            assert config.block_length == block_length
            assert config.n_bootstraps == n_bootstraps
            assert isinstance(config.rng, np.random.Generator)
            assert config.bootstrap_type == "moving"
            assert config.tapered_weights == np.bartlett


class TestHammingBootstrapConfig:
    class TestPassingCases:
        @given(
            block_length=integers(min_value=1),
            n_bootstraps=integers(min_value=1),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
        )
        def test_hamming_bootstrap_config(
            self,
            block_length: int,
            n_bootstraps: int,
            rng: int,
        ) -> None:
            """
            Test if the HammingBootstrapConfig initializes correctly with all integer parameters.
            """
            config = HammingBootstrapConfig(
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )
            assert config.block_length == block_length
            assert config.n_bootstraps == n_bootstraps
            assert isinstance(config.rng, np.random.Generator)
            assert config.bootstrap_type == "moving"
            assert config.tapered_weights == np.hamming


class TestHanningBootstrapConfig:
    class TestPassingCases:
        @given(
            block_length=integers(min_value=1),
            n_bootstraps=integers(min_value=1),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
        )
        def test_hanning_bootstrap_config(
            self,
            block_length: int,
            n_bootstraps: int,
            rng: int,
        ) -> None:
            """
            Test if the HanningBootstrapConfig initializes correctly with all integer parameters.
            """
            config = HanningBootstrapConfig(
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )
            assert config.block_length == block_length
            assert config.n_bootstraps == n_bootstraps
            assert isinstance(config.rng, np.random.Generator)
            assert config.bootstrap_type == "moving"
            assert config.tapered_weights == np.hanning


class TestBlackmanBootstrapConfig:
    class TestPassingCases:
        @given(
            block_length=integers(min_value=1),
            n_bootstraps=integers(min_value=1),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
        )
        def test_blackman_bootstrap_config(
            self,
            block_length: int,
            n_bootstraps: int,
            rng: int,
        ) -> None:
            """
            Test if the BlackmanBootstrapConfig initializes correctly with all integer parameters.
            """
            config = BlackmanBootstrapConfig(
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )
            assert config.block_length == block_length
            assert config.n_bootstraps == n_bootstraps
            assert isinstance(config.rng, np.random.Generator)
            assert config.bootstrap_type == "moving"
            assert config.tapered_weights == np.blackman


class TestTukeyBootstrapConfig:
    class TestPassingCases:
        @given(
            block_length=integers(min_value=1),
            n_bootstraps=integers(min_value=1),
            rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
        )
        def test_tukey_bootstrap_config(
            self,
            block_length: int,
            n_bootstraps: int,
            rng: int,
        ) -> None:
            """
            Test if the TukeyBootstrapConfig initializes correctly with all integer parameters.
            """
            config = TukeyBootstrapConfig(
                block_length=block_length,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )
            assert config.block_length == block_length
            assert config.n_bootstraps == n_bootstraps
            assert isinstance(config.rng, np.random.Generator)
            assert config.bootstrap_type == "moving"
            assert callable(config.tapered_weights)
