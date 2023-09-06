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
    text,
)
from ts_bs.block_bootstrap_configs import BlockBootstrapConfig


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
