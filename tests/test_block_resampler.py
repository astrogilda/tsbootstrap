import pytest
import numpy as np
from numpy.random import Generator
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays
from src.block_resampler import BlockResampler
from utils.validate import validate_weights
from utils.odds_and_ends import check_generator

# Strategies for generating data for the tests
rng_strategy = st.integers(0, 10**6)
blocks_strategy = st.lists(
    arrays(dtype=int, shape=st.integers(1, 100), min_value=0, max_value=100))
weights_strategy = st.one_of(st.none(), arrays(
    dtype=float, shape=st.integers(1, 100), min_value=0, max_value=1e6), st.functions())
X_strategy = arrays(dtype=float, shape=st.integers(1, 100))


class TestPassingCases:
    """Test cases where BlockResampler should work correctly."""

    @given(blocks_strategy, X_strategy, weights_strategy, weights_strategy, rng_strategy)
    def test_init(self, blocks, X, block_weights, tapered_weights, rng):
        """Test initialization of BlockResampler."""
        block_weights = np.random.uniform(low=0, high=100, size=(100))
        br = BlockResampler(blocks, X, block_weights, tapered_weights, rng)
        assert br.blocks == blocks
        assert br.X == X
        assert br.rng == check_generator(rng)
        assert br.block_weights == validate_weights(block_weights)
        assert br.tapered_weights == validate_weights(tapered_weights)

    @given(blocks_strategy, X_strategy, weights_strategy, weights_strategy, rng_strategy)
    def test_block_weights_setter(self, blocks, X, block_weights, tapered_weights, rng):
        """Test block_weights setter method."""
        br = BlockResampler(blocks, X, None, tapered_weights, rng)
        br.block_weights = block_weights
        assert br.block_weights == validate_weights(block_weights)

    @given(blocks_strategy, X_strategy, weights_strategy, weights_strategy, rng_strategy)
    def test_tapered_weights_setter(self, blocks, X, block_weights, tapered_weights, rng):
        """Test tapered_weights setter method."""
        br = BlockResampler(blocks, X, block_weights, None, rng)
        br.tapered_weights = tapered_weights
        assert br.tapered_weights == validate_weights(tapered_weights)

    @given(blocks_strategy, X_strategy, weights_strategy, weights_strategy, rng_strategy)
    def test_rng_setter(self, blocks, X, block_weights, tapered_weights, rng):
        """Test rng setter method."""
        br = BlockResampler(blocks, X, block_weights, tapered_weights, rng)
        new_rng = np.random.default_rng()
        br.rng = new_rng
        assert br.rng == new_rng

    # Tests for getters
    @given(blocks_strategy, X_strategy, weights_strategy, weights_strategy, rng_strategy)
    def test_block_weights_getter(self, blocks, X, block_weights, tapered_weights, rng):
        """Test block_weights getter method."""
        br = BlockResampler(blocks, X, block_weights, tapered_weights, rng)
        assert br.block_weights == validate_weights(block_weights)

    @given(blocks_strategy, X_strategy, weights_strategy, weights_strategy, rng_strategy)
    def test_tapered_weights_getter(self, blocks, X, block_weights, tapered_weights, rng):
        """Test tapered_weights getter method."""
        br = BlockResampler(blocks, X, block_weights, tapered_weights, rng)
        assert br.tapered_weights == validate_weights(tapered_weights)

    @given(blocks_strategy, X_strategy, weights_strategy, weights_strategy, rng_strategy)
    def test_rng_getter(self, blocks, X, block_weights, tapered_weights, rng):
        """Test rng getter method."""
        br = BlockResampler(blocks, X, block_weights, tapered_weights, rng)
        assert br.rng == check_generator(rng)

    # Tests with None values
    @given(blocks_strategy, X_strategy, weights_strategy, rng_strategy)
    def test_none_block_weights(self, blocks, X, tapered_weights, rng):
        """Test initialization with None block weights."""
        br = BlockResampler(blocks, X, None, tapered_weights, rng)
        assert br.block_weights is None

    @given(blocks_strategy, X_strategy, weights_strategy, rng_strategy)
    def test_none_tapered_weights(self, blocks, X, block_weights, rng):
        """Test initialization with None tapered weights."""
        br = BlockResampler(blocks, X, block_weights, None, rng)
        assert br.tapered_weights is None

    @given(blocks_strategy, X_strategy, weights_strategy, weights_strategy)
    def test_none_rng(self, blocks, X, block_weights, tapered_weights):
        """Test initialization with None rng."""
        br = BlockResampler(blocks, X, block_weights, tapered_weights, None)
        assert isinstance(br.rng, np.random.Generator)


class TestFailingCases:
    """Test cases where BlockResampler should raise exceptions."""

    @given(st.lists(st.integers()), X_strategy, weights_strategy, weights_strategy, rng_strategy)
    def test_init_wrong_blocks(self, blocks, X, block_weights, tapered_weights, rng):
        """Test initialization of BlockResampler with invalid blocks."""
        with pytest.raises(ValueError):
            BlockResampler(blocks, X, block_weights, tapered_weights, rng)

    @given(blocks_strategy, st.lists(st.integers()), weights_strategy, weights_strategy, rng_strategy)
    def test_init_wrong_X(self, blocks, X, block_weights, tapered_weights, rng):
        """Test initialization of BlockResampler with invalid X."""
        with pytest.raises(ValueError):
            BlockResampler(blocks, X, block_weights, tapered_weights, rng)

    @given(blocks_strategy, X_strategy, st.lists(st.integers()), weights_strategy, rng_strategy)
    def test_init_wrong_block_weights(self, blocks, X, block_weights, tapered_weights, rng):
        """Test initialization of BlockResampler with invalid block_weights."""
        with pytest.raises(ValueError):
            BlockResampler(blocks, X, block_weights, tapered_weights, rng)

    @given(blocks_strategy, X_strategy, weights_strategy, st.lists(st.integers()), rng_strategy)
    def test_init_wrong_tapered_weights(self, blocks, X, block_weights, tapered_weights, rng):
        """Test initialization of BlockResampler with invalid tapered_weights."""
        with pytest.raises(ValueError):
            BlockResampler(blocks, X, block_weights, tapered_weights, rng)

    @given(blocks_strategy, X_strategy, weights_strategy, weights_strategy, st.lists(st.integers()))
    def test_init_wrong_rng(self, blocks, X, block_weights, tapered_weights, rng):
        """Test initialization of BlockResampler with invalid rng."""
        with pytest.raises(ValueError):
            BlockResampler(blocks, X, block_weights, tapered_weights, rng)

    @given(blocks_strategy, X_strategy, st.lists(st.integers()), weights_strategy, rng_strategy)
    def test_set_wrong_block_weights(self, blocks, X, block_weights, tapered_weights, rng):
        """Test setting block_weights with invalid value."""
        br = BlockResampler(blocks, X, None, tapered_weights, rng)
        with pytest.raises(ValueError):
            br.block_weights = block_weights

    @given(blocks_strategy, X_strategy, weights_strategy, st.lists(st.integers()), rng_strategy)
    def test_set_wrong_tapered_weights(self, blocks, X, block_weights, tapered_weights, rng):
        """Test setting tapered_weights with invalid value."""
        br = BlockResampler(blocks, X, block_weights, None, rng)
        with pytest.raises(ValueError):
            br.tapered_weights = tapered_weights

    @given(blocks_strategy, X_strategy, weights_strategy, weights_strategy, st.lists(st.integers()))
    def test_set_wrong_rng(self, blocks, X, block_weights, tapered_weights, rng):
        """Test setting rng with invalid value."""
        br = BlockResampler(blocks, X, block_weights, tapered_weights, rng)
        with pytest.raises(ValueError):
            br.rng = rng

    @given(st.lists(st.integers()), X_strategy, weights_strategy, weights_strategy, rng_strategy)
    def test_get_wrong_block_weights(self, blocks, X, block_weights, tapered_weights, rng):
        """Test getting block_weights when it's an invalid value."""
        br = BlockResampler(blocks, X, block_weights, tapered_weights, rng)
        with pytest.raises(ValueError):
            _ = br.block_weights

    @given(st.lists(st.integers()), X_strategy, weights_strategy, weights_strategy, rng_strategy)
    def test_get_wrong_tapered_weights(self, blocks, X, block_weights, tapered_weights, rng):
        """Test getting tapered_weights when it's an invalid value."""
        br = BlockResampler(blocks, X, block_weights, tapered_weights, rng)
        with pytest.raises(ValueError):
            _ = br.tapered_weights
