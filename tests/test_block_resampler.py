import pytest
import numpy as np
from numpy.random import Generator
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays
from src.block_resampler import BlockResampler
from utils.validate import validate_weights
from utils.odds_and_ends import check_generator

# Strategies for generating data for the tests
rng_strategy = st.integers(0, 10**6)
X_strategy = arrays(dtype=float, shape=st.integers(2, 100),
                    elements=st.floats(min_value=-1e6, max_value=1e6))


def weights_func(size: int):
    return np.random.uniform(low=0, high=1e6, size=size)


class TestPassingCases:
    """Test cases where BlockResampler should work correctly."""

    @settings(deadline=None)
    @given(X_strategy, rng_strategy)
    def test_init(self, X, rng):
        """Test initialization of BlockResampler."""
        blocks = [np.random.randint(X.shape[0], size=X.shape[0])]
        rng = np.random.default_rng(rng)
        tapered_weights = np.random.choice([None, weights_func])
        block_weights_choice = np.random.choice([0, 1, 2])
        if block_weights_choice == 0:
            block_weights = None
        elif block_weights_choice == 1:
            block_weights = weights_func(int(X.shape[0]))
        else:
            block_weights = weights_func

        br = BlockResampler(blocks, X, block_weights, tapered_weights, rng)
        assert br.blocks == blocks
        np.testing.assert_array_equal(br.X, X)
        assert br.rng == check_generator(rng)

        assert isinstance(br.block_weights, np.ndarray)
        assert np.isclose(br.block_weights.sum(), 1)
        assert len(br.block_weights) == len(X)

        assert isinstance(br.tapered_weights, list)
        assert all([isinstance(br.tapered_weights[i], np.ndarray)
                   for i in range(len(blocks))])
        assert all([np.isclose(br.tapered_weights[i].sum(), 1)
                   for i in range(len(blocks))])
        assert len(br.tapered_weights) == len(blocks)

    @settings(deadline=None)
    @given(X_strategy, rng_strategy)
    def test_block_weights_setter(self, X, rng):
        """Test block_weights setter method."""
        blocks = [np.random.randint(X.shape[0], size=X.shape[0])]
        rng = np.random.default_rng(rng)
        tapered_weights = np.random.choice([None, weights_func])
        block_weights_choice = np.random.choice([0, 1, 2])
        if block_weights_choice == 0:
            block_weights = None
        elif block_weights_choice == 1:
            block_weights = weights_func(int(X.shape[0]))
        else:
            block_weights = weights_func
        br = BlockResampler(blocks, X, None, tapered_weights, rng)
        br.block_weights = block_weights
        assert isinstance(br.block_weights, np.ndarray)
        assert np.isclose(br.block_weights.sum(), 1)
        assert len(br.block_weights) == len(X)

    @settings(deadline=None)
    @given(X_strategy, rng_strategy)
    def test_tapered_weights_setter(self, X, rng):
        """Test block_weights setter method."""
        blocks = [np.random.randint(X.shape[0], size=X.shape[0])]
        rng = np.random.default_rng(rng)
        tapered_weights = np.random.choice([None, weights_func])
        block_weights_choice = np.random.choice([0, 1, 2])
        if block_weights_choice == 0:
            block_weights = None
        elif block_weights_choice == 1:
            block_weights = weights_func(int(X.shape[0]))
        else:
            block_weights = weights_func
        br = BlockResampler(blocks, X, block_weights, None, rng)
        br.tapered_weights = tapered_weights
        assert isinstance(br.tapered_weights, list)
        assert all([isinstance(br.tapered_weights[i], np.ndarray)
                   for i in range(len(blocks))])
        assert all([np.isclose(br.tapered_weights[i].sum(), 1)
                   for i in range(len(blocks))])
        assert len(br.tapered_weights) == len(blocks)

    @settings(deadline=None)
    @given(X_strategy, rng_strategy)
    def test_tapered_weights_setter(self, X, rng):
        """Test block_weights setter method."""
        blocks = [np.random.randint(X.shape[0], size=X.shape[0])]
        rng = np.random.default_rng(rng)
        br = BlockResampler(blocks, X, None, None, rng)
        new_rng = np.random.default_rng()
        br.rng = new_rng
        assert br.rng == new_rng

    # Tests with None values

    @settings(deadline=None)
    @given(X_strategy, rng_strategy)
    def test_none_block_weights(self, X, rng):
        """Test initialization with None block weights."""
        blocks = [np.random.randint(X.shape[0], size=X.shape[0])]
        rng = np.random.default_rng(rng)
        tapered_weights = np.random.choice([None, weights_func])
        br = BlockResampler(blocks, X, None, tapered_weights, rng)
        np.testing.assert_array_almost_equal(
            br.block_weights, np.ones(len(X)) / len(X))

    @settings(deadline=None)
    @given(X_strategy, rng_strategy)
    def test_none_tapered_weights(self, X, rng):
        """Test initialization with None tapered weights."""
        blocks = [np.random.randint(X.shape[0], size=X.shape[0])]
        rng = np.random.default_rng(rng)
        block_weights_choice = np.random.choice([0, 1, 2])
        if block_weights_choice == 0:
            block_weights = None
        elif block_weights_choice == 1:
            block_weights = weights_func(int(X.shape[0]))
        else:
            block_weights = weights_func
        br = BlockResampler(blocks, X, block_weights, None, rng)
        np.testing.assert_array_almost_equal(
            br.tapered_weights[0], np.ones(len(X)) / len(X))

    @settings(deadline=None)
    @given(X_strategy)
    def test_none_rng(self, X):
        """Test initialization with None rng."""
        blocks = [np.random.randint(X.shape[0], size=X.shape[0])]
        br = BlockResampler(blocks, X, None, None, None)
        assert isinstance(br.rng, np.random.Generator)


class TestFailingCases:
    """Test cases where BlockResampler should raise exceptions."""

    @settings(deadline=None)
    @given(X_strategy, rng_strategy)
    def test_init_wrong_blocks(self, X, rng):
        """Test initialization of BlockResampler with invalid blocks."""
        blocks = [np.random.randint(X.shape[0], size=X.shape[0]+1)]
        with pytest.raises(ValueError):
            BlockResampler(blocks, X, None, None, rng)


'''
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
'''
