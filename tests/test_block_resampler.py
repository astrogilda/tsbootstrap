import pytest
import numpy as np
from numpy.random import Generator
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays
from src.block_resampler import BlockResampler
from utils.validate import validate_weights
from utils.odds_and_ends import check_generator
from typing import List, Tuple

# Strategies for generating data for the tests
rng_strategy = st.integers(0, 10**6)
X_strategy = arrays(dtype=float, shape=st.integers(2, 100),
                    elements=st.floats(min_value=-1e6, max_value=1e6))

# Hypothesis strategy for generating valid block indices and corresponding X
valid_block_indices_and_X = st.integers(min_value=2, max_value=100).flatmap(
    lambda n: st.tuples(
        st.builds(
            list,
            st.lists(
                st.builds(
                    np.array,
                    st.lists(
                        st.integers(min_value=0, max_value=n-1),
                        min_size=2, max_size=n
                    )
                ),
                min_size=1, max_size=n
            )
        ),
        st.lists(
            st.lists(
                st.floats(min_value=1e-10, max_value=10,
                          allow_nan=False, allow_infinity=False),
                min_size=2, max_size=2  # two elements in the second axis
            ),
            min_size=n, max_size=n  # n elements in the first axis
        ).map(np.array)
    )
)


def weights_func(size: int):
    return np.random.uniform(low=0, high=1e6, size=size)


class TestBlockResampler:
    """Test the BlockResampler class."""

    class TestInit:
        """Test the __init__ method."""

        class TestPassingCases:
            """Test cases where BlockResampler should work correctly."""

            @settings(deadline=None)
            @given(valid_block_indices_and_X, rng_strategy)
            def test_init(self, block_indices_and_X: Tuple[List[np.ndarray], np.ndarray], rng: Generator) -> None:
                """Test initialization of BlockResampler."""
                blocks, X = block_indices_and_X
                rng = np.random.default_rng(rng)
                tapered_weights = np.random.choice([None, weights_func])
                block_weights_choice = np.random.choice([0, 1, 2])
                if block_weights_choice == 0:
                    block_weights = None
                elif block_weights_choice == 1:
                    block_weights = weights_func(int(X.shape[0]))
                else:
                    block_weights = weights_func

                br = BlockResampler(
                    blocks, X, block_weights, tapered_weights, rng)
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
            @given(valid_block_indices_and_X, rng_strategy)
            def test_block_weights_setter(self, block_indices_and_X: Tuple[List[np.ndarray], np.ndarray], rng: Generator) -> None:
                """Test block_weights setter method."""
                blocks, X = block_indices_and_X
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
            @given(valid_block_indices_and_X, rng_strategy)
            def test_tapered_weights_setter(self, block_indices_and_X: Tuple[List[np.ndarray], np.ndarray], rng: Generator) -> None:
                """Test block_weights setter method."""
                blocks, X = block_indices_and_X
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
            @given(valid_block_indices_and_X, rng_strategy)
            def test_tapered_weights_setter(self, block_indices_and_X: Tuple[List[np.ndarray], np.ndarray], rng: Generator) -> None:
                """Test block_weights setter method."""
                blocks, X = block_indices_and_X
                rng = np.random.default_rng(rng)
                br = BlockResampler(blocks, X, None, None, rng)
                new_rng = np.random.default_rng()
                br.rng = new_rng
                assert br.rng == new_rng

            # Tests with None values

            @settings(deadline=None)
            @given(valid_block_indices_and_X, rng_strategy)
            def test_none_block_weights(self, block_indices_and_X: Tuple[List[np.ndarray], np.ndarray], rng: Generator) -> None:
                """Test initialization with None block weights."""
                blocks, X = block_indices_and_X
                rng = np.random.default_rng(rng)
                tapered_weights = np.random.choice([None, weights_func])
                br = BlockResampler(blocks, X, None, tapered_weights, rng)
                np.testing.assert_array_almost_equal(
                    br.block_weights, np.ones(len(X)) / len(X))

            @settings(deadline=None)
            @given(valid_block_indices_and_X, rng_strategy)
            def test_none_tapered_weights(self, block_indices_and_X: Tuple[List[np.ndarray], np.ndarray], rng: Generator) -> None:
                """Test initialization with None tapered weights."""
                blocks, X = block_indices_and_X
                rng = np.random.default_rng(rng)
                block_weights_choice = np.random.choice([0, 1, 2])
                if block_weights_choice == 0:
                    block_weights = None
                elif block_weights_choice == 1:
                    block_weights = weights_func(int(X.shape[0]))
                else:
                    block_weights = weights_func
                br = BlockResampler(blocks, X, block_weights, None, rng)
                for i in range(len(blocks)):
                    np.testing.assert_array_almost_equal(
                        br.tapered_weights[i], np.ones(len(blocks[i])) / len(blocks[i]))

            @settings(deadline=None)
            @given(valid_block_indices_and_X, rng_strategy)
            def test_none_rng(self, block_indices_and_X: Tuple[List[np.ndarray], np.ndarray], rng: Generator) -> None:
                """Test initialization with None rng."""
                blocks, X = block_indices_and_X
                br = BlockResampler(blocks, X, None, None, None)
                assert isinstance(br.rng, np.random.Generator)

        class TestFailingCases:
            """Test cases where BlockResampler should raise exceptions."""

            @settings(deadline=None)
            @given(valid_block_indices_and_X)
            def test_init_wrong_blocks(self, block_indices_and_X: Tuple[List[np.ndarray], np.ndarray]) -> None:
                """Test initialization of BlockResampler with invalid blocks."""
                blocks, X = block_indices_and_X
                br = BlockResampler(blocks, X, None, None, None)
                with pytest.raises(TypeError):
                    br.blocks = None
                with pytest.raises(TypeError):
                    br.blocks = np.array([])
                with pytest.raises(TypeError):
                    br.blocks = np.array([1])
                with pytest.raises(TypeError):
                    br.blocks = np.array([1, 2])

            @settings(deadline=None)
            @given(valid_block_indices_and_X)
            def test_init_wrong_X(self, block_indices_and_X: Tuple[List[np.ndarray], np.ndarray]) -> None:
                """Test initialization of BlockResampler with invalid X."""
                blocks, X = block_indices_and_X
                br = BlockResampler(blocks, X, None, None, None)
                with pytest.raises(TypeError):
                    br.X = None
                with pytest.raises(ValueError):
                    br.X = np.array([])
                with pytest.raises(ValueError):
                    br.X = np.array([1])

            @settings(deadline=None)
            @given(valid_block_indices_and_X)
            def test_init_wrong_block_weights(self, block_indices_and_X: Tuple[List[np.ndarray], np.ndarray]) -> None:
                """Test initialization of BlockResampler with invalid block_weights."""
                blocks, X = block_indices_and_X
                br = BlockResampler(blocks, X, None, None, None)
                with pytest.raises(ValueError):
                    br.block_weights = np.arange(len(X)+1)
                with pytest.raises(TypeError):
                    br.block_weights = "abc"
                with pytest.raises(ValueError):
                    br.block_weights = X
                with pytest.raises(TypeError):
                    br.block_weights = np.mean

            @settings(deadline=None)
            @given(valid_block_indices_and_X)
            def test_init_wrong_tapered_weights(self, block_indices_and_X: Tuple[List[np.ndarray], np.ndarray]) -> None:
                """Test initialization of BlockResampler with invalid tapered_weights."""
                blocks, X = block_indices_and_X
                br = BlockResampler(blocks, X, None, None, None)
                with pytest.raises(TypeError):
                    br.tapered_weights = "abc"
                with pytest.raises(TypeError):
                    br.tapered_weights = X
                with pytest.raises(TypeError):
                    br.block_weights = np.mean

            @settings(deadline=None)
            @given(valid_block_indices_and_X)
            def test_init_wrong_rng(self, block_indices_and_X: Tuple[List[np.ndarray], np.ndarray]) -> None:
                """Test initialization of BlockResampler with invalid rng."""
                blocks, X = block_indices_and_X
                with pytest.raises(TypeError):
                    BlockResampler(blocks, X, None, None, 3)
