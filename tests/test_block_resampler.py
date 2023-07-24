import pytest
import numpy as np
from numpy.random import Generator
from hypothesis import given, strategies as st, settings
from src.block_resampler import BlockResampler
from utils.odds_and_ends import check_generator
from typing import List, Tuple

# Hypothesis strategy for generating random seeds
rng_strategy = st.integers(0, 10**6)


# Hypothesis strategy for generating valid block indices and corresponding X
valid_block_indices_and_X = st.integers(min_value=2, max_value=100).flatmap(
    lambda n: st.tuples(
        st.builds(
            list,
            st.lists(
                st.builds(
                    np.array,
                    st.sets(
                        st.integers(min_value=0, max_value=n-1),
                        min_size=2, max_size=n
                    ).map(list)  # Convert set to list
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
            def test_init(self, block_indices_and_X: Tuple[List[np.ndarray], np.ndarray], random_seed: int) -> None:
                """Test initialization of BlockResampler."""
                blocks, X = block_indices_and_X
                rng = np.random.default_rng(random_seed)
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
            def test_block_weights_setter(self, block_indices_and_X: Tuple[List[np.ndarray], np.ndarray], random_seed: int) -> None:
                """Test block_weights setter method."""
                blocks, X = block_indices_and_X
                rng = np.random.default_rng(random_seed)
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
            def test_tapered_weights_setter(self, block_indices_and_X: Tuple[List[np.ndarray], np.ndarray], random_seed: int) -> None:
                """Test block_weights setter method."""
                blocks, X = block_indices_and_X
                rng = np.random.default_rng(random_seed)
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
            def test_tapered_weights_setter(self, block_indices_and_X: Tuple[List[np.ndarray], np.ndarray], random_seed: int) -> None:
                """Test block_weights setter method."""
                blocks, X = block_indices_and_X
                rng = np.random.default_rng(random_seed)
                br = BlockResampler(blocks, X, None, None, rng)
                new_rng = np.random.default_rng()
                br.rng = new_rng
                assert br.rng == new_rng

            # Tests with None values

            @settings(deadline=None)
            @given(valid_block_indices_and_X, rng_strategy)
            def test_none_block_weights(self, block_indices_and_X: Tuple[List[np.ndarray], np.ndarray], random_seed: int) -> None:
                """Test initialization with None block weights."""
                blocks, X = block_indices_and_X
                rng = np.random.default_rng(random_seed)
                tapered_weights = np.random.choice([None, weights_func])
                br = BlockResampler(blocks, X, None, tapered_weights, rng)
                np.testing.assert_array_almost_equal(
                    br.block_weights, np.ones(len(X)) / len(X))

            @settings(deadline=None)
            @given(valid_block_indices_and_X, rng_strategy)
            def test_none_tapered_weights(self, block_indices_and_X: Tuple[List[np.ndarray], np.ndarray], random_seed: int) -> None:
                """Test initialization with None tapered weights."""
                blocks, X = block_indices_and_X
                rng = np.random.default_rng(random_seed)
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
            def test_none_rng(self, block_indices_and_X: Tuple[List[np.ndarray], np.ndarray], random_seed: int) -> None:
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


def check_list_of_arrays_equality(list1: List[np.ndarray], list2: List[np.ndarray], equal: bool = True) -> None:
    """
    Check if two lists of NumPy arrays are equal or not equal, based on the `equal` parameter.
    """
    if equal:
        assert len(list1) == len(list2), "Lists are not of the same length"
        for i, (array1, array2) in enumerate(zip(list1, list2)):
            np.testing.assert_array_equal(
                array1, array2, err_msg=f"Arrays at index {i} are not equal")
    else:
        if len(list1) != len(list2):
            return
        else:
            mismatch = False
            for i, (array1, array2) in enumerate(zip(list1, list2)):
                try:
                    np.testing.assert_array_equal(array1, array2)
                except AssertionError:
                    mismatch = True
                    break
            assert mismatch, "All arrays are unexpectedly equal"


def unique_first_indices(blocks: List[np.ndarray]) -> List[np.ndarray]:
    """
    Return a list of blocks with unique first indices.
    """
    seen_first_indices = set()
    unique_blocks = []
    for block in blocks:
        if block[0] not in seen_first_indices:
            unique_blocks.append(block)
            seen_first_indices.add(block[0])
    return unique_blocks


class TestResampleBlocks:
    """Test the resample_blocks method."""
    class TestPassingCases:
        """Test cases where resample_blocks should work correctly."""

        @settings(deadline=1000)
        @given(valid_block_indices_and_X, rng_strategy)
        def test_resample_blocks_valid_inputs(self, block_indices_and_X: Tuple[List[np.ndarray], np.ndarray], random_seed: int) -> None:
            """
            Test that the 'resample_blocks' method works correctly with valid inputs.
            """
            blocks, X = block_indices_and_X
            blocks = unique_first_indices(blocks)
            rng = np.random.default_rng(random_seed)
            br = BlockResampler(blocks, X, rng=rng)
            new_blocks, new_tapered_weights = br.resample_blocks()

            # Check that the total length of the new blocks is equal to n.
            total_length = sum(len(block) for block in new_blocks)
            assert total_length == len(X)

            # Check that the length of new_blocks and new_tapered_weights are equal.
            assert len(new_blocks) == len(new_tapered_weights)

            # We set the len(blocks) to be 3, so we can minimize the chances that resampling blocks a second time, or with a different random seed, gives the same results.
            if len(blocks) > 3:

                # Check that resampling with the same random seed, a second time, gives different results.
                new_blocks_2, new_tapered_weights_2 = br.resample_blocks()
                check_list_of_arrays_equality(
                    new_blocks, new_blocks_2, equal=False)

                # Check that resampling with a new random seed gives different results.
                rng2 = np.random.default_rng((random_seed+1)*2)
                br = BlockResampler(blocks, X, rng=rng2)
                new_blocks_3, new_tapered_weights_3 = br.resample_blocks()
                check_list_of_arrays_equality(
                    new_blocks, new_blocks_3, equal=False)

                # Check that resampling with the same random seed gives the same results.
                rng = np.random.default_rng(random_seed)
                br = BlockResampler(
                    blocks, X, rng=rng)
                new_blocks_4, new_tapered_weights_4 = br.resample_blocks()
                check_list_of_arrays_equality(new_blocks, new_blocks_4)

    class TestFailingCases:
        pass


class TestGenerateBlockIndicesAndData:
    """Test the resample_block_indices_and_data method."""

    class TestPassingCases:
        """Test cases where resample_block_indices_and_data should work correctly."""

        @settings(deadline=1000)
        @given(valid_block_indices_and_X, rng_strategy)
        def test_valid_inputs(self, block_indices_and_X: Tuple[List[np.ndarray], np.ndarray], random_seed: int) -> None:
            """
            Test that the 'resample_blocks' method works correctly with valid inputs.
            """
            blocks, X = block_indices_and_X
            blocks = unique_first_indices(blocks)
            rng = np.random.default_rng(random_seed)
            br = BlockResampler(blocks, X, rng=rng,
                                tapered_weights=weights_func)
            new_blocks, block_data = br.resample_block_indices_and_data()

            # Check that the total length of the new blocks is equal to n.
            total_length = sum(len(block) for block in new_blocks)
            assert total_length == len(X)

            # Check that the length of new_blocks and block_data are equal.
            assert len(new_blocks) == len(block_data)

            # Check that the length of each block in new_blocks is equal to the length of the corresponding block in block_data.
            for i in range(len(new_blocks)):
                assert len(new_blocks[i]) == len(block_data[i])

            # Check that the sum of lengths of all blocks in new_blocks is equal to the sum of lengths of all blocks in block_data is equal to the length of X.
            assert sum(len(block) for block in new_blocks) == sum(
                len(block) for block in block_data) == len(X)

            # We set the len(blocks) to be 3, so we can minimize the chances that resampling blocks a second time, or with a different random seed, gives the same results.
            if len(blocks) > 3:

                # Check that resampling with the same random seed, a second time, gives different results.
                new_blocks_2, block_data_2 = br.resample_block_indices_and_data()
                check_list_of_arrays_equality(
                    new_blocks, new_blocks_2, equal=False)

                # Check that resampling with a new random seed gives different results.
                rng2 = np.random.default_rng((random_seed+1)*2)
                br = BlockResampler(blocks, X, rng=rng2)
                new_blocks_3, block_data_3 = br.resample_block_indices_and_data()
                check_list_of_arrays_equality(
                    new_blocks, new_blocks_3, equal=False)

                # Check that resampling with the same random seed gives the same results.
                rng = np.random.default_rng(random_seed)
                br = BlockResampler(
                    blocks, X, rng=rng)
                new_blocks_4, block_data_4 = br.resample_block_indices_and_data()
                check_list_of_arrays_equality(new_blocks, new_blocks_4)

    class TestFailingCases:
        """Test cases where resample_block_indices_and_data should raise exceptions."""
        pass
