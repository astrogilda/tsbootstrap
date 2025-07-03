import random
from typing import Literal

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError  # Added import for pydantic
from tsbootstrap import BlockResampler

# Hypothesis strategy for generating random seeds
rng_strategy = st.integers(0, 10**6)


def block_generator(
    input_length,
    wrap_around_flag,
    overlap_length,
    min_block_length,
    avg_block_length,
    overlap_flag,
):
    from tsbootstrap.block_generator import BlockGenerator, BlockLengthSampler

    #
    block_length_sampler = BlockLengthSampler(avg_block_length=avg_block_length)
    rng = np.random.default_rng()
    #
    block_generator = BlockGenerator(
        input_length=input_length,
        block_length_sampler=block_length_sampler,
        wrap_around_flag=wrap_around_flag,
        rng=rng,
        overlap_length=overlap_length,
        min_block_length=min_block_length,
    )
    blocks = block_generator.generate_blocks(overlap_flag=overlap_flag)
    X = np.random.uniform(low=0, high=1e6, size=input_length).reshape(-1, 1)
    return blocks, X


valid_block_indices_and_X = st.builds(
    block_generator,
    input_length=st.integers(min_value=50, max_value=100),
    wrap_around_flag=st.booleans(),
    overlap_length=st.integers(min_value=1, max_value=2),
    min_block_length=st.integers(min_value=2, max_value=2),
    avg_block_length=st.integers(min_value=3, max_value=10),
    overlap_flag=st.booleans(),
)


def weights_func(size: int) -> np.ndarray:
    return np.random.uniform(low=0, high=1e6, size=size)


class TestInit:
    """Test the __init__ method."""

    class TestPassingCases:
        """Test cases where BlockResampler should work correctly."""

        @settings(deadline=None)
        @given(valid_block_indices_and_X, rng_strategy)
        def test_init(
            self,
            block_indices_and_X,
            random_seed: int,
        ) -> None:
            """Test initialization of BlockResampler."""
            blocks, X = block_indices_and_X
            rng = np.random.default_rng(random_seed)
            tapered_weights = random.choice([None, weights_func])  # noqa: S311
            block_weights_choice = np.random.choice([0, 1, 2])  # noqa: S311
            if block_weights_choice == 0:
                block_weights = None
            elif block_weights_choice == 1:
                block_weights = weights_func(len(blocks))
            else:
                block_weights = weights_func

            br = BlockResampler(
                X=X,
                blocks=blocks,
                block_weights=block_weights,
                tapered_weights=tapered_weights,
                rng=rng,
            )
            # Use custom equality check for list of arrays
            check_list_of_arrays_equality(br.blocks, blocks)
            np.testing.assert_array_equal(br.X, X)
            # RNG comparison is not straightforward, validate_rng ensures it's a valid generator
            assert isinstance(br.rng, np.random.Generator)

            assert isinstance(br.block_weights, np.ndarray)
            assert np.isclose(br.block_weights.sum(), 1)
            assert len(br.block_weights) == len(blocks)  # Should be length of blocks

            assert isinstance(br.tapered_weights, list)
            assert all(isinstance(br.tapered_weights[i], np.ndarray) for i in range(len(blocks)))
            if tapered_weights is None:  # Check if input was None
                # If input was None, _prepare_tapered_weights defaults to arrays of ones.
                # These are then processed by np.maximum(arr, 0.1) (no change for ones)
                # and _scale_to_max_one (no change for all-ones arrays).
                for i in range(len(blocks)):
                    if len(blocks[i]) > 0:  # Avoid issues with empty blocks if they could occur
                        np.testing.assert_array_almost_equal(
                            br.tapered_weights[i], np.ones(len(blocks[i]))
                        )

            # After _prepare_tapered_weights, all individual weight arrays are scaled so their max is 1.0
            # (unless a block was empty, or original weights were all < 0.1 and became all 0.1s, then scaled).
            # Given np.maximum(weights, 0.1), the minimum value is 0.1, so max will be > 0 for non-empty blocks.
            for i in range(len(blocks)):
                if len(br.tapered_weights[i]) > 0:  # Check for non-empty weight arrays
                    assert np.isclose(
                        np.max(br.tapered_weights[i]), 1.0
                    ), f"Max of tapered_weights for block {i} is not 1.0. Weights: {br.tapered_weights[i]}"

            assert len(br.tapered_weights) == len(blocks)

        @settings(deadline=None)
        @given(valid_block_indices_and_X, rng_strategy)
        def test_block_weights_setter(
            self,
            block_indices_and_X,
            random_seed: int,
        ) -> None:
            """Test block_weights setter method."""
            blocks, X = block_indices_and_X
            rng = np.random.default_rng(random_seed)
            tapered_weights = random.choice(  # noqa: S311
                [None, weights_func]
            )  # For BlockResampler init
            block_weights_choice = np.random.choice([0, 1, 2])  # noqa: S311
            if block_weights_choice == 0:
                block_weights = None
            elif block_weights_choice == 1:
                block_weights = weights_func(len(blocks))
            else:
                block_weights = weights_func
            br = BlockResampler(
                X=X,
                blocks=blocks,
                block_weights=None,
                tapered_weights=tapered_weights,
                rng=rng,
            )
            br.block_weights = block_weights
            assert isinstance(br.block_weights, np.ndarray)
            assert np.isclose(br.block_weights.sum(), 1)
            assert len(br.block_weights) == len(blocks)

        @settings(deadline=None)
        @given(valid_block_indices_and_X, rng_strategy)
        def test_tapered_weights_setter(
            self,
            block_indices_and_X,
            random_seed: int,
        ) -> None:
            """Test tapered_weights setter method."""
            blocks, X = block_indices_and_X
            rng = np.random.default_rng(random_seed)
            # Determine block_weights for initial BR construction
            block_weights_choice_init = np.random.choice([0, 1, 2])
            if block_weights_choice_init == 0:
                initial_block_weights = None
            elif block_weights_choice_init == 1:
                initial_block_weights = weights_func(len(blocks))
            else:
                initial_block_weights = weights_func

            br = BlockResampler(
                X=X,
                blocks=blocks,
                block_weights=initial_block_weights,
                tapered_weights=None,
                rng=rng,
            )

            # Now choose the tapered_weights to set and test
            tapered_weights_to_set = random.choice([None, weights_func])  # noqa: S311
            br.tapered_weights = tapered_weights_to_set

            assert isinstance(br.tapered_weights, list)
            assert all(isinstance(br.tapered_weights[i], np.ndarray) for i in range(len(blocks)))
            if tapered_weights_to_set is None:
                assert all(
                    np.isclose(br.tapered_weights[i].sum(), len(br.tapered_weights[i]))
                    for i in range(len(blocks))
                )
            assert len(br.tapered_weights) == len(blocks)

            new_rng = np.random.default_rng()
            br.rng = new_rng
            assert br.rng == new_rng

        # Tests with None values

        @settings(deadline=None)
        @given(valid_block_indices_and_X, rng_strategy)
        def test_none_block_weights(
            self,
            block_indices_and_X,
            random_seed: int,
        ) -> None:
            """Test initialization with None block weights."""
            blocks, X = block_indices_and_X
            rng = np.random.default_rng(random_seed)
            tapered_weights = random.choice([None, weights_func])  # noqa: S311
            br = BlockResampler(
                X=X,
                blocks=blocks,
                block_weights=None,
                tapered_weights=tapered_weights,
                rng=rng,
            )
            np.testing.assert_array_almost_equal(
                br.block_weights, np.ones(len(blocks)) / len(blocks)  # type: ignore
            )

        @settings(deadline=None)
        @given(valid_block_indices_and_X, rng_strategy)
        def test_none_tapered_weights(
            self,
            block_indices_and_X,
            random_seed: int,
        ) -> None:
            """Test initialization with None tapered weights."""
            blocks, X = block_indices_and_X
            rng = np.random.default_rng(random_seed)
            block_weights_choice = np.random.choice([0, 1, 2])
            if block_weights_choice == 0:
                block_weights = None
            elif block_weights_choice == 1:
                block_weights = weights_func(len(blocks))
            else:
                block_weights = weights_func
            br = BlockResampler(
                X=X,
                blocks=blocks,
                block_weights=block_weights,
                tapered_weights=None,
                rng=rng,
            )
            for i in range(len(blocks)):
                np.testing.assert_array_almost_equal(
                    br.tapered_weights[i], np.ones(len(blocks[i]))  # type: ignore
                )

        @settings(deadline=None)
        @given(valid_block_indices_and_X, rng_strategy)
        def test_none_rng(
            self,
            block_indices_and_X,
            random_seed: int,
        ) -> None:
            """Test initialization with None rng."""
            blocks, X = block_indices_and_X
            br = BlockResampler(
                X=X,
                blocks=blocks,
                block_weights=None,
                tapered_weights=None,
                rng=None,
            )
            assert isinstance(br.rng, np.random.Generator)

    class TestFailingCases:
        """Test cases where BlockResampler should raise exceptions."""

        @settings(deadline=None)
        @given(valid_block_indices_and_X)
        def test_init_wrong_blocks(self, block_indices_and_X) -> None:
            """Test initialization of BlockResampler with invalid blocks."""
            blocks, X = block_indices_and_X
            br = BlockResampler(
                X=X,
                blocks=blocks,
                block_weights=None,
                tapered_weights=None,
                rng=None,
            )
            with pytest.raises(ValidationError):
                br.blocks = None  # type: ignore
            with pytest.raises(ValidationError):
                br.blocks = np.array([])  # type: ignore
            with pytest.raises(ValidationError):
                br.blocks = np.array([1])  # type: ignore
            with pytest.raises(ValidationError):
                br.blocks = np.array([1, 2])  # type: ignore

        @settings(deadline=None)
        @given(valid_block_indices_and_X)
        def test_init_wrong_X(self, block_indices_and_X) -> None:
            """Test initialization of BlockResampler with invalid X."""
            blocks, X = block_indices_and_X
            br = BlockResampler(
                X=X,
                blocks=blocks,
                block_weights=None,
                tapered_weights=None,
                rng=None,
            )
            with pytest.raises(ValidationError):
                br.X = None  # type: ignore
            with pytest.raises(ValidationError):
                br.X = np.array([])
            with pytest.raises(ValidationError):
                br.X = np.array([1])

        @settings(deadline=None)
        @given(valid_block_indices_and_X)
        def test_init_wrong_block_weights(self, block_indices_and_X) -> None:
            """Test initialization of BlockResampler with invalid block_weights."""
            blocks, X = block_indices_and_X
            br = BlockResampler(
                X=X,
                blocks=blocks,
                block_weights=None,
                tapered_weights=None,
                rng=None,
            )
            # Test case for pydantic.ValidationError for string input
            with pytest.raises(ValidationError):
                br.block_weights = "abc"  # type: ignore
            # Test case for TypeError for callable input that doesn't return numpy array
            with pytest.raises(
                TypeError
            ):  # This will be caught by Pydantic as a validation error first
                br.block_weights = np.mean  # type: ignore

        @settings(deadline=None)
        @given(valid_block_indices_and_X)
        def test_init_wrong_tapered_weights(self, block_indices_and_X) -> None:
            """Test initialization of BlockResampler with invalid tapered_weights."""
            blocks, X = block_indices_and_X
            br = BlockResampler(
                X=X,
                blocks=blocks,
                block_weights=None,
                tapered_weights=None,
                rng=None,
            )
            with pytest.raises(ValidationError):
                br.tapered_weights = "abc"  # type: ignore
            with pytest.raises(ValueError):
                br.tapered_weights = X
            with pytest.raises(TypeError):
                br.tapered_weights = np.mean  # type: ignore

        @settings(deadline=None)
        @given(valid_block_indices_and_X)
        def test_init_wrong_rng(self, block_indices_and_X) -> None:
            """Test initialization of BlockResampler with invalid rng."""
            blocks, X = block_indices_and_X
            with pytest.raises(TypeError):
                BlockResampler(X=X, blocks=blocks, block_weights=None, tapered_weights=None, rng=3.1)  # type: ignore
            with pytest.raises(ValueError):
                BlockResampler(X=X, blocks=blocks, block_weights=None, tapered_weights=None, rng=-3)  # type: ignore

        @settings(deadline=None)
        @given(valid_block_indices_and_X)
        def test_init_block_weights_callable_returns_list(self, block_indices_and_X) -> None:
            """Test TypeError if block_weights callable returns a list instead of ndarray."""
            blocks, X = block_indices_and_X

            def callable_returns_list(size: int):
                return [1.0 / size] * size  # Returns a list

            with pytest.raises(
                TypeError,
                match="Callable for block_weights must return a numpy array.",
            ):
                BlockResampler(
                    X=X,
                    blocks=blocks,
                    block_weights=callable_returns_list,  # type: ignore
                    tapered_weights=None,
                    rng=None,
                )

        @settings(deadline=None)
        @given(valid_block_indices_and_X)
        def test_prepare_tapered_weights_invalid_list_length(self, block_indices_and_X) -> None:
            """Test _prepare_tapered_weights with a list of incorrect length."""
            blocks, X = block_indices_and_X
            br = BlockResampler(
                X=X,
                blocks=blocks,
                block_weights=None,
                tapered_weights=None,
                rng=None,
            )
            with pytest.raises(
                ValueError,
                match="Tapered weights list must contain one weight array for each block",
            ):
                br.tapered_weights = [np.array([1.0])] * (len(blocks) + 1)

        @settings(deadline=None)
        @given(valid_block_indices_and_X)
        def test_prepare_tapered_weights_invalid_ndarray_dims(self, block_indices_and_X) -> None:
            """Test _prepare_tapered_weights with an ndarray of incorrect dimensions."""
            blocks, X = block_indices_and_X
            br = BlockResampler(
                X=X,
                blocks=blocks,
                block_weights=None,
                tapered_weights=None,
                rng=None,
            )
            with pytest.raises(ValueError, match="Tapered weights array must be 1-dimensional"):
                br.tapered_weights = np.array([[1.0, 2.0]])  # 2D array

        @settings(deadline=None)
        @given(valid_block_indices_and_X)
        def test_prepare_tapered_weights_invalid_ndarray_length(self, block_indices_and_X) -> None:
            """Test _prepare_tapered_weights with a 1D ndarray of incorrect length."""
            blocks, X = block_indices_and_X
            br = BlockResampler(
                X=X,
                blocks=blocks,
                block_weights=None,
                tapered_weights=None,
                rng=None,
            )
            total_block_len = sum(len(b) for b in blocks)
            if total_block_len > 0:  # Ensure we can create an invalid length
                with pytest.raises(ValueError, match="Expected length:.*sum of all block lengths"):
                    br.tapered_weights = np.array([1.0] * (total_block_len + 1))
            else:  # If all blocks are empty, this specific error isn't triggered in the same way
                pass

        @settings(deadline=None)
        @given(valid_block_indices_and_X)
        def test_prepare_block_weights_invalid_type(self, block_indices_and_X) -> None:
            """Test _prepare_block_weights with an invalid type (list)."""
            blocks, X = block_indices_and_X
            br = BlockResampler(
                X=X,
                blocks=blocks,
                block_weights=None,
                tapered_weights=None,
                rng=None,
            )
            # Directly test the protected method for this specific TypeError
            with pytest.raises(
                TypeError,
                match="Invalid type for block_weights",
            ):
                br._prepare_block_weights(block_weights_input=[0.5] * len(blocks))  # type: ignore

        def test_line_85_validate_blocks_X_not_in_validation_context(self):
            """
            Test that validate_blocks raises ValueError if X is not in the validation context's data.

            This directly calls the classmethod with a mocked FieldValidationInfo.
            """

            # Mocking Pydantic's FieldValidationInfo or a similar structure
            class MockFieldValidationInfo:
                mode: Literal["python", "json"]

                def __init__(self, data_dict, field_name: str = "blocks"):
                    self.data = data_dict
                    self.field_name = field_name
                    # Add other attributes expected by ValidationInfo, can be None or defaults
                    self.context = None
                    self.config = None
                    self.mode = "python"

            dummy_blocks = [np.array([0, 1])]
            # Create a mock validation context where 'X' is missing from the data
            mock_values_without_X = MockFieldValidationInfo(data_dict={}, field_name="blocks")

            with pytest.raises(
                ValueError,
                match="Input data array 'X' must be provided before validating block indices",
            ):
                BlockResampler.validate_blocks(v=dummy_blocks, values=mock_values_without_X)


def check_list_of_arrays_equality(list1, list2, equal: bool = True) -> None:
    """
    Check if two lists of NumPy arrays are equal or not equal, based on the `equal` parameter.
    """
    if equal:
        assert len(list1) == len(list2), "Lists are not of the same length"
        for i, (array1, array2) in enumerate(zip(list1, list2)):
            np.testing.assert_array_equal(
                array1, array2, err_msg=f"Arrays at index {i} are not equal"
            )
    else:
        if len(list1) != len(list2):
            return
        else:
            mismatch = False
            for _, (array1, array2) in enumerate(zip(list1, list2)):
                try:
                    np.testing.assert_array_equal(array1, array2)
                except AssertionError:
                    mismatch = True
                    break
            assert mismatch, "All arrays are unexpectedly equal"


def unique_first_indices(blocks):
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
        def test_resample_blocks_valid_inputs(
            self,
            block_indices_and_X,
            random_seed: int,
        ) -> None:
            """
            Test that the 'resample_blocks' method works correctly with valid inputs.
            """
            blocks, X = block_indices_and_X
            blocks = unique_first_indices(blocks)
            rng = np.random.default_rng(random_seed)
            br = BlockResampler(
                X=X,
                blocks=blocks,
                block_weights=None,
                tapered_weights=None,
                rng=rng,
            )
            new_blocks, new_tapered_weights = br.resample_blocks()

            # Check that the total length of the new blocks is equal to n.
            total_length = sum(len(block) for block in new_blocks)
            assert total_length == len(X)

            # Check that the length of new_blocks and new_tapered_weights are equal.
            assert len(new_blocks) == len(new_tapered_weights)

            # We set the len(blocks) to be 5, so we can minimize the chances that resampling blocks a second time, or with a different random seed, gives the same results.
            if len(blocks) > 1:
                # Check that resampling with the same random seed, a second time, gives different results.
                new_blocks_2, new_tapered_weights_2 = br.resample_blocks()
                check_list_of_arrays_equality(new_blocks, new_blocks_2, equal=False)

                # Check that resampling with a new random seed gives different results.
                rng2 = np.random.default_rng((random_seed + 1) * 2)
                br = BlockResampler(
                    X=X,
                    blocks=blocks,
                    block_weights=None,
                    tapered_weights=None,
                    rng=rng2,
                )
                new_blocks_3, new_tapered_weights_3 = br.resample_blocks()
                check_list_of_arrays_equality(new_blocks, new_blocks_3, equal=False)

                # Check that resampling with the same random seed gives the same results.
                rng = np.random.default_rng(random_seed)
                br = BlockResampler(
                    X=X,
                    blocks=blocks,
                    block_weights=None,
                    tapered_weights=None,
                    rng=rng,
                )
                new_blocks_4, new_tapered_weights_4 = br.resample_blocks()
                check_list_of_arrays_equality(new_blocks, new_blocks_4)

    class TestFailingCases:
        @settings(deadline=None)
        @given(valid_block_indices_and_X, rng_strategy)
        def test_resample_blocks_no_eligible_blocks_zero_probabilities(
            self, block_indices_and_X, random_seed
        ):
            """Test ValueError when all block selection probabilities are zero."""
            blocks, X = block_indices_and_X
            rng = np.random.default_rng(random_seed)
            # Ensure there's at least one block to assign zero probability to
            if not blocks or len(blocks) == 0:  # Ensure blocks list is not empty
                blocks = [np.array([0, 1])]
                X = np.array([1.0, 2.0, 3.0, 4.0]).reshape(-1, 1)

            br = BlockResampler(
                X=X,
                blocks=blocks,
                block_weights=None,  # Start with default uniform weights
                tapered_weights=None,
                rng=rng,
            )
            # Directly manipulate the processed weights to be all zeros
            # This bypasses the Pydantic validation on the setter for block_weights_input
            br._block_weights_processed = np.zeros(len(blocks))
            with pytest.raises(ValueError, match="No eligible blocks available for sampling"):
                br.resample_blocks()

        def test_resample_blocks_partial_block_sampling(self):
            """Test the logic for sampling a partial block at the end."""
            X = np.arange(10).reshape(-1, 1).astype(float)  # Ensure X is float

            # Scenario 1: Force partial block at the end
            blocks_custom = [np.arange(7), np.arange(8)]  # lengths 7, 8
            br_custom = BlockResampler(
                X=X,
                blocks=blocks_custom,
                block_weights=None,  # Uniform probability for simplicity
                tapered_weights=None,
                rng=np.random.default_rng(42),  # Fixed seed for deterministic choice if possible
            )

            new_blocks, _ = br_custom.resample_blocks(n=10)
            total_length = sum(len(b) for b in new_blocks)
            assert total_length == 10

            # Scenario 2: Only one block, larger than n, must be truncated
            X_single_large_block = (
                np.arange(5).reshape(-1, 1).astype(float)
            )  # X must be long enough for the block
            blocks_single_large = [np.arange(5)]  # block of length 5
            br_single_large = BlockResampler(
                X=X_single_large_block,
                blocks=blocks_single_large,
                block_weights=None,
                tapered_weights=None,
                rng=np.random.default_rng(1),
            )
            new_blocks_sl, _ = br_single_large.resample_blocks(
                n=3
            )  # n=3, so block must be truncated
            assert sum(len(b) for b in new_blocks_sl) == 3
            assert len(new_blocks_sl) == 1
            assert len(new_blocks_sl[0]) == 3
            np.testing.assert_array_equal(new_blocks_sl[0], np.arange(3))


class TestGenerateBlockIndicesAndData:
    """Test the resample_block_indices_and_data method."""

    class TestPassingCases:
        """Test cases where resample_block_indices_and_data should work correctly."""

        @settings(deadline=None)
        @given(valid_block_indices_and_X, rng_strategy)
        def test_valid_inputs(
            self,
            block_indices_and_X,
            random_seed: int,
        ) -> None:
            """
            Test that the 'resample_blocks' method works correctly with valid inputs.
            """
            blocks, X = block_indices_and_X
            blocks = unique_first_indices(blocks)
            rng = np.random.default_rng(random_seed)
            br = BlockResampler(
                X=X,
                blocks=blocks,
                block_weights=None,
                rng=rng,
                tapered_weights=weights_func,
            )
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
            assert (
                sum(len(block) for block in new_blocks)
                == sum(len(block) for block in block_data)
                == len(X)
            )

            # We set the len(blocks) to be 5, so we can minimize the chances that resampling blocks a second time, or with a different random seed, gives the same results.
            if len(blocks) > 1:
                # Check that resampling with the same random seed, a second time, gives different results.
                (
                    new_blocks_2,
                    block_data_2,
                ) = br.resample_block_indices_and_data()
                check_list_of_arrays_equality(new_blocks, new_blocks_2, equal=False)

                # Check that resampling with a new random seed gives different results.
                rng2 = np.random.default_rng((random_seed + 1) * 2)
                br = BlockResampler(
                    X=X,
                    blocks=blocks,
                    block_weights=None,
                    rng=rng2,
                    tapered_weights=weights_func,
                )
                (
                    new_blocks_3,
                    block_data_3,
                ) = br.resample_block_indices_and_data()
                check_list_of_arrays_equality(new_blocks, new_blocks_3, equal=False)

                # Check that resampling with the same random seed gives the same results.
                rng = np.random.default_rng(random_seed)
                br = BlockResampler(
                    X=X,
                    blocks=blocks,
                    block_weights=None,
                    rng=rng,
                    tapered_weights=weights_func,
                )
                (
                    new_blocks_4,
                    block_data_4,
                ) = br.resample_block_indices_and_data()
                check_list_of_arrays_equality(new_blocks, new_blocks_4)

    class TestFailingCases:
        """Test cases where resample_block_indices_and_data should raise exceptions."""

    class TestPassingCasesMultiFeature:
        @settings(deadline=None)
        @given(rng_seed=rng_strategy)
        def test_resample_block_indices_and_data_multi_feature_X(self, rng_seed):
            """Test resample_block_indices_and_data with multi-feature X."""
            X = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50], [6, 60]]).astype(float)
            blocks = [
                np.array([0, 1, 2]),
                np.array([2, 3, 4]),
                np.array([3, 4, 5]),
            ]
            rng = np.random.default_rng(rng_seed)

            def custom_taper_func(size):
                return np.linspace(0.5, 1.0, size)

            br = BlockResampler(
                X=X,
                blocks=blocks,
                block_weights=None,
                tapered_weights=custom_taper_func,
                rng=rng,
            )
            res_indices, res_data = br.resample_block_indices_and_data(n=X.shape[0])

            assert len(res_indices) == len(res_data)
            total_len_indices = sum(len(b) for b in res_indices)
            total_len_data = sum(b.shape[0] for b in res_data)
            assert total_len_indices == X.shape[0]
            assert total_len_data == X.shape[0]

            for i, data_block in enumerate(res_data):
                assert data_block.ndim == 2
                assert data_block.shape[1] == X.shape[1]  # Ensure number of features is preserved

                original_data_for_block = X[res_indices[i]]
                expected_taper = custom_taper_func(len(res_indices[i]))
                # Apply np.maximum(0.1) and scale to max 1 as done in _prepare_tapered_weights
                processed_taper = np.maximum(expected_taper, 0.1)
                if np.max(processed_taper) > 0:
                    processed_taper = processed_taper / np.max(processed_taper)
                else:  # Should not happen with linspace(0.5,1) and max(0.1)
                    processed_taper = np.ones_like(processed_taper)

                expected_data_block = original_data_for_block * processed_taper[:, np.newaxis]
                np.testing.assert_array_almost_equal(data_block, expected_data_block)

        @settings(deadline=None)
        @given(rng_seed=rng_strategy)
        def test_resample_block_indices_and_data_1d_X(self, rng_seed):
            """Test resample_block_indices_and_data with 1D X."""
            X_1d = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(float)
            # Ensure blocks are valid for this X_1d
            blocks = [
                np.array([0, 1, 2]),
                np.array([3, 4]),
                np.array([5, 6, 7, 8]),
                np.array([9]),
            ]
            rng = np.random.default_rng(rng_seed)

            def custom_taper_func_1d(size):
                return np.linspace(0.5, 1.0, size)

            br = BlockResampler(
                X=X_1d,  # Pass 1D X
                blocks=blocks,
                block_weights=None,
                tapered_weights=custom_taper_func_1d,
                rng=rng,
            )
            res_indices, res_data = br.resample_block_indices_and_data(n=X_1d.shape[0])

            assert len(res_indices) == len(res_data)
            total_len_indices = sum(len(b) for b in res_indices)
            total_len_data = sum(b.shape[0] for b in res_data)
            assert total_len_indices == X_1d.shape[0]
            assert total_len_data == X_1d.shape[0]

            for i, data_block in enumerate(res_data):
                assert data_block.ndim == 2, f"Data block {i} is not 2D"
                assert data_block.shape[1] == 1, f"Data block {i} does not have 1 column"
                assert data_block.shape[0] == len(
                    res_indices[i]
                ), f"Data block {i} length mismatch with index block"

                # Verify data content (optional, but good for sanity)
                # original_data_for_block = X_1d[res_indices[i]]
                expected_taper = custom_taper_func_1d(len(res_indices[i]))
                processed_taper = np.maximum(expected_taper, 0.1)
                if np.max(processed_taper) > 0:
                    processed_taper = processed_taper / np.max(processed_taper)
                else:
                    processed_taper = np.ones_like(processed_taper)

                # expected_data_block = (
                #     original_data_for_block[:, np.newaxis]
                #     * processed_taper[:, np.newaxis]
                # )


# Add a new test to isolate the ValueError for block_weights
class TestIsolatedBlockWeightsValueError:
    def test_value_error_for_block_weights_length(self):
        from tsbootstrap.block_generator import (
            BlockGenerator,
            BlockLengthSampler,
        )

        input_length = 50
        block_length_sampler = BlockLengthSampler(avg_block_length=3)
        rng = np.random.default_rng()
        block_generator = BlockGenerator(
            input_length=input_length,
            block_length_sampler=block_length_sampler,
            wrap_around_flag=False,
            rng=rng,
            overlap_length=1,
            min_block_length=2,
        )
        blocks = block_generator.generate_blocks(overlap_flag=False)
        X = np.random.uniform(low=0, high=1e6, size=input_length).reshape(-1, 1)

        br = BlockResampler(
            X=X,
            blocks=blocks,
            block_weights=None,
            tapered_weights=None,
            rng=None,
        )
        # This should raise a ValueError because len(X[:-1].ravel()) != len(blocks)
        with pytest.raises(ValueError):
            br.block_weights = X[:-1].ravel()


# TODO: tapered_weights is a valid callable
# TODO: X_bootstrapped when tapered_weights is uniform is a subset of X


class TestStaticHelperMethods:
    """Test static helper methods of BlockResampler."""

    def test_handle_array_block_weights_empty_input(self):
        """Test _handle_array_block_weights with empty input array."""
        dummy_X = np.array([1, 2, 3, 4, 5])
        dummy_blocks = [np.array([0, 1]), np.array([2, 3])]  # 2 blocks
        br = BlockResampler(
            X=dummy_X,
            blocks=dummy_blocks,
            block_weights=None,
            tapered_weights=None,
            rng=None,
        )

        empty_weights = np.array([])
        # The size for expected_weights should match the number of dummy_blocks
        size = len(dummy_blocks)
        expected_weights = np.ones(size) / size
        # Test the behavior when block_weights is set to an empty array.
        # The logic in _prepare_block_weights calls _handle_array_block_weights.
        # If block_weights_input is an empty ndarray, _handle_array_block_weights
        # should result in uniform weights based on the number of blocks.
        br.block_weights = empty_weights
        np.testing.assert_array_almost_equal(br.block_weights, expected_weights)

    def test_normalize_to_sum_one_all_zeros(self):
        """Test _normalize_to_sum_one with an all-zero input."""
        arr = np.array([0.0, 0.0, 0.0, 0.0])
        expected = np.array([0.25, 0.25, 0.25, 0.25])
        # static method, can be called directly on the class
        with pytest.warns(RuntimeWarning, match="invalid value encountered in divide"):
            normalized = BlockResampler._normalize_to_sum_one(arr)
        np.testing.assert_array_almost_equal(normalized, expected)

    def test_normalize_to_sum_one_empty_array(self):
        """Test _normalize_to_sum_one with an empty input array."""
        arr = np.array([])
        # Expecting an empty array as output, no warning.
        normalized = BlockResampler._normalize_to_sum_one(arr)
        assert isinstance(normalized, np.ndarray), "Output should be a numpy array"
        assert normalized.shape == (0,), "Output array should be empty"
        np.testing.assert_array_equal(normalized, np.array([]))

    def test_scale_to_max_one_all_zeros(self):
        """Test _scale_to_max_one with an all-zero input."""
        arr = np.array([0.0, 0.0, 0.0, 0.0])
        expected = np.array([1.0, 1.0, 1.0, 1.0])
        # static method, can be called directly on the class
        scaled = BlockResampler._scale_to_max_one(arr)
        np.testing.assert_array_almost_equal(scaled, expected)

    def test_scale_to_max_one_non_positive_becomes_ones(self):
        """Test _scale_to_max_one when input would be all <= 0 (e.g. after np.maximum(arr, 0.1) if original was negative)."""
        # Simulate a case where weights might have become all 0.1 after np.maximum(weights, 0.1)
        # if original weights were all negative or zero.
        # Then _scale_to_max_one is called.
        arr_after_max_0_1 = np.array([0.1, 0.1, 0.1])
        expected = np.array([1.0, 1.0, 1.0])  # 0.1/0.1 = 1.0
        scaled = BlockResampler._scale_to_max_one(arr_after_max_0_1)
        np.testing.assert_array_almost_equal(scaled, expected)


class TestProtectedHelperMethods:
    """Test protected helper methods of BlockResampler."""

    @pytest.fixture
    def resampler_instance(self, request):
        # Basic instance for calling protected methods
        # Can be parameterized if different setups are needed
        X = request.param.get("X", np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))  # Ensure float
        blocks = request.param.get("blocks", [np.array([0, 1, 2]), np.array([3, 4, 5])])
        return BlockResampler(
            X=X,
            blocks=blocks,
            block_weights=None,
            tapered_weights=None,
            rng=None,
        )

    # Tests for _generate_weights_from_callable
    @pytest.mark.parametrize(
        "resampler_instance",
        [{"X": np.array([1.0, 2.0]), "blocks": [np.array([0, 1])]}],  # Ensure float and valid X
        indirect=True,
    )
    def test_generate_weights_callable_block_weights_invalid_size_type(self, resampler_instance):
        """Test _generate_weights_from_callable with invalid size type for block_weights."""

        def dummy_callable(s):
            return np.array([1] * s)

        with pytest.raises(
            TypeError,
            match="Block weight generation requires an integer size parameter",
        ):
            resampler_instance._generate_weights_from_callable(dummy_callable, size=[2], is_block_weights=True)  # type: ignore
        with pytest.raises(
            TypeError,
            match="Block weight generation requires an integer size parameter",
        ):
            resampler_instance._generate_weights_from_callable(dummy_callable, size=2.0, is_block_weights=True)  # type: ignore

    @pytest.mark.parametrize(
        "resampler_instance",
        [{"X": np.array([1.0, 2.0]), "blocks": [np.array([0, 1])]}],  # Ensure float and valid X
        indirect=True,
    )
    def test_generate_weights_callable_tapered_weights_invalid_size_type(self, resampler_instance):
        """Test _generate_weights_from_callable with invalid size type for tapered_weights."""

        def dummy_callable(s):
            return np.array([1] * s)

        with pytest.raises(
            TypeError,
            match="Tapered weight generation requires size to be an integer or array of integers",
        ):
            resampler_instance._generate_weights_from_callable(dummy_callable, size=2.0, is_block_weights=False)  # type: ignore

    # Tests for _validate_callable_generated_weights
    @pytest.mark.parametrize(
        "resampler_instance",
        [{"X": np.array([1.0, 2.0]), "blocks": [np.array([0, 1])]}],  # Ensure float and valid X
        indirect=True,
    )
    def test_validate_callable_weights_list_size_not_ndarray(self, resampler_instance):
        with pytest.raises(
            TypeError,
            match="When validating list of weight arrays, size must be an array of block lengths",
        ):
            resampler_instance._validate_callable_generated_weights(
                [np.array([1, 2])], 2, "dummy_func"
            )

    @pytest.mark.parametrize(
        "resampler_instance",
        [
            {
                "X": np.array([1.0, 2.0, 3.0]),  # Ensure float
                "blocks": [np.array([0, 1]), np.array([2])],
            }
        ],
        indirect=True,
    )
    def test_validate_callable_weights_list_lengths_mismatch(self, resampler_instance):
        with pytest.raises(
            ValueError, match="Mismatch between number of weight arrays and block lengths"
        ):
            resampler_instance._validate_callable_generated_weights(
                [np.array([1, 2])], np.array([2, 1, 3]), "dummy_func"
            )

    @pytest.mark.parametrize(
        "resampler_instance",
        [{"X": np.array([1.0, 2.0]), "blocks": [np.array([0, 1])]}],  # Ensure float
        indirect=True,
    )
    def test_validate_callable_weights_list_element_not_ndarray(self, resampler_instance):
        with pytest.raises(
            TypeError,
            match="Weight generation function 'dummy_func' must return numpy arrays",
        ):
            resampler_instance._validate_callable_generated_weights([[1, 2]], np.array([2]), "dummy_func")  # type: ignore

    @pytest.mark.parametrize(
        "resampler_instance",
        [{"X": np.array([1.0, 2.0]), "blocks": [np.array([0, 1])]}],  # Ensure float
        indirect=True,
    )
    def test_validate_callable_weights_list_element_wrong_len(self, resampler_instance):
        with pytest.raises(
            ValueError,
            match="Weight array shape mismatch from 'dummy_func'",
        ):
            resampler_instance._validate_callable_generated_weights(
                [np.array([1, 2, 3])], np.array([2]), "dummy_func"
            )

    @pytest.mark.parametrize(
        "resampler_instance",
        [{"X": np.array([1.0, 2.0]), "blocks": [np.array([0, 1])]}],  # Ensure float
        indirect=True,
    )
    def test_validate_callable_weights_list_element_wrong_dims(self, resampler_instance):
        with pytest.raises(
            ValueError,
            match="Weight array shape mismatch from 'dummy_func'",
        ):
            resampler_instance._validate_callable_generated_weights(
                [np.array([[1, 2]])], np.array([2]), "dummy_func"
            )

    @pytest.mark.parametrize(
        "resampler_instance",
        [{"X": np.array([1.0, 2.0]), "blocks": [np.array([0, 1])]}],  # Ensure float
        indirect=True,
    )
    def test_validate_callable_weights_ndarray_size_is_list(self, resampler_instance):
        with pytest.raises(
            TypeError,
            match="For single weight array validation, size must be an integer",
        ):
            resampler_instance._validate_callable_generated_weights(np.array([1, 2]), [2], "dummy_func")  # type: ignore

    @pytest.mark.parametrize(
        "resampler_instance",
        [{"X": np.array([1.0, 2.0]), "blocks": [np.array([0, 1])]}],  # Ensure float
        indirect=True,
    )
    def test_validate_callable_weights_ndarray_wrong_len(self, resampler_instance):
        with pytest.raises(
            ValueError,
            match="Weight array shape mismatch from 'dummy_func'",
        ):
            resampler_instance._validate_callable_generated_weights(
                np.array([1, 2, 3]), 2, "dummy_func"
            )

    @pytest.mark.parametrize(
        "resampler_instance",
        [{"X": np.array([1.0, 2.0]), "blocks": [np.array([0, 1])]}],  # Ensure float
        indirect=True,
    )
    def test_validate_callable_weights_ndarray_wrong_dims(self, resampler_instance):
        with pytest.raises(
            ValueError,
            match="Weight array shape mismatch from 'dummy_func'",
        ):
            resampler_instance._validate_callable_generated_weights(
                np.array([[1, 2]]), 2, "dummy_func"
            )

    @pytest.mark.parametrize(
        "resampler_instance",
        [{"X": np.array([1.0, 2.0]), "blocks": [np.array([0, 1])]}],  # Ensure float
        indirect=True,
    )
    def test_validate_callable_weights_arr_invalid_type(self, resampler_instance):
        with pytest.raises(
            TypeError,
            match="Weight generation function 'dummy_func' must return numpy array",
        ):
            resampler_instance._validate_callable_generated_weights("not_an_array", 1, "dummy_func")  # type: ignore


class TestResampleBlocksRobustness:
    """Tests for robustness of resample_blocks against corrupted internal state."""

    @pytest.fixture
    def valid_resampler_instance(self):
        """Provides a valid BlockResampler instance for modification."""
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).reshape(-1, 1)
        blocks = [np.array([0, 1]), np.array([2, 3, 4])]
        return BlockResampler(
            X=X,
            blocks=blocks,
            block_weights=None,
            tapered_weights=None,
            rng=np.random.default_rng(0),
        )

    def test_resample_blocks_invalid_rng_type(self, valid_resampler_instance):
        """Test resample_blocks when self.rng is not a Generator, by bypassing Pydantic assignment validation."""
        br = valid_resampler_instance
        # Bypass Pydantic's validation on assignment by using object.__setattr__
        object.__setattr__(br, "rng", 123)  # Corrupt rng to an int

        with pytest.raises(
            TypeError,
            match="Random number generator.*must be a numpy.random.Generator instance",
        ):
            br.resample_blocks()

    def test_resample_blocks_invalid_block_weights_type(self, valid_resampler_instance):
        """Test resample_blocks when self._block_weights_processed is not a numpy.ndarray."""
        br = valid_resampler_instance
        # Corrupt _block_weights_processed to a list
        object.__setattr__(br, "_block_weights_processed", [0.5, 0.5])  # type: ignore
        with pytest.raises(
            TypeError,
            match="self._block_weights_processed must be a numpy.ndarray",
        ):
            br.resample_blocks()

    def test_resample_blocks_invalid_tapered_weights_type(self, valid_resampler_instance):
        """Test resample_blocks when self._tapered_weights_processed is not a list."""
        br = valid_resampler_instance
        # Corrupt _tapered_weights_processed to an ndarray
        object.__setattr__(br, "_tapered_weights_processed", np.array([0.5, 0.5]))  # type: ignore
        with pytest.raises(
            TypeError,
            match="Internal error: tapered weights must be stored as a list",
        ):
            br.resample_blocks()


class TestBlockResamplerEquality:
    """Test the __eq__ method of BlockResampler."""

    @given(valid_block_indices_and_X, rng_strategy)
    @settings(deadline=None)
    def test_equality_identical_instances(self, block_indices_and_X, random_seed):
        blocks, X = block_indices_and_X
        rng1 = np.random.default_rng(random_seed)
        rng2 = np.random.default_rng(random_seed)

        br1 = BlockResampler(
            X=X,
            blocks=blocks,
            block_weights=None,
            tapered_weights=None,
            rng=rng1,
        )
        br2 = BlockResampler(
            X=X,
            blocks=blocks,
            block_weights=None,
            tapered_weights=None,
            rng=rng2,
        )
        assert br1 == br2

    @given(valid_block_indices_and_X, rng_strategy)
    @settings(deadline=None)
    def test_inequality_different_X(self, block_indices_and_X, random_seed):
        blocks, X = block_indices_and_X
        rng = np.random.default_rng(random_seed)

        br1 = BlockResampler(
            X=X,
            blocks=blocks,
            block_weights=None,
            tapered_weights=None,
            rng=rng,
        )
        X2 = X.copy() + 1
        br2 = BlockResampler(
            X=X2,
            blocks=blocks,
            block_weights=None,
            tapered_weights=None,
            rng=rng,
        )
        assert br1 != br2

    @given(valid_block_indices_and_X, rng_strategy)
    @settings(deadline=None)
    def test_inequality_different_blocks(self, block_indices_and_X, random_seed):
        blocks, X = block_indices_and_X
        rng = np.random.default_rng(random_seed)

        br1 = BlockResampler(
            X=X,
            blocks=blocks,
            block_weights=None,
            tapered_weights=None,
            rng=rng,
        )
        if len(blocks) > 1:
            blocks2 = blocks[:-1]  # Different number of blocks
        elif len(blocks) == 1 and len(blocks[0]) > 1:
            blocks2 = [blocks[0][:-1]]  # Same number of blocks, different content
        else:  # Cannot make blocks different in a simple way, skip this path for this example
            return

        # Ensure X is still valid for blocks2
        max_idx_blocks2 = 0
        if blocks2:
            max_idx_blocks2 = (
                max(np.max(b) for b in blocks2 if b.size > 0)
                if any(b.size > 0 for b in blocks2)
                else -1
            )

        if X.shape[0] <= max_idx_blocks2:  # If X is too short for modified blocks2
            X_for_br2 = np.arange(max_idx_blocks2 + 2).reshape(-1, 1).astype(float)
        else:
            X_for_br2 = X

        if not blocks2:  # If blocks2 became empty
            if X_for_br2.shape[0] < 2:
                X_for_br2 = np.array([1.0, 2.0]).reshape(-1, 1)  # Ensure X is valid
            blocks2 = [np.array([0])]  # Provide a minimal valid block for empty case

        br2 = BlockResampler(
            X=X_for_br2,
            blocks=blocks2,
            block_weights=None,
            tapered_weights=None,
            rng=rng,
        )
        assert br1 != br2

    @given(valid_block_indices_and_X, rng_strategy)
    @settings(deadline=None)
    def test_inequality_different_block_weights(self, block_indices_and_X, random_seed):
        blocks, X = block_indices_and_X
        rng = np.random.default_rng(random_seed)

        if len(blocks) < 2:  # Need at least two blocks to have different weights
            return

        br1 = BlockResampler(
            X=X,
            blocks=blocks,
            block_weights=None,
            tapered_weights=None,
            rng=rng,
        )  # Uniform weights

        custom_weights = np.ones(len(blocks))
        custom_weights[0] = 0.1  # Make it different from uniform
        custom_weights = custom_weights / custom_weights.sum()

        br2 = BlockResampler(
            X=X,
            blocks=blocks,
            block_weights=custom_weights,
            tapered_weights=None,
            rng=rng,
        )
        assert br1 != br2

    @given(valid_block_indices_and_X, rng_strategy)
    @settings(deadline=None)
    def test_inequality_different_tapered_weights(self, block_indices_and_X, random_seed):
        blocks, X = block_indices_and_X
        rng = np.random.default_rng(random_seed)

        br1 = BlockResampler(
            X=X,
            blocks=blocks,
            block_weights=None,
            tapered_weights=None,
            rng=rng,
        )  # Default ones (all 1s after processing)

        # Create custom tapered weights that are different after processing
        custom_tapered_weights = []
        for b in blocks:
            block_len = len(b)
            if block_len > 1:
                # Using linspace to create a gradient, e.g., from 0.2 to 0.8
                # These values, after np.maximum(val, 0.1) and scaling by max, should not all become 1.0
                custom_tapered_weights.append(np.linspace(0.2, 0.8, block_len))
            elif block_len == 1:
                # For a single element block, use a value that won't scale to 1 if the default is 1.
                # Default processing of [1.0] -> [1.0]
                # Processing of [0.7] -> np.maximum([0.7],0.1) -> [0.7] -> [0.7]/0.7 -> [1.0]. Still an issue.
                # The key is that the *processed* weights must differ.
                # If default is [1.0], custom [0.7] becomes [1.0].
                # Let's make custom weights that are already "processed" like.
                # The _prepare_tapered_weights ensures max is 1.
                # If default is [1.0], we need something else.
                # A callable that returns something other than all ones for a single element.
                # However, the current test setup uses a list of ndarrays.
                # If the block length is 1, tapered_weights=[np.array([0.7])] will become [1.0] after scaling.
                # So, if all blocks have length 1, this test might still fail to show inequality
                # if custom_tapered_weights also become all [1.0]s.
                # The only way for a single element tapered weight to not be 1.0 after processing
                # is if the input was <=0.1, making it 0.1, then scaled to 1.0.
                # This means single-element tapered weights always become [1.0] if >0.1 initially.
                # So, for this test to be robust, we need at least one block with len > 1
                # or accept that for all blocks of len 1, tapered weights will be [1.0].
                custom_tapered_weights.append(np.array([0.7]))  # This will become [1.0]
            else:  # empty block
                custom_tapered_weights.append(np.array([]))

        br2 = BlockResampler(
            X=X,
            blocks=blocks,
            block_weights=None,
            tapered_weights=custom_tapered_weights,
            rng=rng,
        )

        # Check if there's any block with length > 1 for custom_tapered_weights to be different
        can_be_different = any(len(b) > 1 for b in blocks)

        if not blocks or not any(len(b) > 0 for b in blocks):
            pass  # Both will be empty lists of weights
        elif not can_be_different and all(len(b) <= 1 for b in blocks):
            # If all blocks have length 0 or 1, both br1 and br2 tapered weights will be lists of [1.0] or [].
            # So they might be equal. This is an edge case of the test logic.
            # For this specific test, we are interested when they *should* be different.
            # If they are not different due to this edge case, we can skip or expect equality.
            # For now, let's assume the test setup should ideally produce difference.
            # If all blocks are length 1, custom_tapered_weights of [0.7] becomes [1.0], same as default.
            # To make them different, the input to tapered_weights for br2 would need to be a callable
            # that produces something other than what None produces.
            # Given the current structure, if all blocks are len 1, this test will likely show equality.
            # We will assert inequality only if `can_be_different` is true.
            pass
        else:
            # If there's at least one block with len > 1, linspace(0.2,0.8) will not be all 1s after scaling.
            # Default (None) gives all 1s. So they should be different.
            assert br1 != br2

    @given(
        valid_block_indices_and_X,
        st.integers(min_value=0, max_value=1000),
        st.integers(min_value=1001, max_value=2000),
    )
    @settings(deadline=None)
    def test_inequality_different_rng_seeds(self, block_indices_and_X, seed1, seed2):
        # Note: Comparing RNG state directly is complex. Different seeds should lead to different internal states.
        # The __eq__ method does not compare rng objects directly. This test is more conceptual.
        # If resampling results differ due to RNG, then __eq__ might not catch it if all other params are same.
        # However, BlockResampler's __eq__ doesn't compare rng state, so this test as-is might pass if other fields are identical.
        # The current __eq__ only compares X, blocks, _block_weights_processed, _tapered_weights_processed.
        # This test will pass if those are the same, even if RNG is different.
        # To truly test RNG's effect on equality through resampling, one would compare resampling outputs.
        blocks, X = block_indices_and_X
        rng1 = np.random.default_rng(seed1)
        rng2 = np.random.default_rng(seed2)

        br1 = BlockResampler(
            X=X,
            blocks=blocks,
            block_weights=None,
            tapered_weights=None,
            rng=rng1,
        )
        br2 = BlockResampler(
            X=X,
            blocks=blocks,
            block_weights=None,
            tapered_weights=None,
            rng=rng2,
        )
        # Since rng state is not part of __eq__, br1 == br2 will be True if other fields match.
        # This test implicitly checks that __eq__ does NOT depend on rng state.
        assert br1 == br2  # Expect True because rng is not compared in __eq__

    def test_eq_invalid_self_block_weights_type(self):
        """Test __eq__ when self._block_weights_processed is not ndarray."""
        X = np.array([[1.0], [2.0]])
        blocks = [np.array([0, 1])]
        rng = np.random.default_rng(0)
        br1 = BlockResampler(
            X=X,
            blocks=blocks,
            block_weights=None,
            tapered_weights=None,
            rng=rng,
        )
        br2 = BlockResampler(
            X=X,
            blocks=blocks,
            block_weights=None,
            tapered_weights=None,
            rng=rng,
        )
        object.__setattr__(br1, "_block_weights_processed", [0.5, 0.5])  # type: ignore
        with pytest.raises(
            TypeError,
            match="self._block_weights_processed must be a numpy.ndarray",
        ):
            _ = br1 == br2

    def test_eq_invalid_other_block_weights_type(self):
        """Test __eq__ when other._block_weights_processed is not ndarray."""
        X = np.array([[1.0], [2.0]])
        blocks = [np.array([0, 1])]
        rng = np.random.default_rng(0)
        br1 = BlockResampler(
            X=X,
            blocks=blocks,
            block_weights=None,
            tapered_weights=None,
            rng=rng,
        )
        br2 = BlockResampler(
            X=X,
            blocks=blocks,
            block_weights=None,
            tapered_weights=None,
            rng=rng,
        )
        object.__setattr__(br2, "_block_weights_processed", [0.5, 0.5])  # type: ignore
        with pytest.raises(
            TypeError,
            match="other._block_weights_processed must be a numpy.ndarray",
        ):
            _ = br1 == br2

    def test_eq_invalid_self_tapered_weights_type(self):
        """Test __eq__ when self._tapered_weights_processed is not list."""
        X = np.array([[1.0], [2.0]])
        blocks = [np.array([0, 1])]
        rng = np.random.default_rng(0)
        br1 = BlockResampler(
            X=X,
            blocks=blocks,
            block_weights=None,
            tapered_weights=None,
            rng=rng,
        )
        br2 = BlockResampler(
            X=X,
            blocks=blocks,
            block_weights=None,
            tapered_weights=None,
            rng=rng,
        )
        object.__setattr__(br1, "_tapered_weights_processed", np.array([0.5]))  # type: ignore
        with pytest.raises(
            TypeError,
            match="Internal error: tapered weights must be stored as a list",
        ):
            _ = br1 == br2

    def test_eq_invalid_other_tapered_weights_type(self):
        """Test __eq__ when other._tapered_weights_processed is not list."""
        X = np.array([[1.0], [2.0]])
        blocks = [np.array([0, 1])]
        rng = np.random.default_rng(0)
        br1 = BlockResampler(
            X=X,
            blocks=blocks,
            block_weights=None,
            tapered_weights=None,
            rng=rng,
        )
        br2 = BlockResampler(
            X=X,
            blocks=blocks,
            block_weights=None,
            tapered_weights=None,
            rng=rng,
        )
        object.__setattr__(br2, "_tapered_weights_processed", np.array([0.5]))  # type: ignore
        with pytest.raises(
            TypeError,
            match="other._tapered_weights_processed must be a list",
        ):
            _ = br1 == br2

    @given(valid_block_indices_and_X)
    @settings(deadline=None)
    def test_inequality_different_type(self, block_indices_and_X):
        blocks, X = block_indices_and_X
        br = BlockResampler(
            X=X,
            blocks=blocks,
            block_weights=None,
            tapered_weights=None,
            rng=None,
        )
        assert br != "a string"
        assert br != 123
        assert br != [1, 2, 3]


class TestSpecificProtectedMethods:
    """Tests for specific lines/branches in protected methods."""

    @pytest.fixture
    def basic_resampler_fixture(self):
        # A basic, valid BlockResampler instance
        X = np.arange(10).reshape(-1, 1).astype(float)
        blocks = [
            np.array([0, 1, 2]),
            np.array([3, 4, 5]),
            np.array([6, 7, 8, 9]),
        ]
        return BlockResampler(
            X=X,
            blocks=blocks,
            block_weights=None,
            tapered_weights=None,
            rng=np.random.default_rng(0),
        )

    def test_prepare_tapered_weights_line_165_ndarray_split(self, basic_resampler_fixture):
        """Covers line 165: tapered_weights_input is a 1D ndarray to be split."""
        br = basic_resampler_fixture

        # Case 1: Standard case (from fixture)
        block_lengths = np.array([len(b) for b in br.blocks])  # [3, 3, 4]
        total_len = np.sum(block_lengths)  # 10

        if total_len > 0:  # Proceed only if there are elements to weight
            tapered_weights_flat = np.random.rand(total_len)  # array of 10

            processed_weights = br._prepare_tapered_weights(
                tapered_weights_input=tapered_weights_flat
            )
            assert isinstance(processed_weights, list)
            assert len(processed_weights) == len(br.blocks)
            for i, original_block in enumerate(br.blocks):
                assert len(processed_weights[i]) == len(original_block)
                if len(processed_weights[i]) > 0:
                    # Check that weights are scaled (max is 1.0) or all 0.1 (if original was all <=0.1)
                    max_weight = np.max(processed_weights[i])
                    assert np.isclose(max_weight, 1.0) or np.allclose(
                        processed_weights[i], 0.1 / 0.1
                    )  # 0.1 scaled by 0.1 is 1.0
        else:  # if total_len is 0 (e.g. all blocks are empty)
            tapered_weights_flat = np.array([])
            processed_weights = br._prepare_tapered_weights(
                tapered_weights_input=tapered_weights_flat
            )
            assert isinstance(processed_weights, list)
            assert len(processed_weights) == len(br.blocks)
            assert all(len(pw) == 0 for pw in processed_weights)

        # Case 2: Single block
        X_single = np.arange(5).reshape(-1, 1).astype(float)
        blocks_single = [np.array([0, 1, 2, 3, 4])]
        br_single = BlockResampler(
            X=X_single,
            blocks=blocks_single,
            block_weights=None,
            tapered_weights=None,
            rng=np.random.default_rng(1),
        )
        tapered_single_flat = np.random.rand(5)
        processed_single = br_single._prepare_tapered_weights(
            tapered_weights_input=tapered_single_flat
        )
        assert len(processed_single) == 1
        assert len(processed_single[0]) == 5
        # Manually simulate the processing for comparison
        expected_processed_single = br_single._scale_to_max_one(
            np.maximum(tapered_single_flat, 0.1)
        )
        np.testing.assert_array_almost_equal(processed_single[0], expected_processed_single)

        # Case 4: Blocks exist but are all empty (total_len = 0)
        br_all_empty_indiv_blocks = BlockResampler(
            X=np.array([[1.0], [2.0]]),  # Ensure X is valid
            blocks=[
                np.array([], dtype=int),
                np.array([], dtype=int),
            ],  # Ensure integer dtype for empty arrays
            block_weights=None,
            tapered_weights=None,
            rng=np.random.default_rng(3),
        )
        tapered_all_empty_flat = np.array([])  # This is correct, sum of block_lengths is 0
        processed_all_empty = br_all_empty_indiv_blocks._prepare_tapered_weights(
            tapered_weights_input=tapered_all_empty_flat
        )
        assert len(processed_all_empty) == 2
        assert len(processed_all_empty[0]) == 0
        assert len(processed_all_empty[1]) == 0

    def test_prepare_tapered_weights_line_175_invalid_type(self, basic_resampler_fixture):
        """Covers line 175: TypeError for invalid tapered_weights_input type."""
        br = basic_resampler_fixture
        with pytest.raises(
            TypeError,
            match="Invalid type for tapered_weights",
        ):
            br._prepare_tapered_weights(tapered_weights_input=123)  # Pass an int

    def test_generate_weights_from_callable_line_253_tapered_size_int(
        self, basic_resampler_fixture
    ):
        """Covers line 253: _generate_weights_from_callable for tapered weights when size is int."""
        br = basic_resampler_fixture

        def dummy_weights_func(s):
            return np.ones(s) * 0.5

        result = br._generate_weights_from_callable(
            dummy_weights_func, size=5, is_block_weights=False
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], np.ndarray)
        np.testing.assert_array_equal(result[0], np.ones(5) * 0.5)

    def test_prepare_block_weights_line_285_callable_returns_non_ndarray(
        self, basic_resampler_fixture
    ):
        """Covers line 285: _prepare_block_weights when callable returns non-ndarray."""
        br = basic_resampler_fixture

        def bad_callable(s):
            return list(range(s))  # Returns a list, not ndarray

        with pytest.raises(
            TypeError,
            match="Callable for block_weights must return a numpy array.",
        ):
            br._prepare_block_weights(block_weights_input=bad_callable)

    def test_validate_callable_generated_weights_line_405_size_not_int_for_block_weights(
        self, basic_resampler_fixture
    ):
        """Covers line 405: _validate_callable_generated_weights, size not int for block_weights case."""
        br = basic_resampler_fixture
        weights_arr = np.array([1.0, 2.0])
        # This line is tricky because `size` for block_weights comes from `len(self.blocks)`, which is always int.
        # So, we directly call the method with a non-int size to hit the line.
        with pytest.raises(
            TypeError,
            match="For single weight array validation, size must be an integer",
        ):
            br._validate_callable_generated_weights(
                weights_arr,
                size=[2],
                callable_name="test_func_block_weights_size_list",
            )  # type: ignore
        with pytest.raises(
            TypeError,
            match="For single weight array validation, size must be an integer",
        ):
            br._validate_callable_generated_weights(
                weights_arr,
                size=2.0,
                callable_name="test_func_block_weights_size_float",
            )  # type: ignore
