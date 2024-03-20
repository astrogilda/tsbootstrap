import warnings

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.random import default_rng
from tsbootstrap.block_generator import BlockGenerator
from tsbootstrap.block_length_sampler import BlockLengthSampler

MIN_INT_VALUE = 1
MAX_INT_VALUE = 2**32 - 1


class TestInit:
    class TestPassingCases:
        """
        Test class for passing tests of BlockGenerator __init__ method.
        """

        @given(
            st.integers(min_value=10, max_value=MAX_INT_VALUE),
            st.booleans(),
            st.integers(min_value=MIN_INT_VALUE, max_value=MAX_INT_VALUE),
            st.integers(min_value=2, max_value=MAX_INT_VALUE),
            st.integers(min_value=2, max_value=10),
        )
        def test_init_with_valid_args(
            self,
            input_length,
            wrap_around_flag,
            overlap_length,
            min_block_length,
            avg_block_length,
        ):
            """
            Test BlockGenerator initialization with valid arguments.
            """
            block_length_sampler = BlockLengthSampler(
                avg_block_length=avg_block_length
            )
            print(f"block_length_sampler.type: {type(block_length_sampler)}")
            rng = default_rng()

            block_generator = BlockGenerator(
                block_length_sampler,
                input_length,
                wrap_around_flag,
                rng,
                overlap_length=overlap_length,
                min_block_length=min_block_length,
            )

            assert block_generator.block_length_sampler == block_length_sampler
            assert block_generator.input_length == input_length
            assert block_generator.wrap_around_flag == wrap_around_flag
            assert block_generator.rng == rng
            assert block_generator.overlap_length == overlap_length
            assert block_generator.min_block_length == min(
                min_block_length, avg_block_length
            )

    class TestFailingCases:
        """
        Test class for failing tests of BlockGenerator __init__ method.
        """

        @given(
            st.integers(max_value=2),
            st.booleans(),
            st.integers(min_value=1),
            st.integers(min_value=2),
        )
        def test_init_with_invalid_input_length(
            self,
            input_length,
            wrap_around_flag,
            overlap_length,
            min_block_length,
        ):
            """
            Test BlockGenerator initialization with invalid input_length (<= 2).
            """
            block_length_sampler = BlockLengthSampler(avg_block_length=3)
            rng = default_rng()

            with pytest.raises(ValueError):
                BlockGenerator(
                    block_length_sampler,
                    input_length,
                    wrap_around_flag,
                    rng,
                    overlap_length=overlap_length,
                    min_block_length=min_block_length,
                )

        @given(
            st.integers(min_value=3, max_value=MAX_INT_VALUE),
            st.booleans(),
            st.integers(max_value=0),
            st.integers(min_value=2, max_value=MAX_INT_VALUE),
        )
        def test_init_with_invalid_overlap_length(
            self,
            input_length,
            wrap_around_flag,
            overlap_length,
            min_block_length,
        ):
            """
            Test BlockGenerator initialization with invalid overlap_length (< 1).
            """
            block_length_sampler = BlockLengthSampler(avg_block_length=3)
            rng = default_rng()

            with pytest.warns(
                UserWarning, match=r".*'overlap_length' should be >= 1.*"
            ):
                BlockGenerator(
                    block_length_sampler,
                    input_length,
                    wrap_around_flag,
                    rng,
                    overlap_length=overlap_length,
                    min_block_length=min_block_length,
                )

        @given(
            st.integers(min_value=3, max_value=MAX_INT_VALUE),
            st.booleans(),
            st.integers(min_value=MIN_INT_VALUE, max_value=MAX_INT_VALUE),
            st.integers(max_value=0),
        )
        def test_init_with_invalid_min_block_length(
            self,
            input_length,
            wrap_around_flag,
            overlap_length,
            min_block_length,
        ):
            """
            Test BlockGenerator initialization with invalid min_block_length (<= 1).
            """
            # Always display UserWarning
            warnings.simplefilter("always")
            block_length_sampler = BlockLengthSampler(avg_block_length=3)
            rng = default_rng()

            with pytest.warns(
                UserWarning,
                match=r".*'min_block_length' should be >= 1. Setting it to 1.*",
            ):
                BlockGenerator(
                    block_length_sampler,
                    input_length,
                    wrap_around_flag,
                    rng,
                    overlap_length=overlap_length,
                    min_block_length=min_block_length,
                )

        @given(st.integers(min_value=11))
        def test_generate_non_overlapping_blocks_large_block_length(
            self, block_length
        ):
            """
            Test BlockGenerator generate_non_overlapping_blocks method with large block_length.
            """
            block_length_sampler = BlockLengthSampler(
                avg_block_length=block_length
            )
            rng = default_rng()

            with pytest.raises(ValueError):
                BlockGenerator(block_length_sampler, 10, rng=rng)

        @given(st.integers(min_value=1, max_value=2))
        def test_generate_non_overlapping_blocks_invalid_input_length(
            self, input_length
        ):
            """
            Test BlockGenerator generate_non_overlapping_blocks method with invalid input_length.
            """
            block_length_sampler = BlockLengthSampler(avg_block_length=3)
            rng = default_rng()

            with pytest.raises(ValueError):
                BlockGenerator(block_length_sampler, input_length, rng=rng)


def assert_unique_arrays(array_list):
    """
    Asserts if all arrays in a list are unique.

    It converts each array into a tuple and adds it to a set,
    then checks if the size of the set is equal to the length of the list.
    """
    array_set = set()

    for arr in array_list:
        # Convert the array to a tuple and add it to the set
        array_set.add(tuple(arr))

    # Use an assert statement to check if the size of the set is equal to the length of the list
    assert len(array_set) == len(
        array_list
    ), "Some arrays in the list are not unique."


class TestGenerateNonOverlappingBlocks:
    class TestPassingCases:
        """
        Test class for successful tests of BlockGenerator generate_non_overlapping_blocks method.
        """

        @pytest.mark.parametrize(
            "input_length, wrap_around_flag, block_length, expected_output",
            [
                (
                    10,
                    False,
                    3,
                    [
                        np.arange(0, 3),
                        np.arange(3, 6),
                        np.arange(6, 9),
                        np.arange(9, 10),
                    ],
                ),
                (
                    5,
                    False,
                    2,
                    [np.arange(0, 2), np.arange(2, 4), np.arange(4, 5)],
                ),
                (
                    10,
                    True,
                    3,
                    [
                        np.arange(0, 3),
                        np.arange(3, 6),
                        np.arange(6, 9),
                        np.arange(9, 10),
                    ],
                ),
                (
                    5,
                    True,
                    2,
                    [np.arange(0, 2), np.arange(2, 4), np.arange(4, 5)],
                ),
                (10, False, 10, [np.arange(0, 10)]),
                (5, False, 5, [np.arange(0, 5)]),
            ],
        )
        def test_generate_non_overlapping_blocks(
            self, input_length, wrap_around_flag, block_length, expected_output
        ):
            """
            Test BlockGenerator generate_non_overlapping_blocks method with valid arguments.
            """
            block_length_sampler = BlockLengthSampler(
                avg_block_length=block_length
            )
            block_generator = BlockGenerator(
                block_length_sampler, input_length, wrap_around_flag
            )
            generated_blocks = (
                block_generator.generate_non_overlapping_blocks()
            )

            assert len(generated_blocks) == len(expected_output)

            if not wrap_around_flag:
                for gb, eo in zip(generated_blocks, expected_output):
                    assert np.array_equal(gb, eo)

            assert_unique_arrays(generated_blocks)


expected_output0 = [np.arange(i, i + 2) for i in range(9)]
expected_output1 = [
    np.array([0, 1, 2, 3, 4]),
    np.array([4, 5, 6, 7, 8]),
    np.array([8, 9]),
]
expected_output2 = [
    np.array([0, 1, 2, 3, 4]),
    np.array([1, 2, 3, 4, 5]),
    np.array([2, 3, 4, 5, 6]),
    np.array([3, 4, 5, 6, 7]),
    np.array([4, 5, 6, 7, 8]),
    np.array([5, 6, 7, 8, 9]),
    np.array([6, 7, 8, 9]),
    np.array([7, 8, 9]),
    np.array([8, 9]),
]
expected_output3 = [
    np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    np.array([5, 6, 7, 8, 9]),
    np.array([6, 7, 8, 9]),
    np.array([7, 8, 9]),
    np.array([8, 9]),
]
expected_output4 = [
    np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
    np.array([2, 3, 4, 5, 6, 7, 8, 9]),
    np.array([3, 4, 5, 6, 7, 8, 9]),
    np.array([4, 5, 6, 7, 8, 9]),
    np.array([5, 6, 7, 8, 9]),
    np.array([6, 7, 8, 9]),
    np.array([7, 8, 9]),
    np.array([8, 9]),
]
expected_output5 = [np.array([0, 1, 2, 3, 4]), np.array([4, 5, 6, 7, 8])]
expected_output6 = [
    np.array([0, 1, 2, 3, 4]),
    np.array([1, 2, 3, 4, 5]),
    np.array([2, 3, 4, 5, 6]),
    np.array([3, 4, 5, 6, 7]),
    np.array([4, 5, 6, 7, 8]),
    np.array([5, 6, 7, 8, 9]),
]
expected_output7 = [
    np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    np.array([5, 6, 7, 8, 9]),
]
expected_output8 = [
    np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
    np.array([2, 3, 4, 5, 6, 7, 8, 9]),
    np.array([3, 4, 5, 6, 7, 8, 9]),
    np.array([4, 5, 6, 7, 8, 9]),
    np.array([5, 6, 7, 8, 9]),
]


class TestGenerateOverlappingBlocks:
    class TestPassingCases:
        """
        Test class for successful tests of BlockGenerator generate_non_overlapping_blocks method.
        """

        @pytest.mark.parametrize(
            "input_length, wrap_around_flag, block_length, overlap_length, min_block_length, expected_output",
            [
                (10, False, 2, 1, 2, expected_output0),
                (10, False, 5, 1, 2, expected_output1),
                (10, False, 10, 1, 2, [np.arange(10)]),
                (10, False, 2, 5, 2, expected_output0),
                (10, False, 5, 5, 2, expected_output2),
                (10, False, 10, 5, 2, expected_output3),
                (10, False, 2, 10, 2, expected_output0),
                (10, False, 5, 10, 2, expected_output2),
                (10, False, 10, 10, 2, expected_output4),
                (10, False, 2, 11, 2, expected_output0),
                (10, False, 5, 11, 2, expected_output2),
                (10, False, 2, 1, 5, expected_output0),
                (10, False, 5, 1, 5, expected_output5),
                (10, False, 10, 1, 5, [np.arange(10)]),
                (10, False, 2, 5, 5, expected_output0),
                (10, False, 5, 5, 5, expected_output6),
                (10, False, 10, 5, 5, expected_output7),
                (10, False, 2, 10, 5, expected_output0),
                (10, False, 5, 10, 5, expected_output6),
                (10, False, 10, 10, 5, expected_output8),
                (10, False, 2, 11, 5, expected_output0),
                (10, False, 5, 11, 5, expected_output6),
                (10, False, 10, 11, 5, expected_output8),
                (10, False, 2, 1, 10, expected_output0),
                (10, False, 5, 1, 10, expected_output5),
                (10, False, 10, 1, 10, [np.arange(10)]),
                (10, False, 2, 5, 10, expected_output0),
                (10, False, 5, 5, 10, expected_output6),
                (10, False, 10, 5, 10, [np.arange(10)]),
                (10, False, 2, 10, 10, expected_output0),
                (10, False, 5, 10, 10, expected_output6),
                (10, False, 10, 10, 10, [np.arange(10)]),
                (10, False, 2, 11, 10, expected_output0),
                (10, False, 5, 11, 10, expected_output6),
                (10, False, 10, 11, 10, [np.arange(10)]),
                (10, False, 2, 1, 11, expected_output0),
                (10, False, 5, 1, 11, expected_output5),
                (10, False, 10, 1, 11, [np.arange(10)]),
                (10, False, 2, 5, 11, expected_output0),
                (10, False, 5, 5, 11, expected_output6),
                (10, False, 10, 5, 11, [np.arange(10)]),
                (10, False, 2, 10, 11, expected_output0),
                (10, False, 5, 10, 11, expected_output6),
                (10, False, 10, 10, 11, [np.arange(10)]),
                (10, False, 2, 11, 11, expected_output0),
                (10, False, 5, 11, 11, expected_output6),
                (10, False, 10, 11, 11, [np.arange(10)]),
                # (10, True, 10, 11, 11, None)
            ],
        )
        def test_generate_overlapping_blocks(
            self,
            input_length,
            wrap_around_flag,
            block_length,
            overlap_length,
            min_block_length,
            expected_output,
        ):
            """
            Test BlockGenerator generate_non_overlapping_blocks method with valid arguments.
            """
            block_length_sampler = BlockLengthSampler(
                avg_block_length=block_length
            )
            block_generator = BlockGenerator(
                block_length_sampler=block_length_sampler,
                input_length=input_length,
                wrap_around_flag=wrap_around_flag,
                overlap_length=overlap_length,
                min_block_length=min_block_length,
            )
            generated_blocks = block_generator.generate_overlapping_blocks()

            print(f"generated_blocks: {generated_blocks}")
            assert len(generated_blocks) == len(expected_output)

            if not wrap_around_flag:
                for gb, eo in zip(generated_blocks, expected_output):
                    assert np.array_equal(gb, eo)

            assert_unique_arrays(generated_blocks)
