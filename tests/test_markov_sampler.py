from numbers import Integral
from typing import Any

import numpy as np
import pytest
import scipy
from hypothesis import given, settings
from hypothesis import strategies as st
from pytest import approx
from skbase.utils.dependencies import _check_soft_dependencies
from sklearn.decomposition import PCA
from tsbootstrap import (
    BlockCompressor,
    MarkovSampler,
    MarkovTransitionMatrixCalculator,
)


def generate_random_blocks(n_blocks: int, block_size, min_val=0, max_val=10):
    """
    Generate a list of random time series data blocks.

    Parameters
    ----------
    n_blocks : int
        Number of blocks to generate.
    block_size : tuple of int
        Size of each block.
    min_val : int, optional
        Minimum value in each block.
    max_val : int, optional
        Maximum value in each block.

    Returns
    -------
    List[np.ndarray]
        List of numpy arrays, each with shape block_size.
    """
    if n_blocks <= 0 or not isinstance(n_blocks, Integral):
        raise ValueError("'n_blocks' should be a positive integer.")
    if not (isinstance(block_size, tuple) and len(block_size) == 2):
        raise ValueError("'block_size' should be a tuple of 2 integers.")
    return [
        np.random.randint(min_val, max_val, block_size) * np.random.random()
        for _ in range(n_blocks)
    ]


# Use pytest.mark.skipif decorator to skip this class if dtaidistance is not installed
@pytest.mark.skipif(
    not _check_soft_dependencies("dtaidistance", severity="none"),
    reason="skip test if required soft dependency not available",
)
class TestMarkovTransitionMatrixCalculator:
    class TestCalculateTransitionProbabilities:
        class TestPassingCases:
            def test_constant_blocks(self):
                """
                Test calculate_transition_probabilities with constant blocks.
                """
                blocks = [
                    np.ones((10, 2)) for _ in range(3)
                ]  # 3 blocks of constant time series data
                transition_probabilities = MarkovTransitionMatrixCalculator.calculate_transition_probabilities(
                    blocks
                )
                assert transition_probabilities.shape == (
                    len(blocks),
                    len(blocks),
                )
                assert np.allclose(np.sum(transition_probabilities, axis=1), 1)
                # Check that transition probabilities are equal for constant blocks
                expected_probability = 1 / len(blocks)
                assert np.allclose(
                    transition_probabilities, expected_probability
                )

            @pytest.mark.parametrize(
                "n_blocks,n_features", [(2, 2), (5, 3), (10, 4)]
            )
            def test_random_blocks(self, n_blocks, n_features):
                """
                Test calculate_transition_probabilities with random blocks.
                """
                blocks = generate_random_blocks(n_blocks, (10, n_features))
                transition_probabilities = MarkovTransitionMatrixCalculator.calculate_transition_probabilities(
                    blocks
                )
                assert transition_probabilities.shape == (n_blocks, n_blocks)
                assert np.allclose(np.sum(transition_probabilities, axis=1), 1)

            def test_random_blocks_different_sizes(self):
                """
                Test calculate_transition_probabilities with random blocks of different sizes.
                """
                blocks = generate_random_blocks(
                    3, (10, 2)
                ) + generate_random_blocks(2, (20, 2))
                transition_probabilities = MarkovTransitionMatrixCalculator.calculate_transition_probabilities(
                    blocks
                )
                assert transition_probabilities.shape == (
                    len(blocks),
                    len(blocks),
                )
                assert np.allclose(np.sum(transition_probabilities, axis=1), 1)

            @pytest.mark.parametrize("n_blocks", [1, 5, 10])
            def test_multiple_blocks_same_size(self, n_blocks):
                """
                Test calculate_transition_probabilities with multiple blocks of the same size.
                """
                blocks = generate_random_blocks(n_blocks, (10, 2))
                transition_probabilities = MarkovTransitionMatrixCalculator.calculate_transition_probabilities(
                    blocks
                )
                assert transition_probabilities.shape == (n_blocks, n_blocks)
                assert np.allclose(np.sum(transition_probabilities, axis=1), 1)

        class TestFailingCases:
            def test_empty_list(self):
                """
                Test calculate_transition_probabilities with an empty list of blocks.
                """
                blocks = []
                with pytest.raises(ValueError):
                    MarkovTransitionMatrixCalculator.calculate_transition_probabilities(
                        blocks
                    )

            def test_none_blocks(self):
                """
                Test calculate_transition_probabilities where the blocks list contains None.
                """
                blocks = [np.array([[0, 1], [1, 0]]), None]
                with pytest.raises(TypeError):
                    MarkovTransitionMatrixCalculator.calculate_transition_probabilities(
                        blocks
                    )

            def test_incompatible_block_shapes(self):
                """
                Test calculate_transition_probabilities where blocks have incompatible shapes.
                """
                blocks = [np.array([[0, 1], [1, 0]]), np.array([0, 1])]
                with pytest.raises(ValueError):
                    MarkovTransitionMatrixCalculator.calculate_transition_probabilities(
                        blocks
                    )

            @pytest.mark.parametrize("n_blocks", [0, -1])
            def test_invalid_number_of_blocks(self, n_blocks):
                """
                Test calculate_transition_probabilities with an invalid number of blocks.
                """
                with pytest.raises(ValueError):
                    blocks = generate_random_blocks(n_blocks, (10, 2))
                    MarkovTransitionMatrixCalculator.calculate_transition_probabilities(
                        blocks
                    )

            def test_different_number_of_features(self):
                """
                Test calculate_transition_probabilities where blocks have a different number of features.
                """
                blocks = [np.random.rand(10, 2), np.random.rand(10, 3)]
                with pytest.raises(ValueError):
                    MarkovTransitionMatrixCalculator.calculate_transition_probabilities(
                        blocks
                    )

            def test_non_ndarray_blocks(self):
                """
                Test calculate_transition_probabilities where one or more blocks are not numpy ndarrays.
                """
                blocks = [np.random.rand(10, 2), [1, 2, 3]]
                with pytest.raises(TypeError):
                    MarkovTransitionMatrixCalculator.calculate_transition_probabilities(
                        blocks
                    )

            @pytest.mark.parametrize(
                "n_blocks,block_size", [(0, (10, 2)), (-1, (10, 2))]
            )
            def test_invalid_generation_params(self, n_blocks, block_size):
                """
                Test generate_random_blocks with invalid parameters.
                """
                with pytest.raises(ValueError):
                    generate_random_blocks(n_blocks, block_size)


methods = [x["method"] for x in BlockCompressor.get_test_params()]

# Hypothesis strategies
valid_method = st.sampled_from(methods)
invalid_method = st.text().filter(
    lambda x: x
    not in [
        "first",
        "middle",
        "last",
        "mean",
        "mode",
        "median",
        "kmeans",
        "kmedians",
        "kmedoids",
    ]
)
valid_apply_pca = st.booleans()
valid_pca = st.just(PCA(n_components=1))
invalid_pca = st.just(PCA(n_components=2))
rng_generator = st.integers(min_value=0, max_value=2**32 - 1)


class TestBlockCompressor:
    class TestInitAndGettersAndSetters:
        class TestPassingCases:
            """
            A class containing passing test cases for BlockCompressor.
            """

            @given(valid_method, valid_apply_pca, valid_pca, rng_generator)
            def test_initialization_pass(
                self, method, apply_pca_flag, pca, random_seed
            ):
                """
                Test that BlockCompressor can be initialized with valid arguments.
                """
                BlockCompressor(method, apply_pca_flag, pca, random_seed)

            @given(valid_method)
            def test_method_setter_pass(self, method):
                """
                Test that BlockCompressor's method can be set with valid values.
                """
                bc = BlockCompressor()
                bc.method = method

            @given(valid_apply_pca)
            def test_apply_pca_setter_pass(self, apply_pca_flag):
                """
                Test that BlockCompressor's apply_pca_flag can be set with valid values.
                """
                bc = BlockCompressor()
                bc.apply_pca_flag = apply_pca_flag

            @given(valid_pca)
            def test_pca_setter_pass(self, pca):
                """
                Test that BlockCompressor's pca can be set with valid values.
                """
                bc = BlockCompressor()
                bc.pca = pca

            @given(rng_generator)
            def test_rng_setter_pass(self, random_seed):
                """
                Test that BlockCompressor's rng can be set with valid values.
                """
                bc = BlockCompressor()
                bc.random_seed = random_seed

        class TestFailingCases:
            """
            A class containing failing test cases for BlockCompressor.
            """

            @given(invalid_method, valid_apply_pca, valid_pca, rng_generator)
            def test_initialization_fail_invalid_method(
                self, method, apply_pca_flag, pca, random_seed
            ):
                """
                Test that BlockCompressor initialization fails with invalid method.
                """
                with pytest.raises(ValueError):
                    BlockCompressor(method, apply_pca_flag, pca, random_seed)

            @given(valid_method, valid_apply_pca, invalid_pca, rng_generator)
            def test_initialization_fail_invalid_pca(
                self, method, apply_pca_flag, pca, random_seed
            ):
                """
                Test that BlockCompressor initialization fails with invalid pca.
                """
                with pytest.raises(ValueError):
                    BlockCompressor(method, apply_pca_flag, pca, random_seed)

            @given(invalid_method)
            def test_method_setter_fail(self, method):
                """
                Test that BlockCompressor's method setter fails with invalid values.
                """
                bc = BlockCompressor()
                with pytest.raises(ValueError):
                    bc.method = method

            @given(st.integers())
            def test_apply_pca_setter_fail(self, apply_pca_flag):
                """
                Test that BlockCompressor's apply_pca_flag setter fails with non-boolean values.
                """
                bc = BlockCompressor()
                with pytest.raises(TypeError):
                    bc.apply_pca_flag = apply_pca_flag

            @given(invalid_pca)
            def test_pca_setter_fail(self, pca):
                """
                Test that BlockCompressor's pca setter fails with invalid pca.
                """
                bc = BlockCompressor()
                with pytest.raises(ValueError):
                    bc.pca = pca

            @given(st.text())
            def test_rng_setter_fail(self, random_seed):
                """
                Test that BlockCompressor's rng setter fails with non-Generator values.
                """
                bc = BlockCompressor()
                with pytest.raises(TypeError):
                    bc.random_seed = random_seed

    class TestSummarizeBlocks:
        class TestPassingCases:
            @settings(deadline=None, derandomize=True)
            @given(valid_method, valid_apply_pca, valid_pca, rng_generator)
            def test_valid_methods(self, method, apply_pca_flag, pca, rng):
                """
                Test if the function correctly processes blocks for all valid methods.
                """
                blocks = [np.random.rand(10, 2) for _ in range(3)]
                bc = BlockCompressor(
                    method=method,
                    apply_pca_flag=apply_pca_flag,
                    pca=pca,
                    random_seed=rng,
                )
                try:
                    summarized_blocks = bc.summarize_blocks(blocks)
                    assert summarized_blocks.shape == (
                        len(blocks),
                        blocks[0].shape[1],
                    )
                # pyclustering.kmedians raises this error and results in a `flaky test` error from hypothesis
                except OSError:
                    pass

            @settings(deadline=None)
            @given(valid_method, valid_apply_pca, valid_pca, rng_generator)
            def test_unequal_sub_block_sizes(
                self, method, apply_pca_flag, pca, rng
            ):
                """
                Test if the function correctly processes blocks for all valid methods, even when sub-blocks of unequal sizes are provided.
                """
                blocks = [np.random.rand(10, 2), np.random.rand(5, 2)]
                bc = BlockCompressor(
                    method=method,
                    apply_pca_flag=apply_pca_flag,
                    pca=pca,
                    random_seed=rng,
                )
                summarized_blocks = bc.summarize_blocks(blocks)
                assert summarized_blocks.shape == (
                    len(blocks),
                    blocks[0].shape[1],
                )

            @settings(deadline=None)
            @given(valid_method, valid_apply_pca, valid_pca)
            def test_random_seed(self, method, apply_pca_flag, pca):
                """
                Test if the function produces the same output for the same random seed, even when sub-clocks of unequal sizes are provided.
                """
                blocks = [np.random.rand(10, 2), np.random.rand(5, 2)]

                rng1 = 343
                bc1 = BlockCompressor(
                    method=method,
                    apply_pca_flag=apply_pca_flag,
                    pca=pca,
                    random_seed=rng1,
                )
                summarized_blocks1 = bc1.summarize_blocks(blocks)

                rng2 = 343
                bc2 = BlockCompressor(
                    method=method,
                    apply_pca_flag=apply_pca_flag,
                    pca=pca,
                    random_seed=rng2,
                )
                summarized_blocks2 = bc2.summarize_blocks(blocks)

                np.testing.assert_array_equal(
                    summarized_blocks1, summarized_blocks2
                )

            @settings(deadline=None)
            @given(
                st.lists(
                    st.integers(min_value=1, max_value=10),
                    min_size=1,
                    max_size=10,
                ),
                valid_method,
                valid_apply_pca,
                valid_pca,
                rng_generator,
            )
            def test_input_list_various_sizes(
                self, input_list, method, apply_pca_flag, pca, random_seed
            ):
                """
                Test if the function can handle blocks of various sizes correctly.
                """
                blocks = [np.random.rand(size, 2) for size in input_list]
                bc = BlockCompressor(
                    method=method,
                    apply_pca_flag=apply_pca_flag,
                    pca=pca,
                    random_seed=random_seed,
                )
                summarized_blocks = bc.summarize_blocks(blocks)
                assert summarized_blocks.shape == (
                    len(blocks),
                    blocks[0].shape[1],
                )

            @settings(deadline=None)
            @given(valid_method, valid_apply_pca, valid_pca, rng_generator)
            def test_output_values_range(
                self, method, apply_pca_flag, pca, random_seed
            ):
                """
                Test if the output values are in the expected range (between 0 and 1) when the input values are in this range.
                """
                blocks = [np.random.rand(10, 2) for _ in range(3)]
                bc = BlockCompressor(
                    method=method,
                    apply_pca_flag=apply_pca_flag,
                    pca=pca,
                    random_seed=random_seed,
                )
                summarized_blocks = bc.summarize_blocks(blocks)
                if not apply_pca_flag:
                    assert np.min(summarized_blocks) >= 0, print(
                        summarized_blocks
                    )
                    assert np.max(summarized_blocks) <= 1, print(
                        summarized_blocks
                    )

            @settings(deadline=None)
            @given(st.sampled_from(["first", "middle", "last"]), rng_generator)
            def test_output_values_first_last_middle(
                self, method, random_seed
            ):
                """
                Test if the output values have an expected median close to 0.5 when the input values are uniformly distributed between 0 and 1.
                """
                blocks = [np.random.rand(1000, 20) for _ in range(3)]
                bc = BlockCompressor(
                    method=method,
                    apply_pca_flag=False,
                    pca=None,
                    random_seed=random_seed,
                )
                summarized_blocks = bc.summarize_blocks(blocks)
                for i in range(len(summarized_blocks)):
                    if method == "first":
                        np.testing.assert_array_equal(
                            summarized_blocks[i], blocks[i][0]
                        )
                    elif method == "middle":
                        np.testing.assert_array_equal(
                            summarized_blocks[i],
                            blocks[i][blocks[i].shape[0] // 2],
                        )
                    elif method == "last":
                        np.testing.assert_array_equal(
                            summarized_blocks[i], blocks[i][-1]
                        )

            @settings(deadline=None)
            @given(
                st.sampled_from(["mean", "median"]),
                valid_apply_pca,
                rng_generator,
            )
            def test_output_values_mean_median(
                self, method, apply_pca_flag, random_seed
            ):
                """
                Test if the output values have an expected mean/median close to 0.5 when the input values are uniformly distributed between 0 and 1.
                """
                blocks = [np.random.rand(1000, 20) for _ in range(3)]
                bc = BlockCompressor(
                    method=method,
                    apply_pca_flag=apply_pca_flag,
                    pca=None,
                    random_seed=random_seed,
                )
                summarized_blocks = bc.summarize_blocks(blocks)
                print(summarized_blocks)
                expected_output = (
                    approx(0.0, abs=0.05)
                    if apply_pca_flag
                    else approx(0.5, abs=0.05)
                )
                for i in range(len(summarized_blocks)):
                    if method == "mean":
                        assert np.mean(summarized_blocks[i]) == expected_output
                    elif method == "median":
                        assert (
                            np.median(summarized_blocks[i]) == expected_output
                        )

            @settings(deadline=None)
            @given(valid_apply_pca, rng_generator)
            def test_output_values_mode(self, apply_pca_flag, random_seed):
                """
                Test if the output values have an expected mode close to 0.5 when the input values are deterministic.
                """
                blocks = [np.random.rand(1000, 2) for _ in range(3)]

                bc = BlockCompressor(
                    method="mode",
                    apply_pca_flag=apply_pca_flag,
                    pca=None,
                    random_seed=random_seed,
                )
                summarized_blocks = bc.summarize_blocks(blocks)

                for i in range(len(summarized_blocks)):
                    if not apply_pca_flag:
                        block_mode = scipy.stats.mode(
                            blocks[i], keepdims=True
                        )[0][0]
                        np.testing.assert_array_almost_equal(
                            block_mode, summarized_blocks[i]
                        )
                    else:
                        pass

        class TestFailingCases:
            def test_empty_blocks(self):
                """
                Test if the function raises a ValueError when an empty list of blocks is provided.
                """
                bc = BlockCompressor()
                blocks = []
                with pytest.raises(ValueError):
                    bc.summarize_blocks(blocks)

            def test_nan_inf_values(self):
                """
                Test if the function raises a ValueError when NaN or Inf values are included in the blocks.
                """
                bc = BlockCompressor()
                blocks = [np.array([np.nan, np.inf, -np.inf]).reshape(-1, 1)]
                with pytest.raises(ValueError):
                    bc.summarize_blocks(blocks)

            def test_empty_sub_block(self):
                """
                Test if the function raises a ValueError when an empty sub-block is provided.
                """
                bc = BlockCompressor()
                blocks = [np.random.rand(10, 2), np.array([])]
                with pytest.raises(ValueError):
                    bc.summarize_blocks(blocks)

            def test_non_2d_sub_block(self):
                """
                Test if the function raises a ValueError when a non-2D sub-block is provided.
                """
                bc = BlockCompressor()
                blocks = [np.random.rand(10, 2), np.random.rand(10)]
                with pytest.raises(ValueError):
                    bc.summarize_blocks(blocks)


# Prepare strategies to generate valid and invalid inputs
valid_bools = st.booleans()
invalid_bools = st.one_of(st.integers(), st.floats(), st.text())

valid_pcas = st.one_of(st.none(), st.builds(PCA, n_components=st.just(1)))
invalid_pcas = st.builds(
    PCA, n_components=st.integers(min_value=2, max_value=100)
)

valid_ints = st.integers(min_value=1, max_value=1000)
invalid_ints = st.one_of(st.none(), st.floats(), st.text())

valid_random_seed = st.one_of(
    st.none(), st.integers(min_value=0, max_value=2**32 - 1)
)
invalid_random_seed = st.one_of(
    st.integers(max_value=-1),
    st.integers(min_value=2**32),
    st.floats(),
    st.text(),
)


@st.composite
def valid_transmat(draw, min_rows=2, max_rows=2):
    # Set a uniform row size for all the rows of the transition matrix
    row_size = draw(st.integers(min_rows, max_rows))

    row_strategy = st.lists(
        st.floats(
            min_value=0, max_value=1, allow_nan=False, allow_infinity=False
        ),
        min_size=row_size,
        max_size=row_size,
    )
    transmat = draw(
        st.lists(row_strategy, min_size=row_size, max_size=row_size)
    )

    return transmat


@st.composite
def invalid_transmat(draw):
    elements = st.floats(
        min_value=0, max_value=1, allow_nan=False, allow_infinity=False
    )
    # generate a transition matrix with either 1 state or 3 states
    length = draw(st.sampled_from([1, 3]))

    return draw(
        st.lists(
            st.lists(elements, min_size=length, max_size=length).filter(
                lambda row: not np.isclose(np.sum(row), 1)
            ),
            min_size=length,
            max_size=length,
        )
    )


@st.composite
def valid_means(draw):
    elements = st.floats(allow_nan=False, allow_infinity=False)
    # generate either a list of length 1 or a list of length 3 to make it invalid for a HMM with 2 states
    length = draw(st.just(2))

    # Each inner list should have a length different from the number of features in the data
    # If the number of features is 2, we can make the inner list length 1 or 3
    inner_length = draw(st.just(1))

    return draw(
        st.lists(
            st.lists(elements, min_size=inner_length, max_size=inner_length),
            min_size=length,
            max_size=length,
        )
    )


@st.composite
def invalid_means(draw):
    elements = st.floats(allow_nan=False, allow_infinity=False)
    # generate either a list of length 1 or a list of length 3 to make it invalid for a HMM with 2 states
    length = draw(st.sampled_from([1, 3]))

    # Each inner list should have a length different from the number of features in the data
    # If the number of features is 2, we can make the inner list length 1 or 3
    inner_length = draw(st.sampled_from([1, 3]))

    return draw(
        st.lists(
            st.lists(elements, min_size=inner_length, max_size=inner_length),
            min_size=length,
            max_size=length,
        )
    )


valid_test_data_np_array = [
    # Test with random 2D data, n_states=2, n_iter_hmm=100, n_fits_hmm=10
    (np.random.rand(20, 2), 2, 100, 10),
    # Test with increasing 2D data, n_states=2, n_iter_hmm=100, n_fits_hmm=10
    # TODO: figure out why this test fails on ubuntu
    # with size (10,), passes on macos but not ubuntu
    (np.array([[i, i] for i in range(20)]), 2, 100, 10),
    # Test with parabolic 2D data, n_states=3, n_iter_hmm=200, n_fits_hmm=20
    (np.array([[i, i**2] for i in range(10)]), 3, 200, 20),
    # Test with decreasing 2D data, n_states=1, n_iter_hmm=50, n_fits_hmm=5
    (np.array([[i, -i] for i in range(5)]), 1, 50, 5),
    # Test with increasing 2D data, double slope, n_states=3, n_iter_hmm=300, n_fits_hmm=30
    (np.array([[i, 2 * i] for i in range(20)]), 3, 300, 30),
    # Test with larger random 2D data, n_states=5, n_iter_hmm=100, n_fits_hmm=10
    (np.random.rand(100, 2), 5, 100, 10),
    # Test with very large random 2D data, n_states=2, n_iter_hmm=1000, n_fits_hmm=100
    (np.random.rand(100, 2), 2, 1000, 100),
    # Test with cubic 2D data, n_states=4, n_iter_hmm=200, n_fits_hmm=20
    (np.array([[i, i**3] for i in range(20)]), 4, 200, 20),
    # Test with decreasing parabolic 2D data, n_states=3, n_iter_hmm=150, n_fits_hmm=15
    (np.array([[i, -(i**2)] for i in range(10)]), 3, 150, 15),
]

invalid_test_data_np_array = [
    # Test with 1D data
    (
        np.random.rand(
            10,
        ),
        1,
        100,
        10,
    ),
    # Test with n_states=0
    (np.array([[-1, 1], [2, -2], [3, 3], [4, -4], [5, 5]]), 0, 100, 10),
    # Test with negative n_iter_hmm
    (np.array([[-1, 1], [2, -2], [3, 3], [4, -4], [5, 5]]), 2, -100, 10),
    # Test with negative n_fits_hmm
    (np.array([[-1, 1], [2, -2], [3, 3], [4, -4], [5, 5]]), 2, 100, -10),
    # Test with not enough data points
    (np.array([[-1, 1], [2, -2], [3, 3]]), 5, 100, 10),
    # Test with empty data
    (np.array([[]]), 1, 100, 10),
    # Test with non-integer n_states
    (np.array([[i, i] for i in range(5)]), "a", 100, 10),
    # Test with non-integer n_iter_hmm
    (np.array([[i, i] for i in range(5)]), 2, "b", 10),
    # Test with non-integer n_fits_hmm
    (np.array([[i, i] for i in range(5)]), 2, 100, "c"),
    # Test with non-integer n_fits_hmm
    (np.array([[i, i] for i in range(5)]), 2, 100, 10.5),
]


valid_test_data_list = [
    # Test with list of random 2D arrays, n_states=2, n_iter_hmm=100, n_fits_hmm=10
    ([np.random.rand(i + 1, 2) for i in range(10)], 2, 100, 10),
    # Test with list of increasing 2D arrays, n_states=2, n_iter_hmm=100, n_fits_hmm=10
    # TODO: figure out why this test fails on ubuntu
    # with size (5,), passes on macos but not ubuntu
    (
        [np.array([[i, i] for i in range(j + 1)]) for j in range(10)],
        2,
        100,
        10,
    ),
    # Test with list of parabolic 2D arrays, n_states=3, n_iter_hmm=200, n_fits_hmm=20
    (
        [np.array([[i, i**2] for i in range(j + 1)]) for j in range(10)],
        3,
        200,
        20,
    ),
    # Test with list of decreasing 2D arrays, n_states=1, n_iter_hmm=50, n_fits_hmm=5
    ([np.array([[i, -i] for i in range(j + 1)]) for j in range(5)], 1, 50, 5),
    # Test with list of increasing 2D arrays, double slope, n_states=3, n_iter_hmm=300, n_fits_hmm=30
    (
        [np.array([[i, 2 * i] for i in range(j + 1)]) for j in range(20)],
        3,
        300,
        30,
    ),
    # Test with list of larger random 2D arrays, n_states=5, n_iter_hmm=100, n_fits_hmm=10
    ([np.random.rand(i + 1, 2) for i in range(20)], 3, 100, 10),
    # Test with list of very large random 2D arrays, n_states=2, n_iter_hmm=1000, n_fits_hmm=100
    ([np.random.rand(i + 1, 2) for i in range(20)], 2, 100, 100),
    # Test with list of cubic 2D arrays, n_states=4, n_iter_hmm=200, n_fits_hmm=20
    (
        [np.array([[i, i**3] for i in range(j + 1)]) for j in range(20)],
        3,
        200,
        20,
    ),
    # Test with list of increasing 2D arrays, triple slope, n_states=4, n_iter_hmm=400, n_fits_hmm=40
    (
        [np.array([[i, 3 * i] for i in range(j + 1)]) for j in range(20)],
        3,
        400,
        40,
    ),
    # Test with list of decreasing parabolic 2D arrays, n_states=3, n_iter_hmm=150, n_fits_hmm=15
    (
        [np.array([[i, -(i**2)] for i in range(j + 1)]) for j in range(10)],
        3,
        150,
        15,
    ),
]


invalid_test_data_list = [
    # Test with 1D data
    ([np.array([[1]]) for _ in range(5)], 1, 100, 10),
    # Test with n_states=0
    (
        [
            np.array([[-1, 1], [2, -2], [3, 3], [4, -4], [5, 5]])
            for _ in range(5)
        ],
        0,
        100,
        10,
    ),
    # Test with negative n_iter_hmm
    (
        [
            np.array([[-1, 1], [2, -2], [3, 3], [4, -4], [5, 5]])
            for _ in range(5)
        ],
        2,
        -100,
        10,
    ),
    # Test with negative n_fits_hmm
    (
        [
            np.array([[-1, 1], [2, -2], [3, 3], [4, -4], [5, 5]])
            for _ in range(5)
        ],
        2,
        100,
        -10,
    ),
    # Test with not enough data points
    ([np.array([[-1, 1], [2, -2], [3, 3]]) for _ in range(5)], 3, 100, 10),
    # Test with empty data
    ([np.array([[]]) for _ in range(5)], 1, 100, 10),
    # Test with non-integer n_states
    ([np.array([[i, i] for i in range(5)]) for _ in range(5)], "a", 100, 10),
    # Test with non-integer n_iter_hmm
    ([np.array([[i, i] for i in range(5)]) for _ in range(5)], 2, "b", 10),
    # Test with non-integer n_fits_hmm
    ([np.array([[i, i] for i in range(5)]) for _ in range(5)], 2, 100, "c"),
    # Test with non-integer n_fits_hmm
    ([np.array([[i, i] for i in range(5)]) for _ in range(5)], 2, 100, 10.5),
]


@pytest.mark.skipif(
    not _check_soft_dependencies("hmmlearn", severity="none"),
    reason="skip test if required soft dependency not available",
)
class TestMarkovSampler:
    class TestInitAndGettersAndSetters:
        class TestPassingCases:
            @given(valid_bools)
            def test_apply_pca_setter_valid(self, value: bool):
                """Test that the apply_pca_flag setter accepts valid inputs."""
                ms = MarkovSampler()
                ms.apply_pca_flag = value
                assert ms.apply_pca_flag == value

            @given(valid_pcas)
            def test_pca_setter_valid(self, value: PCA):
                """Test that the pca setter accepts valid inputs."""
                ms = MarkovSampler()
                ms.pca = value
                assert ms.pca == value

            @given(valid_ints)
            def test_n_iter_hmm_setter_valid(self, value: int):
                """Test that the n_iter_hmm setter accepts valid inputs."""
                ms = MarkovSampler()
                ms.n_iter_hmm = value
                assert ms.n_iter_hmm == value

            @given(valid_ints)
            def test_n_fits_hmm_setter_valid(self, value: int):
                """Test that the n_fits_hmm setter accepts valid inputs."""
                ms = MarkovSampler()
                ms.n_fits_hmm = value
                assert ms.n_fits_hmm == value

            @given(valid_random_seed)
            def test_random_seed_setter_valid(self, value: int):
                """Test that the random_seed setter accepts valid inputs."""
                ms = MarkovSampler()
                ms.random_seed = value
                assert ms.random_seed == value

        class TestFailingCases:
            @given(invalid_ints)
            def test_n_iter_hmm_setter_invalid(self, value: Any):
                """Test that the n_iter_hmm setter rejects invalid inputs."""
                ms = MarkovSampler()
                with pytest.raises(TypeError):
                    ms.n_iter_hmm = value

            @given(invalid_ints)
            def test_n_fits_hmm_setter_invalid(self, value: Any):
                """Test that the n_fits_hmm setter rejects invalid inputs."""
                ms = MarkovSampler()
                with pytest.raises(TypeError):
                    ms.n_fits_hmm = value

            @given(invalid_random_seed)
            def test_random_seed_setter_invalid(self, value: Any):
                """Test that the random_seed setter rejects invalid inputs."""
                ms = MarkovSampler()
                with pytest.raises((TypeError, ValueError)):
                    ms.random_seed = value

    class TestFitHiddenMarkovModel:
        class TestPassingCases:
            @pytest.mark.parametrize(
                "X, n_states, n_iter_hmm, n_fits_hmm", valid_test_data_np_array
            )
            def test_fit_hidden_markov_model(
                self, X, n_states, n_iter_hmm, n_fits_hmm
            ):
                """
                Test fit_hidden_markov_model with various 2D data, n_states, n_iter_hmm, and n_fits_hmm.

                The test asserts that the returned model is an instance of hmm.GaussianHMM and the number of states matches the input.
                """
                from hmmlearn import hmm

                model = MarkovSampler(
                    n_iter_hmm=n_iter_hmm, n_fits_hmm=n_fits_hmm
                ).fit_hidden_markov_model(X, n_states)
                assert isinstance(model, hmm.GaussianHMM)
                assert model.n_components == n_states

            @settings(deadline=None)
            @given(st.data())
            def test_fit_hidden_markov_model_with_transmat_means_init(
                self, data
            ):
                from hmmlearn import hmm

                X = np.random.rand(50, 1)
                n_states = 2
                transmat_init = data.draw(valid_transmat())
                means_init = data.draw(valid_means())
                ms = MarkovSampler(n_iter_hmm=100, n_fits_hmm=10)
                model = ms.fit_hidden_markov_model(
                    X, n_states, transmat_init, means_init
                )
                assert isinstance(model, hmm.GaussianHMM)

        class TestFailingCases:
            @pytest.mark.parametrize(
                "X, n_states, n_iter_hmm, n_fits_hmm",
                invalid_test_data_np_array,
            )
            def test_fit_hidden_markov_model(
                self, X, n_states, n_iter_hmm, n_fits_hmm
            ):
                """
                Test fit_hidden_markov_model with various invalid inputs.

                The test asserts that the function raises an exception.
                """
                if (
                    not isinstance(n_iter_hmm, Integral)
                    or n_iter_hmm < 1
                    or not isinstance(n_fits_hmm, Integral)
                    or n_fits_hmm < 1
                ):
                    with pytest.raises((ValueError, TypeError)):
                        ms = MarkovSampler(
                            n_iter_hmm=n_iter_hmm, n_fits_hmm=n_fits_hmm
                        )
                else:
                    ms = MarkovSampler(
                        n_iter_hmm=n_iter_hmm, n_fits_hmm=n_fits_hmm
                    )
                with pytest.raises((Exception, ValueError, TypeError)):
                    ms.fit_hidden_markov_model(X, n_states)

            @given(st.data())
            def test_fit_hidden_markov_model_with_invalid_transmat_init(
                self, data
            ):
                X = np.random.rand(50, 2)
                n_states = 2
                transmat_init = data.draw(invalid_transmat())
                means_init = np.random.rand(2, 2)
                ms = MarkovSampler(n_iter_hmm=100, n_fits_hmm=10)
                with pytest.raises(ValueError):
                    ms.fit_hidden_markov_model(
                        X, n_states, transmat_init, means_init
                    )

            @given(st.data())
            def test_fit_hidden_markov_model_with_invalid_means_init(
                self, data
            ):
                X = np.random.rand(50, 2)
                n_states = 2
                transmat_init = np.array([[0.7, 0.3], [0.3, 0.7]])
                means_init = data.draw(invalid_means())
                ms = MarkovSampler(n_iter_hmm=100, n_fits_hmm=10)
                with pytest.raises(ValueError):
                    ms.fit_hidden_markov_model(
                        X, n_states, transmat_init, means_init
                    )

    class TestSample:
        class TestPassingCases:
            @pytest.mark.parametrize(
                "blocks, n_states, n_iter_hmm, n_fits_hmm",
                valid_test_data_list,
            )
            def test_sample_with_list_blocks_passing_blocks_as_hidden_states_flag_false(
                self, blocks, n_states, n_iter_hmm, n_fits_hmm
            ):
                """
                Test `sample` method with a list of blocks for positive cases.
                """
                ms = MarkovSampler(
                    blocks_as_hidden_states_flag=False,
                    random_seed=0,
                    n_iter_hmm=n_iter_hmm,
                    n_fits_hmm=n_fits_hmm,
                )

                total_rows = sum([block.shape[0] for block in blocks])
                ms.fit(blocks, n_states=n_states)
                obs, states = ms.sample()
                assert obs.shape == (total_rows, blocks[0].shape[1])
                assert states.shape == (total_rows,)

            @pytest.mark.skipif(
                not _check_soft_dependencies("dtaidistance", severity="none"),
                reason="skip test if required soft dependency not available",
            )
            @pytest.mark.parametrize(
                "blocks, n_states, n_iter_hmm, n_fits_hmm",
                valid_test_data_list,
            )
            def test_sample_with_list_blocks_passing_blocks_as_hidden_states_flag_true(
                self, blocks, n_states, n_iter_hmm, n_fits_hmm
            ):
                """
                Test `sample` method with a list of blocks for positive cases.
                """
                ms = MarkovSampler(
                    blocks_as_hidden_states_flag=True,
                    random_seed=0,
                    n_iter_hmm=n_iter_hmm,
                    n_fits_hmm=n_fits_hmm,
                )

                total_rows = sum([block.shape[0] for block in blocks])
                lengths = np.array([len(block) for block in blocks])
                if min(lengths) < 10:
                    with pytest.raises(ValueError):
                        ms.fit(blocks, n_states=n_states)
                        # obs, states = ms.sample(blocks, n_states=n_states)
                else:
                    ms.fit(blocks, n_states=n_states)
                    obs, states = ms.sample()
                    assert obs.shape == (total_rows, blocks[0].shape[1])
                    assert states.shape == (total_rows,)

            @pytest.mark.parametrize(
                "blocks, n_states, n_iter_hmm, n_fits_hmm",
                valid_test_data_np_array,
            )
            def test_sample_with_np_array_blocks_passing(
                self, blocks, n_states, n_iter_hmm, n_fits_hmm
            ):
                """
                Test `sample` method with a 2D NumPy array blocks for positive cases.
                """
                ms = MarkovSampler(
                    blocks_as_hidden_states_flag=False,
                    random_seed=0,
                    n_iter_hmm=n_iter_hmm,
                    n_fits_hmm=n_fits_hmm,
                )

                ms.fit(blocks, n_states=n_states)
                obs, states = ms.sample()

                assert obs.shape == (blocks.shape[0], blocks.shape[1])
                assert states.shape == (blocks.shape[0],)

        class TestFailingCases:
            @pytest.mark.parametrize(
                "blocks, n_states, n_iter_hmm, n_fits_hmm",
                invalid_test_data_np_array,
            )
            def test_sample_with_np_array_blocks_failing(
                self, blocks, n_states, n_iter_hmm, n_fits_hmm
            ):
                """
                Test `sample` method with a 2D NumPy array blocks for positive cases.
                """
                try:
                    ms = MarkovSampler(
                        blocks_as_hidden_states_flag=False,
                        random_seed=0,
                        n_iter_hmm=n_iter_hmm,
                        n_fits_hmm=n_fits_hmm,
                    )
                    ms.fit(blocks, n_states=n_states)
                    ms.sample()
                except ValueError:
                    pass
                except TypeError:
                    pass
                else:
                    pytest.fail(
                        "Expected ValueError or TypeError, but got no exception"
                    )

            @pytest.mark.parametrize(
                "blocks, n_states, n_iter_hmm, n_fits_hmm",
                invalid_test_data_list,
            )
            def test_sample_with_list_blocks_failing(
                self, blocks, n_states, n_iter_hmm, n_fits_hmm
            ):
                """
                Test `sample` method with a 2D NumPy array blocks for positive cases.
                """
                try:
                    ms = MarkovSampler(
                        blocks_as_hidden_states_flag=False,
                        random_seed=0,
                        n_iter_hmm=n_iter_hmm,
                        n_fits_hmm=n_fits_hmm,
                    )
                    ms.fit(blocks, n_states=n_states)
                    ms.sample()
                except ValueError:
                    pass
                except TypeError:
                    pass
                else:
                    pytest.fail(
                        "Expected ValueError or TypeError, but got no exception"
                    )
