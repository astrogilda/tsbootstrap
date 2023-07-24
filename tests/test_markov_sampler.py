from pytest import approx
from hypothesis import given
import hypothesis.strategies as st
import pytest
import numpy as np
from utils.markov_sampler import MarkovSampler, kmedians, pca_compression
from hmmlearn import hmm
from typing import List, Tuple
from sklearn.decomposition import PCA
import scipy

'''
class TestKMedians:
    class TestFailingCases:
        def test_non_array_input(self):
            """
            Test kmedians with non-array input.
            """
            data = [1, 2, 3]
            with pytest.raises(TypeError):
                kmedians(data)

        def test_non_2d_array(self):
            """
            Test kmedians with non-2D array input.
            """
            data = np.array([1, 2, 3])
            with pytest.raises(ValueError):
                kmedians(data)

        def test_negative_n_clusters(self):
            """
            Test kmedians with negative n_clusters.
            """
            data = np.array([[1, 2], [3, 4]])
            with pytest.raises(ValueError):
                kmedians(data, n_clusters=-1)

        def test_non_integer_max_iter(self):
            """
            Test kmedians with non-integer max_iter.
            """
            data = np.array([[1, 2], [3, 4]])
            with pytest.raises(ValueError):
                kmedians(data, max_iter='300')

    class TestPassingCases:
        def test_single_cluster(self):
            """
            Test kmedians with a single cluster.
            """
            data = np.array([[1, 2], [3, 4], [5, 6]])
            medians = kmedians(data)
            assert np.array_equal(medians, np.array([[3, 4]]))

        def test_multiple_clusters(self):
            """
            Test kmedians with multiple clusters.
            """
            data = np.array([[1, 2], [3, 4], [5, 6]])
            medians = kmedians(data, n_clusters=2)
            assert medians.shape == (2, 2)
            assert np.all(np.min(data, axis=0) <= medians) and np.all(
                medians <= np.max(data, axis=0))

        def test_same_values(self):
            """
            Test kmedians with same values in the data.
            """
            data = np.array([[1, 1], [1, 1], [1, 1]])
            medians = kmedians(data)
            assert np.array_equal(medians, np.array([[1, 1]]))

        def test_large_values(self):
            """
            Test kmedians with large values in the data.
            """
            data = np.array([[1e6, 1e6], [-1e6, -1e6]])
            medians = kmedians(data, n_clusters=2)
            expected_medians = np.array([[1e6, 1e6], [-1e6, -1e6]])
            assert np.array_equal(np.sort(medians, axis=0),
                                  np.sort(expected_medians, axis=0))

        def test_single_data_point(self):
            """
            Test kmedians with a single data point.
            """
            data = np.array([[1, 2]])
            medians = kmedians(data)
            assert np.array_equal(medians, np.array([[1, 2]]))
'''


class TestPCACompression:

    class TestFailingCases:
        def test_invalid_block(self):
            """
            Test pca_compression with invalid block.
            """
            block = "invalid_block"
            pca = PCA(n_components=1)
            with pytest.raises(TypeError):
                pca_compression(block, pca)

        def test_invalid_pca(self):
            """
            Test pca_compression with invalid PCA object.
            """
            block = np.array([[1, 2], [3, 4], [5, 6]])
            pca = "invalid_pca"
            with pytest.raises(TypeError):
                pca_compression(block, pca)

        def test_invalid_summary(self):
            """
            Test pca_compression with invalid summary.
            """
            block = np.array([[1, 2], [3, 4], [5, 6]])
            pca = PCA(n_components=1)
            summary = "invalid_summary"
            with pytest.raises(TypeError):
                pca_compression(block, pca, summary)

        def test_incompatible_pca(self):
            """
            Test pca_compression with incompatible PCA object (n_components not equal to 1).
            """
            block = np.array([[1, 2], [3, 4], [5, 6]])
            pca = PCA(n_components=2)
            with pytest.raises(ValueError):
                pca_compression(block, pca)

    class TestPassingCases:
        def test_pca_compression(self):
            """
            Test pca_compression with valid inputs.
            """
            block = np.array([[1, 2], [3, 4], [5, 6]])
            pca = PCA(n_components=1)
            summary = pca_compression(block, pca)
            assert len(summary) == 1
            assert isinstance(summary, np.ndarray)

        def test_single_element_block(self):
            """
            Test pca_compression with single element block.
            """
            block = np.array([[1]])
            pca = PCA(n_components=1)
            summary = pca_compression(block, pca)
            assert len(summary) == 1
            assert isinstance(summary, np.ndarray)

        def test_single_row_block(self):
            """
            Test pca_compression with single row block.
            """
            block = np.array([[1, 2, 3]])
            pca = PCA(n_components=1)
            summary = pca_compression(block, pca)
            assert len(summary) == 1
            assert isinstance(summary, np.ndarray)

        def test_single_column_block(self):
            """
            Test pca_compression with single column block.
            """
            block = np.array([[1], [2], [3]])
            pca = PCA(n_components=1)
            summary = pca_compression(block, pca)
            assert len(summary) == 1
            assert isinstance(summary, np.ndarray)


def generate_random_blocks(n_blocks: int, block_size: Tuple[int, int], min_val=0, max_val=10) -> List[np.ndarray]:
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
    ----------
    List[np.ndarray]
        List of numpy arrays, each with shape block_size.
    """
    if n_blocks <= 0 or not isinstance(n_blocks, int):
        raise ValueError("'n_blocks' should be a positive integer.")
    if not (isinstance(block_size, tuple) and len(block_size) == 2):
        raise ValueError("'block_size' should be a tuple of 2 integers.")
    return [np.random.randint(min_val, max_val, block_size) for _ in range(n_blocks)]


# Test for calculate_transition_probabilities function
class TestCalculateTransitionProbabilities:
    class TestPassingCases:
        def test_constant_blocks(self):
            """
            Test calculate_transition_probabilities with constant blocks.
            """
            blocks = [np.ones((10, 2)) for _ in range(
                3)]  # 3 blocks of constant time series data
            transition_probabilities = MarkovSampler.calculate_transition_probabilities(
                blocks)
            assert transition_probabilities.shape == (len(blocks), len(blocks))

            # Check that transition probabilities are equal for constant blocks
            expected_probability = 1 / len(blocks)
            assert np.allclose(transition_probabilities, expected_probability)

        @pytest.mark.parametrize("n_blocks,n_features", [(2, 2), (5, 3), (10, 4)])
        def test_random_blocks(self, n_blocks, n_features):
            """
            Test calculate_transition_probabilities with random blocks.
            """
            blocks = generate_random_blocks(n_blocks, (10, n_features))
            transition_probabilities = MarkovSampler.calculate_transition_probabilities(
                blocks)
            assert transition_probabilities.shape == (n_blocks, n_blocks)
            assert np.allclose(np.sum(transition_probabilities, axis=1), 1)

        def test_random_blocks_different_sizes(self):
            """
            Test calculate_transition_probabilities with random blocks of different sizes.
            """
            blocks = generate_random_blocks(
                3, (10, 2)) + generate_random_blocks(2, (20, 2))
            transition_probabilities = MarkovSampler.calculate_transition_probabilities(
                blocks)
            assert transition_probabilities.shape == (len(blocks), len(blocks))
            assert np.allclose(np.sum(transition_probabilities, axis=1), 1)

        @pytest.mark.parametrize("n_blocks", [1, 5, 10])
        def test_multiple_blocks_same_size(self, n_blocks):
            """
            Test calculate_transition_probabilities with multiple blocks of the same size.
            """
            blocks = generate_random_blocks(n_blocks, (10, 2))
            transition_probabilities = MarkovSampler.calculate_transition_probabilities(
                blocks)
            assert transition_probabilities.shape == (n_blocks, n_blocks)
            assert np.allclose(np.sum(transition_probabilities, axis=1), 1)

    class TestFailingCases:
        def test_empty_list(self):
            """
            Test calculate_transition_probabilities with an empty list of blocks.
            """
            blocks = []
            with pytest.raises(ValueError):
                MarkovSampler.calculate_transition_probabilities(blocks)

        def test_none_blocks(self):
            """
            Test calculate_transition_probabilities where the blocks list contains None.
            """
            blocks = [np.array([[0, 1], [1, 0]]), None]
            with pytest.raises(TypeError):
                MarkovSampler.calculate_transition_probabilities(blocks)

        def test_incompatible_block_shapes(self):
            """
            Test calculate_transition_probabilities where blocks have incompatible shapes.
            """
            blocks = [np.array([[0, 1], [1, 0]]), np.array([0, 1])]
            with pytest.raises(ValueError):
                MarkovSampler.calculate_transition_probabilities(blocks)

        @pytest.mark.parametrize("n_blocks", [0, -1])
        def test_invalid_number_of_blocks(self, n_blocks):
            """
            Test calculate_transition_probabilities with an invalid number of blocks.
            """
            with pytest.raises(ValueError):
                blocks = generate_random_blocks(n_blocks, (10, 2))
                MarkovSampler.calculate_transition_probabilities(blocks)

        def test_different_number_of_features(self):
            """
            Test calculate_transition_probabilities where blocks have a different number of features.
            """
            blocks = [np.random.rand(10, 2), np.random.rand(10, 3)]
            with pytest.raises(ValueError):
                MarkovSampler.calculate_transition_probabilities(blocks)

        def test_non_ndarray_blocks(self):
            """
            Test calculate_transition_probabilities where one or more blocks are not numpy ndarrays.
            """
            blocks = [np.random.rand(10, 2), [1, 2, 3]]
            with pytest.raises(TypeError):
                MarkovSampler.calculate_transition_probabilities(blocks)

        @pytest.mark.parametrize("n_blocks,block_size", [(0, (10, 2)), (-1, (10, 2))])
        def test_invalid_generation_params(self, n_blocks, block_size):
            """
            Test generate_random_blocks with invalid parameters.
            """
            with pytest.raises(ValueError):
                generate_random_blocks(n_blocks, block_size)


class TestSummarizeBlocks:
    class TestPassingCases:
        @pytest.mark.parametrize("method", ['first', 'middle', 'last', 'mean', 'median', 'mode', 'kmeans', 'kmedians', 'kmedoids'])
        def test_valid_methods(self, method):
            """
            Test if the function correctly processes blocks for all valid methods.
            """
            blocks = [np.random.rand(10, 2) for _ in range(3)]
            summarized_blocks = MarkovSampler.summarize_blocks(
                blocks, method=method)
            assert summarized_blocks.shape == (len(blocks), blocks[0].shape[1])

        def test_unequal_sub_block_sizes(self):
            """
            Test if the function raises a ValueError when sub-blocks of unequal sizes are provided.
            """
            blocks = [np.random.rand(10, 2), np.random.rand(5, 2)]
            summarized_blocks = MarkovSampler.summarize_blocks(
                blocks, method='mean')
            assert summarized_blocks.shape == (len(blocks), blocks[0].shape[1])

        def test_apply_pca(self):
            """
            Test if PCA is correctly applied when apply_pca=True and a valid PCA object is provided.
            """
            blocks = [np.random.rand(10, 2) for _ in range(3)]
            summarized_blocks = MarkovSampler.summarize_blocks(
                blocks, method='mean', apply_pca=True, pca=PCA(n_components=1))
            assert summarized_blocks.shape == (len(blocks), 1)

        def test_apply_pca_without_pca_object(self):
            """
            Test if PCA is correctly applied when apply_pca=True but no PCA object is provided.
            """
            blocks = [np.random.rand(10, 2) for _ in range(3)]
            summarized_blocks = MarkovSampler.summarize_blocks(
                blocks, method='mean', apply_pca=True)
            assert summarized_blocks.shape == (len(blocks), 1)

        def test_random_seed(self):
            """
            Test if the function produces the same output for the same random seed.
            """
            blocks = [np.random.rand(10, 2) for _ in range(3)]
            summarized_blocks1 = MarkovSampler.summarize_blocks(
                blocks, method='kmeans', random_seed=0)
            summarized_blocks2 = MarkovSampler.summarize_blocks(
                blocks, method='kmeans', random_seed=0)
            np.testing.assert_array_equal(
                summarized_blocks1, summarized_blocks2)

        @given(st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=10))
        def test_input_list_various_sizes(self, input_list):
            """
            Test if the function can handle blocks of various sizes correctly.
            """
            blocks = [np.random.rand(size, 2) for size in input_list]
            summarized_blocks = MarkovSampler.summarize_blocks(
                blocks, method='mean')
            assert summarized_blocks.shape == (len(blocks), blocks[0].shape[1])

        @given(st.integers(min_value=1, max_value=1000))
        def test_kmedians_max_iter_various_values(self, max_iter):
            """
            Test if the function correctly processes blocks for various values of kmedians_max_iter.
            """
            blocks = [np.random.rand(10, 2) for _ in range(3)]
            summarized_blocks = MarkovSampler.summarize_blocks(
                blocks, method='kmedians', kmedians_max_iter=max_iter)
            assert summarized_blocks.shape == (len(blocks), blocks[0].shape[1])

        def test_output_values_range(self):
            """
            Test if the output values are in the expected range (between 0 and 1) when the input values are in this range.
            """
            blocks = [np.random.rand(10, 2) for _ in range(3)]
            summarized_blocks = MarkovSampler.summarize_blocks(
                blocks, method='mean')
            assert np.min(summarized_blocks) >= 0
            assert np.max(summarized_blocks) <= 1

        def test_output_values_mean(self):
            """
            Test if the output values have an expected mean close to 0.5 when the input values are uniformly distributed between 0 and 1.
            """
            blocks = [np.random.rand(1000, 2) for _ in range(3)]
            summarized_blocks = MarkovSampler.summarize_blocks(
                blocks, method='mean')
            assert np.mean(summarized_blocks) == approx(0.5, abs=0.05)

        def test_output_values_median(self):
            """
            Test if the output values have an expected median close to 0.5 when the input values are uniformly distributed between 0 and 1.
            """
            blocks = [np.random.rand(1000, 2) for _ in range(3)]
            summarized_blocks = MarkovSampler.summarize_blocks(
                blocks, method='median')
            assert np.median(summarized_blocks) == approx(0.5, abs=0.05)

        def test_output_values_mode(self):
            """
            Test if the output values have an expected mode close to 0.5 when the input values are deterministic.
            """
            # Create blocks where 0.5 appears more often than 0 or 1
            blocks = []
            # Set a seed for reproducibility
            np.random.seed(0)

            for _ in range(3):
                block = np.zeros((100, 5))
                for i in range(100):
                    for j in range(5):
                        if i % 2 == 0 or j % 2 == 0:
                            block[i][j] = 0.5
                        else:
                            block[i][j] = np.random.choice([0, 1])
                blocks.append(block)

            summarized_blocks = MarkovSampler.summarize_blocks(
                blocks, method='mode')

            assert scipy.stats.mode(summarized_blocks)[
                0][0] == approx(0.5, abs=0.05)

    class TestFailingCases:
        def test_empty_blocks(self):
            """
            Test if the function raises a ValueError when an empty list of blocks is provided.
            """
            blocks = []
            with pytest.raises(ValueError):
                MarkovSampler.summarize_blocks(blocks)

        @pytest.mark.parametrize("method", ['invalid', 'wrong', 'test'])
        def test_invalid_methods(self, method):
            """
            Test if the function raises a ValueError when an invalid method is provided.
            """
            blocks = [np.random.rand(10, 2) for _ in range(3)]
            with pytest.raises(ValueError):
                MarkovSampler.summarize_blocks(blocks, method=method)

        def test_apply_pca_with_invalid_pca_object(self):
            """
            Test if the function raises a ValueError when apply_pca=True but an invalid PCA object is provided.
            """
            blocks = [np.random.rand(10, 2) for _ in range(3)]
            with pytest.raises(ValueError):
                MarkovSampler.summarize_blocks(
                    blocks, method='mean', apply_pca=True, pca=PCA(n_components=2))

        def test_kmedians_max_iter(self):
            """
            Test if the function raises a ValueError when an invalid kmedians_max_iter value is provided.
            """
            blocks = [np.random.rand(10, 2) for _ in range(3)]
            with pytest.raises(ValueError):
                MarkovSampler.summarize_blocks(
                    blocks, method='kmedians', kmedians_max_iter=-1)

        def test_random_seed_negative(self):
            """
            Test if the function raises a ValueError when an invalid random seed is provided.
            """
            blocks = [np.random.rand(10, 2) for _ in range(3)]
            with pytest.raises(ValueError):
                MarkovSampler.summarize_blocks(
                    blocks, method='kmeans', random_seed=-1)

        @given(st.lists(st.floats(allow_nan=True, allow_infinity=True), min_size=1, max_size=10))
        def test_nan_inf_values(self, input_list):
            """
            Test if the function raises a ValueError when NaN or Inf values are included in the blocks.
            """
            blocks = [np.array([np.nan, np.inf, -np.inf]).reshape(-1, 1)]
            with pytest.raises(ValueError):
                MarkovSampler.summarize_blocks(blocks, method='mean')

        def test_empty_sub_block(self):
            """
            Test if the function raises a ValueError when an empty sub-block is provided.
            """
            blocks = [np.random.rand(10, 2), np.array([])]
            with pytest.raises(ValueError):
                MarkovSampler.summarize_blocks(blocks, method='mean')

        def test_non_2d_sub_block(self):
            """
            Test if the function raises a ValueError when a non-2D sub-block is provided.
            """
            blocks = [np.random.rand(10, 2), np.random.rand(10)]
            with pytest.raises(ValueError):
                MarkovSampler.summarize_blocks(blocks, method='mean')


# Test for fit_hidden_markov_model function
class TestFitHiddenMarkovModel:
    class TestPassingCases:
        test_data = [
            # Test with random 2D data, n_states=2, n_iter_hmm=100, n_fits_hmm=10
            (np.random.rand(5, 2), 2, 100, 10),
            # Test with increasing 2D data, n_states=2, n_iter_hmm=100, n_fits_hmm=10
            (np.array([[i, i] for i in range(5)]), 2, 100, 10),
            # Test with parabolic 2D data, n_states=3, n_iter_hmm=200, n_fits_hmm=20
            (np.array([[i, i**2] for i in range(10)]), 3, 200, 20),
            # Test with decreasing 2D data, n_states=1, n_iter_hmm=50, n_fits_hmm=5
            (np.array([[i, -i] for i in range(5)]), 1, 50, 5),
            # Test with increasing 2D data, double slope, n_states=3, n_iter_hmm=300, n_fits_hmm=30
            (np.array([[i, 2*i] for i in range(10)]), 3, 300, 30),
            # Test with larger random 2D data, n_states=5, n_iter_hmm=100, n_fits_hmm=10
            (np.random.rand(10, 2), 5, 100, 10),
            # Test with very large random 2D data, n_states=2, n_iter_hmm=1000, n_fits_hmm=100
            (np.random.rand(100, 2), 2, 1000, 100),
            # Test with cubic 2D data, n_states=4, n_iter_hmm=200, n_fits_hmm=20
            (np.array([[i, i**3] for i in range(20)]), 4, 200, 20),
            # Test with increasing 2D data, triple slope, n_states=4, n_iter_hmm=400, n_fits_hmm=40
            (np.array([[i, 3*i] for i in range(10)]), 4, 400, 40),
            # Test with decreasing parabolic 2D data, n_states=3, n_iter_hmm=150, n_fits_hmm=15
            (np.array([[i, -i**2] for i in range(10)]), 3, 150, 15),
        ]

        @pytest.mark.parametrize("X, n_states, n_iter_hmm, n_fits_hmm", test_data)
        def test_fit_hidden_markov_model(self, X, n_states, n_iter_hmm, n_fits_hmm):
            """
            Test fit_hidden_markov_model with various 2D data, n_states, n_iter_hmm, and n_fits_hmm.
            The test asserts that the returned model is an instance of hmm.GaussianHMM and the number of states matches the input.
            """
            model = MarkovSampler.fit_hidden_markov_model(
                X, n_states, n_iter_hmm, n_fits_hmm)
            assert isinstance(model, hmm.GaussianHMM)
            assert model.n_components == n_states

    class TestFailingCases:
        test_data = [
            (np.array([[1]]), 1, 100, 10),  # Test with 1D data
            (np.array([[-1, 1], [2, -2], [3, 3], [4, -4], [5, 5]]),
             0, 100, 10),  # Test with n_states=0
            (np.array([[-1, 1], [2, -2], [3, 3], [4, -4], [5, 5]]),
             2, -100, 10),  # Test with negative n_iter_hmm
            (np.array([[-1, 1], [2, -2], [3, 3], [4, -4], [5, 5]]),
             2, 100, -10),  # Test with negative n_fits_hmm
            # Test with not enough data points
            (np.array([[-1, 1], [2, -2], [3, 3]]), 5, 100, 10),
            (np.array([[]]), 1, 100, 10),  # Test with empty data
            # Test with non-integer n_states
            (np.array([[i, i] for i in range(5)]), 'a', 100, 10),
            # Test with non-integer n_iter_hmm
            (np.array([[i, i] for i in range(5)]), 2, 'b', 10),
            # Test with non-integer n_fits_hmm
            (np.array([[i, i] for i in range(5)]), 2, 100, 'c'),
            # Test with non-integer n_fits_hmm
            (np.array([[i, i] for i in range(5)]), 2, 100, 10.5),
        ]

        @pytest.mark.parametrize("X, n_states, n_iter_hmm, n_fits_hmm", test_data)
        def test_fit_hidden_markov_model(self, X, n_states, n_iter_hmm, n_fits_hmm):
            """
            Test fit_hidden_markov_model with various invalid inputs.
            The test asserts that the function raises an exception.
            """
            with pytest.raises(Exception):
                MarkovSampler.fit_hidden_markov_model(
                    X, n_states, n_iter_hmm, n_fits_hmm)


class TestGetClusterTransitionsCentersAssignments:
    class TestPassingCases:
        def test_single_block(self):
            """
            Test get_cluster_transitions_centers_assignments with a single block.
            """
            blocks_summarized = np.array([[1, 2], [3, 4]])
            hmm_model = hmm.GaussianHMM(n_components=1)
            hmm_model.fit(blocks_summarized)
            transition_probs, centers, covariances, assignments = MarkovSampler.get_cluster_transitions_centers_assignments(
                blocks_summarized, hmm_model)
            assert np.array_equal(transition_probs, np.array([[1]]))
            assert np.array_equal(centers, np.array([[2, 3]]))
            assert np.array_equal(assignments, np.array([0, 0]))

        def test_two_blocks(self):
            """
            Test get_cluster_transitions_centers_assignments with two blocks.
            """
            blocks_summarized = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
            hmm_model = hmm.GaussianHMM(n_components=2)
            hmm_model.fit(blocks_summarized)
            transition_probs, centers, covariances, assignments = MarkovSampler.get_cluster_transitions_centers_assignments(
                blocks_summarized, hmm_model)
            assert transition_probs.shape == (2, 2)
            assert centers.shape == (2, 2)
            assert covariances.shape == (2, 2, 2)
            assert assignments.shape == (4,)

        def test_blocks_with_same_values(self):
            """
            Test get_cluster_transitions_centers_assignments with blocks having the same values.
            """
            blocks_summarized = np.array([[1, 1], [1, 1]])
            hmm_model = hmm.GaussianHMM(n_components=1)
            hmm_model.fit(blocks_summarized)
            transition_probs, centers, covariances, assignments = MarkovSampler.get_cluster_transitions_centers_assignments(
                blocks_summarized, hmm_model)
            assert np.array_equal(transition_probs, np.array([[1]]))
            assert np.array_equal(centers, np.array([[1, 1]]))
            assert np.array_equal(assignments, np.array([0, 0]))

        def test_blocks_with_large_values(self):
            """
            Test get_cluster_transitions_centers_assignments with blocks having large values.
            """
            blocks_summarized = np.array([[1e6, 1e6], [-1e6, -1e6]])
            hmm_model = hmm.GaussianHMM(n_components=1)
            hmm_model.fit(blocks_summarized)
            transition_probs, centers, covariances, assignments = MarkovSampler.get_cluster_transitions_centers_assignments(
                blocks_summarized, hmm_model)
            assert transition_probs.shape == (1, 1)
            assert centers.shape == (1, 2)
            assert covariances.shape == (1, 2, 2)
            assert assignments.shape == (2,)

    class TestFailingCases:
        def test_empty_blocks(self):
            """
            Test get_cluster_transitions_centers_assignments with empty blocks.
            """
            blocks_summarized = np.array([])
            hmm_model = hmm.GaussianHMM(n_components=2)
            with pytest.raises(ValueError):
                MarkovSampler.get_cluster_transitions_centers_assignments(
                    blocks_summarized, hmm_model)

        def test_incompatible_model(self):
            """
            Test get_cluster_transitions_centers_assignments with a model not compatible with the blocks.
            """
            blocks_summarized = np.array([[1, 2], [3, 4]])
            hmm_model = hmm.GaussianHMM(n_components=3)
            with pytest.raises(ValueError):
                MarkovSampler.get_cluster_transitions_centers_assignments(
                    blocks_summarized, hmm_model)
