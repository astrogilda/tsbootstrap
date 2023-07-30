from hypothesis import given, settings, strategies as st
import pytest
import numpy as np
from src.markov_sampler import MarkovSampler
from hmmlearn import hmm


# Test for calculate_transition_probs function
class TestCalculateTransitionProbs:
    class TestPassingCases:
        @settings(deadline=None)
        @given(st.lists(st.integers(min_value=0, max_value=10), min_size=2, max_size=100), st.integers(min_value=2, max_value=11))
        def test_random_assignments_random_components(self, assignments, n_components):
            """
            Test calculate_transition_probs with random assignments and random n_components.
            """
            # Update the assignments to be within the correct range
            assignments = np.array(assignments) % n_components
            transition_probabilities = MarkovSampler.calculate_transition_probs(
                assignments, n_components)
            assert transition_probabilities.shape == (
                n_components, n_components)
            assert np.allclose(np.sum(transition_probabilities, axis=1), 1)

        def test_constant_assignments_4_components(self):
            """
            Test calculate_transition_probs with constant assignments and n_components=4.
            """
            assignments = np.array([0, 0, 0, 0])
            n_components = 4
            transition_probabilities = MarkovSampler.calculate_transition_probs(
                assignments, n_components)
            assert transition_probabilities.shape == (
                n_components, n_components)
            assert np.allclose(np.sum(transition_probabilities, axis=1), 1)

        def test_alternating_assignments_2_components(self):
            """
            Test calculate_transition_probs with alternating assignments and n_components=2.
            """
            assignments = np.array([0, 1, 0, 1, 0, 1])
            n_components = 2
            transition_probabilities = MarkovSampler.calculate_transition_probs(
                assignments, n_components)
            assert transition_probabilities.shape == (
                n_components, n_components)
            assert np.allclose(np.sum(transition_probabilities, axis=1), 1)

    class TestFailingCases:
        def test_single_assignment_1_component(self):
            """
            Test calculate_transition_probs with a single assignment and n_components=2.
            """
            assignments = np.array([1])
            n_components = 1
            with pytest.raises(ValueError):
                MarkovSampler.calculate_transition_probs(
                    assignments, n_components)

        def test_random_assignments_0_component(self):
            """
            Test calculate_transition_probs with random assignments and n_components=0.
            """
            assignments = np.array([0, 1, 2, 3, 4])
            n_components = 0
            with pytest.raises(ValueError):
                MarkovSampler.calculate_transition_probs(
                    assignments, n_components)

        def test_non_integer_n_components(self):
            """
            Test calculate_transition_probs with a non-integer n_components value.
            """
            assignments = np.array([0, 1, 2, 3, 4])
            n_components = 2.5
            with pytest.raises(ValueError):
                MarkovSampler.calculate_transition_probs(
                    assignments, n_components)

        def test_negative_n_components(self):
            """
            Test calculate_transition_probs with a negative n_components value.
            """
            assignments = np.array([0, 1, 2, 3, 4])
            n_components = -2
            with pytest.raises(ValueError):
                MarkovSampler.calculate_transition_probs(
                    assignments, n_components)

        def test_assignment_out_of_range(self):
            """
            Test calculate_transition_probs with an assignment value out of the valid range.
            """
            assignments = np.array([0, 1, 2, 3, 4, 11])
            n_components = 5
            with pytest.raises(ValueError):
                MarkovSampler.calculate_transition_probs(
                    assignments, n_components)


# Test for fit_hidden_markov_model function
class TestFitHiddenMarkovModel:
    class TestPassingCases:

        test_data = [
            (np.array([[-100, 50], [200, -150], [300, 400],
             [500, 600], [700, 800], [900, 1000]]), 2),
            (np.array([[-50, 10], [20, -30], [40, 50],
             [-60, 70], [80, -90], [100, 110]]), 3),
            (np.array([[1, 2], [3, 4], [5, 6], [7, 8], [
             9, 10], [11, 12], [13, 14], [15, 16]]), 4),
            (np.array([[10, 20], [-30, 40], [50, -60],
             [70, 80], [90, 100], [110, -120]]), 1),
            (np.array([[-1000, 2000], [3000, -4000],
             [5000, 6000], [7000, 8000], [9000, 10000]]), 2),
            (np.array([[250, -500], [750, 1000],
             [1250, 1500], [1750, 2000], [2250, 2500]]), 3),
            (np.array([[100, 200], [300, 400], [500, 600],
             [700, 800], [900, 1000], [1100, 1200]]), 4),
            (np.array([[-200, 400], [-600, 800],
             [1000, -1200], [1400, 1600], [1800, 2000]]), 5),
            (np.array([[2, 4], [6, 8], [10, 12],
             [14, 16], [18, 20], [22, 24]]), 1),
            (np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5],
             [6, 6], [7, 7], [8, 8], [9, 9], [10, 10]]), 3),
        ]

        @pytest.mark.parametrize("X, n_states", test_data)
        def test_random_data_random_states(self, X, n_states):
            """
            Test fit_hidden_markov_model with random data and random n_states.
            """
            model = MarkovSampler.fit_hidden_markov_model(
                X, n_states, n_fits=1)
            assert isinstance(model, hmm.GaussianHMM)
            assert model.n_components == n_states

        def test_single_2D_point_1_state(self):
            """
            Test fit_hidden_markov_model with a single 2D point and n_states=1.
            """
            X = np.array([[1, 2], [3, 4], [5, 6]])
            n_states = 1
            model = MarkovSampler.fit_hidden_markov_model(X, n_states)
            assert isinstance(model, hmm.GaussianHMM)
            assert model.n_components == n_states

    class TestFailingCases:
        def test_single_2D_point_2_states(self):
            """
            Test fit_hidden_markov_model with a single 2D point and n_states=2.
            """
            X = np.array([[1, 2]])
            n_states = 2
            with pytest.raises(ValueError):
                MarkovSampler.fit_hidden_markov_model(X, n_states)

        def test_no_data_1_state(self):
            """
            Test fit_hidden_markov_model with no data and n_states=1.
            """
            X = np.array([[]])
            n_states = 1
            with pytest.raises(ValueError):
                MarkovSampler.fit_hidden_markov_model(X, n_states)

        def test_random_data_0_state(self):
            """
            Testfit_hidden_markov_model with random data and n_states=0.
            """
            X = np.array([[1, 2], [3, 4], [5, 6]])
            n_states = 0
            with pytest.raises(ValueError):
                MarkovSampler.fit_hidden_markov_model(X, n_states)


class TestGetBlockElement:
    class TestPassingCases:
        def test_first_element(self):
            """
            Test get_block_element with block_index="first".
            """
            block = np.array([[1, 2], [3, 4], [5, 6]])
            block_index = "first"
            result = MarkovSampler.get_block_element(block, block_index)
            assert np.array_equal(result, np.array([1, 2]))

        def test_middle_element(self):
            """
            Test get_block_element with block_index="middle".
            """
            block = np.array([[1, 2], [3, 4], [5, 6]])
            block_index = "middle"
            result = MarkovSampler.get_block_element(block, block_index)
            assert np.array_equal(result, np.array([3, 4]))

        def test_last_element(self):
            """
            Test get_block_element with block_index="last".
            """
            block = np.array([[1, 2], [3, 4], [5, 6]])
            block_index = "last"
            result = MarkovSampler.get_block_element(block, block_index)
            assert np.array_equal(result, np.array([5, 6]))

        def test_single_row_first_element(self):
            """
            Test get_block_element with block_index="first" and single-row block.
            """
            block = np.array([[1, 2]])
            block_index = "first"
            result = MarkovSampler.get_block_element(block, block_index)
            assert np.array_equal(result, np.array([1, 2]))

        def test_single_row_middle_element(self):
            """
            Test get_block_element with block_index="middle" and single-row block.
            """
            block = np.array([[1, 2]])
            block_index = "middle"
            result = MarkovSampler.get_block_element(block, block_index)
            assert np.array_equal(result, np.array([1, 2]))

        def test_single_row_last_element(self):
            """
            Test get_block_element with block_index="last" and single-row block.
            """
            block = np.array([[1, 2]])
            block_index = "last"
            result = MarkovSampler.get_block_element(block, block_index)
            assert np.array_equal(result, np.array([1, 2]))

    class TestFailingCases:
        def test_invalid_block_index(self):
            """
            Test get_block_element with an invalid block_index.
            """
            block = np.array([[1, 2], [3, 4], [5, 6]])
            block_index = "invalid"
            with pytest.raises(ValueError):
                MarkovSampler.get_block_element(block, block_index)

        def test_empty_block(self):
            """
            Test get_block_element with an empty block.
            """
            block = np.array([])
            block_index = "middle"
            with pytest.raises(ValueError):
                MarkovSampler.get_block_element(block, block_index)

        def test_1d_block(self):
            """
            Test get_block_element with a 1D block.
            """
            block = np.array([1, 2, 3, 4, 5, 6])
            block_index = "middle"
            with pytest.raises(ValueError):
                MarkovSampler.get_block_element(block, block_index)


class TestGetClusterTransitionsCentersAssignments:

    class TestFailingCases:
        def test_invalid_clustering_method(self):
            """
            Test get_cluster_transitions_centers_assignments with an invalid clustering method.
            """
            blocks = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
            clustering_method = "invalid"
            with pytest.raises(ValueError):
                MarkovSampler.get_cluster_transitions_centers_assignments(
                    blocks, clustering_method)

        def test_not_enough_blocks(self):
            """
            Test get_cluster_transitions_centers_assignments with less blocks than n_components.
            """
            blocks = [np.array([[1, 2], [3, 4]])]
            n_components = 3
            with pytest.raises(ValueError):
                MarkovSampler.get_cluster_transitions_centers_assignments(
                    blocks, n_components=n_components)

    class TestPassingCases:
        @pytest.mark.parametrize("clustering_method", ["block", "random", "kmeans", "hmm"])
        def test_single_block(self, clustering_method):
            """
            Test get_cluster_transitions_centers_assignments with a single block.
            """
            blocks = [np.array([[1, 2], [3, 4]])]
            n_components = 1
            transition_probs, centers, assignments = MarkovSampler.get_cluster_transitions_centers_assignments(
                blocks, clustering_method, n_components=n_components)
            assert np.array_equal(transition_probs, np.array([[1]]))
            assert np.array_equal(centers, np.array([[3, 4]]))
            assert np.array_equal(assignments, np.array([0]))

        @pytest.mark.parametrize("clustering_method", ["block", "random", "kmeans"])
        def test_two_blocks(self, clustering_method):
            """
            Test get_cluster_transitions_centers_assignments with two blocks and non-HMM clustering methods.
            """
            blocks = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
            n_components = 2
            transition_probs, centers, assignments = MarkovSampler.get_cluster_transitions_centers_assignments(
                blocks, clustering_method, n_components=n_components)
            assert np.array_equal(transition_probs, np.array([[0, 1], [1, 0]]))
            assert len(centers) == n_components
            assert len(assignments) == len(blocks)

        @pytest.mark.parametrize("clustering_method", ["block", "random", "kmeans"])
        def test_three_blocks(self, clustering_method):
            """
            Test get_cluster_transitions_centers_assignments with three blocks and non-HMM clustering methods.
            """
            blocks = [np.array([[1, 2], [3, 4]]), np.array(
                [[5, 6], [7, 8]]), np.array([[9, 10], [11, 12]])]
            n_components = 3
            transition_probs, centers, assignments = MarkovSampler.get_cluster_transitions_centers_assignments(
                blocks, clustering_method, n_components=n_components)
            assert len(transition_probs) == n_components
            assert len(centers) == n_components
            assert len(assignments) == len(blocks)

        @pytest.mark.parametrize("block_index", ["first", "middle", "last"])
        def test_block_clustering_block_index(self, block_index):
            """
            Test get_cluster_transitions_centers_assignments with block clustering and different block index values.
            """
            blocks = [np.array([[1, 2], [3, 4], [5, 6]]),
                      np.array([[7, 8], [9, 10], [11, 12]])]
            clustering_method = "block"
            n_components = 2
            transition_probs, centers, assignments = MarkovSampler.get_cluster_transitions_centers_assignments(
                blocks, clustering_method, block_index=block_index, n_components=n_components)
            assert len(transition_probs) == n_components
            assert len(centers) == n_components
            assert len(assignments) == len(blocks)

        def test_random_clustering_single_component(self):
            """
            Test get_cluster_transitions_centers_assignments with random clustering and a single component.
            """
            blocks = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
            clustering_method = "random"
            n_components = 1
            transition_probs, centers, assignments = MarkovSampler.get_cluster_transitions_centers_assignments(
                blocks=blocks, clustering_method=clustering_method, n_components=n_components)
            assert np.array_equal(transition_probs, np.array([[1]]))
            assert len(centers) == n_components
            assert np.array_equal(assignments, np.array([0, 0]))

        @pytest.mark.parametrize("block_index", ["first", "middle", "last"])
        def test_kmeans_clustering_block_index(self, block_index):
            """
            Test get_cluster_transitions_centers_assignments with kmeans clustering and different block index values.
            """
            blocks = [np.array([[1, 2], [3, 4], [5, 6]]),
                      np.array([[7, 8], [9, 10], [11, 12]])]
            clustering_method = "kmeans"
            n_components = 2
            transition_probs, centers, assignments = MarkovSampler.get_cluster_transitions_centers_assignments(
                blocks, clustering_method, block_index=block_index, n_components=n_components)
            assert len(transition_probs) == n_components
            assert len(centers) == n_components
            assert len(assignments) == len(blocks)

        def test_hmm_clustering_single_component(self):
            """
            Test get_cluster_transitions_centers_assignments with HMM clustering and a single component.
            """
            blocks = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
            clustering_method = "hmm"
            n_components = 1
            transition_probs, centers, assignments = MarkovSampler.get_cluster_transitions_centers_assignments(
                blocks, clustering_method, n_components=n_components)
            assert np.array_equal(transition_probs, np.array([[1]]))
            assert len(centers) == n_components
            assert np.array_equal(assignments, np.array([0, 0, 1, 1]))

        def test_hmm_clustering_two_components(self):
            """
            Test get_cluster_transitions_centers_assignments with HMM clustering and two components.
            """
            blocks = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
            clustering_method = "hmm"
            n_components = 2
            transition_probs, centers, assignments = MarkovSampler.get_cluster_transitions_centers_assignments(
                blocks, clustering_method, n_components=n_components)
            assert len(transition_probs) == n_components
            assert len(centers) == n_components
            assert len(assignments) == len(blocks) * 2

        def test_hmm_clustering_three_components(self):
            """
            Test get_cluster_transitions_centers_assignments with HMM clustering and three components.
            """
            blocks = [np.array([[1, 2], [3, 4]]), np.array(
                [[5, 6], [7, 8]]), np.array([[9, 10], [11, 12]])]
            clustering_method = "hmm"
            n_components = 3
            transition_probs, centers, assignments = MarkovSampler.get_cluster_transitions_centers_assignments(
                blocks, clustering_method, n_components=n_components)
            assert len(transition_probs) == n_components
            assert len(centers) == n_components
            assert len(assignments) == len(blocks) * 2

        def test_random_state_consistency(self):
            """
            Test get_cluster_transitions_centers_assignments with different random_state inputs to ensure consistency.
            """
            blocks = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
            clustering_method = "random"
            n_components = 2
            random_state = np.random.RandomState(42)
            transition_probs1, centers1, assignments1 = MarkovSampler.get_cluster_transitions_centers_assignments(
                blocks, clustering_method, n_components=n_components, random_state=random_state)
            random_state = np.random.RandomState(42)
            transition_probs2, centers2, assignments2 = MarkovSampler.get_cluster_transitions_centers_assignments(
                blocks, clustering_method, n_components=n_components, random_state=random_state)
            assert np.array_equal(transition_probs1, transition_probs2)
            assert np.array_equal(centers1, centers2)
            assert np.array_equal(assignments1, assignments2)
