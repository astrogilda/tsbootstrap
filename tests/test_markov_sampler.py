from hypothesis import given, settings, strategies as st
import pytest
import numpy as np
from utils.markov_sampler import MarkovSampler
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
            assignments = np.array(assignments)
            n_components = max(n_components, np.max(assignments) + 1)
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
            with pytest.raises(AssertionError):
                MarkovSampler.calculate_transition_probs(
                    assignments, n_components)

        def test_random_assignments_0_component(self):
            """
            Test calculate_transition_probs with random assignments and n_components=0.
            """
            assignments = np.array([0, 1, 2, 3, 4])
            n_components = 0
            with pytest.raises(AssertionError):
                MarkovSampler.calculate_transition_probs(
                    assignments, n_components)

        def test_non_integer_n_components(self):
            """
            Test calculate_transition_probs with a non-integer n_components value.
            """
            assignments = np.array([0, 1, 2, 3, 4])
            n_components = 2.5
            with pytest.raises(AssertionError):
                MarkovSampler.calculate_transition_probs(
                    assignments, n_components)

        def test_negative_n_components(self):
            """
            Test calculate_transition_probs with a negative n_components value.
            """
            assignments = np.array([0, 1, 2, 3, 4])
            n_components = -2
            with pytest.raises(AssertionError):
                MarkovSampler.calculate_transition_probs(
                    assignments, n_components)

        def test_assignment_out_of_range(self):
            """
            Test calculate_transition_probs with an assignment value out of the valid range.
            """
            assignments = np.array([0, 1, 2, 3, 4, 11])
            n_components = 5
            with pytest.raises(AssertionError):
                MarkovSampler.calculate_transition_probs(
                    assignments, n_components)


# Test for fit_hidden_markov_model function
class TestFitHiddenMarkovModel:
    class TestPassingCases:
        @settings(deadline=None)
        @given(st.lists(st.lists(st.floats(min_value=-1000, max_value=1000, allow_infinity=False, allow_nan=False), min_size=2, max_size=2), min_size=6, max_size=10), st.integers(min_value=1, max_value=6))
        def test_random_data_random_states(self, X, n_states):
            """
            Test fit_hidden_markov_model with random data and random n_states.
            """
            # Convert X to a 2D numpy array
            X = np.array(X)
            model = MarkovSampler.fit_hidden_markov_model(
                X, n_states, n_fits=1)
            assert isinstance(model, hmm.GaussianHMM)
            assert model.n_components == n_states

        def test_single_2D_point_1_state(self):
            """
            Test fit_hidden_markov_model with a single 2D point and n_states=1.
            """
            X = np.array([[1, 2]])
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
