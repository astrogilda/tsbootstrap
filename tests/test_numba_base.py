from hypothesis import given, settings, strategies as st
from hypothesis import strategies as st
from hypothesis.strategies import builds, floats, integers, just, tuples
from hypothesis.strategies import floats, builds
from hypothesis.extra.numpy import arrays
import pytest
from hypothesis import given, strategies as st, settings
import numpy as np
from utils.numba_base import choice_with_p, time_series_split, har_cov, calculate_transition_probs, fit_hidden_markov_model
from hmmlearn import hmm


class TestChoiceWithP:
    class TestPassingCases:
        @settings(deadline=None)
        @given(st.lists(st.floats(min_value=0, max_value=1), min_size=2, max_size=100))
        def test_valid_weights(self, weights):
            """
            Test that the function works for valid weights and returns an array of indices
            with the same length as the input weights.
            """
            weights = np.array(weights)
            indices = choice_with_p(weights)
            assert len(indices) == len(weights)
            assert np.all(indices >= 0)
            assert np.all(indices < len(weights))

        def test_equal_weights(self):
            """
            Test that the function works for equal weights and returns an array of indices
            with the same length as the input weights.
            """
            weights = np.array([0.25, 0.25, 0.25, 0.25])
            indices = choice_with_p(weights)
            assert len(indices) == len(weights)
            assert np.all(indices >= 0)
            assert np.all(indices < 4)

    class TestFailingCases:
        def test_negative_weights(self):
            """
            Test that the function raises a ValueError if any of the input weights are negative.
            """
            weights = np.array([-0.25, -0.25, -0.25, -0.25])
            with pytest.raises(ValueError):
                choice_with_p(weights)

        def test_weights_not_1d(self):
            """
            Test that the function raises a ValueError if the input weights are not a 1-dimensional array.
            """
            weights = np.array([[0.25, 0.25], [0.25, 0.25]])
            with pytest.raises(ValueError):
                choice_with_p(weights)


class TestTimeSeriesSplit:
    class TestPassingCases:
        @given(st.lists(st.floats(allow_infinity=False, allow_nan=False), min_size=2, max_size=100), st.floats(min_value=0.1, max_value=0.9))
        def test_valid_input(self, X, test_ratio):
            X = np.array(X)
            X_train, X_test = time_series_split(X, test_ratio)
            assert len(X_train) == int(len(X) * (1 - test_ratio))
            assert len(X_test) == len(X) - len(X_train)
            assert np.all(X_train == X[:len(X_train)])
            assert np.all(X_test == X[len(X_train):])

        def test_zero_ratio(self):
            X = np.array([1, 2, 3, 4, 5])
            X_train, X_test = time_series_split(X, 0)
            assert len(X_train) == 5
            assert len(X_test) == 0

        def test_full_ratio(self):
            X = np.array([1, 2, 3, 4, 5])
            X_train, X_test = time_series_split(X, 1)
            assert len(X_train) == 0
            assert len(X_test) == 5

    class TestFailingCases:
        def test_negative_ratio(self):
            X = np.array([1, 2, 3, 4, 5])
            with pytest.raises(ValueError):
                time_series_split(X, -0.5)

        def test_large_ratio(self):
            X = np.array([1, 2, 3, 4, 5])
            with pytest.raises(ValueError):
                time_series_split(X, 1.5)


class TestHarCov:
    """
    Test suite for the har_cov function.
    """
    class TestPassingCases:
        """
        Test cases where the har_cov function is expected to return a result.
        """
        @settings(deadline=None)
        @given(st.lists(st.lists(st.floats(min_value=0, max_value=1), min_size=2, max_size=2), min_size=2, max_size=10))
        def test_random_positive_values(self, X):
            """
            Test the har_cov function with random positive input values. 
            It verifies that the function returns an array of correct shape.
            """
            X = np.array(X)
            H = har_cov(X, 1)
            assert H.shape == (X.shape[1], X.shape[1])

        def test_zero_matrix(self):
            """
            Test the har_cov function with a zero matrix as input. 
            Since the input matrix is all zeros, it is expected to get a zero covariance matrix.
            """
            X = np.zeros((3, 3))
            H = har_cov(X, 1)
            assert np.allclose(H, np.zeros((3, 3)))

        def test_small_matrix(self):
            """
            Test the har_cov function with a small input matrix.
            The function is tested by comparing the result to a manually calculated covariance matrix.
            """
            X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
            H = har_cov(X, 1)

            # Manually calculate covariance matrix for X.
            X_centered = X - np.mean(X, axis=0)
            cov_manual = np.dot(X_centered.T, X_centered) / (X.shape[0] - 1)

            assert np.allclose(H, cov_manual)

        def test_single_column_zeros(self):
            """
            Test the har_cov function with an input array of zeros that has only one column.
            In this case, it should return zero since there is no variation in the input data.
            The HAR Covariance estimation takes into account the autocorrelation properties of the data, thus we can't compare it directly to the variance calculated by np.var.
            """
            X = np.zeros((5, 1))
            # The covariance of zeros is zero.
            expected_output = np.array([[0.0]])
            H = har_cov(X, 1)
            assert np.allclose(H, expected_output)

    class TestFailingCases:
        """
        Test cases where the har_cov function is expected to give nonsensical output.
        """

        def test_too_small_input(self):
            """
            Test the har_cov function with an input array that has too few rows.
            As there are not enough rows, the function will return a matrix full of NaN.
            """
            X = np.array([[1, 2]])
            with pytest.raises(AssertionError, match="h must be less than the number of time steps in X."):
                har_cov(X, 1)

        def test_negative_bandwidth(self):
            """
            Test the har_cov function with a negative bandwidth.
            As the bandwidth is negative, the function will return a matrix full of NaN.
            """
            X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
            with pytest.raises(AssertionError, match="h must be non-negative."):
                har_cov(X, -1)


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
            transition_probabilities = calculate_transition_probs(
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
            transition_probabilities = calculate_transition_probs(
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
            transition_probabilities = calculate_transition_probs(
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
                calculate_transition_probs(
                    assignments, n_components)

        def test_random_assignments_0_component(self):
            """
            Test calculate_transition_probs with random assignments and n_components=0.
            """
            assignments = np.array([0, 1, 2, 3, 4])
            n_components = 0
            with pytest.raises(AssertionError):
                calculate_transition_probs(
                    assignments, n_components)

        def test_non_integer_n_components(self):
            """
            Test calculate_transition_probs with a non-integer n_components value.
            """
            assignments = np.array([0, 1, 2, 3, 4])
            n_components = 2.5
            with pytest.raises(AssertionError):
                calculate_transition_probs(
                    assignments, n_components)

        def test_negative_n_components(self):
            """
            Test calculate_transition_probs with a negative n_components value.
            """
            assignments = np.array([0, 1, 2, 3, 4])
            n_components = -2
            with pytest.raises(AssertionError):
                calculate_transition_probs(
                    assignments, n_components)

        def test_assignment_out_of_range(self):
            """
            Test calculate_transition_probs with an assignment value out of the valid range.
            """
            assignments = np.array([0, 1, 2, 3, 4, 11])
            n_components = 5
            with pytest.raises(AssertionError):
                calculate_transition_probs(
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
            model = fit_hidden_markov_model(X, n_states, n_fits=1)
            assert isinstance(model, hmm.GaussianHMM)
            assert model.n_components == n_states

        def test_single_2D_point_1_state(self):
            """
            Test fit_hidden_markov_model with a single 2D point and n_states=1.
            """
            X = np.array([[1, 2]])
            n_states = 1
            model = fit_hidden_markov_model(X, n_states)
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
                model = fit_hidden_markov_model(X, n_states)

        def test_no_data_1_state(self):
            """
            Test fit_hidden_markov_model with no data and n_states=1.
            """
            X = np.array([[]])
            n_states = 1
            with pytest.raises(ValueError):
                model = fit_hidden_markov_model(X, n_states)

        def test_random_data_0_state(self):
            """
            Testfit_hidden_markov_model with random data and n_states=0.
            """
            X = np.array([[1, 2], [3, 4], [5, 6]])
            n_states = 0
            with pytest.raises(ValueError):
                model = fit_hidden_markov_model(X, n_states)
