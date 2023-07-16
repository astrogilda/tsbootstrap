from hypothesis import given, settings, strategies as st
import pytest
import numpy as np
from future_work.hac_sampler import har_cov


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
