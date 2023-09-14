import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tsbootstrap.utils.odds_and_ends import time_series_split


class TestTimeSeriesSplit:
    class TestPassingCases:
        @given(
            st.lists(
                st.floats(allow_infinity=False, allow_nan=False),
                min_size=2,
                max_size=100,
            ),
            st.floats(
                min_value=0.1,
                max_value=0.9,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
        def test_valid_input(self, X, test_ratio):
            X = np.array(X)
            X_train, X_test = time_series_split(X, test_ratio)
            assert len(X_train) == int(len(X) * (1 - test_ratio))
            assert len(X_test) == len(X) - len(X_train)
            assert np.all(X_train == X[: len(X_train)])
            assert np.all(X_test == X[len(X_train) :])

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
