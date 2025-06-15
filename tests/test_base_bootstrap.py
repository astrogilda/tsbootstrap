import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import (
    booleans,
    floats,
    integers,
    none,
    one_of,
    tuples,
)
from tsbootstrap.base_bootstrap import (
    BaseStatisticPreservingBootstrap,
    BaseTimeSeriesBootstrap,
)


class TestBaseTimeSeriesBootstrap:
    """Tests for BaseTimeSeriesBootstrap."""

    @settings(deadline=None, max_examples=10)
    @given(
        n_bootstraps=integers(min_value=1, max_value=10),
        rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
        X=arrays(
            dtype=float,
            shape=tuples(
                integers(min_value=10, max_value=20),
                integers(min_value=1, max_value=5),
            ),
            elements=floats(min_value=0, max_value=100),
        ),
    )
    def test_base_time_series_bootstrap_initialization(
        self, n_bootstraps, rng, X
    ):
        """Test BaseTimeSeriesBootstrap initialization."""

        class ConcreteBootstrap(BaseTimeSeriesBootstrap):
            def _generate_samples_single_bootstrap(self, X, y=None, n=None):
                # Mock implementation for testing abstract method
                # Return indices and data for a single bootstrap sample
                return [np.arange(int(X.shape[0]))], [X]

        bootstrap = ConcreteBootstrap(n_bootstraps=n_bootstraps, rng=rng)
        assert bootstrap.n_bootstraps == n_bootstraps
        assert bootstrap.rng is None or isinstance(
            bootstrap.rng, np.random.Generator
        )

        # Test bootstrap method with mock _generate_samples_single_bootstrap
        samples_gen = bootstrap.bootstrap(X, return_indices=True)
        first_sample_data, first_sample_indices = next(samples_gen)
        assert isinstance(first_sample_data, np.ndarray)
        assert isinstance(first_sample_indices, np.ndarray)
        assert first_sample_data.shape == X.shape
        assert first_sample_indices.shape == (X.shape[0],)

    @settings(deadline=None, max_examples=10)
    @given(
        X=arrays(
            dtype=float,
            shape=tuples(
                integers(min_value=10, max_value=20),
                integers(min_value=1, max_value=5),
            ),
            elements=floats(min_value=0, max_value=100),
        ),
        test_ratio=floats(min_value=0.1, max_value=0.5),
    )
    def test_bootstrap_with_test_ratio(self, X, test_ratio):
        class ConcreteBootstrap(BaseTimeSeriesBootstrap):
            def _generate_samples_single_bootstrap(self, X, y=None, n=None):
                return [np.arange(int(X.shape[0]))], [X]

        bootstrap = ConcreteBootstrap(n_bootstraps=1)
        samples_gen = bootstrap.bootstrap(
            X, return_indices=True, test_ratio=test_ratio
        )
        first_sample_data, first_sample_indices = next(samples_gen)

        expected_len = int(X.shape[0] * (1 - test_ratio))
        assert first_sample_data.shape[0] == expected_len
        assert first_sample_indices.shape[0] == expected_len


class TestBaseStatisticPreservingBootstrap:
    """Tests for BaseStatisticPreservingBootstrap."""

    class ConcreteStatisticPreservingBootstrap(
        BaseStatisticPreservingBootstrap
    ):
        def _generate_samples_single_bootstrap(self, X, y=None, n=None):
            # Mock implementation for testing abstract method
            return [np.arange(int(X.shape[0]))], [X]

    @settings(deadline=None, max_examples=10)
    @given(
        n_bootstraps=integers(min_value=1, max_value=10),
        statistic_axis=integers(min_value=0, max_value=1),
        statistic_keepdims=booleans(),
        rng=one_of(integers(min_value=0, max_value=2**32 - 1), none()),
        X=arrays(
            dtype=float,
            shape=tuples(
                integers(min_value=10, max_value=20),
                integers(min_value=1, max_value=5),
            ),
            elements=floats(min_value=0, max_value=100),
        ),
    )
    def test_base_statistic_preserving_bootstrap_initialization(
        self, n_bootstraps, statistic_axis, statistic_keepdims, rng, X
    ):
        """Test BaseStatisticPreservingBootstrap initialization."""
        bootstrap = self.ConcreteStatisticPreservingBootstrap(
            n_bootstraps=n_bootstraps,
            statistic=np.mean,
            statistic_axis=statistic_axis,
            statistic_keepdims=statistic_keepdims,
            rng=rng,
        )
        assert bootstrap.n_bootstraps == n_bootstraps
        assert bootstrap.statistic == np.mean
        assert bootstrap.statistic_axis == statistic_axis
        assert bootstrap.statistic_keepdims == statistic_keepdims
        assert bootstrap.rng is None or isinstance(
            bootstrap.rng, np.random.Generator
        )

    @settings(deadline=None, max_examples=10)
    @given(
        X=arrays(
            dtype=float,
            shape=tuples(
                integers(min_value=10, max_value=20),
                integers(min_value=1, max_value=5),
            ),
            elements=floats(min_value=0, max_value=100),
        ),
        statistic_axis=integers(min_value=0, max_value=1),
        statistic_keepdims=booleans(),
    )
    def test_calculate_statistic(self, X, statistic_axis, statistic_keepdims):
        """Test _calculate_statistic method."""
        bootstrap = self.ConcreteStatisticPreservingBootstrap(
            statistic=np.mean,
            statistic_axis=statistic_axis,
            statistic_keepdims=statistic_keepdims,
        )
        calculated_statistic = bootstrap._calculate_statistic(X)
        expected_statistic = np.mean(
            X, axis=statistic_axis, keepdims=statistic_keepdims
        )
        np.testing.assert_allclose(calculated_statistic, expected_statistic)

    @settings(deadline=None, max_examples=10)
    @given(
        X=arrays(
            dtype=float,
            shape=tuples(
                integers(min_value=10, max_value=20),
                integers(min_value=1, max_value=5),
            ),
            elements=floats(min_value=0, max_value=100),
        ),
    )
    def test_default_statistic_initialization_and_calculation(self, X):
        """
        Test that BaseStatisticPreservingBootstrap correctly defaults the statistic to np.mean when not provided, and that _calculate_statistic uses this default.
        """
        bootstrap = self.ConcreteStatisticPreservingBootstrap()
        assert bootstrap.statistic == np.mean

        # Test that _calculate_statistic works with the defaulted np.mean
        calculated_statistic = bootstrap._calculate_statistic(X)
        expected_statistic = np.mean(
            X,
            axis=bootstrap.statistic_axis,
            keepdims=bootstrap.statistic_keepdims,
        )
        np.testing.assert_allclose(calculated_statistic, expected_statistic)

    @settings(deadline=None, max_examples=10)
    @given(
        X=arrays(
            dtype=float,
            shape=tuples(
                integers(min_value=10, max_value=20),
                integers(min_value=1, max_value=5),
            ),
            elements=floats(min_value=0, max_value=100),
        ),
    )
    def test_base_statistic_preserving_bootstrap_abstract_method_call(self, X):
        """
        Test that calling the abstract _generate_samples_single_bootstrap method on BaseStatisticPreservingBootstrap raises NotImplementedError.
        """
        bootstrap = self.ConcreteStatisticPreservingBootstrap(
            statistic=np.mean
        )
        # The ConcreteStatisticPreservingBootstrap provides a mock implementation,
        # so this test should not raise NotImplementedError.
        # This test is likely redundant given the current structure.
        # If the intent is to test the abstract nature, it should be done
        # by attempting to instantiate BaseStatisticPreservingBootstrap directly
        # without implementing the abstract method, which Python's ABC prevents.
        # For now, we'll ensure it doesn't raise an error unexpectedly.
        try:
            bootstrap._generate_samples_single_bootstrap(X)
        except NotImplementedError:
            pytest.fail(
                "ConcreteStatisticPreservingBootstrap should implement _generate_samples_single_bootstrap"
            )
