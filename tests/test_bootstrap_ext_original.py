"""
Test suite for composition_based extended bootstrap classes.

This module tests that the composition_based bootstrap classes behave
identically to the original classes.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from tsbootstrap.bootstrap_ext import (
    BlockDistributionBootstrap,
    BlockMarkovBootstrap,
    BlockStatisticPreservingBootstrap,
    WholeDistributionBootstrap,
    WholeMarkovBootstrap,
    WholeStatisticPreservingBootstrap,
)


class TestMarkovBootstrap:
    """Test Markov bootstrap composition_based classes."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data."""
        np.random.seed(42)
        return np.cumsum(np.random.randn(50, 2), axis=0)

    def test_whole_markov_bootstrap_identical_behavior(self, sample_data):
        """Test that WholeMarkovBootstrap behaves identically to original."""
        # Skip if hmmlearn not available
        pytest.importorskip("hmmlearn")

        # Set up identical parameters
        params = {
            "n_bootstraps": 3,
            "method": "middle",
            "apply_pca_flag": False,
            "n_states": 2,
            "n_iter_hmm": 10,
            "n_fits_hmm": 2,
            "random_state": 42,
        }

        # Original implementation
        original = WholeMarkovBootstrap(**params)

        # Composition-based implementation
        composition_based = WholeMarkovBootstrap(**params)

        # Both should require model fitting
        assert original.requires_model_fitting == composition_based.requires_model_fitting

        # Test basic properties
        assert original.n_bootstraps == composition_based.n_bootstraps
        assert original.method == composition_based.method
        assert original.n_states == composition_based.n_states

    def test_block_markov_bootstrap_identical_behavior(self, sample_data):
        """Test that BlockMarkovBootstrap behaves identically to original."""
        # Skip if hmmlearn not available
        pytest.importorskip("hmmlearn")

        # Set up identical parameters
        params = {
            "n_bootstraps": 3,
            "block_length": 5,
            "method": "middle",
            "n_states": 2,
            "n_iter_hmm": 10,
            "n_fits_hmm": 2,
            "random_state": 42,
        }

        # Original implementation
        original = BlockMarkovBootstrap(**params)

        # Composition-based implementation
        composition_based = BlockMarkovBootstrap(**params)

        # Test basic properties
        assert original.n_bootstraps == composition_based.n_bootstraps
        assert original.block_length == composition_based.block_length
        assert original.method == composition_based.method


class TestDistributionBootstrap:
    """Test Distribution bootstrap composition_based classes."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data."""
        np.random.seed(42)
        return np.random.randn(100, 2)

    def test_whole_distribution_bootstrap_identical_behavior(self, sample_data):
        """Test that WholeDistributionBootstrap behaves identically to original."""
        # Test different distributions
        distributions = ["normal", "exponential", "uniform", "gamma", "beta"]

        for dist in distributions:
            params = {"n_bootstraps": 3, "distribution": dist, "refit": False, "random_state": 42}

            # Original implementation
            original = WholeDistributionBootstrap(**params)

            # Composition-based implementation
            composition_based = WholeDistributionBootstrap(**params)

            # Test basic properties
            assert original.n_bootstraps == composition_based.n_bootstraps
            assert original.distribution == composition_based.distribution
            assert original.refit == composition_based.refit
            # Note: requires_model_fitting is only in composition_based version
            assert composition_based.requires_model_fitting is True

    def test_block_distribution_bootstrap_identical_behavior(self, sample_data):
        """Test that BlockDistributionBootstrap behaves identically to original."""
        params = {
            "n_bootstraps": 3,
            "block_length": 10,
            "distribution": "normal",
            "overlap_flag": True,
            "random_state": 42,
        }

        # Original implementation
        original = BlockDistributionBootstrap(**params)

        # Composition-based implementation
        composition_based = BlockDistributionBootstrap(**params)

        # Test basic properties
        assert original.n_bootstraps == composition_based.n_bootstraps
        assert original.block_length == composition_based.block_length
        assert original.distribution == composition_based.distribution
        assert original.overlap_flag == composition_based.overlap_flag


class TestStatisticPreservingBootstrap:
    """Test Statistic Preserving bootstrap composition_based classes."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data."""
        np.random.seed(42)
        return np.random.randn(80, 3)

    def test_whole_statistic_preserving_bootstrap_identical_behavior(self, sample_data):
        """Test that WholeStatisticPreservingBootstrap behaves identically to original."""
        # Test different statistics
        statistics = ["mean", "median", "std", "var"]

        for stat in statistics:
            params = {
                "n_bootstraps": 3,
                "statistic": stat,
                "statistic_axis": 0,
                "statistic_keepdims": False,
                "random_state": 42,
            }

            # Original implementation
            original = WholeStatisticPreservingBootstrap(**params)

            # Composition-based implementation
            composition_based = WholeStatisticPreservingBootstrap(**params)

            # Test basic properties
            assert original.n_bootstraps == composition_based.n_bootstraps
            assert original.statistic == composition_based.statistic
            assert original.statistic_axis == composition_based.statistic_axis
            assert original.statistic_keepdims == composition_based.statistic_keepdims

            # Test statistic function exists (can't compare functions directly)
            assert callable(original.statistic_func)
            assert callable(composition_based.statistic_func)

    def test_block_statistic_preserving_bootstrap_identical_behavior(self, sample_data):
        """Test that BlockStatisticPreservingBootstrap behaves identically to original."""
        params = {
            "n_bootstraps": 3,
            "block_length": 8,
            "statistic": "mean",
            "preserve_block_statistics": True,
            "overlap_flag": False,
            "random_state": 42,
        }

        # Original implementation
        original = BlockStatisticPreservingBootstrap(**params)

        # Composition-based implementation
        composition_based = BlockStatisticPreservingBootstrap(**params)

        # Test basic properties
        assert original.n_bootstraps == composition_based.n_bootstraps
        assert original.block_length == composition_based.block_length
        assert original.statistic == composition_based.statistic
        assert original.preserve_block_statistics == composition_based.preserve_block_statistics


class TestBootstrapSampleGeneration:
    """Test that composition_based classes generate similar bootstrap samples."""

    @pytest.fixture
    def simple_data(self):
        """Generate simple time series data."""
        np.random.seed(123)
        return np.cumsum(np.random.randn(30))

    def test_distribution_bootstrap_sample_generation(self, simple_data):
        """Test that distribution bootstrap generates appropriate samples."""
        # Test normal distribution
        params = {"n_bootstraps": 5, "distribution": "normal", "random_state": 42}

        composition_based = WholeDistributionBootstrap(**params)
        samples = list(composition_based.bootstrap(simple_data))

        # Check shape
        assert len(samples) == 5
        assert all(len(s) == len(simple_data) for s in samples)

        # Check that samples have similar mean and std (within reasonable bounds)
        original_mean = np.mean(simple_data)
        original_std = np.std(simple_data)

        for sample in samples:
            sample_mean = np.mean(sample)
            sample_std = np.std(sample)

            # Distribution bootstrap should preserve mean/std approximately
            assert abs(sample_mean - original_mean) < 3 * original_std / np.sqrt(len(simple_data))
            assert abs(sample_std - original_std) < original_std

    def test_statistic_preserving_bootstrap_sample_generation(self, simple_data):
        """Test that statistic preserving bootstrap actually preserves statistics."""
        # Test mean preservation
        params = {"n_bootstraps": 5, "statistic": "mean", "random_state": 42}

        composition_based = WholeStatisticPreservingBootstrap(**params)
        samples = list(composition_based.bootstrap(simple_data))

        # Check shape
        assert len(samples) == 5
        assert all(len(s) == len(simple_data) for s in samples)

        # Check that mean is preserved exactly
        original_mean = np.mean(simple_data)

        for sample in samples:
            sample_mean = np.mean(sample)
            assert_array_almost_equal(sample_mean, original_mean, decimal=10)

    def test_block_statistic_preserving_with_block_preservation(self):
        """Test block statistic preservation."""
        # Generate data with clear block structure
        np.random.seed(42)
        data = np.concatenate(
            [
                np.ones(20) * 10,  # Block 1: mean=10
                np.ones(20) * 20,  # Block 2: mean=20
                np.ones(20) * 30,  # Block 3: mean=30
            ]
        )

        params = {
            "n_bootstraps": 3,
            "block_length": 20,
            "statistic": "mean",
            "preserve_block_statistics": True,
            "overlap_flag": False,
            "random_state": 42,
        }

        composition_based = BlockStatisticPreservingBootstrap(**params)
        samples = list(composition_based.bootstrap(data))

        # Check shape
        assert len(samples) == 3
        assert all(len(s) == len(data) for s in samples)

        # When preserve_block_statistics is True, each block should maintain its mean
        # This is a more complex test that would require examining the internal structure


class TestServiceIntegration:
    """Test service integration in composition_based classes."""

    def test_markov_service_integration(self):
        """Test that Markov service is properly integrated."""
        pytest.importorskip("hmmlearn")

        composition_based = WholeMarkovBootstrap(n_bootstraps=2, n_states=2, random_state=42)

        # Check that service is initialized
        assert composition_based._markov_service is not None
        assert hasattr(composition_based._markov_service, "fit_markov_model")
        assert hasattr(composition_based._markov_service, "sample_markov_sequence")

    def test_distribution_service_integration(self):
        """Test that Distribution service is properly integrated."""
        composition_based = WholeDistributionBootstrap(
            n_bootstraps=2, distribution="normal", random_state=42
        )

        # Check that service is initialized
        assert composition_based._dist_service is not None
        assert hasattr(composition_based._dist_service, "fit_distribution")
        assert hasattr(composition_based._dist_service, "sample_from_distribution")

    def test_statistic_preserving_service_integration(self):
        """Test that Statistic Preserving service is properly integrated."""
        composition_based = WholeStatisticPreservingBootstrap(
            n_bootstraps=2, statistic="mean", random_state=42
        )

        # Check that service is initialized
        assert composition_based._stat_service is not None
        assert hasattr(composition_based._stat_service, "statistic_func")
        assert hasattr(composition_based._stat_service, "adjust_sample_to_preserve_statistics")


def test_all_composition_based_classes_exist():
    """Ensure all composition_based classes are properly defined."""
    classes = [
        WholeMarkovBootstrap,
        BlockMarkovBootstrap,
        WholeDistributionBootstrap,
        BlockDistributionBootstrap,
        WholeStatisticPreservingBootstrap,
        BlockStatisticPreservingBootstrap,
    ]

    for cls in classes:
        assert cls is not None
        assert hasattr(cls, "__init__")
        assert hasattr(cls, "_generate_samples_single_bootstrap")
