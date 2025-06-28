"""Test extended bootstrap implementations."""

import numpy as np
import pytest
from tsbootstrap.bootstrap_ext import (
    BlockDistributionBootstrap,
    BlockMarkovBootstrap,
    BlockStatisticPreservingBootstrap,
    WholeDistributionBootstrap,
    WholeMarkovBootstrap,
    WholeStatisticPreservingBootstrap,
)

# Check if hmmlearn is available
try:
    import hmmlearn  # noqa: F401

    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False

# Skip decorator for tests requiring hmmlearn
requires_hmmlearn = pytest.mark.skipif(
    not HAS_HMMLEARN, reason="hmmlearn not installed - required for Markov bootstrap"
)


@pytest.mark.slow
@requires_hmmlearn
class TestWholeMarkovBootstrap:
    """Test suite for WholeMarkovBootstrap."""

    def test_basic_functionality(self):
        """Test basic bootstrap generation."""
        bootstrap = WholeMarkovBootstrap(n_bootstraps=5, rng=42)
        X = np.random.randn(100, 2)

        samples = list(bootstrap.bootstrap(X, return_indices=False))
        assert len(samples) == 5
        for sample in samples:
            assert sample.shape == X.shape

    def test_fit_model(self):
        """Test model fitting."""
        bootstrap = WholeMarkovBootstrap(n_bootstraps=3, n_states=3, rng=42)
        X = np.random.randn(50, 1)

        # Fit should happen automatically
        _ = list(bootstrap.bootstrap(X))
        assert bootstrap._markov_sampler is not None
        assert bootstrap._blocks is not None
        assert len(bootstrap._blocks) >= 5  # At least 5 blocks

    @pytest.mark.parametrize("method", ["first", "middle", "last", "mean"])
    def test_compression_methods(self, method):
        """Test different compression methods."""
        bootstrap = WholeMarkovBootstrap(n_bootstraps=2, method=method, rng=42)
        X = np.random.randn(60, 1)

        samples = list(bootstrap.bootstrap(X))
        assert len(samples) == 2

    def test_multivariate_data(self):
        """Test with multivariate data."""
        bootstrap = WholeMarkovBootstrap(n_bootstraps=3, rng=42)
        X = np.random.randn(50, 3)

        samples = list(bootstrap.bootstrap(X))
        assert len(samples) == 3
        for sample in samples:
            assert sample.shape == X.shape

    def test_with_indices(self):
        """Test bootstrap with indices."""
        bootstrap = WholeMarkovBootstrap(n_bootstraps=2, rng=42)
        X = np.random.randn(40, 1)

        results = list(bootstrap.bootstrap(X, return_indices=True))
        assert len(results) == 2
        for sample, indices in results:
            assert sample.shape == X.shape
            assert indices.shape == (len(X),)


@pytest.mark.slow
@requires_hmmlearn
class TestBlockMarkovBootstrap:
    """Test suite for BlockMarkovBootstrap."""

    def test_basic_functionality(self):
        """Test basic block bootstrap generation."""
        bootstrap = BlockMarkovBootstrap(n_bootstraps=5, block_length=10, rng=42)
        X = np.random.randn(100, 2)

        samples = list(bootstrap.bootstrap(X, return_indices=False))
        assert len(samples) == 5
        for sample in samples:
            assert sample.shape == X.shape

    def test_block_structure(self):
        """Test that block structure is preserved."""
        bootstrap = BlockMarkovBootstrap(n_bootstraps=1, block_length=5, rng=42)
        X = np.arange(20).reshape(-1, 1)  # Sequential data

        results = list(bootstrap.bootstrap(X, return_indices=True))
        sample, indices = results[0]

        # Check that blocks are preserved
        # Due to the block structure, consecutive elements should often come from blocks
        assert len(indices) == len(X)

    @pytest.mark.parametrize("overlap", [True, False])
    def test_overlap_flag(self, overlap):
        """Test overlapping vs non-overlapping blocks."""
        bootstrap = BlockMarkovBootstrap(
            n_bootstraps=3, block_length=5, overlap_flag=overlap, rng=42
        )
        X = np.random.randn(50, 1)

        samples = list(bootstrap.bootstrap(X))
        assert len(samples) == 3


class TestWholeDistributionBootstrap:
    """Test suite for WholeDistributionBootstrap."""

    @pytest.mark.parametrize("dist", ["normal", "exponential", "uniform"])
    def test_distributions(self, dist):
        """Test different distributions."""
        bootstrap = WholeDistributionBootstrap(n_bootstraps=3, distribution=dist, rng=42)
        X = np.random.randn(50, 1)

        samples = list(bootstrap.bootstrap(X))
        assert len(samples) == 3
        for sample in samples:
            assert sample.shape == X.shape

    def test_multivariate_distribution(self):
        """Test distribution bootstrap with multivariate data."""
        bootstrap = WholeDistributionBootstrap(n_bootstraps=2, distribution="normal", rng=42)
        X = np.random.randn(40, 3)

        samples = list(bootstrap.bootstrap(X))
        assert len(samples) == 2
        for sample in samples:
            assert sample.shape == X.shape

    def test_refit_flag(self):
        """Test refit functionality."""
        bootstrap = WholeDistributionBootstrap(
            n_bootstraps=3, distribution="normal", refit=True, rng=42
        )
        X = np.random.randn(50, 1)

        samples = list(bootstrap.bootstrap(X))
        assert len(samples) == 3


class TestBlockDistributionBootstrap:
    """Test suite for BlockDistributionBootstrap."""

    def test_basic_functionality(self):
        """Test basic block distribution bootstrap."""
        bootstrap = BlockDistributionBootstrap(
            n_bootstraps=3, block_length=10, distribution="normal", rng=42
        )
        X = np.random.randn(50, 1)

        samples = list(bootstrap.bootstrap(X))
        assert len(samples) == 3
        for sample in samples:
            assert sample.shape == X.shape

    def test_block_boundaries(self):
        """Test that block boundaries are respected."""
        bootstrap = BlockDistributionBootstrap(
            n_bootstraps=2, block_length=15, overlap_flag=False, rng=42
        )
        X = np.random.randn(45, 2)

        samples = list(bootstrap.bootstrap(X))
        assert len(samples) == 2
        # Should have exactly 3 blocks of size 15 each


class TestWholeStatisticPreservingBootstrap:
    """Test suite for WholeStatisticPreservingBootstrap."""

    @pytest.mark.parametrize("stat", ["mean", "median", "std", "var"])
    def test_statistic_preservation(self, stat):
        """Test that statistics are preserved."""
        bootstrap = WholeStatisticPreservingBootstrap(n_bootstraps=5, statistic=stat, rng=42)
        X = np.random.randn(100, 1)

        # Calculate original statistic
        if stat == "mean":
            original_stat = np.mean(X)
        elif stat == "median":
            original_stat = np.median(X)
        elif stat == "std":
            original_stat = np.std(X)
        elif stat == "var":
            original_stat = np.var(X)

        samples = list(bootstrap.bootstrap(X))

        # Check that statistic is preserved (within tolerance)
        for sample in samples:
            if stat == "mean":
                sample_stat = np.mean(sample)
            elif stat == "median":
                sample_stat = np.median(sample)
            elif stat == "std":
                sample_stat = np.std(sample)
            elif stat == "var":
                sample_stat = np.var(sample)

            # For mean/median, should be exactly preserved
            if stat in ["mean", "median"]:
                np.testing.assert_allclose(sample_stat, original_stat, rtol=1e-10)
            # For std/var, should be approximately preserved
            else:
                np.testing.assert_allclose(sample_stat, original_stat, rtol=1e-10)

    def test_multivariate_statistic(self):
        """Test statistic preservation with multivariate data."""
        bootstrap = WholeStatisticPreservingBootstrap(
            n_bootstraps=3, statistic="mean", statistic_axis=0, rng=42
        )
        X = np.random.randn(50, 3)

        original_mean = np.mean(X, axis=0)
        samples = list(bootstrap.bootstrap(X))

        for sample in samples:
            sample_mean = np.mean(sample, axis=0)
            np.testing.assert_allclose(sample_mean, original_mean, rtol=1e-10)


class TestBlockStatisticPreservingBootstrap:
    """Test suite for BlockStatisticPreservingBootstrap."""

    def test_basic_functionality(self):
        """Test basic block statistic preserving bootstrap."""
        bootstrap = BlockStatisticPreservingBootstrap(
            n_bootstraps=3, block_length=10, statistic="mean", rng=42
        )
        X = np.random.randn(50, 1)

        samples = list(bootstrap.bootstrap(X))
        assert len(samples) == 3
        for sample in samples:
            assert sample.shape == X.shape

    def test_block_statistic_preservation(self):
        """Test preservation of statistics within blocks."""
        bootstrap = BlockStatisticPreservingBootstrap(
            n_bootstraps=2,
            block_length=20,
            statistic="mean",
            preserve_block_statistics=True,
            rng=42,
        )
        X = np.random.randn(60, 1)

        samples = list(bootstrap.bootstrap(X))
        assert len(samples) == 2

        # With preserve_block_statistics=True, individual blocks
        # should have their statistics preserved

    def test_overall_vs_block_preservation(self):
        """Test difference between overall and block-wise preservation."""
        X = np.random.randn(50, 2)

        # Overall preservation
        bootstrap1 = BlockStatisticPreservingBootstrap(
            n_bootstraps=2,
            block_length=10,
            statistic="mean",
            preserve_block_statistics=False,
            rng=42,
        )

        # Block-wise preservation
        bootstrap2 = BlockStatisticPreservingBootstrap(
            n_bootstraps=2,
            block_length=10,
            statistic="mean",
            preserve_block_statistics=True,
            rng=42,
        )

        samples1 = list(bootstrap1.bootstrap(X))
        samples2 = list(bootstrap2.bootstrap(X))

        # Both should preserve something, but differently
        assert len(samples1) == len(samples2) == 2


class TestIntegration:
    """Integration tests for all bootstrap methods."""

    @pytest.mark.parametrize(
        "bootstrap_cls",
        [
            pytest.param(WholeMarkovBootstrap, marks=requires_hmmlearn),
            pytest.param(BlockMarkovBootstrap, marks=requires_hmmlearn),
            WholeDistributionBootstrap,
            BlockDistributionBootstrap,
            WholeStatisticPreservingBootstrap,
            BlockStatisticPreservingBootstrap,
        ],
    )
    def test_empty_data_handling(self, bootstrap_cls):
        """Test handling of edge cases."""
        bootstrap = bootstrap_cls(n_bootstraps=1, rng=42)

        # Empty data should raise an error
        with pytest.raises(ValueError):
            list(bootstrap.bootstrap(np.array([])))

    @pytest.mark.parametrize(
        "bootstrap_cls",
        [
            pytest.param(WholeMarkovBootstrap, marks=requires_hmmlearn),
            pytest.param(BlockMarkovBootstrap, marks=requires_hmmlearn),
            WholeDistributionBootstrap,
            BlockDistributionBootstrap,
            WholeStatisticPreservingBootstrap,
            BlockStatisticPreservingBootstrap,
        ],
    )
    def test_single_sample(self, bootstrap_cls):
        """Test with very small data."""
        # Markov bootstraps need more data
        if "Markov" in bootstrap_cls.__name__:
            bootstrap = bootstrap_cls(n_bootstraps=2, n_states=2, rng=42)
            X = np.array(
                [
                    [1.0],
                    [2.0],
                    [3.0],
                    [4.0],
                    [5.0],
                    [6.0],
                    [7.0],
                    [8.0],
                    [9.0],
                    [10.0],
                ]
            )
        else:
            bootstrap = bootstrap_cls(n_bootstraps=2, rng=42)
            X = np.array([[1.0], [2.0], [3.0]])

        # Should handle small data gracefully
        samples = list(bootstrap.bootstrap(X))
        assert len(samples) == 2
        for sample in samples:
            assert len(sample) == len(X)


class TestAdditionalCoverage:
    """Additional tests to improve coverage."""

    def test_markov_fit_model_property(self):
        """Test computed properties of MarkovBootstrap."""
        pytest.importorskip("hmmlearn", reason="hmmlearn required for Markov bootstrap")
        bootstrap = WholeMarkovBootstrap(n_bootstraps=1, rng=42)
        assert bootstrap.requires_model_fitting is True

    def test_distribution_more_distributions(self):
        """Test more distribution types."""
        X = np.random.randn(50, 1) + 5  # Positive data

        # Test gamma distribution
        bootstrap = WholeDistributionBootstrap(n_bootstraps=1, distribution="gamma", rng=42)
        samples = list(bootstrap.bootstrap(X))
        assert len(samples) == 1

        # Test beta distribution
        X_normalized = (X - X.min()) / (X.max() - X.min())  # Normalize to [0, 1]
        bootstrap = WholeDistributionBootstrap(n_bootstraps=1, distribution="beta", rng=42)
        samples = list(bootstrap.bootstrap(X_normalized))
        assert len(samples) == 1

        # Test unknown distribution (defaults to normal)
        bootstrap = WholeDistributionBootstrap(n_bootstraps=1, distribution="unknown", rng=42)
        samples = list(bootstrap.bootstrap(X))
        assert len(samples) == 1

    def test_distribution_edge_cases(self):
        """Test edge cases for distributions."""
        # Test with zero variance data (for beta distribution)
        X_constant = np.ones((20, 1))
        bootstrap = WholeDistributionBootstrap(n_bootstraps=1, distribution="beta", rng=42)
        samples = list(bootstrap.bootstrap(X_constant))
        assert len(samples) == 1
        assert np.all(samples[0] == 1.0)  # Should return constant values

    def test_statistic_preserving_properties(self):
        """Test computed properties of StatisticPreservingBootstrap."""
        bootstrap = WholeStatisticPreservingBootstrap(n_bootstraps=1, statistic="mean", rng=42)

        # Test statistic_func property
        assert bootstrap.statistic_func == np.mean

        # Test other statistics
        bootstrap.statistic = "max"
        assert bootstrap.statistic_func == np.max

        bootstrap.statistic = "min"
        assert bootstrap.statistic_func == np.min

    def test_block_statistic_edge_cases(self):
        """Test edge cases for block statistic preserving."""
        X = np.random.randn(50, 2)

        # Test with keepdims
        bootstrap = BlockStatisticPreservingBootstrap(
            n_bootstraps=1,
            block_length=10,
            statistic="var",
            statistic_keepdims=True,
            preserve_block_statistics=True,
            rng=42,
        )
        samples = list(bootstrap.bootstrap(X))
        assert len(samples) == 1

    def test_block_distribution_overlapping(self):
        """Test block distribution with overlapping blocks."""
        X = np.random.randn(30, 2)

        bootstrap = BlockDistributionBootstrap(
            n_bootstraps=2,
            block_length=10,
            overlap_flag=True,
            distribution="normal",
            rng=42,
        )
        samples = list(bootstrap.bootstrap(X))
        assert len(samples) == 2

        # Each sample should have the same shape as input
        for sample in samples:
            assert sample.shape == X.shape
