"""
Property-based tests for bootstrap_ext.py using Hypothesis.

Following Jane Street best practices with parametrized tests and property-based testing.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays
from tsbootstrap.bootstrap_ext import (
    BlockDistributionBootstrap,
    DistributionBootstrapService,
    MarkovBootstrapService,
    StatisticPreservingService,
    WholeDistributionBootstrap,
)


class TestMarkovBootstrapServiceProperties:
    """Property-based tests for MarkovBootstrapService."""

    @given(
        n_blocks=st.integers(min_value=2, max_value=20),
        block_size=st.integers(min_value=5, max_value=50),
        n_features=st.integers(min_value=1, max_value=5),
        n_states=st.integers(min_value=1, max_value=5),
        method=st.sampled_from(["first", "middle", "last"]),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    @settings(max_examples=50, deadline=5000)
    def test_markov_model_fitting_properties(
        self, n_blocks, block_size, n_features, n_states, method, seed
    ):
        """Property: Markov model fitting should always produce valid sampler."""
        service = MarkovBootstrapService()

        # Generate random blocks
        blocks = [np.random.randn(block_size, n_features) for _ in range(n_blocks)]
        rng = np.random.default_rng(seed)

        # Fit model
        sampler = service.fit_markov_model(
            blocks=blocks,
            n_states=n_states,
            method=method,
            apply_pca_flag=False,
            n_iter_hmm=10,
            n_fits_hmm=1,
            rng=rng,
        )

        # Properties to verify
        assert sampler is not None
        assert service._markov_sampler is sampler

        # Sampling should produce valid output
        samples = service.sample_markov_sequence(sampler, size=10)
        assert isinstance(samples, np.ndarray)
        assert len(samples) == 10


class TestDistributionBootstrapServiceProperties:
    """Property-based tests for DistributionBootstrapService."""

    @given(
        data=arrays(
            dtype=np.float64, shape=array_shapes(min_dims=1, max_dims=2, min_side=10, max_side=100)
        ),
        distribution=st.sampled_from(["normal", "kde"]),
    )
    def test_distribution_fitting_properties(self, data, distribution):
        """Property: Any valid data should be fittable to a distribution."""
        service = DistributionBootstrapService()

        # Fit distribution
        fitted = service.fit_distribution(data, distribution=distribution)

        # Verify properties
        assert fitted["distribution"] == distribution
        assert fitted["ndim"] == data.ndim

        if distribution == "normal":
            assert "mean" in fitted
            assert "std" in fitted

            if data.ndim == 1:
                assert isinstance(fitted["mean"], (float, np.floating))
                assert isinstance(fitted["std"], (float, np.floating))
            else:
                assert fitted["mean"].shape == (data.shape[1],)
                assert fitted["std"].shape == (data.shape[1],)
        else:  # kde
            assert "kde" in fitted

    @given(
        size=st.integers(min_value=1, max_value=1000),
        mean=st.floats(min_value=-100, max_value=100, allow_nan=False),
        std=st.floats(min_value=0.1, max_value=10, allow_nan=False),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    def test_normal_sampling_preserves_statistics(self, size, mean, std, seed):
        """Property: Sampling from normal distribution preserves statistics."""
        service = DistributionBootstrapService()
        rng = np.random.default_rng(seed)

        fitted = {"distribution": "normal", "mean": mean, "std": std, "ndim": 1}

        samples = service.sample_from_distribution(fitted, size=size, rng=rng)

        # With enough samples, statistics should be close
        if size > 100:
            assert np.abs(np.mean(samples) - mean) < 3 * std / np.sqrt(size)
            assert np.abs(np.std(samples) - std) < std * 0.5


class TestStatisticPreservingServiceProperties:
    """Property-based tests for StatisticPreservingService."""

    @given(
        data=arrays(
            dtype=np.float64,
            shape=array_shapes(min_dims=1, max_dims=2, min_side=10, max_side=100),
            elements=st.floats(min_value=-100, max_value=100, allow_nan=False),
        )
    )
    def test_default_statistics_computation(self, data):
        """Property: Default statistics should be computable for any valid data."""
        service = StatisticPreservingService()

        stats = service._default_statistics(data)

        assert "mean" in stats
        assert "std" in stats
        assert "acf_lag1" in stats

        # Verify shapes
        if data.ndim == 1:
            assert isinstance(stats["mean"], (float, np.floating))
            assert isinstance(stats["std"], (float, np.floating))
        else:
            assert stats["mean"].shape == (data.shape[1],)
            assert stats["std"].shape == (data.shape[1],)

    @given(
        sample_size=st.integers(min_value=10, max_value=100),
        n_features=st.integers(min_value=1, max_value=5),
        target_mean=st.floats(min_value=-10, max_value=10, allow_nan=False),
        target_std=st.floats(min_value=0.1, max_value=5, allow_nan=False),
    )
    def test_statistic_preservation_properties(
        self, sample_size, n_features, target_mean, target_std
    ):
        """Property: Adjustment should move statistics toward target."""
        service = StatisticPreservingService()

        # Generate random sample
        if n_features == 1:
            sample = np.random.randn(sample_size) * 2 + 5
            target_stats = {"mean": target_mean}
        else:
            sample = np.random.randn(sample_size, n_features) * 2 + 5
            target_stats = {"mean": np.full(n_features, target_mean)}

        # Adjust sample
        adjusted = service.adjust_sample_to_preserve_statistics(sample, target_stats, {})

        # Mean should be exactly preserved
        adjusted_mean = np.mean(adjusted, axis=0 if sample.ndim > 1 else None)
        if n_features == 1:
            assert np.allclose(adjusted_mean, target_mean, rtol=1e-10)
        else:
            assert np.allclose(adjusted_mean, target_mean, rtol=1e-10)


@pytest.mark.parametrize("distribution", ["normal", "kde"])
@pytest.mark.parametrize("n_bootstraps", [1, 5, 10])
@pytest.mark.parametrize("data_dim", [1, 2])
class TestBootstrapIntegrationParametrized:
    """Parametrized integration tests for bootstrap classes."""

    def test_distribution_bootstrap_consistency(self, distribution, n_bootstraps, data_dim):
        """Test that distribution bootstrap produces consistent results."""
        # Generate appropriate data
        data = np.random.randn(100) if data_dim == 1 else np.random.randn(100, 3)

        # Create bootstrap instance
        bootstrap = WholeDistributionBootstrap(
            n_bootstraps=n_bootstraps, distribution=distribution, rng=42
        )

        # Generate samples
        samples = list(bootstrap.bootstrap(data))

        # Verify properties
        assert len(samples) == n_bootstraps
        for sample in samples:
            assert sample.shape == data.shape
            assert sample.dtype == data.dtype


@pytest.mark.parametrize(
    "block_length,n_samples,expected_error",
    [
        (10, 5, ValueError),  # Block length > data length
        (0, 100, ValueError),  # Invalid block length
        (-5, 100, ValueError),  # Negative block length
    ],
)
def test_block_bootstrap_validation(block_length, n_samples, expected_error):
    """Test block bootstrap parameter validation."""
    data = np.random.randn(n_samples)

    with pytest.raises(expected_error):
        bootstrap = BlockDistributionBootstrap(n_bootstraps=1, block_length=block_length, rng=42)
        list(bootstrap.bootstrap(data))


class TestEdgeCasesWithHypothesis:
    """Edge case testing with property-based approaches."""

    @given(
        data=arrays(
            dtype=np.float64,
            shape=st.tuples(
                st.integers(min_value=2, max_value=10), st.integers(min_value=1, max_value=5)
            ),
            elements=st.floats(
                min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
            ),
        )
    )
    def test_acf_computation_never_fails(self, data):
        """Property: ACF computation should never fail for valid data."""
        service = StatisticPreservingService()

        # Should handle any lag
        for lag in range(1, len(data)):
            acf = service._compute_acf(data, lag=lag)
            assert isinstance(acf, (float, np.floating))
            assert -1 <= acf <= 1 or np.isnan(acf)  # ACF is correlation

    @given(n_features=st.integers(min_value=1, max_value=10), contains_zero_std=st.booleans())
    def test_zero_variance_handling(self, n_features, contains_zero_std):
        """Property: Should handle zero variance dimensions gracefully."""
        service = StatisticPreservingService()

        # Create data with controlled variance
        sample = np.random.randn(100, n_features)
        if contains_zero_std and n_features > 1:
            # Make first dimension constant
            sample[:, 0] = 1.0

        target_stats = {"std": np.ones(n_features) * 2.0}

        # Should not fail
        adjusted = service.adjust_sample_to_preserve_statistics(sample, target_stats, {})

        assert adjusted.shape == sample.shape
        assert not np.any(np.isnan(adjusted))
        assert not np.any(np.isinf(adjusted))
