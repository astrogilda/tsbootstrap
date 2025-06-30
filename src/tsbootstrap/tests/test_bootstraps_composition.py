"""Direct tests for bootstrap classes using composition architecture."""

import numpy as np
import pytest

# Import bootstrap classes
from tsbootstrap.block_bootstrap import (
    CircularBlockBootstrap,
    HammingBootstrap,
    MovingBlockBootstrap,
    StationaryBlockBootstrap,
    TukeyBootstrap,
)
from tsbootstrap.bootstrap import (
    BlockResidualBootstrap,
    BlockSieveBootstrap,
    WholeResidualBootstrap,
    WholeSieveBootstrap,
)
from tsbootstrap.bootstrap_ext import (
    WholeDistributionBootstrap,
    WholeMarkovBootstrap,
    WholeStatisticPreservingBootstrap,
)


class TestBootstrapsComposition:
    """Test suite for bootstrap classes using service composition."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data."""
        np.random.seed(42)
        n = 100
        return np.random.randn(n).cumsum().reshape(-1, 1)

    def test_whole_residual_bootstrap(self, sample_data):
        """Test WholeResidualBootstrap."""
        bootstrap = WholeResidualBootstrap(model_type="ar", order=2, random_state=42)

        # Generate bootstrap samples
        bootstrap.n_bootstraps = 5
        samples = list(bootstrap.bootstrap(X=sample_data))

        assert len(samples) == 5
        assert all(s.shape == sample_data.shape for s in samples)

    def test_block_residual_bootstrap(self, sample_data):
        """Test BlockResidualBootstrap."""
        bootstrap = BlockResidualBootstrap(
            model_type="ar", order=2, block_length=10, random_state=42
        )

        # Generate bootstrap samples
        bootstrap.n_bootstraps = 5
        samples = list(bootstrap.bootstrap(X=sample_data))

        assert len(samples) == 5
        assert all(s.shape == sample_data.shape for s in samples)

    def test_whole_sieve_bootstrap(self, sample_data):
        """Test WholeSieveBootstrap."""
        bootstrap = WholeSieveBootstrap(model_type="ar", criterion="aic", random_state=42)

        # Generate bootstrap samples
        bootstrap.n_bootstraps = 5
        samples = list(bootstrap.bootstrap(X=sample_data))

        assert len(samples) == 5
        assert all(s.shape == sample_data.shape for s in samples)

    def test_block_sieve_bootstrap(self, sample_data):
        """Test BlockSieveBootstrap."""
        bootstrap = BlockSieveBootstrap(
            model_type="ar", criterion="aic", block_length=10, random_state=42
        )

        # Generate bootstrap samples
        bootstrap.n_bootstraps = 5
        samples = list(bootstrap.bootstrap(X=sample_data))

        assert len(samples) == 5
        assert all(s.shape == sample_data.shape for s in samples)

    @pytest.mark.skipif(
        not pytest.importorskip("hmmlearn", reason="hmmlearn not available"),
        reason="hmmlearn required",
    )
    def test_whole_markov_bootstrap(self, sample_data):
        """Test WholeMarkovBootstrap."""
        bootstrap = WholeMarkovBootstrap(n_states=2, random_state=42)

        # Generate bootstrap samples
        bootstrap.n_bootstraps = 3
        samples = list(bootstrap.bootstrap(X=sample_data))

        assert len(samples) == 3
        assert all(s.shape == sample_data.shape for s in samples)

    def test_whole_distribution_bootstrap(self, sample_data):
        """Test WholeDistributionBootstrap."""
        bootstrap = WholeDistributionBootstrap(distribution="normal", random_state=42)

        # Generate bootstrap samples
        bootstrap.n_bootstraps = 5
        samples = list(bootstrap.bootstrap(X=sample_data))

        assert len(samples) == 5
        assert all(s.shape == sample_data.shape for s in samples)

    def test_whole_statistic_preserving_bootstrap(self, sample_data):
        """Test WholeStatisticPreservingBootstrap."""
        bootstrap = WholeStatisticPreservingBootstrap(
            statistic="mean", random_state=42  # Use string to specify statistic type
        )

        # Generate bootstrap samples
        bootstrap.n_bootstraps = 3
        samples = list(bootstrap.bootstrap(X=sample_data))

        assert len(samples) == 3
        assert all(s.shape == sample_data.shape for s in samples)

    def test_moving_block_bootstrap(self, sample_data):
        """Test MovingBlockBootstrap."""
        bootstrap = MovingBlockBootstrap(block_length=10, random_state=42)

        # Generate bootstrap samples
        bootstrap.n_bootstraps = 5
        samples = list(bootstrap.bootstrap(X=sample_data))

        assert len(samples) == 5
        assert all(s.shape == sample_data.shape for s in samples)

    def test_stationary_block_bootstrap(self, sample_data):
        """Test StationaryBlockBootstrap."""
        bootstrap = StationaryBlockBootstrap(average_block_length=10, random_state=42)

        # Generate bootstrap samples
        bootstrap.n_bootstraps = 5
        samples = list(bootstrap.bootstrap(X=sample_data))

        assert len(samples) == 5
        assert all(s.shape == sample_data.shape for s in samples)

    def test_circular_block_bootstrap(self, sample_data):
        """Test CircularBlockBootstrap."""
        bootstrap = CircularBlockBootstrap(block_length=10, random_state=42)

        # Generate bootstrap samples
        bootstrap.n_bootstraps = 5
        samples = list(bootstrap.bootstrap(X=sample_data))

        assert len(samples) == 5
        assert all(s.shape == sample_data.shape for s in samples)

    def test_hamming_bootstrap(self, sample_data):
        """Test HammingBootstrap."""
        bootstrap = HammingBootstrap(block_length=10, random_state=42)

        # Generate bootstrap samples
        bootstrap.n_bootstraps = 5
        samples = list(bootstrap.bootstrap(X=sample_data))

        assert len(samples) == 5
        assert all(s.shape == sample_data.shape for s in samples)

    def test_tukey_bootstrap(self, sample_data):
        """Test TukeyBootstrap."""
        bootstrap = TukeyBootstrap(block_length=10, alpha=0.5, random_state=42)

        # Generate bootstrap samples
        bootstrap.n_bootstraps = 5
        samples = list(bootstrap.bootstrap(X=sample_data))

        assert len(samples) == 5
        assert all(s.shape == sample_data.shape for s in samples)

    def test_services_composition(self):
        """Test that composition-based classes use service composition."""
        # Create an instance
        bootstrap = WholeResidualBootstrap(model_type="ar", order=2)

        # Check that it has services
        assert hasattr(bootstrap, "_services")

        # Check that it doesn't have internal methods directly (uses services instead)
        assert not hasattr(bootstrap.__class__, "_fit_model")  # Should use services
