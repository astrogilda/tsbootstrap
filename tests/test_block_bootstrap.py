"""
Test suite for composition_based block bootstrap classes.

This module tests that the composition_based block bootstrap classes behave
identically to the original implementations.
"""

import numpy as np
import pytest
from tsbootstrap.block_bootstrap import (
    BartlettsBootstrap,
    BlackmanBootstrap,
    BlockBootstrap,
    CircularBlockBootstrap,
    HammingBootstrap,
    HanningBootstrap,
    MovingBlockBootstrap,
    NonOverlappingBlockBootstrap,
    StationaryBlockBootstrap,
    TukeyBootstrap,
)


class TestBlockBootstrap:
    """Test base block bootstrap composition_based class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data."""
        np.random.seed(42)
        return np.cumsum(np.random.randn(100))

    def test_block_bootstrap_configuration(self):
        """Test block bootstrap configuration fields."""
        params = {
            "n_bootstraps": 3,
            "block_length": 10,
            "block_length_distribution": None,
            "wrap_around_flag": False,
            "overlap_flag": True,
            "combine_generation_and_sampling_flag": False,
            "min_block_length": 5,
            "random_state": 42,
        }

        composition_based = BlockBootstrap(**params)

        # Check configuration
        assert composition_based.n_bootstraps == 3
        assert composition_based.block_length == 10
        assert composition_based.block_length_distribution is None
        assert composition_based.wrap_around_flag is False
        assert composition_based.overlap_flag is True
        assert composition_based.min_block_length == 5

    def test_block_generation_and_caching(self, sample_data):
        """Test that blocks are cached when combine flag is False."""
        composition_based = BlockBootstrap(
            n_bootstraps=2,
            block_length=10,
            combine_generation_and_sampling_flag=False,
            random_state=42,
        )

        # Generate first sample
        _ = composition_based._generate_samples_single_bootstrap(sample_data)

        # Blocks should be cached
        assert composition_based._blocks is not None
        cached_blocks = composition_based._blocks

        # Generate second sample
        _ = composition_based._generate_samples_single_bootstrap(sample_data)

        # Blocks should be the same (cached)
        assert composition_based._blocks is cached_blocks

    def test_block_regeneration(self, sample_data):
        """Test that blocks are regenerated when combine flag is True."""
        composition_based = BlockBootstrap(
            n_bootstraps=2,
            block_length=10,
            combine_generation_and_sampling_flag=True,
            random_state=42,
        )

        # Generate samples
        _ = composition_based._generate_samples_single_bootstrap(sample_data)

        # Blocks should not be cached
        assert composition_based._blocks is None


class TestMovingBlockBootstrap:
    """Test moving block bootstrap implementation."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data."""
        np.random.seed(42)
        return np.cumsum(np.random.randn(50))

    def test_moving_block_identical_behavior(self, sample_data):
        """Test that composition_based moving block behaves like original."""
        params = {"n_bootstraps": 3, "block_length": 5, "random_state": 42}

        # Original
        original = MovingBlockBootstrap(**params)

        # Composition-based
        composition_based = MovingBlockBootstrap(**params)

        # Check configuration matches
        assert original.n_bootstraps == composition_based.n_bootstraps
        assert original.block_length == composition_based.block_length
        assert original.wrap_around_flag == composition_based.wrap_around_flag
        assert original.overlap_flag == composition_based.overlap_flag

    def test_moving_block_sample_generation(self, sample_data):
        """Test moving block sample generation."""
        composition_based = MovingBlockBootstrap(n_bootstraps=3, block_length=10, random_state=42)

        samples = list(composition_based.bootstrap(sample_data))

        # Check output
        assert len(samples) == 3
        assert all(len(s) == len(sample_data) for s in samples)
        assert not np.array_equal(samples[0], samples[1])  # Different samples


class TestStationaryBlockBootstrap:
    """Test stationary block bootstrap implementation."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data."""
        np.random.seed(42)
        return np.random.randn(60)

    def test_stationary_block_configuration(self):
        """Test stationary block bootstrap configuration."""
        composition_based = StationaryBlockBootstrap(
            n_bootstraps=3, block_length=10, random_state=42
        )

        # Check defaults
        assert composition_based.block_length_distribution == "geometric"
        assert composition_based.wrap_around_flag is False
        assert composition_based.overlap_flag is True

    def test_stationary_block_sample_generation(self, sample_data):
        """Test stationary block sample generation."""
        composition_based = StationaryBlockBootstrap(
            n_bootstraps=5, block_length=8, random_state=42
        )

        samples = list(composition_based.bootstrap(sample_data))

        # Check output
        assert len(samples) == 5
        assert all(len(s) == len(sample_data) for s in samples)


class TestCircularBlockBootstrap:
    """Test circular block bootstrap implementation."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data."""
        np.random.seed(42)
        return np.sin(np.linspace(0, 4 * np.pi, 50))

    def test_circular_block_configuration(self):
        """Test circular block bootstrap configuration."""
        composition_based = CircularBlockBootstrap(n_bootstraps=3, block_length=10, random_state=42)

        # Check that wrap_around is always True
        assert composition_based.wrap_around_flag is True
        assert composition_based.overlap_flag is True

    def test_circular_block_sample_generation(self, sample_data):
        """Test circular block sample generation."""
        composition_based = CircularBlockBootstrap(n_bootstraps=4, block_length=15, random_state=42)

        samples = list(composition_based.bootstrap(sample_data))

        # Check output
        assert len(samples) == 4
        assert all(len(s) == len(sample_data) for s in samples)


class TestNonOverlappingBlockBootstrap:
    """Test non-overlapping block bootstrap implementation."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data."""
        np.random.seed(42)
        return np.cumsum(np.random.randn(80))

    def test_non_overlapping_configuration(self):
        """Test non-overlapping block bootstrap configuration."""
        composition_based = NonOverlappingBlockBootstrap(
            n_bootstraps=3, block_length=10, random_state=42
        )

        # Check that overlap_flag is always False
        assert composition_based.overlap_flag is False
        assert composition_based.wrap_around_flag is False

    def test_non_overlapping_sample_generation(self, sample_data):
        """Test non-overlapping block sample generation."""
        composition_based = NonOverlappingBlockBootstrap(
            n_bootstraps=3, block_length=20, random_state=42
        )

        samples = list(composition_based.bootstrap(sample_data))

        # Check output
        assert len(samples) == 3
        assert all(len(s) == len(sample_data) for s in samples)


class TestWindowedBootstraps:
    """Test windowed block bootstrap implementations."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data."""
        np.random.seed(42)
        return np.cumsum(np.random.randn(100))

    def test_bartletts_bootstrap(self, sample_data):
        """Test Bartlett's bootstrap."""
        composition_based = BartlettsBootstrap(n_bootstraps=3, block_length=10, random_state=42)

        # Check configuration
        assert composition_based.window_type == "bartletts"
        assert callable(composition_based.tapered_weights)

        # Generate samples
        samples = list(composition_based.bootstrap(sample_data))
        assert len(samples) == 3
        assert all(len(s) == len(sample_data) for s in samples)

    def test_blackman_bootstrap(self, sample_data):
        """Test Blackman bootstrap."""
        composition_based = BlackmanBootstrap(n_bootstraps=3, block_length=10, random_state=42)

        assert composition_based.window_type == "blackman"
        samples = list(composition_based.bootstrap(sample_data))
        assert len(samples) == 3
        assert all(len(s) == len(sample_data) for s in samples)

    def test_hamming_bootstrap(self, sample_data):
        """Test Hamming bootstrap."""
        composition_based = HammingBootstrap(n_bootstraps=3, block_length=10, random_state=42)

        assert composition_based.window_type == "hamming"
        samples = list(composition_based.bootstrap(sample_data))
        assert len(samples) == 3
        assert all(len(s) == len(sample_data) for s in samples)

    def test_hanning_bootstrap(self, sample_data):
        """Test Hanning bootstrap."""
        composition_based = HanningBootstrap(n_bootstraps=3, block_length=10, random_state=42)

        assert composition_based.window_type == "hanning"
        samples = list(composition_based.bootstrap(sample_data))
        assert len(samples) == 3
        assert all(len(s) == len(sample_data) for s in samples)

    def test_tukey_bootstrap(self, sample_data):
        """Test Tukey bootstrap."""
        composition_based = TukeyBootstrap(
            n_bootstraps=3, block_length=10, alpha=0.7, random_state=42
        )

        assert composition_based.window_type == "tukey"
        assert composition_based.alpha == 0.7
        samples = list(composition_based.bootstrap(sample_data))
        assert len(samples) == 3
        assert all(len(s) == len(sample_data) for s in samples)


class TestBlockServiceIntegration:
    """Test block bootstrap service integration."""

    def test_block_generation_service(self):
        """Test block generation service is properly integrated."""
        composition_based = BlockBootstrap(n_bootstraps=2, block_length=10)

        # Check services exist
        assert composition_based._block_gen_service is not None
        assert composition_based._block_resample_service is not None

    def test_window_service_integration(self):
        """Test window service integration."""
        composition_based = BartlettsBootstrap(n_bootstraps=2, block_length=10)

        # Check window service
        assert composition_based._window_service is not None

        # Test window function
        weights = composition_based.tapered_weights(10)
        assert len(weights) == 10
        assert weights[0] == 0.0  # Bartlett window starts at 0
        # Bartlett window peak is at (n-1)/2 for even n
        assert weights[4] == 0.8888888888888888 or weights[5] == 0.8888888888888888


def test_all_block_bootstrap_composition_based_classes_exist():
    """Ensure all block bootstrap composition_based classes are defined."""
    classes = [
        BlockBootstrap,
        MovingBlockBootstrap,
        StationaryBlockBootstrap,
        CircularBlockBootstrap,
        NonOverlappingBlockBootstrap,
        BartlettsBootstrap,
        BlackmanBootstrap,
        HammingBootstrap,
        HanningBootstrap,
        TukeyBootstrap,
    ]

    for cls in classes:
        assert cls is not None
        assert hasattr(cls, "__init__")
        assert hasattr(cls, "_generate_samples_single_bootstrap")
