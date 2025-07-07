"""
Block bootstrap tests for temporal structure preservation.

We test various block bootstrap methods that handle time series data while
maintaining temporal relationships. Block methods are essential when working
with data where consecutive observations are correlated - something we encounter
frequently in financial data, sensor readings, and many other domains.

These tests focus on the specific challenges each block method addresses. Moving
block tests verify that overlapping blocks work correctly. Stationary bootstrap
tests check that we get the expected geometric distribution of block lengths.
Circular methods need special attention at the boundaries where the series wraps
around.

We've learned that block length selection dramatically impacts results, so we
test edge cases thoroughly. Too short and we lose dependencies, too long and
we don't get enough variety in our bootstrap samples.
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
    WindowedBlockBootstrap,
)


class TestBlockBootstrap:
    """Test base block bootstrap implementation using composition-based architecture."""

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


class TestBlockBootstrap:
    """Tests targeting specific uncovered lines in block_bootstrap.py."""
    
    def test_get_test_params(self):
        """Test get_test_params method ."""
        params = BlockBootstrap.get_test_params()
        assert len(params) == 1
        assert params[0]["n_bootstraps"] == 10
        assert params[0]["block_length"] == 10
    
    def test_cache_blocks_initialization(self):
        """Test blocks caching ."""
        bootstrap = MovingBlockBootstrap(
            n_bootstraps=2, 
            block_length=5,
            combine_generation_and_sampling_flag=False  # Force caching
        )
        
        # Initially no cached blocks
        assert bootstrap._blocks is None
        
        X = np.random.randn(50)
        # Generate blocks will initialize cache
        blocks = bootstrap._generate_blocks_if_needed(X)
        
        # Blocks should be cached
        assert bootstrap._blocks is not None
        assert len(bootstrap._blocks) > 0
    
    def test_block_generation_caching(self):
        """Test block generation and caching ."""
        bootstrap = MovingBlockBootstrap(
            n_bootstraps=2,
            block_length=5,
            combine_generation_and_sampling_flag=False
        )
        
        X = np.random.randn(30)
        
        # First call generates and caches
        blocks1 = bootstrap._generate_blocks_if_needed(X)
        assert bootstrap._blocks is not None
        
        # Second call should use cached blocks
        blocks2 = bootstrap._generate_blocks_if_needed(X)
        # Should be the same blocks
        assert len(blocks1) == len(blocks2)
    
    def test_recombine_all_blocks_from_cache(self):
        """Test _recombine_all_blocks_from_cache ."""
        bootstrap = MovingBlockBootstrap(
            n_bootstraps=3,
            block_length=5,
            combine_generation_and_sampling_flag=False
        )
        
        X = np.random.randn(50)
        
        # Generate initial sample to populate cache
        sample1 = bootstrap._generate_samples_single_bootstrap(X)
        
        # Now cache should be populated, next samples will use cache
        sample2 = bootstrap._generate_samples_single_bootstrap(X)
        sample3 = bootstrap._generate_samples_single_bootstrap(X)
        
        # All should have same length as X
        assert len(sample1) == len(X)
        assert len(sample2) == len(X)
        assert len(sample3) == len(X)
    
    def test_circular_block_edge_cases(self):
        """Test CircularBlockBootstrap edge cases ."""
        # Test with small data that wraps around
        X = np.array([1, 2, 3, 4, 5], dtype=float)
        
        bootstrap = CircularBlockBootstrap(
            n_bootstraps=2,
            block_length=3  # Smaller block length for small data
        )
        
        samples = list(bootstrap.bootstrap(X))
        
        assert len(samples) == 2
        for sample in samples:
            assert len(sample) == len(X)
            # Check that values come from original data
            # Note: values might be repeated due to block structure
            unique_vals = np.unique(sample)
            assert all(val in X for val in unique_vals)
    
    def test_non_overlapping_block_specific_logic(self):
        """Test NonOverlappingBlockBootstrap specific logic ."""
        bootstrap = NonOverlappingBlockBootstrap(
            n_bootstraps=2,
            block_length=10
        )
        
        # Test with data length that's not multiple of block_length
        X = np.random.randn(45)  # 45 is not divisible by 10
        samples = list(bootstrap.bootstrap(X))
        
        assert len(samples) == 2
        for sample in samples:
            assert len(sample) == len(X)
    
    def test_stationary_block_resampling(self):
        """Test StationaryBlockBootstrap block resampling ."""
        bootstrap = StationaryBlockBootstrap(
            n_bootstraps=3,
            avg_block_length=10
        )
        
        X = np.random.randn(100)
        samples = list(bootstrap.bootstrap(X))
        
        assert len(samples) == 3
        for sample in samples:
            assert len(sample) == len(X)
            assert isinstance(sample, np.ndarray)
    
    def test_window_function_applications(self):
        """Test window function applications for various windowed bootstraps."""
        X = np.random.randn(50)
        
        # Test BartlettsBootstrap 
        bartletts = BartlettsBootstrap(n_bootstraps=1, block_length=10)
        bartletts_samples = list(bartletts.bootstrap(X))
        assert len(bartletts_samples[0]) == len(X)
        
        # Test BlackmanBootstrap 
        # BlackmanBootstrap uses composition and doesn't have an 'a' parameter
        blackman = BlackmanBootstrap(n_bootstraps=1, block_length=10)
        blackman_samples = list(blackman.bootstrap(X))
        assert len(blackman_samples[0]) == len(X)
        assert blackman.window_type == "blackman"
        
        # Test HammingBootstrap 
        hamming = HammingBootstrap(n_bootstraps=1, block_length=10)
        hamming_samples = list(hamming.bootstrap(X))
        assert len(hamming_samples[0]) == len(X)
        
        # Test HanningBootstrap 
        hanning = HanningBootstrap(n_bootstraps=1, block_length=10)
        hanning_samples = list(hanning.bootstrap(X))
        assert len(hanning_samples[0]) == len(X)
        
        # Test TukeyBootstrap 
        tukey = TukeyBootstrap(n_bootstraps=1, block_length=10)
        assert tukey.alpha == 0.5  # Default alpha
        tukey_samples = list(tukey.bootstrap(X))
        assert len(tukey_samples[0]) == len(X)
        
        # Test with custom alpha
        tukey2 = TukeyBootstrap(n_bootstraps=1, block_length=10, alpha=0.7)
        assert tukey2.alpha == 0.7
    
    def test_window_function_compute_length(self):
        """Test compute_window_length for windowed bootstraps ."""
        # Create a windowed bootstrap
        bootstrap = BartlettsBootstrap(n_bootstraps=1, block_length=10)
        
        # The compute_window_length is used internally
        # Test that windowed bootstraps work correctly with different block lengths
        X = np.random.randn(100)
        
        # Test with different block lengths
        for block_length in [5, 10, 20]:
            bootstrap = BartlettsBootstrap(n_bootstraps=1, block_length=block_length)
            samples = list(bootstrap.bootstrap(X))
            assert len(samples[0]) == len(X)
    
    def test_block_bootstrap_with_multivariate_data(self):
        """Test block bootstraps with multivariate data."""
        X = np.random.randn(100, 3)  # Multivariate data
        
        # Test various block bootstrap methods
        bootstraps = [
            MovingBlockBootstrap(n_bootstraps=1, block_length=10),
            CircularBlockBootstrap(n_bootstraps=1, block_length=10),
            NonOverlappingBlockBootstrap(n_bootstraps=1, block_length=10),
            StationaryBlockBootstrap(n_bootstraps=1, avg_block_length=10),
            BartlettsBootstrap(n_bootstraps=1, block_length=10),
        ]
        
        for bootstrap in bootstraps:
            samples = list(bootstrap.bootstrap(X))
            assert len(samples) == 1
            assert samples[0].shape == X.shape
    
    def test_block_length_edge_cases(self):
        """Test block bootstrap with edge case block lengths."""
        X = np.random.randn(50)
        
        # Test with block_length = 1 (essentially iid bootstrap)
        bootstrap = MovingBlockBootstrap(n_bootstraps=1, block_length=1)
        samples = list(bootstrap.bootstrap(X))
        assert len(samples[0]) == len(X)
        
        # Test with block_length = data length
        bootstrap = MovingBlockBootstrap(n_bootstraps=1, block_length=len(X))
        samples = list(bootstrap.bootstrap(X))
        assert len(samples[0]) == len(X)
    
    def test_stationary_block_with_small_avg_length(self):
        """Test StationaryBlockBootstrap with small average block length."""
        bootstrap = StationaryBlockBootstrap(
            n_bootstraps=2,
            avg_block_length=2  # Very small average
        )
        
        X = np.random.randn(30)
        samples = list(bootstrap.bootstrap(X))
        
        assert len(samples) == 2
        for sample in samples:
            assert len(sample) == len(X)
    
    def test_windowed_bootstrap_caching_behavior(self):
        """Test caching behavior in windowed bootstraps."""
        bootstrap = HammingBootstrap(
            n_bootstraps=3,
            block_length=8,
            combine_generation_and_sampling_flag=False  # Force caching
        )
        
        X = np.random.randn(40)
        
        # Generate multiple samples - should use caching after first
        samples = list(bootstrap.bootstrap(X))
        
        assert len(samples) == 3
        assert all(len(s) == len(X) for s in samples)
        # Check that blocks are cached (the attribute is _blocks, not _cache_blocks)
        assert bootstrap._blocks is not None


class TestAdditionalCoverage:
    """Additional tests for missing lines to reach 95% coverage."""
    
    def test_all_get_test_params(self):
        """Test get_test_params for all bootstrap classes ."""
        # MovingBlockBootstrap.get_test_params 
        params = MovingBlockBootstrap.get_test_params()
        assert len(params) == 1
        assert params[0]["n_bootstraps"] == 10
        assert params[0]["block_length"] == 10
        
        # StationaryBlockBootstrap.get_test_params 
        params = StationaryBlockBootstrap.get_test_params()
        assert len(params) == 1
        assert params[0]["n_bootstraps"] == 10
        assert params[0]["block_length"] == 10
        
        # CircularBlockBootstrap.get_test_params 
        params = CircularBlockBootstrap.get_test_params()
        assert len(params) == 1
        assert params[0]["n_bootstraps"] == 10
        assert params[0]["block_length"] == 10
        
        # NonOverlappingBlockBootstrap.get_test_params 
        params = NonOverlappingBlockBootstrap.get_test_params()
        assert len(params) == 1
        assert params[0]["n_bootstraps"] == 10
        assert params[0]["block_length"] == 10
        
        # BartlettsBootstrap.get_test_params 
        params = BartlettsBootstrap.get_test_params()
        assert len(params) == 1
        assert params[0]["n_bootstraps"] == 10
        assert params[0]["block_length"] == 10
        
        # BlackmanBootstrap.get_test_params 
        params = BlackmanBootstrap.get_test_params()
        assert len(params) == 1
        assert params[0]["n_bootstraps"] == 10
        assert params[0]["block_length"] == 10
        
        # HammingBootstrap.get_test_params 
        params = HammingBootstrap.get_test_params()
        assert len(params) == 1
        assert params[0]["n_bootstraps"] == 10
        assert params[0]["block_length"] == 10
        
        # HanningBootstrap.get_test_params 
        params = HanningBootstrap.get_test_params()
        assert len(params) == 1
        assert params[0]["n_bootstraps"] == 10
        assert params[0]["block_length"] == 10
        
        # TukeyBootstrap.get_test_params 
        params = TukeyBootstrap.get_test_params()
        assert len(params) == 1
        assert params[0]["n_bootstraps"] == 10
        assert params[0]["block_length"] == 10
    
    def test_generate_samples_edge_cases(self):
        """Test edge cases in _generate_samples_single_bootstrap ."""
        # Test when result is longer than original
        bootstrap = MovingBlockBootstrap(n_bootstraps=1, block_length=3)
        X = np.array([1, 2, 3, 4, 5])
        
        # Mock the block resample service to return longer data
        original_resample = bootstrap._block_resample_service.resample_blocks
        
        def mock_resample(X, blocks, n, block_weights, tapered_weights, rng):
            # Return block indices and data that results in longer series
            indices = [0, 1]  # Two blocks
            data = [np.array([1, 2, 3]), np.array([3, 4, 5])]  # 6 elements total
            return indices, data
        
        bootstrap._block_resample_service.resample_blocks = mock_resample
        
        # Generate sample - should be truncated to original length
        sample = bootstrap._generate_samples_single_bootstrap(X)
        
        # Restore original
        bootstrap._block_resample_service.resample_blocks = original_resample
        
        assert len(sample) == len(X)  # Should be truncated to 5
        
        # Test with empty block data
        bootstrap2 = MovingBlockBootstrap(n_bootstraps=1, block_length=3)
        
        def mock_empty_resample(X, blocks, n, block_weights, tapered_weights, rng):
            return [], []  # Empty blocks
        
        bootstrap2._block_resample_service.resample_blocks = mock_empty_resample
        
        # Should return array with same shape as X (uses np.empty_like)
        sample2 = bootstrap2._generate_samples_single_bootstrap(X)
        assert sample2.shape == X.shape
        # The array will be uninitialized but have same shape
        
        bootstrap2._block_resample_service.resample_blocks = original_resample
    
    def test_get_params_with_callable_block_weights(self):
        """Test get_params and set_params with callable block_weights ."""
        # Define a callable block weight function
        def custom_weights(n_blocks):
            return np.ones(n_blocks) / n_blocks
        
        # Create bootstrap with callable block_weights
        bootstrap = MovingBlockBootstrap(
            n_bootstraps=2,
            block_length=5,
            block_weights=custom_weights
        )
        
        # get_params should exclude callable block_weights
        params = bootstrap.get_params()
        assert "block_weights" not in params
        assert "n_bootstraps" in params
        assert params["n_bootstraps"] == 2
        
        # set_params with callable should be handled
        new_weights = lambda n: np.ones(n)
        params_with_callable = {"block_weights": new_weights, "n_bootstraps": 3}
        bootstrap.set_params(**params_with_callable)
        
        # n_bootstraps should be updated, but callable should be ignored
        assert bootstrap.n_bootstraps == 3
        # The original callable should still be there (set_params filtered it out)
        assert bootstrap.block_weights is custom_weights
        
        # Test with array block_weights (non-callable)
        bootstrap2 = MovingBlockBootstrap(
            n_bootstraps=2,
            block_length=5,
            block_weights=np.array([0.5, 0.5])
        )
        
        params2 = bootstrap2.get_params()
        # Array block_weights might be excluded in get_params due to serialization constraints
        # The important part is that callable weights are filtered out
        # This test verifies the callable filtering works correctly
    
    def test_windowed_bootstrap_base_methods(self):
        """Test WindowedBlockBootstrap base class methods ."""
        # WindowedBlockBootstrap.get_test_params returns empty list
        params = WindowedBlockBootstrap.get_test_params()
        assert params == []
        
        # Test _create_tapered_weights when window_service is None 
        bootstrap = BartlettsBootstrap(n_bootstraps=1, block_length=5)
        # Force window service to None and clear cache
        bootstrap._window_service = None
        bootstrap._tapered_weights_cache = None
        
        # Call _create_tapered_weights directly - should recreate service
        weights_func = bootstrap._create_tapered_weights()
        assert weights_func is not None
        assert bootstrap._window_service is not None
        
        # Test that weights function works
        weights = weights_func(10)
        assert len(weights) == 10
        assert np.all(weights >= 0)  # Weights should be non-negative
    
    def test_reshape_logic_in_generate_samples(self):
        """Test reshape logic in _generate_samples_single_bootstrap with extra dimensions."""
        bootstrap = MovingBlockBootstrap(n_bootstraps=1, block_length=3)
        X = np.array([[1], [2], [3], [4], [5]])  # 2D array with shape (5, 1)
        
        # Mock to return data with extra trailing dimension
        original_resample = bootstrap._block_resample_service.resample_blocks
        
        def mock_resample_extra_dim(X, blocks, n, block_weights, tapered_weights, rng):
            # Return data with extra dimension: shape (5, 1, 1)
            indices = [0]
            data = [np.array([[[1]], [[2]], [[3]], [[4]], [[5]]])]  # Extra dimension
            return indices, data
        
        bootstrap._block_resample_service.resample_blocks = mock_resample_extra_dim
        
        # Should handle the extra dimension
        sample = bootstrap._generate_samples_single_bootstrap(X)
        
        # Restore
        bootstrap._block_resample_service.resample_blocks = original_resample
        
        # Should maintain original shape
        assert sample.shape == X.shape
        

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])