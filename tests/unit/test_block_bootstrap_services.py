"""Tests for block_bootstrap_services.py."""

import numpy as np
import pytest

from tsbootstrap.services.block_bootstrap_services import (
    BlockGenerationService,
    BlockResamplingService,
    WindowFunctionService,
    MarkovBootstrapService,
    DistributionBootstrapService,
    StatisticPreservingService,
)


class TestBlockBootstrapServices:
    """Tests targeting specific uncovered lines in block_bootstrap_services.py."""
    
    def test_block_generation_length_validation(self):
        """Test block_length validation error ."""
        service = BlockGenerationService()
        X = np.random.randn(10)  # Small array
        
        # Test with block_length greater than array size
        with pytest.raises(ValueError, match="block_length cannot be greater than the size of the input array"):
            service.generate_blocks(X, block_length=15)  # 15 > 10
        
        # Test with block_length equal to array size (should work)
        blocks = service.generate_blocks(X, block_length=10)
        assert len(blocks) > 0
        
        # Test with valid block_length
        blocks = service.generate_blocks(X, block_length=5)
        assert len(blocks) > 0
    
    def test_markov_bootstrap_service(self):
        """Test MarkovBootstrapService ."""
        # Test initialization 
        service = MarkovBootstrapService()
        assert service.transition_matrix is None
        
        # Test fit_markov_model 
        X = np.random.randn(50)
        order = 3
        service.fit_markov_model(X, order=order)
        
        # Should have set transition_matrix 
        assert service.transition_matrix is not None
        assert service.transition_matrix.shape == (order, order)
        assert np.allclose(service.transition_matrix, np.eye(order))
        
        # Test generate_markov_sample 
        rng = np.random.default_rng(42)
        n_samples = 20
        sample = service.generate_markov_sample(n_samples, rng)
        
        assert isinstance(sample, np.ndarray)
        assert len(sample) == n_samples
    
    def test_distribution_bootstrap_service(self):
        """Test DistributionBootstrapService ."""
        # Test initialization 
        service = DistributionBootstrapService()
        assert service.distribution is None
        
        # Test fit_distribution 
        residuals = np.random.randn(100)
        service.fit_distribution(residuals)
        
        # Should have set distribution 
        assert service.distribution is not None
        assert "mean" in service.distribution
        assert "std" in service.distribution
        assert service.distribution["mean"] == np.mean(residuals)
        assert service.distribution["std"] == np.std(residuals)
        
        # Test sample_from_distribution with fitted distribution 
        rng = np.random.default_rng(42)
        n_samples = 25
        sample = service.sample_from_distribution(n_samples, rng)
        
        assert isinstance(sample, np.ndarray)
        assert len(sample) == n_samples
        
        # Test sample_from_distribution without fitted distribution
        service2 = DistributionBootstrapService()
        sample2 = service2.sample_from_distribution(n_samples, rng)
        assert isinstance(sample2, np.ndarray)
        assert len(sample2) == n_samples
    
    def test_statistic_preserving_service(self):
        """Test StatisticPreservingService ."""
        # Test initialization 
        service = StatisticPreservingService()
        assert service.target_statistics == {}
        
        # Test compute_statistics 
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = service.compute_statistics(X)
        
        assert isinstance(stats, dict)
        assert "mean" in stats
        assert "variance" in stats
        assert "skewness" in stats
        assert "kurtosis" in stats
        assert stats["mean"] == np.mean(X)
        assert stats["variance"] == np.var(X)
        assert stats["skewness"] == 0.0  # Placeholder
        assert stats["kurtosis"] == 3.0  # Placeholder
        
        # Test adjust_sample with valid standard deviation 
        sample = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        target_stats = {"mean": 10.0, "variance": 4.0}
        
        adjusted_sample = service.adjust_sample(sample, target_stats)
        
        assert isinstance(adjusted_sample, np.ndarray)
        assert len(adjusted_sample) == len(sample)
        # Check that the adjustment actually changed the sample
        assert not np.array_equal(sample, adjusted_sample)
        # Check that the mean is close to target
        assert abs(np.mean(adjusted_sample) - target_stats["mean"]) < 1e-10
        
        # Test adjust_sample with zero standard deviation (edge case)
        constant_sample = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        adjusted_constant = service.adjust_sample(constant_sample, target_stats)
        
        # Should return the original sample when std is 0
        assert np.array_equal(constant_sample, adjusted_constant)
    
    def test_additional_coverage_for_remaining_lines(self):
        """Test additional scenarios to reach closer to 95% coverage."""
        # Test BlockGenerationService with various parameters
        service = BlockGenerationService()
        X = np.random.randn(20)
        
        # Test with wrap_around_flag
        blocks = service.generate_blocks(X, block_length=5, wrap_around_flag=True)
        assert len(blocks) > 0
        
        # Test with overlap
        blocks = service.generate_blocks(X, block_length=5, overlap_flag=True, overlap_length=2)
        assert len(blocks) > 0
    
    def test_edge_cases_and_error_conditions(self):
        """Test edge cases and error conditions for all services."""
        # Test BlockGenerationService with edge cases
        service = BlockGenerationService()
        
        # Small array with minimum size (3 is the minimum for BlockGenerator)
        X_small = np.array([1, 2, 3])
        blocks = service.generate_blocks(X_small, block_length=2)
        assert len(blocks) > 0
        
        # Test with None block_length (should use default)
        X = np.random.randn(100)
        blocks = service.generate_blocks(X, block_length=None)
        assert len(blocks) > 0
        
        # Test MarkovBootstrapService edge cases
        markov_service = MarkovBootstrapService()
        
        # Test with different orders
        for order in [1, 2, 5]:
            markov_service.fit_markov_model(X, order=order)
            assert markov_service.transition_matrix.shape == (order, order)
        
        # Test DistributionBootstrapService edge cases
        dist_service = DistributionBootstrapService()
        
        # Test with constant residuals
        constant_residuals = np.ones(50)
        dist_service.fit_distribution(constant_residuals)
        assert dist_service.distribution["std"] == 0.0
        
        # Test StatisticPreservingService edge cases
        stat_service = StatisticPreservingService()
        
        # Test with single-value array
        single_val = np.array([42.0])
        stats = stat_service.compute_statistics(single_val)
        assert stats["mean"] == 42.0
        assert stats["variance"] == 0.0
        
        # Test adjust_sample with empty target_stats
        sample = np.array([1, 2, 3])
        adjusted = stat_service.adjust_sample(sample, {})
        # Should use default values (variance=1.0, mean=0.0)
        assert not np.array_equal(sample, adjusted)
    
    def test_block_resampling_service_comprehensive(self):
        """Test BlockResamplingService ."""
        # Test initialization 
        service = BlockResamplingService()
        assert service._block_resampler is None
        
        # Test resample_blocks method 
        X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        blocks = [X[:3], X[3:6], X[6:9]]  # Three blocks
        
        # Test basic resampling
        block_indices, block_data = service.resample_blocks(X, blocks, n=12)
        
        assert isinstance(block_indices, list)
        assert isinstance(block_data, list)
        assert len(block_indices) > 0
        assert len(block_data) > 0
        
        # Test with custom RNG
        rng = np.random.default_rng(42)
        block_indices, block_data = service.resample_blocks(X, blocks, n=12, rng=rng)
        assert len(block_indices) > 0
        
        # Test with block weights
        block_weights = np.array([0.5, 0.3, 0.2])
        block_indices, block_data = service.resample_blocks(
            X, blocks, n=12, block_weights=block_weights
        )
        assert len(block_indices) > 0
        
        # Test with tapered weights function
        def tapered_weights_func(size):
            # Function receives the size, not block data
            return np.ones(size) * 0.8
        
        block_indices, block_data = service.resample_blocks(
            X, blocks, n=12, tapered_weights=tapered_weights_func
        )
        assert len(block_indices) > 0
        
        # Test with all parameters
        block_indices, block_data = service.resample_blocks(
            X, blocks, n=15, 
            block_weights=block_weights, 
            tapered_weights=tapered_weights_func,
            rng=rng
        )
        assert len(block_indices) > 0
        assert len(block_data) > 0
    
    def test_window_function_service_comprehensive(self):
        """Test WindowFunctionService ."""
        service = WindowFunctionService()
        
        # Test all static window methods
        block_length = 10
        
        # Test bartletts_window 
        bartlett_window = service.bartletts_window(block_length)
        assert isinstance(bartlett_window, np.ndarray)
        assert len(bartlett_window) == block_length
        np.testing.assert_array_equal(bartlett_window, np.bartlett(block_length))
        
        # Test blackman_window 
        blackman_window = service.blackman_window(block_length)
        assert isinstance(blackman_window, np.ndarray)
        assert len(blackman_window) == block_length
        np.testing.assert_array_equal(blackman_window, np.blackman(block_length))
        
        # Test hamming_window 
        hamming_window = service.hamming_window(block_length)
        assert isinstance(hamming_window, np.ndarray)
        assert len(hamming_window) == block_length
        np.testing.assert_array_equal(hamming_window, np.hamming(block_length))
        
        # Test hanning_window 
        hanning_window = service.hanning_window(block_length)
        assert isinstance(hanning_window, np.ndarray)
        assert len(hanning_window) == block_length
        np.testing.assert_array_equal(hanning_window, np.hanning(block_length))
        
        # Test tukey_window 
        tukey_window = service.tukey_window(block_length, alpha=0.5)
        assert isinstance(tukey_window, np.ndarray)
        assert len(tukey_window) == block_length
        
        # Test tukey_window with different alpha
        tukey_window_alpha = service.tukey_window(block_length, alpha=0.25)
        assert isinstance(tukey_window_alpha, np.ndarray)
        assert len(tukey_window_alpha) == block_length
        
        # Test get_window_function method 
        window_types = ["bartletts", "blackman", "hamming", "hanning", "tukey"]
        
        for window_type in window_types:
            window_func = service.get_window_function(window_type)
            assert callable(window_func)
            
            # Test that the function works
            if window_type == "tukey":
                # Tukey requires alpha parameter
                window = window_func(block_length, alpha=0.5)
            else:
                window = window_func(block_length)
            
            assert isinstance(window, np.ndarray)
            assert len(window) == block_length
        
        # Test window function mapping 
        assert service.get_window_function("bartletts") == service.bartletts_window
        assert service.get_window_function("blackman") == service.blackman_window
        assert service.get_window_function("hamming") == service.hamming_window
        assert service.get_window_function("hanning") == service.hanning_window
        assert service.get_window_function("tukey") == service.tukey_window
        
        # Test invalid window type 
        with pytest.raises(ValueError, match="Window type 'invalid' not recognized"):
            service.get_window_function("invalid")
        
        with pytest.raises(ValueError, match="Available window functions"):
            service.get_window_function("unknown")
        
        with pytest.raises(ValueError, match="For custom windows, extend WindowFunctionService"):
            service.get_window_function("nonexistent")
    
    def test_block_generation_service_comprehensive_parameters(self):
        """Test BlockGenerationService with comprehensive parameter coverage."""
        service = BlockGenerationService()
        X = np.random.randn(50)
        
        # Test with block_length_distribution parameter
        blocks = service.generate_blocks(
            X, 
            block_length=8, 
            block_length_distribution="exponential"
        )
        assert len(blocks) > 0
        
        # Test with min_block_length parameter
        blocks = service.generate_blocks(
            X, 
            block_length=10, 
            min_block_length=3
        )
        assert len(blocks) > 0
        
        # Test with all parameters combined
        rng = np.random.default_rng(42)
        blocks = service.generate_blocks(
            X,
            block_length=12,
            block_length_distribution="uniform",
            wrap_around_flag=True,
            overlap_flag=True,
            overlap_length=3,
            min_block_length=4,
            rng=rng
        )
        assert len(blocks) > 0
        
        # Test default block_length calculation (sqrt of array length)
        X_large = np.random.randn(144)  # sqrt(144) = 12
        blocks = service.generate_blocks(X_large, block_length=None)
        assert len(blocks) > 0
    
    def test_service_integration_workflow(self):
        """Test integration between all services."""
        # Initialize all services
        block_gen = BlockGenerationService()
        block_resample = BlockResamplingService()
        window_func = WindowFunctionService()
        markov = MarkovBootstrapService()
        dist = DistributionBootstrapService()
        stat_preserve = StatisticPreservingService()
        
        # Generate sample data
        np.random.seed(42)
        X = np.random.randn(60)
        
        # Test workflow: generate blocks
        blocks = block_gen.generate_blocks(X, block_length=10)
        assert len(blocks) > 0
        
        # Test workflow: resample blocks
        block_indices, block_data = block_resample.resample_blocks(X, blocks, n=60)
        assert len(block_indices) > 0
        assert len(block_data) > 0
        
        # Test workflow: apply window function
        window = window_func.get_window_function("hanning")
        weights = window(10)
        assert len(weights) == 10
        
        # Test workflow: use markov bootstrap
        markov.fit_markov_model(X, order=2)
        markov_sample = markov.generate_markov_sample(30, np.random.default_rng(42))
        assert len(markov_sample) == 30
        
        # Test workflow: use distribution bootstrap
        dist.fit_distribution(X)
        dist_sample = dist.sample_from_distribution(25, np.random.default_rng(42))
        assert len(dist_sample) == 25
        
        # Test workflow: preserve statistics
        original_stats = stat_preserve.compute_statistics(X)
        adjusted_sample = stat_preserve.adjust_sample(X[:20], original_stats)
        assert len(adjusted_sample) == 20
        
        # Verify all services worked together
        assert markov.transition_matrix is not None
        assert dist.distribution is not None
        assert len(original_stats) == 4


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])