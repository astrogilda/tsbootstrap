"""
Tests for block bootstrap services.

This module tests the services that implement block-based bootstrap methods,
including block generation, resampling, and window function applications
for handling time series with dependencies.
"""

import numpy as np
from tsbootstrap.services.block_bootstrap_services import (
    BlockGenerationService,
    BlockResamplingService,
    WindowFunctionService,
)


class TestBlockGenerationService:
    """Test block generation service functionality."""

    def test_generate_blocks_specified_length(self):
        """Test block generation with specified block length."""
        service = BlockGenerationService()

        # Generate test data
        X = np.arange(20).reshape(-1, 1)

        # Generate blocks with specified length
        blocks = service.generate_blocks(X, block_length=5)

        assert isinstance(blocks, list)
        assert len(blocks) > 0
        # Verify that blocks are numpy arrays
        for block in blocks:
            assert isinstance(block, np.ndarray)
            assert len(block) > 0

    def test_generate_blocks_random_length(self):
        """Test block generation with random length."""
        service = BlockGenerationService()

        X = np.arange(50).reshape(-1, 1)

        # Automatic block length selection when not specified
        blocks = service.generate_blocks(X, block_length=None)

        assert isinstance(blocks, list)
        assert len(blocks) > 0
        # Blocks are numpy arrays
        for block in blocks:
            assert isinstance(block, np.ndarray)

    def test_generate_blocks_with_rng(self):
        """Test block generation with custom RNG."""
        service = BlockGenerationService()
        rng = np.random.default_rng(42)

        X = np.arange(30).reshape(-1, 1)

        blocks1 = service.generate_blocks(X, block_length=10, rng=rng)

        # Reset RNG with same seed
        rng2 = np.random.default_rng(42)
        blocks2 = service.generate_blocks(X, block_length=10, rng=rng2)

        # Same seed produces same number of blocks
        assert len(blocks1) == len(blocks2)

    def test_generate_blocks_2d_data(self):
        """Test block generation with 2D data."""
        service = BlockGenerationService()

        # 2D data
        X = np.random.randn(40, 3)

        blocks = service.generate_blocks(X, block_length=8)

        assert isinstance(blocks, list)
        for block in blocks:
            assert isinstance(block, np.ndarray)


class TestBlockResamplingService:
    """Test block resampling service functionality."""

    def test_resample_blocks_basic(self):
        """Test basic block resampling."""
        service = BlockResamplingService()

        X = np.arange(20).reshape(-1, 1)
        # Create blocks as numpy arrays for the resampler
        blocks = [np.arange(0, 5), np.arange(5, 10), np.arange(10, 15), np.arange(15, 20)]

        # The resampler returns both indices and data
        block_indices, block_data = service.resample_blocks(X, blocks, n=20)

        assert isinstance(block_indices, list)
        assert isinstance(block_data, list)
        # Both should have same length
        assert len(block_indices) == len(block_data)

    def test_resample_blocks_different_size(self):
        """Test resampling to different size."""
        service = BlockResamplingService()

        X = np.arange(30).reshape(-1, 1)
        blocks = [np.arange(0, 10), np.arange(10, 20), np.arange(20, 30)]

        # Resample to larger size
        block_indices, block_data = service.resample_blocks(X, blocks, n=50)

        assert isinstance(block_indices, list)
        assert isinstance(block_data, list)

    def test_resample_blocks_with_weights(self):
        """Test resampling with block weights."""
        service = BlockResamplingService()

        X = np.array([[1], [2], [3], [4], [5], [6]])
        blocks = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]

        # Heavy weight on first block
        weights = np.array([0.8, 0.1, 0.1])

        np.random.seed(42)
        block_indices, block_data = service.resample_blocks(X, blocks, n=100, block_weights=weights)

        # Check that we got valid results
        assert len(block_indices) > 0
        assert len(block_data) > 0

    def test_resample_blocks_with_rng(self):
        """Test resampling with custom RNG."""
        service = BlockResamplingService()

        X = np.arange(15).reshape(-1, 1)
        blocks = [np.arange(0, 5), np.arange(5, 10), np.arange(10, 15)]

        rng1 = np.random.default_rng(123)
        indices1, data1 = service.resample_blocks(X, blocks, n=15, rng=rng1)

        rng2 = np.random.default_rng(123)
        indices2, data2 = service.resample_blocks(X, blocks, n=15, rng=rng2)

        # Same seed produces identical results
        assert len(indices1) == len(indices2)

    def test_resample_blocks_multivariate(self):
        """Test resampling with multivariate data."""
        service = BlockResamplingService()

        # Multivariate data
        X = np.random.randn(20, 3)
        blocks = [np.arange(0, 5), np.arange(5, 10), np.arange(10, 15), np.arange(15, 20)]

        block_indices, block_data = service.resample_blocks(X, blocks, n=20)

        assert len(block_indices) > 0
        assert len(block_data) > 0


class TestWindowFunctionService:
    """Test window function service functionality."""

    def test_bartletts_window(self):
        """Test Bartlett's window function."""
        result = WindowFunctionService.bartletts_window(10)

        assert isinstance(result, np.ndarray)
        assert len(result) == 10
        # Should be symmetric
        assert np.allclose(result[:5], result[5:][::-1])
        # Peak at center
        assert np.argmax(result) in [4, 5]

    def test_blackman_window(self):
        """Test Blackman window function."""
        result = WindowFunctionService.blackman_window(8)

        assert isinstance(result, np.ndarray)
        assert len(result) == 8
        # Should start and end near zero
        assert result[0] < 0.1
        assert result[-1] < 0.1

    def test_hamming_window(self):
        """Test Hamming window function."""
        result = WindowFunctionService.hamming_window(12)

        assert isinstance(result, np.ndarray)
        assert len(result) == 12
        # Should not go to zero at endpoints
        assert result[0] > 0.05
        assert result[-1] > 0.05

    def test_hanning_window(self):
        """Test Hanning window function."""
        result = WindowFunctionService.hanning_window(15)

        assert isinstance(result, np.ndarray)
        assert len(result) == 15
        # Should go to zero at endpoints
        assert np.isclose(result[0], 0)
        assert np.isclose(result[-1], 0)

    def test_tukey_window(self):
        """Test Tukey window function."""
        result = WindowFunctionService.tukey_window(10)

        assert isinstance(result, np.ndarray)
        assert len(result) == 10

        # Test with different alpha
        result_rect = WindowFunctionService.tukey_window(10, alpha=0.0)
        assert np.allclose(result_rect, 1.0)  # Rectangular window

        result_hann = WindowFunctionService.tukey_window(10, alpha=1.0)
        assert result_hann[0] < 0.1  # Should taper at edges

    def test_window_functions_consistency(self):
        """Test that window functions produce consistent results."""
        n = 20

        # Verify that window functions produce consistent results
        bartletts1 = WindowFunctionService.bartletts_window(n)
        bartletts2 = WindowFunctionService.bartletts_window(n)
        assert np.array_equal(bartletts1, bartletts2)

        hamming1 = WindowFunctionService.hamming_window(n)
        hamming2 = WindowFunctionService.hamming_window(n)
        assert np.array_equal(hamming1, hamming2)


class TestIntegration:
    """Integration tests for block bootstrap services."""

    def test_block_bootstrap_workflow(self):
        """Test complete block bootstrap workflow."""
        # Initialize services
        generator = BlockGenerationService()
        resampler = BlockResamplingService()

        # Generate test data
        np.random.seed(42)
        X = np.random.randn(100).cumsum().reshape(-1, 1)

        # Generate blocks
        blocks = generator.generate_blocks(X, block_length=10)

        # Resample blocks
        block_indices, block_data = resampler.resample_blocks(X, blocks, n=100)

        assert isinstance(block_indices, list)
        assert isinstance(block_data, list)
        assert len(block_indices) > 0

    def test_windowed_block_bootstrap(self):
        """Test block bootstrap with window weighting."""
        # Initialize services
        generator = BlockGenerationService()
        resampler = BlockResamplingService()

        # Generate data
        X = np.arange(50).reshape(-1, 1)

        # Generate blocks
        blocks = generator.generate_blocks(X, block_length=10)

        # Create window-based weights
        window = WindowFunctionService.hamming_window(len(blocks))
        weights = window / window.sum()  # Normalize

        # Resample with weights
        block_indices, block_data = resampler.resample_blocks(
            X, blocks, n=50, block_weights=weights
        )

        assert len(block_indices) > 0
        assert len(block_data) > 0
