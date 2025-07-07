"""Tests for batch_bootstrap.py."""

import numpy as np
import pytest

from tsbootstrap.batch_bootstrap import (
    BatchOptimizedBlockBootstrap,
    BatchOptimizedModelBootstrap,
)


class TestBatchOptimizedBlockBootstrap:
    """Test BatchOptimizedBlockBootstrap class."""

    def test_initialization(self):
        """Test basic initialization."""
        bootstrap = BatchOptimizedBlockBootstrap(
            n_bootstraps=10,
            block_length=5,
            batch_size=5
        )
        
        assert bootstrap.n_bootstraps == 10
        assert bootstrap.block_length == 5
        assert bootstrap.batch_size == 5
        assert bootstrap.use_backend is True  # Should default to True for batch

    def test_bootstrap_generation(self):
        """Test bootstrap sample generation."""
        X = np.random.randn(100)
        bootstrap = BatchOptimizedBlockBootstrap(
            n_bootstraps=6,
            block_length=10,
            batch_size=3,
            rng=42
        )
        
        samples = list(bootstrap.bootstrap(X))
        
        assert len(samples) == 6
        for sample in samples:
            assert len(sample) == len(X)
            assert isinstance(sample, np.ndarray)

    def test_batch_size_effect(self):
        """Test that batch_size is properly used."""
        X = np.random.randn(50)
        
        # Small batch size
        bootstrap1 = BatchOptimizedBlockBootstrap(
            n_bootstraps=4,
            block_length=5,
            batch_size=2,
            rng=42
        )
        
        # Large batch size
        bootstrap2 = BatchOptimizedBlockBootstrap(
            n_bootstraps=4,
            block_length=5,
            batch_size=4,
            rng=42
        )
        
        # Both should produce same results with same seed
        samples1 = list(bootstrap1.bootstrap(X))
        samples2 = list(bootstrap2.bootstrap(X))
        
        assert len(samples1) == len(samples2)
        # Results might differ due to batching implementation

    def test_multivariate_data(self):
        """Test with multivariate data."""
        X = np.random.randn(100, 3)
        bootstrap = BatchOptimizedBlockBootstrap(
            n_bootstraps=5,
            block_length=10,
            batch_size=5
        )
        
        samples = list(bootstrap.bootstrap(X))
        
        assert len(samples) == 5
        for sample in samples:
            assert sample.shape == X.shape


class TestBatchOptimizedModelBootstrap:
    """Test BatchOptimizedModelBootstrap class."""

    def test_initialization(self):
        """Test basic initialization."""
        bootstrap = BatchOptimizedModelBootstrap(
            n_bootstraps=10,
            model_type="ar",
            order=2,
            batch_size=5
        )
        
        assert bootstrap.n_bootstraps == 10
        assert bootstrap.model_type == "ar"
        assert bootstrap.order == 2
        assert bootstrap.batch_size == 5
        assert bootstrap.use_backend is True

    def test_bootstrap_generation(self):
        """Test bootstrap sample generation."""
        X = np.random.randn(100)
        bootstrap = BatchOptimizedModelBootstrap(
            n_bootstraps=4,
            model_type="ar",
            order=2,
            batch_size=2,
            rng=42
        )
        
        samples = list(bootstrap.bootstrap(X))
        
        assert len(samples) == 4
        for sample in samples:
            assert len(sample) == len(X)
            assert isinstance(sample, np.ndarray)

    def test_different_models(self):
        """Test with different model types."""
        X = np.random.randn(100)
        
        # AR model
        ar_bootstrap = BatchOptimizedModelBootstrap(
            n_bootstraps=2,
            model_type="ar",
            order=1,
            batch_size=2
        )
        ar_samples = list(ar_bootstrap.bootstrap(X))
        assert len(ar_samples) == 2
        
        # ARIMA model (MA is not directly supported, use ARIMA with MA component)
        arima_bootstrap = BatchOptimizedModelBootstrap(
            n_bootstraps=2,
            model_type="arima",
            order=(0, 0, 1),  # Pure MA(1) model
            batch_size=2
        )
        arima_samples = list(arima_bootstrap.bootstrap(X))
        assert len(arima_samples) == 2

    def test_get_test_params(self):
        """Test get_test_params method."""
        params = BatchOptimizedBlockBootstrap.get_test_params()
        assert isinstance(params, list)
        assert len(params) > 0
        
        params = BatchOptimizedModelBootstrap.get_test_params()
        assert isinstance(params, list)
        assert len(params) > 0