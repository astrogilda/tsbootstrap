"""Test the refactored BaseTimeSeriesBootstrap class."""

import numpy as np
import pytest
from sklearn.base import clone
from tsbootstrap.base_bootstrap import BaseTimeSeriesBootstrap


class ConcreteBootstrap(BaseTimeSeriesBootstrap):
    """Concrete implementation for testing."""

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y: np.ndarray = None
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Simple implementation that returns shuffled indices."""
        n_samples = X.shape[0]
        indices = self.rng.choice(n_samples, size=n_samples, replace=True)
        return indices, [X[indices]]


class TestBaseTimeSeriesBootstrap:
    """Test suite for refactored base bootstrap class."""

    def test_initialization(self):
        """Test basic initialization."""
        bootstrap = ConcreteBootstrap(n_bootstraps=5, rng=42)
        assert bootstrap.n_bootstraps == 5
        assert isinstance(bootstrap.rng, np.random.Generator)

    def test_computed_fields(self):
        """Test computed fields work correctly."""
        bootstrap = ConcreteBootstrap(n_bootstraps=5)
        assert bootstrap.parallel_capable is False
        assert bootstrap.is_fitted is False

        bootstrap_parallel = ConcreteBootstrap(n_bootstraps=15)
        assert bootstrap_parallel.parallel_capable is True

    def test_sklearn_compatibility(self):
        """Test sklearn get_params/set_params work correctly."""
        bootstrap = ConcreteBootstrap(n_bootstraps=5, rng=42)

        # Test get_params
        params = bootstrap.get_params()
        assert params["n_bootstraps"] == 5
        assert params["rng"] == 42  # Should return original value

        # Test set_params
        bootstrap.set_params(n_bootstraps=10)
        assert bootstrap.n_bootstraps == 10

        # Test clone
        bootstrap_clone = clone(bootstrap)
        assert bootstrap_clone.n_bootstraps == 10
        assert bootstrap_clone is not bootstrap

    def test_numpy_serialization(self):
        """Test numpy array serialization."""
        bootstrap = ConcreteBootstrap()
        X = np.array([[1, 2], [3, 4], [5, 6]])

        # Validate array input
        X_validated = bootstrap._validate_array_input(X, "X")
        assert isinstance(X_validated, np.ndarray)

        # Test 2D conversion
        X_1d = np.array([1, 2, 3])
        X_2d = bootstrap._ensure_2d(X_1d)
        assert X_2d.shape == (3, 1)

    def test_validation_mixin(self):
        """Test validation methods from mixin."""
        bootstrap = ConcreteBootstrap()

        # Test positive int validation
        assert bootstrap._validate_positive_int(5, "test") == 5
        with pytest.raises(ValueError):
            bootstrap._validate_positive_int(-1, "test")

        # Test probability validation
        assert bootstrap._validate_probability(0.5, "test") == 0.5
        with pytest.raises(ValueError):
            bootstrap._validate_probability(1.5, "test")

    def test_bootstrap_generation(self):
        """Test bootstrap sample generation."""
        bootstrap = ConcreteBootstrap(n_bootstraps=3, rng=42)
        X = np.array([[1], [2], [3], [4], [5]])

        # Test without indices
        samples = list(bootstrap.bootstrap(X, return_indices=False))
        assert len(samples) == 3
        for sample in samples:
            assert isinstance(sample, np.ndarray)
            assert sample.shape == X.shape

        # Test with indices
        samples_with_idx = list(bootstrap.bootstrap(X, return_indices=True))
        assert len(samples_with_idx) == 3
        for sample, indices in samples_with_idx:
            assert isinstance(sample, np.ndarray)
            assert isinstance(indices, np.ndarray)
            assert sample.shape == X.shape
            assert indices.shape == (X.shape[0],)

    def test_repr_and_str(self):
        """Test string representations."""
        bootstrap = ConcreteBootstrap(n_bootstraps=5, rng=42)

        repr_str = repr(bootstrap)
        assert "ConcreteBootstrap" in repr_str
        assert "n_bootstraps=5" in repr_str

        str_str = str(bootstrap)
        assert "ConcreteBootstrap" in str_str
        assert "5 bootstrap samples" in str_str

    def test_is_fitted_property(self):
        """Test the is_fitted computed property."""
        bootstrap = ConcreteBootstrap(rng=42)  # Provide rng to avoid None
        assert bootstrap.is_fitted is False

        X = np.array([[1], [2], [3]])
        _ = list(bootstrap.bootstrap(X))
        assert bootstrap.is_fitted is True

    def test_rng_validation(self):
        """Test RNG validation and conversion."""
        # Test with None (default)
        bootstrap1 = ConcreteBootstrap(rng=None)
        assert isinstance(bootstrap1.rng, np.random.Generator)

        # Test with int seed
        bootstrap2 = ConcreteBootstrap(rng=42)
        assert isinstance(bootstrap2.rng, np.random.Generator)

        # Test with Generator instance
        rng = np.random.default_rng(42)
        bootstrap3 = ConcreteBootstrap(rng=rng)
        assert bootstrap3.rng is rng

        # Test with invalid type
        with pytest.raises(TypeError):
            ConcreteBootstrap(rng="invalid")

    def test_model_config_optimizations(self):
        """Test that Pydantic config optimizations are applied."""
        bootstrap = ConcreteBootstrap()
        config = bootstrap.model_config

        # Check performance optimizations
        assert config.get("validate_default") is False
        assert config.get("use_enum_values") is True
        assert config.get("arbitrary_types_allowed") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
