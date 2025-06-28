"""
Test migrated bootstrap implementations.

Follows TestPassingCases/TestFailingCases pattern with hypothesis and parametrize.
"""

import contextlib

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

# Then import bootstrap classes - this should trigger registration
from tsbootstrap.bootstrap import (
    BlockResidualBootstrap,
    BlockSieveBootstrap,
    WholeResidualBootstrap,
    WholeSieveBootstrap,
)

# Import factory first to ensure it's available
from tsbootstrap.bootstrap_factory import BootstrapFactory


class TestWholeResidualBootstrap:
    """Test suite for WholeResidualBootstrap."""

    class TestPassingCases:
        """Valid bootstrap operations."""

        @pytest.mark.parametrize(
            "model_type,order",
            [
                ("ar", 2),
                ("ar", [1, 3]),
                ("arima", (1, 1, 1)),
                ("var", 3),
            ],
        )
        def test_initialization_with_models(self, model_type, order):
            """Test initialization with different model types."""
            bootstrap = WholeResidualBootstrap(
                n_bootstraps=5,
                model_type=model_type,
                order=order,
                rng=42,
            )
            assert bootstrap.model_type == model_type
            assert bootstrap.order == order
            assert bootstrap.requires_model_fitting is True

        @given(
            n_samples=st.integers(min_value=20, max_value=100),
            n_features=st.integers(min_value=1, max_value=5),
            n_bootstraps=st.integers(min_value=1, max_value=10),
        )
        @settings(deadline=None, max_examples=10)
        def test_bootstrap_generation(self, n_samples, n_features, n_bootstraps):
            """Test bootstrap sample generation."""
            # Generate synthetic time series data
            X = np.random.randn(n_samples, n_features)

            bootstrap = WholeResidualBootstrap(
                n_bootstraps=n_bootstraps,
                model_type="ar",
                order=1,
                rng=42,
            )

            # Generate bootstrap samples
            samples = list(bootstrap.bootstrap(X, return_indices=False))

            assert len(samples) == n_bootstraps
            for sample in samples:
                assert sample.shape == X.shape
                assert not np.array_equal(sample, X)  # Should be different from original

        def test_factory_registration(self):
            """Test that bootstrap is registered with factory."""
            # Import all bootstrap classes to trigger decorator registration
            from tsbootstrap.bootstrap import (
                BlockResidualBootstrap,  # noqa: F811
                BlockSieveBootstrap,  # noqa: F811
                WholeResidualBootstrap,  # noqa: F811
                WholeSieveBootstrap,  # noqa: F811
            )

            # Create instances to ensure full initialization
            test_instances = [
                WholeResidualBootstrap(n_bootstraps=1),
                WholeSieveBootstrap(n_bootstraps=1),
                BlockResidualBootstrap(n_bootstraps=1, block_length=5),
                BlockSieveBootstrap(n_bootstraps=1, block_length=5),
            ]
            assert all(inst is not None for inst in test_instances)

            # Check all registrations
            assert BootstrapFactory.is_registered("whole_residual")
            assert BootstrapFactory.is_registered("whole_sieve")
            assert BootstrapFactory.is_registered("block_residual")
            assert BootstrapFactory.is_registered("block_sieve")

            # Test creating from registry
            bootstrap = BootstrapFactory._registry["whole_sieve"](
                n_bootstraps=5,
                min_lag=2,
                max_lag=15,
            )

            assert isinstance(bootstrap, WholeSieveBootstrap)
            assert bootstrap.min_lag == 2
            assert bootstrap.n_bootstraps == 5

    class TestFailingCases:
        """Invalid sieve bootstrap operations."""

        def test_invalid_lag_order(self):
            """Test invalid lag configuration."""
            with pytest.raises(ValidationError):
                WholeSieveBootstrap(
                    min_lag=10,
                    max_lag=5,  # max < min
                )

        def test_zero_min_lag(self):
            """Test zero min lag."""
            with pytest.raises(ValidationError):
                WholeSieveBootstrap(
                    min_lag=0,  # Must be positive
                )


class TestBootstrapCompatibility:
    """Test compatibility with existing code."""

    class TestPassingCases:
        """Test that new implementations are compatible."""

        @pytest.mark.parametrize(
            "bootstrap_type,params",
            [
                ("whole_residual", {"model_type": "ar", "order": 2}),
                (
                    "block_residual",
                    {"model_type": "ar", "order": 1, "block_length": 5},
                ),
                ("whole_sieve", {"min_lag": 1, "max_lag": 10}),
            ],
        )
        def test_basic_interface(self, bootstrap_type, params):
            """Test basic bootstrap interface."""
            # Import and reference to ensure registrations
            from tsbootstrap.bootstrap import (
                BlockResidualBootstrap,  # noqa: F811
                BlockSieveBootstrap,  # noqa: F811
                WholeResidualBootstrap,  # noqa: F811
                WholeSieveBootstrap,  # noqa: F811
            )

            # Reference imports to keep them
            assert all(
                [
                    BlockResidualBootstrap,
                    BlockSieveBootstrap,
                    WholeResidualBootstrap,
                    WholeSieveBootstrap,
                ]
            )

            # Create directly from registry
            bootstrap_cls = BootstrapFactory._registry[bootstrap_type]
            bootstrap = bootstrap_cls(n_bootstraps=3, rng=42, **params)

            # Test basic attributes
            assert hasattr(bootstrap, "n_bootstraps")
            assert hasattr(bootstrap, "bootstrap")
            assert hasattr(bootstrap, "get_params")
            assert hasattr(bootstrap, "set_params")

            # Test bootstrap generation
            X = np.random.randn(50, 2)
            samples = list(bootstrap.bootstrap(X))

            assert len(samples) == 3
            for sample in samples:
                assert sample.shape == X.shape

        def test_sklearn_compatibility(self):
            """Test sklearn interface compatibility."""
            from sklearn.base import clone

            bootstrap = WholeResidualBootstrap(
                n_bootstraps=5,
                model_type="ar",
                order=2,
                rng=42,
            )

            # Test get_params
            params = bootstrap.get_params()
            assert "n_bootstraps" in params
            assert "model_type" in params

            # Test set_params
            bootstrap.set_params(n_bootstraps=10)
            assert bootstrap.n_bootstraps == 10

            # Test clone
            cloned = clone(bootstrap)
            assert cloned.n_bootstraps == 10
            assert cloned is not bootstrap


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
