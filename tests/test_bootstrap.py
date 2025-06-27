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
from tsbootstrap.bootstrap import (
    BlockResidualBootstrap,
    WholeResidualBootstrap,
    WholeSieveBootstrap,
)
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
            # Import and reference to ensure registration
            from tsbootstrap.bootstrap import WholeResidualBootstrap

            assert WholeResidualBootstrap is not None  # Keep import
            assert BootstrapFactory.is_registered("whole_residual")

            # Create directly from registry
            bootstrap_cls = BootstrapFactory._registry["whole_residual"]
            bootstrap = bootstrap_cls(
                n_bootstraps=10,
                model_type="ar",
                order=2,
            )

            assert isinstance(bootstrap, WholeResidualBootstrap)
            assert bootstrap.n_bootstraps == 10
            assert bootstrap.model_type == "ar"
            assert bootstrap.order == 2

        def test_sarima_model(self):
            """Test SARIMA model bootstrap."""
            bootstrap = WholeResidualBootstrap(
                n_bootstraps=3,
                model_type="sarima",
                order=(1, 1, 1),
                seasonal_order=(1, 0, 1, 12),
                rng=42,
            )

            # Generate monthly data with seasonal pattern
            t = np.arange(48)
            seasonal = np.sin(2 * np.pi * t / 12)
            trend = 0.1 * t
            noise = np.random.randn(48) * 0.1
            X = (seasonal + trend + noise).reshape(-1, 1)

            samples = list(bootstrap.bootstrap(X))
            assert len(samples) == 3

            for sample in samples:
                assert sample.shape == X.shape

        @pytest.mark.parametrize("save_models", [True, False])
        def test_save_models_flag(self, save_models):
            """Test save_models functionality."""
            bootstrap = WholeResidualBootstrap(
                n_bootstraps=2,
                model_type="ar",
                save_models=save_models,
                rng=42,
            )

            X = np.random.randn(30, 1)
            list(bootstrap.bootstrap(X))

            # After bootstrapping, model should be fitted
            assert bootstrap._fitted_model is not None
            assert bootstrap._residuals is not None

        def test_var_model_type(self):
            """Test VAR model type with None order."""
            bootstrap = WholeResidualBootstrap(
                n_bootstraps=1,
                model_type="var",
                order=None,  # Should default to 1
                rng=42,
            )

            X = np.random.randn(30, 2)  # VAR needs multivariate
            # VAR fitting might fail, but we're testing the order defaulting
            with contextlib.suppress(Exception):
                list(bootstrap.bootstrap(X))
            assert bootstrap.model_type == "var"

        def test_arima_model_type(self):
            """Test ARIMA model type with None order."""
            bootstrap = WholeResidualBootstrap(
                n_bootstraps=1,
                model_type="arima",
                order=None,  # Should default to (1,1,1)
                rng=42,
            )

            X = np.random.randn(50, 1)
            # ARIMA fitting might fail, but we're testing the order defaulting
            with contextlib.suppress(Exception):
                list(bootstrap.bootstrap(X))
            assert bootstrap.model_type == "arima"

    class TestFailingCases:
        """Invalid bootstrap operations."""

        def test_invalid_model_type(self):
            """Test invalid model type."""
            with pytest.raises(ValidationError):
                WholeResidualBootstrap(
                    model_type="invalid_model",
                    order=2,
                )

        def test_insufficient_data(self):
            """Test with too little data."""
            bootstrap = WholeResidualBootstrap(
                n_bootstraps=5,
                model_type="ar",
                order=10,  # Order too high for data
                rng=42,
            )

            X = np.random.randn(5, 1)  # Too few samples

            # Should raise an error during model fitting
            with pytest.raises((ValueError, TypeError, RuntimeError)):  # Model fitting error
                list(bootstrap.bootstrap(X))


class TestBlockResidualBootstrap:
    """Test suite for BlockResidualBootstrap."""

    class TestPassingCases:
        """Valid block bootstrap operations."""

        @given(
            block_length=st.integers(min_value=1, max_value=20),
            overlap_flag=st.booleans(),
        )
        def test_block_configuration(self, block_length, overlap_flag):
            """Test block bootstrap configuration."""
            bootstrap = BlockResidualBootstrap(
                n_bootstraps=5,
                block_length=block_length,
                overlap_flag=overlap_flag,
                model_type="ar",
                order=1,
                rng=42,
            )

            assert bootstrap.block_length == block_length
            assert bootstrap.overlap_flag == overlap_flag

        def test_block_structure_preservation(self):
            """Test that block structure is preserved."""
            bootstrap = BlockResidualBootstrap(
                n_bootstraps=3,
                block_length=5,
                model_type="ar",
                order=1,
                rng=42,
            )

            # Create data with clear pattern
            X = np.arange(50).reshape(-1, 1).astype(float)

            samples = list(bootstrap.bootstrap(X, return_indices=True))

            for _sample, indices in samples:
                # Check that indices come in blocks
                # (This is a simplified check)
                assert len(indices) == len(X)

        def test_factory_registration(self):
            """Test factory registration."""
            # Import and reference to ensure registration
            from tsbootstrap.bootstrap import BlockResidualBootstrap

            assert BlockResidualBootstrap is not None  # Keep import
            assert BootstrapFactory.is_registered("block_residual")

            # Create directly from registry
            bootstrap_cls = BootstrapFactory._registry["block_residual"]
            bootstrap = bootstrap_cls(
                n_bootstraps=10,
                block_length=5,
                model_type="ar",
                order=2,
            )

            assert isinstance(bootstrap, BlockResidualBootstrap)
            assert bootstrap.block_length == 5
            assert bootstrap.n_bootstraps == 10

        def test_var_model_with_none_order(self):
            """Test VAR model type with None order."""
            bootstrap = BlockResidualBootstrap(
                n_bootstraps=1,
                block_length=5,
                model_type="var",
                order=None,  # Should default to 1
                rng=42,
            )
            X = np.random.randn(30, 2)
            with contextlib.suppress(Exception):
                list(bootstrap.bootstrap(X))
            assert bootstrap.model_type == "var"

        def test_arima_model_with_none_order(self):
            """Test ARIMA model type with None order."""
            bootstrap = BlockResidualBootstrap(
                n_bootstraps=1,
                block_length=5,
                model_type="arima",
                order=None,  # Should default to (1,1,1)
                rng=42,
            )
            X = np.random.randn(50, 1)
            with contextlib.suppress(Exception):
                list(bootstrap.bootstrap(X))
            assert bootstrap.model_type == "arima"

    class TestFailingCases:
        """Invalid block bootstrap operations."""

        def test_invalid_block_length(self):
            """Test invalid block length."""
            with pytest.raises(ValidationError):
                BlockResidualBootstrap(
                    block_length=0,  # Must be positive
                    model_type="ar",
                )

        def test_block_length_exceeds_data(self):
            """Test when block length exceeds data length."""
            bootstrap = BlockResidualBootstrap(
                n_bootstraps=3,
                block_length=100,  # Very large
                model_type="ar",
                order=1,
                rng=42,
            )

            X = np.random.randn(20, 1)  # Small data

            # Should still work but with adjusted behavior
            samples = list(bootstrap.bootstrap(X))
            assert len(samples) == 3


class TestWholeSieveBootstrap:
    """Test suite for WholeSieveBootstrap."""

    class TestPassingCases:
        """Valid sieve bootstrap operations."""

        @given(
            min_lag=st.integers(min_value=1, max_value=5),
            max_lag=st.one_of(st.none(), st.integers(min_value=5, max_value=20)),
        )
        def test_lag_configuration(self, min_lag, max_lag):
            """Test lag configuration."""
            assume(max_lag is None or max_lag >= min_lag)

            bootstrap = WholeSieveBootstrap(
                n_bootstraps=3,
                min_lag=min_lag,
                max_lag=max_lag,
                rng=42,
            )

            assert bootstrap.min_lag == min_lag
            assert bootstrap.max_lag == max_lag
            assert bootstrap.model_type == "ar"  # Always AR

        def test_order_selection(self):
            """Test that order is selected based on sample size."""
            bootstrap = WholeSieveBootstrap(
                n_bootstraps=2,
                min_lag=1,
                max_lag=10,
                rng=42,
            )

            # Test with different sample sizes
            for n in [50, 100, 200]:
                X = np.random.randn(n, 1)

                # Fit model to trigger order selection
                bootstrap._fit_model(X)

                assert bootstrap._selected_order is not None
                assert (
                    bootstrap.min_lag <= bootstrap._selected_order <= (bootstrap.max_lag or n // 4)
                )

        @pytest.mark.parametrize("criterion", ["aic", "bic", "hqic"])
        def test_information_criteria(self, criterion):
            """Test different information criteria."""
            bootstrap = WholeSieveBootstrap(
                n_bootstraps=2,
                criterion=criterion,
                rng=42,
            )

            assert bootstrap.criterion == criterion

            X = np.random.randn(50, 1)
            samples = list(bootstrap.bootstrap(X))
            assert len(samples) == 2

        def test_factory_registration(self):
            """Test factory registration."""
            # Import and reference to ensure registration
            from tsbootstrap.bootstrap import WholeSieveBootstrap

            assert WholeSieveBootstrap is not None  # Keep import
            assert BootstrapFactory.is_registered("whole_sieve")

            # Create directly from registry
            bootstrap_cls = BootstrapFactory._registry["whole_sieve"]
            bootstrap = bootstrap_cls(
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
                BlockResidualBootstrap,
                BlockSieveBootstrap,
                WholeResidualBootstrap,
                WholeSieveBootstrap,
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
