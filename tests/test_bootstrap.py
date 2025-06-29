"""
Test composition-based bootstrap implementations.

This mirrors tests/test_bootstrap.py but for composition-based classes.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError
from tsbootstrap.bootstrap import (
    BlockResidualBootstrap,
    BlockSieveBootstrap,
    WholeResidualBootstrap,
    WholeSieveBootstrap,
)


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
            assert bootstrap.n_bootstraps == 5
            assert bootstrap.model_type == model_type
            assert bootstrap.order == order

        @given(
            n_bootstraps=st.integers(min_value=1, max_value=100),
            rng=st.one_of(st.none(), st.integers(min_value=0, max_value=2**32 - 1)),
        )
        @settings(max_examples=10)
        def test_hypothesis_valid_params(self, n_bootstraps, rng):
            """Test with hypothesis-generated valid parameters."""
            bootstrap = WholeResidualBootstrap(
                n_bootstraps=n_bootstraps,
                rng=rng,
            )
            assert bootstrap.n_bootstraps == n_bootstraps
            if rng is not None:
                assert isinstance(bootstrap.rng, np.random.Generator)

        def test_bootstrap_generation(self):
            """Test bootstrap sample generation."""
            rng = np.random.default_rng(42)
            bootstrap = WholeResidualBootstrap(
                n_bootstraps=3,
                model_type="ar",
                order=1,
                rng=rng,
            )

            # Generate sample data
            X = np.random.randn(100, 1)

            # Generate bootstrap samples
            samples = list(bootstrap.bootstrap(X))

            assert len(samples) == 3
            for sample in samples:
                assert sample.shape == X.shape
                assert not np.array_equal(sample, X)  # Should be different due to resampling

        def test_model_persistence(self):
            """Test that models can be saved."""
            bootstrap = WholeResidualBootstrap(
                n_bootstraps=2,
                model_type="ar",
                order=1,
                save_models=True,
            )

            X = np.random.randn(50, 1)
            list(bootstrap.bootstrap(X))

            # When save_models=True, the model should be retained
            assert bootstrap._fitted_model is not None

    class TestFailingCases:
        """Invalid bootstrap operations."""

        @pytest.mark.parametrize(
            "n_bootstraps",
            [0, -1, -10],
        )
        def test_invalid_n_bootstraps(self, n_bootstraps):
            """Test that invalid n_bootstraps raises error."""
            with pytest.raises(ValidationError):
                WholeResidualBootstrap(n_bootstraps=n_bootstraps)

        def test_invalid_model_type(self):
            """Test that invalid model type raises error."""
            with pytest.raises(ValidationError):
                WholeResidualBootstrap(model_type="invalid_model")

        def test_empty_data(self):
            """Test that empty data raises error."""
            bootstrap = WholeResidualBootstrap(n_bootstraps=1)
            with pytest.raises(ValueError):
                list(bootstrap.bootstrap(np.array([])))


class TestBlockResidualBootstrap:
    """Test suite for BlockResidualBootstrap."""

    class TestPassingCases:
        """Valid block bootstrap operations."""

        @pytest.mark.parametrize(
            "block_length,overlap_flag",
            [
                (5, True),
                (10, False),
                (3, True),
            ],
        )
        def test_block_parameters(self, block_length, overlap_flag):
            """Test block-specific parameters."""
            bootstrap = BlockResidualBootstrap(
                n_bootstraps=5,
                block_length=block_length,
                overlap_flag=overlap_flag,
            )
            assert bootstrap.block_length == block_length
            assert bootstrap.overlap_flag == overlap_flag

        def test_block_bootstrap_generation(self):
            """Test block bootstrap sample generation."""
            bootstrap = BlockResidualBootstrap(
                n_bootstraps=3,
                block_length=5,
                model_type="ar",
                order=1,
                rng=42,
            )

            # Generate sample data
            X = np.random.randn(50, 1)

            # Generate bootstrap samples
            samples = list(bootstrap.bootstrap(X))

            assert len(samples) == 3
            for sample in samples:
                assert sample.shape == X.shape

    class TestFailingCases:
        """Invalid block bootstrap operations."""

        def test_block_length_too_large(self):
            """Test that block length larger than data raises error."""
            bootstrap = BlockResidualBootstrap(
                n_bootstraps=1,
                block_length=100,
            )

            X = np.random.randn(50, 1)
            with pytest.raises(ValueError, match="block_length.*cannot be larger than"):
                list(bootstrap.bootstrap(X))

        @pytest.mark.parametrize(
            "block_length",
            [0, -1, -5],
        )
        def test_invalid_block_length(self, block_length):
            """Test that invalid block length raises error."""
            with pytest.raises(ValidationError):
                BlockResidualBootstrap(block_length=block_length)


class TestSieveBootstrap:
    """Test suite for Sieve bootstrap implementations."""

    class TestPassingCases:
        """Valid sieve bootstrap operations."""

        def test_whole_sieve_bootstrap(self):
            """Test WholeSieveBootstrap."""
            bootstrap = WholeSieveBootstrap(
                n_bootstraps=3,
                min_lag=1,
                max_lag=5,
                criterion="aic",
                rng=42,
            )

            X = np.random.randn(100, 1)
            samples = list(bootstrap.bootstrap(X))

            assert len(samples) == 3
            for sample in samples:
                assert sample.shape == X.shape

        def test_block_sieve_bootstrap(self):
            """Test BlockSieveBootstrap."""
            bootstrap = BlockSieveBootstrap(
                n_bootstraps=3,
                block_length=10,
                min_lag=1,
                max_lag=5,
                criterion="bic",
                rng=42,
            )

            X = np.random.randn(100, 1)
            samples = list(bootstrap.bootstrap(X))

            assert len(samples) == 3
            for sample in samples:
                assert sample.shape == X.shape

        @pytest.mark.parametrize(
            "criterion",
            ["aic", "bic", "hqic"],
        )
        def test_sieve_criteria(self, criterion):
            """Test different information criteria."""
            bootstrap = WholeSieveBootstrap(
                n_bootstraps=2,
                criterion=criterion,
            )
            assert bootstrap.criterion == criterion


class TestBootstrapCompatibility:
    """Test compatibility and integration of composition-based bootstrap classes."""

    class TestPassingCases:
        """Valid compatibility tests."""

        def test_service_injection(self):
            """Test that custom services can be injected."""
            from tsbootstrap.services.service_container import BootstrapServices

            # Create custom services
            services = BootstrapServices.create_for_model_based_bootstrap()

            # Create bootstrap with custom services
            bootstrap = WholeResidualBootstrap(
                n_bootstraps=2,
                services=services,
            )

            assert bootstrap._services is services

        def test_sklearn_compatibility(self):
            """Test sklearn-compatible methods."""
            bootstrap = WholeResidualBootstrap(
                n_bootstraps=10,
                model_type="ar",
                order=2,
            )

            # Test get_params
            params = bootstrap.get_params()
            assert params["n_bootstraps"] == 10
            assert params["model_type"] == "ar"
            assert params["order"] == 2

            # Test set_params
            bootstrap.set_params(n_bootstraps=20)
            assert bootstrap.n_bootstraps == 20

            # Test clone
            cloned = bootstrap.clone()
            assert cloned.n_bootstraps == 20
            assert cloned is not bootstrap

        def test_multivariate_data(self):
            """Test with multivariate time series."""
            bootstrap = WholeResidualBootstrap(
                n_bootstraps=2,
                model_type="var",
                order=1,
            )

            # Multivariate data
            X = np.random.randn(50, 3)

            samples = list(bootstrap.bootstrap(X))
            assert len(samples) == 2
            for sample in samples:
                assert sample.shape == (50, 3)

        def test_with_exogenous_variables(self):
            """Test bootstrap with exogenous variables."""
            bootstrap = WholeResidualBootstrap(
                n_bootstraps=2,
                model_type="ar",
                order=1,
            )

            X = np.random.randn(50, 1)
            y = np.random.randn(50, 2)  # Exogenous variables

            samples = list(bootstrap.bootstrap(X, y=y))
            assert len(samples) == 2
            for sample in samples:
                if isinstance(sample, tuple):
                    X_boot, y_boot = sample
                    assert X_boot.shape == X.shape
                    assert y_boot.shape == y.shape
                else:
                    assert sample.shape == X.shape
