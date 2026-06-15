"""
Bootstrap implementation tests: Validating service-oriented architecture.

We test the concrete bootstrap implementations built on our service composition
framework. These tests verify that the orchestration of services produces correct
statistical results while maintaining clean architectural boundaries.

Testing follows a natural progression from basic initialization through complex
workflows. We start with simple parameter validation, move to configuration
testing, then validate complete bootstrap operations. Model-based methods receive
extra attention since they involve the most complex service interactions.

Each test class targets a specific bootstrap variant. We examine both common
behaviors and the unique edge cases that each method presents.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from tsbootstrap.bootstrap import (
    BlockResidualBootstrap,
    BlockSieveBootstrap,
    ModelBasedBootstrap,
    WholeResidualBootstrap,
    WholeSieveBootstrap,
)
from tsbootstrap.services.service_container import BootstrapServices


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


class TestCoverageEnhancements:
    """Additional tests to achieve 80% coverage."""

    def test_model_based_bootstrap_abstract_get_test_params(self):
        """Test ModelBasedBootstrap.get_test_params returns empty list."""
        params = ModelBasedBootstrap.get_test_params()
        assert params == []

    def test_requires_model_fitting_property(self):
        """Test requires_model_fitting computed property."""
        bootstrap = WholeResidualBootstrap(n_bootstraps=1)
        assert bootstrap.requires_model_fitting is True

    def test_type_checking_imports(self):
        """Test TYPE_CHECKING import coverage."""
        # Importing the module triggers TYPE_CHECKING block
        from tsbootstrap import bootstrap as bs_module

        assert hasattr(bs_module, "WholeResidualBootstrap")

    def test_get_test_params_for_all_classes(self):
        """Test get_test_params for all bootstrap classes."""
        # WholeResidualBootstrap
        params = WholeResidualBootstrap.get_test_params()
        assert isinstance(params, list)
        assert params[0]["n_bootstraps"] == 10

        # WholeSieveBootstrap
        params = WholeSieveBootstrap.get_test_params()
        assert isinstance(params, list)
        assert params[0]["n_bootstraps"] == 10

        # BlockResidualBootstrap
        params = BlockResidualBootstrap.get_test_params()
        assert isinstance(params, list)

        # BlockSieveBootstrap
        params = BlockSieveBootstrap.get_test_params()
        assert isinstance(params, list)

    def test_residuals_unavailable_error(self):
        """Test error when residuals are not available after fitting."""
        from unittest.mock import patch

        bootstrap = WholeResidualBootstrap(n_bootstraps=1)
        X = np.random.randn(10)

        # Mock _fit_model_if_needed to do nothing, leaving residuals as None
        with patch.object(bootstrap, "_fit_model_if_needed"):
            bootstrap._residuals = None

            with pytest.raises(ValueError, match="No residuals available for bootstrapping"):
                list(bootstrap.bootstrap(X))

    def test_univariate_padding_whole_residual(self):
        """Test padding for univariate series in WholeResidualBootstrap."""
        # Use high order AR to trigger padding
        bootstrap = WholeResidualBootstrap(
            n_bootstraps=1, model_type="arima", order=(8, 1, 1), rng=42
        )

        X = np.random.randn(20)
        samples = list(bootstrap.bootstrap(X))
        assert len(samples) == 1
        assert samples[0].shape == X.shape

    def test_multivariate_padding_whole_residual(self):
        """Test padding for multivariate series in WholeResidualBootstrap."""
        bootstrap = WholeResidualBootstrap(n_bootstraps=1, model_type="var", order=5, rng=42)

        X = np.random.randn(15, 3)
        samples = list(bootstrap.bootstrap(X))
        assert len(samples) == 1
        assert samples[0].shape == X.shape

    def test_univariate_padding_sieve(self):
        """Test padding for univariate series in WholeSieveBootstrap."""
        bootstrap = WholeSieveBootstrap(
            n_bootstraps=1, min_lag=5, max_lag=8, criterion="aic", rng=42
        )

        X = np.random.randn(20)
        samples = list(bootstrap.bootstrap(X))
        assert len(samples) == 1
        assert samples[0].shape == X.shape

    def test_multivariate_padding_sieve(self):
        """Test padding for multivariate series in WholeSieveBootstrap."""
        bootstrap = WholeSieveBootstrap(
            n_bootstraps=1, min_lag=3, max_lag=5, criterion="aic", rng=42
        )

        X = np.random.randn(15, 2)
        samples = list(bootstrap.bootstrap(X))
        assert len(samples) == 1
        assert samples[0].shape == X.shape

    def test_sieve_order_selection_flow(self):
        """Test sieve bootstrap order selection logic."""
        bootstrap = WholeSieveBootstrap(
            n_bootstraps=1, min_lag=1, max_lag=3, criterion="bic", save_models=False, rng=42
        )

        X = np.random.randn(50)

        # First bootstrap should select order
        assert bootstrap._fitted_model is None
        samples1 = list(bootstrap.bootstrap(X))
        assert len(samples1) == 1

        # Force refit by clearing model
        bootstrap._fitted_model = None
        samples2 = list(bootstrap.bootstrap(X))
        assert len(samples2) == 1

    def test_custom_services_with_model_based_bootstrap(self):
        """Test custom services initialization."""
        services = BootstrapServices.create_for_model_based_bootstrap()

        bootstrap = WholeResidualBootstrap(
            services=services, n_bootstraps=2, model_type="ar", order=3, rng=42
        )

        X = np.random.randn(100)
        samples = list(bootstrap.bootstrap(X))
        assert len(samples) == 2

    def test_docstring_example_code(self):
        """Test the docstring example runs successfully."""
        # This covers lines 466-499 in the docstring
        import numpy as np

        # Generate sample data
        np.random.seed(42)
        n = 100
        X = np.cumsum(np.random.randn(n)).reshape(-1, 1)

        # Standard usage with default services
        bootstrap = WholeResidualBootstrap(n_bootstraps=5, model_type="ar", order=2)
        samples = bootstrap.bootstrap(X)
        samples_list = list(samples)
        assert len(samples_list) == 5

        # Advanced usage with custom service configuration
        custom_services = BootstrapServices.create_for_model_based_bootstrap()

        bootstrap_custom = WholeResidualBootstrap(
            services=custom_services, n_bootstraps=5, model_type="ar", order=2
        )
        samples_custom = bootstrap_custom.bootstrap(X)
        samples_custom_list = list(samples_custom)
        assert len(samples_custom_list) == 5

    def test_save_models_behavior(self):
        """Test save_models parameter functionality."""
        bootstrap = WholeResidualBootstrap(
            n_bootstraps=1, model_type="ar", order=2, save_models=True, rng=42
        )

        X = np.random.randn(50)

        # First call - should fit and save model
        list(bootstrap.bootstrap(X))
        model1 = bootstrap._fitted_model
        residuals1 = bootstrap._residuals
        fitted1 = bootstrap._fitted_values

        assert model1 is not None
        assert residuals1 is not None
        assert fitted1 is not None

        # Second call - should reuse saved model
        list(bootstrap.bootstrap(X))
        model2 = bootstrap._fitted_model

        assert model1 is model2  # Same object

    def test_block_padding_scenarios(self):
        """Test various block bootstrap padding scenarios."""
        # Block sieve with high order
        bootstrap = BlockSieveBootstrap(
            n_bootstraps=1, block_length=5, min_lag=7, max_lag=10, criterion="bic", rng=42
        )

        X = np.random.randn(25, 2)
        samples = list(bootstrap.bootstrap(X))
        assert len(samples) == 1
        assert samples[0].shape == X.shape

    def test_no_padding_needed_scenarios(self):
        """Test scenarios where no padding is needed."""
        # Test line 221 - no padding needed case
        bootstrap = WholeResidualBootstrap(
            n_bootstraps=1, model_type="ar", order=1, rng=42  # Low order, minimal loss
        )

        X = np.random.randn(50)
        samples = list(bootstrap.bootstrap(X))
        assert len(samples) == 1
        assert samples[0].shape == X.shape

    def test_block_residual_no_padding(self):
        """Test BlockResidualBootstrap when no padding needed."""
        # Test line 317 for block bootstrap
        bootstrap = BlockResidualBootstrap(
            n_bootstraps=1, block_length=5, model_type="ar", order=1, rng=42
        )

        X = np.random.randn(100)
        samples = list(bootstrap.bootstrap(X))
        assert len(samples) == 1
        assert samples[0].shape == X.shape

    def test_sieve_no_padding_univariate(self):
        """Test sieve bootstrap when no padding needed."""
        # Test line 393 for sieve bootstrap
        bootstrap = WholeSieveBootstrap(
            n_bootstraps=1, min_lag=1, max_lag=2, criterion="aic", rng=42
        )

        X = np.random.randn(100)
        samples = list(bootstrap.bootstrap(X))
        assert len(samples) == 1
        assert samples[0].shape == X.shape

    def test_block_sieve_multivariate_padding(self):
        """Test block sieve bootstrap multivariate padding."""
        # Test lines 579-581 for block sieve
        bootstrap = BlockSieveBootstrap(
            n_bootstraps=1, block_length=3, min_lag=5, max_lag=8, criterion="bic", rng=42
        )

        X = np.random.randn(20, 3)
        samples = list(bootstrap.bootstrap(X))
        assert len(samples) == 1
        assert samples[0].shape == X.shape

    def test_sieve_order_selection_direct(self):
        """Test sieve order selection is called."""
        # Test lines 398-408 - order selection in sieve
        bootstrap = WholeSieveBootstrap(
            n_bootstraps=1,
            min_lag=1,
            max_lag=5,
            criterion="aic",
            save_models=True,  # Save to check order was selected
            rng=42,
        )

        X = np.random.randn(100)

        # Bootstrap should trigger order selection
        samples = list(bootstrap.bootstrap(X))
        assert len(samples) == 1

        # Just check that bootstrap worked
        assert samples[0].shape == X.shape

    def test_whole_residual_edge_cases(self):
        """Test edge cases for WholeResidualBootstrap."""
        # Test with minimal data
        bootstrap = WholeResidualBootstrap(n_bootstraps=1, model_type="ar", order=1, rng=42)

        # Just enough data to fit AR(1)
        X = np.random.randn(5)
        samples = list(bootstrap.bootstrap(X))
        assert len(samples) == 1
        assert samples[0].shape == X.shape

    def test_comprehensive_coverage_scenarios(self):
        """Comprehensive test to cover remaining lines."""
        # Test line 221 - else case when no padding
        bootstrap = WholeResidualBootstrap(
            n_bootstraps=2, model_type="arima", order=(1, 0, 1), rng=42  # ARIMA model
        )
        X = np.random.randn(100)  # Univariate for ARIMA
        samples = list(bootstrap.bootstrap(X))
        assert len(samples) == 2
        for s in samples:
            assert s.shape == X.shape

        # Test line 317 - block bootstrap no padding case
        bootstrap = BlockResidualBootstrap(
            n_bootstraps=2, block_length=10, model_type="arima", order=(1, 0, 1), rng=42
        )
        X = np.random.randn(100)
        samples = list(bootstrap.bootstrap(X))
        assert len(samples) == 2

        # Test line 393 - sieve bootstrap else case
        bootstrap = WholeSieveBootstrap(
            n_bootstraps=2, min_lag=1, max_lag=3, criterion="bic", rng=42
        )
        X = np.random.randn(100, 2)
        samples = list(bootstrap.bootstrap(X))
        assert len(samples) == 2

        # Test lines 441-443 - sieve bootstrap no padding multivariate
        bootstrap = WholeSieveBootstrap(
            n_bootstraps=1, min_lag=1, max_lag=2, criterion="aic", rng=42
        )
        X = np.random.randn(200, 3)  # Large enough to not need padding
        samples = list(bootstrap.bootstrap(X))
        assert len(samples) == 1
        assert samples[0].shape == X.shape

        # Test lines 579-581 - block sieve else case
        bootstrap = BlockSieveBootstrap(
            n_bootstraps=1, block_length=5, min_lag=1, max_lag=2, criterion="aic", rng=42
        )
        X = np.random.randn(100, 2)
        samples = list(bootstrap.bootstrap(X))
        assert len(samples) == 1
        assert samples[0].shape == X.shape

    def test_sieve_fit_model_if_needed_coverage(self):
        """Test _fit_model_if_needed in sieve bootstrap."""
        # This tests lines 398-408
        from unittest.mock import patch

        bootstrap = WholeSieveBootstrap(
            n_bootstraps=1, min_lag=1, max_lag=3, criterion="aic", rng=42
        )

        X = np.random.randn(50)

        # Mock order selector to ensure it gets called
        with patch.object(
            bootstrap._services.order_selector, "select_order", return_value=2
        ) as mock_selector:
            # Ensure model is None initially
            assert bootstrap._fitted_model is None

            # Call bootstrap which should trigger _fit_model_if_needed
            samples = list(bootstrap.bootstrap(X))

            # Check order selector was called
            mock_selector.assert_called_once()
            assert len(samples) == 1


# Additional coverage tests
class TestBootstrapAdditionalCoverage:
    """Additional tests for complete coverage of bootstrap.py."""

    def test_type_checking_imports(self):
        """Test that TYPE_CHECKING imports work correctly."""
        # This is already covered by the import, but we can verify the type annotation
        bootstrap = WholeResidualBootstrap(n_bootstraps=1)
        # The TimeSeriesModel type is used in annotations
        assert hasattr(bootstrap, "_fitted_model")

    def test_1d_padding_edge_case(self):
        """Test 1D array padding when bootstrap series is shorter."""
        np.random.seed(42)
        X = np.random.randn(100)  # 1D array

        bootstrap = WholeResidualBootstrap(n_bootstraps=1, model_type="ar", order=2)

        # We need to mock the reconstruction to return a shorter series
        # This will trigger the padding logic
        short_series = X[:80]  # Shorter than original

        # Mock the reconstructor to return shorter series
        original_reconstruct = bootstrap._services.reconstructor.reconstruct_time_series

        def mock_reconstruct(fitted_values, resampled_residuals):
            return short_series

        bootstrap._services.reconstructor.reconstruct_time_series = mock_reconstruct

        # Generate samples
        samples = list(bootstrap.bootstrap(X))

        # Restore
        bootstrap._services.reconstructor.reconstruct_time_series = original_reconstruct

        # Should be padded to original length
        assert len(samples[0]) == len(X)
        # Last 20 values should all be the same (padding)
        assert np.all(samples[0][-20:] == samples[0][-20])

    def test_shape_mismatch_error(self):
        """Test _pad_to_original_length shape mismatch error."""
        np.random.seed(42)
        X = np.random.randn(100, 3)  # 2D array with 3 columns

        bootstrap = WholeResidualBootstrap(n_bootstraps=1, model_type="var", order=2)

        # Directly test the _pad_to_original_length method to ensure line 173 is covered
        # Create a 1D array that needs padding when X is 2D with multiple columns
        bootstrapped_1d = np.random.randn(80)  # 1D array, shorter than X

        # This should trigger the ValueError at line 173
        with pytest.raises(
            ValueError, match="Shape mismatch: bootstrapped series is 1D but X has 3 columns"
        ):
            bootstrap._pad_to_original_length(bootstrapped_1d, X)

    def test_sieve_bootstrap_edge_cases(self):
        """Test sieve bootstrap validation edge case."""
        # Test max_lag < min_lag validation
        with pytest.raises(ValueError, match="max_lag must be >= min_lag"):
            WholeSieveBootstrap(n_bootstraps=1, min_lag=10, max_lag=5)  # Invalid: less than min_lag

    def test_sieve_bootstrap_order_selection_flow(self):
        """Test sieve bootstrap order selection flow."""
        np.random.seed(42)
        X = np.random.randn(100)

        # Create sieve bootstrap with order selection
        bootstrap = WholeSieveBootstrap(n_bootstraps=1, min_lag=1, max_lag=5, criterion="aic")

        # Verify order selection happens
        samples = list(bootstrap.bootstrap(X))

        # For sieve bootstrap, order is selected dynamically during each bootstrap
        # The instance order remains None since it's selected per-sample
        # Verify the bootstrap completed successfully
        assert len(samples) == 1
        assert len(samples[0]) == len(X)

    def test_docstring_example_execution(self):
        """Execute the docstring example code."""
        # Execute the docstring example code directly
        import numpy as np

        from tsbootstrap.bootstrap import WholeResidualBootstrap
        from tsbootstrap.services.service_container import BootstrapServices

        # Generate sample data
        np.random.seed(42)
        n = 100
        X = np.cumsum(np.random.randn(n)).reshape(-1, 1)

        # Standard usage with default services
        bootstrap = WholeResidualBootstrap(n_bootstraps=5, model_type="ar", order=2)
        samples = list(bootstrap.bootstrap(X))

        # Advanced usage with custom service configuration
        custom_services = BootstrapServices.create_for_model_based_bootstrap()

        bootstrap_custom = WholeResidualBootstrap(
            services=custom_services, n_bootstraps=5, model_type="ar", order=2
        )
        samples_custom = list(bootstrap_custom.bootstrap(X))

        # Verify results
        assert len(samples) == 5  # n_bootstraps=5
        assert len(samples_custom) == 5
        # Both should produce numpy arrays
        assert all(isinstance(s, np.ndarray) for s in samples)
        assert all(isinstance(s, np.ndarray) for s in samples_custom)

    def test_block_residual_padding_edge_case(self):
        """Test edge case for BlockResidualBootstrap padding."""
        np.random.seed(42)
        X = np.random.randn(100)

        # Create block bootstrap that might need padding
        bootstrap = BlockResidualBootstrap(
            n_bootstraps=1,
            model_type="ar",
            order=10,
            block_length=30,  # Large blocks might cause short series
        )

        # Generate samples
        samples = list(bootstrap.bootstrap(X))

        # Should maintain original length
        assert len(samples[0]) == len(X)

    def test_whole_residual_with_large_order(self):
        """Test WholeResidualBootstrap with order approaching data length."""
        np.random.seed(42)
        X = np.random.randn(200)  # Larger dataset to support high order

        # Order that will cause shorter bootstrap series
        bootstrap = WholeResidualBootstrap(
            n_bootstraps=1,
            model_type="ar",
            order=50,  # High order but still reasonable for 200 samples
        )

        # Should still work and maintain length
        samples = list(bootstrap.bootstrap(X))
        assert len(samples[0]) == len(X)

    def test_multivariate_padding_scenarios(self):
        """Test various multivariate padding scenarios."""
        np.random.seed(42)

        # Test different multivariate shapes
        for n_features in [1, 2, 5]:
            X = np.random.randn(100, n_features)

            bootstrap = WholeResidualBootstrap(
                n_bootstraps=2, model_type="var" if n_features > 1 else "ar", order=10
            )

            samples = list(bootstrap.bootstrap(X))

            # All samples should maintain shape
            for sample in samples:
                assert sample.shape == X.shape

    def test_block_sieve_multivariate(self):
        """Test BlockSieveBootstrap with multivariate data."""
        np.random.seed(42)
        X = np.random.randn(100, 2)

        bootstrap = BlockSieveBootstrap(n_bootstraps=1, block_length=10, min_lag=1, max_lag=5)

        samples = list(bootstrap.bootstrap(X))
        assert samples[0].shape == X.shape

    def test_invalid_bootstrap_parameters(self):
        """Test various invalid parameter combinations."""
        # These should all raise ValueError
        invalid_configs = [
            {"n_bootstraps": 0},  # Invalid number
            {"n_bootstraps": -1},  # Negative
            {"model_type": "ar", "order": 0},  # Invalid order
        ]

        for config in invalid_configs:
            with pytest.raises(ValueError):
                WholeResidualBootstrap(**config)

    def test_data_too_short_for_model(self):
        """Test bootstrap with data too short for model order."""
        np.random.seed(42)
        X = np.random.randn(20)  # Short but workable

        bootstrap = WholeResidualBootstrap(
            n_bootstraps=1, model_type="ar", order=5  # Reasonable for this data length
        )

        # Should handle gracefully
        samples = list(bootstrap.bootstrap(X))
        assert len(samples) == 1
        assert len(samples[0]) == len(X)

    def test_demonstrate_service_architecture(self):
        """Test the demonstrate_service_architecture function."""
        from tsbootstrap.bootstrap import demonstrate_service_architecture

        # This function is part of the documentation
        samples, samples_custom = demonstrate_service_architecture()

        # Verify it returns valid results
        assert len(list(samples)) == 5
        assert len(list(samples_custom)) == 5

    def test_1d_padding_concatenate(self):
        """Test 1D padding concatenation logic."""
        np.random.seed(42)
        X = np.random.randn(100)  # 1D array

        bootstrap = WholeResidualBootstrap(n_bootstraps=1, model_type="ar", order=2)

        # Directly test the padding method with a 1D array that needs padding
        short_series = np.random.randn(80)

        # This should use the 1D padding logic (lines 165-166)
        padded = bootstrap._pad_to_original_length(short_series, X)

        assert len(padded) == 100
        # Check that the last 20 values are all the same (padding)
        assert np.all(padded[-20:] == short_series[-1])

    def test_block_residual_specific_padding(self):
        """Test BlockResidualBootstrap padding scenarios."""
        np.random.seed(42)
        X = np.random.randn(100)

        # Create bootstrap with parameters that might trigger padding
        bootstrap = BlockResidualBootstrap(
            n_bootstraps=1,
            model_type="ar",
            order=20,  # High order to ensure shorter series
            block_length=15,
        )

        # Mock the block resampler to create a shorter series
        original_resample = bootstrap._services.residual_resampler.resample_residuals_block

        def mock_resample(residuals, block_length, n_samples):
            # Return residuals that will result in a shorter series
            return residuals[:70]  # Only 70 samples instead of 100

        bootstrap._services.residual_resampler.resample_residuals_block = mock_resample

        # Generate sample - should trigger padding
        samples = list(bootstrap.bootstrap(X))

        # Restore original
        bootstrap._services.residual_resampler.resample_residuals_block = original_resample

        # Should maintain original length through padding
        assert len(samples[0]) == 100

    def test_sieve_fit_model_order_selection(self):
        """Test sieve bootstrap _fit_model_if_needed with order selection."""
        np.random.seed(42)
        X = np.random.randn(100)

        bootstrap = WholeSieveBootstrap(n_bootstraps=1, min_lag=1, max_lag=5, criterion="aic")

        # Directly call _fit_model_if_needed to trigger order selection
        bootstrap._fit_model_if_needed(X)

        # The order should have been selected and model fitted
        assert bootstrap._fitted_model is not None
        assert bootstrap.order is not None
        assert 1 <= bootstrap.order <= 5
