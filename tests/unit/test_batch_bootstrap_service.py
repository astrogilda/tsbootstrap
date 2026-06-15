"""Tests for batch_bootstrap_service.py."""

from unittest.mock import ANY, Mock, patch

import numpy as np
import pytest

from tsbootstrap.services.batch_bootstrap_service import (
    BatchBootstrapService,
    IndividualModelWrapper,
)


class TestIndividualModelWrapper:
    """Tests targeting specific uncovered lines in IndividualModelWrapper."""

    def test_init_with_params_list_underscore(self):
        """Test initialization with _params_list attribute ."""
        mock_backend = Mock()
        mock_backend._params_list = [{"param1": 1}, {"param2": 2}]
        mock_backend._residuals = np.array([[1, 2, 3], [4, 5, 6]])
        mock_backend._fitted_values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        wrapper = IndividualModelWrapper(mock_backend, 0, "ar", 1)

        assert wrapper.params == {"param1": 1}
        assert wrapper.series_index == 0
        assert wrapper.model_type == "ar"
        assert wrapper.order == 1

    def test_init_with_params_list_no_underscore(self):
        """Test initialization with params_list attribute ."""
        mock_backend = Mock()
        del mock_backend._params_list  # Remove _params_list
        mock_backend.params_list = [{"param1": 1}, {"param2": 2}]
        mock_backend._residuals = np.array([[1, 2, 3], [4, 5, 6]])
        mock_backend._fitted_values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        wrapper = IndividualModelWrapper(mock_backend, 1, "arima", (1, 0, 1))

        assert wrapper.params == {"param2": 2}

    def test_init_with_params_fallback_dict(self):
        """Test initialization with params fallback for dict with series_params ."""
        mock_backend = Mock()
        del mock_backend._params_list
        del mock_backend.params_list
        mock_backend.params = {"series_params": [{"param1": 1}, {"param2": 2}]}
        mock_backend._residuals = np.array([[1, 2, 3], [4, 5, 6]])
        mock_backend._fitted_values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        wrapper = IndividualModelWrapper(mock_backend, 0, "ar", 1)

        assert wrapper.params == {"param1": 1}

    def test_init_with_params_fallback_direct(self):
        """Test initialization with params fallback for direct params ."""
        mock_backend = Mock()
        del mock_backend._params_list
        del mock_backend.params_list
        mock_backend.params = {"direct_param": 42}
        mock_backend._residuals = np.array([[1, 2, 3], [4, 5, 6]])
        mock_backend._fitted_values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        wrapper = IndividualModelWrapper(mock_backend, 0, "ar", 1)

        assert wrapper.params == {"direct_param": 42}

    def test_init_residuals_underscore_attribute(self):
        """Test residual extraction with _residuals attribute ."""
        mock_backend = Mock()
        mock_backend._params_list = [{"param1": 1}, {"param2": 2}]  # Need 2 elements for index 1
        # Use a real numpy array - it already has the ndim attribute
        mock_backend._residuals = np.array([[1, 2, 3], [4, 5, 6]])
        mock_backend._fitted_values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        wrapper = IndividualModelWrapper(mock_backend, 1, "ar", 1)

        np.testing.assert_array_equal(wrapper.residuals, np.array([4, 5, 6]))
        assert wrapper.params == {"param2": 2}

    def test_init_residuals_no_underscore_attribute(self):
        """Test residual extraction with residuals attribute ."""
        mock_backend = Mock()
        mock_backend._params_list = [{"param1": 1}, {"param2": 2}]
        del mock_backend._residuals
        mock_backend.residuals = np.array([[1, 2, 3], [4, 5, 6]])
        mock_backend._fitted_values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        wrapper = IndividualModelWrapper(mock_backend, 0, "ar", 1)

        np.testing.assert_array_equal(wrapper.residuals, np.array([1, 2, 3]))

    def test_init_residuals_1d_fallback(self):
        """Test residual extraction with 1D array fallback ."""
        mock_backend = Mock()
        mock_backend._params_list = [{"param1": 1}, {"param2": 2}]
        mock_backend._residuals = np.array([1, 2, 3])  # 1D array
        mock_backend._fitted_values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        wrapper = IndividualModelWrapper(mock_backend, 0, "ar", 1)

        np.testing.assert_array_equal(wrapper.residuals, np.array([1, 2, 3]))

    def test_init_residuals_exception_handling(self):
        """Test residual extraction exception handling ."""
        mock_backend = Mock()
        mock_backend._params_list = [{"param1": 1}, {"param2": 2}]
        # Make residuals access raise an exception
        mock_backend._residuals = Mock(side_effect=AttributeError("No residuals"))
        mock_backend.residuals = Mock(side_effect=TypeError("Type error"))
        mock_backend._fitted_values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        wrapper = IndividualModelWrapper(mock_backend, 0, "ar", 1)

        assert wrapper.residuals is None

    def test_init_fitted_values_underscore_attribute(self):
        """Test fitted values extraction with _fitted_values attribute ."""
        mock_backend = Mock()
        mock_backend._params_list = [{"param1": 1}, {"param2": 2}]
        mock_backend._residuals = np.array([[1, 2, 3], [4, 5, 6]])
        # Use a real numpy array - it already has the ndim attribute
        mock_backend._fitted_values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        wrapper = IndividualModelWrapper(mock_backend, 1, "ar", 1)

        np.testing.assert_array_equal(wrapper.fitted_values, np.array([0.4, 0.5, 0.6]))

    def test_init_fitted_values_no_underscore_attribute(self):
        """Test fitted values extraction with fitted_values attribute ."""
        mock_backend = Mock()
        mock_backend._params_list = [{"param1": 1}, {"param2": 2}]
        mock_backend._residuals = np.array([[1, 2, 3], [4, 5, 6]])
        del mock_backend._fitted_values
        mock_backend.fitted_values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        wrapper = IndividualModelWrapper(mock_backend, 0, "ar", 1)

        np.testing.assert_array_equal(wrapper.fitted_values, np.array([0.1, 0.2, 0.3]))

    def test_init_fitted_values_1d_fallback(self):
        """Test fitted values extraction with 1D array fallback ."""
        mock_backend = Mock()
        mock_backend._params_list = [{"param1": 1}, {"param2": 2}]
        mock_backend._residuals = np.array([[1, 2, 3], [4, 5, 6]])
        mock_backend._fitted_values = np.array([0.1, 0.2, 0.3])  # 1D array

        wrapper = IndividualModelWrapper(mock_backend, 0, "ar", 1)

        np.testing.assert_array_equal(wrapper.fitted_values, np.array([0.1, 0.2, 0.3]))

    def test_init_fitted_values_exception_handling(self):
        """Test fitted values extraction exception handling ."""
        mock_backend = Mock()
        mock_backend._params_list = [{"param1": 1}, {"param2": 2}]
        mock_backend._residuals = np.array([[1, 2, 3], [4, 5, 6]])
        # Make fitted_values access raise an exception
        mock_backend._fitted_values = Mock(side_effect=AttributeError("No fitted values"))
        mock_backend.fitted_values = Mock(side_effect=TypeError("Type error"))

        wrapper = IndividualModelWrapper(mock_backend, 0, "ar", 1)

        assert wrapper.fitted_values is None

    def test_predict_multidimensional(self):
        """Test predict with multidimensional predictions ."""
        mock_backend = Mock()
        mock_backend._params_list = [{"param1": 1}, {"param2": 2}]
        mock_backend._residuals = np.array([[1, 2, 3], [4, 5, 6]])
        mock_backend._fitted_values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        # Mock backend predict to return 2D array
        mock_backend.predict.return_value = np.array([[1, 2, 3], [4, 5, 6]])

        wrapper = IndividualModelWrapper(mock_backend, 1, "ar", 1)
        result = wrapper.predict(steps=3)

        np.testing.assert_array_equal(result, np.array([4, 5, 6]))
        mock_backend.predict.assert_called_once_with(steps=3, X=None)

    def test_predict_1d_fallback(self):
        """Test predict with 1D prediction fallback ."""
        mock_backend = Mock()
        mock_backend._params_list = [{"param1": 1}, {"param2": 2}]
        mock_backend._residuals = np.array([[1, 2, 3], [4, 5, 6]])
        mock_backend._fitted_values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        # Mock backend predict to return 1D array
        mock_backend.predict.return_value = np.array([1, 2, 3])

        wrapper = IndividualModelWrapper(mock_backend, 0, "ar", 1)
        result = wrapper.predict(steps=3, X=np.array([1, 2, 3]))

        np.testing.assert_array_equal(result, np.array([1, 2, 3]))
        # Use ANY to avoid array comparison issues
        mock_backend.predict.assert_called_once_with(steps=3, X=ANY)

    def test_simulate_multidimensional(self):
        """Test simulate with multidimensional simulations ."""
        mock_backend = Mock()
        mock_backend._params_list = [{"param1": 1}, {"param2": 2}]
        mock_backend._residuals = np.array([[1, 2, 3], [4, 5, 6]])
        mock_backend._fitted_values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        # Mock backend simulate to return 3D array (n_series, steps, n_paths)
        mock_backend.simulate.return_value = np.array(
            [[[1, 2], [2, 3], [3, 4]], [[4, 5], [5, 6], [6, 7]]]
        )

        wrapper = IndividualModelWrapper(mock_backend, 1, "ar", 1)
        result = wrapper.simulate(steps=3, n_paths=2, random_state=42)

        np.testing.assert_array_equal(result, np.array([[4, 5], [5, 6], [6, 7]]))
        mock_backend.simulate.assert_called_once_with(steps=3, n_paths=2, X=None, random_state=42)

    def test_simulate_fallback(self):
        """Test simulate with fallback for lower dimensional arrays ."""
        mock_backend = Mock()
        mock_backend._params_list = [{"param1": 1}, {"param2": 2}]
        mock_backend._residuals = np.array([[1, 2, 3], [4, 5, 6]])
        mock_backend._fitted_values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        # Mock backend simulate to return 2D array
        mock_backend.simulate.return_value = np.array([[1, 2], [2, 3]])

        wrapper = IndividualModelWrapper(mock_backend, 0, "ar", 1)
        result = wrapper.simulate(steps=2, n_paths=2)

        np.testing.assert_array_equal(result, np.array([[1, 2], [2, 3]]))

    def test_forecast_alias(self):
        """Test forecast method as alias for predict ."""
        mock_backend = Mock()
        mock_backend._params_list = [{"param1": 1}, {"param2": 2}]
        mock_backend._residuals = np.array([[1, 2, 3], [4, 5, 6]])
        mock_backend._fitted_values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        mock_backend.predict.return_value = np.array([1, 2, 3])

        wrapper = IndividualModelWrapper(mock_backend, 0, "ar", 1)
        result = wrapper.forecast(steps=3)

        np.testing.assert_array_equal(result, np.array([1, 2, 3]))
        mock_backend.predict.assert_called_once_with(steps=3, X=None)

    def test_get_prediction_with_backend_method(self):
        """Test get_prediction when backend has the method ."""
        mock_backend = Mock()
        mock_backend._params_list = [{"param1": 1}, {"param2": 2}]
        mock_backend._residuals = np.array([[1, 2, 3], [4, 5, 6]])
        mock_backend._fitted_values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        mock_backend.get_prediction.return_value = "prediction_result"

        wrapper = IndividualModelWrapper(mock_backend, 0, "ar", 1)
        result = wrapper.get_prediction(start=0, end=5)

        assert result == "prediction_result"
        mock_backend.get_prediction.assert_called_once_with(start=0, end=5)

    def test_get_prediction_fallback_with_defaults(self):
        """Test get_prediction fallback with default parameters ."""
        mock_backend = Mock()
        mock_backend._params_list = [{"param1": 1}, {"param2": 2}]
        mock_backend._residuals = np.array([1, 2, 3])  # Length 3
        mock_backend._fitted_values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        # Remove get_prediction method to trigger fallback
        del mock_backend.get_prediction
        mock_backend.predict.return_value = np.array([4, 5, 6])

        wrapper = IndividualModelWrapper(mock_backend, 0, "ar", 1)
        result = wrapper.get_prediction()  # No start/end provided

        # Should use defaults: start=0, end=len(residuals)=3, steps=3
        np.testing.assert_array_equal(result, np.array([4, 5, 6]))
        mock_backend.predict.assert_called_once_with(steps=3, X=None)

    def test_get_prediction_fallback_with_parameters(self):
        """Test get_prediction fallback with explicit parameters ."""
        mock_backend = Mock()
        mock_backend._params_list = [{"param1": 1}, {"param2": 2}]
        mock_backend._residuals = np.array([1, 2, 3, 4, 5])
        mock_backend._fitted_values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        del mock_backend.get_prediction
        mock_backend.predict.return_value = np.array([6, 7])

        wrapper = IndividualModelWrapper(mock_backend, 0, "ar", 1)
        result = wrapper.get_prediction(start=1, end=3)

        # steps = end - start = 3 - 1 = 2
        np.testing.assert_array_equal(result, np.array([6, 7]))
        mock_backend.predict.assert_called_once_with(steps=2, X=None)


class TestBatchBootstrapService:
    """Tests targeting specific uncovered lines in BatchBootstrapService."""

    def test_init(self):
        """Test initialization ."""
        # Test with default use_backend=False
        service = BatchBootstrapService()
        assert service.use_backend is False

        # Test with use_backend=True
        service = BatchBootstrapService(use_backend=True)
        assert service.use_backend is True

    def test_fit_models_batch_fallback_no_backend(self):
        """Test fit_models_batch fallback when use_backend=False ."""
        service = BatchBootstrapService(use_backend=False)

        bootstrap_samples = [
            np.array([1, 2, 3, 4, 5]),
            np.array([2, 3, 4, 5, 6]),
            np.array([3, 4, 5, 6, 7]),
        ]

        with patch.object(service, "_fit_models_sequential") as mock_sequential:
            mock_sequential.return_value = ["model1", "model2", "model3"]

            result = service.fit_models_batch(bootstrap_samples, model_type="ar", order=2)

            assert result == ["model1", "model2", "model3"]
            mock_sequential.assert_called_once_with(bootstrap_samples, "ar", 2, None)

    def test_fit_models_batch_fallback_unsupported_model(self):
        """Test fit_models_batch fallback for unsupported model type ."""
        service = BatchBootstrapService(use_backend=True)

        bootstrap_samples = [np.array([1, 2, 3, 4, 5]), np.array([2, 3, 4, 5, 6])]

        with patch.object(service, "_fit_models_sequential") as mock_sequential:
            mock_sequential.return_value = ["model1", "model2"]

            # VAR model should trigger fallback
            result = service.fit_models_batch(bootstrap_samples, model_type="var", order=2)

            assert result == ["model1", "model2"]
            mock_sequential.assert_called_once_with(bootstrap_samples, "var", 2, None)

    def test_fit_models_batch_length_validation(self):
        """Test fit_models_batch length validation ."""
        service = BatchBootstrapService(use_backend=True)

        bootstrap_samples = [
            np.array([1, 2, 3, 4, 5]),  # length 5
            np.array([2, 3, 4, 5]),  # length 4 - different!
        ]

        with pytest.raises(ValueError, match="All bootstrap samples must have same length"):
            service.fit_models_batch(bootstrap_samples, model_type="ar", order=1)

        with pytest.raises(ValueError, match="Sample 0 has length 5, sample 1 has length 4"):
            service.fit_models_batch(bootstrap_samples, model_type="ar", order=1)

    def test_fit_models_batch_2d_data_handling(self):
        """Test fit_models_batch with 2D data handling ."""
        service = BatchBootstrapService(use_backend=True)

        bootstrap_samples = [
            np.array([1, 2, 3, 4, 5]),
            np.array([2, 3, 4, 5, 6]),
        ]

        with patch("tsbootstrap.services.batch_bootstrap_service.create_backend") as mock_create:
            mock_backend = Mock()
            mock_fitted = Mock()
            mock_fitted._params_list = [{"param1": 1}, {"param2": 2}]
            mock_fitted._residuals = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
            mock_fitted._fitted_values = np.array(
                [[0.1, 0.2, 0.3, 0.4, 0.5], [0.2, 0.3, 0.4, 0.5, 0.6]]
            )

            mock_backend.fit.return_value = mock_fitted
            mock_create.return_value = mock_backend

            result = service.fit_models_batch(bootstrap_samples, model_type="ar", order=1)

            # Check that stacked data has correct shape
            call_args = mock_backend.fit.call_args[0][0]
            assert call_args.shape == (2, 5)  # (n_samples, n_obs)

            assert len(result) == 2
            assert all(isinstance(model, IndividualModelWrapper) for model in result)

    def test_fit_models_batch_3d_data_handling(self):
        """Test fit_models_batch with 3D data handling ."""
        service = BatchBootstrapService(use_backend=True)

        # Create 3D bootstrap samples (multivariate)
        bootstrap_samples = [
            np.array([[1, 2], [2, 3], [3, 4]]),  # shape (3, 2)
            np.array([[2, 3], [3, 4], [4, 5]]),  # shape (3, 2)
        ]

        with patch("tsbootstrap.services.batch_bootstrap_service.create_backend") as mock_create:
            mock_backend = Mock()
            mock_fitted = Mock()
            mock_fitted._params_list = [{"param1": 1}, {"param2": 2}]
            mock_fitted._residuals = np.array([[1, 2, 3], [2, 3, 4]])
            mock_fitted._fitted_values = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])

            mock_backend.fit.return_value = mock_fitted
            mock_create.return_value = mock_backend

            result = service.fit_models_batch(
                bootstrap_samples, model_type="arima", order=(1, 0, 1)
            )

            # Check that 3D data was converted to 2D by taking first variable
            call_args = mock_backend.fit.call_args[0][0]
            assert call_args.shape == (2, 3)  # (n_samples, n_obs)
            np.testing.assert_array_equal(call_args[0], [1, 2, 3])  # First variable of first sample
            np.testing.assert_array_equal(
                call_args[1], [2, 3, 4]
            )  # First variable of second sample

    def test_fit_models_batch_backend_creation(self):
        """Test fit_models_batch backend creation and fitting ."""
        service = BatchBootstrapService(use_backend=True)

        bootstrap_samples = [
            np.array([1, 2, 3]),
            np.array([2, 3, 4]),
        ]

        with patch("tsbootstrap.services.batch_bootstrap_service.create_backend") as mock_create:
            mock_backend = Mock()
            mock_fitted = Mock()
            mock_fitted._params_list = [{"param1": 1}, {"param2": 2}]
            mock_fitted._residuals = np.array([[1, 2, 3], [2, 3, 4]])
            mock_fitted._fitted_values = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])

            mock_backend.fit.return_value = mock_fitted
            mock_create.return_value = mock_backend

            result = service.fit_models_batch(
                bootstrap_samples,
                model_type="sarima",
                order=(1, 1, 1),
                seasonal_order=(1, 0, 1, 12),
            )

            # Verify backend creation
            mock_create.assert_called_once_with(
                model_type="SARIMA", order=(1, 1, 1), force_backend="statsforecast"
            )

            # Verify fitting was called
            mock_backend.fit.assert_called_once()

            # Verify individual model wrappers were created
            assert len(result) == 2
            assert all(isinstance(model, IndividualModelWrapper) for model in result)
            assert result[0].series_index == 0
            assert result[1].series_index == 1

    def test_fit_models_sequential(self):
        """Test _fit_models_sequential method ."""
        service = BatchBootstrapService()

        bootstrap_samples = [
            np.array([1, 2, 3, 4, 5]),
            np.array([2, 3, 4, 5, 6]),
        ]

        with patch("tsbootstrap.time_series_model.TimeSeriesModel") as mock_ts_model:
            # Create mock instances
            mock_instance1 = Mock()
            mock_instance2 = Mock()
            mock_fitted1 = Mock()
            mock_fitted2 = Mock()

            mock_instance1.fit.return_value = mock_fitted1
            mock_instance2.fit.return_value = mock_fitted2

            # Mock the constructor to return our instances
            mock_ts_model.side_effect = [mock_instance1, mock_instance2]

            result = service._fit_models_sequential(
                bootstrap_samples, "ar", 2, (1, 0, 1, 12), extra_param="test"
            )

            # Verify TimeSeriesModel was called correctly
            assert mock_ts_model.call_count == 2
            # Check call arguments manually to avoid array comparison issues
            calls = mock_ts_model.call_args_list
            assert len(calls) == 2

            # Check first call
            call0_kwargs = calls[0].kwargs
            assert call0_kwargs["model_type"] == "ar"
            np.testing.assert_array_equal(call0_kwargs["X"], bootstrap_samples[0])

            # Check second call
            call1_kwargs = calls[1].kwargs
            assert call1_kwargs["model_type"] == "ar"
            np.testing.assert_array_equal(call1_kwargs["X"], bootstrap_samples[1])

            # Verify fit was called correctly
            mock_instance1.fit.assert_called_once_with(
                order=2, seasonal_order=(1, 0, 1, 12), extra_param="test"
            )
            mock_instance2.fit.assert_called_once_with(
                order=2, seasonal_order=(1, 0, 1, 12), extra_param="test"
            )

            # Verify results
            assert result == [mock_fitted1, mock_fitted2]

    def test_simulate_batch_with_batch_support(self):
        """Test simulate_batch when first model has simulate_batch method ."""
        service = BatchBootstrapService()

        mock_model = Mock()
        mock_model.simulate_batch.return_value = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

        fitted_models = [mock_model]

        result = service.simulate_batch(fitted_models, steps=2, n_paths=2)

        np.testing.assert_array_equal(result, np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
        mock_model.simulate_batch.assert_called_once_with(steps=2, n_paths=2)

    def test_simulate_batch_fallback_with_simulate(self):
        """Test simulate_batch fallback with simulate method ."""
        service = BatchBootstrapService()

        mock_model1 = Mock()
        mock_model2 = Mock()
        del mock_model1.simulate_batch  # No batch support
        del mock_model2.simulate_batch

        mock_model1.simulate.return_value = np.array([[1, 2], [3, 4]])
        mock_model2.simulate.return_value = np.array([[5, 6], [7, 8]])

        fitted_models = [mock_model1, mock_model2]

        result = service.simulate_batch(fitted_models, steps=2, n_paths=2)

        expected = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        np.testing.assert_array_equal(result, expected)

        mock_model1.simulate.assert_called_once_with(steps=2, n_paths=2)
        mock_model2.simulate.assert_called_once_with(steps=2, n_paths=2)

    def test_simulate_batch_fallback_with_forecast_single_path(self):
        """Test simulate_batch fallback with forecast method for single path ."""
        service = BatchBootstrapService()

        mock_model = Mock()
        del mock_model.simulate_batch
        del mock_model.simulate  # No simulate method

        mock_model.forecast.return_value = np.array([1, 2, 3])

        fitted_models = [mock_model]

        result = service.simulate_batch(fitted_models, steps=3, n_paths=1)

        np.testing.assert_array_equal(result, np.array([[1, 2, 3]]))
        mock_model.forecast.assert_called_once_with(steps=3)

    def test_simulate_batch_fallback_with_forecast_multiple_paths(self):
        """Test simulate_batch fallback with forecast method for multiple paths ."""
        service = BatchBootstrapService()

        mock_model = Mock()
        del mock_model.simulate_batch
        del mock_model.simulate

        mock_model.forecast.return_value = np.array([1, 2, 3])

        fitted_models = [mock_model]

        result = service.simulate_batch(fitted_models, steps=3, n_paths=2)

        # Should replicate forecast for multiple paths
        expected = np.array([[[1, 1], [2, 2], [3, 3]]])
        np.testing.assert_array_equal(result, expected)
        mock_model.forecast.assert_called_once_with(steps=3)

    def test_simulate_batch_fallback_unsupported_model(self):
        """Test simulate_batch fallback with unsupported model ."""
        service = BatchBootstrapService()

        mock_model = Mock()
        del mock_model.simulate_batch
        del mock_model.simulate
        del mock_model.forecast  # No simulation methods

        fitted_models = [mock_model]

        with pytest.raises(ValueError, match="does not support simulation"):
            service.simulate_batch(fitted_models, steps=3, n_paths=1)

    def test_comprehensive_integration(self):
        """Test comprehensive integration scenario."""
        service = BatchBootstrapService(use_backend=True)

        # Create realistic bootstrap samples
        np.random.seed(42)
        bootstrap_samples = [np.random.randn(20) + i for i in range(2)]  # Use 2 instead of 3

        with patch("tsbootstrap.services.batch_bootstrap_service.create_backend") as mock_create:
            mock_backend = Mock()
            mock_fitted = Mock()

            # Mock fitted backend attributes
            mock_fitted._params_list = [{"ar_coef": [0.5]}, {"ar_coef": [0.6]}]  # Only 2 elements
            mock_fitted._residuals = np.random.randn(2, 20)  # 2 series
            mock_fitted._fitted_values = np.random.randn(2, 20)  # 2 series

            mock_backend.fit.return_value = mock_fitted
            mock_create.return_value = mock_backend

            # Test batch fitting
            fitted_models = service.fit_models_batch(bootstrap_samples, model_type="ar", order=1)

            assert len(fitted_models) == 2

            # Test that each model has correct attributes
            for i, model in enumerate(fitted_models):
                assert isinstance(model, IndividualModelWrapper)
                assert model.series_index == i
                assert model.model_type == "ar"
                assert model.order == 1
                assert model.params["ar_coef"] == [0.5 + i * 0.1]

            # Test simulation - create a simple mock that doesn't rely on array indexing
            mock_sims = [np.random.randn(5, 3) for _ in range(2)]  # Individual simulations
            for i, model in enumerate(fitted_models):
                model.simulate = Mock(return_value=mock_sims[i])

            simulations = service.simulate_batch(fitted_models, steps=5, n_paths=3)

            assert simulations.shape == (2, 5, 3)  # (n_models, steps, n_paths)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
