"""
Batch bootstrap service for high-performance bootstrap operations.

This service leverages the statsforecast backend's batch processing capabilities
to achieve 10-50x speedup for Method A (data bootstrap) operations.
"""

from typing import Any, List, Optional, Tuple

import numpy as np

from tsbootstrap.backends import create_backend
from tsbootstrap.utils.types import ModelTypes


class IndividualModelWrapper:
    """Wrapper for an individual model from batch fitting.

    This class provides access to a single model's parameters and methods
    from a batch-fitted backend that contains multiple models.
    """

    def __init__(self, backend, series_index: int, model_type: str, order: Any):
        """Initialize wrapper for a specific model from the batch.

        Parameters
        ----------
        backend : StatsForecastFittedBackend
            The fitted backend containing all models
        series_index : int
            Index of this specific model in the batch
        model_type : str
            Type of model (AR, ARIMA, etc.)
        order : Any
            Model order parameters
        """
        self.backend = backend
        self.series_index = series_index
        self.model_type = model_type
        self.order = order

        # Extract this model's specific attributes
        # Check if backend has params_list attribute
        if hasattr(backend, "_params_list"):
            self.params = backend._params_list[series_index]
        elif hasattr(backend, "params_list"):
            self.params = backend.params_list[series_index]
        else:
            # Fallback: extract from params property
            params = backend.params
            if isinstance(params, dict) and "series_params" in params:
                self.params = params["series_params"][series_index]
            else:
                self.params = params

        # Extract residuals and fitted values
        try:
            if hasattr(backend, "_residuals"):
                all_residuals = backend._residuals
            else:
                all_residuals = backend.residuals

            # Handle numpy arrays and mock objects
            if hasattr(all_residuals, "ndim") and all_residuals.ndim > 1:
                self.residuals = all_residuals[series_index]
            else:
                self.residuals = all_residuals
        except (AttributeError, TypeError):
            # For mocked objects or when residuals not available
            self.residuals = None

        try:
            if hasattr(backend, "_fitted_values"):
                all_fitted = backend._fitted_values
            else:
                all_fitted = backend.fitted_values

            # Handle numpy arrays and mock objects
            if hasattr(all_fitted, "ndim") and all_fitted.ndim > 1:
                self.fitted_values = all_fitted[series_index]
            else:
                self.fitted_values = all_fitted
        except (AttributeError, TypeError):
            # For mocked objects or when fitted values not available
            self.fitted_values = None

    def predict(self, steps: int, X: Optional[np.ndarray] = None, **kwargs: Any) -> np.ndarray:
        """Generate predictions for this individual model.

        Parameters
        ----------
        steps : int
            Number of steps to predict
        X : np.ndarray, optional
            Exogenous variables
        **kwargs : Any
            Additional prediction arguments

        Returns
        -------
        np.ndarray
            Predictions for this specific model
        """
        # Get predictions from the backend
        all_predictions = self.backend.predict(steps=steps, X=X, **kwargs)

        # Extract this model's predictions
        if all_predictions.ndim > 1 and all_predictions.shape[0] > 1:
            return all_predictions[self.series_index]
        return all_predictions

    def simulate(
        self,
        steps: int,
        n_paths: int = 1,
        X: Optional[np.ndarray] = None,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Generate simulations for this individual model.

        Parameters
        ----------
        steps : int
            Number of steps to simulate
        n_paths : int, default 1
            Number of simulation paths
        X : np.ndarray, optional
            Exogenous variables
        random_state : int, optional
            Random seed
        **kwargs : Any
            Additional simulation arguments

        Returns
        -------
        np.ndarray
            Simulations for this specific model
        """
        # Get simulations from the backend
        all_simulations = self.backend.simulate(
            steps=steps, n_paths=n_paths, X=X, random_state=random_state, **kwargs
        )

        # Extract this model's simulations
        if all_simulations.ndim > 2 and all_simulations.shape[0] > 1:
            return all_simulations[self.series_index]
        return all_simulations

    def forecast(self, steps: int, **kwargs: Any) -> np.ndarray:
        """Generate forecasts (alias for predict).

        This method provides compatibility with statsmodels interface.
        """
        return self.predict(steps=steps, **kwargs)

    def get_prediction(
        self, start: Optional[int] = None, end: Optional[int] = None, **kwargs: Any
    ) -> Any:
        """Get prediction with confidence intervals.

        This is primarily for statsmodels compatibility.
        """
        if hasattr(self.backend, "get_prediction"):
            # If backend supports this method
            result = self.backend.get_prediction(start=start, end=end, **kwargs)
            # Would need to extract series-specific results
            return result
        else:
            # Fallback to basic predict
            if start is None:
                start = 0
            if end is None:
                end = len(self.residuals)
            steps = end - start
            return self.predict(steps=steps, **kwargs)


class BatchBootstrapService:
    """
    Service for performing batch bootstrap operations.

    This service coordinates batch model fitting for bootstrap samples,
    leveraging backend systems that support batch operations for massive
    performance improvements.
    """

    def __init__(self, use_backend: bool = False):
        """
        Initialize batch bootstrap service.

        Parameters
        ----------
        use_backend : bool, default False
            Whether to use backend system for batch operations.
        """
        self.use_backend = use_backend

    def fit_models_batch(
        self,
        bootstrap_samples: List[np.ndarray],
        model_type: ModelTypes = "ar",
        order: Any = 1,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        **kwargs,
    ) -> List[Any]:
        """
        Fit models to multiple bootstrap samples in batch.

        Parameters
        ----------
        bootstrap_samples : List[np.ndarray]
            List of bootstrap samples, each of shape (n_obs,) or (n_obs, n_features)
        model_type : str, default "ar"
            Type of model to fit
        order : Any, default 1
            Model order
        seasonal_order : Optional[Tuple[int, int, int, int]], default None
            Seasonal order for SARIMA models
        **kwargs
            Additional model fitting arguments

        Returns
        -------
        List[Any]
            List of fitted models, one per bootstrap sample
        """
        if not self.use_backend or model_type.lower() not in ["ar", "arima", "sarima"]:
            # Fall back to sequential fitting
            return self._fit_models_sequential(
                bootstrap_samples, model_type, order, seasonal_order, **kwargs
            )

        # Prepare data for batch fitting
        # Stack all samples into a single array with shape (n_series, n_obs)
        n_samples = len(bootstrap_samples)
        n_obs = len(bootstrap_samples[0])

        # Ensure all samples have same length
        for i, sample in enumerate(bootstrap_samples):
            if len(sample) != n_obs:
                raise ValueError(
                    f"All bootstrap samples must have same length. "
                    f"Sample 0 has length {n_obs}, sample {i} has length {len(sample)}"
                )

        # Stack into batch array
        batch_data = np.array(bootstrap_samples)
        if batch_data.ndim == 2:
            # Shape is already (n_series, n_obs)
            pass
        elif batch_data.ndim == 3:
            # Multivariate case - for now, only use first variable
            batch_data = batch_data[:, :, 0]

        # Create backend and fit in batch
        backend = create_backend(
            model_type=model_type.upper(), order=order, force_backend="statsforecast"
        )

        # Fit all models at once
        fitted_backend = backend.fit(batch_data)

        # Extract individual fitted models
        fitted_models = []
        for i in range(n_samples):
            # Create a wrapper that represents a single fitted model
            individual_model = IndividualModelWrapper(
                backend=fitted_backend, series_index=i, model_type=model_type, order=order
            )
            fitted_models.append(individual_model)

        return fitted_models

    def _fit_models_sequential(
        self,
        bootstrap_samples: List[np.ndarray],
        model_type: ModelTypes,
        order: Any,
        seasonal_order: Optional[Tuple[int, int, int, int]],
        **kwargs,
    ) -> List[Any]:
        """Sequential model fitting fallback."""
        from tsbootstrap.time_series_model import TimeSeriesModel

        fitted_models = []
        for sample in bootstrap_samples:
            ts_model = TimeSeriesModel(X=sample, model_type=model_type)
            fitted = ts_model.fit(order=order, seasonal_order=seasonal_order, **kwargs)
            fitted_models.append(fitted)

        return fitted_models

    def simulate_batch(self, fitted_models: List[Any], steps: int, n_paths: int = 1) -> np.ndarray:
        """
        Simulate from multiple fitted models in batch.

        Parameters
        ----------
        fitted_models : List[Any]
            List of fitted models
        steps : int
            Number of steps to simulate
        n_paths : int, default 1
            Number of simulation paths per model

        Returns
        -------
        np.ndarray
            Array of shape (n_models, steps, n_paths) with simulated values
        """
        # For backend models that support batch simulation
        if hasattr(fitted_models[0], "simulate_batch"):
            return fitted_models[0].simulate_batch(steps=steps, n_paths=n_paths)

        # Fallback to sequential simulation
        simulations = []
        for model in fitted_models:
            if hasattr(model, "simulate"):
                sim = model.simulate(steps=steps, n_paths=n_paths)
            elif hasattr(model, "forecast"):
                # For statsmodels compatibility
                sim = model.forecast(steps=steps)
                if n_paths > 1:
                    # Replicate forecast for multiple paths
                    sim = np.tile(sim, (n_paths, 1)).T
            else:
                raise ValueError(f"Model {type(model)} does not support simulation")

            simulations.append(sim)

        return np.array(simulations)
