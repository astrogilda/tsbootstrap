"""Common utilities and shared code for bootstrap implementations."""

from typing import Optional, Tuple, Union

import numpy as np

from tsbootstrap.backends.adapter import BackendToStatsmodelsAdapter, fit_with_backend
from tsbootstrap.tsfit_compat import TSFit
from tsbootstrap.utils.types import ModelTypesWithoutArch


class BootstrapUtilities:
    """Shared utilities for bootstrap implementations."""

    @staticmethod
    def fit_time_series_model(
        X: np.ndarray,
        y: Optional[np.ndarray],
        model_type: ModelTypesWithoutArch,
        order: Optional[Union[int, Tuple]] = None,
        seasonal_order: Optional[tuple] = None,
        use_tsfit_compat: bool = False,
    ) -> Tuple[Union[TSFit, BackendToStatsmodelsAdapter], np.ndarray]:
        """
        Common model fitting logic for bootstrap methods.

        Parameters
        ----------
        X : np.ndarray
            Time series data
        y : Optional[np.ndarray]
            Exogenous variables
        model_type : ModelTypesWithoutArch
            Type of time series model
        order : Optional[Union[int, Tuple]]
            Model order
        seasonal_order : Optional[tuple]
            Seasonal order for SARIMA
        use_tsfit_compat : bool, default=False
            If True, use TSFit for compatibility. If False, use backends directly.

        Returns
        -------
        fitted_model : Union[TSFit, BackendToStatsmodelsAdapter]
            Fitted time series model
        residuals : np.ndarray
            Model residuals
        """
        # Ensure X is properly shaped for time series models
        if model_type == "var":
            # VAR needs multivariate data in shape (n_obs, n_vars)
            if X.ndim == 2:
                X_model = X  # Keep as is - VAR expects (n_obs, n_vars)
            else:
                raise ValueError("VAR models require 2D multivariate data")
        else:
            # For univariate models, ensure we have a 1D array
            if X.ndim == 2:
                if X.shape[1] == 1:
                    # Single column, flatten it
                    X_model = X.flatten()
                else:
                    # Multiple columns, take first column and flatten
                    X_model = X[:, 0].flatten()
            else:
                # Already 1D
                X_model = X

        # Handle None order by using default based on model type
        if order is None:
            if model_type == "var":
                order = 1
            elif model_type in ["arima", "sarima"]:
                order = (1, 1, 1)
            else:  # ar, ma, arma
                order = 1

        if use_tsfit_compat:
            # Use TSFit for backward compatibility
            ts_fit = TSFit(
                order=order,
                model_type=model_type,
                seasonal_order=seasonal_order,
            )
            fitted = ts_fit.fit(X=X_model, y=y)
            model = fitted.model
        else:
            # Use backend system directly for better performance and stability
            fitted = fit_with_backend(
                model_type=model_type,
                endog=X_model,
                exog=y,
                order=order,
                seasonal_order=seasonal_order,
                force_backend="statsmodels",  # Use statsmodels for stability
                return_backend=False,  # Get adapter for statsmodels compatibility
            )
            model = fitted

        # Extract residuals
        if hasattr(model, "resid"):
            residuals = model.resid
            # For VAR models, handle backend shape issues
            if model_type == "var":
                # Backend bug workaround: VAR residuals come as (1, n_obs*n_vars) instead of (n_obs, n_vars)
                if residuals.shape[0] == 1 and residuals.shape[1] > len(X):
                    # Reshape from (1, n_obs*n_vars) to (n_obs, n_vars)
                    # First, figure out the actual shape
                    n_vars = X.shape[1]
                    n_obs_resid = residuals.shape[1] // n_vars
                    residuals = residuals.reshape(n_obs_resid, n_vars)
                elif residuals.ndim == 2 and residuals.shape == (len(X) - order, X.shape[1]):
                    # Already in correct shape (n_obs - order, n_vars)
                    pass
        else:
            # Fallback: compute residuals from predictions
            try:
                if model_type == "var":
                    # VAR predictions need special handling
                    predictions = model.fittedvalues
                    residuals = X - predictions  # X is original (n_obs, n_vars)
                else:
                    predictions = model.predict(start=0, end=len(X_model) - 1)
                    residuals = X_model.flatten() - predictions.flatten()
            except Exception:
                # If prediction fails, return zeros
                if model_type == "var":
                    residuals = np.zeros_like(X)
                else:
                    residuals = np.zeros(len(X_model))

        # Ensure residuals have same length as input by padding if needed
        if model_type == "var":
            # For VAR, ensure residuals match X's shape
            if residuals.shape[0] < X.shape[0]:
                padding_length = X.shape[0] - residuals.shape[0]
                padding = np.zeros((padding_length, X.shape[1]))
                residuals = np.concatenate([padding, residuals], axis=0)
        else:
            # For univariate models
            if len(residuals) < len(X_model):
                padding_length = len(X_model) - len(residuals)
                if residuals.ndim == 2:
                    # Multivariate residuals (shouldn't happen for univariate models)
                    padding = np.zeros((padding_length, residuals.shape[1]))
                else:
                    # Univariate residuals
                    padding = np.zeros(padding_length)
                residuals = np.concatenate([padding, residuals])

        # Return the appropriate fitted model
        if use_tsfit_compat:
            return fitted, residuals
        else:
            # For direct backend usage, wrap in a simple container
            # that provides TSFit-like interface
            class FittedModelWrapper:
                def __init__(self, model):
                    self.model = model

            return FittedModelWrapper(model), residuals

    @staticmethod
    def resample_residuals_whole(
        residuals: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
        replace: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample residuals with replacement (whole bootstrap).

        Parameters
        ----------
        residuals : np.ndarray
            Model residuals to resample
        n_samples : int
            Number of samples to generate
        rng : np.random.Generator
            Random number generator
        replace : bool
            Whether to sample with replacement

        Returns
        -------
        indices : np.ndarray
            Indices of resampled residuals
        resampled_residuals : np.ndarray
            Resampled residuals
        """
        indices = rng.choice(len(residuals), size=n_samples, replace=replace)
        resampled_residuals = residuals[indices]
        return indices, resampled_residuals

    @staticmethod
    def resample_residuals_block(
        residuals: np.ndarray,
        n_samples: int,
        block_length: int,
        rng: np.random.Generator,
        overlap: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample residuals in blocks (block bootstrap).

        Parameters
        ----------
        residuals : np.ndarray
            Model residuals to resample
        n_samples : int
            Number of samples to generate
        block_length : int
            Length of blocks
        rng : np.random.Generator
            Random number generator
        overlap : bool
            Whether blocks can overlap

        Returns
        -------
        indices : np.ndarray
            Indices of resampled residuals
        resampled_residuals : np.ndarray
            Resampled residuals
        """
        n_residuals = len(residuals)
        n_blocks = (n_samples + block_length - 1) // block_length

        # Handle case where block_length exceeds data length
        effective_block_length = min(block_length, n_residuals)

        max_start = (
            max(1, n_residuals - effective_block_length + 1)
            if overlap
            else max(1, n_residuals - effective_block_length + 1)
        )

        block_starts = rng.choice(max_start, size=n_blocks, replace=True)

        # Collect residuals from blocks
        resampled_residuals = []
        indices = []

        for start in block_starts:
            block_indices = np.arange(start, min(start + effective_block_length, n_residuals))
            indices.extend(block_indices)
            resampled_residuals.extend(residuals[block_indices])

        # Ensure we have exactly n_samples
        resampled_residuals = np.array(resampled_residuals)
        indices = np.array(indices)

        if len(resampled_residuals) == 0:
            resampled_residuals = np.zeros(n_samples)
            indices = np.arange(n_samples)
        elif len(resampled_residuals) < n_samples:
            repeats = (n_samples // len(resampled_residuals)) + 1
            resampled_residuals = np.tile(resampled_residuals, repeats)
            indices = np.tile(indices, repeats)

        resampled_residuals = resampled_residuals[:n_samples]
        indices = indices[:n_samples]

        return indices, resampled_residuals

    @staticmethod
    def reconstruct_time_series(
        fitted_values: np.ndarray,
        resampled_residuals: np.ndarray,
        original_shape: tuple,
        indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Reconstruct time series from fitted values and resampled residuals.

        Parameters
        ----------
        fitted_values : np.ndarray
            Fitted values from the model
        resampled_residuals : np.ndarray
            Resampled residuals
        original_shape : tuple
            Shape of original data
        indices : Optional[np.ndarray]
            Indices used for resampling (for multivariate data)

        Returns
        -------
        bootstrapped_series : np.ndarray
            Reconstructed bootstrap sample
        """
        n_samples = len(resampled_residuals)

        # Ensure fitted values have correct length
        if len(fitted_values) < n_samples:
            if fitted_values.ndim == 2:
                # Multivariate fitted values
                padding = np.full(
                    (n_samples - len(fitted_values), fitted_values.shape[1]),
                    np.mean(fitted_values, axis=0),
                )
            else:
                # Univariate fitted values
                padding = np.full(n_samples - len(fitted_values), np.mean(fitted_values))
            fitted_values = np.concatenate([padding, fitted_values])

        # Reconstruct series
        bootstrapped_series = fitted_values[:n_samples] + resampled_residuals

        # Handle shape matching
        if len(original_shape) == 2 and bootstrapped_series.ndim == 1:
            # Univariate series, but need 2D output
            bootstrapped_series = bootstrapped_series.reshape(-1, 1)
            # For multivariate, bootstrapped_series should already be the right shape

        return bootstrapped_series
