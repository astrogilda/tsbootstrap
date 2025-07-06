"""
Shared bootstrap utilities: Battle-tested code for the heavy lifting.

After implementing dozens of bootstrap variants, we noticed the same patterns
emerging: fitting models, resampling residuals, reconstructing series. Rather
than duplicate this logic across every bootstrap class, we centralized it here.
This module contains the workhorses that power our bootstrap implementations.

The utilities here embody hard-won knowledge about edge cases and numerical
quirks. Why do we pad residuals? Because some models produce fewer residuals
than observations. Why the special VAR handling? Because backends disagree
on matrix shapes. Each function represents solutions to problems we've
encountered in production.

By sharing this code, we ensure consistency across bootstrap methods while
making it easier to fix bugs and add enhancements. When we discover a better
way to handle model fitting or residual resampling, updating it here improves
every bootstrap variant simultaneously.
"""

from typing import Any, Optional, Tuple, Union

import numpy as np

from tsbootstrap.backends.adapter import BackendToStatsmodelsAdapter, fit_with_backend

# TSFit removed - using backends directly
from tsbootstrap.utils.types import ModelTypesWithoutArch


class BootstrapUtilities:
    """Core utilities that power all bootstrap implementations.

    We designed this class as a central repository for the operations that
    every bootstrap method needs: model fitting, residual resampling, and
    series reconstruction. The static methods reflect our functional approach—
    these are pure transformations without side effects, making them easy to
    test and reason about.

    The implementation handles the messy realities of different backends,
    model types, and data shapes. We've encountered every edge case you can
    imagine, from backends that return transposed matrices to models that
    produce fewer residuals than observations. This class encapsulates those
    hard-won solutions.
    """

    @staticmethod
    def fit_time_series_model(
        X: np.ndarray,
        y: Optional[np.ndarray],
        model_type: ModelTypesWithoutArch,
        order: Optional[Union[int, Tuple]] = None,
        seasonal_order: Optional[tuple] = None,
    ) -> Tuple[Union[BackendToStatsmodelsAdapter, Any], np.ndarray]:
        """
        Fit time series models with intelligent shape handling and backend selection.

        This method embodies years of debugging shape mismatches and backend
        quirks. We handle the impedance mismatch between how users think about
        data (observations in rows) and how different models expect it. VAR wants
        matrices, univariate models want vectors, and we make it all work.

        The residual extraction logic here is particularly battle-tested. Some
        backends return residuals directly, others require computing them from
        predictions, and VAR models have their own special shape requirements.
        We've seen it all and handle it all.

        Parameters
        ----------
        X : np.ndarray
            Time series data in any reasonable shape. We'll figure out what
            the model needs and transform accordingly.
        y : Optional[np.ndarray]
            Exogenous variables for models that support them
        model_type : ModelTypesWithoutArch
            The model family—each has its own shape expectations
        order : Optional[Union[int, Tuple]]
            Model complexity. We provide sensible defaults when None
        seasonal_order : Optional[tuple]
            For SARIMA models that capture periodic patterns

        Returns
        -------
        fitted_model : Union[BackendToStatsmodelsAdapter, Any]
            The fitted model, wrapped for consistent interface
        residuals : np.ndarray
            Model residuals, carefully extracted and shape-corrected
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
                # Use ternary operator for cleaner code
                X_model = X.flatten() if X.shape[1] == 1 else X[:, 0].flatten()
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

        # Always use backend system directly for better performance and stability
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
                residuals = np.zeros_like(X) if model_type == "var" else np.zeros(len(X_model))

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

        # Return the fitted model wrapped for backward compatibility
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
        Implement whole residual resampling: the simplest bootstrap approach.

        Whole resampling treats each residual as independent, ignoring any
        remaining temporal structure. While this assumption is often violated,
        the method remains useful when model fitting has successfully removed
        serial correlation. We return both indices and values to support
        different use cases—some methods need to track which residuals were
        selected.

        Parameters
        ----------
        residuals : np.ndarray
            Model residuals, ideally white noise after successful fitting
        n_samples : int
            How many residuals to draw. Often matches original series length
        rng : np.random.Generator
            For reproducible randomness—critical for research
        replace : bool
            With replacement is standard, but without can be useful

        Returns
        -------
        indices : np.ndarray
            Which residuals were selected—useful for diagnostics
        resampled_residuals : np.ndarray
            The actual resampled values
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
