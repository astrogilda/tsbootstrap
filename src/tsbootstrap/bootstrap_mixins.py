"""
Bootstrap-specific mixins to reduce code duplication.

This module provides mixins for common bootstrap patterns used across
different bootstrap implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from pydantic import BaseModel

from tsbootstrap.bootstrap_common import BootstrapUtilities

if TYPE_CHECKING:
    from tsbootstrap.time_series_model import TimeSeriesModel


class ModelFittingMixin(BaseModel):
    """Mixin for bootstrap methods that require model fitting."""

    # These attributes are expected to be defined by the class using this mixin
    model_type: str
    order: Optional[object] = None
    seasonal_order: Optional[object] = None
    _fitted_model: Optional[TimeSeriesModel] = None  # type: ignore[assignment]
    _residuals: Optional[np.ndarray] = None  # type: ignore[assignment]

    def _fit_model(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Fit the time series model and compute residuals.

        shared implementation for model fitting that can be used
        by any bootstrap method requiring model fitting.

        Parameters
        ----------
        X : np.ndarray
            The time series data.
        y : Optional[np.ndarray]
            Exogenous variables.
        """
        # Use shared utility for model fitting
        fitted_model, residuals = BootstrapUtilities.fit_time_series_model(
            X=X,
            y=y,
            model_type=self.model_type,
            order=self.order,
            seasonal_order=self.seasonal_order,
        )
        self._fitted_model = fitted_model.model
        self._residuals = residuals

    def _get_fitted_values(self, X: np.ndarray) -> np.ndarray:
        """
        Extract fitted values from the model.

        Parameters
        ----------
        X : np.ndarray
            Original time series data.

        Returns
        -------
        np.ndarray
            Fitted values from the model.
        """
        if self._fitted_model is None:
            raise ValueError("Model not fitted yet")

        if hasattr(self._fitted_model, "fittedvalues"):
            return self._fitted_model.fittedvalues
        else:
            # Fallback: compute fitted values as X - residuals
            X_univariate = X[:, 0] if X.ndim == 2 and X.shape[1] > 1 else X.flatten()

            if self._residuals is None:
                raise ValueError("No residuals available")

            n_samples = len(X)
            return X_univariate - self._residuals[:n_samples]


class ResidualResamplingMixin(BaseModel):
    """Mixin for bootstrap methods that resample residuals."""

    # Expected attributes
    _residuals: Optional[np.ndarray] = None  # type: ignore[assignment]
    rng: Optional[object] = None

    def _resample_residuals_whole(
        self, n_samples: int, replace: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample residuals for whole bootstrap.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        replace : bool, default=True
            Whether to sample with replacement.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Indices and resampled residuals.
        """
        if self._residuals is None:
            raise ValueError("No residuals available for resampling")

        return BootstrapUtilities.resample_residuals_whole(
            residuals=self._residuals,
            n_samples=n_samples,
            rng=self.rng,
            replace=replace,
        )

    def _resample_residuals_block(
        self,
        n_samples: int,
        block_length: int,
        overlap: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample residuals in blocks.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        block_length : int
            Length of blocks for resampling.
        overlap : bool, default=True
            Whether blocks can overlap.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Indices and resampled residuals.
        """
        if self._residuals is None:
            raise ValueError("No residuals available for resampling")

        return BootstrapUtilities.resample_residuals_block(
            residuals=self._residuals,
            n_samples=n_samples,
            block_length=block_length,
            rng=self.rng,
            overlap=overlap,
        )


class TimeSeriesReconstructionMixin(BaseModel):
    """Mixin for reconstructing time series from fitted values and residuals."""

    @staticmethod
    def _reconstruct_series(
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
            Fitted values from the model.
        resampled_residuals : np.ndarray
            Resampled residuals.
        original_shape : tuple
            Shape of the original time series.
        indices : Optional[np.ndarray]
            Indices used for resampling (for block methods).

        Returns
        -------
        np.ndarray
            Reconstructed time series.
        """
        return BootstrapUtilities.reconstruct_time_series(
            fitted_values=fitted_values,
            resampled_residuals=resampled_residuals,
            original_shape=original_shape,
            indices=indices,
        )


class SieveOrderSelectionMixin(BaseModel):
    """Mixin for sieve bootstrap order selection."""

    # Expected attributes
    min_lag: int = 1
    max_lag: Optional[int] = None

    def _select_order(self, X: np.ndarray, c: float = 1.5) -> int:
        """
        Select AR order for sieve bootstrap based on sample size.

        Parameters
        ----------
        X : np.ndarray
            Time series data.
        c : float, default=1.5
            Tuning parameter for order selection.

        Returns
        -------
        int
            Selected AR order.
        """
        n_samples = len(X)

        # Set max_lag based on sample size if not provided
        if self.max_lag is None:
            # Rule of thumb: max_lag = min(10 * log10(n), n/4)
            max_lag = min(int(10 * np.log10(n_samples)), n_samples // 4)
        else:
            max_lag = self.max_lag

        # Ensure min_lag <= max_lag
        max_lag = max(max_lag, self.min_lag)

        # Sieve bootstrap order formula: order = floor(c * n^(1/3))
        sieve_order = int(c * (n_samples ** (1 / 3)))

        # Constrain to the specified range
        return max(self.min_lag, min(sieve_order, max_lag))
