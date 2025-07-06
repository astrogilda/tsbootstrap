"""
Rescaling service for numerical stability in time series models.

This service provides standardized data rescaling functionality to ensure
numerical stability across different backends. We implement rescaling to
handle extreme data ranges that could cause numerical issues during model
fitting, while preserving the statistical properties of the time series.

The rescaling approach uses mean-centering and variance normalization,
which maintains the autocorrelation structure essential for time series
models while improving numerical conditioning.
"""

from typing import Dict, Tuple

import numpy as np


class RescalingService:
    """
    Service providing data rescaling capabilities for numerical stability.

    This service implements intelligent rescaling that preserves time series
    properties while ensuring numerical stability. We automatically detect
    when rescaling is beneficial based on data characteristics and model
    requirements.

    The implementation follows the principle of transparent rescaling—all
    transformations are reversible, ensuring that predictions and parameters
    can be interpreted in the original scale.
    """

    def check_if_rescale_needed(self, data: np.ndarray) -> Tuple[bool, Dict[str, float]]:
        """
        Determine if data rescaling would improve numerical stability.

        We analyze the data range and magnitude to identify potential numerical
        issues. Large ranges or extreme values can cause convergence problems
        or precision loss in optimization algorithms.

        Parameters
        ----------
        data : np.ndarray
            Time series data to analyze

        Returns
        -------
        needs_rescaling : bool
            True if rescaling is recommended
        rescale_factors : dict
            Dictionary containing scale and shift parameters
        """
        # Compute data statistics
        data_range = np.ptp(data)
        data_mean = np.mean(data)
        data_std = np.std(data)
        data_abs_mean = np.mean(np.abs(data))

        # Determine if rescaling needed based on multiple criteria
        needs_rescaling = bool(
            data_range > 1000
            or data_abs_mean < 0.001  # Large range can cause numerical issues
            or data_abs_mean > 1e6  # Very small values lose precision
            or data_std < 1e-6  # Very large values cause overflow
            or data_std  # Near-constant series need scaling
            > 1e6  # Extreme variance needs normalization
        )

        rescale_factors = {}
        if needs_rescaling:
            # Use robust scaling to handle outliers
            rescale_factors["shift"] = float(data_mean)
            rescale_factors["scale"] = float(max(data_std, 1e-8))  # Avoid division by zero

        return needs_rescaling, rescale_factors

    def rescale_data(self, data: np.ndarray, rescale_factors: Dict[str, float]) -> np.ndarray:
        """
        Apply rescaling transformation to improve numerical stability.

        We use standardization (z-score normalization) which preserves the
        autocorrelation structure while improving numerical properties. This
        transformation is particularly effective for gradient-based optimization.

        Parameters
        ----------
        data : np.ndarray
            Data to rescale
        rescale_factors : dict
            Dictionary with 'scale' and 'shift' parameters

        Returns
        -------
        np.ndarray
            Rescaled data with improved numerical properties
        """
        if not rescale_factors:
            return data

        shift = rescale_factors.get("shift", 0.0)
        scale = rescale_factors.get("scale", 1.0)

        # Standardize: (x - mean) / std
        return (data - shift) / scale

    def rescale_back_data(self, data: np.ndarray, rescale_factors: Dict[str, float]) -> np.ndarray:
        """
        Reverse the rescaling transformation to original scale.

        This ensures that all outputs (predictions, fitted values, parameters)
        are interpretable in the original data scale. We maintain full numerical
        precision during the back-transformation.

        Parameters
        ----------
        data : np.ndarray
            Rescaled data to transform back
        rescale_factors : dict
            Dictionary with 'scale' and 'shift' parameters

        Returns
        -------
        np.ndarray
            Data in original scale
        """
        if not rescale_factors:
            return data

        shift = rescale_factors.get("shift", 0.0)
        scale = rescale_factors.get("scale", 1.0)

        # Reverse standardization: x * std + mean
        return data * scale + shift

    def rescale_residuals(
        self, residuals: np.ndarray, rescale_factors: Dict[str, float]
    ) -> np.ndarray:
        """
        Rescale residuals accounting for scale but not shift.

        Residuals represent deviations from fitted values, so they need only
        scale adjustment, not mean-shifting. This preserves their zero-mean
        property while adjusting for the scale transformation.

        Parameters
        ----------
        residuals : np.ndarray
            Model residuals in transformed scale
        rescale_factors : dict
            Dictionary with 'scale' parameter

        Returns
        -------
        np.ndarray
            Residuals in original scale
        """
        if not rescale_factors:
            return residuals

        scale = rescale_factors.get("scale", 1.0)

        # Residuals only need scale adjustment
        return residuals * scale

    def rescale_parameters(self, params: Dict, rescale_factors: Dict[str, float]) -> Dict:
        """
        Adjust model parameters for rescaling effects.

        Some parameters (like innovation variance) need adjustment when data
        is rescaled. This method handles parameter transformations to ensure
        correct interpretation in the original scale.

        Parameters
        ----------
        params : dict
            Model parameters in rescaled space
        rescale_factors : dict
            Dictionary with rescaling parameters

        Returns
        -------
        dict
            Parameters adjusted for original scale
        """
        if not rescale_factors:
            return params

        adjusted_params = params.copy()
        scale = rescale_factors.get("scale", 1.0)

        # Adjust variance parameters
        if "sigma2" in adjusted_params:
            adjusted_params["sigma2"] = adjusted_params["sigma2"] * (scale**2)

        # Note: AR and MA coefficients don't need adjustment for standardization
        # as they operate on the standardized scale

        return adjusted_params
