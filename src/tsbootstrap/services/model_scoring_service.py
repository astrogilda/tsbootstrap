"""Model scoring service for consistent metric calculations across backends.

This module provides a unified scoring interface for all model backends,
supporting various error metrics for both in-sample and out-of-sample evaluation.
"""


import numpy as np


class ModelScoringService:
    """Service for calculating model performance metrics.

    Provides consistent scoring functionality across all backend implementations,
    supporting common time series evaluation metrics.
    """

    def score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric: str = "r2",
    ) -> float:
        """Calculate score between true and predicted values.

        Parameters
        ----------
        y_true : np.ndarray
            True values. Shape: (n_obs,) or (n_obs, n_features)
        y_pred : np.ndarray
            Predicted values. Must have same shape as y_true.
        metric : str, default="r2"
            Scoring metric to use. Options:
            - 'r2': R-squared (coefficient of determination)
            - 'mse': Mean Squared Error
            - 'mae': Mean Absolute Error
            - 'rmse': Root Mean Squared Error
            - 'mape': Mean Absolute Percentage Error

        Returns
        -------
        float
            Score value. Higher is better for r2, lower is better for error metrics.

        Raises
        ------
        ValueError
            If shapes don't match or metric is unknown.
        """
        # Validate inputs
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

        # Flatten if needed for consistent calculations
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred.ravel()

        # Calculate metric
        if metric == "r2":
            return self._r2_score(y_true_flat, y_pred_flat)
        elif metric == "mse":
            return self._mse(y_true_flat, y_pred_flat)
        elif metric == "mae":
            return self._mae(y_true_flat, y_pred_flat)
        elif metric == "rmse":
            return self._rmse(y_true_flat, y_pred_flat)
        elif metric == "mape":
            return self._mape(y_true_flat, y_pred_flat)
        else:
            raise ValueError(
                f"Unknown metric: {metric}. Available: 'r2', 'mse', 'mae', 'rmse', 'mape'"
            )

    def calculate_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Squared Error.

        Convenience method that calls score with metric='mse'.

        Parameters
        ----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values

        Returns
        -------
        float
            Mean Squared Error
        """
        return self.score(y_true, y_pred, metric="mse")

    def calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error.

        Convenience method that calls score with metric='mae'.

        Parameters
        ----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values

        Returns
        -------
        float
            Mean Absolute Error
        """
        return self.score(y_true, y_pred, metric="mae")

    def _r2_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared (coefficient of determination).

        R² = 1 - (SS_res / SS_tot)
        where SS_res = Σ(y_true - y_pred)²
              SS_tot = Σ(y_true - y_mean)²
        """
        # Handle edge cases
        if len(y_true) == 0:
            return np.nan

        # Calculate mean
        y_mean = np.mean(y_true)

        # Total sum of squares
        ss_tot = np.sum((y_true - y_mean) ** 2)

        # Handle constant y_true
        if ss_tot == 0:
            # If predictions are also constant and equal, R² = 1
            # Otherwise R² is undefined (we return 0)
            return 1.0 if np.allclose(y_true, y_pred) else 0.0

        # Residual sum of squares
        ss_res = np.sum((y_true - y_pred) ** 2)

        # R-squared
        r2 = 1 - (ss_res / ss_tot)

        return r2

    def _mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Squared Error."""
        return np.mean((y_true - y_pred) ** 2)

    def _mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return np.mean(np.abs(y_true - y_pred))

    def _rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        return np.sqrt(self._mse(y_true, y_pred))

    def _mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error.

        MAPE = 100 * mean(|y_true - y_pred| / |y_true|)

        Note: Excludes points where y_true = 0 to avoid division by zero.
        """
        # Avoid division by zero
        mask = y_true != 0

        if not np.any(mask):
            # All values are zero
            return np.inf

        # Calculate MAPE only for non-zero true values
        abs_percentage_errors = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
        mape = np.mean(abs_percentage_errors) * 100

        return mape
