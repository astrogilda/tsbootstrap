"""
Core services for bootstrap operations and time series modeling.

This module provides the essential services that power the bootstrap
architecture. Each service encapsulates a specific domain of functionality,
enabling clean composition and testability. The services here handle the
heavy lifting of time series analysis while maintaining clear interfaces.

The service-oriented design enables:
- Independent testing of each component
- Easy extension with new algorithms
- Performance optimization per service
- Clean dependency injection patterns
"""

from typing import Any, Optional, Tuple, Union

import numpy as np
from arch import arch_model

from tsbootstrap.bootstrap_common import BootstrapUtilities


class ModelFittingService:
    """
    Service for fitting time series models.

    Provides model fitting functionality as a composable service.
    """

    def __init__(self):
        """Initialize the model fitting service."""
        self.utilities = BootstrapUtilities()
        self._fitted_model = None
        self._residuals = None

    def fit_model(
        self,
        X: np.ndarray,
        model_type: str = "ar",
        order: Union[int, Tuple[int, int, int]] = 1,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        **model_kwargs,
    ) -> Tuple[Any, np.ndarray, np.ndarray]:
        """
        Fit a time series model and compute residuals.

        Parameters
        ----------
        X : np.ndarray
            Time series data (n_samples, n_features)
        model_type : str, default="ar"
            Type of model to fit
        order : int or tuple, default=1
            Model order
        seasonal_order : tuple, optional
            Seasonal order for SARIMA models
        **model_kwargs
            Additional arguments for model fitting

        Returns
        -------
        fitted_model : Any
            The fitted model object
        fitted_values : np.ndarray
            Fitted values from the model
        residuals : np.ndarray
            Residuals from the model fit
        """
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Fit model based on type
        if model_type.lower() in ["ar", "arima", "sarima"]:
            # For multivariate data with AR, switch to VAR
            if X.shape[1] > 1 and model_type.lower() == "ar":
                return self.fit_model(X, "var", order, **model_kwargs)

            from statsmodels.tsa.arima.model import ARIMA

            # Handle order parameter
            arima_order = (order, 0, 0) if isinstance(order, int) else order

            # Fit ARIMA model
            arima_kwargs = model_kwargs.copy()
            if seasonal_order is not None:
                arima_kwargs["seasonal_order"] = seasonal_order

            model = ARIMA(X[:, 0], order=arima_order, **arima_kwargs)  # ARIMA expects 1D
            fitted_model = model.fit()
            fitted_values = fitted_model.fittedvalues
            residuals = fitted_model.resid

        elif model_type.lower() == "var":
            from statsmodels.tsa.api import VAR

            # VAR requires multivariate data
            if X.shape[1] == 1:
                # Convert to ARIMA for univariate case
                return self.fit_model(X, "ar", order, **model_kwargs)

            # Fit VAR model
            model = VAR(X)
            fitted_model = model.fit(maxlags=order if isinstance(order, int) else order[0])
            fitted_values = fitted_model.fittedvalues
            residuals = fitted_model.resid

        elif model_type.lower() in ["arch", "garch", "egarch", "tgarch"]:
            # Use arch package for GARCH models
            fitted_model, residuals = self._fit_arch_model(
                X[:, 0], model_type, order, **model_kwargs
            )
            fitted_values = X[:, 0] - residuals

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Store results
        self._fitted_model = fitted_model
        self._residuals = residuals

        return fitted_model, fitted_values, residuals

    def _fit_arch_model(
        self, y: np.ndarray, model_type: str, order: Union[int, Tuple[int, int]], **kwargs
    ) -> Tuple[Any, np.ndarray]:
        """Fit ARCH/GARCH family models."""
        # Create appropriate volatility model
        if model_type.lower() == "arch":
            vol_model = "ARCH"
            vol_params = {"p": order if isinstance(order, int) else order[0]}
        elif model_type.lower() == "garch":
            if isinstance(order, int):
                vol_params = {"p": order, "q": 1}
            else:
                vol_params = {"p": order[0], "q": order[1] if len(order) > 1 else 1}
            vol_model = "GARCH"
        elif model_type.lower() == "egarch":
            if isinstance(order, int):
                vol_params = {"p": order, "q": 1}
            else:
                vol_params = {"p": order[0], "q": order[1] if len(order) > 1 else 1}
            vol_model = "EGARCH"
        elif model_type.lower() == "tgarch":
            if isinstance(order, int):
                vol_params = {"p": order, "q": 1}
            else:
                vol_params = {"p": order[0], "q": order[1] if len(order) > 1 else 1}
            vol_model = "TGARCH"
        else:
            raise ValueError(f"Unknown ARCH model type: {model_type}")

        # Fit model
        model = arch_model(y, vol=vol_model, **vol_params, **kwargs)
        fitted = model.fit(disp="off")

        return fitted, fitted.resid

    @property
    def fitted_model(self):
        """Get the fitted model."""
        if self._fitted_model is None:
            raise ValueError("Model not fitted yet. Call fit_model first.")
        return self._fitted_model

    @property
    def residuals(self):
        """Get the residuals."""
        if self._residuals is None:
            raise ValueError("Model not fitted yet. Call fit_model first.")
        return self._residuals


class ResidualResamplingService:
    """
    Service for resampling residuals.

    Provides residual resampling functionality as a composable service.
    """

    def __init__(self, rng: Optional[np.random.Generator] = None):
        """
        Initialize the residual resampling service.

        Parameters
        ----------
        rng : np.random.Generator, optional
            Random number generator
        """
        self.rng = rng if rng is not None else np.random.default_rng()

    def resample_residuals_whole(
        self, residuals: np.ndarray, n_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Resample residuals using whole (IID) bootstrap.

        Parameters
        ----------
        residuals : np.ndarray
            Residuals to resample (can be 1D or 2D)
        n_samples : int, optional
            Number of samples to generate. If None, uses length of residuals

        Returns
        -------
        np.ndarray
            Resampled residuals
        """
        if n_samples is None:
            n_samples = residuals.shape[0] if residuals.ndim > 1 else len(residuals)

        # Simple random sampling with replacement
        indices = self.rng.integers(
            0, residuals.shape[0] if residuals.ndim > 1 else len(residuals), size=n_samples
        )
        return residuals[indices]

    def resample_residuals_block(
        self, residuals: np.ndarray, block_length: int, n_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Resample residuals using block bootstrap.

        Parameters
        ----------
        residuals : np.ndarray
            Residuals to resample (can be 1D or 2D)
        block_length : int
            Length of blocks
        n_samples : int, optional
            Number of samples to generate

        Returns
        -------
        np.ndarray
            Resampled residuals
        """
        if n_samples is None:
            n_samples = residuals.shape[0] if residuals.ndim > 1 else len(residuals)

        n_residuals = residuals.shape[0] if residuals.ndim > 1 else len(residuals)

        if residuals.ndim == 1:
            # Univariate case
            resampled = []
            while len(resampled) < n_samples:
                # Select random block start
                block_start = self.rng.integers(0, n_residuals - block_length + 1)
                block_end = min(block_start + block_length, n_residuals)

                # Add block to resampled data
                resampled.extend(residuals[block_start:block_end])

            return np.array(resampled[:n_samples])
        else:
            # Multivariate case
            resampled = []
            while len(resampled) < n_samples:
                # Select random block start
                block_start = self.rng.integers(0, n_residuals - block_length + 1)
                block_end = min(block_start + block_length, n_residuals)

                # Add block to resampled data
                block = residuals[block_start:block_end]
                for row in block:
                    if len(resampled) < n_samples:
                        resampled.append(row)

            return np.array(resampled)


class TimeSeriesReconstructionService:
    """
    Service for reconstructing time series from components.

    Provides time series reconstruction functionality.
    """

    @staticmethod
    def reconstruct_time_series(
        fitted_values: np.ndarray, resampled_residuals: np.ndarray
    ) -> np.ndarray:
        """
        Reconstruct time series from fitted values and residuals.

        Parameters
        ----------
        fitted_values : np.ndarray
            Fitted values from the model
        resampled_residuals : np.ndarray
            Resampled residuals

        Returns
        -------
        np.ndarray
            Reconstructed time series
        """
        # Handle both univariate and multivariate cases
        if fitted_values.ndim == 1 and resampled_residuals.ndim == 1:
            # Univariate case
            min_len = min(len(fitted_values), len(resampled_residuals))
            return fitted_values[:min_len] + resampled_residuals[:min_len]
        else:
            # Multivariate case - ensure shapes match
            min_len = min(fitted_values.shape[0], resampled_residuals.shape[0])
            return fitted_values[:min_len] + resampled_residuals[:min_len]


class SieveOrderSelectionService:
    """
    Service for selecting AR order in sieve bootstrap.

    Provides automatic order selection for time series models.
    """

    def __init__(self):
        """Initialize the order selection service."""
        pass

    def _get_criterion_score(self, fitted, criterion: str) -> float:
        """Get the score for the given criterion."""
        criterion_lower = criterion.lower()
        if criterion_lower == "aic":
            return fitted.aic
        elif criterion_lower == "bic":
            return fitted.bic
        elif criterion_lower == "hqic":
            return fitted.hqic
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

    def select_order(
        self, X: np.ndarray, min_lag: int = 1, max_lag: int = 10, criterion: str = "aic"
    ) -> int:
        """
        Select optimal AR order using information criterion.

        Parameters
        ----------
        X : np.ndarray
            Time series data
        min_lag : int, default=1
            Minimum lag to consider
        max_lag : int, default=10
            Maximum lag to consider
        criterion : str, default="aic"
            Information criterion to use ('aic', 'bic', 'hqic')

        Returns
        -------
        int
            Selected order
        """
        from statsmodels.tsa.ar_model import AutoReg

        # Ensure 1D
        if X.ndim > 1:
            X = X[:, 0]

        best_score = np.inf
        best_order = min_lag

        for order in range(min_lag, max_lag + 1):
            try:
                model = AutoReg(X, lags=order)
                fitted = model.fit()
                score = self._get_criterion_score(fitted, criterion)

                if score < best_score:
                    best_score = score
                    best_order = order

            except Exception:
                # Skip orders that fail to fit - this is expected for some orders
                # Log at debug level for diagnostics if needed
                import logging

                logger = logging.getLogger(__name__)
                logger.debug(f"Failed to fit AR model with order {order}", exc_info=True)
                continue

        return best_order
