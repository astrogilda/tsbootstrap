"""Protocol definitions for model backends.

This module defines the interface that all model backends must implement,
enabling seamless switching between different time series libraries.
"""

from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ModelBackend(Protocol):
    """Protocol for model fitting backends.

    All backend implementations must conform to this interface to ensure
    compatibility with the tsbootstrap framework.
    """

    def fit(
        self,
        y: np.ndarray,
        X: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> "FittedModelBackend":
        """Fit model to data.

        Parameters
        ----------
        y : np.ndarray
            Target time series data. Shape depends on backend:
            - For sequential backends: (n_obs,)
            - For batch backends: (n_series, n_obs)
        X : np.ndarray, optional
            Exogenous variables. Shape must align with y.
        **kwargs : Any
            Additional backend-specific parameters.

        Returns
        -------
        FittedModelBackend
            Fitted model instance conforming to the protocol.
        """
        ...


@runtime_checkable
class FittedModelBackend(Protocol):
    """Protocol for fitted model instances.

    Provides a unified interface for accessing model parameters,
    residuals, and generating predictions/simulations.
    """

    @property
    def params(self) -> dict[str, Any]:
        """Model parameters in standardized format.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing model parameters. Structure:
            - 'ar': AR coefficients (if applicable)
            - 'ma': MA coefficients (if applicable)
            - 'sigma2': Residual variance
            - Additional model-specific parameters
        """
        ...

    @property
    def residuals(self) -> np.ndarray:
        """Model residuals.

        Returns
        -------
        np.ndarray
            Residuals with shape:
            - Sequential backend: (n_obs,)
            - Batch backend: (n_series, n_obs)
        """
        ...

    @property
    def fitted_values(self) -> np.ndarray:
        """Fitted values from the model.

        Returns
        -------
        np.ndarray
            Fitted values with same shape as residuals.
        """
        ...

    def predict(
        self,
        steps: int,
        X: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Generate point predictions.

        Parameters
        ----------
        steps : int
            Number of steps ahead to predict.
        X : np.ndarray, optional
            Future exogenous variables.
        **kwargs : Any
            Additional backend-specific parameters.

        Returns
        -------
        np.ndarray
            Predictions with shape:
            - Sequential: (steps,)
            - Batch: (n_series, steps)
        """
        ...

    def simulate(
        self,
        steps: int,
        n_paths: int = 1,
        X: Optional[np.ndarray] = None,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Generate simulated paths.

        Parameters
        ----------
        steps : int
            Number of steps to simulate.
        n_paths : int, default=1
            Number of simulation paths per series.
        X : np.ndarray, optional
            Future exogenous variables.
        random_state : int, optional
            Random seed for reproducibility.
        **kwargs : Any
            Additional backend-specific parameters.

        Returns
        -------
        np.ndarray
            Simulated paths with shape:
            - Sequential: (n_paths, steps)
            - Batch: (n_series, n_paths, steps)
        """
        ...

    def get_info_criteria(self) -> dict[str, float]:
        """Get information criteria.

        Returns
        -------
        Dict[str, float]
            Dictionary containing:
            - 'aic': Akaike Information Criterion
            - 'bic': Bayesian Information Criterion
            - 'hqic': Hannan-Quinn Information Criterion (if available)
        """
        ...
