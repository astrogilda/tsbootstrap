"""
Prediction methods for TSFit class.
"""

from typing import Optional, Union

import numpy as np
from numpy import ndarray
from numpy.typing import NDArray
from sklearn.utils.validation import NotFittedError, check_is_fitted


class TSFitPredictors:
    """Mixin class providing prediction methods for TSFit."""

    def predict(
        self,
        X_history: Optional[Union[np.ndarray, NDArray[np.float64]]] = None,
        n_steps: int = 1,
        exog: Optional[Union[np.ndarray, NDArray[np.float64]]] = None,
        X: Optional[Union[np.ndarray, NDArray[np.float64]]] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        dynamic: bool = False,
        **kwargs,
    ) -> ndarray:
        """
        Generate predictions from the fitted model.

        This method supports two interfaces:
        1. New interface: X, start, end, dynamic
        2. Old interface: X_history, n_steps

        Parameters
        ----------
        X_history : Optional[Union[np.ndarray, NDArray[np.float64]]], default=None
            Historical data for prediction (old interface).
        n_steps : int, default=1
            Number of steps to predict (old interface).
        exog : Optional[Union[np.ndarray, NDArray[np.float64]]], default=None
            Exogenous variables for prediction.
        X : Optional[Union[np.ndarray, NDArray[np.float64]]], default=None
            Not used, present for API consistency (new interface).
        start : Optional[int], default=None
            The start of the prediction period (new interface).
        end : Optional[int], default=None
            The end of the prediction period (new interface).
        dynamic : bool, default=False
            Whether to use dynamic prediction (new interface).
        **kwargs
            Additional keyword arguments passed to the model's predict method.

        Returns
        -------
        ndarray
            The predicted values.

        Raises
        ------
        NotFittedError
            If the model has not been fitted yet.
        """
        check_is_fitted(self, "model")
        if self.model is None:
            raise NotFittedError("This TSFit instance is not fitted yet.")

        # Check which interface is being used
        if X_history is not None:
            # Old interface
            if not isinstance(X_history, np.ndarray):
                X_history = np.asarray(X_history)
            if exog is not None and not isinstance(exog, np.ndarray):
                exog = np.asarray(exog)

            if self.model_type == "var":
                predictions = self.model.forecast(y=X_history, steps=n_steps, exog_future=exog)
            elif self.model_type == "arch":
                forecast_result = self.model.forecast(
                    horizon=n_steps, x=exog, method="analytic"
                ).mean
                predictions = forecast_result.values.T
                if predictions.shape[1] == 1:
                    predictions = predictions.ravel()
                    return predictions  # Return early to avoid reshaping
            elif self.model_type in ["ar", "arima", "sarima"]:
                predictions = self.model.forecast(steps=n_steps, exog=exog)
            else:
                raise ValueError(f"Unsupported model_type for predict: {self.model_type}")
        else:
            # New interface (backward compatible)
            # Set default values for start and end if not provided
            if hasattr(self.model, "nobs"):
                n_obs = self.model.nobs
            elif hasattr(self.model, "_nobs"):
                n_obs = self.model._nobs
            else:
                # For ARCH models
                n_obs = len(self.model.resid)

            if start is None:
                start = 0
            if end is None:
                end = n_obs - 1

            # Generate predictions based on model type
            if self.model_type == "var":
                # VAR models have a different prediction interface
                steps = end - start + 1
                predictions = self.model.forecast(self.model.endog[-self.order :], steps=steps)
            elif self.model_type == "arch":
                # ARCH models use forecast method
                forecast = self.model.forecast(horizon=end - start + 1)
                predictions = forecast.mean.values.flatten()
            else:
                # Other models use predict method
                predictions = self.model.predict(
                    start=start, end=end, dynamic=dynamic, exog=exog, **kwargs
                )

        # Ensure predictions are numpy array
        if hasattr(predictions, "values"):
            predictions = predictions.values

        # Ensure consistent output shape
        if isinstance(predictions, np.ndarray):
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
            elif predictions.ndim > 2:
                # Handle case where predictions might be 3D
                predictions = predictions.reshape(predictions.shape[0], -1)

        return predictions
