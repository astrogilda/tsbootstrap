"""
Scoring methods for TSFit class.
"""

from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_is_fitted


class TSFitScorers:
    """Mixin class providing scoring methods for TSFit."""

    def score(
        self,
        X: Union[np.ndarray, NDArray[np.float64]],
        y: Optional[Union[np.ndarray, NDArray[np.float64]]] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        dynamic: bool = False,
        **kwargs,
    ) -> float:
        """
        Calculate the R² score of the model predictions.

        Parameters
        ----------
        X : Union[np.ndarray, NDArray[np.float64]]
            Time series data (used as ground truth).
        y : Optional[Union[np.ndarray, NDArray[np.float64]]], default=None
            Not used, present for API consistency.
        start : Optional[int], default=None
            The start of the prediction period.
        end : Optional[int], default=None
            The end of the prediction period.
        dynamic : bool, default=False
            Whether to use dynamic prediction.
        **kwargs
            Additional keyword arguments passed to predict method.

        Returns
        -------
        float
            The R² score of the predictions.

        Raises
        ------
        NotFittedError
            If the model has not been fitted yet.
        """
        check_is_fitted(self, "model")

        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Get predictions
        predictions = self.predict(X=X, start=start, end=end, dynamic=dynamic, **kwargs)

        # Determine the actual range of data to compare
        if start is None:
            start = 0
        if end is None:
            end = len(X) - 1

        # Extract the relevant portion of X for comparison
        X_subset = X[start : end + 1]

        # Ensure shapes match
        if X_subset.shape != predictions.shape and X_subset.shape[0] == predictions.shape[0]:
            # Handle case where predictions might be 1D
            if predictions.shape[1] == 1 and X_subset.shape[1] > 1:
                # Use only first column of X for univariate predictions
                X_subset = X_subset[:, 0:1]
            elif X_subset.shape[1] == 1 and predictions.shape[1] > 1:
                # This shouldn't happen, but handle it gracefully
                predictions = predictions[:, 0:1]

        # Calculate R² score
        return r2_score(X_subset, predictions)
