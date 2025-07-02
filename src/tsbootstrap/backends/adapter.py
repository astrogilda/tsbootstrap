"""Adapter for integrating backends with legacy TimeSeriesModel.

This module provides compatibility between the new backend architecture
and the existing TimeSeriesModel API, ensuring backward compatibility
while enabling performance improvements.
"""

from typing import Any, Optional, Union

import numpy as np

from tsbootstrap.backends.factory import create_backend
from tsbootstrap.backends.protocol import FittedModelBackend


class BackendToStatsmodelsAdapter:
    """Adapts FittedModelBackend to statsmodels ResultsWrapper interface.

    This adapter allows the new backend architecture to seamlessly
    integrate with existing code that expects statsmodels result objects.

    Parameters
    ----------
    fitted_backend : FittedModelBackend
        The fitted backend instance to adapt.
    model_type : str
        Type of model for proper adaptation.
    """

    def __init__(self, fitted_backend: FittedModelBackend, model_type: str) -> None:
        self._backend = fitted_backend
        self._model_type = model_type.upper()
        self._params_dict = fitted_backend.params

        # Extract key parameters
        if "series_params" in self._params_dict:
            # Multiple series - use first for compatibility
            self._params_dict = self._params_dict["series_params"][0]

    @property
    def params(self) -> Union[np.ndarray, dict[str, Any]]:
        """Model parameters in statsmodels format."""
        # Return parameters based on model type
        if self._model_type in ["AR", "ARIMA", "SARIMA"]:
            # Combine AR and MA parameters
            ar_params = self._params_dict.get("ar", np.array([]))
            ma_params = self._params_dict.get("ma", np.array([]))

            # Return as dict with labeled parameters
            params = {}
            for i, coef in enumerate(ar_params):
                params[f"ar.L{i+1}"] = coef
            for i, coef in enumerate(ma_params):
                params[f"ma.L{i+1}"] = coef

            # Add sigma2 if present
            if "sigma2" in self._params_dict:
                params["sigma2"] = self._params_dict["sigma2"]

            return params
        # Return raw params dict for other models
        return self._params_dict

    @property
    def resid(self) -> np.ndarray:
        """Residuals in statsmodels format."""
        return self._backend.residuals

    @property
    def fittedvalues(self) -> np.ndarray:
        """Fitted values in statsmodels format."""
        return self._backend.fitted_values

    @property
    def aic(self) -> float:
        """AIC in statsmodels format."""
        criteria = self._backend.get_info_criteria()
        return criteria.get("aic", np.nan)

    @property
    def bic(self) -> float:
        """BIC in statsmodels format."""
        criteria = self._backend.get_info_criteria()
        return criteria.get("bic", np.nan)

    @property
    def hqic(self) -> float:
        """HQIC in statsmodels format."""
        criteria = self._backend.get_info_criteria()
        return criteria.get("hqic", np.nan)

    @property
    def sigma2(self) -> float:
        """Residual variance."""
        return self._params_dict.get("sigma2", 1.0)

    def forecast(
        self, steps: int = 1, exog: Optional[np.ndarray] = None, **kwargs: Any
    ) -> np.ndarray:
        """Generate forecasts in statsmodels format."""
        return self._backend.predict(steps=steps, X=exog, **kwargs)

    def predict(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        exog: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Generate predictions in statsmodels format.

        For compatibility with statsmodels, predict returns in-sample predictions
        when start/end are within the training range.
        """
        if start is None and end is None:
            # Return fitted values for in-sample prediction
            return self._backend.fitted_values
        elif start is not None and end is not None:
            # Return slice of fitted values if within training range
            return self._backend.fitted_values[start : end + 1]
        else:
            # For out-of-sample, use forecast
            steps = 1 if end is None else end - (start or 0) + 1
            return self._backend.predict(steps=steps, X=exog, **kwargs)

    def simulate(
        self,
        nsimulations: int,
        repetitions: int = 1,
        exog: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Generate simulations in statsmodels format."""
        # Map statsmodels parameters to backend
        return self._backend.simulate(
            steps=nsimulations,
            n_paths=repetitions,
            X=exog,
            **kwargs,
        )

    def summary(self) -> str:
        """Return summary in statsmodels format."""
        # Basic summary information
        summary_str = f"{self._model_type} Model Results\n"
        summary_str += "=" * 40 + "\n"
        summary_str += f"AIC: {self.aic:.4f}\n"
        summary_str += f"BIC: {self.bic:.4f}\n"
        summary_str += f"HQIC: {self.hqic:.4f}\n"
        summary_str += f"Sigma2: {self.sigma2:.4f}\n"
        return summary_str

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attributes to backend."""
        return getattr(self._backend, name)


def fit_with_backend(
    model_type: str,
    endog: np.ndarray,
    exog: Optional[np.ndarray] = None,
    order: Optional[Union[int, tuple[int, ...]]] = None,
    seasonal_order: Optional[tuple[int, int, int, int]] = None,
    force_backend: Optional[str] = None,
    return_backend: bool = False,
    **kwargs: Any,
) -> Union[BackendToStatsmodelsAdapter, FittedModelBackend]:
    """Fit a time series model using the backend architecture.

    This function provides a high-level interface for fitting time series
    models using either statsforecast or statsmodels backends, with
    automatic selection based on feature flags.

    Parameters
    ----------
    model_type : str
        Type of model ('AR', 'ARIMA', 'SARIMA', 'VAR').
    endog : np.ndarray
        Endogenous variable (time series data).
    exog : np.ndarray, optional
        Exogenous variables.
    order : Union[int, tuple[int, ...]], optional
        Model order.
    seasonal_order : tuple[int, int, int, int], optional
        Seasonal order for SARIMA.
    force_backend : str, optional
        Force specific backend.
    return_backend : bool, default False
        If True, return FittedModelBackend directly.
        If False, return adapted statsmodels-compatible object.
    **kwargs : Any
        Additional model parameters.

    Returns
    -------
    Union[BackendToStatsmodelsAdapter, FittedModelBackend]
        Fitted model, either adapted or raw backend.
    """
    # Create backend
    backend = create_backend(
        model_type=model_type,
        order=order,
        seasonal_order=seasonal_order,
        force_backend=force_backend,
        **kwargs,
    )

    # Fit the model
    fitted_backend = backend.fit(endog, exog, **kwargs)

    # Return appropriate format
    if return_backend:
        return fitted_backend
    return BackendToStatsmodelsAdapter(fitted_backend, model_type)
