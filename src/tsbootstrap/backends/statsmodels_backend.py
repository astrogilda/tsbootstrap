"""StatsModels backend implementation for legacy support and VAR models.

This module provides a backend using statsmodels, maintaining compatibility
with existing functionality and supporting model types not available in
statsforecast (e.g., VAR models).
"""

from typing import Any, Optional, Union

import numpy as np
from statsmodels.tsa.ar_model import AutoReg, AutoRegResultsWrapper
from statsmodels.tsa.arima.model import ARIMA, ARIMAResultsWrapper
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper
from statsmodels.tsa.vector_ar.var_model import VAR, VARResultsWrapper


class StatsModelsBackend:
    """Backend implementation using statsmodels library.

    This backend provides compatibility with the existing statsmodels-based
    implementation and supports model types not available in statsforecast,
    particularly VAR models.

    Parameters
    ----------
    model_type : str
        Type of model ('AR', 'ARIMA', 'SARIMA', 'VAR').
    order : Union[int, Tuple[int, ...]]
        Model order specification.
    seasonal_order : Tuple[int, int, int, int], optional
        Seasonal order for SARIMA models.
    **kwargs : Any
        Additional model-specific parameters.
    """

    def __init__(
        self,
        model_type: str,
        order: Union[int, tuple[int, ...]],
        seasonal_order: Optional[tuple[int, int, int, int]] = None,
        **kwargs: Any,
    ):
        self.model_type = model_type.upper()
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_params = kwargs
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        valid_types = ["AR", "ARIMA", "SARIMA", "VAR"]
        if self.model_type not in valid_types:
            raise ValueError(
                f"Invalid model type: {self.model_type}. Must be one of {valid_types}",
            )

        if self.model_type == "SARIMA" and self.seasonal_order is None:
            raise ValueError("seasonal_order required for SARIMA models")

    def fit(
        self,
        y: np.ndarray,
        X: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> "StatsModelsFittedBackend":
        """Fit model to data.

        Note: StatsModels does not support batch fitting, so for multiple
        series (y.shape[0] > 1), models are fit sequentially.

        Parameters
        ----------
        y : np.ndarray
            Time series data. Shape (n_obs,) for single series or
            (n_series, n_obs) for multiple series.
        X : np.ndarray, optional
            Exogenous variables.
        **kwargs : Any
            Additional fitting parameters.

        Returns
        -------
        StatsModelsFittedBackend
            Fitted model instance.
        """
        # Handle both single and multiple series
        if y.ndim == 1:
            y = y.reshape(1, -1)

        n_series, n_obs = y.shape

        # Fit models (sequentially for statsmodels)
        fitted_models = []
        for i in range(n_series):
            series_data = y[i, :]
            series_exog = X[i, :] if X is not None and X.ndim > 1 else X

            model = self._create_model(series_data, series_exog)

            # Fit with appropriate method
            if self.model_type == "VAR":
                # VAR models need multivariate data
                if n_series == 1:
                    raise ValueError(
                        "VAR models require multivariate time series data",
                    )
                # For VAR, we fit on the full multivariate series
                if i == 0:  # Only fit once for VAR
                    fitted = model.fit(**kwargs)
                    fitted_models.append(fitted)
                break
            fitted = model.fit(**kwargs)
            fitted_models.append(fitted)

        return StatsModelsFittedBackend(
            fitted_models=fitted_models,
            model_type=self.model_type,
            n_series=n_series,
        )

    def _create_model(self, y: np.ndarray, X: Optional[np.ndarray] = None):
        """Create appropriate statsmodels model instance."""
        if self.model_type == "AR":
            return AutoReg(
                y,
                lags=self.order,
                exog=X,
                **self.model_params,
            )
        if self.model_type == "ARIMA":
            return ARIMA(
                y,
                order=self.order,
                exog=X,
                **self.model_params,
            )
        if self.model_type == "SARIMA":
            return SARIMAX(
                y,
                order=self.order,
                seasonal_order=self.seasonal_order,
                exog=X,
                **self.model_params,
            )
        if self.model_type == "VAR":
            # VAR requires full multivariate series
            return VAR(y.T, exog=X, **self.model_params)
        raise ValueError(f"Unknown model type: {self.model_type}")


class StatsModelsFittedBackend:
    """Fitted model backend for statsmodels.

    Wraps statsmodels fitted model objects to conform to the
    FittedModelBackend protocol.
    """

    def __init__(
        self,
        fitted_models: list[Any],
        model_type: str,
        n_series: int,
    ):
        self._fitted_models = fitted_models
        self._model_type = model_type
        self._n_series = n_series

    @property
    def params(self) -> dict[str, Any]:
        """Extract model parameters in standardized format."""
        if self._n_series == 1 or self._model_type == "VAR":
            return self._extract_params(self._fitted_models[0])
        return {
            "series_params": [self._extract_params(model) for model in self._fitted_models],
        }

    def _extract_params(self, fitted_model) -> dict[str, Any]:
        """Extract parameters from single fitted model."""
        params = {"model_type": self._model_type}

        if isinstance(fitted_model, AutoRegResultsWrapper):
            params.update(
                {
                    "ar": fitted_model.params,
                    "sigma2": fitted_model.sigma2,
                    "order": fitted_model.model.lags,
                }
            )
        elif isinstance(fitted_model, (ARIMAResultsWrapper, SARIMAXResultsWrapper)):
            # Extract ARIMA parameters directly from params
            ar_params = []
            ma_params = []

            # Extract based on parameter names
            for key, value in fitted_model.params.items():
                if key.startswith("ar.L"):
                    ar_params.append((int(key[4:]), value))  # Extract lag number
                elif key.startswith("ma.L"):
                    ma_params.append((int(key[4:]), value))  # Extract lag number

            # Sort by lag number and extract values
            ar_params.sort(key=lambda x: x[0])
            ma_params.sort(key=lambda x: x[0])

            ar_values = [val for _, val in ar_params]
            ma_values = [val for _, val in ma_params]

            # Get order from model specification
            if hasattr(fitted_model, "model"):
                if hasattr(fitted_model.model, "order"):
                    order = fitted_model.model.order  # (p, d, q)
                else:
                    # Default fallback
                    order = (len(ar_values), 0, len(ma_values))
            else:
                order = (len(ar_values), 0, len(ma_values))

            params.update(
                {
                    "ar": np.array(ar_values),
                    "ma": np.array(ma_values),
                    "d": order[1] if len(order) > 1 else 0,
                    "sigma2": fitted_model.scale if hasattr(fitted_model, "scale") else 1.0,
                    "order": order,
                }
            )

            # Seasonal parameters for SARIMA
            if hasattr(fitted_model.model, "seasonal_order"):
                params["seasonal_order"] = fitted_model.model.seasonal_order

        elif isinstance(fitted_model, VARResultsWrapper):
            params.update(
                {
                    "coefs": fitted_model.coefs,
                    "sigma_u": fitted_model.sigma_u,
                    "order": fitted_model.k_ar,
                }
            )

        return params

    @property
    def residuals(self) -> np.ndarray:
        """Return model residuals."""
        if self._model_type == "VAR":
            return self._fitted_models[0].resid.T  # Transpose for consistency
        if self._n_series == 1:
            return self._fitted_models[0].resid
        return np.array([model.resid for model in self._fitted_models])

    @property
    def fitted_values(self) -> np.ndarray:
        """Return fitted values."""
        if self._model_type == "VAR":
            return self._fitted_models[0].fittedvalues.T
        if self._n_series == 1:
            return self._fitted_models[0].fittedvalues
        return np.array([model.fittedvalues for model in self._fitted_models])

    def predict(
        self,
        steps: int,
        X: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Generate predictions using statsmodels."""
        if self._model_type == "VAR":
            # VAR prediction
            forecast = self._fitted_models[0].forecast(
                self._fitted_models[0].endog[-self._fitted_models[0].k_ar :],
                steps,
            )
            return forecast.T  # Transpose for consistency
        if self._n_series == 1:
            # Single series prediction
            return self._fitted_models[0].forecast(steps=steps, exog=X)
        # Multiple series predictions
        predictions = []
        for i, model in enumerate(self._fitted_models):
            exog_i = X[i] if X is not None and X.ndim > 1 else X
            pred = model.forecast(steps=steps, exog=exog_i)
            predictions.append(pred)
        return np.array(predictions)

    def simulate(
        self,
        steps: int,
        n_paths: int = 1,
        X: Optional[np.ndarray] = None,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Generate simulated paths using statsmodels."""
        if random_state is not None:
            np.random.seed(random_state)

        if self._model_type == "VAR":
            # VAR simulation - returns (steps, n_vars) for each path
            simulations = []
            for _ in range(n_paths):
                sim = self._fitted_models[0].simulate_var(steps)
                simulations.append(sim.T)  # Transpose for consistency
            return np.array(simulations).transpose(1, 0, 2)  # (n_vars, n_paths, steps)

        if self._n_series == 1:
            # Single series simulation
            model = self._fitted_models[0]
            simulations = []

            for _ in range(n_paths):
                if hasattr(model, "simulate"):
                    sim = model.simulate(
                        nsimulations=steps,
                        exog=X,
                        **kwargs,
                    )
                else:
                    # Fallback for models without simulate method
                    # Generate using model parameters
                    sim = self._simulate_from_params(
                        self._extract_params(model),
                        steps,
                    )
                simulations.append(sim)

            return np.array(simulations)
        # Multiple series simulation
        all_simulations = []
        for model in self._fitted_models:
            series_sims = []
            for _ in range(n_paths):
                if hasattr(model, "simulate"):
                    sim = model.simulate(nsimulations=steps, exog=X)
                else:
                    sim = self._simulate_from_params(
                        self._extract_params(model),
                        steps,
                    )
                series_sims.append(sim)
            all_simulations.append(np.array(series_sims))

        return np.array(all_simulations)

    def _simulate_from_params(self, params: dict[str, Any], steps: int) -> np.ndarray:
        """Simulate from extracted parameters when simulate method not available."""
        # Simple AR simulation as fallback
        ar_coefs = params.get("ar", np.array([]))
        sigma = np.sqrt(params.get("sigma2", 1.0))

        # Generate innovations
        innovations = np.random.normal(0, sigma, steps + 100)

        # Apply AR filter if coefficients exist
        if len(ar_coefs) > 0:
            from scipy import signal

            ar_filt = np.r_[1, -ar_coefs]
            series = signal.lfilter([1], ar_filt, innovations)
        else:
            series = innovations

        return series[-steps:]

    def get_info_criteria(self) -> dict[str, float]:
        """Get information criteria from fitted models."""
        if self._n_series == 1 or self._model_type == "VAR":
            model = self._fitted_models[0]
            return {
                "aic": getattr(model, "aic", np.nan),
                "bic": getattr(model, "bic", np.nan),
                "hqic": getattr(model, "hqic", np.nan),
            }
        # Return criteria for all series
        criteria = []
        for model in self._fitted_models:
            criteria.append(
                {
                    "aic": getattr(model, "aic", np.nan),
                    "bic": getattr(model, "bic", np.nan),
                    "hqic": getattr(model, "hqic", np.nan),
                }
            )
        return {"series_criteria": criteria}
