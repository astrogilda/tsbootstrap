"""
StatsModels backend: Bridging classical econometrics with modern architecture.

This module represents a critical architectural component in our backend system,
providing comprehensive support for classical time series models through the
statsmodels library. While newer backends offer performance advantages for certain
model types, statsmodels remains indispensable for its breadth of econometric
methods and mature implementations.

We maintain this backend for several compelling reasons: VAR models for
multivariate analysis, ARCH/GARCH for volatility modeling, and the extensive
diagnostic tools that statsmodels provides. The implementation follows our
backend protocol precisely, ensuring seamless interchangeability while preserving
the unique capabilities that make statsmodels valuable for rigorous time series
analysis.
"""

from typing import Any, Optional, Union

import numpy as np
from arch import arch_model
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR

from tsbootstrap.backends.stationarity_mixin import StationarityMixin
from tsbootstrap.services.model_scoring_service import ModelScoringService
from tsbootstrap.services.tsfit_services import TSFitHelperService


class StatsModelsBackend:
    """
    Comprehensive statsmodels integration for advanced time series modeling.

    This backend serves as the foundation for sophisticated econometric analyses,
    providing access to statsmodels' extensive model catalog. We've carefully
    wrapped each model type to present a consistent interface while preserving
    the unique capabilities that make statsmodels essential for certain analyses.

    The implementation handles the subtle differences between model APIs, parameter
    conventions, and output formats across the statsmodels ecosystem. This
    abstraction enables users to leverage advanced models without navigating the
    complexities of individual implementations.

    Parameters
    ----------
    model_type : str
        Model specification: 'AR' for autoregressive, 'ARIMA' for integrated
        models, 'SARIMA' for seasonal variants, 'VAR' for vector autoregression,
        or 'ARCH' for volatility modeling. Each type activates specialized
        handling for that model family.

    order : Union[int, Tuple[int, ...]]
        Model order parameters. Format varies by model type: single integer
        for AR/VAR/ARCH, tuple (p,d,q) for ARIMA, following standard conventions.

    seasonal_order : Tuple[int, int, int, int], optional
        Seasonal specification (P,D,Q,s) for SARIMA models. Required only
        for seasonal models, where s represents the seasonal period.

    **kwargs : Any
        Model-specific parameters passed through to the underlying implementation.
        Enables access to advanced features while maintaining interface simplicity.
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
        valid_types = ["AR", "ARIMA", "SARIMA", "VAR", "ARCH"]
        if self.model_type not in valid_types:
            raise ValueError(
                f"Model type '{self.model_type}' is not supported by this backend. "
                f"Available models are: {', '.join(valid_types)}. "
                f"Each model type provides specific capabilities - AR for simple "
                f"autoregression, ARIMA for integrated series, SARIMA for seasonal "
                f"patterns, VAR for multivariate analysis, and ARCH for volatility."
            )

        if self.model_type == "SARIMA" and self.seasonal_order is None:
            raise ValueError(
                "SARIMA models require seasonal_order specification in format "
                "(P, D, Q, s) where P=seasonal AR order, D=seasonal differences, "
                "Q=seasonal MA order, and s=seasonal period (e.g., 12 for monthly)."
            )

        # seasonal_order only valid for SARIMA
        if self.model_type != "SARIMA" and self.seasonal_order is not None:
            raise ValueError(
                f"seasonal_order is only valid for SARIMA models, not {self.model_type}"
            )

        # VAR models require integer order
        if self.model_type == "VAR":
            # Accept numpy integers as well as Python ints
            if not isinstance(self.order, (int, np.integer)):
                raise TypeError(
                    f"Order must be an integer for VAR model. Got {type(self.order).__name__}."
                )
            # Convert to Python int to avoid issues downstream
            self.order = int(self.order)

        # ARCH models require integer order
        if self.model_type == "ARCH":
            # Accept numpy integers as well as Python ints
            if not isinstance(self.order, (int, np.integer)):
                raise TypeError(
                    f"Order must be an integer for ARCH model. Got {type(self.order).__name__}."
                )
            # Convert to Python int to avoid issues downstream
            self.order = int(self.order)

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        dict
            Parameter names mapped to their values.
        """
        return {
            "model_type": self.model_type,
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            **self.model_params,
        }

    def set_params(self, **params) -> "StatsModelsBackend":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        StatsModelsBackend
            Self, for method chaining.
        """
        for key, value in params.items():
            if key == "model_type":
                self.model_type = value.upper()
            elif key == "order":
                self.order = value
            elif key == "seasonal_order":
                self.seasonal_order = value
            else:
                self.model_params[key] = value
        self._validate_inputs()
        return self

    def fit(
        self,
        y: np.ndarray,
        X: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> "StatsModelsBackend":
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

        # Fit models
        fitted_models = []

        if self.model_type == "VAR":
            # VAR models need multivariate data
            if n_series == 1:
                raise ValueError(
                    "VAR (Vector Autoregression) models require multivariate time series data "
                    "with at least 2 series to capture cross-series dynamics. Received only 1 series. "
                    "For univariate analysis, consider using AR, ARIMA, or SARIMA models instead."
                )
            # For VAR, we pass all series at once
            model = self._create_model(y, X)
            fitted = model.fit(**kwargs)
            fitted_models.append(fitted)
        else:
            # For univariate models, fit each series separately
            for i in range(n_series):
                series_data = y[i, :]
                # Handle exogenous variables properly
                if X is not None:
                    if X.ndim == 1:
                        series_exog = X
                    elif n_series == 1:
                        # If single series but X is 2D (n_obs, n_features), use it as is
                        series_exog = X
                    else:
                        # Multiple series, X should be (n_series, n_obs, n_features)
                        series_exog = X[i, :]
                else:
                    series_exog = None

                model = self._create_model(series_data, series_exog)
                # Filter out model creation parameters from fit kwargs
                if self.model_type == "ARCH":
                    fit_kwargs = {
                        k: v for k, v in kwargs.items() if k not in ["p", "q", "arch_model_type"]
                    }
                else:
                    fit_kwargs = kwargs
                fitted = model.fit(**fit_kwargs)
                fitted_models.append(fitted)

        return StatsModelsFittedBackend(
            fitted_models=fitted_models,
            model_type=self.model_type,
            n_series=n_series,
            y=y,
            X=X,
        )

    def _create_model(self, y: np.ndarray, X: Optional[np.ndarray] = None):
        """Create appropriate statsmodels model instance."""
        if self.model_type == "AR":
            # Handle both int and tuple order formats
            ar_order = self.order[0] if isinstance(self.order, tuple) else self.order
            return AutoReg(
                y,
                lags=ar_order,
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
            # y should already be shape (n_vars, n_obs)
            return VAR(y.T if y.ndim == 2 else y, exog=X, **self.model_params)
        if self.model_type == "ARCH":
            # ARCH model from arch package
            # Default to GARCH(1,1) if no specific volatility params given
            p = self.order if isinstance(self.order, int) else 1
            q = self.model_params.get("q", 1)
            # Remove p, q, and arch_model_type from model_params to avoid duplication
            arch_params = {
                k: v for k, v in self.model_params.items() if k not in ["p", "q", "arch_model_type"]
            }
            return arch_model(y, vol="GARCH", p=p, q=q, **arch_params)
        raise ValueError(
            f"Unknown model type: {self.model_type}. This should not occur as model types "
            f"are validated during initialization. Please report this as a bug if encountered."
        )


class StatsModelsFittedBackend(StationarityMixin):
    """Fitted model backend for statsmodels.

    Wraps statsmodels fitted model objects to conform to the
    FittedModelBackend protocol.
    """

    def __init__(
        self,
        fitted_models: list[Any],
        model_type: str,
        n_series: int,
        y: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
    ):
        self._fitted_models = fitted_models
        self._model_type = model_type
        self._n_series = n_series
        self._y_train = y
        self._X_train = X
        self._scoring_service = ModelScoringService()

    @property
    def params(self) -> dict[str, Any]:
        """Model parameters in standardized format."""
        if self._n_series == 1:
            return self._extract_params(self._fitted_models[0])
        return {"series_params": [self._extract_params(m) for m in self._fitted_models]}

    def _extract_params(self, model: Any) -> dict[str, Any]:
        """Extract parameters from a fitted model."""
        helper = TSFitHelperService()
        params = {}

        # Handle VAR models differently
        if self._model_type == "VAR":
            # For VAR, params returns coefficients matrix
            if hasattr(model, "params"):
                params["coef_matrix"] = np.asarray(model.params)
            if hasattr(model, "sigma_u"):
                params["sigma_u"] = np.asarray(model.sigma_u)
            if hasattr(model, "k_ar"):
                params["k_ar"] = model.k_ar
            return params

        # For ARIMA-type models
        if hasattr(model, "arparams"):
            params["ar"] = np.asarray(model.arparams)
        elif hasattr(model, "params") and self._model_type == "AR":
            # For AR models, params include constant term
            params["ar"] = np.asarray(model.params[1:])  # Skip constant

        if hasattr(model, "maparams"):
            params["ma"] = np.asarray(model.maparams)

        # Get sigma2 (residual variance)
        if hasattr(model, "sigma2"):
            params["sigma2"] = float(model.sigma2)
        elif hasattr(model, "scale"):
            params["sigma2"] = float(model.scale)
        else:
            # Fallback: compute from residuals
            residuals = helper.get_residuals(model)
            params["sigma2"] = float(np.var(residuals))

        # Include seasonal parameters if available
        if hasattr(model, "seasonalarparams"):
            params["seasonal_ar"] = np.asarray(model.seasonalarparams)
        if hasattr(model, "seasonalmaparams"):
            params["seasonal_ma"] = np.asarray(model.seasonalmaparams)

        # Include trend parameters
        if hasattr(model, "trend") and model.trend != "n" and hasattr(model, "trendparams"):
            params["trend"] = np.asarray(model.trendparams)

        return params

    @property
    def residuals(self) -> np.ndarray:
        """Model residuals."""
        helper = TSFitHelperService()
        if self._n_series == 1:
            return helper.get_residuals(self._fitted_models[0]).ravel()
        return np.array([helper.get_residuals(m).ravel() for m in self._fitted_models])

    @property
    def aic(self) -> float:
        """Akaike Information Criterion."""
        criteria = self.get_info_criteria()
        return criteria.get("aic", np.nan)

    @property
    def bic(self) -> float:
        """Bayesian Information Criterion."""
        criteria = self.get_info_criteria()
        return criteria.get("bic", np.nan)

    @property
    def hqic(self) -> float:
        """Hannan-Quinn Information Criterion."""
        criteria = self.get_info_criteria()
        return criteria.get("hqic", np.nan)

    @property
    def fitted_values(self) -> np.ndarray:
        """Fitted values from the model."""
        helper = TSFitHelperService()
        if self._n_series == 1:
            # For single series, return 1D array
            return helper.get_fitted_values(self._fitted_models[0]).ravel()
        # For multiple series, return 2D array
        return np.array([helper.get_fitted_values(m).ravel() for m in self._fitted_models])

    def predict(
        self,
        steps: int,
        X: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Generate point predictions."""
        predictions = []
        for i, model in enumerate(self._fitted_models):
            if self._model_type == "VAR":
                # VAR models require last observations for forecasting
                if X is None:
                    raise ValueError(
                        "VAR models require the last observations (X) for generating predictions. "
                        "Please provide a numpy array containing the most recent observations "
                        "with shape (n_obs, n_vars) where n_obs is the number of lagged observations "
                        "needed by the model and n_vars matches the number of variables in the system."
                    )
                # X should be the last observations of the time series
                # VAR expects (n_obs, n_vars) format
                pred = model.forecast(X, steps=steps, **kwargs)
            elif self._model_type == "ARCH":
                # ARCH models use 'horizon' parameter instead of 'steps'
                pred = model.forecast(horizon=steps, **kwargs)
                # Extract mean predictions
                if hasattr(pred, "mean"):
                    pred = pred.mean.values[-steps:]  # Get last 'steps' predictions
            else:
                # Other models can use exog
                exog = X[i] if X is not None and X.ndim > 1 else X
                pred = model.forecast(steps=steps, exog=exog, **kwargs)
            predictions.append(pred)

        if self._n_series == 1:
            return predictions[0]
        elif self._model_type == "VAR":
            # VAR returns predictions for all series at once
            return predictions[0]
        return np.array(predictions)

    def simulate(
        self,
        steps: int,
        n_paths: int = 1,
        X: Optional[np.ndarray] = None,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Generate simulated paths."""
        rng = np.random.RandomState(random_state)
        simulations = []

        for i, model in enumerate(self._fitted_models):
            exog = X[i] if X is not None and X.ndim > 1 else X

            # Handle different model types
            if hasattr(model, "simulate"):
                # Most statsmodels models have simulate method
                sim = model.simulate(
                    nsimulations=steps,
                    repetitions=n_paths,
                    exog=exog,
                    random_state=rng,
                    **kwargs,
                )
                # Ensure correct shape: (n_paths, steps)
                if sim.ndim == 1:
                    sim = sim.reshape(1, -1)
                elif sim.shape[0] == steps and n_paths > 1:
                    # Some models return (steps, n_paths), we need (n_paths, steps)
                    sim = sim.T
            else:
                # Fallback for models without simulate
                sim = self._simulate_from_params(
                    model=model,
                    steps=steps,
                    n_paths=n_paths,
                    rng=rng,
                )

            simulations.append(sim)

        if self._n_series == 1:
            return simulations[0]
        return np.array(simulations)

    def _simulate_from_params(
        self,
        model: Any,
        steps: int,
        n_paths: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Simulate from model parameters when simulate method not available."""
        params = self._extract_params(model)
        sigma = np.sqrt(params.get("sigma2", 1.0))

        # Generate random shocks
        shocks = rng.normal(0, sigma, size=(n_paths, steps))

        # For now, return random walk
        # This is a simplified fallback - in practice would implement
        # proper ARIMA simulation
        return np.cumsum(shocks, axis=1)

    def get_info_criteria(self) -> dict[str, float]:
        """Get information criteria."""
        criteria = {}
        models = self._fitted_models[:1] if self._n_series > 1 else self._fitted_models

        for model in models:
            if hasattr(model, "aic"):
                criteria["aic"] = float(model.aic)
            if hasattr(model, "bic"):
                criteria["bic"] = float(model.bic)
            if hasattr(model, "hqic"):
                criteria["hqic"] = float(model.hqic)

        return criteria

    def score(
        self,
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
        metric: str = "r2",
    ) -> float:
        """Score model predictions."""
        # Use fitted values for in-sample scoring if y_pred not provided
        if y_pred is None:
            y_pred = self.fitted_values

        # Use training data if y_true not provided
        if y_true is None:
            if self._y_train is None:
                raise ValueError(
                    "True values (y_true) must be provided for scoring when the model "
                    "was not fitted with training data retained. Either provide y_true "
                    "explicitly or ensure the model retains training data during fitting."
                )
            y_true = self._y_train
            # If y_train is 2D with shape (1, n), flatten it
            if y_true.ndim == 2 and y_true.shape[0] == 1:
                y_true = y_true.ravel()

        # Ensure compatible shapes
        if y_true.ndim == 2 and y_true.shape[0] == 1:
            y_true = y_true.ravel()
        if y_pred.ndim == 2 and y_pred.shape[0] == 1:
            y_pred = y_pred.ravel()

        # Ensure shapes match
        if y_true.shape != y_pred.shape:
            # Handle case where fitted values might be shorter due to lags
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[-min_len:]
            y_pred = y_pred[-min_len:]

        return self._scoring_service.score(y_true, y_pred, metric)

    def summary(self) -> str:
        """Get model summary.

        Returns
        -------
        str
            Model summary information
        """
        # For now, return a basic summary
        # In production, could delegate to underlying model's summary
        summary_lines = [
            f"{self._model_type} Model Results",
            "=" * 40,
            f"Number of series: {self._n_series}",
        ]

        # Add information criteria if available
        criteria = {}
        try:
            criteria = self.get_info_criteria()
        except Exception:
            # Information criteria may not be available for all model types
            criteria = {}

        if "aic" in criteria:
            summary_lines.append(f"AIC: {criteria['aic']:.4f}")
        if "bic" in criteria:
            summary_lines.append(f"BIC: {criteria['bic']:.4f}")
        if "hqic" in criteria:
            summary_lines.append(f"HQIC: {criteria['hqic']:.4f}")

        # For statsmodels models, we could delegate to the actual summary
        if self._n_series == 1 and hasattr(self._fitted_models[0], "summary"):
            summary_lines.append("\nDetailed Summary:")
            summary_lines.append(str(self._fitted_models[0].summary()))

        return "\n".join(summary_lines)
