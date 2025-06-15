from __future__ import annotations

import math
import warnings
from numbers import Integral
from typing import List, Union  # Add Union and List for type hints

import numpy as np
from arch.univariate.base import ARCHModelResult
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from sklearn.utils.validation import NotFittedError, check_is_fitted
from statsmodels.tsa.ar_model import AutoRegResultsWrapper
from statsmodels.tsa.arima.model import ARIMAResultsWrapper
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

from tsbootstrap.ranklags import RankLags
from tsbootstrap.time_series_model import TimeSeriesModel
from tsbootstrap.utils.types import (
    ModelTypes,
    OrderTypes,
    OrderTypesWithoutNone,
)
from tsbootstrap.utils.validate import (
    validate_literal_type,
    validate_X_and_y,
)


class TSFit(BaseEstimator, RegressorMixin):
    """
    Performs fitting for various time series models including 'ar', 'arima', 'sarima', 'var', and 'arch'.

    Attributes
    ----------
    rescale_factors : dict
        Rescaling factors for the input data and exogenous variables.
    model: Union[
        AutoRegResultsWrapper,
        ARIMAResultsWrapper,
        SARIMAXResultsWrapper,
        VARResultsWrapper,
        ARCHModelResult,
    ]
        The fitted model.

    Methods
    -------
    fit(X, y=None)
        Fit the chosen model to the data.
    get_coefs()
        Return the coefficients of the fitted model.
    get_intercepts()
        Return the intercepts of the fitted model.
    get_residuals()
        Return the residuals of the fitted model.
    get_fitted_X()
        Return the fitted values of the model.
    get_order()
        Return the order of the fitted model.
    predict(X, n_steps=1)
        Predict future values using the fitted model.
    score(X, y)
        Compute the R-squared score for the fitted model.

    Raises
    ------
    ValueError
        If the model type or the model order is invalid.

    Notes
    -----
    The following table shows the valid model types and their corresponding orders.

    +--------+-------------------+-------------------+
    | Model  | Valid orders      | Invalid orders    |
    +========+===================+===================+
    | 'ar'   | int, List[int]    | tuple             |
    +--------+-------------------+-------------------+
    | 'arima'| tuple of length 3 | int, list         |
    +--------+-------------------+-------------------+
    | 'sarima'| tuple of length 4| int, list         |
    +--------+-------------------+-------------------+
    | 'var'  | int               | list, tuple       |
    +--------+-------------------+-------------------+
    | 'arch' | int               | list, tuple       |
    +--------+-------------------+-------------------+

    Examples
    --------
    >>> from tsbootstrap import TSFit
    >>> import numpy as np
    >>> X = np.random.normal(size=(100, 1))
    >>> fit_obj = TSFit(order=2, model_type='ar')  # doctest: +SKIP
    >>> fit_obj.fit(X)  # doctest: +SKIP
    TSFit(order=2, model_type='ar')
    >>> fit_obj.get_coefs()  # doctest: +SKIP
    array([[ 0.003, -0.002]])
    """

    _tags = {"python_dependencies": ["arch", "statsmodels"]}

    def __init__(
        self, order: OrderTypesWithoutNone, model_type: ModelTypes, **kwargs
    ) -> None:
        """
        Initialize the TSFit object.

        Parameters
        ----------
        order : OrderTypesWithoutNone
            Order of the model.
        model_type : ModelTypes
            Type of the model.
        **kwargs
            Additional parameters to be passed to the model.
        """
        self.model_type = model_type  # Setter will be called
        self.order = order  # Setter will be called
        self.rescale_factors: dict = {}
        self.model_params = kwargs
        self.model: (
            Union[
                AutoRegResultsWrapper,
                ARIMAResultsWrapper,
                SARIMAXResultsWrapper,
                VARResultsWrapper,
                ARCHModelResult,
            ]
            | None
        ) = None

    @property
    def model_type(self) -> ModelTypes:
        """The type of the model."""
        return self._model_type

    @model_type.setter
    def model_type(self, value: ModelTypes) -> None:
        """Set the model type."""
        validate_literal_type(value, ModelTypes)
        self._model_type = value  # Already lowercase from ModelTypes

    @property
    def order(self) -> OrderTypesWithoutNone:
        """The order of the model."""
        return self._order

    @order.setter
    def order(self, value: OrderTypesWithoutNone) -> None:
        """Set the order of the model."""
        if not isinstance(value, (Integral, list, tuple)):
            raise TypeError(
                f"Invalid order '{value}', should be an integer, list, or tuple."
            )

        current_model_type = getattr(
            self, "_model_type", None
        )  # Get if already set

        if isinstance(value, list):
            if len(value) == 0:
                raise ValueError("Order list cannot be empty.")
            if not all(isinstance(v, Integral) for v in value):
                raise TypeError("All elements in order list must be integers.")
            if len(value) > 1:
                value_orig = value
                value = sorted(set(value))  # Sort and remove duplicates
                if (
                    value != value_orig
                ):  # Check if original was already sorted and unique
                    warning_msg = f"Order '{value_orig}' is a list. Using sorted unique lags: '{value}'."
                    warnings.warn(warning_msg, stacklevel=2)

        if isinstance(value, tuple):
            if len(value) == 0:
                raise ValueError("Order tuple cannot be empty.")
            if not all(isinstance(v, Integral) for v in value):
                raise TypeError(
                    "All elements in order tuple must be integers."
                )

        if (
            current_model_type
        ):  # Check if model_type is available for validation
            if isinstance(value, tuple):
                if current_model_type == "arima" and len(value) != 3:
                    raise ValueError(
                        f"ARIMA order must be a tuple of 3 integers, got {value}"
                    )
                if current_model_type == "sarima" and len(value) != 4:
                    raise ValueError(
                        f"SARIMA order must be a tuple of 4 integers, got {value}"
                    )
                if current_model_type not in ["arima", "sarima"]:
                    raise ValueError(
                        f"Tuple order '{value}' is only valid for ARIMA/SARIMA, not '{current_model_type}'"
                    )

            if isinstance(value, (Integral, list)) and current_model_type in [
                "arima",
                "sarima",
            ]:
                raise ValueError(
                    f"Integer/List order '{value}' is not valid for '{current_model_type}'. Use a tuple."
                )

            if isinstance(value, Integral) and current_model_type in [
                "sarima",
                "arima",
            ]:
                # This block should ideally not be reached if above checks are correct
                # but kept for safety from original logic.
                if current_model_type == "sarima":
                    s_period = 2
                    new_value = (value, 0, 0, s_period)
                    warning_msg = f"{current_model_type.upper()} model requires a tuple of order (p, d, q, s). Integer order {value} converted to {new_value}."
                else:  # arima
                    new_value = (value, 0, 0)
                    warning_msg = f"{current_model_type.upper()} model requires a tuple of order (p, d, q). Integer order {value} converted to {new_value}."
                warnings.warn(warning_msg, stacklevel=2)
                value = new_value

        self._order = value

    def get_params(self, deep=True):
        return {
            "order": self.order,
            "model_type": self.model_type,
            **self.model_params,
        }

    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.model_params[key] = value
        return self

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> TSFit:
        X_orig, y_orig = X, y  # Keep original for validation
        X_val, y_val = validate_X_and_y(
            X_orig,
            y_orig,
            model_is_var=(self.model_type == "var"),
            model_is_arch=(self.model_type == "arch"),
        )

        X_fit, y_fit = X_val, y_val  # Data to be used for fitting

        def _rescale_inputs(
            data_X: np.ndarray, data_y: np.ndarray | None = None
        ):
            def rescale_array(arr: np.ndarray, max_iter: int = 100):
                variance = np.var(arr)
                if math.isclose(
                    variance, 0, abs_tol=1e-8
                ):  # More robust zero check
                    warnings.warn(
                        "Variance of input data is close to 0. Rescaling might be unstable or ineffective.",
                        UserWarning,
                        stacklevel=2,
                    )
                    return arr, 1.0  # No rescaling if variance is zero

                total_rescale_factor = 1.0
                iterations = 0
                current_arr = arr.copy()

                while not (1.0 <= np.var(current_arr) <= 1000.0):
                    if iterations >= max_iter:
                        warnings.warn(
                            f"Max iterations ({max_iter}) for rescaling reached. Variance {np.var(current_arr)} not in [1, 1000]. ARCH/GARCH results might be untrustworthy.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        break

                    variance = np.var(current_arr)
                    if math.isclose(
                        variance, 0, abs_tol=1e-8
                    ):  # Check again inside loop
                        warnings.warn(
                            "Variance became zero during rescaling. Stopping.",
                            UserWarning,
                            stacklevel=2,
                        )
                        break

                    rescale_factor = np.sqrt(100.0 / variance)
                    current_arr = current_arr * rescale_factor
                    total_rescale_factor *= rescale_factor
                    iterations += 1
                return current_arr, total_rescale_factor

            rescaled_X, x_factor = rescale_array(data_X)
            rescaled_y = None
            y_factors = None

            if data_y is not None:
                rescaled_y_list = []
                y_factors_list = []
                for i in range(data_y.shape[1]):
                    col_y, factor_y = rescale_array(data_y[:, i])
                    rescaled_y_list.append(col_y)
                    y_factors_list.append(factor_y)
                rescaled_y = np.column_stack(rescaled_y_list)
                y_factors = y_factors_list

            return rescaled_X, rescaled_y, (x_factor, y_factors)

        if self.model_type == "arch":
            X_rescaled, y_rescaled, (x_factor, y_factors) = _rescale_inputs(
                X_val, y_val
            )
            self.rescale_factors["x"] = x_factor
            self.rescale_factors["y"] = y_factors
            X_fit, y_fit = X_rescaled, y_rescaled

        fit_func = TimeSeriesModel(
            X=X_fit, y=y_fit, model_type=self.model_type
        )
        self.model = fit_func.fit(order=self.order, **self.model_params)
        return self

    def get_coefs(self) -> np.ndarray:
        check_is_fitted(self, "model")
        if self.model is None:
            raise NotFittedError("Model not fitted")

        if self.model_type != "arch":
            n_features = (
                self.model.model.endog.shape[1]
                if self.model.model.endog.ndim > 1
                else 1
            )
        else:
            n_features = (
                self.model.model.y.shape[1]
                if self.model.model.y.ndim > 1
                else 1
            )
        return self._get_coefs_helper(self.model, n_features)

    def get_intercepts(self) -> np.ndarray:
        check_is_fitted(self, "model")
        if self.model is None:
            raise NotFittedError("Model not fitted")

        if self.model_type != "arch":
            n_features = (
                self.model.model.endog.shape[1]
                if self.model.model.endog.ndim > 1
                else 1
            )
        else:  # ARCH model might not always have 'endog', use 'y'
            n_features = (
                self.model.model.y.shape[1]
                if self.model.model.y.ndim > 1
                else 1
            )
        return self._get_intercepts_helper(self.model, n_features)

    def get_residuals(self) -> np.ndarray:
        check_is_fitted(self, "model")
        if self.model is None:
            raise NotFittedError("Model not fitted")
        return self._get_residuals_helper(self.model)

    def get_fitted_X(self) -> np.ndarray:
        check_is_fitted(self, "model")
        if self.model is None:
            raise NotFittedError("Model not fitted")
        return self._get_fitted_X_helper(self.model)

    def get_order(self) -> OrderTypesWithoutNone:
        check_is_fitted(self, "model")
        if self.model is None:
            raise NotFittedError("Model not fitted")
        return self._get_order_helper(self.model)

    def predict(
        self, X: np.ndarray, y: np.ndarray | None = None, n_steps: int = 1
    ) -> np.ndarray:
        check_is_fitted(self, "model")
        if self.model is None:
            raise NotFittedError("Model not fitted")

        # X for prediction might not need full validation like training X,
        # but ensure it's a NumPy array.
        # For VAR, X is the last `k_ar` observations.
        # For others, it's often ignored by statsmodels' forecast if history is in model.
        # However, exog (y) is important.

        # Minimal validation for X_pred
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        if self.model_type == "var":
            # VAR.forecast expects `y` to be the last `k_ar` observations of the series.
            # `exog_future` is for future exogenous variables.
            return self.model.forecast(X, steps=n_steps, exog_future=y)
        elif self.model_type == "arch":
            # ARCH forecast uses `x` for future exogenous regressors.
            # `horizon` is n_steps.
            return (
                self.model.forecast(horizon=n_steps, x=y, method="analytic")
                .mean.values[-1]
                .ravel()
            )
        elif self.model_type in ["ar", "arima", "sarima"]:
            return self.model.forecast(steps=n_steps, exog=y)
        else:
            raise ValueError(
                f"Unsupported model_type for predict: {self.model_type}"
            )

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> float:
        # sample_weight is ignored for now
        y_pred = self.predict(X)
        # Ensure y_pred and y have compatible shapes for r2_score
        # r2_score expects (n_samples,) or (n_samples, n_outputs)
        if y_pred.ndim == 1 and y.ndim == 2 and y.shape[1] == 1:
            y_true_adj = y.ravel()
        elif y_pred.ndim == 2 and y_pred.shape[1] == 1 and y.ndim == 1:
            y_pred_adj = y_pred.ravel()
            y_true_adj = y
        elif (
            y_pred.ndim == 2 and y.ndim == 2 and y_pred.shape[0] == y.shape[0]
        ):
            # If y_pred is shorter due to lags, adjust y_true
            if y_pred.shape[0] < y.shape[0]:
                y_true_adj = y[-y_pred.shape[0] :]
                y_pred_adj = y_pred
            else:
                y_true_adj = y
                y_pred_adj = y_pred
        else:
            y_true_adj = y
            y_pred_adj = y_pred

        if y_pred_adj.shape != y_true_adj.shape:
            # If shapes still don't match after basic adjustments,
            # this often happens if predict returns forecasts beyond original X length.
            # For a fair score, predict up to len(X) or align y_true.
            # Simplest for now: if y_pred is shorter (e.g. due to lags), score on available part.
            min_len = min(len(y_pred_adj), len(y_true_adj))
            if min_len == 0:
                return -np.inf  # Cannot score if no overlapping predictions
            y_pred_adj = y_pred_adj[:min_len]
            y_true_adj = y_true_adj[:min_len]

        return r2_score(y_true_adj, y_pred_adj, sample_weight=sample_weight)

    def _get_coefs_helper(self, model, n_features) -> np.ndarray:
        trend_terms = TSFit._calculate_trend_terms(self.model_type, model)
        order_val = self.get_order()  # Get the actual order

        if self.model_type == "var":
            if not isinstance(order_val, Integral):  # VAR order should be int
                raise TypeError(
                    f"VAR order must be an integer, got {order_val}"
                )
            return (
                model.params[trend_terms:]
                .reshape(n_features, order_val, n_features)
                .transpose(0, 2, 1)
            )
        elif self.model_type == "ar" or self.model_type in ["arima", "sarima"]:
            return model.params[trend_terms:].reshape(1, -1)
        elif self.model_type == "arch":
            return model.params  # ARCH params are typically direct
        else:
            raise ValueError(
                f"Unsupported model_type for _get_coefs_helper: {self.model_type}"
            )

    def _get_intercepts_helper(self, model, n_features) -> np.ndarray:
        trend_terms = TSFit._calculate_trend_terms(self.model_type, model)
        if trend_terms == 0:
            return np.array([])  # No intercept if no trend terms

        if self.model_type == "var":
            return model.params[:trend_terms].reshape(n_features, trend_terms)
        elif self.model_type in ["ar", "arima", "sarima"]:
            return model.params[:trend_terms].reshape(1, trend_terms)
        elif self.model_type == "arch":
            return np.array([])  # ARCH models usually handle mean separately
        else:
            raise ValueError(
                f"Unsupported model_type for _get_intercepts_helper: {self.model_type}"
            )

    @staticmethod
    def _calculate_trend_terms(model_type: str, model) -> int:
        trend_terms = 0
        if model_type in ["ar", "arima", "sarima"]:
            trend_attr = getattr(
                model.model, "trend", "n"
            )  # Default to 'n' if no trend
            if trend_attr == "c" or trend_attr == "t":
                trend_terms = 1
            elif trend_attr == "ct":
                trend_terms = 2
            elif trend_attr != "n" and trend_attr is not None:
                raise ValueError(f"Unknown trend term: {trend_attr}")
        elif model_type == "var":
            trend_attr = getattr(model, "trend", "nc")  # Default to 'nc'
            if trend_attr == "c":
                trend_terms = 1
            elif trend_attr == "ct":
                trend_terms = 2
            elif trend_attr == "ctt":
                trend_terms = 3
            elif trend_attr != "nc":
                raise ValueError(f"Unknown trend term: {trend_attr}")
        return trend_terms

    def _get_residuals_helper(self, model) -> np.ndarray:
        model_resid = np.asarray(model.resid)
        if model_resid.ndim == 1:
            model_resid = model_resid.reshape(-1, 1)

        if self.model_type in ["ar", "var"]:
            # Prepend initial values that are lost due to lag
            num_initial_lost = (
                self.model.model.endog.shape[0] - model_resid.shape[0]
            )
            if num_initial_lost > 0:
                values_to_add_back = np.asarray(
                    self.model.model.endog[:num_initial_lost]
                )
                if values_to_add_back.ndim != model_resid.ndim:
                    values_to_add_back = values_to_add_back.reshape(
                        -1, model_resid.shape[1] if model_resid.ndim > 1 else 1
                    )
                model_resid = np.vstack((values_to_add_back, model_resid))

        if (
            self.model_type == "arch"
            and "x" in self.rescale_factors
            and self.rescale_factors["x"] != 1.0
        ):
            model_resid = model_resid / self.rescale_factors["x"]
        return model_resid

    def _get_fitted_X_helper(self, model) -> np.ndarray:
        if self.model_type == "arch":
            # For ARCH, fitted = y - resid (standardized)
            # resid = (y - mu) / sigma, so y - resid * sigma = mu
            # Or, more directly, fitted = y - resid (where resid are from the model)
            # conditional_volatility is sigma_t
            # fitted = y_t - resid_t (where resid_t = (y_t - mu_t)/sigma_t)
            # This seems more like mu_t = y_t - resid_t * sigma_t
            # Let's use y - resid (after unscaling resid)
            unscaled_resid = self._get_residuals_helper(
                model
            )  # gets unscaled resid
            # Original X was used to fit if not ARCH, or rescaled X if ARCH
            # If ARCH, self.model.model.y is the rescaled X
            original_X_for_fit = (
                self.model.model.y
                if self.model_type == "arch"
                else self.model.model.endog
            )

            # Ensure shapes match for subtraction
            min_len = min(original_X_for_fit.shape[0], unscaled_resid.shape[0])
            fitted_values = (
                original_X_for_fit[-min_len:] - unscaled_resid[-min_len:]
            )

            # Prepend initial values if any were lost from original_X_for_fit compared to fitted_values
            if fitted_values.shape[0] < original_X_for_fit.shape[0]:
                num_to_prepend = (
                    original_X_for_fit.shape[0] - fitted_values.shape[0]
                )
                prepend_values = original_X_for_fit[:num_to_prepend]
                fitted_values = np.vstack((prepend_values, fitted_values))

            return fitted_values

        else:  # For AR, ARIMA, SARIMA, VAR
            model_fittedvalues = np.asarray(model.fittedvalues)
            if model_fittedvalues.ndim == 1:
                model_fittedvalues = model_fittedvalues.reshape(-1, 1)

            if self.model_type in ["ar", "var"]:
                num_initial_lost = (
                    self.model.model.endog.shape[0]
                    - model_fittedvalues.shape[0]
                )
                if num_initial_lost > 0:
                    values_to_add_back = np.asarray(
                        self.model.model.endog[:num_initial_lost]
                    )
                    if values_to_add_back.ndim != model_fittedvalues.ndim:
                        values_to_add_back = values_to_add_back.reshape(
                            -1,
                            (
                                model_fittedvalues.shape[1]
                                if model_fittedvalues.ndim > 1
                                else 1
                            ),
                        )
                    model_fittedvalues = np.vstack(
                        (values_to_add_back, model_fittedvalues)
                    )
            return model_fittedvalues

    def _get_order_helper(self, model) -> OrderTypesWithoutNone:
        if self.model_type == "arch":
            # For ARCH, p from GARCH(p,q) is usually considered the AR order of variance
            return model.model.volatility.p
        elif self.model_type == "var":
            return model.k_ar  # VAR model's lag order
        elif self.model_type == "ar":
            # self.order is already processed (int or sorted list of ints)
            return self.order
        elif self.model_type == "arima":
            # self.order is (p,d,q)
            return self.order
        elif self.model_type == "sarima":
            # self.order is (p,d,q)(P,D,Q,s)
            return self.order
        else:
            # Fallback, though should be caught by model_type validation
            raise ValueError(
                f"Invalid or unhandled model_type '{self.model_type}' in _get_order_helper."
            )

    def _lag(self, X: np.ndarray, n_lags: int) -> np.ndarray:
        if len(X) < n_lags:
            raise ValueError(
                "Number of lags is greater than the length of the input data."
            )
        return np.column_stack(
            [X[i : len(X) - n_lags + 1 + i, :] for i in range(n_lags)]
        )


class TSFitBestLag(BaseEstimator, RegressorMixin):
    """
    A class used to fit time series data and find the best lag for forecasting.
    """

    def __init__(
        self,
        model_type: ModelTypes,
        max_lag: int = 10,
        order: OrderTypes = None,  # Can be None initially
        save_models=False,
        **kwargs,
    ):
        self.model_type = model_type
        self.max_lag = max_lag
        self.order: OrderTypesWithoutNone | None = (
            order  # Allow None initially, will be set in fit
        )
        self.save_models = save_models
        self.model_params = kwargs
        self.rank_lagger: RankLags | None = None
        self.ts_fit: TSFit | None = None
        self.model: (
            Union[
                AutoRegResultsWrapper,
                ARIMAResultsWrapper,
                SARIMAXResultsWrapper,
                VARResultsWrapper,
                ARCHModelResult,
            ]
            | None
        ) = None
        self.rescale_factors: dict = {}

    def _compute_best_order(self, X) -> OrderTypesWithoutNone:
        self.rank_lagger = RankLags(
            X=X,
            max_lag=self.max_lag,
            model_type=self.model_type,
            save_models=self.save_models,  # Pass save_models to RankLags
        )
        # estimate_conservative_lag returns int, but TSFit order can be more complex
        # For now, assume RankLags gives an appropriate int order for non-ARIMA/SARIMA
        # or that this will be handled/overridden if self.order is explicitly set.
        best_lag_int = self.rank_lagger.estimate_conservative_lag()

        # Convert integer lag to appropriate tuple for ARIMA/SARIMA if needed by TSFit
        if self.model_type == "arima":
            return (best_lag_int, 0, 0)
        elif self.model_type == "sarima":
            return (best_lag_int, 0, 0, 2)  # Default s=2 as in TSFit
        return best_lag_int

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        if self.order is None:
            self.order = self._compute_best_order(X)

        if self.order is None:  # Should be set by _compute_best_order
            raise ValueError("Order could not be determined.")

        self.ts_fit = TSFit(
            order=self.order,  # Now OrderTypesWithoutNone
            model_type=self.model_type,
            **self.model_params,
        )
        self.model = self.ts_fit.fit(X, y=y).model
        self.rescale_factors = self.ts_fit.rescale_factors
        return self

    def get_coefs(self) -> np.ndarray:
        check_is_fitted(self, "ts_fit")
        if self.ts_fit is None:
            raise NotFittedError("ts_fit not available.")
        return self.ts_fit.get_coefs()

    def get_intercepts(self) -> np.ndarray:
        check_is_fitted(self, "ts_fit")
        if self.ts_fit is None:
            raise NotFittedError("ts_fit not available.")
        return self.ts_fit.get_intercepts()

    def get_residuals(self) -> np.ndarray:
        check_is_fitted(self, "ts_fit")
        if self.ts_fit is None:
            raise NotFittedError("ts_fit not available.")
        return self.ts_fit.get_residuals()

    def get_fitted_X(self) -> np.ndarray:
        check_is_fitted(self, "ts_fit")
        if self.ts_fit is None:
            raise NotFittedError("ts_fit not available.")
        return self.ts_fit.get_fitted_X()

    def get_order(self) -> OrderTypesWithoutNone:
        check_is_fitted(self, "ts_fit")
        if self.ts_fit is None:
            raise NotFittedError("ts_fit not available.")
        return self.ts_fit.get_order()

    def get_model(self):  # Returns the fitted model instance
        check_is_fitted(self, "model")
        if self.model is None:
            raise NotFittedError("Model not fitted.")
        return self.model

    def predict(
        self, X: np.ndarray, y: np.ndarray | None = None, n_steps: int = 1
    ):
        check_is_fitted(self, "ts_fit")
        if self.ts_fit is None:
            raise NotFittedError("ts_fit not available.")
        return self.ts_fit.predict(X, y=y, n_steps=n_steps)

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> float:
        check_is_fitted(self, "ts_fit")
        if self.ts_fit is None:
            raise NotFittedError("ts_fit not available.")
        return self.ts_fit.score(X, y, sample_weight=sample_weight)

    def __repr__(self, N_CHAR_MAX=700) -> str:
        params_str = ", ".join(
            f"{k!r}={v!r}" for k, v in self.model_params.items()
        )
        return f"{self.__class__.__name__}(model_type={self.model_type!r}, order={self.order!r}, max_lag={self.max_lag!r}, save_models={self.save_models!r}, model_params={{{params_str}}})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__} using model_type='{self.model_type}' with order={self.order}, max_lag={self.max_lag}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TSFitBestLag):
            return False
        return (
            self.model_type == other.model_type
            and self.order == other.order
            and self.max_lag == other.max_lag
            and self.save_models == other.save_models
            and self.rescale_factors == other.rescale_factors
            and self.model_params == other.model_params
            # Model comparison can be tricky. For now, check if both are None or same type.
            # A deeper comparison might involve checking model parameters if available and consistent.
            and (
                (self.model is None and other.model is None)
                or (
                    self.model is not None
                    and other.model is not None
                    and type(self.model) is type(other.model)
                )
            )
        )
