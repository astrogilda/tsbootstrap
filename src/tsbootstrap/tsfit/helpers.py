"""
Helper methods for TSFit class.
"""

import math
import warnings
from numbers import Integral
from typing import List, Union

import numpy as np
from arch.univariate.base import ARCHModelResult
from numpy import ndarray
from statsmodels.tsa.ar_model import AutoRegResultsWrapper
from statsmodels.tsa.arima.model import ARIMAResultsWrapper
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

from tsbootstrap.utils.types import OrderTypes


class TSFitHelpers:
    """Mixin class providing helper methods for TSFit."""

    def _get_coefs_helper(self, model, n_features) -> ndarray:
        """
        Extract coefficients from the fitted model.

        Parameters
        ----------
        model : fitted model object
            The fitted model.
        n_features : int
            Number of features in the model.

        Returns
        -------
        ndarray
            Model coefficients.
        """
        trend_terms = self._calculate_trend_terms(self.model_type, model)
        order_val = self.get_order()  # Get the actual order

        if self.model_type == "var":
            if not isinstance(order_val, Integral):  # VAR order should be int
                raise TypeError(f"VAR order must be an integer, got {order_val}")
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
            raise ValueError(f"Unsupported model_type for _get_coefs_helper: {self.model_type}")

    def _get_intercepts_helper(self, model, n_features) -> ndarray:
        """
        Extract intercepts from the fitted model.

        Parameters
        ----------
        model : fitted model object
            The fitted model.
        n_features : int
            Number of features in the model.

        Returns
        -------
        ndarray
            Model intercepts.
        """
        trend_terms = self._calculate_trend_terms(self.model_type, model)
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
        """
        Calculate the number of trend terms in the model.

        Parameters
        ----------
        model_type : str
            Type of the model.
        model : fitted model object
            The fitted model.

        Returns
        -------
        int
            Number of trend terms.
        """
        trend_terms = 0
        if model_type in ["ar", "arima", "sarima"]:
            trend_attr = getattr(model.model, "trend", "n")  # Default to 'n' if no trend
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

    def _get_residuals_helper(self) -> ndarray:
        """
        Extract residuals from the fitted model.

        Returns
        -------
        ndarray
            Model residuals.
        """
        if hasattr(self.model, "resid"):
            residuals = self.model.resid
            if hasattr(residuals, "values"):
                residuals = residuals.values
            residuals = np.asarray(residuals)
            if residuals.ndim == 1:
                residuals = residuals.reshape(-1, 1)
        elif hasattr(self.model, "residuals"):
            residuals = self.model.residuals
            if hasattr(residuals, "values"):
                residuals = residuals.values
            residuals = np.asarray(residuals)
            if residuals.ndim == 1:
                residuals = residuals.reshape(-1, 1)
        else:
            raise AttributeError(
                f"Model type {type(self.model)} does not have accessible residuals."
            )

        # For AR and VAR models, prepend initial values that are lost due to lag
        if self.model_type in ["ar", "var"]:
            # Get the underlying model object
            actual_model_obj = self.model.model
            if hasattr(actual_model_obj, "endog"):
                num_initial_lost = actual_model_obj.endog.shape[0] - residuals.shape[0]
                if num_initial_lost > 0:
                    values_to_add_back = np.asarray(actual_model_obj.endog[:num_initial_lost])
                    if values_to_add_back.ndim != residuals.ndim:
                        values_to_add_back = values_to_add_back.reshape(
                            -1, residuals.shape[1] if residuals.ndim > 1 else 1
                        )
                    residuals = np.vstack((values_to_add_back, residuals))

        return residuals

    def _get_fitted_X_helper(self) -> ndarray:
        """
        Extract fitted values from the model.

        Returns
        -------
        ndarray
            Fitted values.
        """
        if isinstance(self.model, VARResultsWrapper):
            # VAR models
            fitted = self.model.fittedvalues
            if hasattr(fitted, "values"):
                fitted = fitted.values
            # Handle padding for VAR
            if self.model_type == "var":
                actual_model_obj = self.model.model
                if hasattr(actual_model_obj, "endog"):
                    num_initial_lost = actual_model_obj.endog.shape[0] - fitted.shape[0]
                    if num_initial_lost > 0:
                        values_to_add_back = np.asarray(actual_model_obj.endog[:num_initial_lost])
                        fitted = np.vstack((values_to_add_back, fitted))
            return fitted
        elif isinstance(self.model, ARCHModelResult):
            # ARCH models - reconstruct fitted values
            residuals = self._get_residuals_helper()
            # Get the original y data
            if hasattr(self.model, "_y"):
                y = self.model._y
            elif hasattr(self.model.model, "_y"):
                y = self.model.model._y
            elif hasattr(self.model.model, "y"):
                y = self.model.model.y
            else:
                raise AttributeError("Cannot find original data for ARCH model")

            # Ensure y is the right shape
            y = np.asarray(y)
            if y.ndim == 1:
                y = y.reshape(-1, 1)

            # Calculate fitted values
            fitted = y - residuals
            return fitted
        elif hasattr(self.model, "fittedvalues"):
            # Most statsmodels models
            fitted = self.model.fittedvalues
            if hasattr(fitted, "values"):
                fitted = fitted.values
            fitted = np.asarray(fitted)
            if fitted.ndim == 1:
                fitted = fitted.reshape(-1, 1)

            # Handle padding for AR models
            if self.model_type == "ar":
                actual_model_obj = self.model.model
                if hasattr(actual_model_obj, "endog"):
                    num_initial_lost = actual_model_obj.endog.shape[0] - fitted.shape[0]
                    if num_initial_lost > 0:
                        values_to_add_back = np.asarray(actual_model_obj.endog[:num_initial_lost])
                        if values_to_add_back.ndim != fitted.ndim:
                            values_to_add_back = values_to_add_back.reshape(
                                -1, fitted.shape[1] if fitted.ndim > 1 else 1
                            )
                        fitted = np.vstack((values_to_add_back, fitted))

            return fitted
        else:
            # Fallback: calculate from residuals
            try:
                residuals = self._get_residuals_helper()
                # Need original data to compute fitted values
                if hasattr(self.model, "data"):
                    if hasattr(self.model.data, "endog"):
                        endog = self.model.data.endog
                    else:
                        endog = self.model.data
                elif hasattr(self.model, "endog"):
                    endog = self.model.endog
                else:
                    raise

                if hasattr(endog, "values"):
                    endog = endog.values

                fitted = endog - residuals.flatten()
                return fitted.reshape(-1, 1)
            except Exception as e:
                raise AttributeError(
                    f"Cannot compute fitted values for model type {type(self.model)}: {e}"
                ) from e

    def _get_order_helper(self) -> OrderTypes:
        """
        Extract the order from the fitted model.

        Returns
        -------
        OrderTypes
            Model order.
        """
        if isinstance(self.model, VARResultsWrapper):
            return self.model.k_ar
        elif isinstance(self.model, (ARIMAResultsWrapper, SARIMAXResultsWrapper)):
            # Return ARIMA order (p, d, q)
            return self.model.specification["order"]
        elif isinstance(self.model, ARCHModelResult):
            # ARCH model order
            if hasattr(self.model, "volatility"):
                return self.model.volatility.p
            else:
                return 1  # Default ARCH(1)
        elif hasattr(self.model, "k_ar"):
            return self.model.k_ar
        else:
            # Return the order that was set during initialization
            return self.order

    @staticmethod
    def _lag(x: ndarray, n_lags: int) -> ndarray:
        """
        Create lagged versions of the input array.

        Parameters
        ----------
        x : ndarray
            Input array.
        n_lags : int
            Number of lags.

        Returns
        -------
        ndarray
            Array with lagged values.
        """
        if n_lags == 0:
            return x

        n_obs = len(x)
        lagged = np.zeros((n_obs - n_lags, n_lags))

        for i in range(n_lags):
            lagged[:, i] = x[i : n_obs - n_lags + i]

        return lagged
