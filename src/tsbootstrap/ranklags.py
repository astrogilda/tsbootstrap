"""Ranklags module."""

from __future__ import annotations

import logging
from numbers import Integral

# Keep for now, might be used elsewhere or can be removed if not.
from typing import Optional, cast  # Added Optional

import numpy as np

from tsbootstrap.utils.types import ModelTypes
from tsbootstrap.utils.validate import validate_integers, validate_literal_type

logger = logging.getLogger(__name__)


class RankLags:
    """
    A class that uses several metrics to rank lags for time series models.

    Methods
    -------
    rank_lags_by_aic_bic()
        Rank lags based on Akaike information criterion (AIC) and Bayesian information criterion (BIC).
    rank_lags_by_pacf()
        Rank lags based on Partial Autocorrelation Function (PACF) values.
    estimate_conservative_lag()
        Estimate a conservative lag value by considering various metrics.
    get_model(order)
        Retrieve a previously fitted model given an order.

    Examples
    --------
    >>> from tsbootstrap import RankLags
    >>> import numpy as np
    >>> X = np.random.normal(size=(100, 1))
    >>> rank_obj = RankLags(X, model_type='ar')
    >>> rank_obj.estimate_conservative_lag()
    2
    >>> rank_obj.rank_lags_by_aic_bic()
    (array([2, 1]), array([2, 1]))
    >>> rank_obj.rank_lags_by_pacf()
    array([1, 2])
    """

    _tags = {"python_dependencies": "statsmodels"}
    _model_type: ModelTypes  # Class-level annotation for the backing field

    def __init__(
        self,
        X: np.ndarray,
        model_type: ModelTypes,
        max_lag: int = 10,  # Changed Integral to int
        y: Optional[np.ndarray] = None,  # Added Optional and type hint
        save_models: bool = False,
    ) -> None:
        """
        Initialize the RankLags object.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        model_type : str
            The type of model to fit. One of 'ar', 'arima', 'sarima', 'var', 'arch'.
        max_lag : int, optional, default=10
            Maximum lag to consider.
        y : np.ndarray, optional, default=None
            Exogenous variables to include in the model.
        save_models : bool, optional, default=False
            Whether to save the models.
        """
        self.X = X
        self.max_lag = max_lag
        self.model_type = model_type  # Reverted: Let the property setter handle it
        self.y = y
        self.save_models = save_models
        self.models = []

    @property
    def X(self) -> np.ndarray:
        """
        The input data.

        Returns
        -------
        np.ndarray
            The input data.
        """
        return self._X

    @X.setter
    def X(self, value: np.ndarray) -> None:
        """
        Set the input data.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("X must be a numpy array.")
        self._X = value

    @property
    def max_lag(self) -> int:  # Changed Integral to int
        """
        Maximum lag to consider.

        Returns
        -------
        int
            Maximum lag to consider.
        """
        return self._max_lag

    @max_lag.setter
    def max_lag(self, value: int) -> None:  # Changed Integral to int
        """
        Set the maximum lag to consider.

        Parameters
        ----------
        max_lag : int
            Maximum lag to consider.
        """
        validate_integers(value, min_value=cast(Integral, 1))
        self._max_lag = value

    @property
    def model_type(self) -> ModelTypes:
        """
        The type of model to fit.

        Returns
        -------
        ModelTypes
            The type of model to fit.
        """
        return self._model_type

    @model_type.setter
    def model_type(self, value: ModelTypes) -> None:
        """
        Set the type of model to fit.

        Parameters
        ----------
        value : ModelTypes
            The type of model to fit. One of 'ar', 'arima', 'sarima', 'var', 'arch'.
        """
        validate_literal_type(value, ModelTypes)
        self._model_type = value  # Removed .lower() as ModelTypes are already lowercase literals

    @property
    def y(self) -> Optional[np.ndarray]:  # Added Optional
        """
        Exogenous variables to include in the model.

        Returns
        -------
        np.ndarray
            Exogenous variables to include in the model.
        """
        return self._y

    @y.setter
    def y(self, value: Optional[np.ndarray]) -> None:  # Added Optional
        """
        Set the exogenous variables to include in the model.

        Parameters
        ----------
        y : np.ndarray
            Exogenous variables to include in the model.
        """
        if value is not None and not isinstance(value, np.ndarray):
            raise TypeError("y must be a numpy array or None.")  # Modified error message
        self._y = value

    def rank_lags_by_aic_bic(self):
        """
        Rank lags based on Akaike information criterion (AIC) and Bayesian information criterion (BIC).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            aic_ranked_lags: Lags ranked by AIC.
            bic_ranked_lags: Lags ranked by BIC.
        """
        from tsbootstrap.tsfit_compat import TSFit

        aic_values = []
        bic_values = []
        for lag in range(1, self.max_lag + 1):
            try:
                fit_obj = TSFit(order=lag, model_type=self.model_type)
                model = fit_obj.fit(X=self.X, y=self.y).model
            except Exception as e:
                # raise RuntimeError(f"An error occurred during fitting: {e}")
                logger.warning(
                    f"An error occurred during fitting for lag {lag}. Skipping remaining lags."
                )
                logger.debug(f"{e}")
                # If fitting fails for a lag, assign a high AIC/BIC to deprioritize it
                aic_values.append(np.inf)
                bic_values.append(np.inf)
                if self.save_models:
                    self.models.append(None)  # Add None to keep index alignment if saving
                continue  # Continue to the next lag

            if model is not None:
                if self.save_models:
                    self.models.append(model)

                current_aic = np.inf
                current_bic = np.inf

                if hasattr(model, "aic"):
                    current_aic = model.aic
                else:
                    logger.warning(f"Model for lag {lag} does not have 'aic' attribute. Using inf.")

                if hasattr(model, "bic"):
                    current_bic = model.bic
                else:
                    logger.warning(f"Model for lag {lag} does not have 'bic' attribute. Using inf.")

                aic_values.append(current_aic)
                bic_values.append(current_bic)
            else:
                # Model is None, even if no exception was caught (should be rare)
                logger.warning(f"Model for lag {lag} is None. Assigning inf to AIC/BIC.")
                aic_values.append(np.inf)
                bic_values.append(np.inf)
                if self.save_models:
                    self.models.append(None)

        aic_ranked_lags = np.argsort(aic_values) + 1
        bic_ranked_lags = np.argsort(bic_values) + 1

        return aic_ranked_lags, bic_ranked_lags

    def rank_lags_by_pacf(self) -> np.ndarray:
        """
        Rank lags based on Partial Autocorrelation Function (PACF) values.

        Returns
        -------
        np.ndarray
            Lags ranked by PACF values.
        """
        from statsmodels.tsa.stattools import pacf

        # Can only compute partial correlations for lags up to 50% of the sample size. We use the minimum of max_lag and third of the sample size, to allow for other parameters and trends to be included in the model.
        pacf_values = pacf(self.X, nlags=max(min(self.max_lag, self.X.shape[0] // 3 - 1), 1))[1:]
        ci = 1.96 / np.sqrt(len(self.X))
        significant_lags = np.where(np.abs(pacf_values) > ci)[0] + 1
        return significant_lags

    def estimate_conservative_lag(self) -> int:
        """
        Estimate a conservative lag value by considering various metrics.

        Returns
        -------
        int
            A conservative lag value.
        """
        aic_ranked_lags, bic_ranked_lags = self.rank_lags_by_aic_bic()

        if not aic_ranked_lags.size:  # Check if aic_ranked_lags is empty
            logger.warning(
                "No lags identified by AIC/BIC (possibly due to model fitting issues). "
                "Cannot estimate a conservative lag. Defaulting to lag 1."
            )
            return 1  # Default to 1 if no information from AIC/BIC

        # Start with the intersection of AIC and BIC ranked lags
        candidate_lags = set(aic_ranked_lags).intersection(bic_ranked_lags)

        # If univariate data, try to incorporate PACF results
        if self.X.shape[1] == 1:
            pacf_ranked_lags = self.rank_lags_by_pacf()
            if pacf_ranked_lags.size > 0:  # If PACF found significant lags
                # Refine candidate_lags with PACF results
                candidate_lags = candidate_lags.intersection(pacf_ranked_lags)
            # If pacf_ranked_lags is empty, we proceed with the AIC/BIC intersection

        if not candidate_lags:
            # If no consensus lag is found (either initially or after PACF),
            # default to the best AIC-ranked lag.
            logger.info(
                "No consensus lag found among AIC, BIC (and PACF if applicable). "
                "Using the best lag according to AIC (lag %d).",
                aic_ranked_lags[0],
            )
            return aic_ranked_lags[0]  # Best AIC lag
        else:
            # Return the smallest lag from the consensus set.
            selected_lag = min(candidate_lags)
            logger.info("Estimated conservative lag: %d", selected_lag)
            return selected_lag

    def get_model(self, order: int):
        """
        Retrieve a previously fitted model given an order.

        Parameters
        ----------
        order : int
            Order of the model to retrieve.

        Returns
        -------
        Union[AutoRegResultsWrapper, ARIMAResultsWrapper, SARIMAXResultsWrapper, VARResultsWrapper, ARCHModelResult]
            The fitted model.
        """
        return self.models[order - 1] if self.save_models else None
