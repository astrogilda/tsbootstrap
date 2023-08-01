from typing import List, Union, Optional, Dict, Any
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper
from statsmodels.tsa.arima.model import ARIMAResultsWrapper
from arch.univariate.base import ARCHModelResult
from statsmodels.tsa.ar_model import AutoRegResultsWrapper
import numpy as np
from numpy.random import Generator
from numbers import Integral
from src.tsfit import TSFit
from utils.validate import validate_fitted_model, validate_X, validate_literal_type
from utils.types import ModelTypes, FittedModelType


class TimeSeriesSimulator:
    """
    Class to simulate various types of time series models.
    """

    def __init__(
        self,
        fitted_model: FittedModelType,
        X_fitted: np.ndarray,
        rng: Optional[Union[Integral, Generator]] = None,
    ):
        """
        Initialize the TimeSeriesSimulator class.

        Args:
            fitted_model (FittedModelType): A fitted model object.
            X_fitted (np.ndarray): Array of fitted values.
            rng (Optional[Union[Integral, Generator]], optional): Random number generator instance. Defaults to None.
        """
        self.fitted_model = fitted_model
        self.X_fitted = X_fitted
        self.rng = rng
        self.n_samples, self.n_features = self.X_fitted.shape
        self.burnin = min(100, self.n_samples)

    @property
    def fitted_model(self) -> FittedModelType:
        return self._fitted_model

    @fitted_model.setter
    def fitted_model(self, fitted_model: FittedModelType) -> None:
        validate_fitted_model(fitted_model)
        self._fitted_model = fitted_model

    @property
    def X_fitted(self) -> np.ndarray:
        return self._X_fitted

    @X_fitted.setter
    def X_fitted(self, X_fitted: np.ndarray) -> None:
        validate_X(X_fitted)
        self._X_fitted = X_fitted

    @property
    def rng(self) -> Union[Integral, Generator]:
        return self._rng

    @rng.setter
    def rng(self, rng: Optional[Union[Integral, Generator]]) -> None:
        # Lazy-import check_generator
        from utils.odds_and_ends import check_generator
        self._rng = check_generator(rng)

    def _validate_ar_simulation_params(self, params: Dict[str, Any]) -> None:
        """
        Validate the parameters necessary for the simulation.
        """
        required_params = ['resids_lags', 'resids_coefs', 'resids']
        for param in required_params:
            if params.get(param) is None:
                # logger.error(f"{param} is not provided.")
                raise ValueError(f"{param} must be provided for the AR model.")

    def _simulate_ar_residuals(self, lags: np.ndarray, coefs: np.ndarray, init: np.ndarray,
                               max_lag: int) -> np.ndarray:
        """
        Simulates an Autoregressive (AR) process with given lags, coefficients, initial values, and random errors.

        Args:
            lags (np.ndarray): The lags to be used in the AR process. Can be non-consecutive, but when called from `generate_samples_sieve_autoreg`, it will be sorted.
            coefs (np.ndarray): The coefficients corresponding to each lag. Of shape (1, len(lags)). Sorted by `generate_samples_sieve_autoreg` corresponding to the sorted `lags`.
            init (np.ndarray): The initial values for the simulation. Should be at least as long as the maximum lag.


        Returns:
            np.ndarray: The simulated AR process as a 1D NumPy array.

        Raises:
            ValueError: If `init` is not long enough to cover the maximum lag.
        """

        random_errors = self.rng.normal(size=self.n_samples)

        if len(init) < max_lag:
            raise ValueError(
                "Length of 'init' must be at least as long as the maximum lag in 'lags'")

        # In case init is 2d with shape (X, 1), convert it to 1d
        init = init.ravel()
        series = np.zeros(self.n_samples, dtype=init.dtype)
        series[:max_lag] = init

        trend_terms = TSFit._calculate_trend_terms(
            model_type='ar', model=self.fitted_model)
        intercepts = self.fitted_model.params[:trend_terms].reshape(
            1, trend_terms)

        # Loop through the series, calculating each value based on the lagged values, coefficients, random error, and trend term
        for t in range(max_lag, self.n_samples):
            ar_term = 0
            for i in range(len(lags)):
                ar_term += coefs[0, i] * series[t - lags[i]]

            trend_term = 0
            # If the trend is 'c' or 'ct', add a constant term
            if self.fitted_model.model.trend in ['c', 'ct']:
                trend_term += intercepts[0, 0]
            # If the trend is 't' or 'ct', add a linear time trend
            if self.fitted_model.model.trend in ['t', 'ct']:
                trend_term += intercepts[0, -1] * t

            series[t] = ar_term + trend_term + random_errors[t]

        return series

    def simulate_ar_process(self,
                            resids_lags: Union[Integral, List[int]],
                            resids_coefs: np.ndarray,
                            resids: np.ndarray) -> np.ndarray:
        """
        Simulate AR process from the fitted model.

        Args:
            resids_lags (Union[Integral, List[int]]): The lags of the residuals.
            resids_coefs (np.ndarray): Coefficients of the residuals.
            resids (np.ndarray): Residuals of the fitted model.

        Returns:
            np.ndarray: Simulated AR process.
        """
        self._validate_ar_simulation_params(
            {'resids_lags': resids_lags, 'resids_coefs': resids_coefs, 'resids': resids})

        if not isinstance(self.fitted_model, AutoRegResultsWrapper):
            # logger.error("fitted_model must be an instance of AutoRegResultsWrapper.")
            raise ValueError(
                f"fitted_model must be an instance of AutoRegResultsWrapper. Got {type(self.fitted_model)}.")

        if self.n_features > 1:
            raise ValueError(
                "Only univariate time series are supported for the AR model.")
        if self.n_samples != len(resids):
            raise ValueError(
                "Length of 'resids' must be the same as the number of samples in 'X_fitted'.")

        # In case resids is 2d with shape (X, 1), convert it to 1d
        resids = resids.ravel()
        # In case X_fitted is 2d with shape (X, 1), convert it to 1d
        X_fitted = self.X_fitted.ravel()
        # Generate the bootstrap series
        bootstrap_series = np.zeros(self.n_samples, dtype=X_fitted.dtype)
        # Convert resids_lags to a NumPy array if it is not already
        resids_lags = np.arange(1, resids_lags + 1) if isinstance(
            resids_lags, Integral) else np.array(sorted(resids_lags))
        # Convert lags to a NumPy array if it is not already. When called from `generate_samples_sieve_autoreg`, it will be sorted.
        resids_lags = np.array(sorted(resids_lags))
        max_lag = np.max(resids_lags)
        if resids_coefs.shape[0] != 1:
            raise ValueError(
                "AR coefficients must be a 1D NumPy array of shape (1, X)")
        if resids_coefs.shape[1] != len(resids_lags):
            raise ValueError(
                "Length of 'resids_coefs' must be the same as the length of 'lags'")

        # Simulate residuals using the AR model
        simulated_residuals = self._simulate_ar_residuals(
            lags=resids_lags, coefs=resids_coefs, init=resids[:max_lag], max_lag=max_lag)
        # simulated_residuals.shape: (n_samples,)

        bootstrap_series[:max_lag] = X_fitted[:max_lag]

        # Loop through the series, calculating each value based on the lagged values, coefficients, and random error
        for t in range(max_lag, self.n_samples):
            lagged_values = bootstrap_series[t - resids_lags]
            # lagged_values.shape: (n_lags,)
            lagged_values = lagged_values.reshape(-1, 1)
            # lagged_values.shape: (n_lags, 1)
            # print(f"lagged_values.shape: {lagged_values.shape}")
            bootstrap_series[t] = resids_coefs @ lagged_values + \
                simulated_residuals[t]

        return bootstrap_series.reshape(-1, 1)

    def _simulate_non_ar_residuals(self) -> np.ndarray:
        """
        Simulate residuals according to the model type.
        Returns:
            np.ndarray: Simulated residuals.
        """
        rng_seed = self.rng.integers(
            0, 2**32 - 1) if not isinstance(self.rng, Integral) else self.rng

        '''
        simulators = {
            ARIMAResultsWrapper: self.fitted_model.simulate(burnin=self.burnin, nsimulations=self.n_samples + self.burnin, random_state=self.rng),
            SARIMAXResultsWrapper: self.fitted_model.simulate(burnin=self.burnin, nsimulations=self.n_samples + self.burnin, random_state=self.rng),
            VARResultsWrapper: self.fitted_model.simulate_var(steps=self.n_samples + self.burnin, seed=rng_seed),
            ARCHModelResult: self.fitted_model.model.simulate(
                params=self.fitted_model.params, nobs=self.n_samples + self.burnin, random_state=self.rng)['data'].values
        }
        '''

        if isinstance(self.fitted_model, (ARIMAResultsWrapper, SARIMAXResultsWrapper)):
            return self.fitted_model.simulate(burnin=self.burnin, nsimulations=self.n_samples + self.burnin, random_state=self.rng)
        elif isinstance(self.fitted_model, VARResultsWrapper):
            return self.fitted_model.simulate_var(steps=self.n_samples + self.burnin, seed=rng_seed)[self.burnin:]
        elif isinstance(self.fitted_model, ARCHModelResult):
            return self.fitted_model.model.simulate(
                params=self.fitted_model.params, nobs=self.n_samples + self.burnin, random_state=self.rng)['data'].values[self.burnin:]
        raise ValueError(f"Unsupported fitted model type {self.fitt}.")

    def simulate_non_ar_process(self) -> np.ndarray:
        """
        Simulate a time series from the fitted model.

        Returns:
            np.ndarray: The simulated time series.
        """
        simulated_residuals = self._simulate_non_ar_residuals()
        # Discard the burn-in samples for certain models
        if isinstance(self.fitted_model, (VARResultsWrapper, ARCHModelResult)):
            simulated_residuals = simulated_residuals[self.burnin:]
        return self.X_fitted + simulated_residuals

    def generate_samples_sieve(self,
                               model_type: ModelTypes,
                               resids_lags: Optional[Union[int,
                                                           List[int]]] = None,
                               resids_coefs: Optional[np.ndarray] = None,
                               resids: Optional[np.ndarray] = None,
                               ) -> np.ndarray:
        """
        Generate a bootstrap sample using the sieve bootstrap.

        Args:
            model_type (ModelTypes): The model type used for the simulation.
            resids_lags (Optional[Union[int, List[int]]], optional): The lags to be used in the AR process. Can be non-consecutive.
            resids_coefs (Optional[np.ndarray], optional): The coefficients corresponding to each lag. Of shape (1, len(lags)).
            resids (Optional[np.ndarray], optional): The initial values for the simulation. Should be at least as long as the maximum lag.

        Returns:
            np.ndarray: The simulated bootstrap series.

        Raises:
            ValueError: If model_type is not supported or necessary parameters are not provided.
        """
        validate_literal_type(model_type, ModelTypes)
        if model_type == 'ar':
            return self.simulate_ar_process(
                resids_lags, resids_coefs, resids)
        else:
            return self.simulate_non_ar_process()
