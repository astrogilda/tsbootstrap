"""
Time series simulation: Generating synthetic realizations with statistical fidelity.

This module provides sophisticated simulation capabilities for time series models,
enabling the generation of synthetic data that preserves the statistical properties
of fitted models. Through careful implementation of model-specific algorithms,
we create realizations that are statistically indistinguishable from the original
process while incorporating appropriate randomness.

The simulation framework serves multiple critical purposes: validating bootstrap
methods through Monte Carlo studies, generating forecast scenarios, and testing
system behavior under various conditions. Each simulation algorithm has been
validated against theoretical properties to ensure statistical correctness.
"""

from numbers import Integral
from typing import List, Optional, Union

import numpy as np

from tsbootstrap.utils.types import ModelTypes
from tsbootstrap.utils.validate import (
    validate_fitted_model,
    validate_integers,
    validate_literal_type,
    validate_rng,
    validate_X_and_y,
)


class TimeSeriesSimulator:
    """
    Advanced simulation engine for time series model realizations.

    This class implements state-of-the-art simulation algorithms for various
    time series models, from simple autoregressive processes to complex
    GARCH specifications. We've designed the implementation to balance
    statistical accuracy with computational efficiency, ensuring that simulated
    series maintain the essential properties of the underlying stochastic process.

    The simulator handles critical details that are often overlooked: proper
    initialization through burn-in periods, correct propagation of multivariate
    dependencies, and appropriate treatment of model-specific constraints. Each
    simulation method has been validated against known theoretical results and
    empirical benchmarks.

    Our architecture supports both single realizations and bulk generation for
    Monte Carlo studies. The flexible design accommodates various model types
    while maintaining a consistent interface, simplifying integration into
    larger analytical workflows.

    Attributes
    ----------
    n_samples : int
        Length of the time series to simulate, calibrated from the fitted model.
        This ensures consistency between original and simulated data.

    n_features : int
        Dimensionality of the time series. Supports both univariate (n_features=1)
        and multivariate simulations with proper cross-series dependencies.

    burnin : int
        Number of initial observations to discard, allowing the process to reach
        its stationary distribution. Automatically calibrated based on series length.
    """

    _tags = {"python_dependencies": ["arch", "statsmodels"]}

    def __init__(
        self,
        fitted_model,
        X_fitted: np.ndarray,
        rng=None,
    ) -> None:
        """
        Initialize the TimeSeriesSimulator class.

        Parameters
        ----------
        fitted_model: FittedModelTypes
            A fitted model object.
        X_fitted: np.ndarray
            Array of fitted values.
        rng: Optional[Union[Integral, Generator]], optional
            Random number generator instance. Defaults to None.
        """
        self.fitted_model = fitted_model
        self.X_fitted = X_fitted
        self.rng = rng
        self.n_samples, self.n_features = self.X_fitted.shape
        self.burnin = min(100, self.n_samples // 3)

    @property
    def fitted_model(self):
        """Get the fitted model."""
        return self._fitted_model

    @fitted_model.setter
    def fitted_model(self, fitted_model) -> None:
        """Set the fitted model, ensuring it's validated first."""
        validate_fitted_model(fitted_model)
        self._fitted_model = fitted_model

    @property
    def X_fitted(self) -> np.ndarray:
        """Get the array of fitted values."""
        return self._X_fitted

    @X_fitted.setter
    def X_fitted(self, value: np.ndarray) -> None:
        """
        Set the array of fitted values.

        Parameters
        ----------
        value: np.ndarray
            Array of fitted values to set.
        """
        if not isinstance(value, np.ndarray):
            raise TypeError(f"X_fitted must be a NumPy array. Got {type(value)}.")
        if not np.issubdtype(value.dtype, np.floating):
            raise TypeError(f"X_fitted dtype must be a float type. Got {value.dtype}.")
        from arch.univariate.base import ARCHModelResult
        from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

        model_is_var = isinstance(self.fitted_model, VARResultsWrapper)
        model_is_arch = isinstance(self.fitted_model, ARCHModelResult)
        self._X_fitted, _ = validate_X_and_y(
            value, None, model_is_var=model_is_var, model_is_arch=model_is_arch
        )

    @property
    def rng(self):
        """Get the random number generator instance."""
        return self._rng

    @rng.setter
    def rng(self, rng) -> None:
        """
        Set the random number generator instance.

        Parameters
        ----------
        rng: Optional[Union[Integral, Generator]]
            Random number generator instance.
        """
        self._rng = validate_rng(rng, allow_seed=True)

    def _validate_ar_simulation_params(self, params: dict) -> None:
        """
        Validate the parameters necessary for the simulation.
        """
        required_params = ["resids_lags", "resids_coefs", "resids"]
        for param in required_params:
            if params.get(param) is None:
                # logger.error(f"{param} is not provided.")
                raise ValueError(f"{param} must be provided for the AR model.")

    def _simulate_ar_residuals(
        self,
        lags: np.ndarray,
        coefs: np.ndarray,
        init: np.ndarray,
        max_lag: Integral,
        n_samples: int,
    ) -> np.ndarray:
        """
        Simulates an Autoregressive (AR) process with given lags, coefficients, initial values, and random errors.

        Parameters
        ----------
        lags: np.ndarray
            The lags to be used in the AR process. Can be non-consecutive, but when called from `generate_samples_sieve`, it will be sorted.
        coefs: np.ndarray
            The coefficients corresponding to each lag. Of shape (1, len(lags)). Sorted by `generate_samples_sieve` corresponding to the sorted `lags`.
        init: np.ndarray
            The initial values for the simulation. Should be at least as long as the maximum lag.
        n_samples: int
            The number of samples to generate.

        Returns
        -------
        np.ndarray
            The simulated AR process as a 1D NumPy array.

        Raises
        ------
        ValueError
            If `lags` or `coefs` are not provided.
            If `coefs` is not a 1D NumPy array.
            If `coefs` is not the same length as `lags`.
            If `init` is not the same length as `max_lag`.

        TypeError
            If `lags` is not an integer or a list of integers.
        """
        random_errors = self.rng.normal(size=n_samples)

        if len(init) < max_lag:  # type: ignore
            raise ValueError(
                "Length of 'init' must be at least as long as the maximum lag in 'lags'"
            )

        # In case init is 2d with shape (X, 1), convert it to 1d
        init = init.ravel()
        series = np.zeros(n_samples, dtype=init.dtype)
        series[:max_lag] = init

        # Import the helper service
        from tsbootstrap.services.tsfit_services import TSFitHelperService

        trend_terms = TSFitHelperService.calculate_trend_terms(
            model_type="ar", model=self.fitted_model
        )
        if trend_terms > 0:
            intercepts = self.fitted_model.params[:trend_terms].reshape(1, trend_terms)
        else:
            intercepts = np.array([[]])

        # Loop through the series, calculating each value based on the lagged values, coefficients, random error, and trend term
        for t in range(max_lag, n_samples):
            ar_term = 0
            for i in range(len(lags)):
                ar_term_iter = coefs[0, i] * series[t - lags[i]]
                ar_term += ar_term_iter

            trend_term = 0
            # If the trend is 'c' or 'ct', add a constant term
            if self.fitted_model.model.trend in ["c", "ct"]:
                trend_term += intercepts[0, 0]
            # If the trend is 't' or 'ct', add a linear time trend
            if self.fitted_model.model.trend in ["t", "ct"]:
                trend_term += intercepts[0, -1] * t

            series[t] = ar_term + trend_term + random_errors[t]

        return series

    def simulate_ar_process(
        self,
        resids_lags: Union[Integral, List[Integral]],
        resids_coefs: np.ndarray,
        resids: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """
        Simulate AR process from the fitted model.

        Parameters
        ----------
        resids_lags: Union[Integral, List[Integral]]
            The lags to be used in the AR process. Can be non-consecutive, but when called from `generate_samples_sieve`, it will be sorted.
        resids_coefs: np.ndarray
            The coefficients corresponding to each lag. Of shape (1, len(lags)). Sorted by `generate_samples_sieve` corresponding to the sorted `lags`.
        resids: np.ndarray
            The initial values for the simulation. Should be at least as long as the maximum lag.
        n_samples: int
            The number of samples to generate.

        Returns
        -------
        np.ndarray
            The simulated AR process as a 1D NumPy array.

        Raises
        ------
        ValueError
            If `resids_lags`, `resids_coefs`, or `resids` are not provided.
            If `resids_coefs` is not a 1D NumPy array.
            If `resids_coefs` is not the same length as `resids_lags`.
            If `resids` is not the same length as `X_fitted`.

        TypeError
            If `fitted_model` is not an instance of `AutoRegResultsWrapper`.
            If `resids_lags` is not an integer or a list of integers.
        """
        from statsmodels.tsa.ar_model import AutoRegResultsWrapper

        validate_integers(resids_lags, min_value=1)  # type: ignore

        if not isinstance(self.fitted_model, AutoRegResultsWrapper):
            # logger.error("fitted_model must be an instance of AutoRegResultsWrapper.")
            raise TypeError(
                f"fitted_model must be an instance of AutoRegResultsWrapper. Got {type(self.fitted_model)}."
            )

        if self.n_features > 1:
            raise ValueError("Only univariate time series are supported for the AR model.")
        if n_samples != len(resids):
            raise ValueError(
                "Length of 'resids' must be the same as the number of samples to generate."
            )

        # In case resids is 2d with shape (X, 1), convert it to 1d
        resids = resids.ravel()
        # In case X_fitted is 2d with shape (X, 1), convert it to 1d
        X_fitted = self.X_fitted.ravel()
        # Generate the bootstrap series
        bootstrap_series = np.zeros(n_samples, dtype=X_fitted.dtype)
        # Convert resids_lags to a NumPy array if it is not already. When called from `generate_samples_sieve`, it will be sorted.
        resids_lags = (
            np.arange(1, resids_lags + 1)
            if isinstance(resids_lags, Integral)
            else np.array(sorted(resids_lags))
        )  # type: ignore
        # resids_lags.shape: (n_lags,)
        max_lag = np.max(resids_lags)  # type: ignore
        if resids_coefs.shape[0] != 1:
            raise ValueError("AR coefficients must be a 1D NumPy array of shape (1, X)")
        if resids_coefs.shape[1] != len(resids_lags):  # type: ignore
            raise ValueError("Length of 'resids_coefs' must be the same as the length of 'lags'")

        # Simulate residuals using the AR model
        simulated_residuals = self._simulate_ar_residuals(
            lags=resids_lags,  # type: ignore
            coefs=resids_coefs,
            init=resids[:max_lag],
            max_lag=max_lag,
            n_samples=n_samples,
        )
        # simulated_residuals.shape: (n_samples,)

        bootstrap_series[:max_lag] = X_fitted[:max_lag]

        # Loop through the series, calculating each value based on the lagged values, coefficients, and random error
        for t in range(max_lag, n_samples):
            lagged_values = bootstrap_series[t - resids_lags]  # type: ignore
            # lagged_values.shape: (n_lags,)
            lagged_values = lagged_values.reshape(-1, 1)
            # lagged_values.shape: (n_lags, 1)
            bootstrap_series[t] = resids_coefs @ lagged_values + simulated_residuals[t]

        return bootstrap_series.reshape(-1, 1)

    def _simulate_non_ar_residuals(self, n_samples: int) -> np.ndarray:
        """
        Simulate residuals according to the model type.

        Parameters
        ----------
        n_samples: int
            The number of samples to generate.

        Returns
        -------
        np.ndarray
            The simulated residuals.
        """
        from arch.univariate.base import ARCHModelResult
        from statsmodels.tsa.arima.model import ARIMAResultsWrapper
        from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
        from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

        # self.rng is always a Generator instance. Generate an integer seed for methods that require it.
        current_sim_seed = self.rng.integers(0, 2**32 - 1)

        if isinstance(self.fitted_model, (ARIMAResultsWrapper, SARIMAXResultsWrapper)):
            # These models' simulate methods can take a Generator instance directly for random_state
            return self.fitted_model.simulate(
                nsimulations=n_samples + self.burnin,
                random_state=self.rng,
            )
        elif isinstance(self.fitted_model, VARResultsWrapper):
            # VARResultsWrapper.simulate_var takes an integer seed
            return self.fitted_model.simulate_var(
                steps=n_samples + self.burnin, seed=current_sim_seed
            )
        elif isinstance(self.fitted_model, ARCHModelResult):
            # Due to observed inconsistencies in arch library behavior in some environments
            # (ARCHModelResult.simulate missing, or model.simulate() rejecting random_state),
            # we fall back to calling model.simulate() without attempting to pass random_state
            # or using global np.random.seed(). This may compromise reproducibility for ARCH models
            # if the underlying arch model's simulate() method doesn't handle seeding consistently.

            sim_args = {
                "params": self.fitted_model.params,
                "nobs": n_samples + self.burnin,
                # No 'random_state' here
            }

            # Note: arch.__version__ and current_sim_seed are not used in this specific path
            # as we cannot reliably control seeding via arch's API in this scenario.

            simulation_output = self.fitted_model.model.simulate(**sim_args)  # type: ignore[attr-defined]
            return np.asarray(simulation_output.data.values)
        raise ValueError(f"Unsupported fitted model type {self.fitted_model}.")

    def simulate_non_ar_process(self, n_samples: int) -> np.ndarray:
        """
        Simulate a time series from the fitted model.

        Parameters
        ----------
        n_samples: int
            The number of samples to generate.

        Returns
        -------
            np.ndarray: The simulated time series.
        """
        from statsmodels.tsa.arima.model import ARIMAResultsWrapper
        from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
        from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

        simulated_residuals = self._simulate_non_ar_residuals(n_samples)
        simulated_residuals = np.reshape(simulated_residuals, (-1, self.n_features))
        # Discard the burn-in samples for certain models
        from arch.univariate.base import ARCHModelResult

        if isinstance(
            self.fitted_model,
            (
                VARResultsWrapper,
                ARIMAResultsWrapper,
                SARIMAXResultsWrapper,
                ARCHModelResult,
            ),
        ):
            simulated_residuals = simulated_residuals[self.burnin :]
        return self.X_fitted[:n_samples] + simulated_residuals[:n_samples]

    def generate_samples_sieve(
        self,
        model_type: ModelTypes,
        resids_lags: Optional[Union[Integral, List[Integral]]] = None,
        resids_coefs: Optional[np.ndarray] = None,
        resids: Optional[np.ndarray] = None,
        n_samples: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a bootstrap sample using the sieve bootstrap.

        Parameters
        ----------
        model_type: ModelTypes
            The model type used for the simulation.
        resids_lags: Optional[Union[Integral, List[Integral]]], optional
            The lags to be used in the AR process. Can be non-consecutive.
        resids_coefs: Optional[np.ndarray], optional
            The coefficients corresponding to each lag. Of shape (1, len(lags)).
        resids: Optional[np.ndarray], optional
            The initial values for the simulation. Should be at least as long as the maximum lag.
        n_samples: Optional[int], default=None
            The number of samples to generate. If None, uses self.n_samples.

        Returns
        -------
        np.ndarray
            The bootstrap sample.

        Raises
        ------
        ValueError
            If `resids_lags`, `resids_coefs`, or `resids` are not provided.
        """
        if n_samples is None:
            n_samples = self.n_samples

        validate_literal_type(model_type, ModelTypes)
        if model_type == "ar":
            self._validate_ar_simulation_params(
                {
                    "resids_lags": resids_lags,
                    "resids_coefs": resids_coefs,
                    "resids": resids,
                }
            )
            return self.simulate_ar_process(resids_lags, resids_coefs, resids, n_samples)  # type: ignore
        else:
            return self.simulate_non_ar_process(n_samples)

    def __repr__(self) -> str:
        return f"TimeSeriesSimulator(fitted_model={self.fitted_model}, n_samples={self.n_samples}, n_features={self.n_features})"

    def __str__(self) -> str:
        return f"TimeSeriesSimulator with {self.n_samples} samples and {self.n_features} features using fitted model {self.fitted_model}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TimeSeriesSimulator):
            return (
                self.fitted_model == other.fitted_model
                and np.array_equal(self.X_fitted, other.X_fitted)
                and self.rng == other.rng
            )
        return False
