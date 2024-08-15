from __future__ import annotations

import inspect
from collections.abc import Callable
from multiprocessing import Pool
from numbers import Integral
from typing import Any, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator
from scipy import stats
from skbase.base import BaseObject
from sklearn.decomposition import PCA  # type: ignore

from tsbootstrap.tsfit import TSFitBestLag
from tsbootstrap.utils.odds_and_ends import time_series_split
from tsbootstrap.utils.types import (
    BlockCompressorTypes,
    DistributionTypes,
    ModelTypes,
    ModelTypesWithoutArch,
    OrderTypes,
)
from tsbootstrap.utils.validate import validate_order, validate_rng


class BaseTimeSeriesBootstrap(BaseModel, BaseObject):
    """
    Base class for time series bootstrapping.

    This class provides the foundation for implementing various time series bootstrapping techniques.

    Raises
    ------
    ValueError
        If n_bootstraps is not greater than 0.
    """

    _tags = {
        "object_type": "bootstrap",
        "bootstrap_type": "other",
        "capability:multivariate": True,
    }

    n_bootstraps: int = Field(default=10, ge=1)
    rng: Optional[np.random.Generator] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    @field_validator("rng")
    @classmethod
    def validate_rng(cls, v):
        return validate_rng(v)

    def bootstrap(
        self,
        X: np.ndarray,
        return_indices: bool = False,
        y=None,
        test_ratio: Optional[float] = None,  # noqa: UP007
    ):
        """
        Generate bootstrapped samples of time series data.

        Parameters
        ----------
        X : 2D array-like of shape (n_timepoints, n_features)
            The endogenous time series to bootstrap.
        return_indices : bool, default=False
            If True, return index references for the bootstrap sample.
        y : array-like of shape (n_timepoints, n_features_exog), default=None
            Exogenous time series to use in bootstrapping.
        test_ratio : float, default=None
            If provided, this fraction of data is removed from the end before bootstrapping.

        Yields
        ------
        X_boot_i : 2D np.ndarray of shape (n_timepoints_boot_i, n_features)
            i-th bootstrapped sample of X.
        indices_i : 1D np.nparray of shape (n_timepoints_boot_i,), optional
            Index references for the i-th bootstrapped sample, if return_indices=True.
        """
        X, y = self._check_X_y(X, y)

        if test_ratio is not None:
            X_inner, _ = time_series_split(X, test_ratio=test_ratio)
            if y is not None:
                y_inner, _ = time_series_split(y, test_ratio=test_ratio)
            else:
                y_inner = None
        else:
            X_inner = X
            y_inner = y

        yield from self._bootstrap(
            X=X_inner, return_indices=return_indices, y=y_inner
        )

    def _bootstrap(self, X: np.ndarray, return_indices: bool = False, y=None):
        """
        Generate bootstrapped samples. To be implemented by derived classes.

        Parameters
        ----------
        X : 2D array-like of shape (n_timepoints, n_features)
            The endogenous time series to bootstrap.
        return_indices : bool, default=False
            If True, return index references for the bootstrap sample.
        y : array-like, default=None
            Exogenous time series to use in bootstrapping.

        Yields
        ------
        X_boot_i : 2D np.ndarray
            i-th bootstrapped sample of X.
        indices_i : 1D np.nparray, optional
            Index references if return_indices=True.
        """
        yield from self._generate_samples(
            X=X, return_indices=return_indices, y=y
        )

    def _generate_samples(
        self,
        X: np.ndarray,
        return_indices: bool = False,
        y=None,
        n_jobs: int = 1,
    ):
        """
        Generate bootstrapped samples directly.

        Parameters
        ----------
        X : array-like of shape (n_timepoints, n_features)
            The input samples.
        return_indices : bool, default=False
            If True, return index references for the bootstrap sample.
        y : array-like, default=None
            Exogenous time series to use in bootstrapping.
        n_jobs : int, default=1
            The number of jobs to run in parallel.

        Yields
        ------
        Iterator[np.ndarray]
            Bootstrapped samples, and optionally their indices.
        """
        if n_jobs == 1:
            # Run bootstrap generation sequentially in the main process
            for _ in range(self.n_bootstraps):
                indices, data = self._generate_samples_single_bootstrap(X, y)
                data = np.concatenate(data, axis=0)
                if return_indices:
                    # hack to fix known issue with non-concatenated index sets
                    # see bug issue #81
                    if isinstance(indices, list):
                        indices = np.concatenate(indices, axis=0)
                    yield data, indices
                else:
                    yield data
        else:
            # Use multiprocessing to handle bootstrapping
            args = [(X, y) for _ in range(self.n_bootstraps)]
            with Pool(n_jobs) as pool:
                results = pool.starmap(
                    self._generate_samples_single_bootstrap, args
                )

            for indices, data in results:
                data = np.concatenate(data, axis=0)
                if return_indices:
                    # hack to fix known issue with non-concatenated index sets
                    # see bug issue #81
                    if isinstance(indices, list):
                        indices = np.concatenate(indices, axis=0)
                    yield data, indices
                else:
                    yield data

    def _generate_samples_single_bootstrap(self, X: np.ndarray, y=None):
        """Generate list of bootstraps for a single bootstrap iteration."""
        raise NotImplementedError("abstract method")

    def _check_X_y(self, X, y):
        """Check X and y inputs, for bootstrap and get_n_bootstraps methods.

        Checks X to be a 2D array-like, and y to be a 2D array-like or None.
        If X is 1D np.ndarray, it is expanded to 2D via np.expand_dims.

        Parameters
        ----------
        X : checked 2D array-like of shape (n_timepoints, n_features)
            The endogenous time series to bootstrap.
            Dimension 0 is assumed to be the time dimension, ordered
        y : checked array-like of shape (n_timepoints, n_features_exog), default=None
            Exogenous time series to use in bootstrapping.

        Returns
        -------
        X : np.ndarray, coerced to 2D array-like of shape (n_timepoints, n_features)
            The checked endogenous time series.
        y : np.ndarray or None, identical with y
            The checked exogenous time series.

        Raises
        ------
        ValueError : If the input is not valid.
        """
        if X is not None:
            X = np.asarray(X)
            if len(X.shape) < 2:
                print(X)
                X = np.expand_dims(X, 1)

            X = self._check_input(X)
        if y is not None:
            y = self._check_input(y, enforce_univariate=False)

        return X, y

    def _check_input(self, X, enforce_univariate=True):
        """Checks if the input is valid.

        Parameters
        ----------
        X : list of np.ndarray
            The input to check.
        enforce_univariate : bool, default=True
            Whether to enforce univariate input.

        Returns
        -------
        object : The input object if it is valid.

        Raises
        ------
        ValueError
            If the input is not valid.
        """
        if np.any(np.diff([len(x) for x in X]) != 0):
            raise ValueError("All time series must be of the same length.")

        self_can_only_univariate = not self.get_tag("capability:multivariate")
        check_univariate = enforce_univariate and self_can_only_univariate
        if check_univariate and X.shape[1] > 1:
            raise ValueError(
                f"Unsupported input type: the estimator {type(self)} "
                "does not support multivariate endogeneous time series (X argument). "
                "Pass an 1D np.array, or a 2D np.array with a single column."
            )

        return X

    def get_n_bootstraps(self, X=None, y=None) -> int:
        """Returns the number of bootstrap instances produced by the bootstrap.

        Parameters
        ----------
        X : 2D array-like of shape (n_timepoints, n_features)
            The endogenous time series to bootstrap.
            Dimension 0 is assumed to be the time dimension, ordered
        y : array-like of shape (n_timepoints, n_features_exog), default=None
            Exogenous time series to use in bootstrapping.

        Returns
        -------
        int : The number of bootstrap instances produced by the bootstrap.
        """
        return self.n_bootstraps  # type: ignore


class BaseResidualBootstrap(BaseTimeSeriesBootstrap):
    """Base class for residual bootstrap.

    Parameters
    ----------
    n_bootstraps : int, default=10
        The number of bootstrap samples to create.
    rng : int or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.
    model_type : {'ar', 'arima', 'sarima', 'var'}, default='ar'
        The type of model to use for fitting.
    model_params : dict, default=None
        Additional keyword arguments to pass to the TSFit model.
    order : int or tuple or list, default=None
        The order of the model. If None, the best order is chosen via TSFitBestLag.
        - For AR: int or list of non-consecutive ints
        - For ARIMA: tuple (p, d, q)
        - For SARIMA: tuple (p, d, q, s)
        - For VAR: int
        Note: TSFitBestLag only chooses the best lag (p), not full order tuples.
    save_models : bool, default=False
        Whether to save the fitted models.

    Attributes
    ----------
    fit_model : TSFitBestLag
        The fitted model.
    resids : ndarray
        The residuals of the fitted model.
    X_fitted : ndarray
        The fitted values of the model.
    coefs : ndarray
        The coefficients of the fitted model.

    Methods
    -------
    _fit_model(X, y=None)
        Fits the model to the data and stores the residuals.
    """

    _tags = {
        "python_dependencies": "statsmodels",
        "bootstrap_type": "residual",
        "capability:multivariate": False,
    }

    model_type: ModelTypesWithoutArch = Field(default="ar")
    order: OrderTypes = Field(default=None)
    save_models: bool = Field(default=False)
    model_params: dict[str, Any] = Field(default_factory=dict)

    # Additional attributes
    fit_model: Optional[Any] = Field(default=None, init=False)
    resids: Optional[np.ndarray] = Field(default=None, init=False)
    X_fitted: Optional[np.ndarray] = Field(default=None, init=False)
    coefs: Optional[np.ndarray] = Field(default=None, init=False)

    @field_validator("model_type", mode="before")
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        """Validate and normalize the model type."""
        return v.lower()

    @field_validator("order")
    @classmethod
    def validate_order(cls, v):
        """Validate the order parameter."""
        return validate_order(v)

    def __repr__(self):
        """Return a string representation of the object."""
        return (
            f"BaseResidualBootstrap(n_bootstraps={self.n_bootstraps}, "
            f"model_type='{self.model_type}', order={self.order}, "
            f"save_models={self.save_models}, "
            f"model_params={self.model_params})"
        )

    def _fit_model(self, X: np.ndarray, y=None) -> None:
        """Fit the model to the data and store the residuals.

        Parameters
        ----------
        X : ndarray
            The input time series data.
        y : ndarray, optional
            Additional exogenous variables.

        Notes
        -----
        This method fits the model and updates the following attributes:
        fit_model, X_fitted, resids, order, and coefs.
        """
        if (
            self.resids is None
            or self.X_fitted is None
            or self.fit_model is None
            or self.coefs is None
        ):
            fit_obj = TSFitBestLag(
                model_type=self.model_type,
                order=self.order,
                save_models=self.save_models,
                **self.model_params,
            )
            self.fit_model = fit_obj.fit(X=X, y=y).model
            self.X_fitted = fit_obj.get_fitted_X()
            self.resids = fit_obj.get_residuals()
            self.order = fit_obj.get_order()
            self.coefs = fit_obj.get_coefs()


class BaseMarkovBootstrap(BaseResidualBootstrap):
    """
    Base class for Markov bootstrap.

    Parameters
    ----------
    n_bootstraps : int, default=10
        The number of bootstrap samples to create.
    method : {'first', 'middle', 'last', 'mean', 'mode', 'median', 'kmeans', 'kmedians', 'kmedoids'}, default='middle'
        The method to use for compressing the blocks.
    apply_pca_flag : bool, default=False
        Whether to apply PCA to the residuals before fitting the HMM.
    pca : PCA, optional
        The PCA object to use for applying PCA to the residuals.
    n_iter_hmm : int, default=10
        Number of iterations for fitting the HMM.
    n_fits_hmm : int, default=1
        Number of times to fit the HMM.
    blocks_as_hidden_states_flag : bool, default=False
        Whether to use blocks as hidden states.
    n_states : int, default=2
        Number of states for the HMM.
    model_type : {'ar', 'arima', 'sarima', 'var', 'arch'}, default='ar'
        The type of model to use for fitting.
    model_params : dict, optional
        Additional keyword arguments to pass to the TSFit model.
    order : int or tuple or list, optional
        The order of the model. If None, the best order is chosen via TSFitBestLag.
        - For AR: int or list of non-consecutive ints
        - For ARIMA: tuple (p, d, q)
        - For SARIMA: tuple (p, d, q, s)
        - For VAR and ARCH: int
        Note: TSFitBestLag only chooses the best lag (p), not full order tuples.
    save_models : bool, default=False
        Whether to save the fitted models.
    rng : int or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

    Attributes
    ----------
    hmm_object : MarkovSampler or None
        The MarkovSampler object used for sampling.

    Notes
    -----
    Fitting Markov models is expensive, so the model is fitted once to the residuals.
    New samples are generated by changing the random seed.
    """

    method: BlockCompressorTypes = Field(default="middle")
    apply_pca_flag: bool = Field(default=False)
    pca: Optional[PCA] = Field(default=None)
    n_iter_hmm: Integral = Field(default=10, ge=1)
    n_fits_hmm: Integral = Field(default=1, ge=1)
    blocks_as_hidden_states_flag: bool = Field(default=False)
    n_states: Integral = Field(default=2, ge=2)

    hmm_object: Optional[Any] = Field(default=None, init=False)

    def __repr__(self):
        """Return a string representation of the object."""
        return (
            f"BaseMarkovBootstrap(n_bootstraps={self.n_bootstraps}, "
            f"method='{self.method}', apply_pca_flag={self.apply_pca_flag}, "
            f"pca={self.pca}, n_iter_hmm={
                self.n_iter_hmm}, n_fits_hmm={self.n_fits_hmm}, "
            f"blocks_as_hidden_states_flag={
                self.blocks_as_hidden_states_flag}, "
            f"n_states={self.n_states}, model_type='{
                self.model_type}', order={self.order}, "
            f"save_models={self.save_models})"
        )


class BaseStatisticPreservingBootstrap(BaseTimeSeriesBootstrap):
    """Bootstrap class that generates samples preserving a specific statistic.

    This class generates bootstrapped time series data, preserving a given statistic
    (e.g., mean, median) calculated from the original data.

    Parameters
    ----------
    n_bootstraps : int, default=10
        The number of bootstrap samples to create.
    statistic : callable, default=np.mean
        Function to compute the statistic to be preserved.
    statistic_axis : int, default=0
        The axis along which to compute the statistic.
    statistic_keepdims : bool, default=False
        Whether to keep the dimensions of the statistic.
    rng : int or np.random.Generator, default=np.random.default_rng()
        Random number generator or seed for bootstrap samples.

    Attributes
    ----------
    statistic_x : ndarray
        The statistic calculated from the original data, used as a parameter
        for generating bootstrapped samples.

    Methods
    -------
    _calculate_statistic(X)
        Calculate the statistic from the input data.
    """

    statistic: Optional[Callable] = Field(default=np.mean)
    statistic_axis: Integral = Field(default=0)
    statistic_keepdims: bool = Field(default=False)

    statistic_x: Optional[np.ndarray] = Field(default=None, init=False)

    def __repr__(self):
        """Return a string representation of the object."""
        return (
            f"BaseStatisticPreservingBootstrap(n_bootstraps={
                self.n_bootstraps}, "
            f"statistic={self.statistic}, statistic_axis={
                self.statistic_axis}, "
            f"statistic_keepdims={self.statistic_keepdims}, rng={self.rng})"
        )

    def _calculate_statistic(self, X: np.ndarray) -> np.ndarray:
        """Calculate the statistic from the input data.

        Parameters
        ----------
        X : ndarray
            The input data.

        Returns
        -------
        ndarray
            The calculated statistic.

        Raises
        ------
        ValueError
            If self.statistic is None.
        """
        if self.statistic is None:
            raise ValueError("The statistic function is not set.")

        params = inspect.signature(self.statistic).parameters
        kwargs_stat = {
            "axis": self.statistic_axis,
            "keepdims": self.statistic_keepdims,
        }
        kwargs_stat = {k: v for k, v in kwargs_stat.items() if k in params}
        return self.statistic(X, **kwargs_stat)


# We can only fit uni-variate distributions, so X must be a 1D array, and `model_type` in BaseResidualBootstrap must not be "var".


class BaseDistributionBootstrap(BaseResidualBootstrap):
    r"""
    Implementation of the Distribution Bootstrap (DB) method for time series data.

    Generates bootstrapped samples by fitting a distribution to the residuals
    and generating new residuals from the fitted distribution.

    Parameters
    ----------
    n_bootstraps : int, default=10
        Number of bootstrap samples to create.
    distribution : {'poisson', 'exponential', 'normal', 'gamma', 'beta', 'lognormal', 'weibull', 'pareto', 'geometric', 'uniform'}, default='normal'
        Distribution to use for generating bootstrapped samples.
    refit : bool, default=False
        Whether to refit the distribution for each bootstrap.
    model_type : {'ar', 'arima', 'sarima', 'arch'}, default='ar'
        Type of model to use.
    model_params : dict, optional
        Additional keyword arguments for the TSFit model.
    order : int or tuple or list, optional
        Order of the model. If None, best order chosen via TSFitBestLag.
        For specifics, see class docstring.
    save_models : bool, default=False
        Whether to save fitted models.
    rng : int or np.random.Generator, default=np.random.default_rng()
        Random number generator or seed for bootstrap samples.

    Attributes
    ----------
    resids_dist : scipy.stats.rv_continuous or None
        Distribution object for generating bootstrapped samples.
    resids_dist_params : tuple or None
        Parameters of the distribution.

    Notes
    -----
    The DB method is defined as:

    .. math::
        \hat{X}_t = \hat{\mu} + \epsilon_t

    where :math:`\epsilon_t \sim F_{\hat{\epsilon}}` is sampled from the
    distribution :math:`F_{\hat{\epsilon}}` fitted to the residuals.

    References
    ----------
    .. [1] Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap.
           Journal of the American Statistical Association, 89(428), 1303-1313.
    """

    @staticmethod
    def fit_continuous(
        dist: stats.rv_continuous, data: np.ndarray
    ) -> tuple[float, ...]:
        return dist.fit(data)

    @staticmethod
    def fit_poisson(
        dist: stats.rv_discrete, data: np.ndarray
    ) -> tuple[Union[float, np.floating], ...]:
        return (np.mean(data),)

    @staticmethod
    def fit_geometric(
        dist: stats.rv_discrete, data: np.ndarray
    ) -> tuple[Union[float, np.floating], ...]:
        return (1 / (np.mean(data) + 1),)

    DistributionMethod = tuple[
        Any, Callable[[Any, np.ndarray], tuple[Union[float, np.floating], ...]]
    ]

    distribution_methods: dict[DistributionTypes, DistributionMethod] = Field(
        default_factory=lambda: {
            DistributionTypes.POISSON: (
                stats.poisson,
                BaseDistributionBootstrap.fit_poisson,
            ),
            DistributionTypes.EXPONENTIAL: (
                stats.expon,
                BaseDistributionBootstrap.fit_continuous,
            ),
            DistributionTypes.NORMAL: (
                stats.norm,
                BaseDistributionBootstrap.fit_continuous,
            ),
            DistributionTypes.GAMMA: (
                stats.gamma,
                BaseDistributionBootstrap.fit_continuous,
            ),
            DistributionTypes.BETA: (
                stats.beta,
                BaseDistributionBootstrap.fit_continuous,
            ),
            DistributionTypes.LOGNORMAL: (
                stats.lognorm,
                BaseDistributionBootstrap.fit_continuous,
            ),
            DistributionTypes.WEIBULL: (
                stats.weibull_min,
                BaseDistributionBootstrap.fit_continuous,
            ),
            DistributionTypes.PARETO: (
                stats.pareto,
                BaseDistributionBootstrap.fit_continuous,
            ),
            DistributionTypes.GEOMETRIC: (
                stats.geom,
                BaseDistributionBootstrap.fit_geometric,
            ),
            DistributionTypes.UNIFORM: (
                stats.uniform,
                BaseDistributionBootstrap.fit_continuous,
            ),
        }
    )

    distribution: DistributionTypes = Field(default=DistributionTypes.NORMAL)
    refit: bool = Field(default=False)

    resids_dist: Optional[Any] = Field(default=None, init=False)
    resids_dist_params: tuple[Union[float, np.floating], ...] = Field(
        default_factory=tuple, init=False
    )

    @field_validator("distribution", mode="before")
    @classmethod
    def validate_distribution_type(cls, v: str) -> str:
        """Validate and normalize the distribution type."""
        return v.lower()

    def __repr__(self):
        """Return a string representation of the object."""
        return (
            f"BaseDistributionBootstrap(n_bootstraps={self.n_bootstraps}, "
            f"distribution='{self.distribution}', refit={self.refit}, "
            f"model_type='{self.model_type}', order={self.order}, "
            f"save_models={self.save_models})"
        )

    def _fit_distribution(
        self, resids: np.ndarray
    ) -> tuple[Any, tuple[Union[float, np.floating], ...]]:
        """Fit the specified distribution to the residuals.

        Parameters
        ----------
        resids : np.ndarray
            The residuals to fit the distribution to.

        Returns
        -------
        dist_class : Any
            The distribution class (from scipy.stats) that was fitted.
        params : Tuple[Union[float, np.floating], ...]
            The parameters of the fitted distribution.

        Notes
        -----
        This method uses the distribution and fitting function specified in
        `self.distribution_methods` for the current `self.distribution`.
        """
        dist_class, fit_func = self.distribution_methods[self.distribution]
        params = fit_func(dist_class, resids)
        return dist_class, params


class BaseSieveBootstrap(BaseResidualBootstrap):
    """
    Base class for Sieve bootstrap.

    Implements the Sieve bootstrap method, fitting models to residuals
    and generating bootstrapped samples.

    Parameters
    ----------
    resids_model_type : {'ar', 'arima', 'sarima', 'var', 'arch'}, default='ar'
        Model type for fitting residuals.
    resids_order : int or tuple or list, optional
        Order of the residuals model. If None, automatically determined.
    save_resids_models : bool, default=False
        Whether to save fitted residuals models.
    resids_model_params : dict, optional
        Additional parameters for the SieveBootstrap class.
    model_type : {'ar', 'arima', 'sarima', 'var', 'arch'}, default='ar'
        Model type for the main time series.
    model_params : dict, optional
        Additional parameters for the TSFit model.
    order : int or tuple or list, optional
        Order of the main model. If None, best order chosen via TSFitBestLag.
        For specifics, see class docstring.
    n_bootstraps : int, default=10
        Number of bootstrap samples to create.
    save_models : bool, default=False
        Whether to save fitted models.

    Attributes
    ----------
    resids_coefs : ndarray or None
        Coefficients of the fitted residual model.
    resids_fit_model : object or None
        Fitted residual model object.

    Notes
    -----
    The Sieve bootstrap is a parametric method that generates bootstrapped
    samples by fitting a model to the residuals and then generating new
    residuals from the fitted model. These new residuals are added to the
    fitted values to create the bootstrapped samples.
    """

    resids_model_type: ModelTypes = Field(default="ar")
    resids_order: OrderTypes = Field(default=None)
    save_resids_models: bool = Field(default=False)
    resids_model_params: dict[str, Any] = Field(default_factory=dict)

    resids_fit_model: Optional[Any] = Field(default=None, init=False)
    resids_coefs: Optional[np.ndarray] = Field(default=None, init=False)

    @model_validator(mode="after")
    def validate_model(self):
        """Validate model consistency."""
        if self.resids_model_type == "var" and self.model_type != "var":
            raise ValueError(
                "resids_model_type can be 'var' only if model_type is also 'var'."
            )
        return self

    @field_validator("resids_model_type", mode="before")
    @classmethod
    def validate_resids_model_type(cls, v: str) -> str:
        """Validate and normalize residuals model type."""
        return v.lower()

    @field_validator("resids_order")
    @classmethod
    def validate_resids_order(cls, v):
        """Validate residuals order."""
        return validate_order(v)

    def __repr__(self):
        """Return a string representation of the object."""
        return (
            f"BaseSieveBootstrap(n_bootstraps={self.n_bootstraps}, "
            f"resids_model_type='{self.resids_model_type}', resids_order={
                self.resids_order}, "
            f"save_resids_models={self.save_resids_models}, resids_model_params={
                self.resids_model_params}, "
            f"model_type='{self.model_type}', order={
                self.order}, save_models={self.save_models})"
        )

    def _fit_resids_model(self, X: np.ndarray) -> None:
        """Fit the residual model to the residuals.

        Parameters
        ----------
        X : ndarray
            The residuals to fit the model to.
        """
        if self.resids_fit_model is None or self.resids_coefs is None:
            resids_fit_obj = TSFitBestLag(
                model_type=self.resids_model_type,
                order=self.resids_order,
                save_models=self.save_resids_models,
                **self.resids_model_params,
            )
            resids_fit_model = resids_fit_obj.fit(X, y=None).model
            resids_order = resids_fit_obj.get_order()
            resids_coefs = resids_fit_obj.get_coefs()
            self.resids_fit_model = resids_fit_model
            self.resids_order = resids_order
            self.resids_coefs = resids_coefs
