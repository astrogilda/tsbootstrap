from __future__ import annotations

import inspect
from abc import ABCMeta, abstractmethod
from collections.abc import Iterator
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import rv_continuous
from skbase.base import BaseObject

from tsbootstrap.utils.odds_and_ends import time_series_split

if TYPE_CHECKING:
    from tsbootstrap.base_bootstrap_configs import (
        BaseDistributionBootstrapConfig,
        BaseMarkovBootstrapConfig,
        BaseResidualBootstrapConfig,
        BaseSieveBootstrapConfig,
        BaseStatisticPreservingBootstrapConfig,
        BaseTimeSeriesBootstrapConfig,
    )

from tsbootstrap.tsfit import TSFitBestLag


class BaseTimeSeriesBootstrap(BaseObject):
    """
    Base class for time series bootstrapping.

    Raises
    ------
    ValueError
        If n_bootstraps is not greater than 0.
    """

    _tags = {"object_type": "bootstrap"}

    def __init__(self, config: BaseTimeSeriesBootstrapConfig) -> None:
        self.config = config

    # TODO 0.2.0: change default value of test_ratio to 0.0
    def bootstrap(
        self,
        X: np.ndarray,
        return_indices: bool = False,
        exog: np.ndarray | None = None,
        test_ratio: float = None,
    ) -> Iterator[np.ndarray] | Iterator[tuple[list[np.ndarray], np.ndarray]]:
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : 2D array-like of shape (n_timepoints, n_features)
            The endogenous time series to bootstrap.
            Dimension 0 is assumed to be the time dimension, ordered
        return_indices : bool, default=False
            If True, a second output is retured, integer locations of
            index references for the bootstrap sample, in reference to original indices.
            Indexed values do are not necessarily identical with bootstrapped values.
        exog : array-like of shape (n_timepoints, n_features_exog), default=None
            Exogenous time series to use in bootstrapping.
        test_ratio : float, default=0.2
            The ratio of test samples to total samples.
            If provided, test_ratio fraction the data (rounded up)
            is removed from the end before applying the bootstrap logic.

        Yields
        ------
        X_boot_i : 2D np.ndarray-like of shape (n_timepoints_boot_i, n_features)
            i-th bootstrapped sample of X.
        indices_i : 1D np.nparray of shape (n_timepoints_boot_i,) integer values,
            only returned if return_indices=True.
            Index references for the i-th bootstrapped sample of X.
            Indexed values do are not necessarily identical with bootstrapped values.
        """
        # TODO 0.2.0: remove this block, change default value to 0.0
        if test_ratio is None:
            from warnings import warn

            test_ratio = 0.2
            warn(
                "in bootstrap, the default value for test_ratio will chage to 0.0 "
                "from tsbootstrap version 0.2.0 onwards. "
                "To avoid chages in logic, please specify test_ratio explicitly. ",
                stacklevel=2,
            )

        X = np.asarray(X)
        if len(X.shape) < 2:
            X = np.expand_dims(X, 1)

        self._check_input(X)

        X_train, X_test = time_series_split(X, test_ratio=test_ratio)

        if exog is not None:
            self._check_input(exog)
            exog_train, _ = time_series_split(exog, test_ratio=test_ratio)
        else:
            exog_train = None

        tuple_iter = self._generate_samples(
            X=X_train, return_indices=return_indices, exog=exog_train
        )

        yield from tuple_iter

    def _generate_samples(
        self,
        X: np.ndarray,
        return_indices: bool = False,
        exog: np.ndarray | None = None,
    ) -> Iterator[np.ndarray] | Iterator[tuple[list[np.ndarray], np.ndarray]]:
        """Generates bootstrapped samples directly.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Yields
        ------
        Iterator[np.ndarray]
            An iterator over the bootstrapped samples.

        """
        for _ in range(self.config.n_bootstraps):
            indices, data = self._generate_samples_single_bootstrap(
                X=X, exog=exog
            )
            data = np.concatenate(data, axis=0)
            if return_indices:
                yield indices, data  # type: ignore
            else:
                yield data

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, exog: np.ndarray | None = None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Generates list of bootstrapped indices and samples for a single bootstrap iteration.

        Should be implemented in derived classes.
        """
        raise NotImplementedError("abstract method")

    def _check_input(self, X):
        """Checks if the input is valid."""
        if np.any(np.diff([len(x) for x in X]) != 0):
            raise ValueError("All time series must be of the same length.")

    def get_n_bootstraps(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> Integral:
        """Returns the number of bootstrapping iterations."""
        return self.config.n_bootstraps  # type: ignore


class BaseResidualBootstrap(BaseTimeSeriesBootstrap):
    """
    Base class for residual bootstrap.

    Attributes
    ----------
    fit_model : TSFitBestLag
        The fitted model.
    resids : np.ndarray
        The residuals of the fitted model.
    X_fitted : np.ndarray
        The fitted values of the fitted model.
    coefs : np.ndarray
        The coefficients of the fitted model.

    Methods
    -------
    __init__ : Initialize self.
    _fit_model : Fits the model to the data and stores the residuals.
    """

    def __init__(
        self,
        config: BaseResidualBootstrapConfig,
    ):
        """
        Initialize self.

        Parameters
        ----------
        config : BaseResidualBootstrapConfig
            The configuration object.
        """
        super().__init__(config=config)
        self.config = config

        self.fit_model = None
        self.resids = None
        self.X_fitted = None
        self.coefs = None

    def _fit_model(
        self, X: np.ndarray, exog: np.ndarray | None = None
    ) -> None:
        """Fits the model to the data and stores the residuals."""
        if (
            self.resids is None
            or self.X_fitted is None
            or self.fit_model is None
            or self.coefs is None
        ):
            fit_obj = TSFitBestLag(
                model_type=self.config.model_type,
                order=self.config.order,
                save_models=self.config.save_models,
                **self.config.model_params,
            )
            self.fit_model = fit_obj.fit(X=X, exog=exog).model
            self.X_fitted = fit_obj.get_fitted_X()
            self.resids = fit_obj.get_residuals()
            self.order = fit_obj.get_order()
            self.coefs = fit_obj.get_coefs()

    def __repr__(self) -> str:
        """Returns the string representation of the object."""
        return f"{self.__class__.__name__}(config={self.config})"

    def __str__(self) -> str:
        """Returns the string representation of the object."""
        return f"{self.__class__.__name__}(config={self.config})"

    def __eq__(self, __value: object) -> bool:
        """Returns True if the objects are equal, False otherwise."""
        if not isinstance(__value, BaseResidualBootstrap):
            return NotImplemented
        return self.config == __value.config

    def __hash__(self) -> int:
        """Returns the hash of the object."""
        return hash((super().__hash__(), self.config))


class BaseMarkovBootstrap(BaseResidualBootstrap):
    """
    Base class for Markov bootstrap.

    Attributes
    ----------
    hmm_object : MarkovSampler or None
        The MarkovSampler object used for sampling.

    Methods
    -------
    __init__ : Initialize the Markov bootstrap.

    Notes
    -----
    Fitting Markov models is expensive, hence we do not allow re-fititng. We instead fit once to the residuals and generate new samples by changing the random_seed.
    """

    def __init__(
        self,
        config: BaseMarkovBootstrapConfig,
    ):
        """
        Initialize self.

        Parameters
        ----------
        config : BaseMarkovBootstrapConfig
            The configuration object.
        """
        super().__init__(config=config)
        self.config = config

        self.hmm_object = None

    def __repr__(self) -> str:
        """Returns the string representation of the object."""
        return f"{self.__class__.__name__}(config={self.config})"

    def __str__(self) -> str:
        """Returns the string representation of the object."""
        return self.__repr__()

    def __eq__(self, __value: object) -> bool:
        """Returns True if the objects are equal, False otherwise."""
        if not isinstance(__value, BaseMarkovBootstrap):
            return NotImplemented
        return self.config == __value.config

    def __hash__(self) -> int:
        """Returns the hash of the object."""
        return hash((super().__hash__(), self.config))


class BaseStatisticPreservingBootstrap(BaseTimeSeriesBootstrap):
    """Bootstrap class that generates bootstrapped samples preserving a specific statistic.

    This class generates bootstrapped time series data, preserving a given statistic (such as mean, median, etc.)
    The statistic is calculated from the original data and then used as a parameter for generating the bootstrapped samples.
    For example, if the statistic is np.mean, then the mean of the original data is calculated and then used as a parameter for generating the bootstrapped samples.

    Attributes
    ----------
    statistic_X : np.ndarray, default=None
        The statistic calculated from the original data. This is used as a parameter for generating the bootstrapped samples.

    Methods
    -------
    __init__ : Initialize the BaseStatisticPreservingBootstrap class.
    _calculate_statistic(X: np.ndarray) -> np.ndarray : Calculate the statistic from the input data.
    """

    def __init__(
        self,
        config: BaseStatisticPreservingBootstrapConfig,
    ) -> None:
        """
        Initialize the BaseStatisticPreservingBootstrap class.

        Parameters
        ----------
        config : BaseStatisticPreservingBootstrapConfig
            The configuration object.
        """
        super().__init__(config=config)
        self.config = config

        self.statistic_X = None

    def _calculate_statistic(self, X: np.ndarray) -> np.ndarray:
        params = inspect.signature(self.config.statistic).parameters
        kwargs_stat = {
            "axis": self.config.statistic_axis,
            "keepdims": self.config.statistic_keepdims,
        }
        kwargs_stat = {k: v for k, v in kwargs_stat.items() if k in params}
        statistic_X = self.config.statistic(X, **kwargs_stat)
        return statistic_X

    def __repr__(self) -> str:
        """Returns the string representation of the object."""
        return f"{self.__class__.__name__}(config={self.config})"

    def __str__(self) -> str:
        """Returns the string representation of the object."""
        return self.__repr__()

    def __eq__(self, __value: object) -> bool:
        """Returns True if the objects are equal, False otherwise."""
        if not isinstance(__value, BaseStatisticPreservingBootstrap):
            return NotImplemented
        return self.config == __value.config

    def __hash__(self) -> int:
        """Returns the hash of the object."""
        return hash((super().__hash__(), self.config))


# We can only fit uni-variate distributions, so X must be a 1D array, and `model_type` in BaseResidualBootstrap must not be "var".
class BaseDistributionBootstrap(BaseResidualBootstrap):
    r"""
    Implementation of the Distribution Bootstrap (DB) method for time series data.

    The DB method is a non-parametric method that generates bootstrapped samples by fitting a distribution to the residuals and then generating new residuals from the fitted distribution. The new residuals are then added to the fitted values to create the bootstrapped samples.

    Attributes
    ----------
    resids_dist : scipy.stats.rv_continuous or None
        The distribution object used to generate the bootstrapped samples. If None, the distribution has not been fit yet.
    resids_dist_params : tuple or None
        The parameters of the distribution used to generate the bootstrapped samples. If None, the distribution has not been fit yet.

    Methods
    -------
    __init__ : Initialize the BaseDistributionBootstrap class.
    fit_distribution(resids: np.ndarray) -> tuple[rv_continuous, tuple]
        Fit the specified distribution to the residuals and return the distribution object and the parameters of the distribution.

    Notes
    -----
    The DB method is defined as:

    .. math::
        \\hat{X}_t = \\hat{\\mu} + \\epsilon_t

    where :math:`\\epsilon_t \\sim F_{\\hat{\\epsilon}}` is a random variable
    sampled from the distribution :math:`F_{\\hat{\\epsilon}}` fitted to the
    residuals :math:`\\hat{\\epsilon}`.

    References
    ----------
    .. [^1^] Politis, Dimitris N., and Joseph P. Romano. "The stationary bootstrap." Journal of the American Statistical Association 89.428 (1994): 1303-1313.
    """

    def __init__(
        self,
        config: BaseDistributionBootstrapConfig,
    ) -> None:
        """
        Initialize the BaseStatisticPreservingBootstrap class.

        Parameters
        ----------
        config : BaseStatisticPreservingBootstrapConfig
            The configuration object.
        """
        super().__init__(config=config)
        self.config = config

        self.resids_dist = None
        self.resids_dist_params = ()

    def _fit_distribution(
        self, resids: np.ndarray
    ) -> tuple[rv_continuous, tuple]:
        """
        Fit the specified distribution to the residuals and return the distribution object and the parameters of the distribution.

        Parameters
        ----------
        resids : np.ndarray
            The residuals to fit the distribution to.

        Returns
        -------
        resids_dist : scipy.stats.rv_continuous
            The distribution object used to generate the bootstrapped samples.
        resids_dist_params : tuple
            The parameters of the distribution used to generate the bootstrapped samples.
        """
        resids_dist = self.config.distribution_methods[
            self.config.distribution
        ]
        # Fit the distribution to the residuals
        resids_dist_params = resids_dist.fit(resids)
        return resids_dist, resids_dist_params

    def __repr__(self) -> str:
        """Returns the string representation of the object."""
        return f"{self.__class__.__name__}(config={self.config})"

    def __str__(self) -> str:
        """Returns the string representation of the object."""
        return self.__repr__()

    def __eq__(self, __value: object) -> bool:
        """Returns True if the objects are equal, False otherwise."""
        if not isinstance(__value, BaseDistributionBootstrap):
            return NotImplemented
        return self.config == __value.config

    def __hash__(self) -> int:
        """Returns the hash of the object."""
        return hash((super().__hash__(), self.config))


class BaseSieveBootstrap(BaseResidualBootstrap):
    """
    Base class for Sieve bootstrap.

    This class provides the core functionalities for implementing the Sieve bootstrap method, allowing for the fitting of various models to the residuals and generation of bootstrapped samples. The Sieve bootstrap is a parametric method that generates bootstrapped samples by fitting a model to the residuals and then generating new residuals from the fitted model. The new residuals are then added to the fitted values to create the bootstrapped samples.

    Attributes
    ----------
    resids_coefs : type or None
        Coefficients of the fitted residual model. Replace "type" with the specific type if known.
    resids_fit_model : type or None
        Fitted residual model object. Replace "type" with the specific type if known.

    Methods
    -------
    __init__ : Initialize the BaseSieveBootstrap class.
    _fit_resids_model : Fit the residual model to the residuals.
    """

    def __init__(
        self,
        config: BaseSieveBootstrapConfig,
    ) -> None:
        """
        Initialize the BaseSieveBootstrap class.

        Parameters
        ----------
        config : BaseSieveBootstrapConfig
            The configuration object.
        """
        super().__init__(config=config)
        self.config = config
        self.resids_coefs = None
        self.resids_fit_model = None

    def _fit_resids_model(self, X: np.ndarray) -> None:
        """
        Fit the residual model to the residuals.

        Parameters
        ----------
        X : np.ndarray
            The residuals to fit the model to.

        Returns
        -------
        resids_fit_model : type
            The fitted residual model object. Replace "type" with the specific type if known.
        resids_order : Integral or list or tuple
            The order of the fitted residual model.
        resids_coefs : np.ndarray
            The coefficients of the fitted residual model.
        """
        if self.resids_fit_model is None or self.resids_coefs is None:
            resids_fit_obj = TSFitBestLag(
                model_type=self.config.resids_model_type,
                order=self.config.resids_order,
                save_models=self.config.save_resids_models,
                **self.config.resids_model_params,
            )
            resids_fit_model = resids_fit_obj.fit(X, exog=None).model
            resids_order = resids_fit_obj.get_order()
            resids_coefs = resids_fit_obj.get_coefs()
            self.resids_fit_model = resids_fit_model
            self.resids_order = resids_order
            self.resids_coefs = resids_coefs

    def __repr__(self) -> str:
        """Returns the string representation of the object."""
        return f"{self.__class__.__name__}(config={self.config})"

    def __str__(self) -> str:
        """Returns the string representation of the object."""
        return self.__repr__()

    def __eq__(self, __value: object) -> bool:
        """Returns True if the objects are equal, False otherwise."""
        if not isinstance(__value, BaseSieveBootstrap):
            return NotImplemented
        return self.config == __value.config

    def __hash__(self) -> int:
        """Returns the hash of the object."""
        return hash((super().__hash__(), self.config))
