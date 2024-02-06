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
    )

from sklearn.decomposition import PCA  # type: ignore

from tsbootstrap.tsfit import TSFitBestLag
from tsbootstrap.utils.types import (
    BlockCompressorTypes,
    ModelTypes,
    ModelTypesWithoutArch,
    OrderTypes,
    RngTypes,
)
from tsbootstrap.utils.validate import (
    validate_literal_type,
    validate_order,
    validate_rng,
    validate_single_integer,
)


class BaseTimeSeriesBootstrap(BaseObject):
    """
    Base class for time series bootstrapping.

    Raises
    ------
    ValueError
        If n_bootstraps is not greater than 0.
    """

    _tags = {
        "object_type": "bootstrap",
        "bootstrap_type": "other",
        "api_rework_complete": False,
    }

    def __init__(
        self,
        n_bootstraps: Integral = 10,  # type: ignore
        rng=None,
    ) -> None:
        """
        Initialize self.

        Parameters
        ----------
        n_bootstraps : Integral, default=10
            The number of bootstrap samples to create.
        rng : Integral or np.random.Generator, default=np.random.default_rng()
            The random number generator or seed used to generate the bootstrap samples.
        """
        self.n_bootstraps = n_bootstraps
        self.rng = rng

    @property
    def rng(self) -> np.random.Generator:
        """Getter for rng."""
        return self._rng

    @rng.setter
    def rng(self, value: RngTypes) -> None:
        """Setter for rng. Performs validation on assignment."""
        self._rng = validate_rng(value)

    @property
    def n_bootstraps(self) -> Integral:
        """Getter for n_bootstraps."""
        return self._n_bootstraps

    @n_bootstraps.setter
    def n_bootstraps(self, value) -> None:
        """Setter for n_bootstraps. Performs validation on assignment."""
        validate_single_integer(value, min_value=1)  # type: ignore
        self._n_bootstraps = value

    # TODO 0.1.0: change default value of test_ratio to 0.0
    def bootstrap(
        self,
        X: np.ndarray,
        return_indices: bool = False,
        y=None,
        test_ratio: float = None,
    ):
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
        y : array-like of shape (n_timepoints, n_features_exog), default=None
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

        if y is not None:
            self._check_input(y)
            exog_train, _ = time_series_split(y, test_ratio=test_ratio)
        else:
            exog_train = None

        tuple_iter = self._generate_samples(
            X=X_train, return_indices=return_indices, y=exog_train
        )

        yield from tuple_iter

    def _generate_samples(
        self,
        X: np.ndarray,
        return_indices: bool = False,
        y=None,
    ):
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
        for _ in range(self.n_bootstraps):
            indices, data = self._generate_samples_single_bootstrap(X=X, y=y)
            data = np.concatenate(data, axis=0)
            if return_indices:
                yield indices, data  # type: ignore
            else:
                yield data

    def _generate_samples_single_bootstrap(self, X: np.ndarray, y=None):
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
        X=None,
        y=None,
        groups=None,
    ) -> Integral:
        """Returns the number of bootstrapping iterations."""
        return self.n_bootstraps  # type: ignore


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
        n_bootstraps: Integral = 10,  # type: ignore
        rng=None,
        model_type: ModelTypesWithoutArch = "ar",
        order: OrderTypes = None,
        save_models: bool = False,
        **kwargs,
    ):
        """
        Initialize self.

        Parameters
        ----------
        n_bootstraps : Integral, default=10
            The number of bootstrap samples to create.
        model_type : str, default="ar"
            The model type to use. Must be one of "ar", "arima", "sarima", "var", or "arch".
        order : Integral or list or tuple, default=None
            The order of the model. If None, the best order is chosen via TSFitBestLag. If Integral, it is the lag order for AR, ARIMA, and SARIMA, and the lag order for ARCH. If list or tuple, the order is a tuple of (p, o, q) for ARIMA and (p, d, q, s) for SARIMAX. It is either a single Integral or a list of non-consecutive ints for AR, and an Integral for VAR and ARCH. If None, the best order is chosen via TSFitBestLag. Do note that TSFitBestLag only chooses the best lag, not the best order, so for the tuple values, it only chooses the best p, not the best (p, o, q) or (p, d, q, s). The rest of the values are set to 0.
        save_models : bool, default=False
            Whether to save the fitted models.
        rng : Integral or np.random.Generator, default=np.random.default_rng()
            The random number generator or seed used to generate the bootstrap samples.
        **kwargs
            Additional keyword arguments to pass to the TSFit model.

        Raises
        ------
        ValueError
            If model_type is not one of "ar", "arima", "sarima", "var", or "arch".

        Notes
        -----
        The model_type and order parameters are passed to TSFitBestLag, which
        chooses the best lag and order for the model. The best lag and order are
        then used to fit the model to the data. The residuals are then stored
        for use in the bootstrap.

        References
        ----------
        .. [^1^] https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Residual_bootstrap
        """
        super().__init__(n_bootstraps=n_bootstraps, rng=rng)

        self._model_type = model_type
        self.model_type = model_type
        self.order = order
        self.save_models = save_models
        self.model_params = kwargs

        self.fit_model = None
        self.resids = None
        self.X_fitted = None
        self.coefs = None

    @property
    def model_type(self) -> str:
        """Getter for model_type."""
        return self._model_type

    @model_type.setter
    def model_type(self, value: str) -> None:
        """Setter for model_type. Performs validation on assignment."""
        value = value.lower()
        validate_literal_type(value, ModelTypesWithoutArch)  # type: ignore
        self._model_type = value

    @property
    def order(self) -> OrderTypes:
        """Getter for order."""
        return self._order

    @order.setter
    def order(self, value) -> None:
        """Setter for order. Performs validation on assignment."""
        validate_order(value)
        self._order = value

    @property
    def save_models(self) -> bool:
        """Getter for save_models."""
        return self._save_models

    @save_models.setter
    def save_models(self, value: bool) -> None:
        """Setter for save_models. Performs validation on assignment."""
        if not isinstance(value, bool):
            raise TypeError("save_models must be a boolean.")
        self._save_models = value

    def _fit_model(self, X: np.ndarray, y=None) -> None:
        """Fits the model to the data and stores the residuals."""
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
        n_bootstraps: Integral = 10,  # type: ignore
        rng=None,
        method: BlockCompressorTypes = "middle",
        apply_pca_flag: bool = False,
        pca=None,
        n_iter_hmm: Integral = 10,  # type: ignore
        n_fits_hmm: Integral = 1,  # type: ignore
        blocks_as_hidden_states_flag: bool = False,
        n_states: Integral = 2,  # type: ignore
        **kwargs,
    ):
        """
        Initialize self.

        Parameters
        ----------
        n_bootstraps : Integral, default=10
            The number of bootstrap samples to create.
        rng : Integral or np.random.Generator, default=np.random.default_rng()
            The random number generator or seed used to generate the bootstrap samples.
        method : str, default="middle"
            The method to use for compressing the blocks. Must be one of "first", "middle", "last", "mean", "mode", "median", "kmeans", "kmedians", "kmedoids".
        apply_pca_flag : bool, default=False
            Whether to apply PCA to the residuals before fitting the HMM.
        pca : PCA, default=None
            The PCA object to use for applying PCA to the residuals.
        n_iter_hmm : Integral, default=10
            Number of iterations for fitting the HMM.
        n_fits_hmm : Integral, default=1
            Number of times to fit the HMM.
        blocks_as_hidden_states_flag : bool, default=False
            Whether to use blocks as hidden states.
        n_states : Integral, default=2
            Number of states for the HMM.
        **kwargs
            Additional keyword arguments to pass to the BaseResidualBootstrapConfig class,
            except for n_bootstraps and rng, which are passed directly to the parent BaseTimeSeriesBootstrapConfig class.
            See the documentation for BaseResidualBootstrapConfig for more information.
        """
        super().__init__(n_bootstraps=n_bootstraps, rng=rng, **kwargs)

        self.method = method
        self.apply_pca_flag = apply_pca_flag
        self.pca = pca
        self.n_iter_hmm = n_iter_hmm
        self.n_fits_hmm = n_fits_hmm
        self.blocks_as_hidden_states_flag = blocks_as_hidden_states_flag
        self.n_states = n_states

        self.hmm_object = None

    @property
    def method(self) -> str:
        """Getter for method."""
        return self._method

    @method.setter
    def method(self, value: BlockCompressorTypes) -> None:
        """Setter for method. Performs validation on assignment."""
        validate_literal_type(value, BlockCompressorTypes)  # type: ignore
        self._method = value.lower()

    @property
    def apply_pca_flag(self) -> bool:
        """Getter for apply_pca_flag."""
        return self._apply_pca_flag

    @apply_pca_flag.setter
    def apply_pca_flag(self, value: bool) -> None:
        """Setter for apply_pca_flag. Performs validation on assignment."""
        if not isinstance(value, bool):
            raise TypeError("apply_pca_flag must be a boolean.")
        self._apply_pca_flag = value

    @property
    def pca(self):
        """Getter for pca."""
        return self._pca

    @pca.setter
    def pca(self, value) -> None:
        """Setter for pca. Performs validation on assignment."""
        if value is not None and not isinstance(value, PCA):
            raise TypeError("pca must be an instance of PCA.")
        self._pca = value

    @property
    def n_iter_hmm(self) -> Integral:
        """Getter for n_iter_hmm."""
        return self._n_iter_hmm

    @n_iter_hmm.setter
    def n_iter_hmm(self, value: Integral) -> None:
        """Setter for n_iter_hmm. Performs validation on assignment."""
        validate_single_integer(value, min_value=10)  # type: ignore
        self._n_iter_hmm = value

    @property
    def n_fits_hmm(self) -> Integral:
        """Getter for n_fits_hmm."""
        return self._n_fits_hmm

    @n_fits_hmm.setter
    def n_fits_hmm(self, value: Integral) -> None:
        """Setter for n_fits_hmm. Performs validation on assignment."""
        validate_single_integer(value, min_value=1)  # type: ignore
        self._n_fits_hmm = value

    @property
    def blocks_as_hidden_states_flag(self) -> bool:
        """Getter for blocks_as_hidden_states_flag."""
        return self._blocks_as_hidden_states_flag

    @blocks_as_hidden_states_flag.setter
    def blocks_as_hidden_states_flag(self, value: bool) -> None:
        """Setter for blocks_as_hidden_states_flag. Performs validation on assignment."""
        if not isinstance(value, bool):
            raise TypeError("blocks_as_hidden_states_flag must be a boolean.")
        self._blocks_as_hidden_states_flag = value

    @property
    def n_states(self) -> Integral:
        """Getter for n_states."""
        return self._n_states

    @n_states.setter
    def n_states(self, value: Integral) -> None:
        """Setter for n_states. Performs validation on assignment."""
        validate_single_integer(value, min_value=2)  # type: ignore
        self._n_states = value


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

    def _fit_distribution(self, resids: np.ndarray):
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
            resids_fit_model = resids_fit_obj.fit(X, y=None).model
            resids_order = resids_fit_obj.get_order()
            resids_coefs = resids_fit_obj.get_coefs()
            self.resids_fit_model = resids_fit_model
            self.resids_order = resids_order
            self.resids_coefs = resids_coefs
