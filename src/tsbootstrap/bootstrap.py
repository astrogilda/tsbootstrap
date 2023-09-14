from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tsbootstrap.block_bootstrap import BaseBlockBootstrap

if TYPE_CHECKING:
    from tsbootstrap.base_bootstrap_configs import (
        BaseDistributionBootstrapConfig,
        BaseMarkovBootstrapConfig,
        BaseResidualBootstrapConfig,
        BaseSieveBootstrapConfig,
        BaseStatisticPreservingBootstrapConfig,
    )
    from tsbootstrap.block_bootstrap_configs import BaseBlockBootstrapConfig

from tsbootstrap.base_bootstrap import (
    BaseDistributionBootstrap,
    BaseMarkovBootstrap,
    BaseResidualBootstrap,
    BaseSieveBootstrap,
    BaseStatisticPreservingBootstrap,
)
from tsbootstrap.markov_sampler import MarkovSampler
from tsbootstrap.time_series_simulator import TimeSeriesSimulator
from tsbootstrap.utils.odds_and_ends import (
    generate_random_indices,
)

# TODO: add a check if generated block is only one unit long
# TODO: ensure docstrings align with functionality
# TODO: test -- check len(returned_indices) == X.shape[0]
# TODO: ensure x is 2d only for var, otherwise 1d or 2d with 1 feature
# TODO: block_weights=p with block_length=1 should be equivalent to the iid bootstrap
# TODO: add test to fit_ar to ensure input lags, if list, are unique


# Fit, then resample residuals.
class WholeResidualBootstrap(BaseResidualBootstrap):
    """
    Whole Residual Bootstrap class for time series data.

    This class applies residual bootstrapping to the entire time series,
    without any block structure. This is the most basic form of residual
    bootstrapping. The residuals are resampled with replacement and added to
    the fitted values to generate new samples.

    Methods
    -------
    __init__ : Initialize self.
    _generate_samples_single_bootstrap : Generate a single bootstrap sample.
    """

    def __init__(self, config: BaseResidualBootstrapConfig):
        super().__init__(config)

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, exog: np.ndarray | None = None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        self._fit_model(X=X, exog=exog)

        # Resample residuals
        resampled_indices = generate_random_indices(
            self.resids.shape[0], self.config.rng  # type: ignore
        )

        resampled_residuals = self.resids[resampled_indices]  # type: ignore
        # Add the bootstrapped residuals to the fitted values
        bootstrap_samples = self.X_fitted + resampled_residuals
        return [resampled_indices], [bootstrap_samples]


class BlockResidualBootstrap(BaseResidualBootstrap):
    """
    Block Residual Bootstrap class for time series data.

    This class applies residual bootstrapping to blocks of the time series.
    The residuals are bootstrapped using the specified block structure and
    added to the fitted values to generate new samples.

    Methods
    -------
    __init__ : Initialize self.
    _generate_samples_single_bootstrap : Generate a single bootstrap sample.
    """

    def __init__(
        self,
        block_config: BaseBlockBootstrapConfig,
        residual_config: BaseResidualBootstrapConfig,
    ) -> None:
        """
        Initialize self.

        Parameters
        ----------
        residual_config : BaseResidualBootstrapConfig
            The configuration object for the residual bootstrap.
        block_config : BaseBlockBootstrapConfig
            The configuration object for the block bootstrap.
        """
        BaseResidualBootstrap.__init__(self, config=residual_config)
        self.block_bootstrap = BaseBlockBootstrap(config=block_config)

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, exog: np.ndarray | None = None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        # Fit the model and store residuals, fitted values, etc.
        BaseResidualBootstrap._fit_model(self, X=X, exog=exog)

        # Generate blocks of residuals
        (
            block_indices,
            block_data,
        ) = self.block_bootstrap._generate_samples_single_bootstrap(
            X=self.resids  # type: ignore
        )

        # Add the bootstrapped residuals to the fitted values
        bootstrap_samples = self.X_fitted + np.concatenate(block_data, axis=0)
        return block_indices, [bootstrap_samples]


class WholeMarkovBootstrap(BaseMarkovBootstrap):
    """
    Whole Markov Bootstrap class for time series data.

    This class applies Markov bootstrapping to the entire time series,
    without any block structure. This is the most basic form of Markov
    bootstrapping. The residuals are fit to a Markov model, and then
    resampled using the Markov model. The resampled residuals are added to
    the fitted values to generate new samples.

    Methods
    -------
    __init__ : Initialize self.
    _generate_samples_single_bootstrap : Generate a single bootstrap sample.

    Notes
    -----
    Fitting Markov models is expensive, hence we do not allow re-fititng. We instead fit once to the residuals and generate new samples by changing the random_seed.
    """

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, exog: np.ndarray | None = None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        # Fit the model and store residuals, fitted values, etc.
        self._fit_model(X=X, exog=exog)

        # Fit HMM to residuals, just once.
        random_seed = self.config.rng.integers(0, 1000)
        if self.hmm_object is None:
            markov_sampler = MarkovSampler(
                apply_pca_flag=self.config.apply_pca_flag,
                pca=self.config.pca,
                n_iter_hmm=self.config.n_iter_hmm,
                n_fits_hmm=self.config.n_fits_hmm,
                method=self.config.method,  # type: ignore
                blocks_as_hidden_states_flag=self.config.blocks_as_hidden_states_flag,
                random_seed=random_seed,  # type: ignore
            )

            markov_sampler.fit(
                blocks=self.resids, n_states=self.config.n_states  # type: ignore
            )
            self.hmm_object = markov_sampler

        # Resample the fitted values using the HMM.
        bootstrapped_resids = self.hmm_object.sample(
            random_seed=random_seed + self.config.rng.integers(0, 1000)  # type: ignore
        )[0]

        # Add the bootstrapped residuals to the fitted values
        bootstrap_samples = self.X_fitted + bootstrapped_resids

        return [np.arange(X.shape[0])], [bootstrap_samples]


class BlockMarkovBootstrap(BaseMarkovBootstrap):
    """
    Block Markov Bootstrap class for time series data.

    This class applies Markov bootstrapping to blocks of the time series. The
    residuals are fit to a Markov model, then resampled using the specified
    block structure. The resampled residuals are added to the fitted values
    to generate new samples. This class is a combination of the
    `BlockResidualBootstrap` and `WholeMarkovBootstrap` classes.

    Methods
    -------
    __init__ : Initialize self.
    _generate_samples_single_bootstrap : Generate a single bootstrap sample.

    Notes
    -----
    Fitting Markov models is expensive, hence we do not allow re-fititng. We instead fit once to the residuals, resample using blocks once, and generate new samples by changing the random_seed.
    """

    def __init__(
        self,
        markov_config: BaseMarkovBootstrapConfig,
        block_config: BaseBlockBootstrapConfig,
    ) -> None:
        """
        Initialize self.

        Parameters
        ----------
        markov_config : BaseMarkovBootstrapConfig
            The configuration object for the markov bootstrap.
        block_config : BaseBlockBootstrapConfig
            The configuration object for the block bootstrap.
        """
        BaseMarkovBootstrap.__init__(self, config=markov_config)
        self.block_bootstrap = BaseBlockBootstrap(config=block_config)

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, exog: np.ndarray | None = None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        # Fit the model and store residuals, fitted values, etc.
        super()._fit_model(X=X, exog=exog)

        # Generate blocks of residuals
        (
            block_indices,
            block_data,
        ) = self.block_bootstrap._generate_samples_single_bootstrap(
            X=self.resids  # type: ignore
        )

        random_seed = self.config.rng.integers(0, 1000)
        if self.hmm_object is None:
            markov_sampler = MarkovSampler(
                apply_pca_flag=self.config.apply_pca_flag,
                pca=self.config.pca,
                n_iter_hmm=self.config.n_iter_hmm,
                n_fits_hmm=self.config.n_fits_hmm,
                method=self.config.method,  # type: ignore
                blocks_as_hidden_states_flag=self.config.blocks_as_hidden_states_flag,
                random_seed=random_seed,  # type: ignore
            )

            markov_sampler.fit(
                blocks=block_data, n_states=self.config.n_states
            )
            self.hmm_object = markov_sampler

        # Resample the fitted values using the HMM.
        bootstrapped_resids = self.hmm_object.sample(
            random_seed=random_seed + self.config.rng.integers(0, 1000)  # type: ignore
        )[0]

        # Add the bootstrapped residuals to the fitted values
        bootstrap_samples = self.X_fitted + bootstrapped_resids

        return block_indices, [bootstrap_samples]


class WholeStatisticPreservingBootstrap(BaseStatisticPreservingBootstrap):
    """
    Whole Bias Corrected Bootstrap class for time series data.

    This class applies bias corrected bootstrapping to the entire time series,
    without any block structure. This is the most basic form of bias corrected
    bootstrapping. The residuals are resampled with replacement and added to
    the fitted values to generate new samples.

    Attributes
    ----------
    statistic_X : np.ndarray, default=None
        The statistic calculated from the original data. This is used as a parameter for generating the bootstrapped samples.

    Methods
    -------
    __init__ : Initialize self.
    _generate_samples_single_bootstrap : Generate a single bootstrap sample.
    """

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, exog: np.ndarray | None = None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if self.statistic_X is None:
            self.statistic_X = self._calculate_statistic(X=X)

        # Resample residuals
        resampled_indices = generate_random_indices(
            X.shape[0], self.config.rng
        )
        bootstrapped_sample = X[resampled_indices]
        # Calculate the bootstrapped statistic
        statistic_bootstrapped = self._calculate_statistic(bootstrapped_sample)
        # Calculate the bias
        bias = self.statistic_X - statistic_bootstrapped
        # Add the bias to the bootstrapped sample
        bootstrap_sample_bias_corrected = bootstrapped_sample + bias
        return [resampled_indices], [bootstrap_sample_bias_corrected]


class BlockStatisticPreservingBootstrap(BaseStatisticPreservingBootstrap):
    """
    Block Bias Corrected Bootstrap class for time series data.

    This class applies bias corrected bootstrapping to blocks of the time series.
    The residuals are resampled using the specified block structure and added to
    the fitted values to generate new samples.

    Attributes
    ----------
    statistic_X : np.ndarray, default=None
        The statistic calculated from the original data. This is used as a parameter for generating the bootstrapped samples.

    Methods
    -------
    __init__ : Initialize self.
    _generate_samples_single_bootstrap : Generate a single bootstrap sample.
    """

    def __init__(
        self,
        statistic_config: BaseStatisticPreservingBootstrapConfig,
        block_config: BaseBlockBootstrapConfig,
    ) -> None:
        """
        Initialize self.

        Parameters
        ----------
        statistic_config : BaseStatisticPreservingBootstrapConfig
            The configuration object for the bias corrected bootstrap.
        block_config : BaseBlockBootstrapConfig
            The configuration object for the block bootstrap.
        """
        BaseStatisticPreservingBootstrap.__init__(
            self, config=statistic_config
        )
        self.block_bootstrap = BaseBlockBootstrap(config=block_config)

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, exog: np.ndarray | None = None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if self.statistic_X is None:
            self.statistic_X = super()._calculate_statistic(X=X)
        (
            block_indices,
            block_data,
        ) = self.block_bootstrap._generate_samples_single_bootstrap(X=X)

        block_data_concat = np.concatenate(block_data, axis=0)
        # Calculate the bootstrapped statistic
        statistic_bootstrapped = self._calculate_statistic(block_data_concat)
        # Calculate the bias
        bias = self.statistic_X - statistic_bootstrapped
        # Add the bias to the bootstrapped sample
        bootstrap_samples = block_data_concat + bias
        return block_indices, [bootstrap_samples]


class WholeDistributionBootstrap(BaseDistributionBootstrap):
    """
    Whole Distribution Bootstrap class for time series data.

    This class applies distribution bootstrapping to the entire time series,
    without any block structure. This is the most basic form of distribution
    bootstrapping. The residuals are fit to a distribution, and then
    resampled using the distribution. The resampled residuals are added to
    the fitted values to generate new samples.

    Attributes
    ----------
    resids_dist : scipy.stats.rv_continuous or None
        The distribution object used to generate the bootstrapped samples. If None, the distribution has not been fit yet.
    resids_dist_params : tuple or None
        The parameters of the distribution used to generate the bootstrapped samples. If None, the distribution has not been fit yet.

    Methods
    -------
    __init__ : Initialize self.
    _generate_samples_single_bootstrap : Generate a single bootstrap sample.

    Notes
    -----
    We either fit the distribution to the residuals once and generate new samples from the fitted distribution with a new random seed, or resample the residuals once and fit the distribution to the resampled residuals, then generate new samples from the fitted distribution with the same random seed n_bootstrap times.
    """

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, exog: np.ndarray | None = None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        # Fit the model and residuals
        self._fit_model(X=X, exog=exog)
        # Fit the specified distribution to the residuals
        if not self.config.refit:
            if self.resids_dist is None or self.resids_dist_params == ():
                (
                    self.resids_dist,
                    self.resids_dist_params,
                ) = super()._fit_distribution(self.resids)

            # Generate new residuals from the fitted distribution
            bootstrap_residuals = self.resids_dist.rvs(
                *self.resids_dist_params,
                size=X.shape[0],
                random_state=self.config.rng.integers(0, 2**32 - 1),
            ).reshape(-1, 1)

            # Add new residuals to the fitted values to create the bootstrap time series
            bootstrap_samples = self.X_fitted + bootstrap_residuals
            return [np.arange(0, X.shape[0])], [bootstrap_samples]

        else:
            # Resample residuals
            resampled_indices = generate_random_indices(
                self.resids.shape[0], self.config.rng
            )
            resampled_residuals = self.resids[resampled_indices]
            resids_dist, resids_dist_params = super()._fit_distribution(
                resampled_residuals
            )
            # Generate new residuals from the fitted distribution
            bootstrap_residuals = resids_dist.rvs(
                *resids_dist_params,
                size=X.shape[0],
                random_state=self.config.rng,
            ).reshape(-1, 1)

            # Add the bootstrapped residuals to the fitted values
            bootstrap_samples = self.X_fitted + resampled_residuals
            return [resampled_indices], [bootstrap_samples]


class BlockDistributionBootstrap(BaseDistributionBootstrap):
    """
    Block Distribution Bootstrap class for time series data.

    This class applies distribution bootstrapping to blocks of the time series.
    The residuals are fit to a distribution, then resampled using the specified
    block structure. Then new residuals are generated from the fitted
    distribution and added to the fitted values to generate new samples.

    Attributes
    ----------
    resids_dist : scipy.stats.rv_continuous or None
        The distribution object used to generate the bootstrapped samples. If None, the distribution has not been fit yet.
    resids_dist_params : tuple or None
        The parameters of the distribution used to generate the bootstrapped samples. If None, the distribution has not been fit yet.

    Methods
    -------
    __init__ : Initialize self.
    _generate_samples_single_bootstrap : Generate a single bootstrap sample.

    Notes
    -----
    We either fit the distribution to the residuals once and generate new samples from the fitted distribution with a new random seed, or resample the residuals once and fit the distribution to the resampled residuals, then generate new samples from the fitted distribution with the same random seed n_bootstrap times.
    """

    def __init__(
        self,
        distribution_config: BaseDistributionBootstrapConfig,
        block_config: BaseBlockBootstrapConfig,
    ) -> None:
        """
        Initialize self.

        Parameters
        ----------
        distribution_config : BaseDistributionBootstrapConfig
            The configuration object for the distribution bootstrap.
        block_config : BaseBlockBootstrapConfig
            The configuration object for the block bootstrap.
        """
        BaseDistributionBootstrap.__init__(self, config=distribution_config)
        self.block_bootstrap = BaseBlockBootstrap(config=block_config)

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, exog: np.ndarray | None = None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        # Fit the model and residuals
        super()._fit_model(X=X, exog=exog)
        (
            block_indices,
            block_data,
        ) = self.block_bootstrap._generate_samples_single_bootstrap(
            X=self.resids
        )
        block_data_concat = np.concatenate(block_data, axis=0)
        # Fit the specified distribution to the residuals
        if not self.config.refit:
            if self.resids_dist is None or self.resids_dist_params == ():
                (
                    self.resids_dist,
                    self.resids_dist_params,
                ) = super()._fit_distribution(block_data_concat)

            # Generate new residuals from the fitted distribution
            bootstrap_residuals = self.resids_dist.rvs(
                *self.resids_dist_params,
                size=block_data_concat.shape[0],
                random_state=self.config.rng.integers(0, 2**32 - 1),
            ).reshape(-1, 1)

            # Add new residuals to the fitted values to create the bootstrap time series
            bootstrap_samples = self.X_fitted + bootstrap_residuals
            return [np.arange(0, block_data_concat.shape[0])], [
                bootstrap_samples
            ]

        else:
            # Resample residuals
            resids_dist, resids_dist_params = super()._fit_distribution(
                block_data_concat
            )
            # Generate new residuals from the fitted distribution
            bootstrap_residuals = resids_dist.rvs(
                *resids_dist_params,
                size=block_data_concat.shape[0],
                random_state=self.config.rng,
            ).reshape(-1, 1)

            # Add the bootstrapped residuals to the fitted values
            bootstrap_samples = self.X_fitted + bootstrap_residuals
            return block_indices, [bootstrap_samples]


class WholeSieveBootstrap(BaseSieveBootstrap):
    """
    Implementation of the Sieve bootstrap method for time series data.

    This class applies Sieve bootstrapping to the entire time series,
    without any block structure. This is the most basic form of Sieve
    bootstrapping. The residuals are fit to a second model, and then new
    samples are generated by adding the new residuals to the fitted values.

    Methods
    -------
    _generate_samples_single_bootstrap : Generate a single bootstrapped sample.
    """

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, exog: np.ndarray | None = None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        self._fit_model(X=X, exog=exog)
        self._fit_resids_model(X=self.resids)

        ts_simulator = TimeSeriesSimulator(
            X_fitted=self.X_fitted,
            rng=self.config.rng,
            fitted_model=self.resids_fit_model,
        )

        simulated_samples = ts_simulator.generate_samples_sieve(
            model_type=self.config.resids_model_type,
            resids_lags=self.resids_order,
            resids_coefs=self.resids_coefs,
            resids=self.resids,
        )

        return [np.arange(X.shape[0])], [simulated_samples]


class BlockSieveBootstrap(BaseSieveBootstrap):
    """
    Implementation of the Sieve bootstrap method for time series data.

    This class applies Sieve bootstrapping to blocks of the time series.
    The residuals are fit to a second model, then resampled using the
    specified block structure. The new residuals are then added to the
    fitted values to generate new samples.

    Methods
    -------
    _init_ : Initialize self.
    _generate_samples_single_bootstrap : Generate a single bootstrapped sample.
    """

    def __init__(
        self,
        sieve_config: BaseSieveBootstrapConfig,
        block_config: BaseBlockBootstrapConfig,
    ) -> None:
        """
        Initialize self.

        Parameters
        ----------
        sieve_config : BaseSieveBootstrapConfig
            The configuration object for the sieve bootstrap.
        block_config : BaseBlockBootstrapConfig
            The configuration object for the block bootstrap.
        """
        BaseSieveBootstrap.__init__(self, config=sieve_config)
        self.block_bootstrap = BaseBlockBootstrap(config=block_config)

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, exog: np.ndarray | None = None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        # Fit the model and residuals
        super()._fit_model(X=X, exog=exog)
        super()._fit_resids_model(X=self.resids)

        ts_simulator = TimeSeriesSimulator(
            X_fitted=self.X_fitted,
            rng=self.config.rng,
            fitted_model=self.resids_fit_model,
        )

        simulated_samples = ts_simulator.generate_samples_sieve(
            model_type=self.config.resids_model_type,
            resids_lags=self.resids_order,
            resids_coefs=self.resids_coefs,
            resids=self.resids,
        )

        resids_resids = self.X_fitted - simulated_samples
        (
            block_indices,
            resids_resids_resampled,
        ) = self.block_bootstrap._generate_samples_single_bootstrap(
            X=resids_resids
        )
        resids_resids_resampled_concat = np.concatenate(
            resids_resids_resampled, axis=0
        )

        bootstrapped_samples = self.X_fitted + resids_resids_resampled_concat

        return block_indices, [bootstrapped_samples]
