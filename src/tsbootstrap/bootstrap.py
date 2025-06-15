from __future__ import annotations

import logging  # Added logging
from typing import Any, Callable, Optional, Union  # Added Any, Callable, Union

import numpy as np
from pydantic import Field  # Added Field
from sklearn.decomposition import PCA  # Added PCA

from tsbootstrap.base_bootstrap import (
    BaseDistributionBootstrap,
    BaseMarkovBootstrap,
    BaseResidualBootstrap,
    BaseSieveBootstrap,
    BaseStatisticPreservingBootstrap,
)
from tsbootstrap.block_bootstrap import (
    BlockBootstrap,
    MovingBlockBootstrap,
)
from tsbootstrap.markov_sampler import MarkovSampler
from tsbootstrap.time_series_simulator import TimeSeriesSimulator
from tsbootstrap.utils.odds_and_ends import generate_random_indices
from tsbootstrap.utils.types import (
    BlockCompressorTypes,
    DistributionTypes,  # Added DistributionTypes
    ModelTypes,
    ModelTypesWithoutArch,
    OrderTypes,
    RngTypes,
)

# TODO: add a check if generated block is only one unit long
# TODO: ensure docstrings align with functionality
# TODO: test -- check len(returned_indices) == X.shape[0]
# TODO: ensure x is 2d only for var, otherwise 1d or 2d with 1 feature
# TODO: block_weights=p with block_length=1 should be equivalent to the iid bootstrap
# TODO: add test to fit_ar to ensure input lags, if list, are unique
# TODO: for `StatisticPreservingBootstrap`, see if the statistic on the bootstrapped
# sample is close to the statistic on the original sample
# TODO: in `DistributionBootstrap`, allow mixture of distributions


# Fit, then resample residuals.
class WholeResidualBootstrap(BaseResidualBootstrap):
    """
    Whole Residual Bootstrap class for time series data.

    This class applies residual bootstrapping to the entire time series,
    without any block structure. This is the most basic form of residual
    bootstrapping. The residuals are resampled with replacement and added to
    the fitted values to generate new samples.

    Parameters
    ----------
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    model_type : str, default="ar"
        The model type to use. Must be one of "ar", "arima", "sarima", "var", or "arch".
    model_params : dict, default=None
        Additional keyword arguments to pass to the TSFit model.
    order : OrderTypes, default=None
        The order of the model. If None, the best order is chosen via TSFitBestLag.
        If Integral, it is the lag order for AR, ARIMA, and SARIMA,
        and the lag order for ARCH. If list or tuple, the order is a
        tuple of (p, o, q) for ARIMA and (p, d, q, s) for SARIMAX.
        It is either a single Integral or a list of non-consecutive ints for AR,
        and an Integral for VAR and ARCH. If None, the best order is chosen via
        TSFitBestLag. Do note that TSFitBestLag only chooses the best lag,
        not the best order, so for the tuple values, it only chooses the best p,
        not the best (p, o, q) or (p, d, q, s). The rest of the values are set to 0.
    save_models : bool, default=False
        Whether to save the fitted models.
    rng : RngTypes, default=None
        The random number generator or seed used to generate the bootstrap samples.

    Methods
    -------
    __init__ : Initialize self.
    _generate_samples_single_bootstrap : Generate a single bootstrap sample.
    """

    def __init__(
        self,
        n_bootstraps: int = 10,
        rng: Optional[np.random.Generator] = None,
        model_type: ModelTypesWithoutArch = "ar",
        model_params: Optional[dict[str, Any]] = None,  # Align with base
        order: Optional[Union[int, tuple, list]] = None,  # Align with base
        save_models: bool = False,
    ):
        self._model_type = model_type  # This seems to be a local attribute, not passed to super for BaseResidualBootstrap
        # BaseResidualBootstrap defines its own model_type field.

        super().__init__(  # Calls BaseResidualBootstrap.__init__
            n_bootstraps=n_bootstraps,
            rng=rng,
            model_type=model_type,  # This should align with BaseResidualBootstrap.model_type
            model_params=(
                model_params if model_params is not None else {}
            ),  # Ensure dict, not None
            order=order,
            save_models=save_models,
        )

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y=None, n: Optional[int] = None
    ):
        self._fit_model(X=X, y=y)

        # Resample residuals
        resampled_indices = generate_random_indices(
            X.shape[0] if n is None else n, self.rng  # type: ignore
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

    Parameters
    ----------
    block_bootstrap : BlockBootstrap, default=MovingBlockBootstrap()
        The block bootstrap algorithm.
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    model_type : str, default="ar"
        The model type to use. Must be one of "ar", "arima", "sarima", "var", or "arch".
    model_params : dict, default=None
        Additional keyword arguments to pass to the TSFit model.
    order : OrderTypes, default=None
        The order of the model. If None, the best order is chosen via TSFitBestLag.
        If Integral, it is the lag order for AR, ARIMA, and SARIMA,
        and the lag order for ARCH. If list or tuple, the order is a
        tuple of (p, o, q) for ARIMA and (p, d, q, s) for SARIMAX.
        It is either a single Integral or a list of non-consecutive ints for AR,
        and an Integral for VAR and ARCH. If None, the best order is chosen via
        TSFitBestLag. Do note that TSFitBestLag only chooses the best lag,
        not the best order, so for the tuple values, it only chooses the best p,
        not the best (p, o, q) or (p, d, q, s). The rest of the values are set to 0.
    save_models : bool, default=False
        Whether to save the fitted models.
    rng : RngTypes, default=None
        The random number generator or seed used to generate the bootstrap samples.

    Methods
    -------
    __init__ : Initialize self.
    _generate_samples_single_bootstrap : Generate a single bootstrap sample.
    """

    def __init__(
        self,
        n_bootstraps: int = 10,
        block_bootstrap: Optional[BlockBootstrap] = None,
        model_type: ModelTypesWithoutArch = "ar",
        model_params: Optional[dict[str, Any]] = None,  # Align with base
        order: Optional[Union[int, tuple, list]] = None,  # Align with base
        save_models: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(  # Calls BaseResidualBootstrap.__init__
            n_bootstraps=n_bootstraps,
            rng=rng,
            model_type=model_type,
            model_params=(
                model_params if model_params is not None else {}
            ),  # Ensure dict, not None
            order=order,
            save_models=save_models,
        )
        if block_bootstrap is None:
            self.block_bootstrap = MovingBlockBootstrap()
        else:
            self.block_bootstrap = block_bootstrap

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y=None, n: Optional[int] = None
    ):
        # Fit the model and store residuals, fitted values, etc.
        BaseResidualBootstrap._fit_model(self, X=X, y=y)

        # Generate blocks of residuals
        if self.block_bootstrap is None:
            raise ValueError("block_bootstrap must be initialized.")
        (
            block_indices,
            block_data,
        ) = self.block_bootstrap._generate_samples_single_bootstrap(
            X=self.resids, n=X.shape[0] if n is None else n  # type: ignore
        )

        # Add the bootstrapped residuals to the fitted values
        bootstrap_samples = np.concatenate(block_data, axis=0)
        return block_indices, [bootstrap_samples]

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from tsbootstrap.block_bootstrap import MovingBlockBootstrap

        bs = MovingBlockBootstrap()
        return {"block_bootstrap": bs}


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
    _generate_samples_single_bootstrap : Generate a single bootstrap sample.

    Notes
    -----
    Fitting Markov models is expensive, hence we do not allow re-fititng. We instead fit once to the residuals and generate new samples by changing the random_seed.
    """

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y=None, n: Optional[int] = None
    ):
        # Fit the model and store residuals, fitted values, etc.
        self._fit_model(X=X, y=y)

        if self.X_fitted is None:
            raise RuntimeError(
                f"{self.__class__.__name__}._fit_model did not set self.X_fitted."
            )
        if (
            self.resids is None
        ):  # resids are needed to fit the HMM if it's not already fit
            raise RuntimeError(
                f"{self.__class__.__name__}._fit_model did not set self.resids."
            )

        # Fit HMM to residuals, just once.
        random_seed = self.rng.integers(0, 2**32 - 1)  # type: ignore
        if self.hmm_object is None:
            markov_sampler = MarkovSampler(
                apply_pca_flag=self.apply_pca_flag,  # type: ignore
                pca=self.pca,  # type: ignore
                n_iter_hmm=self.n_iter_hmm,  # type: ignore
                n_fits_hmm=self.n_fits_hmm,  # type: ignore
                method=self.method,  # type: ignore
                blocks_as_hidden_states_flag=self.blocks_as_hidden_states_flag,  # type: ignore
                random_seed=random_seed,  # type: ignore[arg-type]
            )

            markov_sampler.fit(
                blocks=self.resids, n_states=self.n_states  # type: ignore
            )
            self.hmm_object = markov_sampler

        # Resample the residuals using the HMM.
        # target_output_length is the desired length for the final bootstrapped sample.
        target_output_length = X.shape[0] if n is None else n
        new_sample_seed = self.rng.integers(0, 2**32 - 1)  # type: ignore

        # Call HMM sample method to get bootstrapped_resids of target_output_length
        _sample_output = self.hmm_object.sample(
            n_to_sample=target_output_length,
            random_seed=new_sample_seed,  # type: ignore
        )
        bootstrapped_resids = _sample_output[0]

        # Ensure HMM produced the correct number of residuals
        if bootstrapped_resids.shape[0] != target_output_length:
            # This would indicate an issue with the HMM's sample method
            raise RuntimeError(  # Changed to RuntimeError as it's an unexpected internal state
                f"HMM sampling failed in {self.__class__.__name__}: "
                f"Expected {target_output_length} residuals, but got {bootstrapped_resids.shape[0]}."
            )

        # Prepare the X_fitted component for addition. It must match target_output_length.
        X_fitted_component = self.X_fitted
        if self.X_fitted.shape[0] > target_output_length:
            X_fitted_component = self.X_fitted[:target_output_length]
        elif self.X_fitted.shape[0] < target_output_length:
            # This means we're asked to generate a longer series than the model was fit on (X_fitted is too short).
            raise ValueError(
                f"Cannot generate bootstrap sample of length {target_output_length} in {self.__class__.__name__}: "
                f"Fitted values (X_fitted) only have length {self.X_fitted.shape[0]}. "
                f"This usually means the 'n' parameter ({n}) is larger than the length of the original series "
                f"used for model fitting ({X.shape[0]}), or model fitting significantly shortened the series."
            )

        # Debug print (updated)
        # Ensure the type ignore is correctly placed if self.X_fitted or bootstrapped_resids can be None by type hints elsewhere, though logic implies they are ndarray here.
        print(f"[DEBUG {self.__class__.__name__}] X_fitted_component shape: {X_fitted_component.shape}, Bootstrapped_resids shape: {bootstrapped_resids.shape}, Target output length: {target_output_length}")  # type: ignore

        # Add the bootstrapped residuals to the (potentially truncated) fitted values
        # At this point, X_fitted_component.shape[0] == bootstrapped_resids.shape[0] == target_output_length
        bootstrap_samples = X_fitted_component + bootstrapped_resids

        return [np.arange(int(X.shape[0]))], [bootstrap_samples]


class BlockMarkovBootstrap(BaseMarkovBootstrap):
    block_bootstrap: Optional[BlockBootstrap] = Field(
        default_factory=MovingBlockBootstrap,
        description="The block bootstrap algorithm.",
    )
    """
    Block Markov Bootstrap class for time series data.

    This class applies Markov bootstrapping to blocks of the time series. The
    residuals are fit to a Markov model, then resampled using the specified
    block structure. The resampled residuals are added to the fitted values
    to generate new samples. This class is a combination of the
    `BlockResidualBootstrap` and `WholeMarkovBootstrap` classes.

    Parameters
    ----------
    block_bootstrap : BlockBootstrap, default=MovingBlockBootstrap()
        The block bootstrap algorithm.
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    method : str, default="middle"
        The method to use for compressing the blocks.
        Must be one of "first", "middle", "last", "mean", "mode", "median",
        "kmeans", "kmedians", "kmedoids".
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
    model_type : str, default="ar"
        The model type to use. Must be one of "ar", "arima", "sarima", "var", or "arch".
    model_params : dict, default=None
        Additional keyword arguments to pass to the TSFit model.
    order : Integral or list or tuple, default=None
        The order of the model. If None, the best order is chosen via TSFitBestLag.
        If Integral, it is the lag order for AR, ARIMA, and SARIMA, and the lag order
        for ARCH. If list or tuple, the order is a tuple of (p, o, q) for ARIMA
        and (p, d, q, s) for SARIMAX. It is either a single Integral or a
        list of non-consecutive ints for AR, and an Integral for VAR and ARCH.
        If None, the best order is chosen via TSFitBestLag. Do note that TSFitBestLag
        only chooses the best lag, not the best order, so for the tuple values,
        it only chooses the best p, not the best (p, o, q) or (p, d, q, s).
        The rest of the values are set to 0.
    save_models : bool, default=False
        Whether to save the fitted models.
    rng : Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

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
        n_bootstraps: int = 10,
        block_bootstrap: Optional[BlockBootstrap] = None,
        method: BlockCompressorTypes = "middle",
        apply_pca_flag: bool = False,
        pca: Optional[PCA] = None,  # Align with BaseMarkovBootstrap.pca
        n_iter_hmm: int = 10,  # Align with BaseMarkovBootstrap.n_iter_hmm
        n_fits_hmm: int = 1,  # Align with BaseMarkovBootstrap.n_fits_hmm
        blocks_as_hidden_states_flag: bool = False,
        n_states: int = 2,  # Align with BaseMarkovBootstrap.n_states
        model_type: ModelTypesWithoutArch = "ar",
        model_params: Optional[dict[str, Any]] = None,  # Align with base
        order: Optional[Union[int, tuple, list]] = None,  # Align with base
        save_models: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(  # Calls BaseMarkovBootstrap.__init__
            n_bootstraps=n_bootstraps,
            method=method,
            apply_pca_flag=apply_pca_flag,
            pca=pca,
            n_iter_hmm=n_iter_hmm,
            n_fits_hmm=n_fits_hmm,
            blocks_as_hidden_states_flag=blocks_as_hidden_states_flag,
            n_states=n_states,
            model_type=model_type,
            model_params=(
                model_params if model_params is not None else {}
            ),  # Ensure dict
            order=order,
            save_models=save_models,
            rng=rng,
        )
        if block_bootstrap is None:
            self.block_bootstrap = MovingBlockBootstrap()
        else:
            self.block_bootstrap = block_bootstrap

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y=None, n: Optional[int] = None
    ):
        # Fit the model and store residuals, fitted values, etc.
        super()._fit_model(X=X, y=y)

        if self.X_fitted is None:
            raise RuntimeError(
                f"{self.__class__.__name__}._fit_model did not set self.X_fitted."
            )
        if self.resids is None:  # resids are needed for block generation
            raise RuntimeError(
                f"{self.__class__.__name__}._fit_model did not set self.resids."
            )

        # Generate blocks of residuals
        if self.block_bootstrap is None:
            raise ValueError("block_bootstrap must be initialized.")
        (
            block_indices,
            block_data,
        ) = self.block_bootstrap._generate_samples_single_bootstrap(
            X=self.resids, n=X.shape[0] if n is None else n  # type: ignore
        )

        random_seed = self.rng.integers(0, 2**32 - 1)  # type: ignore
        if self.hmm_object is None:
            markov_sampler = MarkovSampler(
                apply_pca_flag=self.apply_pca_flag,  # type: ignore
                pca=self.pca,  # type: ignore
                n_iter_hmm=self.n_iter_hmm,  # type: ignore
                n_fits_hmm=self.n_fits_hmm,  # type: ignore
                method=self.method,  # type: ignore
                blocks_as_hidden_states_flag=self.blocks_as_hidden_states_flag,  # type: ignore
                random_seed=random_seed,  # type: ignore[arg-type]
            )

            markov_sampler.fit(
                blocks=block_data, n_states=self.n_states  # type: ignore
            )
            self.hmm_object = markov_sampler

        # Resample the residuals using the HMM.
        # target_output_length is the desired length for the final bootstrapped sample.
        target_output_length = X.shape[0] if n is None else n
        new_sample_seed = self.rng.integers(0, 2**32 - 1)  # type: ignore

        # Call HMM sample method to get bootstrapped_resids of target_output_length
        _sample_output = self.hmm_object.sample(
            n_to_sample=target_output_length,
            random_seed=new_sample_seed,  # type: ignore
        )
        bootstrapped_resids = _sample_output[0]

        # Ensure HMM produced the correct number of residuals
        if bootstrapped_resids.shape[0] != target_output_length:
            # This would indicate an issue with the HMM's sample method
            raise RuntimeError(  # Changed to RuntimeError as it's an unexpected internal state
                f"HMM sampling failed in {self.__class__.__name__}: "
                f"Expected {target_output_length} residuals, but got {bootstrapped_resids.shape[0]}."
            )

        # Prepare the X_fitted component for addition. It must match target_output_length.
        X_fitted_component = self.X_fitted
        if self.X_fitted.shape[0] > target_output_length:
            X_fitted_component = self.X_fitted[:target_output_length]
        elif self.X_fitted.shape[0] < target_output_length:
            # This means we're asked to generate a longer series than the model was fit on (X_fitted is too short).
            raise ValueError(
                f"Cannot generate bootstrap sample of length {target_output_length} in {self.__class__.__name__}: "
                f"Fitted values (X_fitted) only have length {self.X_fitted.shape[0]}. "
                f"This usually means the 'n' parameter ({n}) is larger than the length of the original series "
                f"used for model fitting ({X.shape[0]}), or model fitting significantly shortened the series."
            )

        # Debug print (updated)
        # Ensure the type ignore is correctly placed if self.X_fitted or bootstrapped_resids can be None by type hints elsewhere, though logic implies they are ndarray here.
        print(f"[DEBUG {self.__class__.__name__}] X_fitted_component shape: {X_fitted_component.shape}, Bootstrapped_resids shape: {bootstrapped_resids.shape}, Target output length: {target_output_length}")  # type: ignore

        # Add the bootstrapped residuals to the (potentially truncated) fitted values
        # At this point, X_fitted_component.shape[0] == bootstrapped_resids.shape[0] == target_output_length
        bootstrap_samples = X_fitted_component + bootstrapped_resids

        return block_indices, [bootstrap_samples]

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from tsbootstrap.block_bootstrap import MovingBlockBootstrap

        bs = MovingBlockBootstrap()
        return {"block_bootstrap": bs}


class WholeStatisticPreservingBootstrap(BaseStatisticPreservingBootstrap):
    """
    Whole Statistic Preserving Bootstrap class for time series data.

    This class applies statistic-preserving bootstrapping to the entire time series,
    without any block structure. This is the most basic form of statistic-preserving
    bootstrapping. The residuals are resampled with replacement and added to
    the fitted values to generate new samples.

    Methods
    -------
    _generate_samples_single_bootstrap : Generate a single bootstrap sample.
    """

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y=None, n: Optional[int] = None
    ):
        current_statistic_to_match = self._calculate_statistic(X=X)

        # Resample residuals
        resampled_indices = generate_random_indices(
            X.shape[0] if n is None else n, self.rng  # type: ignore
        )
        bootstrapped_sample = X[resampled_indices]
        # Calculate the bootstrapped statistic
        statistic_bootstrapped = self._calculate_statistic(bootstrapped_sample)
        # Ensure statistic_bootstrapped has the same number of dimensions as bootstrapped_sample for broadcasting
        if statistic_bootstrapped.ndim == 1 and bootstrapped_sample.ndim == 2:
            if self.statistic_axis == 0:
                statistic_bootstrapped = np.expand_dims(
                    statistic_bootstrapped, axis=0
                )
            elif self.statistic_axis == 1:
                statistic_bootstrapped = np.expand_dims(
                    statistic_bootstrapped, axis=1
                )

        # Ensure current_statistic_to_match has the same number of dimensions as X for broadcasting
        if current_statistic_to_match.ndim == 1 and X.ndim == 2:
            if self.statistic_axis == 0:
                current_statistic_to_match = np.expand_dims(
                    current_statistic_to_match, axis=0
                )
            elif self.statistic_axis == 1:
                current_statistic_to_match = np.expand_dims(
                    current_statistic_to_match, axis=1
                )

        # Calculate the bias
        bias = current_statistic_to_match - statistic_bootstrapped
        # Add the bias to the bootstrapped sample
        bootstrap_sample_bias_corrected = bootstrapped_sample + bias
        return [resampled_indices], [bootstrap_sample_bias_corrected]


class BlockStatisticPreservingBootstrap(BaseStatisticPreservingBootstrap):
    block_bootstrap: Optional[BlockBootstrap] = Field(
        default_factory=MovingBlockBootstrap,
        description="The block bootstrap algorithm.",
    )
    """
    Block Statistic Preserving Bootstrap class for time series data.

    This class applies statistic-preserving bootstrapping to blocks of the time series.
    The residuals are resampled using the specified block structure and added to
    the fitted values to generate new samples.

    Parameters
    ----------
    block_bootstrap : BlockBootstrap, default=MovingBlockBootstrap()
        The block bootstrap algorithm.
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    statistic : Callable, default=np.mean
        A callable function to compute the statistic that should be preserved.
    statistic_axis : Integral, default=0
        The axis along which the statistic should be computed.
    statistic_keepdims : bool, default=False
        Whether to keep the dimensions of the statistic or not.
    rng :  Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

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
        n_bootstraps: int = 10,
        block_bootstrap: Optional[BlockBootstrap] = None,
        statistic: Optional[
            Callable
        ] = np.mean,  # Align with BaseStatisticPreservingBootstrap
        statistic_axis: int = 0,  # Align with BaseStatisticPreservingBootstrap
        statistic_keepdims: bool = False,
        rng: Optional[np.random.Generator] = None,
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
        super().__init__(  # Calls BaseStatisticPreservingBootstrap.__init__
            n_bootstraps=n_bootstraps,
            statistic=statistic,  # type: ignore[arg-type]
            statistic_axis=statistic_axis,
            statistic_keepdims=statistic_keepdims,
            rng=rng,
        )
        if block_bootstrap is None:
            self.block_bootstrap = MovingBlockBootstrap()
        else:
            self.block_bootstrap = block_bootstrap

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y=None, n: Optional[int] = None
    ):
        current_statistic_to_match = super()._calculate_statistic(
            X=X
        )  # Always calculate based on current X
        if self.block_bootstrap is None:
            raise ValueError("block_bootstrap must be initialized.")
        (
            block_indices,
            block_data,
        ) = self.block_bootstrap._generate_samples_single_bootstrap(
            X=X, n=X.shape[0] if n is None else n
        )

        block_data_concat = np.concatenate(block_data, axis=0)
        logging.debug(
            f"DEBUG: BlockStatisticPreservingBootstrap - X.shape: {X.shape}"
        )
        logging.debug(
            f"DEBUG: BlockStatisticPreservingBootstrap - block_data_concat.shape: {block_data_concat.shape}"
        )
        # Calculate the bootstrapped statistic
        statistic_bootstrapped = self._calculate_statistic(block_data_concat)
        logging.debug(
            f"DEBUG: BlockStatisticPreservingBootstrap - statistic_bootstrapped.shape (before expand): {statistic_bootstrapped.shape}"
        )
        # Ensure statistic_bootstrapped has the same number of dimensions as block_data_concat for broadcasting
        if statistic_bootstrapped.ndim == 1 and block_data_concat.ndim == 2:
            if self.statistic_axis == 0:
                statistic_bootstrapped = np.expand_dims(
                    statistic_bootstrapped, axis=0
                )
            elif self.statistic_axis == 1:
                statistic_bootstrapped = np.expand_dims(
                    statistic_bootstrapped, axis=1
                )
        logging.debug(
            f"DEBUG: BlockStatisticPreservingBootstrap - statistic_bootstrapped.shape (after expand): {statistic_bootstrapped.shape}"
        )

        # Ensure current_statistic_to_match has the same number of dimensions as X for broadcasting
        if current_statistic_to_match.ndim == 1 and X.ndim == 2:
            if self.statistic_axis == 0:
                current_statistic_to_match = np.expand_dims(
                    current_statistic_to_match, axis=0
                )
            elif self.statistic_axis == 1:
                current_statistic_to_match = np.expand_dims(
                    current_statistic_to_match, axis=1
                )

        # Calculate the bias
        # Check for shape compatibility before subtraction
        if current_statistic_to_match.shape != statistic_bootstrapped.shape:
            raise ValueError(
                f"Shape mismatch for statistic preservation. Original statistic "
                f"shape {current_statistic_to_match.shape} is not compatible "
                f"with bootstrapped statistic shape {statistic_bootstrapped.shape}. "
                f"This often indicates an issue with the block bootstrap "
                f"generating a different number of samples than the original data, "
                f"especially when statistic_axis=1. "
                f"Original data length: {X.shape[0]}, "
                f"Bootstrapped data length: {block_data_concat.shape[0]}"
            )

        bias = (
            current_statistic_to_match - statistic_bootstrapped
        )  # Use the freshly calculated current_statistic_to_match
        logging.debug(
            f"DEBUG: BlockStatisticPreservingBootstrap - current_statistic_to_match.shape: {current_statistic_to_match.shape}"
        )
        logging.debug(
            f"DEBUG: BlockStatisticPreservingBootstrap - bias.shape: {bias.shape}"
        )
        # Add the bias to the bootstrapped sample
        bootstrap_samples = block_data_concat + bias
        return block_indices, [bootstrap_samples]

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from tsbootstrap.block_bootstrap import MovingBlockBootstrap

        bs = MovingBlockBootstrap()
        return {"block_bootstrap": bs}


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
        self, X: np.ndarray, y=None, n: Optional[int] = None
    ):
        # Fit the model and residuals
        self._fit_model(X=X, y=y)
        # Fit the specified distribution to the residuals
        if not self.refit:  # type: ignore
            if self.resids_dist is None or self.resids_dist_params == ():
                if self.resids is None:
                    raise ValueError(
                        "Residuals must be computed before fitting their distribution."
                    )
                (
                    self.resids_dist,
                    self.resids_dist_params,
                ) = super()._fit_distribution(self.resids)

            # Generate new residuals from the fitted distribution
            bootstrap_residuals = self.resids_dist.rvs(
                *self.resids_dist_params,
                size=X.shape[0] if n is None else n,
                random_state=self.rng.integers(0, 2**32 - 1),  # type: ignore
            ).reshape(-1, 1)

            # Add new residuals to the fitted values to create the bootstrap time series
            bootstrap_samples = self.X_fitted + bootstrap_residuals
            return [np.arange(0, X.shape[0])], [bootstrap_samples]

        else:
            # Resample residuals
            if self.resids is None:
                raise ValueError(
                    "Residuals must be computed before resampling."
                )
            if self.rng is None:
                raise ValueError("RNG must be initialized for resampling.")
            resampled_indices = generate_random_indices(
                self.resids.shape[0] if n is None else n, self.rng  # type: ignore[arg-type]
            )
            resampled_residuals = self.resids[resampled_indices]
            resids_dist, resids_dist_params = super()._fit_distribution(
                resampled_residuals
            )
            # Generate new residuals from the fitted distribution
            bootstrap_residuals = resids_dist.rvs(
                *resids_dist_params,
                size=X.shape[0] if n is None else n,
                random_state=self.rng,  # type: ignore
            ).reshape(-1, 1)

            # Add the bootstrapped residuals to the fitted values
            bootstrap_samples = (
                self.X_fitted + bootstrap_residuals
            )  # This was resampled_residuals, should be bootstrap_residuals
            return [resampled_indices], [bootstrap_samples]


class BlockDistributionBootstrap(BaseDistributionBootstrap):
    block_bootstrap: (
        BlockBootstrap  # Declare block_bootstrap as a Pydantic field
    )
    """
    Block Distribution Bootstrap class for time series data.

    This class applies distribution bootstrapping to blocks of the time series.
    The residuals are fit to a distribution, then resampled using the specified
    block structure. Then new residuals are generated from the fitted
    distribution and added to the fitted values to generate new samples.

    Parameters
    ----------
    block_bootstrap : BlockBootstrap, default=MovingBlockBootstrap()
        The block bootstrap algorithm.
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    distribution: str, default='normal'
        The distribution to use for generating the bootstrapped samples.
        Must be one of 'poisson', 'exponential', 'normal', 'gamma', 'beta',
        'lognormal', 'weibull', 'pareto', 'geometric', or 'uniform'.
    refit: bool, default=False
        Whether to refit the distribution to the resampled residuals for each
        bootstrap. If False, the distribution is fit once to the residuals and
        the same distribution is used for all bootstraps.
    model_type : str, default="ar"
        The model type to use. Must be one of "ar", "arima", "sarima", "var", or "arch".
    model_params : dict, default=None
        Additional keyword arguments to pass to the TSFit model.
    order : Integral or list or tuple, default=None
        The order of the model. If None, the best order is chosen via TSFitBestLag.
        If Integral, it is the lag order for AR, ARIMA, and SARIMA, and the lag order
        for ARCH. If list or tuple, the order is a tuple of (p, o, q) for ARIMA
        and (p, d, q, s) for SARIMAX. It is either a single Integral or a
        list of non-consecutive ints for AR, and an Integral for VAR and ARCH.
        If None, the best order is chosen via TSFitBestLag. Do note that TSFitBestLag
        only chooses the best lag, not the best order, so for the tuple values,
        it only chooses the best p, not the best (p, o, q) or (p, d, q, s).
        The rest of the values are set to 0.
    save_models : bool, default=False
        Whether to save the fitted models.
    rng : Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

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
        n_bootstraps: int = 10,
        block_bootstrap: Optional[BlockBootstrap] = None,
        distribution: DistributionTypes = DistributionTypes.NORMAL,  # Align with BaseDistributionBootstrap
        refit: bool = False,
        model_type: ModelTypesWithoutArch = "ar",
        model_params: Optional[dict[str, Any]] = None,  # Align with base
        order: Optional[Union[int, tuple, list]] = None,  # Align with base
        save_models: bool = False,
        rng: Optional[np.random.Generator] = None,
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
        super().__init__(  # Calls BaseDistributionBootstrap.__init__
            n_bootstraps=n_bootstraps,
            distribution=distribution,
            refit=refit,
            save_models=save_models,
            order=order,
            model_type=model_type,
            model_params=(
                model_params if model_params is not None else {}
            ),  # Ensure dict
            rng=rng,
        )
        if block_bootstrap is None:
            block_bootstrap = MovingBlockBootstrap()
        self.block_bootstrap = block_bootstrap

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y=None, n: Optional[int] = None
    ):
        # Fit the model and residuals
        super()._fit_model(X=X, y=y)
        if self.resids is None:
            raise ValueError(
                "Residuals must be computed before generating blocks from them."
            )
        (
            block_indices,
            block_data,
        ) = self.block_bootstrap._generate_samples_single_bootstrap(
            X=self.resids, n=X.shape[0] if n is None else n
        )
        block_data_concat = np.concatenate(block_data, axis=0)

        # Logic for distribution fitting and sampling
        if not self.refit:
            if self.resids_dist is None or self.resids_dist_params == ():
                self.resids_dist, self.resids_dist_params = (
                    super()._fit_distribution(block_data_concat)
                )

            bootstrap_residuals = self.resids_dist.rvs(
                *self.resids_dist_params,
                size=(
                    X.shape[0] if n is None else n
                ),  # Use X.shape[0] or n for the size of the sample
                random_state=self.rng.integers(0, 2**32 - 1),  # type: ignore
            ).reshape(-1, 1)
        else:  # self.refit is True
            resids_dist, resids_dist_params = super()._fit_distribution(
                block_data_concat
            )
            bootstrap_residuals = resids_dist.rvs(
                *resids_dist_params,
                size=(
                    X.shape[0] if n is None else n
                ),  # Use X.shape[0] or n for the size of the sample
                random_state=self.rng.integers(0, 2**32 - 1),  # type: ignore
            ).reshape(-1, 1)

        # Add the bootstrapped residuals to the fitted values
        bootstrap_samples = self.X_fitted + bootstrap_residuals
        return block_indices, [bootstrap_samples]

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from tsbootstrap.block_bootstrap import MovingBlockBootstrap

        bs = MovingBlockBootstrap()
        return {"block_bootstrap": bs}


class WholeSieveBootstrap(BaseSieveBootstrap):
    """
    Implementation of the Sieve bootstrap method for time series data.

    This class applies Sieve bootstrapping to the entire time series,
    without any block structure. This is the most basic form of Sieve
    bootstrapping. The residuals are fit to a second model, and then new
    samples are generated by adding the new residuals to the fitted values.

    Parameters
    ----------
    resids_model_type : str, default="ar"
        The model type to use for fitting the residuals. Must be one of "ar", "arima", "sarima", "var", or "arch".
    resids_order : Integral or list or tuple, default=None
        The order of the model to use for fitting the residuals. If None, the order is automatically determined.
    save_resids_models : bool, default=False
        Whether to save the fitted models for the residuals.
    kwargs_base_sieve : dict, default=None
        Keyword arguments to pass to the SieveBootstrap class.
    model_type : str, default="ar"
        The model type to use. Must be one of "ar", "arima", "sarima", "var", or "arch".
    model_params : dict, default=None
        Additional keyword arguments to pass to the TSFit model.
    order : Integral or list or tuple, default=None
        The order of the model. If None, the best order is chosen via TSFitBestLag.
        If Integral, it is the lag order for AR, ARIMA, and SARIMA,
        and the lag order for ARCH. If list or tuple, the order is a
        tuple of (p, o, q) for ARIMA and (p, d, q, s) for SARIMAX.
        It is either a single Integral or a list of non-consecutive ints for AR,
        and an Integral for VAR and ARCH. If None, the best order is chosen via
        TSFitBestLag. Do note that TSFitBestLag only chooses the best lag,
        not the best order, so for the tuple values, it only chooses the best p,
        not the best (p, o, q) or (p, d, q, s). The rest of the values are set to 0.

    Methods
    -------
    _generate_samples_single_bootstrap : Generate a single bootstrapped sample.
    """

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y=None, n: Optional[int] = None
    ):
        self._fit_model(X=X, y=y)
        if self.resids is None:
            raise ValueError("Residuals must be computed by _fit_model.")
        self._fit_resids_model(X=self.resids)

        if self.X_fitted is None:
            raise ValueError("X_fitted must be computed by _fit_model.")
        if self.rng is None:
            raise ValueError("RNG must be initialized.")
        if self.resid_fit_model is None:
            raise ValueError(
                "Residuals model must be fitted by _fit_resids_model."
            )
        ts_simulator = TimeSeriesSimulator(
            X_fitted=self.X_fitted,
            rng=self.rng,
            fitted_model=self.resid_fit_model,  # type: ignore[attr-defined]
        )

        if self.resid_model_type is None:
            raise ValueError("resid_model_type must be set.")
        if self.resids is None:
            raise ValueError(
                "Residuals must be available for sieve simulation."
            )
        simulated_samples = ts_simulator.generate_samples_sieve(
            model_type=self.resid_model_type,  # type: ignore[attr-defined]
            resids_lags=self.resid_order,  # type: ignore[attr-defined]
            resids_coefs=self.resid_coefs,  # type: ignore[attr-defined]
            resids=self.resids,
            n_samples=X.shape[0] if n is None else n,
        )

        return [np.arange(int(X.shape[0]))], [simulated_samples]


class BlockSieveBootstrap(BaseSieveBootstrap):
    block_bootstrap: Optional[BlockBootstrap] = Field(
        default_factory=MovingBlockBootstrap,
        description="The block bootstrap algorithm.",
    )
    """
    Implementation of the Sieve bootstrap method for time series data.

    This class applies Sieve bootstrapping to blocks of the time series.
    The residuals are fit to a second model, then resampled using the
    specified block structure. The new residuals are then added to the
    fitted values to generate new samples.

    Parameters
    ----------
    block_bootstrap : BlockBootstrap, default=MovingBlockBootstrap()
        The block bootstrap algorithm.
    resids_model_type : str, default="ar"
        The model type to use for fitting the residuals. Must be one of "ar", "arima", "sarima", "var", or "arch".
    resids_order : Integral or list or tuple, default=None
        The order of the model to use for fitting the residuals. If None, the order is automatically determined.
    save_resids_models : bool, default=False
        Whether to save the fitted models for the residuals.
    kwargs_base_sieve : dict, default=None
        Keyword arguments to pass to the SieveBootstrap class.
    model_type : str, default="ar"
        The model type to use. Must be one of "ar", "arima", "sarima", "var", or "arch".
    model_params : dict, default=None
        Additional keyword arguments to pass to the TSFit model.
    order : Integral or list or tuple, default=None
        The order of the model. If None, the best order is chosen via TSFitBestLag.
        If Integral, it is the lag order for AR, ARIMA, and SARIMA,
        and the lag order for ARCH. If list or tuple, the order is a
        tuple of (p, o, q) for ARIMA and (p, d, q, s) for SARIMAX.
        It is either a single Integral or a list of non-consecutive ints for AR,
        and an Integral for VAR and ARCH. If None, the best order is chosen via
        TSFitBestLag. Do note that TSFitBestLag only chooses the best lag,
        not the best order, so for the tuple values, it only chooses the best p,
        not the best (p, o, q) or (p, d, q, s). The rest of the values are set to 0.
    save_models : bool, default=False
        Whether to save the fitted models.

    Methods
    -------
    _init_ : Initialize self.
    _generate_samples_single_bootstrap : Generate a single bootstrapped sample.
    """

    def __init__(
        self,
        n_bootstraps: int = 10,
        block_bootstrap: Optional[BlockBootstrap] = None,
        resid_model_type: ModelTypes = "ar",  # Changed from resids_model_type
        resid_order: Optional[
            Union[int, tuple, list]
        ] = None,  # Changed from resids_order
        resid_save_models: bool = False,  # Changed from save_resids_models
        # kwargs_base_sieve was removed
        model_type: ModelTypesWithoutArch = "ar",
        model_params: Optional[dict[str, Any]] = None,
        order: Optional[Union[int, tuple, list]] = None,
        save_models: bool = False,
        rng: Optional[np.random.Generator] = None,
        # resid_model_params is also a field in BaseSieveBootstrap, should be added if used
        resid_model_params: Optional[dict[str, Any]] = None,
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
        super().__init__(  # Calls BaseSieveBootstrap.__init__
            n_bootstraps=n_bootstraps,
            resid_model_type=resid_model_type,  # Changed
            resid_order=resid_order,  # Changed
            resid_save_models=resid_save_models,  # Changed
            resid_model_params=(
                resid_model_params if resid_model_params is not None else {}
            ),  # Added
            model_type=model_type,
            model_params=model_params if model_params is not None else {},
            order=order,
            save_models=save_models,
            rng=rng,
        )
        if block_bootstrap is None:
            self.block_bootstrap = MovingBlockBootstrap()
        else:
            self.block_bootstrap = block_bootstrap

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y=None, n: Optional[int] = None
    ):
        # Fit the model and residuals
        super()._fit_model(X=X, y=y)
        if self.resids is None:
            raise ValueError("Residuals must be computed by _fit_model.")
        super()._fit_resids_model(X=self.resids)

        if self.X_fitted is None:
            raise ValueError("X_fitted must be computed by _fit_model.")
        if self.rng is None:
            raise ValueError("RNG must be initialized.")
        if self.resid_fit_model is None:
            raise ValueError(
                "Residuals model must be fitted by _fit_resids_model."
            )
        ts_simulator = TimeSeriesSimulator(
            X_fitted=self.X_fitted,
            rng=self.rng,
            fitted_model=self.resid_fit_model,  # type: ignore[attr-defined]
        )

        if self.resid_model_type is None:
            raise ValueError("resid_model_type must be set.")
        if self.resids is None:
            raise ValueError(
                "Residuals must be available for sieve simulation."
            )
        simulated_samples = ts_simulator.generate_samples_sieve(
            model_type=self.resid_model_type,  # type: ignore[attr-defined]
            resids_lags=self.resid_order,  # type: ignore[attr-defined]
            resids_coefs=self.resid_coefs,  # type: ignore[attr-defined]
            resids=self.resids,
            n_samples=X.shape[0] if n is None else n,
        )

        if self.X_fitted is None:
            raise ValueError("X_fitted must be computed for difference.")
        resids_resids = self.X_fitted - simulated_samples
        if self.block_bootstrap is None:
            raise ValueError("block_bootstrap must be initialized.")
        (
            block_indices,
            resids_resids_resampled,
        ) = self.block_bootstrap._generate_samples_single_bootstrap(
            X=resids_resids, n=X.shape[0] if n is None else n
        )
        resids_resids_resampled_concat = np.concatenate(
            resids_resids_resampled, axis=0
        )

        bootstrapped_samples = self.X_fitted + resids_resids_resampled_concat

        return block_indices, [bootstrapped_samples]

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from tsbootstrap.block_bootstrap import MovingBlockBootstrap

        bs = MovingBlockBootstrap()
        return {"block_bootstrap": bs}
