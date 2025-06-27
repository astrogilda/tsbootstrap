"""
Migrated bootstrap implementations using the new architecture.

This module contains bootstrap implementations refactored to use:
- Mixin architecture from base_mixins
- Enhanced configuration types from bootstrap_types_v2
- Factory pattern from bootstrap_factory
- Custom validators from validators
- Shared utilities from bootstrap_common
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from tsbootstrap.time_series_model import TimeSeriesModel
    from tsbootstrap.tsfit import TSFit

import numpy as np
from pydantic import Field, computed_field, model_validator

from tsbootstrap.base_bootstrap import BaseTimeSeriesBootstrap
from tsbootstrap.bootstrap_factory import BootstrapFactory
from tsbootstrap.bootstrap_intermediate import (
    BlockBasedBootstrap,
    ModelBasedBlockBootstrap,
    ModelBasedBootstrap,
    ModelBasedWholeDataBootstrap,
    WholeDataBootstrap,
)
from tsbootstrap.bootstrap_mixins import (
    ModelFittingMixin,
    ResidualResamplingMixin,
    SieveOrderSelectionMixin,
    TimeSeriesReconstructionMixin,
)
from tsbootstrap.common_fields import (
    BLOCK_LENGTH_REQUIRED_FIELD,
    MODEL_TYPE_NO_ARCH_FIELD,
    ORDER_FIELD,
    OVERLAP_FLAG_FIELD,
    SAVE_MODELS_FIELD,
    SEASONAL_ORDER_FIELD,
    create_block_length_field,
)
from tsbootstrap.utils.types import ModelTypesWithoutArch
from tsbootstrap.validators import ModelOrder, PositiveInt


@BootstrapFactory.register("whole_residual")
class WholeResidualBootstrap(
    ModelFittingMixin,
    ResidualResamplingMixin,
    TimeSeriesReconstructionMixin,
    ModelBasedWholeDataBootstrap,
    ModelBasedBootstrap,
):
    """
    Whole Residual Bootstrap implementation using new architecture.

    This bootstrap method fits a time series model and resamples the residuals
    with replacement to generate new time series.
    """

    # Configuration fields
    model_type: ModelTypesWithoutArch = Field(default="ar", description="The model type to use. Options are 'ar', 'ma', 'arma', 'arima', 'sarima', 'var' (no 'arch').")  # type: ignore[assignment]
    order: Optional[ModelOrder] = ORDER_FIELD
    seasonal_order: Optional[ModelOrder] = SEASONAL_ORDER_FIELD
    save_models: bool = SAVE_MODELS_FIELD

    # Private attributes for fitted models
    _fitted_model: Optional[TimeSeriesModel] = None  # type: ignore[assignment]
    _residuals: Optional[np.ndarray] = None  # type: ignore[assignment]

    @computed_field
    @property
    def requires_model_fitting(self) -> bool:
        """Whether this bootstrap requires model fitting."""
        return True

    # The _fit_model method is provided by ModelFittingMixin

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Generate a single bootstrap sample by resampling residuals."""
        if self._fitted_model is None:
            self._fit_model(X, y)

        if self._residuals is None:
            raise ValueError("No residuals available for bootstrapping")

        n_samples = len(X)

        # Use mixin method for residual resampling
        indices, resampled_residuals = self._resample_residuals_whole(
            n_samples=n_samples, replace=True
        )

        # Get fitted values from the model
        fitted_values = self._get_fitted_values(X)

        # Use mixin method for reconstruction
        bootstrapped_series = self._reconstruct_series(
            fitted_values=fitted_values,
            resampled_residuals=resampled_residuals,
            original_shape=X.shape,
            indices=None,  # Not using indices for whole bootstrap
        )

        return indices, [bootstrapped_series]


@BootstrapFactory.register("block_residual")
class BlockResidualBootstrap(
    ModelFittingMixin,
    ResidualResamplingMixin,
    TimeSeriesReconstructionMixin,
    ModelBasedBlockBootstrap,
):
    """
    Block Residual Bootstrap implementation using new architecture.

    This bootstrap method fits a time series model and resamples the residuals
    in blocks to preserve temporal dependencies.
    """

    # Override model_type to exclude ARCH
    model_type: ModelTypesWithoutArch = Field(default="ar", description="The model type to use. Options are 'ar', 'ma', 'arma', 'arima', 'sarima', 'var' (no 'arch').")  # type: ignore[assignment]

    # Use common field definition with custom default
    block_length: Optional[PositiveInt] = create_block_length_field(default=10)

    # Private attributes
    _fitted_model: Optional[TimeSeriesModel] = None  # type: ignore[assignment]
    _residuals: Optional[np.ndarray] = None  # type: ignore[assignment]

    # The _fit_model method is provided by ModelFittingMixin

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Generate a single bootstrap sample by resampling residuals in blocks."""
        if self._fitted_model is None:
            self._fit_model(X, y)

        if self._residuals is None:
            raise ValueError("No residuals available for bootstrapping")

        n_samples = len(X)

        # Use mixin method for block residual resampling
        if self.block_length is None:
            raise ValueError("Block length must be specified for block bootstrap")
        indices, resampled_residuals = self._resample_residuals_block(
            n_samples=n_samples,
            block_length=self.block_length,
            overlap=self.overlap_flag,
        )

        # Get fitted values from the model
        fitted_values = self._get_fitted_values(X)

        # Use mixin method for reconstruction
        bootstrapped_series = self._reconstruct_series(
            fitted_values=fitted_values,
            resampled_residuals=resampled_residuals,
            original_shape=X.shape,
            indices=indices,  # Use indices for block bootstrap
        )

        return indices, [bootstrapped_series]


@BootstrapFactory.register("whole_sieve")
class WholeSieveBootstrap(
    SieveOrderSelectionMixin,
    ModelFittingMixin,
    ResidualResamplingMixin,
    TimeSeriesReconstructionMixin,
    ModelBasedWholeDataBootstrap,
    ModelBasedBootstrap,
):
    """
    Whole Sieve Bootstrap implementation using new architecture.

    The sieve bootstrap uses an autoregressive model with order selected
    based on the sample size.
    """

    # Configuration fields
    model_type: str = Field(default="ar", description="Sieve bootstrap always uses AR models")
    min_lag: PositiveInt = Field(default=1, description="Minimum lag for AR order selection")
    max_lag: Optional[PositiveInt] = Field(
        default=None, description="Maximum lag for AR order selection"
    )
    criterion: str = Field(default="aic", description="Information criterion for order selection")
    save_models: bool = Field(default=False, description="Whether to save fitted models")

    @model_validator(mode="after")
    def validate_lag_order(self) -> WholeSieveBootstrap:
        """Validate that max_lag >= min_lag if max_lag is specified."""
        if self.max_lag is not None and self.max_lag < self.min_lag:
            raise ValueError(f"max_lag ({self.max_lag}) must be >= min_lag ({self.min_lag})")
        return self

    # Private attributes
    _fitted_model: Optional[TimeSeriesModel] = None  # type: ignore[assignment]
    _residuals: Optional[np.ndarray] = None  # type: ignore[assignment]
    _selected_order: Optional[int] = None

    # Sieve bootstrap always uses AR models, but this is enforced in _fit_model

    # The _select_order method is provided by SieveOrderSelectionMixin
    # The _fit_model method is provided by ModelFittingMixin but we override it for sieve

    def _fit_model(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Fit AR model with selected order."""
        # Select order based on sieve bootstrap criterion
        X_for_order = X[:, 0].reshape(-1, 1) if X.ndim == 2 and X.shape[1] > 1 else X

        self._selected_order = self._select_order(X_for_order)

        # Use shared utility for model fitting
        from tsbootstrap.bootstrap_common import BootstrapUtilities

        fitted_model, residuals = BootstrapUtilities.fit_time_series_model(
            X=X,
            y=y,
            model_type="ar",
            order=self._selected_order,
            seasonal_order=None,
        )

        self._fitted_model = fitted_model.model
        self._residuals = residuals

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Generate a single sieve bootstrap sample."""
        if self._fitted_model is None:
            self._fit_model(X, y)

        if self._residuals is None:
            raise ValueError("No residuals available for bootstrapping")

        n_samples = len(X)

        # Use mixin method for residual resampling
        indices, resampled_residuals = self._resample_residuals_whole(
            n_samples=n_samples, replace=True
        )

        # Get fitted values from the model
        fitted_values = self._get_fitted_values(X)

        # Use mixin method for reconstruction
        bootstrapped_series = self._reconstruct_series(
            fitted_values=fitted_values,
            resampled_residuals=resampled_residuals,
            original_shape=X.shape,
            indices=None,  # Not using indices for whole bootstrap
        )

        return indices, [bootstrapped_series]


@BootstrapFactory.register("block_sieve")
class BlockSieveBootstrap(BlockBasedBootstrap, WholeSieveBootstrap):
    """
    Block Sieve Bootstrap implementation using new architecture.

    This bootstrap method fits an AR model and resamples the residuals
    in blocks, preserving temporal dependencies while adapting the AR order.
    """

    # Additional configuration for block structure
    block_length: Optional[PositiveInt] = create_block_length_field(default=10)
    overlap_flag: bool = OVERLAP_FLAG_FIELD

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Generate a single bootstrap sample by resampling residuals in blocks."""
        if self._fitted_model is None:
            self._fit_model(X, y)

        if self._residuals is None:
            raise ValueError("No residuals available for bootstrapping")

        n_samples = len(X)

        # Use mixin method for block residual resampling
        if self.block_length is None:
            raise ValueError("Block length must be specified for block bootstrap")
        indices, resampled_residuals = self._resample_residuals_block(
            n_samples=n_samples,
            block_length=self.block_length,
            overlap=self.overlap_flag,
        )

        # Get fitted values from the model
        fitted_values = self._get_fitted_values(X)

        # Use mixin method for reconstruction
        bootstrapped_series = self._reconstruct_series(
            fitted_values=fitted_values,
            resampled_residuals=resampled_residuals,
            original_shape=X.shape,
            indices=indices,  # Use indices for block bootstrap
        )

        return indices, [bootstrapped_series]
