"""
Intermediate bootstrap classes for better code organization.

This module provides intermediate base classes that group bootstraps by their
processing approach (whole data vs block-based), reducing code duplication
and improving maintainability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from tsbootstrap.tsfit import TSFit

import numpy as np

from tsbootstrap.base_bootstrap import BaseTimeSeriesBootstrap
from tsbootstrap.common_fields import (
    BLOCK_LENGTH_FIELD,
    MODEL_TYPE_FIELD,
    ORDER_FIELD,
    OVERLAP_FLAG_FIELD,
    SAVE_MODELS_FIELD,
    SEASONAL_ORDER_FIELD,
)
from tsbootstrap.utils.types import ModelTypes, OrderTypes


class WholeDataBootstrap(BaseTimeSeriesBootstrap):
    """
    Base class for bootstraps that work on whole data.

    This intermediate class provides common functionality for bootstrap methods
    that process the entire time series at once, without dividing it into blocks.

    Examples of whole data bootstraps:
    - WholeResidualBootstrap
    - WholeSieveBootstrap
    - WholeMarkovBootstrap
    - WholeDistributionBootstrap
    - WholeStatisticPreservingBootstrap
    """

    def __init__(self, **data):
        """Initialize WholeDataBootstrap with validation."""
        super().__init__(**data)


class ModelBasedWholeDataBootstrap(WholeDataBootstrap):
    """
    Base class for model-based whole data bootstraps.

    This class provides common fields for bootstraps that fit models.
    """

    # Common fields for model-based whole data bootstraps
    model_type: Optional[ModelTypes] = MODEL_TYPE_FIELD
    order: Optional[OrderTypes] = ORDER_FIELD
    seasonal_order: Optional[tuple[int, int, int, int]] = SEASONAL_ORDER_FIELD
    save_models: bool = SAVE_MODELS_FIELD


class BlockBasedBootstrap(BaseTimeSeriesBootstrap):
    """
    Base class for bootstraps that work on blocks of data.

    This intermediate class provides common functionality for bootstrap methods
    that divide the time series into blocks and resample from these blocks.

    Examples of block-based bootstraps:
    - BlockResidualBootstrap
    - BlockSieveBootstrap
    - All BlockBootstrap variants (Moving, Stationary, Circular, etc.)
    - BlockMarkovBootstrap
    - BlockDistributionBootstrap
    - BlockStatisticPreservingBootstrap
    """

    # Common fields for all block-based bootstraps
    block_length: Optional[int] = BLOCK_LENGTH_FIELD
    overlap_flag: bool = OVERLAP_FLAG_FIELD

    def __init__(self, **data):
        """Initialize BlockBasedBootstrap with validation."""
        # Set default block_length if not provided
        if data.get("block_length") is None and "X" in data:
            X = data["X"]
            if isinstance(X, np.ndarray):
                data["block_length"] = int(np.sqrt(len(X)))

        super().__init__(**data)

    def _validate_block_length(self, X: np.ndarray) -> int:
        """
        Validate and set block length.

        Parameters
        ----------
        X : np.ndarray
            The input data array.

        Returns
        -------
        int
            The validated block length.
        """
        # Default to square root of data length if not provided
        block_length = int(np.sqrt(len(X))) if self.block_length is None else self.block_length

        if block_length > len(X):
            raise ValueError(
                f"Block length ({block_length}) cannot be greater than "
                f"the length of the time series ({len(X)})."
            )

        return block_length


class ModelBasedBlockBootstrap(BlockBasedBootstrap):
    """
    Base class for model-based block bootstraps.

    This class provides common fields for block bootstraps that fit models.
    """

    # Common fields for model-based block bootstraps
    model_type: Optional[ModelTypes] = MODEL_TYPE_FIELD
    order: Optional[OrderTypes] = ORDER_FIELD
    seasonal_order: Optional[tuple[int, int, int, int]] = SEASONAL_ORDER_FIELD
    save_models: bool = SAVE_MODELS_FIELD


class ModelBasedBootstrap:
    """
    Mixin class for bootstrap methods that require model fitting.

    This mixin provides common functionality for bootstraps that fit
    time series models to the data before resampling.
    """

    @property
    def requires_model_fitting(self) -> bool:
        """Check if this bootstrap requires model fitting."""
        return True

    def _fit_model_common(self, X: np.ndarray, **kwargs):
        """
        Common model fitting logic that can be shared across bootstrap types.

        This method should be called by the specific _fit_model implementations
        in derived classes.

        Parameters
        ----------
        X : np.ndarray
            The input time series data.
        **kwargs
            Additional keyword arguments for model fitting.

        Returns
        -------
        tuple
            Fitted model and residuals.
        """
        # Initialize the model
        tsfit = TSFit(
            order=self.order,
            model_type=self.model_type,
            seasonal_order=self.seasonal_order,
        )

        # Fit the model
        tsfit.fit(X=X, **kwargs)

        # Get residuals
        residuals = tsfit.get_residuals()

        return tsfit, residuals
