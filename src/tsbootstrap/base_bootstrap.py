"""
Refactored base classes for time series bootstrap algorithms using mixin architecture.

This module provides a cleaner implementation of base bootstrap classes using
Pydantic 2.x features and simplified inheritance through mixins.
"""

from __future__ import annotations

import abc
import sys
from multiprocessing import Pool
from typing import TYPE_CHECKING, Any, ClassVar, Iterator, Optional, Union

if sys.version_info >= (3, 11) or sys.version_info >= (3, 10):
    pass
else:
    pass

import numpy as np
from pydantic import (
    ConfigDict,
    Field,
    PrivateAttr,
    computed_field,
    field_validator,
)
from skbase.base import BaseObject

from tsbootstrap.base_mixins import (
    NumpySerializationMixin,
    SklearnCompatMixin,
    ValidationMixin,
)
from tsbootstrap.utils.odds_and_ends import time_series_split


class BaseTimeSeriesBootstrap(
    NumpySerializationMixin,
    SklearnCompatMixin,
    ValidationMixin,
    BaseObject,
    abc.ABC,
):
    """
    Refactored base class for time series bootstrapping using mixin architecture.

    This class simplifies the inheritance hierarchy by using focused mixins for
    cross-cutting concerns like sklearn compatibility and numpy serialization.

    Parameters
    ----------
    n_bootstraps : int, default=10
        The number of bootstrap samples to create.
    rng : Optional[Union[int, np.random.Generator]], default=None
        The random number generator or seed. If None, np.random.default_rng() is used.

    Attributes
    ----------
    _rng_init_val : Any
        Private attribute storing the original rng value for sklearn compatibility.
    """

    # Class tags for skbase compatibility
    _tags: ClassVar[dict] = {
        "object_type": "bootstrap",
        "bootstrap_type": "other",
        "capability:multivariate": True,
    }

    # Model configuration with Pydantic 2.x optimizations
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        protected_namespaces=(),
        # Performance optimizations
        validate_default=False,  # Skip validation of defaults
        use_enum_values=True,  # Use enum values directly
    )

    # Fields
    n_bootstraps: int = Field(
        default=10,
        ge=1,
        description="The number of bootstrap samples to create.",
    )

    rng: Optional[np.random.Generator] = Field(
        default=None,
        description="The random number generator or seed.",
    )

    # Private attributes
    _rng_init_val: Any = PrivateAttr(default=None)  # type: ignore[assignment]
    _X: Optional[np.ndarray] = PrivateAttr(default=None)  # type: ignore[assignment]
    _y: Optional[np.ndarray] = PrivateAttr(default=None)  # type: ignore[assignment]

    def __init__(self, **data: Any) -> None:
        """Initialize with sklearn-compatible parameter tracking."""
        # Store original rng value before validation
        rng_original = data.get("rng")

        # Initialize Pydantic model
        super().__init__(**data)

        # Store original value for sklearn compatibility
        self._rng_init_val = rng_original

        # Initialize BaseObject
        BaseObject.__init__(self)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Override to return original rng value for sklearn compatibility."""
        params = super().get_params(deep)
        # Replace rng with the original value
        if "rng" in params:
            params["rng"] = self._rng_init_val
        return params

    @field_validator("rng", mode="before")
    @classmethod
    def _validate_rng_field(cls, v: Any) -> np.random.Generator:
        """Validate and convert rng to Generator instance."""
        if v is None:
            return np.random.default_rng()
        if isinstance(v, np.random.Generator):
            return v
        if isinstance(v, (int, np.integer)):
            return np.random.default_rng(int(v))
        raise TypeError(
            f"Invalid type for rng: {type(v)}. Expected None, int, or np.random.Generator."
        )

    @computed_field
    @property
    def parallel_capable(self) -> bool:
        """Whether parallel bootstrap generation is beneficial."""
        return self.n_bootstraps > 10

    @computed_field
    @property
    def is_fitted(self) -> bool:
        """Whether the bootstrap has been fitted to data."""
        return self._X is not None

    def _check_X_y(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Validate input data."""
        # Use mixin's validation method
        X = self._validate_array_input(X, "X")
        if y is not None:
            y = self._validate_array_input(y, "y")

        # For our refactored implementation, we handle multivariate data properly
        # in the bootstrap implementations, so we'll do basic validation here
        if X.ndim not in [1, 2]:
            raise ValueError("X must be 1-dimensional or 2-dimensional")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if len(X) == 0:
            raise ValueError("X cannot be empty")

        # Validate y if provided
        if y is not None:
            if y.ndim > 2:
                raise ValueError("y must be at most 2-dimensional")

            if len(y) != len(X):
                raise ValueError(f"X and y must have the same length. Got {len(X)} and {len(y)}")

        return X, y

    def bootstrap(
        self,
        X: np.ndarray,
        return_indices: bool = False,
        y: Optional[np.ndarray] = None,
        test_ratio: Optional[float] = None,
    ) -> Iterator[Union[np.ndarray, tuple[np.ndarray, np.ndarray]]]:
        """
        Generate bootstrapped samples of time series data.

        Parameters
        ----------
        X : array-like of shape (n_timepoints, n_features)
            The endogenous time series to bootstrap.
        return_indices : bool, default=False
            If True, return index references for the bootstrap sample.
        y : array-like of shape (n_timepoints, n_features_exog), default=None
            Exogenous time series to use in bootstrapping.
        test_ratio : float, default=None
            If provided, this fraction of data is removed from the end before bootstrapping.

        Yields
        ------
        X_boot_i : np.ndarray of shape (n_timepoints_boot_i, n_features)
            i-th bootstrapped sample of X.
        indices_i : np.ndarray of shape (n_timepoints_boot_i,), optional
            Index references for the i-th bootstrapped sample, if return_indices=True.
        """
        X_checked, y_checked = self._check_X_y(X, y)

        # Store for later use
        self._X = X_checked
        self._y = y_checked

        if test_ratio is not None:
            X_inner, _ = time_series_split(X_checked, test_ratio=test_ratio)
            y_inner = None
            if y_checked is not None:
                y_inner, _ = time_series_split(y_checked, test_ratio=test_ratio)
        else:
            X_inner = X_checked
            y_inner = y_checked

        yield from self._bootstrap(X=X_inner, return_indices=return_indices, y=y_inner)

    def _bootstrap(
        self,
        X: np.ndarray,
        return_indices: bool = False,
        y: Optional[np.ndarray] = None,
        n_jobs: int = 1,
    ) -> Iterator[Union[np.ndarray, tuple[np.ndarray, np.ndarray]]]:
        """
        Generate bootstrapped samples with optional parallel processing.

        This method handles both sequential and parallel bootstrap generation
        based on the n_jobs parameter.
        """
        actual_n_jobs = n_jobs
        if actual_n_jobs == -1:
            import os

            actual_n_jobs = os.cpu_count() or 1

        if actual_n_jobs <= 0:
            actual_n_jobs = 1

        if actual_n_jobs == 1 or not self.parallel_capable:
            # Sequential processing
            for _ in range(self.n_bootstraps):
                indices, data_list = self._generate_samples_single_bootstrap(X, y)
                result = self._process_bootstrap_result(indices, data_list, return_indices)
                yield result
        else:
            # Parallel processing
            args = [(X, y) for _ in range(self.n_bootstraps)]
            with Pool(processes=actual_n_jobs) as pool:
                results = pool.starmap(self._generate_samples_single_bootstrap, args)

            for indices, data_list in results:
                result = self._process_bootstrap_result(indices, data_list, return_indices)
                yield result

    def _process_bootstrap_result(
        self,
        indices: Union[np.ndarray, list[np.ndarray]],
        data_list: list[np.ndarray],
        return_indices: bool,
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Process raw bootstrap results into final format."""
        # Process data
        processed_data = [np.asarray(d) for d in data_list if d is not None]
        data_concat = np.concatenate(processed_data, axis=0) if processed_data else np.array([])

        if return_indices:
            # Process indices
            if isinstance(indices, list):
                processed_indices = [np.asarray(idx) for idx in indices if idx is not None]
            else:
                processed_indices = [np.asarray(indices)] if indices is not None else []

            final_indices = (
                np.concatenate(processed_indices, axis=0) if processed_indices else np.array([])
            )
            return data_concat, final_indices
        else:
            return data_concat

    def get_n_bootstraps(self, X=None, y=None) -> int:
        """
        Returns the number of bootstrap instances produced by the bootstrap.

        Parameters
        ----------
        X : 2D array-like of shape (n_timepoints, n_features), optional
            The endogenous time series to bootstrap.
        y : array-like of shape (n_timepoints, n_features_exog), optional
            Exogenous time series to use in bootstrapping.

        Returns
        -------
        int
            The number of bootstrap instances produced by the bootstrap.
        """
        return self.n_bootstraps

    @abc.abstractmethod
    def _generate_samples_single_bootstrap(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> tuple[Union[np.ndarray, list[np.ndarray]], list[np.ndarray]]:
        """
        Generate a single bootstrap sample.

        To be implemented by derived classes.

        Parameters
        ----------
        X : np.ndarray
            Input time series data
        y : Optional[np.ndarray]
            Exogenous variables

        Returns
        -------
        indices : Union[np.ndarray, list[np.ndarray]]
            Bootstrap indices
        data_list : list[np.ndarray]
            List of data arrays for the bootstrap sample
        """

    def __repr__(self) -> str:
        """String representation of the bootstrap object."""
        class_name = self.__class__.__name__
        params = self.get_params(deep=False)
        param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{class_name}({param_str})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.__class__.__name__} with {self.n_bootstraps} bootstrap samples"
