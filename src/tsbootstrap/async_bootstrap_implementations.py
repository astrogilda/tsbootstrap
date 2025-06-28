"""
Async-enabled bootstrap implementations.

This module provides async versions of bootstrap implementations that support
parallel generation for improved performance.
"""

import asyncio
from typing import Any, List, Optional, Tuple

import numpy as np
from pydantic import Field, PrivateAttr, computed_field

from tsbootstrap.async_bootstrap import AsyncBootstrap, AsyncBootstrapMixin
from tsbootstrap.bootstrap import (
    BlockResidualBootstrap,
    WholeResidualBootstrap,
    WholeSieveBootstrap,
)
from tsbootstrap.bootstrap_factory import BootstrapFactory


@BootstrapFactory.register("async_whole_residual")
class AsyncWholeResidualBootstrap(AsyncBootstrapMixin, WholeResidualBootstrap):
    """
    Async-enabled Whole Residual Bootstrap.

    Inherits all functionality from WholeResidualBootstrap and adds
    async/parallel generation capabilities.
    """

    @computed_field
    @property
    def recommended_workers(self) -> int:
        """Recommend number of workers based on bootstrap count."""
        import os

        cpu_count = os.cpu_count() or 4

        if self.n_bootstraps < 10:
            return 1
        elif self.n_bootstraps < 100:
            return min(4, cpu_count)
        else:
            return min(8, cpu_count)


@BootstrapFactory.register("async_block_residual")
class AsyncBlockResidualBootstrap(AsyncBootstrapMixin, BlockResidualBootstrap):
    """
    Async-enabled Block Residual Bootstrap.

    Inherits all functionality from BlockResidualBootstrap and adds
    async/parallel generation capabilities.
    """

    @computed_field
    @property
    def optimal_parallelization(self) -> bool:
        """Determine if parallelization is beneficial."""
        # Block bootstrap benefits from parallelization when
        # we have many bootstraps or large block sizes
        return self.n_bootstraps > 10 or self.block_length > 50


@BootstrapFactory.register("async_whole_sieve")
class AsyncWholeSieveBootstrap(AsyncBootstrapMixin, WholeSieveBootstrap):
    """
    Async-enabled Whole Sieve Bootstrap.

    Inherits all functionality from WholeSieveBootstrap and adds
    async/parallel generation capabilities.
    """

    # Override to use threads for model fitting operations
    use_processes: bool = Field(
        default=False,  # Sieve bootstrap involves model fitting, better with threads
        description="Use threads for better model fitting performance",
    )


class AsyncBootstrapEnsemble(AsyncBootstrap):
    """
    Ensemble bootstrap that can run multiple bootstrap methods in parallel.

    This allows comparing different bootstrap methods or combining their results.
    """

    # Store default as class variable to ensure identity preservation
    _DEFAULT_METHODS = ("whole_residual", "block_residual")

    bootstrap_methods: Tuple[str, ...] = Field(
        default=_DEFAULT_METHODS,
        description="Tuple of bootstrap methods to run in ensemble",
    )

    combine_method: str = Field(
        default="concatenate",
        description="How to combine results: 'concatenate', 'average', 'median'",
    )

    # Private attributes
    _bootstrappers: Optional[List[AsyncBootstrap]] = PrivateAttr(default=None)
    _bootstrap_methods_init: Any = PrivateAttr(default=None)

    def __init__(self, **data: Any) -> None:
        """Initialize with sklearn-compatible parameter tracking."""
        # Store original bootstrap_methods value before validation
        bootstrap_methods_original = data.get("bootstrap_methods", self.__class__._DEFAULT_METHODS)

        # Initialize parent
        super().__init__(**data)

        # Store original value for sklearn compatibility
        self._bootstrap_methods_init = bootstrap_methods_original

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Override to return original bootstrap_methods value for sklearn compatibility."""
        params = super().get_params(deep)
        # Replace bootstrap_methods with the original value
        if "bootstrap_methods" in params:
            params["bootstrap_methods"] = self._bootstrap_methods_init
        return params

    def _create_bootstrappers(self) -> List[AsyncBootstrap]:
        """Create bootstrap instances for ensemble."""
        if self._bootstrappers is None:
            self._bootstrappers = []

            for method in self.bootstrap_methods:
                # Create async version of each method
                async_method = f"async_{method}"

                if BootstrapFactory.is_registered(async_method):
                    bootstrap_cls = BootstrapFactory._registry[async_method]
                elif BootstrapFactory.is_registered(method):
                    # Wrap non-async bootstrap with async mixin
                    base_cls = BootstrapFactory._registry[method]

                    # Create dynamic async class
                    class DynamicAsyncBootstrap(AsyncBootstrapMixin, base_cls):
                        pass

                    bootstrap_cls = DynamicAsyncBootstrap
                else:
                    raise ValueError(f"Unknown bootstrap method: {method}")

                # Create instance with shared parameters
                bootstrap = bootstrap_cls(
                    n_bootstraps=self.n_bootstraps // len(self.bootstrap_methods),
                    rng=self.rng,
                )
                self._bootstrappers.append(bootstrap)

        return self._bootstrappers

    async def generate_ensemble_async(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        return_indices: bool = False,
    ) -> np.ndarray:
        """
        Generate bootstrap samples from all methods in parallel.

        Parameters
        ----------
        X : np.ndarray
            Input time series data
        y : Optional[np.ndarray]
            Exogenous variables
        return_indices : bool, default=False
            Whether to return indices (not supported for ensemble)

        Returns
        -------
        np.ndarray
            Combined bootstrap samples from all methods
        """
        if return_indices:
            raise NotImplementedError("Ensemble bootstrap does not support returning indices")

        bootstrappers = self._create_bootstrappers()

        # Run all bootstrappers in parallel
        tasks = [bootstrapper.generate_samples_async(X, y, False) for bootstrapper in bootstrappers]

        all_results = await asyncio.gather(*tasks)

        # Combine results based on method
        if self.combine_method == "concatenate":
            # Flatten and concatenate all samples
            combined = []
            for results in all_results:
                combined.extend(results)
            return combined

        elif self.combine_method == "average":
            # Average samples element-wise
            all_samples = []
            for results in all_results:
                all_samples.extend(results)

            return np.mean(all_samples, axis=0)

        elif self.combine_method == "median":
            # Median of samples element-wise
            all_samples = []
            for results in all_results:
                all_samples.extend(results)

            return np.median(all_samples, axis=0)

        else:
            raise ValueError(f"Unknown combine method: {self.combine_method}")

    def _generate_samples_single_bootstrap(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Not used for ensemble - implements abstract method."""
        raise NotImplementedError("Ensemble bootstrap uses generate_ensemble_async instead")
