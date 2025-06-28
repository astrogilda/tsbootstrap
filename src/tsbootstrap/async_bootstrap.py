"""
Async/Parallel Bootstrap Support.

This module provides async and parallel execution capabilities for bootstrap
generation, improving performance for large-scale bootstrap operations.
"""

import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr, computed_field

from tsbootstrap.base_bootstrap import BaseTimeSeriesBootstrap


class AsyncBootstrapMixin(BaseModel):
    """
    Mixin that adds async/parallel capabilities to bootstrap classes.

    This mixin provides:
    - Async bootstrap generation using asyncio
    - Process-based parallelization for CPU-intensive operations
    - Thread-based parallelization for I/O-bound operations
    - Automatic selection of optimal executor based on workload
    """

    model_config = {"arbitrary_types_allowed": True}

    # Executor configuration
    max_workers: Optional[int] = Field(
        default=None,
        description="Maximum number of workers for parallel execution",
    )
    use_processes: bool = Field(
        default=False,
        description="Use processes (True) or threads (False) for parallelization",
    )
    chunk_size: int = Field(
        default=10,
        gt=0,
        description="Number of bootstraps to process in each chunk",
    )

    # Private attributes
    _executor: Optional[Union[ProcessPoolExecutor, ThreadPoolExecutor]] = PrivateAttr(default=None)

    @computed_field
    @property
    def optimal_chunk_size(self) -> int:
        """Calculate optimal chunk size based on number of bootstraps."""
        if hasattr(self, "n_bootstraps"):
            n_bootstraps = self.n_bootstraps
            if n_bootstraps <= 10:
                return 1
            elif n_bootstraps <= 100:
                return 10
            else:
                return max(10, n_bootstraps // 10)
        return self.chunk_size

    def _get_executor(self) -> Union[ProcessPoolExecutor, ThreadPoolExecutor]:
        """Get or create the appropriate executor."""
        if self._executor is None:
            if self.use_processes:
                self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
            else:
                self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self._executor

    def _cleanup_executor(self):
        """Clean up executor resources."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    async def generate_samples_async(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        return_indices: bool = False,
    ) -> List[Union[np.ndarray, tuple[np.ndarray, np.ndarray]]]:
        """
        Generate bootstrap samples asynchronously.

        Parameters
        ----------
        X : np.ndarray
            Input time series data
        y : Optional[np.ndarray]
            Exogenous variables
        return_indices : bool, default=False
            Whether to return bootstrap indices

        Returns
        -------
        List[Union[np.ndarray, tuple]]
            List of bootstrap samples (and indices if requested)
        """
        # Validate inputs
        X_checked, y_checked = self._check_X_y(X, y)

        # Create async tasks
        loop = asyncio.get_event_loop()
        executor = self._get_executor()

        # Calculate chunks
        n_bootstraps = getattr(self, "n_bootstraps", 10)
        chunk_size = self.optimal_chunk_size
        n_chunks = (n_bootstraps + chunk_size - 1) // chunk_size

        tasks = []
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n_bootstraps)
            n_samples_chunk = end_idx - start_idx

            task = loop.run_in_executor(
                executor,
                self._generate_chunk,
                X_checked,
                y_checked,
                n_samples_chunk,
                return_indices,
            )
            tasks.append(task)

        # Gather results
        chunk_results = await asyncio.gather(*tasks)

        # Flatten results
        results = []
        for chunk in chunk_results:
            results.extend(chunk)

        return results

    def _generate_chunk(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        n_samples: int,
        return_indices: bool,
    ) -> List[Union[np.ndarray, tuple[np.ndarray, np.ndarray]]]:
        """Generate a chunk of bootstrap samples."""
        results = []

        for _ in range(n_samples):
            indices, data_list = self._generate_samples_single_bootstrap(X, y)

            if return_indices:
                # Process indices
                if isinstance(indices, list):
                    processed_indices = [np.asarray(idx) for idx in indices if idx is not None]
                else:
                    processed_indices = [np.asarray(indices)] if indices is not None else []

                indices_concat = (
                    np.concatenate(processed_indices, axis=0) if processed_indices else np.array([])
                )

                # Process data
                processed_data = [np.asarray(d) for d in data_list if d is not None]
                data_concat = (
                    np.concatenate(processed_data, axis=0) if processed_data else np.array([])
                )

                results.append((data_concat, indices_concat))
            else:
                # Process data only
                processed_data = [np.asarray(d) for d in data_list if d is not None]
                data_concat = (
                    np.concatenate(processed_data, axis=0) if processed_data else np.array([])
                )

                results.append(data_concat)

        return results

    def bootstrap_parallel(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        return_indices: bool = False,
        batch_size: Optional[int] = None,
    ) -> List[Union[np.ndarray, tuple[np.ndarray, np.ndarray]]]:
        """
        Generate bootstrap samples using parallel execution.

        synchronous wrapper around the async functionality
        for easier integration with existing code.

        Parameters
        ----------
        X : np.ndarray
            Input time series data
        y : Optional[np.ndarray]
            Exogenous variables
        return_indices : bool, default=False
            Whether to return bootstrap indices
        batch_size : Optional[int]
            Override chunk size for this operation

        Returns
        -------
        List[Union[np.ndarray, tuple]]
            List of bootstrap samples (and indices if requested)
        """
        # Temporarily override chunk size if specified
        original_chunk_size = self.chunk_size
        if batch_size is not None:
            self.chunk_size = batch_size

        try:
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                results = loop.run_until_complete(self.generate_samples_async(X, y, return_indices))
                return results
            finally:
                loop.close()
        finally:
            # Restore original chunk size
            self.chunk_size = original_chunk_size
            # Clean up executor
            self._cleanup_executor()


class AsyncBootstrap(AsyncBootstrapMixin, BaseTimeSeriesBootstrap):
    """
    Base class for async-enabled bootstrap implementations.

    This class combines the async mixin with the refactored bootstrap base
    to provide both standard and async bootstrap generation capabilities.
    """

    _tags = {
        "object_type": "bootstrap",
        "capability:multivariate": True,
        "bootstrap-type": "async-base",
        "X-y-must-have-same-index": True,
        "y_inner_mtype": "None",
        "requires_y": False,
        "python_version": None,
        "python_dependencies": None,
        "python_dependencies_alias": None,
        # Mark as abstract/non-instantiable for tests
        "_skip_test": True,
    }

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Override setattr to allow test attributes for skbase compatibility.

        This allows setting arbitrary attributes that start with 'test_' to support
        skbase's test suite which checks for side effects between tests.
        """
        if name.startswith("test_"):
            # For test attributes, bypass Pydantic validation
            object.__setattr__(self, name, value)
        else:
            # Use Pydantic's normal setattr
            super().__setattr__(name, value)

    def __del__(self):
        """Ensure executor cleanup on deletion."""
        # Handle edge case where __del__ is called before full initialization
        try:
            if hasattr(self, "_executor"):
                self._cleanup_executor()
        except AttributeError:
            pass
