"""
Async execution service for bootstrap operations.

This service provides async and parallel execution capabilities,
providing async and parallel execution capabilities.
"""

import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Callable, List, Optional, Union

import numpy as np

from tsbootstrap.services.async_compatibility import async_compat


class AsyncExecutionService:
    """
    Service for async and parallel bootstrap execution.

    This service provides:
    - Async bootstrap generation using asyncio
    - Process-based parallelization for CPU-intensive operations
    - Thread-based parallelization for I/O-bound operations
    - Automatic selection of optimal executor based on workload
    """

    def __init__(
        self, max_workers: Optional[int] = None, use_processes: bool = False, chunk_size: int = 10
    ):
        """
        Initialize async execution service.

        Parameters
        ----------
        max_workers : Optional[int]
            Maximum number of workers for parallel execution
        use_processes : bool
            Use processes (True) or threads (False) for parallelization
        chunk_size : int
            Number of bootstraps to process in each chunk
        """
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.chunk_size = chunk_size
        self._executor: Optional[Union[ProcessPoolExecutor, ThreadPoolExecutor]] = None

    def calculate_optimal_chunk_size(self, n_bootstraps: int) -> int:
        """
        Calculate optimal chunk size based on number of bootstraps.

        Parameters
        ----------
        n_bootstraps : int
            Total number of bootstraps

        Returns
        -------
        int
            Optimal chunk size
        """
        if n_bootstraps <= 10:
            return 1
        elif n_bootstraps <= 100:
            return 10
        else:
            return max(10, n_bootstraps // 10)

    def _get_executor(self) -> Union[ProcessPoolExecutor, ThreadPoolExecutor]:
        """Get or create the appropriate executor."""
        if self._executor is None:
            if self.use_processes:
                self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
            else:
                self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self._executor

    def cleanup_executor(self):
        """Clean up executor resources."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    async def execute_async_chunks(
        self,
        generate_func: Callable,
        n_bootstraps: int,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        chunk_size: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Execute bootstrap generation in async chunks.

        Parameters
        ----------
        generate_func : Callable
            Function to generate single bootstrap sample
        n_bootstraps : int
            Total number of bootstraps
        X : np.ndarray
            Input data
        y : Optional[np.ndarray]
            Target data
        chunk_size : Optional[int]
            Override default chunk size

        Returns
        -------
        List[np.ndarray]
            List of bootstrap samples
        """
        # Use optimal chunk size if not specified
        if chunk_size is None:
            chunk_size = self.calculate_optimal_chunk_size(n_bootstraps)

        # Get executor
        executor = self._get_executor()

        # Calculate chunks
        n_chunks = (n_bootstraps + chunk_size - 1) // chunk_size

        # Create tasks using compatibility service
        tasks = []
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n_bootstraps)
            n_samples_chunk = end_idx - start_idx

            task = async_compat.run_in_executor(
                executor, self._generate_chunk, generate_func, X, y, n_samples_chunk
            )
            tasks.append(task)

        # Gather results using compatibility service
        chunk_results = await async_compat.gather_tasks(*tasks)

        # Flatten results
        results = []
        for chunk in chunk_results:
            results.extend(chunk)

        return results

    def _generate_chunk(
        self, generate_func: Callable, X: np.ndarray, y: Optional[np.ndarray], n_samples: int
    ) -> List[np.ndarray]:
        """
        Generate a chunk of bootstrap samples.

        Parameters
        ----------
        generate_func : Callable
            Function to generate single bootstrap sample
        X : np.ndarray
            Input data
        y : Optional[np.ndarray]
            Target data
        n_samples : int
            Number of samples in this chunk

        Returns
        -------
        List[np.ndarray]
            List of bootstrap samples for this chunk
        """
        results = []

        for _ in range(n_samples):
            sample = generate_func(X, y)
            results.append(sample)

        return results

    def execute_parallel(
        self,
        generate_func: Callable,
        n_bootstraps: int,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        batch_size: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Execute bootstrap generation in parallel (synchronous wrapper).

        Parameters
        ----------
        generate_func : Callable
            Function to generate single bootstrap sample
        n_bootstraps : int
            Total number of bootstraps
        X : np.ndarray
            Input data
        y : Optional[np.ndarray]
            Target data
        batch_size : Optional[int]
            Override chunk size for this operation

        Returns
        -------
        List[np.ndarray]
            List of bootstrap samples
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
                results = loop.run_until_complete(
                    self.execute_async_chunks(generate_func, n_bootstraps, X, y, self.chunk_size)
                )
                return results
            finally:
                loop.close()
        finally:
            # Restore original chunk size
            self.chunk_size = original_chunk_size
            # Clean up executor
            self.cleanup_executor()

    def __del__(self):
        """Ensure executor cleanup on deletion."""
        import contextlib

        with contextlib.suppress(Exception):
            self.cleanup_executor()
