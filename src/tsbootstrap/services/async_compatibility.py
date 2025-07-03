"""
Async compatibility: Unified interface across Python's async ecosystem.

In the evolving landscape of Python async programming, we face a fundamental
challenge: how to write async code that works seamlessly across different
async frameworks without sacrificing performance or clarity. This module
represents our solution—a carefully designed compatibility layer that abstracts
away framework differences while maintaining zero-cost abstractions.

We've built this service around anyio, the emerging standard for async
framework interoperability. However, recognizing that many users only need
asyncio support, we've made anyio optional. Users who stick with asyncio
pay no runtime penalty—the service detects missing dependencies and falls
back to pure asyncio implementations. Those who need trio compatibility
can install our async extras to unlock full cross-framework support.

The architecture follows a principle we call "progressive enhancement."
Basic async operations work out of the box with stdlib asyncio. Advanced
features like structured concurrency and cancellation scopes become available
when anyio is present. This design ensures that simple use cases remain
simple while complex requirements are fully supported.

Installation:
- Basic async support (asyncio only): No additional dependencies needed
- Full async support (asyncio + trio): pip install tsbootstrap[async-extras]
"""

import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable, List, Optional, TypeVar, Union

try:
    import anyio
    import sniffio

    HAS_ANYIO = True
except ImportError:
    HAS_ANYIO = False
    anyio = None
    sniffio = None

# Type checking imports that won't be executed at runtime
if TYPE_CHECKING and anyio is None:
    import anyio  # noqa: F401


T = TypeVar("T")


class AsyncCompatibilityService:
    """
    Cross-framework async orchestration service.

    We've designed this service to solve a critical problem in modern Python:
    the fragmentation of the async ecosystem. While asyncio ships with Python,
    alternative frameworks like trio offer compelling advantages—structured
    concurrency, better cancellation semantics, and more predictable behavior.
    Yet most libraries only support asyncio, creating compatibility barriers.

    This service acts as a universal translator between async dialects. It
    detects the running async framework and provides appropriate implementations
    for common operations. The abstraction is zero-cost: asyncio users see
    pure asyncio calls, while trio users get proper trio semantics. No
    performance penalty, no behavioral compromises.

    The implementation leverages anyio when available but gracefully degrades
    to asyncio-only mode when it's not. This progressive enhancement strategy
    ensures that basic users aren't forced to install extra dependencies while
    power users can unlock full cross-framework support.
    """

    def __init__(self):
        """Initialize the compatibility service."""
        self._executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]] = None

    def detect_backend(self) -> str:
        """
        Detect the current async backend.

        Returns
        -------
        str
            'asyncio', 'trio', or 'unknown'
        """
        if HAS_ANYIO and sniffio:
            try:
                return sniffio.current_async_library()
            except sniffio.AsyncLibraryNotFoundError:
                pass

        # Check if we're in an asyncio context
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            return "asyncio"

        return "unknown"

    async def get_current_backend(self) -> str:
        """Get current backend (async version)."""
        return self.detect_backend()

    async def run_in_thread(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run a sync function in a thread."""
        backend = self.detect_backend()

        if backend == "trio" or (HAS_ANYIO and backend != "asyncio"):
            # Use anyio for trio compatibility
            if not HAS_ANYIO:
                raise RuntimeError(
                    "Trio async backend detected but anyio is not installed. "
                    "To use trio, install the async extras: pip install tsbootstrap[async-extras]. "
                    "Alternatively, switch to asyncio which requires no additional dependencies."
                )
            return await anyio.to_thread.run_sync(func, *args, **kwargs)
        else:
            # Use asyncio's run_in_executor
            loop = asyncio.get_event_loop()
            # Create partial function to handle kwargs
            if kwargs:
                from functools import partial

                func = partial(func, **kwargs)
                return await loop.run_in_executor(None, func, *args)
            else:
                return await loop.run_in_executor(None, func, *args)

    async def sleep(self, seconds: float) -> None:
        """Sleep for given seconds."""
        backend = self.detect_backend()

        if backend == "trio" or (HAS_ANYIO and backend != "asyncio"):
            # Use anyio for trio compatibility
            if not HAS_ANYIO:
                raise RuntimeError(
                    "Trio async backend detected but anyio is not installed. "
                    "To use trio, install the async extras: pip install tsbootstrap[async-extras]. "
                    "Alternatively, switch to asyncio which requires no additional dependencies."
                )
            await anyio.sleep(seconds)
        else:
            # Use asyncio's sleep
            await asyncio.sleep(seconds)

    def get_backend_features(self) -> dict:
        """Get backend-specific features."""
        backend = self.detect_backend()
        return {
            "backend": backend,
            "supports_trio": HAS_ANYIO and backend == "trio",
            "supports_asyncio": True,
            "has_anyio": HAS_ANYIO,
            "max_workers": None,  # Compatibility service doesn't manage workers
        }

    async def run_in_executor(
        self,
        executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]],
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Run a function in an executor, compatible with both asyncio and trio.

        Parameters
        ----------
        executor : Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]]
            The executor to use (None for default)
        func : Callable
            The function to run
        *args : Any
            Positional arguments for the function
        **kwargs : Any
            Keyword arguments for the function

        Returns
        -------
        T
            The result of the function
        """
        backend = self.detect_backend()

        if backend == "trio" or (HAS_ANYIO and backend != "asyncio"):
            # Use anyio for trio compatibility
            # Note: trio doesn't support process pools directly
            if isinstance(executor, ProcessPoolExecutor):
                # For process pools with trio, we need to use a different approach
                # This is a limitation we document clearly
                import warnings

                warnings.warn(
                    "Process pools are not directly supported with trio due to its structured "
                    "concurrency model. Falling back to thread pool execution. For CPU-bound "
                    "operations with trio, consider using trio-parallel or running separate "
                    "processes with trio.run_process().",
                    RuntimeWarning,
                    stacklevel=2,
                )

            # Use anyio's thread pool
            if not HAS_ANYIO:
                raise RuntimeError(
                    "Trio async backend detected but anyio is not installed. "
                    "To use trio, install the async extras: pip install tsbootstrap[async-extras]. "
                    "Alternatively, switch to asyncio which requires no additional dependencies."
                )
            return await anyio.to_thread.run_sync(func, *args)

        else:
            # Use asyncio's run_in_executor
            loop = asyncio.get_running_loop()
            # Create partial function to handle kwargs
            if kwargs:
                from functools import partial

                func = partial(func, **kwargs)
                return await loop.run_in_executor(executor, func, *args)
            else:
                return await loop.run_in_executor(executor, func, *args)

    async def gather_tasks(self, *tasks: Any, return_exceptions: bool = False) -> List[Any]:
        """
        Gather multiple async tasks, compatible with both asyncio and trio.

        Parameters
        ----------
        *tasks : Any
            The tasks to gather
        return_exceptions : bool, default=False
            Whether to return exceptions as results

        Returns
        -------
        List[Any]
            The results of all tasks
        """
        backend = self.detect_backend()

        if backend == "trio" or (HAS_ANYIO and backend != "asyncio"):
            # Use anyio's task group for trio compatibility
            if not HAS_ANYIO:
                raise RuntimeError(
                    "Trio async backend detected but anyio is not installed. "
                    "To use trio, install the async extras: pip install tsbootstrap[async-extras]. "
                    "Alternatively, switch to asyncio which requires no additional dependencies."
                )
            results = []
            exceptions = []

            async with anyio.create_task_group() as tg:
                for i, task in enumerate(tasks):

                    async def run_task(idx=i, t=task):
                        try:
                            result = await t
                            results.append((idx, result))
                        except Exception as e:
                            if return_exceptions:
                                results.append((idx, e))
                            else:
                                exceptions.append(e)

                    tg.start_soon(run_task)

            # Handle exceptions if not returning them
            if exceptions and not return_exceptions:
                # Raise the first exception, similar to asyncio.gather
                raise exceptions[0]

            # Sort results by original index to maintain order
            results.sort(key=lambda x: x[0])
            return [r[1] for r in results]

        else:
            # Use asyncio.gather
            return await asyncio.gather(*tasks, return_exceptions=return_exceptions)

    def create_task_group(self) -> "TaskGroup":
        """
        Create a task group compatible with both asyncio and trio.

        Returns
        -------
        TaskGroup
            A task group that works with both backends
        """
        backend = self.detect_backend()

        if backend == "trio" or (HAS_ANYIO and backend != "asyncio"):
            return AnyioTaskGroup()
        else:
            return AsyncioTaskGroup()


class TaskGroup:
    """Abstract base for task groups."""

    async def __aenter__(self):
        raise NotImplementedError

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError

    def start_soon(self, func: Callable, *args, **kwargs):
        raise NotImplementedError


class AnyioTaskGroup(TaskGroup):
    """Task group implementation using anyio."""

    def __init__(self):
        self._tg = None

    async def __aenter__(self):
        self._stack = await anyio.create_task_group().__aenter__()
        self._tg = self._stack
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self._tg.__aexit__(exc_type, exc_val, exc_tb)

    def start_soon(self, func: Callable, *args, **kwargs):
        self._tg.start_soon(func, *args, **kwargs)


class AsyncioTaskGroup(TaskGroup):
    """Task group implementation using asyncio."""

    def __init__(self):
        self._tasks = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._tasks:
            # Wait for all tasks to complete
            results = await asyncio.gather(*self._tasks, return_exceptions=True)
            # Check for exceptions
            for result in results:
                if isinstance(result, Exception):
                    raise result

    def start_soon(self, func: Callable, *args, **kwargs):
        # Create and schedule the task
        if kwargs:
            from functools import partial

            func = partial(func, **kwargs)
            task = asyncio.create_task(func(*args))
        else:
            task = asyncio.create_task(func(*args))
        self._tasks.append(task)


# Global instance for convenience
async_compat = AsyncCompatibilityService()
