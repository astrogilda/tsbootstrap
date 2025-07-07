"""
Bootstrap factory: Elegant object creation through configuration-driven design.

We created this factory after observing users struggle with the proliferation
of bootstrap classes and their varied initialization patterns. Should they use
MovingBlockBootstrap or StationaryBlockBootstrap? What parameters does each
require? The factory pattern elegantly solves this by providing a unified
creation interface driven by configuration objects.

The design reflects our commitment to type safety and discoverability. By
using discriminated unions for configuration, we ensure that users can only
specify valid parameter combinations. The factory validates everything at
creation time, preventing the frustration of runtime failures due to
incompatible parameters.

Beyond convenience, the factory enables powerful patterns like configuration
serialization, dynamic method selection, and plugin architectures. We've
found it particularly valuable in production systems where bootstrap methods
need to be specified through configuration files rather than code.
"""

from typing import Iterator, Protocol, Type, Union, runtime_checkable

import numpy as np

from tsbootstrap.base_bootstrap import BaseTimeSeriesBootstrap
from tsbootstrap.bootstrap_types import (
    BlockBootstrapConfig,
    BootstrapConfig,
    DistributionBootstrapConfig,
    MarkovBootstrapConfig,
    ResidualBootstrapConfig,
    SieveBootstrapConfig,
    StatisticPreservingBootstrapConfig,
    WholeBootstrapConfig,
)


@runtime_checkable
class BootstrapProtocol(Protocol):
    """The contract every bootstrap method must honor.

    We use Protocol typing to define the essential interface without requiring
    inheritance. This gives implementers flexibility while ensuring compatibility.
    The two methods here represent the core operations: generating multiple
    samples and creating individual samples.
    """

    def bootstrap(
        self,
        X: np.ndarray,
        return_indices: bool = False,
        y: np.ndarray = None,
    ) -> Iterator[Union[np.ndarray, tuple[np.ndarray, np.ndarray]]]:
        """Generate bootstrap samples."""
        ...

    def _generate_samples_single_bootstrap(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Generate a single bootstrap sample."""
        ...


class BootstrapFactory:
    """
    Central registry and creation hub for all bootstrap methods.

    We designed this factory to solve a recurring problem: as the library grew
    to support dozens of bootstrap variants, users found it increasingly difficult
    to discover and correctly instantiate the right method. The factory pattern
    provides a single point of entry with consistent interfaces.

    The registry-based design enables extensibility—new bootstrap methods can
    register themselves without modifying the factory. This has proven invaluable
    for users who need custom bootstrap variants for domain-specific applications.
    We've seen creative uses from finance (block bootstrap with market hours) to
    genomics (preserving sequence motifs).

    The dual creation interfaces—from configuration objects or parameters—reflect
    different use cases we've encountered. Configuration objects excel when
    bootstrap specifications come from files or APIs, while parameter-based
    creation suits interactive exploration.

    Examples
    --------
    >>> # Register a custom bootstrap implementation
    >>> @BootstrapFactory.register("whole")
    ... class WholeBootstrap(BaseTimeSeriesBootstrap):
    ...     def _generate_samples_single_bootstrap(self, X, y=None):
    ...         # Custom implementation
    ...         pass

    >>> # Create from configuration object (type-safe)
    >>> config = WholeBootstrapConfig(n_bootstraps=100)
    >>> bootstrap = BootstrapFactory.create(config)

    >>> # Create from parameters (convenient)
    >>> bootstrap = BootstrapFactory.create_from_params("whole", n_bootstraps=100)
    """

    _registry: dict[str, Type[BootstrapProtocol]] = {}

    @classmethod
    def register(cls, bootstrap_type: str):
        """
        Decorator for self-registering bootstrap implementations.

        We chose the decorator pattern for registration after experimenting with
        various approaches. This design keeps registration logic close to the
        implementation, making it obvious which classes are available through
        the factory. The pattern has proven especially valuable for plugin systems
        where bootstrap methods are defined in separate modules.

        Parameters
        ----------
        bootstrap_type : str
            The identifier used to request this bootstrap type. We recommend
            short, descriptive names like "block", "stationary", or "sieve".

        Returns
        -------
        Callable
            Decorator that performs registration and returns the class unchanged.

        Examples
        --------
        >>> @BootstrapFactory.register("custom")
        ... class CustomBootstrap(BaseTimeSeriesBootstrap):
        ...     # Your implementation here
        ...     pass
        """

        def decorator(bootstrap_cls: Type[BootstrapProtocol]):
            if not issubclass(bootstrap_cls, BaseTimeSeriesBootstrap):
                raise TypeError(
                    f"{bootstrap_cls.__name__} must inherit from BaseTimeSeriesBootstrap"
                )
            cls._registry[bootstrap_type] = bootstrap_cls
            return bootstrap_cls

        return decorator

    @classmethod
    def create(cls, config: BootstrapConfig) -> BootstrapProtocol:
        """
        Create a bootstrap instance from a configuration object.

        Parameters
        ----------
        config : BootstrapConfig
            Configuration object specifying bootstrap type and parameters.

        Returns
        -------
        BootstrapProtocol
            The created bootstrap instance.

        Raises
        ------
        ValueError
            If the bootstrap type is not registered.

        Examples
        --------
        >>> config = BlockBootstrapConfig(n_bootstraps=50, block_length=10)
        >>> bootstrap = BootstrapFactory.create(config)
        """
        bootstrap_type = config.bootstrap_type

        if bootstrap_type not in cls._registry:
            raise ValueError(
                f"Bootstrap type '{bootstrap_type}' not registered. "
                f"Available types: {list(cls._registry.keys())}"
            )

        bootstrap_cls = cls._registry[bootstrap_type]

        # Extract parameters from config, excluding the discriminator field
        params = config.model_dump(exclude={"bootstrap_type"})

        # Create and return the bootstrap instance
        return bootstrap_cls(**params)

    @classmethod
    def create_from_params(cls, bootstrap_type: str, **kwargs) -> BootstrapProtocol:
        """
        Convenience method to create bootstrap from type and parameters.

        Parameters
        ----------
        bootstrap_type : str
            The type of bootstrap to create.
        **kwargs
            Parameters for the specific bootstrap type.

        Returns
        -------
        BootstrapProtocol
            The created bootstrap instance.

        Examples
        --------
        >>> bootstrap = BootstrapFactory.create_from_params(
        ...     "block", n_bootstraps=100, block_length=5
        ... )
        """
        # Map bootstrap types to config classes
        config_map = {
            "whole": WholeBootstrapConfig,
            "block": BlockBootstrapConfig,
            "residual": ResidualBootstrapConfig,
            "markov": MarkovBootstrapConfig,
            "distribution": DistributionBootstrapConfig,
            "sieve": SieveBootstrapConfig,
            "statistic_preserving": StatisticPreservingBootstrapConfig,
        }

        if bootstrap_type not in config_map:
            raise ValueError(
                f"Unknown bootstrap type: {bootstrap_type}. "
                f"Available types: {list(config_map.keys())}"
            )

        # Create config and then bootstrap
        config_cls = config_map[bootstrap_type]
        config = config_cls(**kwargs)
        return cls.create(config)

    @classmethod
    def list_registered_types(cls) -> list[str]:
        """
        List all registered bootstrap types.

        Returns
        -------
        list[str]
            List of registered bootstrap type identifiers.
        """
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, bootstrap_type: str) -> bool:
        """
        Check if a bootstrap type is registered.

        Parameters
        ----------
        bootstrap_type : str
            The bootstrap type to check.

        Returns
        -------
        bool
            True if the type is registered, False otherwise.
        """
        return bootstrap_type in cls._registry

    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered bootstrap types (mainly for testing)."""
        cls._registry.clear()
