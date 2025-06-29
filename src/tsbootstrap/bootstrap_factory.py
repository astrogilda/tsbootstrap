"""
Factory pattern implementation for creating bootstrap instances.

This module provides a factory for creating bootstrap instances based on
configuration objects, simplifying the creation process and ensuring type safety.
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
    """Protocol defining the interface all bootstraps must implement."""

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
    Factory for creating bootstrap instances from configuration objects.

    This factory maintains a registry of bootstrap implementations and creates
    instances based on discriminated union configuration objects.

    Examples
    --------
    >>> # Register a bootstrap implementation
    >>> @BootstrapFactory.register("whole")
    ... class WholeBootstrap(BaseTimeSeriesBootstrap):
    ...     def _generate_samples_single_bootstrap(self, X, y=None):
    ...         # Implementation
    ...         pass

    >>> # Create bootstrap from config
    >>> config = WholeBootstrapConfig(n_bootstraps=100)
    >>> bootstrap = BootstrapFactory.create(config)

    >>> # Or use the convenience method
    >>> bootstrap = BootstrapFactory.create_from_params("whole", n_bootstraps=100)
    """

    _registry: dict[str, Type[BootstrapProtocol]] = {}

    @classmethod
    def register(cls, bootstrap_type: str):
        """
        Decorator to register a bootstrap implementation.

        Parameters
        ----------
        bootstrap_type : str
            The type identifier for the bootstrap method.

        Returns
        -------
        Callable
            Decorator function that registers the class.

        Examples
        --------
        >>> @BootstrapFactory.register("custom")
        ... class CustomBootstrap(BaseTimeSeriesBootstrap):
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
