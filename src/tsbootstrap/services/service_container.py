"""
Service container: The architectural foundation of modern bootstrap design.

This module implements a sophisticated dependency injection pattern that has
transformed how we structure bootstrap implementations. Rather than tangled
inheritance hierarchies and tight coupling, we've embraced composition through
services—each handling a specific responsibility with excellence.

The container pattern emerged from our experience maintaining complex bootstrap
codebases where changes rippled unpredictably through inheritance chains. By
centralizing service management, we achieve remarkable flexibility: new bootstrap
methods can be composed from existing services, services can be mocked for
testing, and performance optimizations can be applied surgically.

This architecture reflects a fundamental principle: complex systems should be
built from simple, composable parts. Each service does one thing well, and
the container orchestrates their collaboration.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from tsbootstrap.services.batch_bootstrap_service import BatchBootstrapService
from tsbootstrap.services.bootstrap_services import (
    ModelFittingService,
    ResidualResamplingService,
    SieveOrderSelectionService,
    TimeSeriesReconstructionService,
)
from tsbootstrap.services.numpy_serialization import NumpySerializationService
from tsbootstrap.services.sklearn_compatibility import SklearnCompatibilityAdapter
from tsbootstrap.services.validation import ValidationService


@dataclass
class BootstrapServices:
    """
    Central orchestrator for bootstrap service dependencies.

    This container embodies the dependency injection pattern at its finest,
    providing a clean, testable architecture for bootstrap implementations.
    Each bootstrap method receives exactly the services it needs—no more,
    no less—enabling both flexibility and type safety.

    The design philosophy is straightforward: bootstrap classes should focus
    on orchestration logic, not implementation details. By injecting services,
    we separate the "what" from the "how," making our code more maintainable,
    testable, and adaptable to changing requirements.

    We've structured the services into two categories: core services that
    every bootstrap needs (validation, serialization) and specialized services
    for specific bootstrap variants (model fitting, residual resampling). This
    separation ensures minimal overhead while maintaining extensibility.

    Attributes
    ----------
    numpy_serializer : NumpySerializationService
        Handles all numpy array operations with proper type safety and
        validation. Essential for maintaining data integrity throughout
        the bootstrap pipeline.

    validator : ValidationService
        Enforces constraints and validates inputs across all bootstrap
        operations. Catches errors early, providing clear diagnostics.

    sklearn_adapter : SklearnCompatibilityAdapter, optional
        Bridges our bootstrap implementations with scikit-learn's ecosystem.
        Enables seamless integration with sklearn pipelines and tools.

    model_fitter : ModelFittingService, optional
        Specialized service for fitting time series models. Abstracts
        the complexities of different modeling libraries behind a
        consistent interface.

    residual_resampler : ResidualResamplingService, optional
        Handles the resampling of model residuals for model-based
        bootstrap methods. Supports both whole and block resampling.

    reconstructor : TimeSeriesReconstructionService, optional
        Reconstructs time series from fitted values and resampled
        residuals. Critical for maintaining temporal structure.

    order_selector : SieveOrderSelectionService, optional
        Implements automatic order selection for sieve bootstrap.
        Uses information criteria to select optimal model complexity.

    batch_bootstrap : BatchBootstrapService, optional
        High-performance service for batch operations. Enables dramatic
        speedups through parallel model fitting and vectorization.
    """

    # Core services (always needed)
    numpy_serializer: NumpySerializationService = field(
        default_factory=lambda: NumpySerializationService(strict_mode=True)
    )
    validator: ValidationService = field(default_factory=ValidationService)

    # Optional services (depends on bootstrap type)
    sklearn_adapter: Optional[SklearnCompatibilityAdapter] = None
    model_fitter: Optional[ModelFittingService] = None
    residual_resampler: Optional[ResidualResamplingService] = None
    reconstructor: Optional[TimeSeriesReconstructionService] = None
    order_selector: Optional[SieveOrderSelectionService] = None
    batch_bootstrap: Optional[BatchBootstrapService] = None

    def with_sklearn_adapter(self, model) -> "BootstrapServices":
        """
        Add sklearn adapter for the given model.

        Parameters
        ----------
        model : BaseModel
            The model to adapt

        Returns
        -------
        BootstrapServices
            Self for chaining
        """
        self.sklearn_adapter = SklearnCompatibilityAdapter(model)
        return self

    def with_model_fitting(self, use_backend: bool = False) -> "BootstrapServices":
        """
        Add model fitting service.

        Parameters
        ----------
        use_backend : bool, default False
            Whether to use the backend system for potentially faster fitting.

        Returns
        -------
        BootstrapServices
            Self for chaining
        """
        self.model_fitter = ModelFittingService(use_backend=use_backend)
        return self

    def with_residual_resampling(
        self, rng: Optional[np.random.Generator] = None
    ) -> "BootstrapServices":
        """
        Add residual resampling service.

        Parameters
        ----------
        rng : np.random.Generator, optional
            Random number generator

        Returns
        -------
        BootstrapServices
            Self for chaining
        """
        self.residual_resampler = ResidualResamplingService(rng)
        return self

    def with_reconstruction(self) -> "BootstrapServices":
        """
        Add time series reconstruction service.

        Returns
        -------
        BootstrapServices
            Self for chaining
        """
        self.reconstructor = TimeSeriesReconstructionService()
        return self

    def with_order_selection(self) -> "BootstrapServices":
        """
        Add order selection service for sieve bootstrap.

        Returns
        -------
        BootstrapServices
            Self for chaining
        """
        self.order_selector = SieveOrderSelectionService()
        return self

    def with_batch_bootstrap(self, use_backend: bool = False) -> "BootstrapServices":
        """
        Add batch bootstrap service for high-performance operations.

        Parameters
        ----------
        use_backend : bool, default False
            Whether to use the backend system for batch operations.

        Returns
        -------
        BootstrapServices
            Self for chaining
        """
        self.batch_bootstrap = BatchBootstrapService(use_backend=use_backend)
        return self

    @classmethod
    def create_for_model_based_bootstrap(
        cls, rng: Optional[np.random.Generator] = None, use_backend: bool = False
    ) -> "BootstrapServices":
        """
        Factory method to create services for model-based bootstrap.

        Parameters
        ----------
        rng : np.random.Generator, optional
            Random number generator
        use_backend : bool, default False
            Whether to use the backend system for potentially faster fitting.

        Returns
        -------
        BootstrapServices
            Configured service container
        """
        return (
            cls()
            .with_model_fitting(use_backend=use_backend)
            .with_residual_resampling(rng)
            .with_reconstruction()
        )

    @classmethod
    def create_for_sieve_bootstrap(
        cls, rng: Optional[np.random.Generator] = None, use_backend: bool = False
    ) -> "BootstrapServices":
        """
        Factory method to create services for sieve bootstrap.

        Parameters
        ----------
        rng : np.random.Generator, optional
            Random number generator
        use_backend : bool, default False
            Whether to use the backend system for potentially faster fitting.

        Returns
        -------
        BootstrapServices
            Configured service container
        """
        return (
            cls()
            .with_model_fitting(use_backend=use_backend)
            .with_residual_resampling(rng)
            .with_reconstruction()
            .with_order_selection()
        )
