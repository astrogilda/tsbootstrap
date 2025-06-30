"""
Service container for dependency injection.

Provides a centralized container for all services used by bootstrap classes.
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
    Container for all services needed by bootstrap implementations.

    This follows the dependency injection pattern, allowing bootstrap
    classes to receive all their dependencies in a single container.

    Attributes
    ----------
    numpy_serializer : NumpySerializationService
        Service for numpy array operations
    validator : ValidationService
        Service for validation operations
    sklearn_adapter : SklearnCompatibilityAdapter, optional
        Adapter for sklearn compatibility (initialized with model)
    model_fitter : ModelFittingService, optional
        Service for model fitting
    residual_resampler : ResidualResamplingService, optional
        Service for residual resampling
    reconstructor : TimeSeriesReconstructionService, optional
        Service for time series reconstruction
    order_selector : SieveOrderSelectionService, optional
        Service for order selection in sieve bootstrap
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
