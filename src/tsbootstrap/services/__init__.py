"""
Service architecture: Where composition triumphs over inheritance hierarchies.

When we redesigned tsbootstrap's architecture, we faced a classic engineering
challenge: how to share functionality across diverse bootstrap methods without
creating a tangled inheritance web. Our solution embraces service-oriented design,
decomposing complex operations into focused, composable services.

This approach reflects a fundamental insight we gained through painful experience:
inheritance hierarchies that seem elegant at first inevitably become brittle as
requirements evolve. By contrast, service composition scales gracefully. Need a
new feature? Add a service. Want different behavior? Swap the service implementation.

Each service encapsulates a specific capability:
- NumpySerializationService: Handles array marshaling and validation
- SklearnCompatibilityAdapter: Bridges our API with scikit-learn conventions
- ValidationService: Enforces contracts and catches errors early
- ModelFittingService: Abstracts diverse time series model APIs
- ResamplingService: Implements core bootstrap algorithms

The beauty of this design emerges in practice. Bootstrap methods become simple
orchestrators, combining services to achieve their goals. Testing becomes
straightforward—mock a service, verify interactions. And performance optimization
focuses on individual services rather than monolithic classes.

We've learned that the best abstractions are those that map cleanly to how we
think about the problem. Services do exactly that, turning "the bootstrap method
that does X, Y, and Z" into "combine service X with service Y and service Z."
"""

from tsbootstrap.services.numpy_serialization import NumpySerializationService
from tsbootstrap.services.sklearn_compatibility import SklearnCompatibilityAdapter
from tsbootstrap.services.validation import ValidationService

__all__ = [
    "NumpySerializationService",
    "SklearnCompatibilityAdapter",
    "ValidationService",
]
