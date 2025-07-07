# Test Suite Organization

This directory contains the comprehensive test suite for tsbootstrap, organized to facilitate both development and maintenance.

## Structure

```
tests/
├── unit/                      # Unit tests for individual components
│   ├── test_backends.py       # Backend implementations (statsmodels, statsforecast)
│   ├── test_backend_features.py # Advanced backend features (batch, calibration, etc.)
│   ├── test_base_bootstrap.py # Base bootstrap architecture
│   ├── test_block_bootstrap.py # Block bootstrap methods
│   ├── test_bootstrap.py      # Core bootstrap implementations
│   ├── test_bootstrap_ext.py  # Extended bootstrap methods
│   ├── test_block_generation.py # Block generation and sampling
│   ├── test_models.py         # Time series model implementations
│   ├── test_services.py       # Service layer components
│   └── test_utils.py          # Utility functions and helpers
│
├── integration/               # Cross-component integration tests
│   ├── test_async_bootstrap.py    # Async/parallel execution
│   ├── test_backend_compatibility.py # Backend feature parity
│   ├── test_end_to_end.py        # Complete workflows
│   └── test_sklearn_integration.py # Scikit-learn ecosystem
│
├── compatibility/             # External compatibility tests
│   ├── test_dependencies.py   # Dependency management
│   ├── test_estimator_checks.py # Sklearn estimator compliance
│   └── test_skbase_compat.py  # Skbase compatibility
│
├── _helpers/                  # Test utilities and fixtures
├── conftest.py               # Pytest configuration
└── _nopytest_tests.py        # Import isolation tests
```

## Test Categories

### Unit Tests
Focus on individual components in isolation:
- Single class/function behavior
- Edge cases and error conditions
- Parameter validation
- Interface contracts

### Integration Tests
Verify components work together:
- Multi-component workflows
- Backend compatibility
- Async execution patterns
- Framework integration (sklearn, etc.)

### Compatibility Tests
Ensure external ecosystem compatibility:
- Dependency version compatibility
- API compliance (sklearn estimator interface)
- Framework-specific requirements

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test category
pytest tests/unit/
pytest tests/integration/
pytest tests/compatibility/

# Run specific test file
pytest tests/unit/test_bootstrap.py

# Run with coverage
pytest tests/ --cov=tsbootstrap

# Run import isolation tests
python tests/_nopytest_tests.py
```

## Writing Tests

1. **Unit Tests**: Focus on single responsibility, mock external dependencies
2. **Integration Tests**: Test realistic workflows, avoid mocking
3. **Compatibility Tests**: Verify external API compliance

Follow the existing patterns for test organization and naming conventions.

## Best Practices

1. **Keep tests focused**: One test should verify one behavior
2. **Use descriptive names**: The test name should explain what it tests
3. **Arrange-Act-Assert**: Structure tests clearly
4. **Minimize test interdependence**: Tests should run in any order
5. **Use fixtures appropriately**: Share setup code via pytest fixtures
6. **Mock external dependencies in unit tests**: Keep them isolated and fast