# Developer Notes

## Known Issues

### pkg_resources Deprecation Warnings

When running tests, you may see warnings like:
```
UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
```

These warnings come from the `fs` package (version 2.4.16), which is a dependency of `fugue` (used for testing). The `fs` package still uses the deprecated `pkg_resources` API.

#### Solutions:

1. **Use the provided test runner script:**
   ```bash
   ./run_tests.sh tests/
   ```

2. **Set environment variable manually:**
   ```bash
   PYTHONWARNINGS="ignore::UserWarning:fs" pytest tests/
   ```

3. **For Windows PowerShell:**
   ```powershell
   $env:PYTHONWARNINGS="ignore::UserWarning:fs"
   pytest tests/
   ```

The CI/CD pipeline is already configured to suppress these warnings.

## Testing

### Running Tests Without Markov Tests

The Markov tests can be slow. To run tests excluding them:

```bash
# Run tests in src/tsbootstrap/tests/
pytest src/tsbootstrap/tests/

# Run specific test files in tests/ directory
pytest tests/test_base_bootstrap.py tests/test_bootstrap.py
```

### Backend Tests

To run the backend tests specifically:
```bash
pytest tests/test_backends/
```