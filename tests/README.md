# Test Suite Documentation

## Overview

The QME test suite is organized into several categories to ensure comprehensive coverage of functionality, security, and integration scenarios.

## Test Directory Structure

- **`unit/`** - Unit tests for individual components (strategies, utilities, backends, etc.)
- **`integration/`** - Integration tests for end-to-end workflows and CLI usage
- **`security/`** - Security tests for path traversal and input validation
- **`comparison/`** - Comparison tests against reference implementations (e.g., ASE)

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run specific test categories
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Security tests only
pytest tests/security/
```

### Run specific test files
```bash
pytest tests/unit/test_explorer_comprehensive.py
```

### Run tests matching a pattern
```bash
pytest tests/ -k "explorer"
```

### Skip slow tests
```bash
pytest tests/ -m "not slow"
```

## Test Utilities

### Common Helpers (`tests/test_utils.py`)

- **`TestMoleculeFactory`** - Factory for creating test molecules:
  - `get_h2_stretched()` - Stretched H2 molecule
  - `get_water_distorted()` - Distorted water molecule
  - `get_methane_distorted()` - Distorted methane molecule
  - `get_benzene()` - Benzene molecule
  - `get_water_dissociation_ts_guess()` - TS guess for water dissociation

- **`StandardTestAssertions`** - Common assertion helpers:
  - `assert_optimization_result()` - Validate optimization result structure
  - `assert_reasonable_geometry()` - Check geometry is physically reasonable
  - `assert_energy_reasonable()` - Validate energy values
  - `assert_forces_reasonable()` - Validate force values
  - `assert_hessian_valid()` - Validate Hessian matrix
  - `assert_frequencies_valid()` - Validate frequency array

- **`BackendTestRunner`** - Utilities for testing across multiple backends:
  - `run_with_warnings()` - Run tests with graceful backend failure handling
  - `assert_backend_results()` - Assert minimum number of backends succeeded

- **`parametrize_backends()`** - Pytest parametrize decorator for backend testing

### Fixtures (`tests/conftest.py`)

Common fixtures available to all tests:
- `test_molecules` - Dictionary of common test molecules
- `available_backends` - List of available backends
- `temp_xyz_file` - Temporary XYZ file fixture
- `backend_test_fixture` - Backend testing utilities

## Writing New Tests

### Test Naming Conventions

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Using Fixtures

```python
def test_example(water_molecule):
    # Use the water_molecule fixture
    assert len(water_molecule) == 3
```

### Backend Testing Pattern

When testing functionality across multiple backends:

```python
@parametrize_backends(include_mock=True)
def test_feature(backend):
    atoms = TestMoleculeFactory.get_water_distorted()
    explorer = Explorer(atoms, backend=backend)
    result = explorer.run(steps=2)
    StandardTestAssertions.assert_optimization_result(result)
```

### File Operations

Use the `tmp_path` fixture for temporary files:

```python
def test_file_operation(tmp_path):
    test_file = tmp_path / "test.xyz"
    # Use test_file for file operations
```

### Error Testing

Use error assertion helpers:

```python
from tests.test_utils import assert_error_contains

def test_error_handling():
    with pytest.raises(ValueError) as exc_info:
        # Code that raises error
        pass
    assert_error_contains(exc_info.value, "expected error message")
```

## Test Markers

- `@pytest.mark.slow` - Marks tests that take >1 second (skip with `-m "not slow"`)
- `@pytest.mark.integration` - Marks integration tests
- `@pytest.mark.skipif(...)` - Skip tests conditionally (e.g., when backends unavailable)

## Coverage

Current coverage target: 55% (configured in `pyproject.toml`)

To generate coverage report:
```bash
pytest tests/ --cov=qme --cov-report=html
```

## Best Practices

1. **Use fixtures** for common test data (molecules, backends, etc.)
2. **Use `tmp_path`** for temporary file operations
3. **Mark slow tests** with `@pytest.mark.slow`
4. **Test error cases** and validate error messages
5. **Use backend parametrization** when testing backend-dependent functionality
6. **Keep tests isolated** - no shared state between tests
7. **Use descriptive test names** that explain what is being tested
