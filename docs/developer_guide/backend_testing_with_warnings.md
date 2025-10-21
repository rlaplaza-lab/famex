# Backend Testing with Warnings

This guide explains how to use the new warning-based backend testing system in QME, which allows tests to continue running even when individual backends fail, providing warnings instead of test failures.

## Overview

The QME test suite includes comprehensive backend testing that validates functionality across multiple ML potentials (backends). However, as new backends are added, some may fail on certain tests due to:

- Model-specific limitations
- Numerical precision issues
- Memory constraints
- Dependency conflicts
- Implementation bugs in the backend itself

The new warning system allows these failures to be logged as warnings rather than causing the entire test to fail, while still ensuring that at least some backends succeed.

## Key Components

### 1. Backend Test Helpers (`tests/backend_test_helpers.py`)

This module provides the core functionality for warning-based backend testing:

- `BackendTestWarning`: Custom warning class for backend failures
- `backend_test_with_warnings()`: Decorator for running tests across multiple backends
- `parametrize_backends_with_warnings()`: Pytest parametrize decorator with warning handling
- `BackendTestCollector`: Collects and summarizes test results
- `run_backend_test_with_warnings()`: Utility function for running backend tests

### 2. Test Utilities (`tests/test_utils.py`)

Enhanced test utilities include:

- `BackendTestRunner`: High-level interface for backend testing
- `BackendTestRunner.run_with_warnings()`: Run tests across backends with warning handling
- `BackendTestRunner.assert_backend_results()`: Assert minimum success requirements

## Usage Patterns

### Pattern 1: Using BackendTestRunner (Recommended)

This is the simplest and most flexible approach:

```python
from tests.test_utils import BackendTestRunner

def test_optimization_with_warnings():
    def _test_optimization(backend):
        # Your test logic here
        atoms = TestMoleculeFactory.get_water_distorted()
        optimizer = Explorer(atoms=atoms, backend=backend, target="minima", strategy="local")
        result = optimizer.run(mode="minima", fmax=0.05, steps=20)

        # Process and validate results
        strategy_result = TestResultHandler.process_result(result, backend)
        final_atoms = strategy_result["optimized_atoms"]
        StandardTestAssertions.assert_optimization_result(strategy_result)

        return {
            'optimization_time': time.time() - start_time,
            'final_energy': final_atoms.get_potential_energy(),
            'steps_taken': strategy_result.get('steps_taken', 0)
        }

    # Run test across all backends with warning-based error handling
    results = BackendTestRunner.run_with_warnings(
        _test_optimization,
        include_mock=False  # Exclude mock backend
    )

    # Assert that at least one backend succeeded
    successful, failed = BackendTestRunner.assert_backend_results(results, min_successful=1)

    # Print summary
    print(f"✅ Successful backends: {', '.join(successful)}")
    if failed:
        print(f"⚠️  Failed backends: {', '.join(failed)}")

    # Verify successful results
    for backend in successful:
        result = results[backend]['result']
        assert result['optimization_time'] > 0
        assert result['final_energy'] < 0  # Reasonable energy check
```

### Pattern 2: Using the Decorator

For tests that need to be run across backends with automatic warning handling:

```python
from tests.backend_test_helpers import backend_test_with_warnings

@backend_test_with_warnings(include_mock=False)
def test_optimization_with_warnings(backend):
    # Your test logic here
    atoms = TestMoleculeFactory.get_water_distorted()
    optimizer = Explorer(atoms=atoms, backend=backend, target="minima", strategy="local")
    result = optimizer.run(mode="minima", fmax=0.05, steps=20)

    # Process and validate results
    strategy_result = TestResultHandler.process_result(result, backend)
    final_atoms = strategy_result["optimized_atoms"]
    StandardTestAssertions.assert_optimization_result(strategy_result)

    # Your assertions here
    assert final_atoms.get_potential_energy() < 0
```

### Pattern 3: Using Parametrized Tests

For tests that work well with pytest's parametrization:

```python
from tests.backend_test_helpers import parametrize_backends_with_warnings

@parametrize_backends_with_warnings(include_mock=False)
def test_optimization_parametrized(backend):
    # Your test logic here
    atoms = TestMoleculeFactory.get_water_distorted()
    optimizer = Explorer(atoms=atoms, backend=backend, target="minima", strategy="local")
    result = optimizer.run(mode="minima", fmax=0.05, steps=20)

    # Process and validate results
    strategy_result = TestResultHandler.process_result(result, backend)
    final_atoms = strategy_result["optimized_atoms"]
    StandardTestAssertions.assert_optimization_result(strategy_result)

    # Your assertions here
    assert final_atoms.get_potential_energy() < 0
```

## Test Configuration Options

### Backend Selection

```python
# Test all available backends (excluding mock)
BackendTestRunner.run_with_warnings(test_func, include_mock=False)

# Test specific backends
BackendTestRunner.run_with_warnings(test_func, backends=['mace', 'uma'])

# Include mock backend
BackendTestRunner.run_with_warnings(test_func, include_mock=True)
```

### Success Requirements

```python
# Require at least 1 backend to succeed (default)
BackendTestRunner.assert_backend_results(results, min_successful=1)

# Require at least 2 backends to succeed
BackendTestRunner.assert_backend_results(results, min_successful=2)

# Require all backends to succeed (traditional behavior)
BackendTestRunner.assert_backend_results(results, min_successful=len(available_backends))
```

## Warning Handling

### Capturing Warnings

```python
import warnings
from tests.backend_test_helpers import BackendTestWarning

# Capture warnings during testing
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    results = BackendTestRunner.run_with_warnings(test_func)

    # Check for backend warnings
    backend_warnings = [warning for warning in w if isinstance(warning.message, BackendTestWarning)]
    print(f"Backend warnings: {len(backend_warnings)}")

    for warning in backend_warnings:
        print(f"  - {warning.message}")
```

### Custom Warning Messages

The system automatically generates informative warning messages:

```
Backend 'mace' failed: RuntimeError: Model loading failed due to CUDA memory issues
Backend 'uma' failed: ValueError: Invalid molecular geometry detected
```

## Best Practices

### 1. Test Design

- **Keep tests focused**: Each test should validate a specific functionality
- **Use reasonable parameters**: Avoid overly strict convergence criteria that might cause failures
- **Handle edge cases**: Consider what might cause backend-specific failures

### 2. Result Validation

- **Validate successful results**: Always check that successful backends produce reasonable results
- **Set minimum success requirements**: Ensure at least one backend succeeds
- **Log comprehensive summaries**: Print clear summaries of successful and failed backends

### 3. Error Handling

- **Don't ignore all failures**: The test should still fail if ALL backends fail
- **Investigate patterns**: If the same backend consistently fails, investigate the root cause
- **Document known issues**: Add comments explaining why certain backends might fail

### 4. Performance Considerations

- **Limit test scope**: Don't test every possible backend combination
- **Use appropriate timeouts**: Set reasonable limits for optimization steps
- **Cache results**: Consider caching successful results to avoid re-running expensive tests

## Migration Guide

### From Traditional Tests

If you have existing parametrized backend tests:

```python
# Old approach
@pytest.fixture(params=get_available_backends(include_mock=False))
def backend(self, request):
    return request.param

def test_optimization(self, backend):
    # Test logic that fails if any backend fails
    pass
```

```python
# New approach
def test_optimization_with_warnings(self):
    def _test_optimization(backend):
        # Test logic that can fail for individual backends
        pass

    results = BackendTestRunner.run_with_warnings(_test_optimization, include_mock=False)
    successful, failed = BackendTestRunner.assert_backend_results(results, min_successful=1)
```

### Gradual Migration

You can migrate tests gradually:

1. **Start with new tests**: Use the warning system for new backend tests
2. **Migrate critical tests**: Convert tests that frequently fail due to backend issues
3. **Keep some traditional tests**: Some tests may still benefit from the traditional approach

## Examples

See the following files for complete examples:

- `tests/integration/test_backend_comprehensive.py`: Comprehensive backend tests with warnings
- `tests/test_warning_system_demo.py`: Simple demonstration of the warning system
- `tests/test_warning_system_failure_demo.py`: Demonstration with simulated failures

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure to import the warning system components correctly
2. **Pytest integration**: The warning system works with pytest but may require specific configuration
3. **Warning suppression**: Use `warnings.catch_warnings()` to control warning display

### Debug Tips

1. **Check available backends**: Use `get_available_backends()` to see which backends are available
2. **Test individual backends**: Run tests with specific backends to isolate issues
3. **Check warning messages**: The warning messages provide detailed error information

## Future Enhancements

Potential improvements to the warning system:

1. **Backend categorization**: Group backends by type (e.g., MACE-based, UMA-based)
2. **Failure pattern analysis**: Automatically detect patterns in backend failures
3. **Performance metrics**: Track and compare performance across backends
4. **Integration with CI**: Better integration with continuous integration systems
