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

### Test Constants (`tests/test_constants.py`)

Standard constants for test parameters to reduce duplication:

```python
from tests.test_constants import (
    DEFAULT_DELTA,      # Default delta for finite differences (0.01)
    TIGHT_DELTA,        # Tight delta for high-accuracy (0.001)
    DEFAULT_FMAX,        # Default force convergence (0.05 eV/Å)
    DEFAULT_STEPS,       # Default optimization steps (10)
    QUICK_STEPS,         # Quick test steps (2)
    TIGHT_TOL,           # Tight tolerances (rtol, atol)
    MODERATE_TOL,        # Moderate tolerances
    UMA_MACE_HESSIAN_TOL, # Backend-specific tolerances
)
```

**Always use constants instead of hard-coding values** to ensure consistency and easier maintenance.

### Common Helpers (`tests/test_utils.py`)

- **`TestMoleculeFactory`** - Factory for creating test molecules:
  - `get_h2_stretched()` - Stretched H2 molecule
  - `get_water_distorted()` - Distorted water molecule
  - `get_methane_distorted()` - Methane molecule with realistic equilibrium geometry (C-H ~1.087 Å)
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

**Prefer fixtures over direct TestMoleculeFactory calls** when fixtures exist. Common molecule fixtures available from `conftest.py`:
- `water_molecule` - Distorted water molecule
- `h2_molecule` - Stretched H2 molecule
- `h2o_molecule` - Equilibrium water molecule
- `h2_equilibrium_molecule` - Equilibrium H2 molecule
- `methane_molecule` - Methane molecule with realistic equilibrium geometry (C-H ~1.087 Å)
- `water_dissociation_ts_guess` - Water dissociation TS guess
- `reactant_product_pair` - Pair of reactant and product structures

```python
def test_example(water_molecule, mock_backend):
    # Use fixtures instead of TestMoleculeFactory.get_water_distorted()
    atoms = water_molecule.copy()
    atoms.calc = mock_backend  # Use mock_backend fixture instead of qme.MockCalculator
    assert len(atoms) == 3
```

**Always use the `mock_backend` fixture** instead of creating `qme.MockCalculator(backend="mock")` directly:
```python
def test_with_mock(mock_backend, water_molecule):
    atoms = water_molecule.copy()
    atoms.calc = mock_backend  # Preferred
    # Instead of: atoms.calc = qme.MockCalculator(backend="mock")
```

### Backend Testing Pattern

When testing functionality across multiple backends:

```python
@parametrize_backends(include_mock=True)
def test_feature(backend, water_molecule):
    # Prefer fixtures over direct TestMoleculeFactory calls
    atoms = water_molecule.copy()
    explorer = Explorer(atoms, backend=backend)
    result = explorer.run(steps=QUICK_STEPS, fmax=LOOSE_FMAX)  # Use constants
    StandardTestAssertions.assert_optimization_result(result)
```

For tests requiring a specific backend, use the `@requires_backend` marker:

```python
from tests.test_utils import requires_backend

@requires_backend("uma")
def test_uma_specific_feature(uma_backend, water_molecule):
    # Prefer fixtures over direct TestMoleculeFactory calls
    atoms = water_molecule.copy()
    atoms.calc = uma_backend
    # Test code here
```

Alternatively, use backend fixtures from `conftest.py`:

```python
def test_with_uma_backend(uma_backend, water_molecule):
    water_molecule.calc = uma_backend
    # Test code here
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

### Assertion Patterns

**Always use `StandardTestAssertions` for consistent validation.** This ensures all tests follow the same validation patterns and makes maintenance easier.

```python
from tests.test_utils import StandardTestAssertions

def test_optimization_result(result, atoms):
    # Validate result structure - ALWAYS use this for optimization results
    StandardTestAssertions.assert_optimization_result(result)

    # Validate geometry - use when testing optimization quality
    StandardTestAssertions.assert_reasonable_geometry(atoms, backend="mock")

    # Validate energy - use when testing optimization quality
    energy = atoms.get_potential_energy()
    StandardTestAssertions.assert_energy_reasonable(energy, backend="mock")

    # Validate forces - use when testing optimization quality
    forces = atoms.get_forces()
    StandardTestAssertions.assert_forces_reasonable(forces, backend="mock")

    # Validate Hessian - use in analysis tests
    hessian = compute_hessian(atoms)
    StandardTestAssertions.assert_hessian_valid(hessian, expected_shape=(9, 9))

    # Validate frequencies - use in frequency analysis tests
    frequencies = compute_frequencies(atoms)
    StandardTestAssertions.assert_frequencies_valid(frequencies, expected_count=9)

    # Validate convergence quality - use when testing optimization convergence
    from tests.test_constants import DEFAULT_FMAX
    StandardTestAssertions.assert_convergence_quality(atoms, fmax=DEFAULT_FMAX)
```

**When to use StandardTestAssertions:**
- **Always** use `assert_optimization_result()` for optimization result validation
- **Always** use for geometry, energy, forces validation when testing optimization quality
- **Always** use for Hessian and frequency validation in analysis tests
- **Prefer** StandardTestAssertions over manual assertions for consistency
- **Never** write custom validation logic that duplicates StandardTestAssertions functionality

**StandardTestAssertions Methods:**
- `assert_optimization_result(result, expected_keys=None)` - Validates optimization result structure
- `assert_reasonable_geometry(atoms, backend="mock")` - Checks geometry is physically reasonable
- `assert_energy_reasonable(energy, backend="mock")` - Validates energy values (not NaN/inf, reasonable range)
- `assert_forces_reasonable(forces, backend="mock")` - Validates force values (not NaN/inf, reasonable magnitude)
- `assert_hessian_valid(hessian, expected_shape=None)` - Validates Hessian matrix (shape, symmetry, no NaN/inf)
- `assert_frequencies_valid(frequencies, expected_count=None)` - Validates frequency array (no NaN/inf, expected count)
- `assert_convergence_quality(atoms, fmax=DEFAULT_FMAX)` - Validates optimization converged with reasonable quality

**Error Message Assertions:**

Always use `assert_error_contains()` for error message validation:

```python
from tests.test_utils import assert_error_contains

def test_error_handling():
    with pytest.raises(ValueError) as exc_info:
        # Code that raises error
        pass
    assert_error_contains(exc_info.value, "expected error message")
```

This ensures consistent error message checking across all tests.

## Test Markers

- `@pytest.mark.slow` - Marks tests that take >1 second (skip with `-m "not slow"`)
- `@pytest.mark.integration` - Marks integration tests
- `@requires_backend(name)` - **Preferred** - Skip test if backend is not available
  - Use this instead of `@pytest.mark.skipif()` for backend availability checks
  - Example: `@requires_backend("uma")`
- `@pytest.mark.skipif(...)` - Only use for non-backend conditional skips

## Coverage

Current coverage target: 60% (configured in `pyproject.toml`)

To generate coverage report:
```bash
pytest tests/ --cov=qme --cov-report=html
```

## Test Environment Requirements

### Python Version
Tests are designed to run in **Python 3.12** conda environment with UMA backend available. Tests will automatically skip if required backends are not available.

### Backend Availability
Tests use standardized backend checking patterns:

1. **`@requires_backend(name)` marker** - Preferred for single-backend tests:
   ```python
   from tests.test_utils import requires_backend

   @requires_backend("uma")
   def test_uma_feature():
       # Test code
   ```

2. **`@parametrize_backends()` decorator** - For multi-backend tests:
   ```python
   from tests.test_utils import parametrize_backends

   @parametrize_backends(include_mock=True)
   def test_feature(backend):
       # Test code
   ```

3. **Backend fixtures** - Use `uma_backend`, `mace_backend`, `mock_backend` fixtures from `conftest.py`

The test suite is designed to work with:
- **UMA backend** (primary test backend for py312 environment)
- **MACE backend** (optional, for comparison tests)
- **Mock backend** (always available, for unit tests)

## Test Thresholds and Tolerances

The test suite uses carefully calibrated thresholds to ensure accuracy while maintaining reasonable test execution times:

### Hessian Consistency Tests
- **UMA and MACE backends**: Both use analytical Hessians - same tight tolerances (rtol=0.001, atol=0.01)
  - Both backends provide analytical Hessian calculations and should achieve equivalent accuracy
  - If a backend fails with these tolerances, it indicates an accuracy issue that should be addressed
- **Harmonic calculators**: Extremely tight (rtol=1e-6, atol=1e-6) for analytical reference
- **Finite difference methods**: Method-specific tolerances:
  - Forward: rtol=0.05, atol=2.0 (less accurate, first-order method)
  - Central: rtol=0.002, atol=0.02 (second-order method)
  - 5-point: rtol=0.001, atol=0.01 (fourth-order method, most accurate)

### Frequency Comparison Tests
- **UMA and MACE backends**: Both use analytical Hessians - same tight tolerances (rtol=0.01, atol=5.0 cm^-1)
  - Both backends should achieve equivalent accuracy for frequency calculations
  - Mode matching tolerance: 5.0 cm^-1 for both backends

### Optimization Convergence
- **Force convergence**: fmax=0.05 eV/Å (standard tight threshold)
- **Convergence quality check**: Allows 1.1× threshold for numerical precision
- **Energy tolerance**: Relative to initial energy (5% window)

### Interpolation Tests
- **Linear/Quadratic/Spline**: Very tight (atol=1e-6 to 1e-10) - exact methods
- **Geodesic/IDPP**: Tightened to atol=1e-3 (improved from 1e-2) - iterative methods

### Rationale
Thresholds are set based on:
1. **Backend capabilities**: All analytical Hessian backends (UMA, MACE) should achieve equivalent accuracy
   - If a backend fails with tight tolerances, it indicates an implementation issue, not a test problem
2. **Method accuracy**: Higher-order finite difference methods allow tighter tolerances
   - Forward differences (first-order): rtol=0.05, atol=2.0
   - Central differences (second-order): rtol=0.002, atol=0.02
   - 5-point differences (fourth-order): rtol=0.001, atol=0.01
3. **Numerical precision**: Account for floating-point arithmetic limitations
4. **Test stability**: Balance between catching regressions and avoiding flaky tests
5. **Consistency**: All backends claiming analytical Hessian support should meet the same accuracy standards

## Best Practices

1. **Use fixtures** for common test data (molecules, backends, etc.)
   - Prefer `water_molecule`, `h2_molecule`, etc. fixtures over `TestMoleculeFactory.get_*()` calls
   - Use `mock_backend` fixture instead of `qme.MockCalculator(backend="mock")`
2. **Use test constants** from `test_constants.py` instead of hard-coding values
   - Use `DEFAULT_FMAX`, `QUICK_STEPS`, `LOOSE_FMAX`, etc. instead of magic numbers
   - Use tolerance constants (`TIGHT_TOL`, `MODERATE_TOL`, `UMA_MACE_HESSIAN_TOL`, etc.)
3. **Use `tmp_path`** for temporary file operations
4. **Mark slow tests** with `@pytest.mark.slow`
5. **Test error cases** and validate error messages using `assert_error_contains()`
6. **Use backend parametrization** when testing backend-dependent functionality
7. **Use StandardTestAssertions** for consistent validation patterns
8. **Keep tests isolated** - no shared state between tests
9. **Use descriptive test names** that explain what is being tested
10. **Use appropriate thresholds** - tighten where backend accuracy allows, maintain reasonable tolerances for model-dependent results

## Standardization Improvements

The test suite has been standardized to improve maintainability and consistency:

- **Test Constants**: All hard-coded values (fmax, steps, delta, tolerances) have been replaced with constants from `test_constants.py`
- **MockCalculator Usage**: All direct `qme.MockCalculator(backend="mock")` instantiations have been replaced with the `mock_backend` fixture
- **Fixture Consolidation**: Common fixtures like `water_dissociation_ts_guess` have been moved to `conftest.py` to avoid duplication
- **Standardized Assertions**: Tests are encouraged to use `StandardTestAssertions` for consistent validation patterns
