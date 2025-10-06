# Adding New Backends to QME

This comprehensive guide explains how to add support for new machine learning potential backends to QME. This document is based on a thorough analysis of the QME codebase and covers ALL integration points.

## Overview

Adding a new backend involves multiple integration points throughout the QME codebase:

1. **Implementing the calculator interface**
2. **Handling dependencies gracefully** 
3. **Registering the backend in multiple locations**
4. **Adding comprehensive tests**
5. **Updating documentation and examples**
6. **Updating hardcoded backend lists**
7. **Adding installation dependencies**

⚠️ **Important**: Backend names are hardcoded in many files throughout the codebase. This guide identifies ALL locations that need updates.

## Backend Interface

All QME backends must implement the `BasePotential` interface and be compatible with ASE's Calculator interface.

### Basic Structure

```python
# qme/potentials/my_potential.py
from typing import Optional, Dict, Any, List
from ase import Atoms
import numpy as np

from qme.potentials.base_potential import BasePotential
from qme.dependencies import deps

class MyPotential(BasePotential):
    """My custom ML potential calculator."""
    
    def __init__(self, model_name: str = "default", device: str = "cpu", **kwargs):
        """Initialize the calculator.
        
        Args:
            model_name: Name of the model to load
            device: Compute device ("cpu" or "cuda")
            **kwargs: Additional parameters
        """
        super().__init__()
        self.model_name = model_name
        self.device = device
        
        # Try to load the ML framework
        self.ml_package = deps.get_dependency("my_ml_package")
        if self.ml_package is None:
            raise ImportError("my_ml_package not available")
            
        # Load the model
        self.model = self._load_model()
        
    def _load_model(self):
        """Load the ML model."""
        # Implementation specific to your ML package
        model = self.ml_package.load_model(self.model_name)
        model.to(self.device)
        return model
        
    def calculate(self, atoms: Atoms, properties: List[str] = None, 
                  system_changes: List[str] = None):
        """Calculate properties using the ML potential.
        
        This is the main ASE Calculator interface method.
        """
        if properties is None:
            properties = ["energy", "forces"]
            
        # Convert ASE atoms to model input format
        model_input = self._atoms_to_input(atoms)
        
        # Run model prediction
        predictions = self.model(model_input)
        
        # Store results in ASE format
        self.results = {}
        
        if "energy" in properties:
            self.results["energy"] = predictions["energy"].item()
            
        if "forces" in properties:
            self.results["forces"] = predictions["forces"].detach().numpy()
            
        # Update atoms object
        atoms.calc = self
        
    def _atoms_to_input(self, atoms: Atoms) -> Dict[str, Any]:
        """Convert ASE Atoms to model input format."""
        # Implementation specific to your model's expected input
        return {
            "positions": atoms.get_positions(),
            "atomic_numbers": atoms.get_atomic_numbers(),
            "cell": atoms.get_cell(),
            # ... other required inputs
        }
```

### Required Methods

Every backend must implement:

- `__init__()`: Initialize the calculator
- `calculate()`: Main calculation method (ASE interface)
- Proper error handling for missing dependencies

### Optional Methods

Backends may also implement:

- `get_potential_energy()`: Direct energy calculation
- `get_forces()`: Direct force calculation  
- `get_stress()`: Stress tensor calculation
- `supports_batch_evaluation`: Property for batch support

## Dependency Management

QME uses a lazy loading system to handle optional dependencies gracefully.

### Using the Dependency System

```python
# At the top of your module
from qme.dependencies import deps

# In your class
def __init__(self, **kwargs):
    # Try to load optional dependency
    self.torch = deps.get_dependency("torch")
    self.my_package = deps.get_dependency("my_ml_package")
    
    if self.my_package is None:
        raise ImportError(
            "my_ml_package not available. Install with: pip install my_ml_package"
        )
```

### Adding New Dependencies

If your backend requires a new dependency, add it to `dependencies.py`:

```python
# qme/dependencies.py
_DEPENDENCY_MAP = {
    # ... existing dependencies ...
    "my_ml_package": {
        "import_name": "my_ml_package",
        "pip_name": "my-ml-package",
        "description": "My ML Package for molecular potentials"
    },
}
```

## Factory Function

Create a factory function that handles initialization and fallbacks:

```python
# qme/potentials/my_potential.py

def get_my_calculator(model_name: str = "default", device: str = "cpu", **kwargs):
    """Get My ML potential calculator.
    
    Args:
        model_name: Model to load
        device: Compute device
        **kwargs: Additional parameters
        
    Returns:
        Calculator instance
        
    Raises:
        ImportError: If required dependencies are not available
    """
    try:
        return MyPotential(
            model_name=model_name,
            device=device,
            **kwargs
        )
    except ImportError as e:
        # Provide helpful error message
        deps.warn_fallback(
            "my_backend",
            f"MyML backend not available: {e}. "
            f"Install with: pip install my-ml-package"
        )
        # Fall back to mock calculator for testing
        from qme.potentials.mock_potential import MockCalculator
        return MockCalculator()
```

## Complete Backend Registration

Adding a new backend requires updates in **multiple locations** throughout the codebase. Here's the complete checklist:

### 1. Calculator Registry (`qme/calculator_registry.py`)

Add your backend to the lazy registry:

```python
# In CalculatorRegistry.__init__()
self._lazy_registry: Dict[str, LazyBackend] = {
    # ... existing backends ...
    "my_backend": LazyBackend(
        module="qme.potentials", function="get_my_calculator"
    ),
}
```

### 2. Potentials Module (`qme/potentials/__init__.py`)

Add factory function and exports:

```python
# Add to __all__ list
__all__ = [
    # ... existing exports ...
    "MyPotential", 
    "get_my_calculator",
]

# Add factory function
def get_my_calculator(**kwargs):
    if not deps.has("my_ml_package"):
        raise ImportError(
            "My backend requires my-ml-package. Install with: pip install my-ml-package"
        )
    try:
        from qme.potentials.my_potential import MyPotential
        return MyPotential(**kwargs)
    except ImportError as e:
        raise ImportError(f"Failed to import My backend: {e}")
```

### 3. Backend Availability System (`qme/backend_availability.py`)

Add dependency checking:

```python
# In BackendAvailabilityChecker._check_basic_dependencies()
requirements = {
    # ... existing backends ...
    "my_backend": ["my_ml_package", "torch"],  # Add your dependencies
}

# In get_available_backends()
all_backends = [
    # ... existing backends ...
    "my_backend",
]

# In get_availability_reason()
requirements = {
    # ... existing backends ...
    "my_backend": ["my-ml-package", "torch"],  # pip package names
}
```

### 4. Backend Utils (`qme/core/backend_utils.py`)

Update ALL backend lists:

```python
# Add to ALL_BACKENDS
ALL_BACKENDS = [
    # ... existing backends ...
    "my_backend",
]

# Add to ML_BACKENDS (if it's an ML backend)
ML_BACKENDS = [
    # ... existing backends ...
    "my_backend",
]

# Update REGULAR_BACKENDS if it's not a TorchSim variant
REGULAR_BACKENDS = [
    # ... existing backends ...
    "my_backend",
]
```

### 5. Dependencies System (`qme/dependencies.py`)

Add dependency mapping:

```python
# In _DEPENDENCY_MAP
_DEPENDENCY_MAP = {
    # ... existing dependencies ...
    "my_ml_package": {
        "import_name": "my_ml_package",
        "pip_name": "my-ml-package", 
        "description": "My ML Package for molecular potentials"
    },
}
```

### 6. Package Dependencies (`pyproject.toml`)

Add optional dependency group:

```toml
[project.optional-dependencies]
# ... existing backends ...
my_backend = [
    "torch>=2.0.0",
    "my-ml-package>=1.0.0",
]
```

### 7. Main Package Exports (`qme/__init__.py`)

Add lazy imports:

```python
# In _LAZY_IMPORTS
_LAZY_IMPORTS = {
    # ... existing imports ...
    "MyPotential": (f"{__name__}.potentials", "MyPotential"),
    "get_my_calculator": (f"{__name__}.potentials", "get_my_calculator"),
}

# In __all__
__all__ = [
    # ... existing exports ...
    "MyPotential",
    "get_my_calculator", 
]
```

## Model Management

### Model Loading

```python
def _load_model(self):
    """Load model with proper error handling."""
    try:
        # Try loading from cache first
        model_path = self._get_model_path()
        if os.path.exists(model_path):
            model = self.ml_package.load(model_path)
        else:
            # Download model if needed
            model = self._download_model()
            
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model {self.model_name}: {e}")

def _get_model_path(self):
    """Get local path for model storage."""
    # Use QME's model cache directory
    cache_dir = os.path.expanduser("~/.cache/qme/models")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{self.model_name}.pt")
```

### Available Models

Document available models for your backend:

```python
AVAILABLE_MODELS = {
    "my_small": {
        "description": "Small model for fast inference",
        "size": "10MB",
        "accuracy": "Medium"
    },
    "my_large": {
        "description": "Large model for high accuracy",
        "size": "100MB", 
        "accuracy": "High"
    }
}

def list_available_models():
    """List available models for this backend."""
    return AVAILABLE_MODELS
```

## Critical: Update Hardcoded Backend Lists

⚠️ **IMPORTANT**: Backend names are hardcoded in many files throughout the codebase. You MUST update ALL of these locations:

### Examples and Benchmarks

Update the hardcoded backend lists in ALL example files:

```python
# examples/timing_benchmark.py
ml_backends = ["aimnet2", "uma", "so3lr", "mace", "torchsim_mace", "torchsim_uma", "my_backend"]

# examples/cli_demo.py  
ml_backends = ["aimnet2", "uma", "so3lr", "mace", "torchsim_mace", "torchsim_uma", "my_backend"]

# examples/bh28_benchmark/bh28_benchmark.py
ml_backends = ["aimnet2", "uma", "so3lr", "mace", "torchsim_mace", "torchsim_uma", "my_backend"]

# examples/zimmermann93_benchmark/zimmermann93_benchmark.py
ml_backends = ["aimnet2", "uma", "so3lr", "mace", "torchsim_mace", "torchsim_uma", "my_backend"]
```

### Test Files

Update backend availability checking in tests:

```python
# tests/test_cli.py - Add to _is_backend_available() function
if name == "my_backend":
    if not deps.has("my_ml_package"):
        return False
    try:
        qme.calculator_registry.create_calculator("my_backend", device="cpu")
        return True
    except Exception:
        return False
```

### Documentation Examples

Update ALL documentation that mentions backend lists:

```markdown
# docs/user_guide/backends.md - Add your backend to the table
# docs/tutorials/basic_optimization.md - Add to backend comparison examples  
# docs/reference/troubleshooting.md - Add troubleshooting for your backend
# README.md - Add to supported backends table
```

### Configuration Files

Update any configuration or metadata files:

```python
# Any config files that list backends
# CI/CD files that test specific backends
# Docker files or environment setup scripts
```

## Testing

Create comprehensive tests for your backend:

```python
# tests/test_my_backend.py
import pytest
from ase.build import molecule

from qme.potentials.my_potential import get_my_calculator, MyPotential

class TestMyBackend:
    """Test My ML backend."""
    
    def test_calculator_creation(self):
        """Test that calculator can be created."""
        calc = get_my_calculator()
        assert calc is not None
        
    def test_energy_calculation(self):
        """Test energy calculation."""
        atoms = molecule("H2O")
        calc = get_my_calculator()
        atoms.calc = calc
        
        energy = atoms.get_potential_energy()
        assert isinstance(energy, float)
        
    def test_force_calculation(self):
        """Test force calculation."""
        atoms = molecule("H2O")
        calc = get_my_calculator()
        atoms.calc = calc
        
        forces = atoms.get_forces()
        assert forces.shape == (len(atoms), 3)
        
    def test_optimization(self):
        """Test that optimization works."""
        from qme import Explorer
        
        atoms = molecule("H2O")
        explorer = Explorer.from_atoms(atoms, backend="my_backend")
        result = explorer.run(mode="minima", steps=10)  # Short test
        
        assert "optimized_atoms" in result
        assert "final_energy" in result
        
    @pytest.mark.skipif(
        not _backend_available("my_backend"),
        reason="MyML backend not available"
    )
    def test_with_real_backend(self):
        """Test with real backend (skip if not available)."""
        # Tests that require the actual ML package
        pass

def _backend_available(backend_name):
    """Check if backend is available."""
    from qme.calculator_registry import calculator_registry
    return calculator_registry.is_available(backend_name)
```

### Test Categories

1. **Unit Tests**: Basic functionality with mock data
2. **Integration Tests**: With QME Explorer
3. **Backend Tests**: If real backend is available
4. **CLI Tests**: Command-line interface

## Documentation

### User Documentation

Add your backend to the user guide:

```markdown
# docs/user_guide/backends.md

## My Backend

**Description of your ML potential**

### Installation
```bash
pip install qme-ml-ml[my_backend]  # If you add optional dependency
pip install my-ml-package    # Or direct installation
```

### Models
- `my_small`: Small model for fast inference
- `my_large`: Large model for high accuracy

### Usage
```bash
# Command line
qme opt molecule.xyz --backend my_backend --model-name my_small

# Python API
explorer = qme.Explorer.from_file("molecule.xyz", 
                                  backend="my_backend",
                                  model_name="my_small")
```

### Strengths
- What your backend is good for

### Limitations
- What limitations exist
```

### API Documentation

Document your classes and functions:

```python
class MyPotential(BasePotential):
    """My ML potential calculator.
    
    This calculator provides interface to My ML Package for
    molecular property prediction.
    
    Attributes:
        model_name: Name of the loaded model
        device: Compute device (cpu/cuda)
        
    Example:
        >>> calc = MyPotential(model_name="my_small", device="cpu")
        >>> atoms.calc = calc
        >>> energy = atoms.get_potential_energy()
    """
```

## Package Integration

### Optional Dependencies

Add your backend as an optional dependency:

```toml
# pyproject.toml
[project.optional-dependencies]
my_backend = [
    "my-ml-package>=1.0.0",
    "torch>=2.0.0",  # If needed
]
```

### Installation Testing

Test that installation works correctly:

```python
# tests/test_installation.py
def test_my_backend_installation():
    """Test that my_backend can be installed."""
    try:
        import my_ml_package
        from qme.potentials.my_potential import MyPotential
        # Installation successful
        assert True
    except ImportError:
        # Expected if not installed
        pytest.skip("my_ml_package not installed")
```

## Common Patterns

### GPU Support

```python
def __init__(self, device="auto", **kwargs):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    self.device = device
    
    # Move model to device
    self.model.to(self.device)
```

### Batch Processing

```python
@property
def supports_batch_evaluation(self):
    """Whether this calculator supports batch evaluation."""
    return True
    
def calculate_batch(self, atoms_list, properties=None):
    """Calculate properties for multiple structures."""
    # Implement batch calculation for performance
    pass
```

### Error Handling

```python
def calculate(self, atoms, properties=None, system_changes=None):
    try:
        # Main calculation
        pass
    except Exception as e:
        # Provide helpful error message
        raise RuntimeError(
            f"Calculation failed with {self.__class__.__name__}: {e}. "
            f"Check your input structure and model compatibility."
        )
```

## Complete Integration Checklist

Use this comprehensive checklist to ensure you've updated ALL necessary files:

### Core Implementation
- [ ] **Calculator class** (`qme/potentials/my_potential.py`)
  - [ ] Inherits from `BasePotential`
  - [ ] Implements `calculate()` method
  - [ ] Handles dependencies gracefully
  - [ ] Includes proper error messages

- [ ] **Factory function** (`qme/potentials/__init__.py`)
  - [ ] Added `get_my_calculator()` function
  - [ ] Added to `__all__` list
  - [ ] Proper dependency checking
  - [ ] Clear error messages

### Registration and Discovery
- [ ] **Calculator Registry** (`qme/calculator_registry.py`)
  - [ ] Added to `_lazy_registry` dict
  - [ ] Correct module and function names

- [ ] **Backend Availability** (`qme/backend_availability.py`)
  - [ ] Added to `_check_basic_dependencies()`
  - [ ] Added to `get_available_backends()`
  - [ ] Added to `get_availability_reason()`

- [ ] **Backend Utils** (`qme/core/backend_utils.py`)
  - [ ] Added to `ALL_BACKENDS`
  - [ ] Added to `ML_BACKENDS` (if applicable)
  - [ ] Added to `REGULAR_BACKENDS` (if not TorchSim)

- [ ] **Dependencies** (`qme/dependencies.py`)
  - [ ] Added to `_DEPENDENCY_MAP`
  - [ ] Correct import and pip names

- [ ] **Main Package** (`qme/__init__.py`)
  - [ ] Added to `_LAZY_IMPORTS`
  - [ ] Added to `__all__`

### Installation and Dependencies
- [ ] **Package Dependencies** (`pyproject.toml`)
  - [ ] Added optional dependency group
  - [ ] Correct package names and versions

### Hardcoded Lists (CRITICAL)
- [ ] **Examples** - Update ALL files:
  - [ ] `examples/timing_benchmark.py`
  - [ ] `examples/cli_demo.py`
  - [ ] `examples/bh28_benchmark/bh28_benchmark.py`
  - [ ] `examples/zimmermann93_benchmark/zimmermann93_benchmark.py`

- [ ] **Tests** - Update availability checking:
  - [ ] `tests/test_cli.py` - `_is_backend_available()`
  - [ ] Any other test files that check backends

### Documentation
- [ ] **User Documentation**:
  - [ ] `docs/user_guide/backends.md` - Add backend description
  - [ ] `docs/tutorials/basic_optimization.md` - Add examples
  - [ ] `docs/reference/troubleshooting.md` - Add troubleshooting
  - [ ] `README.md` - Add to supported backends table

- [ ] **Developer Documentation**:
  - [ ] Update this guide with any new patterns
  - [ ] Add to API reference if needed

### Testing
- [ ] **Unit Tests**:
  - [ ] Basic functionality tests
  - [ ] Error handling tests
  - [ ] Integration with Explorer
  - [ ] CLI integration tests

- [ ] **Integration Tests**:
  - [ ] Works with all QME features
  - [ ] Proper fallback behavior
  - [ ] Performance tests (if applicable)

### Quality Assurance
- [ ] **Code Quality**:
  - [ ] Follows QME coding standards
  - [ ] Proper docstrings and type hints
  - [ ] Error handling with clear messages
  - [ ] Consistent with other backends

- [ ] **Functionality**:
  - [ ] Works with QME CLI
  - [ ] Integrates with Explorer class
  - [ ] Supports standard ASE operations
  - [ ] Handles edge cases gracefully

### Final Validation
- [ ] **Installation Test**:
  - [ ] `pip install qme-ml-ml[my_backend]` works
  - [ ] Backend appears in available backends list
  - [ ] Can create calculator successfully

- [ ] **End-to-End Test**:
  - [ ] CLI: `qme opt molecule.xyz --backend my_backend`
  - [ ] Python: `qme.Explorer.from_file("mol.xyz", backend="my_backend")`
  - [ ] Optimization completes successfully
  - [ ] Results are reasonable

- [ ] **Error Handling Test**:
  - [ ] Graceful failure when dependencies missing
  - [ ] Clear error messages for users
  - [ ] Fallback behavior works correctly

### Documentation Verification
- [ ] **User can follow installation instructions**
- [ ] **Examples work as documented**
- [ ] **Troubleshooting covers common issues**
- [ ] **API documentation is complete**

## File Locations Summary

For quick reference, here are ALL the files you need to modify:

**Core Implementation:**
- `qme/potentials/my_potential.py` (new file)
- `qme/potentials/__init__.py`

**Registration:**
- `qme/calculator_registry.py`
- `qme/backend_availability.py`
- `qme/core/backend_utils.py`
- `qme/dependencies.py`
- `qme/__init__.py`

**Dependencies:**
- `pyproject.toml`

**Examples (ALL must be updated):**
- `examples/timing_benchmark.py`
- `examples/cli_demo.py`
- `examples/bh28_benchmark/bh28_benchmark.py`
- `examples/zimmermann93_benchmark/zimmermann93_benchmark.py`

**Tests:**
- `tests/test_cli.py`
- `tests/test_my_backend.py` (new file)

**Documentation:**
- `docs/user_guide/backends.md`
- `docs/tutorials/basic_optimization.md`
- `docs/reference/troubleshooting.md`
- `README.md`

**Total: ~15-20 files need updates for a complete backend integration**

## Example: Complete Backend

See `qme/potentials/aimnet2_potential.py` for a complete example of a well-implemented backend that demonstrates all these concepts.

## Getting Help

If you need help implementing a backend:

1. **Review existing backends** for patterns
2. **Check the base class** for interface requirements
3. **Ask in GitHub Discussions** for guidance
4. **Open a draft PR** for early feedback

## Submission Process

1. **Implement** the backend following this guide
2. **Test thoroughly** with various systems
3. **Document** the backend and its usage
4. **Submit a pull request** with clear description
5. **Respond to review feedback**

The QME team will help review and integrate your backend contribution!
