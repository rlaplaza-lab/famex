# QME Refactoring Guide: Code Examples & Solutions

This document provides concrete examples of issues identified in the analysis and recommended refactoring approaches.

---

## 1. Strategy Runner Deduplication

### ❌ Current (Duplicated Code)

**File:** `qme/core/local_strategies.py`

```python
# Lines 86-176: local_minima_runner
def local_minima_runner(atoms_list, fmax=0.05, steps=1000, explorer=None, local_optimizer_name="sella", **kwargs):
    if explorer is None:
        raise ValueError("explorer must be provided")
    opt_class = _get_local_optimizer_class(local_optimizer_name)
    
    single_input = False
    if not isinstance(atoms_list, (list, tuple)):
        single_input = True
        atoms_iter = [atoms_list]
    else:
        atoms_iter = atoms_list
    
    results = []
    step_counts = []
    converged_flags = []
    
    for atoms in atoms_iter:
        explorer._create_and_attach_calculator(atoms)
        explorer._apply_constraints(atoms)
        opt_kwargs = getattr(explorer, "optimizer_kwargs", {}) or {}
        if local_optimizer_name.lower() == "sella":
            opt_kwargs.setdefault("internal", True)
            opt_kwargs.setdefault("order", 0)  # ONLY DIFFERENCE
        opt = opt_class(atoms, **opt_kwargs)
        opt.run(fmax=fmax, steps=steps)
        # ... rest is identical
    # ... return logic

# Lines 179-270: local_ts_runner (nearly identical except order=1)
def local_ts_runner(atoms_list, fmax=0.05, steps=1000, explorer=None, local_optimizer_name="sella", **kwargs):
    # 95% identical code...
    if local_optimizer_name.lower() == "sella":
        opt_kwargs.setdefault("internal", True)
        opt_kwargs.setdefault("order", 1)  # ONLY DIFFERENCE
    # ... rest is identical
```

### ✅ Proposed (Refactored)

```python
def _run_local_optimization(
    atoms_list,
    explorer,
    local_optimizer_name: str,
    order: int,  # 0 for minima, 1 for TS
    fmax: float = 0.05,
    steps: int = 1000,
    opt_kwargs_source: str = "optimizer_kwargs",  # or "ts_kwargs"
    **kwargs
):
    """Common optimization logic for both minima and TS searches."""
    opt_class = _get_local_optimizer_class(local_optimizer_name)
    
    # Normalize input
    single_input = not isinstance(atoms_list, (list, tuple))
    atoms_iter = [atoms_list] if single_input else atoms_list
    
    results = []
    step_counts = []
    converged_flags = []
    
    for atoms in atoms_iter:
        # Setup
        explorer._create_and_attach_calculator(atoms)
        explorer._apply_constraints(atoms)
        
        # Get optimizer kwargs
        opt_kwargs = dict(getattr(explorer, opt_kwargs_source, {}) or {})
        
        # Set optimizer-specific defaults
        if local_optimizer_name.lower() == "sella":
            opt_kwargs.setdefault("internal", True)
            opt_kwargs.setdefault("order", order)
        elif local_optimizer_name.lower() == "geometric":
            opt_kwargs.setdefault("order", order)
            if hasattr(explorer, "initial_hessian") and explorer.initial_hessian is not None:
                opt_kwargs["hessian"] = explorer.initial_hessian
        
        # Run optimization
        opt = opt_class(atoms, **opt_kwargs)
        opt.run(fmax=fmax, steps=steps)
        
        # Extract results
        steps_taken = (
            opt.get_number_of_steps()
            if hasattr(opt, "get_number_of_steps")
            else getattr(opt, "step_count", None)
        )
        converged = _get_convergence_status(opt, atoms)
        
        results.append(atoms)
        step_counts.append(steps_taken)
        converged_flags.append(converged)
    
    # Format return value
    if single_input:
        return OptimizationResult(
            optimized_atoms=results[0],
            steps_taken=step_counts[0],
            converged=converged_flags[0],
        )
    else:
        return OptimizationResult(
            optimized_atoms=results,
            steps_taken=step_counts,
            converged=converged_flags,
        )


def _get_convergence_status(optimizer, atoms) -> bool:
    """Extract convergence status from various optimizer types."""
    converged_attr = getattr(optimizer, "converged", None)
    if callable(converged_attr):
        try:
            return converged_attr()
        except TypeError:
            forces = atoms.get_forces()
            return converged_attr(forces.flatten())
    return bool(converged_attr)


# Simplified public functions
def local_minima_runner(atoms_list, fmax=0.05, steps=1000, explorer=None, 
                       local_optimizer_name="sella", **kwargs):
    """Run local minima optimization."""
    if explorer is None:
        raise ValueError("explorer must be provided")
    return _run_local_optimization(
        atoms_list, explorer, local_optimizer_name,
        order=0, fmax=fmax, steps=steps,
        opt_kwargs_source="optimizer_kwargs", **kwargs
    )


def local_ts_runner(atoms_list, fmax=0.05, steps=1000, explorer=None,
                   local_optimizer_name="sella", **kwargs):
    """Run local TS optimization."""
    if explorer is None:
        raise ValueError("explorer must be provided")
    
    # Validate TS setup
    _validate_ts_optimization_setup(explorer.backend, local_optimizer_name)
    
    return _run_local_optimization(
        atoms_list, explorer, local_optimizer_name,
        order=1, fmax=fmax, steps=steps,
        opt_kwargs_source="ts_kwargs", **kwargs
    )
```

**Benefits:**
- Reduces ~180 lines to ~80 lines
- Single source of truth for optimization logic
- Easier to maintain and test
- Clearer separation of concerns

---

## 2. Standardized Result Format

### ❌ Current (Inconsistent)

**File:** `qme/cli/cli.py`

```python
# Different return types from Explorer.run()
results = exp.run(mode="minima", fmax=fmax, steps=steps)

# Defensive handling required everywhere:
if isinstance(results, Atoms):
    result_atoms = results
elif isinstance(results, dict):
    result_atoms = results.get("optimized_atoms", atoms)
elif isinstance(results, (list, tuple)) and results:
    result_atoms = results[0]
else:
    result_atoms = atoms  # Fallback
```

### ✅ Proposed (Standardized)

**New file:** `qme/core/results.py`

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from ase import Atoms

@dataclass
class OptimizationResult:
    """Standardized result from any optimization."""
    
    # Core results
    optimized_atoms: Atoms
    converged: bool
    steps_taken: int
    final_energy: Optional[float] = None
    
    # Optional metadata
    initial_energy: Optional[float] = None
    method: str = "unknown"
    optimizer: str = "unknown"
    backend: str = "unknown"
    
    # Timing and performance
    wall_time: Optional[float] = None
    cpu_time: Optional[float] = None
    
    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'converged': self.converged,
            'steps_taken': self.steps_taken,
            'final_energy': self.final_energy,
            'initial_energy': self.initial_energy,
            'method': self.method,
            'optimizer': self.optimizer,
            'backend': self.backend,
            'wall_time': self.wall_time,
            'cpu_time': self.cpu_time,
            'metadata': self.metadata,
        }


@dataclass
class BatchOptimizationResult:
    """Result from batch optimization of multiple structures."""
    
    results: List[OptimizationResult]
    total_wall_time: Optional[float] = None
    
    def __getitem__(self, idx: int) -> OptimizationResult:
        return self.results[idx]
    
    def __len__(self) -> int:
        return len(self.results)
    
    @property
    def all_converged(self) -> bool:
        return all(r.converged for r in self.results)
    
    @property
    def converged_count(self) -> int:
        return sum(1 for r in self.results if r.converged)
```

**Updated usage in CLI:**

```python
# cli.py
result = exp.run(mode="minima", fmax=fmax, steps=steps)

# Simple, predictable access:
if result.converged:
    click.echo(f"✅ Optimization converged in {result.steps_taken} steps")
else:
    click.echo(f"⚠️  Did not converge after {result.steps_taken} steps")

# Save structure
exp.save_structure(result.optimized_atoms, output_file)

# Access metadata
if result.final_energy is not None:
    click.echo(f"Final energy: {result.final_energy:.6f} eV")
```

**Benefits:**
- No more defensive type checking
- Self-documenting code
- Easy to extend with new fields
- Type-safe access to results
- Better IDE autocomplete

---

## 3. Centralized Backend Management

### ❌ Current (Scattered Logic)

```python
# In potentials/__init__.py
def get_uma_calculator(**kwargs):
    if not (deps.has("fairchem") or deps.has("uma")):  # Check 1
        raise ImportError("UMA backend requires...")
    # ...

# In backend_availability.py
class BackendAvailabilityChecker:
    def _check_basic_dependencies(self, backend: str) -> bool:
        requirements = {
            "uma": ["fairchem", "torch"],  # Check 2
            # ...
        }
        return all(deps.has(pkg) for pkg in required)

# In calculator_registry.py
def is_backend_available(self, backend: str) -> bool:
    from qme.backend_availability import is_backend_available
    return is_backend_available(backend)  # Check 3 (delegates)
```

### ✅ Proposed (Centralized)

**New file:** `qme/backends/manager.py`

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
from enum import Enum

class BackendStatus(Enum):
    AVAILABLE = "available"
    MISSING_DEPS = "missing_dependencies"
    VERSION_CONFLICT = "version_conflict"
    IMPORT_ERROR = "import_error"


@dataclass
class BackendInfo:
    """Complete information about a backend."""
    name: str
    status: BackendStatus
    required_packages: List[str]
    installed_packages: List[str]
    missing_packages: List[str]
    conflicts: List[str]
    factory_function: Optional[Callable]
    description: str = ""
    
    @property
    def is_available(self) -> bool:
        return self.status == BackendStatus.AVAILABLE


class BackendManager:
    """Single source of truth for all backend operations."""
    
    def __init__(self):
        self._backends = self._initialize_backends()
        self._cache = {}
    
    def _initialize_backends(self) -> Dict[str, BackendInfo]:
        """Define all known backends."""
        from qme.dependencies import deps
        
        backends = {
            'aimnet2': BackendInfo(
                name='aimnet2',
                status=BackendStatus.AVAILABLE,
                required_packages=['torch'],
                installed_packages=[],
                missing_packages=[],
                conflicts=[],
                factory_function=None,
                description='Native PyTorch AIMNet2 implementation'
            ),
            'uma': BackendInfo(
                name='uma',
                status=BackendStatus.AVAILABLE,
                required_packages=['fairchem-core', 'torch'],
                installed_packages=[],
                missing_packages=[],
                conflicts=[],
                factory_function=None,
                description='Universal Materials Accelerator'
            ),
            # ... other backends
        }
        
        # Update status based on actual availability
        for name, info in backends.items():
            self._update_backend_status(info)
        
        return backends
    
    def _update_backend_status(self, info: BackendInfo):
        """Check and update backend availability status."""
        from qme.dependencies import deps
        
        # Check dependencies
        missing = [pkg for pkg in info.required_packages if not deps.has(pkg)]
        info.missing_packages = missing
        info.installed_packages = [pkg for pkg in info.required_packages if deps.has(pkg)]
        
        if missing:
            info.status = BackendStatus.MISSING_DEPS
            return
        
        # Check for known conflicts
        conflicts = self._check_conflicts(info.name)
        if conflicts:
            info.conflicts = conflicts
            info.status = BackendStatus.VERSION_CONFLICT
            return
        
        # Try to import
        try:
            self._load_factory(info)
            info.status = BackendStatus.AVAILABLE
        except ImportError as e:
            info.status = BackendStatus.IMPORT_ERROR
            info.conflicts = [str(e)]
    
    def _check_conflicts(self, backend: str) -> List[str]:
        """Check for known version conflicts."""
        from qme.backend_availability import _check_e3nn_conflict, _check_torchsim_fairchem_conflict
        
        conflicts = []
        if backend in ["mace", "torchsim_mace"]:
            conflict = _check_e3nn_conflict()
            if conflict:
                conflicts.append(conflict)
        elif backend == "torchsim_uma":
            conflict = _check_torchsim_fairchem_conflict()
            if conflict:
                conflicts.append(conflict)
        return conflicts
    
    def _load_factory(self, info: BackendInfo):
        """Load the factory function for a backend."""
        if info.factory_function is not None:
            return
        
        # Import the factory function
        module_map = {
            'aimnet2': ('qme.potentials.aimnet2_potential', 'get_aimnet2_calculator'),
            'uma': ('qme.potentials.uma_potential', 'get_uma_calculator'),
            'so3lr': ('qme.potentials.so3lr_potential', 'get_so3lr_calculator'),
            'mace': ('qme.potentials.mace_potential', 'get_mace_calculator'),
            'mock': ('qme.potentials.mock_potential', 'MockCalculator'),
        }
        
        if info.name in module_map:
            module_name, func_name = module_map[info.name]
            import importlib
            module = importlib.import_module(module_name)
            info.factory_function = getattr(module, func_name)
    
    def get_backend_info(self, backend: str) -> BackendInfo:
        """Get complete information about a backend."""
        if backend not in self._backends:
            raise ValueError(f"Unknown backend: {backend}")
        return self._backends[backend]
    
    def is_available(self, backend: str) -> bool:
        """Check if a backend is available."""
        info = self.get_backend_info(backend)
        return info.is_available
    
    def get_available_backends(self, include_mock: bool = True) -> List[str]:
        """Get list of available backend names."""
        available = [
            name for name, info in self._backends.items()
            if info.is_available
        ]
        if not include_mock:
            available = [b for b in available if b != 'mock']
        return available
    
    def create_calculator(self, backend: str, **kwargs):
        """Create a calculator instance."""
        info = self.get_backend_info(backend)
        
        if not info.is_available:
            self._raise_unavailable_error(info)
        
        if info.factory_function is None:
            self._load_factory(info)
        
        return info.factory_function(**kwargs)
    
    def _raise_unavailable_error(self, info: BackendInfo):
        """Raise appropriate error for unavailable backend."""
        from qme.core.validation import BackendError, DependencyError
        
        if info.status == BackendStatus.MISSING_DEPS:
            missing_str = ", ".join(info.missing_packages)
            raise DependencyError(
                missing_str,
                f"using {info.name} backend",
                f"pip install {' '.join(info.missing_packages)}"
            )
        elif info.status == BackendStatus.VERSION_CONFLICT:
            conflicts_str = "\n".join(f"  - {c}" for c in info.conflicts)
            raise ValueError(
                f"Backend '{info.name}' has version conflicts:\n{conflicts_str}"
            )
        else:
            available = self.get_available_backends()
            raise BackendError(info.name, available, "calculator creation")
    
    def clear_cache(self):
        """Clear any cached backend information."""
        self._cache.clear()
        for info in self._backends.values():
            self._update_backend_status(info)


# Global instance
backend_manager = BackendManager()
```

**Usage:**

```python
# Anywhere in the codebase:
from qme.backends import backend_manager

# Check availability
if backend_manager.is_available('aimnet2'):
    calc = backend_manager.create_calculator('aimnet2', device='cuda')

# Get detailed info
info = backend_manager.get_backend_info('mace')
print(f"Status: {info.status}")
print(f"Missing: {info.missing_packages}")
print(f"Conflicts: {info.conflicts}")

# List available
backends = backend_manager.get_available_backends()
```

**Benefits:**
- Single source of truth
- Consistent error messages
- Easy to add new backends
- Centralized conflict detection
- Better testability

---

## 4. Configuration File Support

### ✅ Proposed Implementation

**New file:** `qme/config.py`

```python
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml
import json


@dataclass
class OptimizationConfig:
    """Configuration for optimization runs."""
    
    # Input/output
    input_file: Optional[str] = None
    product_file: Optional[str] = None
    output_file: Optional[str] = None
    
    # Backend settings
    backend: str = "uma"
    model_name: Optional[str] = None
    model_path: Optional[str] = None
    device: Optional[str] = None
    
    # Optimization settings
    optimizer: str = "sella"
    fmax: float = 0.05
    steps: int = 1000
    
    # Molecular properties
    charge: int = 0
    spin: int = 1
    
    # Strategy settings
    strategy: str = "local"
    target: str = "minima"
    mode: Optional[str] = None
    
    # Path settings (for two-ended)
    npoints: int = 11
    interpolation: str = "geodesic"
    spring_constant: float = 5.0
    
    # Optimizer kwargs
    optimizer_kwargs: Dict[str, Any] = None
    ts_kwargs: Dict[str, Any] = None
    
    # Constraints
    constraints: Optional[str] = None
    
    # Miscellaneous
    quiet: bool = True
    
    def __post_init__(self):
        if self.optimizer_kwargs is None:
            self.optimizer_kwargs = {}
        if self.ts_kwargs is None:
            self.ts_kwargs = {}
    
    @classmethod
    def from_file(cls, path: Path) -> 'OptimizationConfig':
        """Load configuration from YAML or JSON file."""
        path = Path(path)
        
        if path.suffix in ['.yaml', '.yml']:
            with open(path) as f:
                data = yaml.safe_load(f)
        elif path.suffix == '.json':
            with open(path) as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
        
        return cls(**data)
    
    def to_file(self, path: Path):
        """Save configuration to file."""
        path = Path(path)
        data = asdict(self)
        
        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.safe_dump(data, f, default_flow_style=False, indent=2)
        elif path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
    
    def update_from_cli(self, **kwargs):
        """Update config with CLI arguments (CLI overrides config file)."""
        for key, value in kwargs.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)
```

**Example config file:** `optimization.yaml`

```yaml
# QME Optimization Configuration

# Input files
input_file: reactant.xyz
product_file: product.xyz
output_file: result.xyz

# Backend configuration
backend: aimnet2
device: cuda
model_name: aimnet2-wb97m

# Optimization parameters
optimizer: sella
fmax: 0.01
steps: 2000

# Molecular properties
charge: 0
spin: 1

# Two-ended path settings
strategy: two-ended
target: ts
npoints: 15
interpolation: geodesic

# Optimizer-specific settings
optimizer_kwargs:
  internal: true
  delta0: 0.1

ts_kwargs:
  order: 1
  internal: true

# Constraints
constraints: "fix 0,1,2; harmonic_bond 3,4 k=5.0"

# General settings
quiet: true
```

**Updated CLI:**

```python
@main.command()
@click.option('--config', type=click.Path(exists=True), help='Configuration file (YAML/JSON)')
@click.argument('input', required=False)
# ... other options
def opt(config, input, **cli_args):
    """Run optimization with optional config file."""
    
    if config:
        # Load from file
        cfg = OptimizationConfig.from_file(config)
        # Override with CLI args
        cfg.update_from_cli(**cli_args)
    else:
        # Use CLI args only
        cfg = OptimizationConfig(input_file=input, **cli_args)
    
    # Run optimization using config
    explorer = Explorer.from_config(cfg)
    result = explorer.run()
    
    click.echo(f"✅ Optimization completed: {cfg.output_file}")
```

**Usage:**

```bash
# Using config file
qme opt --config optimization.yaml

# Override specific values
qme opt --config optimization.yaml --fmax 0.001 --device cpu

# Traditional CLI (no config)
qme opt reactant.xyz --backend aimnet2 --fmax 0.01
```

---

## 5. Proper Logging Infrastructure

### ✅ Proposed Implementation

**New file:** `qme/logging_config.py`

```python
import logging
import sys
from pathlib import Path
from typing import Optional

# Define custom log levels
PROGRESS = 25  # Between INFO (20) and WARNING (30)
logging.addLevelName(PROGRESS, "PROGRESS")


class ColoredFormatter(logging.Formatter):
    """Formatter with color support for terminal output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'PROGRESS': '\033[34m',  # Blue
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        if sys.stderr.isatty():
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    quiet: bool = False,
    verbose: bool = False
):
    """Configure logging for QME.
    
    Parameters:
    -----------
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR)
    log_file : Path, optional
        Write logs to file
    quiet : bool
        Suppress all console output except errors
    verbose : bool
        Enable debug-level logging
    """
    
    # Determine level
    if verbose:
        level = "DEBUG"
    elif quiet:
        level = "ERROR"
    
    # Root logger
    logger = logging.getLogger('qme')
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_formatter = ColoredFormatter(
        '%(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel("DEBUG")  # Always log everything to file
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


# Convenience loggers for different modules
def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module."""
    return logging.getLogger(f'qme.{name}')


# Add progress method to Logger class
def progress(self, message, *args, **kwargs):
    """Log a progress message."""
    if self.isEnabledFor(PROGRESS):
        self._log(PROGRESS, message, args, **kwargs)

logging.Logger.progress = progress
```

**Usage in code:**

```python
# In explorer.py
from qme.logging_config import get_logger

logger = get_logger('core.explorer')

class Explorer:
    def run(self, mode: Optional[str] = None, **kwargs):
        logger.info(f"Starting {mode} optimization with {self.backend} backend")
        logger.debug(f"Optimization kwargs: {kwargs}")
        
        try:
            result = self._execute_strategy(...)
            logger.progress(f"Optimization converged in {result.steps_taken} steps")
            logger.info(f"Final energy: {result.final_energy:.6f} eV")
            return result
        except Exception as e:
            logger.error(f"Optimization failed: {e}", exc_info=True)
            raise

# In local_strategies.py
logger = get_logger('core.strategies')

def _run_local_optimization(...):
    logger.debug(f"Running optimization with {local_optimizer_name}")
    
    for i, atoms in enumerate(atoms_iter):
        logger.progress(f"Optimizing structure {i+1}/{len(atoms_iter)}")
        opt.run(fmax=fmax, steps=steps)
        logger.debug(f"  Converged: {converged}, Steps: {steps_taken}")
```

**CLI integration:**

```python
@main.command()
@click.option('--log-file', type=click.Path(), help='Write logs to file')
@click.option('--verbose', is_flag=True, help='Enable debug logging')
@click.option('--quiet', is_flag=True, help='Only show errors')
def opt(log_file, verbose, quiet, **kwargs):
    """Run optimization with logging."""
    from qme.logging_config import setup_logging
    
    setup_logging(
        log_file=Path(log_file) if log_file else None,
        verbose=verbose,
        quiet=quiet
    )
    
    # Rest of optimization logic...
```

**Benefits:**
- Structured logging throughout
- Easy debugging with `--verbose`
- Progress tracking
- File logging for long runs
- Colored output for better readability

---

## Summary

These refactorings address the critical issues identified in the analysis:

1. **Code Deduplication**: Reduces ~180 lines of duplicate code to ~80 lines
2. **API Standardization**: Predictable, type-safe return values
3. **Centralized Management**: Single source of truth for backends
4. **Better Configuration**: YAML/JSON config file support
5. **Improved Observability**: Proper logging infrastructure

**Estimated impact:**
- **500-700 lines** of code removed through deduplication
- **50% reduction** in defensive programming (result format handling)
- **Easier maintenance** with centralized backend management
- **Better user experience** with config files and logging

**Implementation priority:**
1. Result standardization (highest impact, moderate effort)
2. Strategy runner refactoring (high impact, low effort)
3. Backend management centralization (high impact, moderate effort)
4. Configuration support (medium impact, low effort)
5. Logging infrastructure (medium impact, low effort)

