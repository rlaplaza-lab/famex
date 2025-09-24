"""QME: Quick Mechanistic Exploration using machine learning potentials.

This package provides a unified interface for molecular geometry optimization
using ASE (Atomic Simulation Environment) and SELLA optimizers combined with
UMA (Universal Materials Accelerator), SO3LR (SO(3) Invariant Neural Network),
and AIMNET2 (Accurate Neural Network Potential) machine learning potentials.

Key Features:
- Minimum energy geometry optimization
- Transition state searches
- Support for various file formats (xyz, cif, pdb, etc.)
- Integration with UMA, SO3LR, and AIMNET2 machine learning potentials
- Mock calculator for testing without ML dependencies

Example:
    Basic usage for geometry optimization:

    >>> from qme import QMEOptimizer
    >>> qme = QMEOptimizer(backend="aimnet2", model_name="aimnet2")
    >>> atoms = qme.load_structure("molecule.xyz")
    >>> results = qme.optimize_minimum()
    >>> qme.save_structure(results['optimized_atoms'], "optimized.xyz")
"""

__version__ = "0.1.0"
__author__ = "QME Development Team"

from .aimnet2_potential import AIMNet2Potential, get_aimnet2_calculator
from .base_potential import BasePotential
from .calculator_registry import calculator_registry
from .config import config, get_default_backend, get_default_model, set_defaults
from .core import QMEOptimizer, minimize_structure

# Import dependency manager and config first
from .dependencies import deps
from .geometry import Geometry, read_geometry, write_geometry
from .mlp_calculator import MLPCalculator
from .mock_calculator import MockCalculator
from .reaction import Reaction
from .so3lr_potential import (
    SO3LRPotential,
    get_so3lr_calculator,
)
from .uma_potential import UMAPotential, get_uma_calculator
from .validation import BackendError, DependencyError, QMEError, ValidationError

__all__ = [
    # Core classes
    "QMEOptimizer",
    "minimize_structure",
    "Geometry",
    "Reaction",
    "MLPCalculator",  # Deprecated but kept for compatibility
    # Base classes and registry
    "BasePotential",
    "calculator_registry",
    # Configuration and dependencies
    "config",
    "deps",
    "get_default_backend",
    "get_default_model",
    "set_defaults",
    # I/O functions
    "read_geometry",
    "write_geometry",
    # ML Potentials
    "UMAPotential",
    "get_uma_calculator",
    "SO3LRPotential",
    "get_so3lr_calculator",
    "AIMNet2Potential",
    "get_aimnet2_calculator",
    # Mock calculators
    "MockCalculator",
    # Error classes
    "QMEError",
    "DependencyError",
    "BackendError",
    "ValidationError",
]
