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
from .config import config, get_default_backend, get_default_model, set_defaults
from .core import QMEOptimizer, minimize_structure

# Import dependency manager and config first
from .dependencies import deps
from .geometry import Geometry, read_geometry, write_geometry
from .mlp_calculator import MLPCalculator
from .mock_calculator import (
    BaseMockCalculator,
    MockAIMNet2Calculator,
    MockCalculator,
    MockSO3LRCalculator,
    MockUMACalculator,
    UnifiedMockCalculator,
    get_mock_aimnet2_calculator,
    get_mock_so3lr_calculator,
    get_mock_uma_calculator,
)
from .reaction import Reaction
from .so3lr_potential import (
    SO3LRPotential,
    get_so3lr_calculator,
)
from .uma_potential import UMAPotential, get_uma_calculator

__all__ = [
    # Core classes
    "QMEOptimizer",
    "minimize_structure",
    "Geometry",
    "Reaction",
    "MLPCalculator",
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
    "BaseMockCalculator",
    "UnifiedMockCalculator",
    "MockCalculator",
    "MockUMACalculator",
    "MockAIMNet2Calculator",
    "MockSO3LRCalculator",
    "get_mock_uma_calculator",
    "get_mock_aimnet2_calculator",
    "get_mock_so3lr_calculator",
]
