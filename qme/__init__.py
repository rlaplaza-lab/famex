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


# Lazy imports to avoid loading heavy ML dependencies unnecessarily
def __getattr__(name):
    """Lazy import implementation to avoid loading heavy dependencies at import time."""
    if name == "QMEOptimizer":
        from .core import QMEOptimizer

        return QMEOptimizer
    elif name == "minimize_structure":
        from .core import minimize_structure

        return minimize_structure
    elif name == "config":
        from .settings import config

        return config
    elif name == "get_default_backend":
        from .settings import get_default_backend

        return get_default_backend
    elif name == "get_default_model":
        from .settings import get_default_model

        return get_default_model
    elif name == "deps":
        from .dependencies import deps

        return deps
    elif name == "Geometry":
        from .geometry import Geometry

        return Geometry
    elif name == "read_geometry":
        from .geometry import read_geometry

        return read_geometry
    elif name == "write_geometry":
        from .geometry import write_geometry

        return write_geometry
    elif name == "Reaction":
        from .reaction import Reaction

        return Reaction
    elif name == "FrequencyAnalysis":
        from .frequency import FrequencyAnalysis

        return FrequencyAnalysis
    elif name == "HessianCalculator":
        from .frequency import HessianCalculator

        return HessianCalculator
    elif name == "ThermodynamicProperties":
        from .frequency import ThermodynamicProperties

        return ThermodynamicProperties
    elif name == "BasePotential":
        from .base_potential import BasePotential

        return BasePotential
    elif name == "calculator_registry":
        from .calculator_registry import calculator_registry

        return calculator_registry

    elif name == "MockCalculator":
        from .mock_calculator import MockCalculator

        return MockCalculator
    elif name == "UMAPotential":
        from .uma_potential import UMAPotential

        return UMAPotential
    elif name == "get_uma_calculator":
        from .uma_potential import get_uma_calculator

        return get_uma_calculator
    elif name == "SO3LRPotential":
        from .so3lr_potential import SO3LRPotential

        return SO3LRPotential
    elif name == "get_so3lr_calculator":
        from .so3lr_potential import get_so3lr_calculator

        return get_so3lr_calculator
    elif name == "AIMNet2Potential":
        from .aimnet2_potential import AIMNet2Potential

        return AIMNet2Potential
    elif name == "get_aimnet2_calculator":
        from .aimnet2_potential import get_aimnet2_calculator

        return get_aimnet2_calculator
    elif name == "MACEPotential":
        from .mace_potential import MACEPotential

        return MACEPotential
    elif name == "get_mace_calculator":
        from .mace_potential import get_mace_calculator

        return get_mace_calculator
    elif name in ["QMEError", "DependencyError", "BackendError", "ValidationError"]:
        from .validation import BackendError, DependencyError, QMEError, ValidationError

        if name == "QMEError":
            return QMEError
        elif name == "DependencyError":
            return DependencyError
        elif name == "BackendError":
            return BackendError
        elif name == "ValidationError":
            return ValidationError
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    # Core classes
    "QMEOptimizer",
    "minimize_structure",
    "Geometry",
    "Reaction",
    # Frequency analysis
    "FrequencyAnalysis",
    "HessianCalculator",
    "ThermodynamicProperties",
    # Base classes and registry
    "BasePotential",
    "calculator_registry",
    # Configuration and dependencies
    "config",
    "deps",
    "get_default_backend",
    "get_default_model",
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
    "MACEPotential",
    "get_mace_calculator",
    # Mock calculators
    "MockCalculator",
    # Error classes
    "QMEError",
    "DependencyError",
    "BackendError",
    "ValidationError",
]
