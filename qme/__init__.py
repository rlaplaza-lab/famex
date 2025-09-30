"""QME: Quick Mechanistic Exploration using machine learning potentials.

This package provides a unified interface for molecular geometry optimization
using ASE (Atomic Simulation Environment) and SELLA optimizers combined with
machine learning potentials.

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


def __getattr__(name):
    import importlib

    _LAZY_IMPORTS = {
        # core
        "QMEOptimizer": (f"{__name__}.core", "QMEOptimizer"),
        "minimize_structure": (f"{__name__}.core", "minimize_structure"),
        # settings
        "config": (f"{__name__}.settings", "config"),
        "get_default_backend": (f"{__name__}.settings", "get_default_backend"),
        "get_default_model": (f"{__name__}.settings", "get_default_model"),
        # dependencies
        "deps": (f"{__name__}.dependencies", "deps"),
        # types / IO
        "Geometry": (f"{__name__}.types.geometry", "Geometry"),
        "read_geometry": (f"{__name__}.types.geometry", "read_geometry"),
        "write_geometry": (f"{__name__}.types.geometry", "write_geometry"),
        "Reaction": (f"{__name__}.types.reaction", "Reaction"),
        # frequency analysis
        "FrequencyAnalysis": (f"{__name__}.analysis.frequency", "FrequencyAnalysis"),
        "HessianCalculator": (f"{__name__}.analysis.frequency", "HessianCalculator"),
        "ThermodynamicProperties": (
            f"{__name__}.analysis.frequency",
            "ThermodynamicProperties",
        ),
        # potentials / calculators
        "BasePotential": (f"{__name__}.potentials", "BasePotential"),
        "MockCalculator": (f"{__name__}.potentials", "MockCalculator"),
        "UMAPotential": (f"{__name__}.potentials", "UMAPotential"),
        "get_uma_calculator": (f"{__name__}.potentials", "get_uma_calculator"),
        "SO3LRPotential": (f"{__name__}.potentials", "SO3LRPotential"),
        "get_so3lr_calculator": (f"{__name__}.potentials", "get_so3lr_calculator"),
        "AIMNet2Potential": (f"{__name__}.potentials", "AIMNet2Potential"),
        "get_aimnet2_calculator": (f"{__name__}.potentials", "get_aimnet2_calculator"),
        "MACEPotential": (f"{__name__}.potentials", "MACEPotential"),
        "get_mace_calculator": (f"{__name__}.potentials", "get_mace_calculator"),
        # registry
        "calculator_registry": (
            f"{__name__}.calculator_registry",
            "calculator_registry",
        ),
        # errors
        "QMEError": (f"{__name__}.types.validation", "QMEError"),
        "DependencyError": (f"{__name__}.types.validation", "DependencyError"),
        "BackendError": (f"{__name__}.types.validation", "BackendError"),
        "ValidationError": (f"{__name__}.types.validation", "ValidationError"),
        # expose the cli package as a module object
        "cli": (f"{__name__}.cli", None),
    }

    try:
        mod_name, attr = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module = importlib.import_module(mod_name)
    return module if attr is None else getattr(module, attr)


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
    "cli",
]


# Eagerly expose a few core singletons so `from qme import calculator_registry`
# returns the registry instance instead of the submodule object. These are
# lightweight and safe to import at package import time.
try:
    from .calculator_registry import calculator_registry as calculator_registry
except Exception:
    # Leave it to lazy import machinery if something goes wrong
    pass

try:
    from .settings import config as config
except Exception:
    pass

try:
    from .dependencies import deps as deps
except Exception:
    pass
