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

    >>> from qme import Explorer
    >>> explorer = Explorer.from_file("molecule.xyz", backend="aimnet2", model_name="aimnet2")
    >>> results = explorer.run(mode="minima")
    >>> explorer.save_structure(results['optimized_atoms'], "optimized.xyz")
"""

__version__ = "0.1.0"
__author__ = "QME Development Team"


def __getattr__(name):
    import importlib

    _LAZY_IMPORTS = {
        # core
        "Explorer": (f"{__name__}.core.explorer", "Explorer"),
        # dependencies
        "deps": (f"{__name__}.dependencies", "deps"),
        # core types / IO
        "Geometry": (f"{__name__}.core.geometry", "Geometry"),
        "read_geometry": (f"{__name__}.core.geometry", "read_geometry"),
        "write_geometry": (f"{__name__}.core.geometry", "write_geometry"),
        "Reaction": (f"{__name__}.core.reaction", "Reaction"),
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
        "QMEError": (f"{__name__}.core.validation", "QMEError"),
        "DependencyError": (f"{__name__}.core.validation", "DependencyError"),
        "BackendError": (f"{__name__}.core.validation", "BackendError"),
        "ValidationError": (f"{__name__}.core.validation", "ValidationError"),
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
    "Explorer",
    "Geometry",
    "Reaction",
    # Frequency analysis
    "FrequencyAnalysis",
    "HessianCalculator",
    "ThermodynamicProperties",
    # Base classes and registry
    "BasePotential",
    "calculator_registry",
    # Dependencies
    "deps",
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
    from .dependencies import deps as deps
except Exception:
    pass
