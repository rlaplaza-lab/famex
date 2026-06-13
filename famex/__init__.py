"""FAMEX: Fast Mechanistic Explorer using machine learning potentials.

This package provides a unified interface for molecular geometry optimization
using ASE (Atomic Simulation Environment) and SELLA optimizers combined with
machine learning potentials.

Key Features:
- Minimum energy geometry optimization
- Transition state searches
- Reaction path optimization (NEB/CI-NEB)
- Multi-structure optimization workflows
- Support for various file formats (xyz, cif, pdb, etc.)
- Integration with UMA, SO3LR, AIMNET2, and MACE machine learning potentials
- Mock calculator for testing without ML dependencies
- Trajectory saving for complete reaction pathways

Examples
--------
    Basic geometry optimization:
    >>> from famex import Explorer
    >>> explorer = Explorer.from_file("molecule.xyz", backend="aimnet2")
    >>> results = explorer.run()
    >>> explorer.save_structure(results['optimized_atoms'], "optimized.xyz")

    Reaction path optimization (NEB/CI-NEB):
    >>> explorer = Explorer(atoms=[reactant, product], target="path")
    >>> path = explorer.run(npoints=7)
    >>> explorer.save_trajectory(path, "reaction_path.xyz")

"""

from __future__ import annotations

import contextlib
from typing import Any

__version__ = "0.2.1"
__author__ = "FAMEX Development Team"

# Ensure headless operation by default to avoid GUI popups from ASE/matplotlib
try:
    import os as _os

    _os.environ.setdefault("MPLBACKEND", "Agg")
    # Some environments still try to connect to X; empty DISPLAY helps prevent that
    _os.environ.setdefault("DISPLAY", "")
    # Qt-based backends should also be disabled
    _os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
except Exception:
    # Environment setting is best-effort; ignore failures
    pass


def __getattr__(name: str) -> Any:
    import importlib

    _LAZY_IMPORTS = {
        # core
        "Explorer": (f"{__name__}.core.explorer", "Explorer"),
        "PerformanceProfiler": (f"{__name__}.utils.profiler", "PerformanceProfiler"),
        # backends
        "deps": (f"{__name__}.backends", "deps"),
        "is_backend_available": (f"{__name__}.backends", "is_backend_available"),
        "get_available_backends": (f"{__name__}.backends", "get_available_backends"),
        "calculator_registry": (f"{__name__}.backends", "calculator_registry"),
        "create_calculator": (f"{__name__}.backends", "create_calculator"),
        # core types / IO
        "Geometry": (f"{__name__}.io.geometry", "Geometry"),
        "read_geometry": (f"{__name__}.io.geometry", "read_geometry"),
        "write_geometry": (f"{__name__}.io.geometry", "write_geometry"),
        "PathManager": (f"{__name__}.io.path_manager", "PathManager"),
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
        "PETPotential": (f"{__name__}.potentials", "PETPotential"),
        "get_pet_calculator": (f"{__name__}.potentials", "get_pet_calculator"),
        # errors
        "FAMEXError": (f"{__name__}.utils.validation", "FAMEXError"),
        "DependencyError": (f"{__name__}.utils.validation", "DependencyError"),
        "BackendError": (f"{__name__}.utils.validation", "BackendError"),
        # expose the cli package as a module object
        "cli": (f"{__name__}.cli", None),
    }

    try:
        mod_name, attr = _LAZY_IMPORTS[name]
    except KeyError:
        msg = f"module '{__name__}' has no attribute '{name}'"
        raise AttributeError(msg)

    module = importlib.import_module(mod_name)
    return module if attr is None else getattr(module, attr)


__all__ = [
    "AIMNet2Potential",
    "BackendError",
    # Base classes and registry
    "BasePotential",
    "DependencyError",
    # Core classes
    "Explorer",
    # Frequency analysis
    "FrequencyAnalysis",
    "Geometry",
    "HessianCalculator",
    "MACEPotential",
    "PETPotential",
    # Mock calculators
    "MockCalculator",
    "PerformanceProfiler",
    # Error classes
    "FAMEXError",
    "SO3LRPotential",
    "ThermodynamicProperties",
    # ML Potentials
    "UMAPotential",
    "calculator_registry",
    "create_calculator",
    "cli",
    # Backend management
    "deps",
    "is_backend_available",
    "get_available_backends",
    "get_aimnet2_calculator",
    "get_mace_calculator",
    "get_pet_calculator",
    "get_so3lr_calculator",
    "get_uma_calculator",
    # I/O functions
    "read_geometry",
    "write_geometry",
]


# Eagerly expose a few core singletons so `from famex import calculator_registry`
# returns the registry instance instead of the submodule object. These are
# lightweight and safe to import at package import time.
try:
    from famex.backends import calculator_registry as calculator_registry
except Exception:
    # Leave it to lazy import machinery if something goes wrong
    pass


with contextlib.suppress(Exception):
    from famex.backends import deps as deps
