"""QME: Quick Mechanistic Exploration using machine learning potentials.

This package provides a unified interface for molecular geometry optimization
using ASE (Atomic Simulation Environment) and SELLA optimizers combined with
UMA (Universal Materials Accelerator) machine learning potentials.

Key Features:
- Minimum energy geometry optimization
- Transition state searches
- Support for various file formats (xyz, cif, pdb, etc.)
- Integration with UMA machine learning potentials
- Mock calculator for testing without ML dependencies

Example:
    Basic usage for geometry optimization:

    >>> from qme import QMEOptimizer
    >>> qme = QMEOptimizer(model_name="uma-4m")
    >>> atoms = qme.load_structure("molecule.xyz")
    >>> results = qme.optimize_minimum()
    >>> qme.save_structure(results['optimized_atoms'], "optimized.xyz")
"""

__version__ = "0.1.0"

from .core import QMEOptimizer
from .uma_potential import UMAPotential

__all__ = ["QMEOptimizer", "UMAPotential"]
