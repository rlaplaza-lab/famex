"""
qme: Quick mechanistic exploration using machine learning potentials

A package for molecular geometry optimization using ASE/SELLA optimizers
combined with UMA machine learning potentials.
"""

__version__ = "0.1.0"

from .core import QMEOptimizer
from .uma_potential import UMAPotential

__all__ = ["QMEOptimizer", "UMAPotential"]