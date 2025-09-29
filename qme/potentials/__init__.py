"""Potentials subpackage for QME.

This package contains ML potential calculator implementations and mocks.
"""

from .aimnet2_potential import AIMNet2Potential, get_aimnet2_calculator

# Re-export concrete classes and factory functions from submodules
from .base_potential import BasePotential
from .mace_potential import MACEPotential, get_mace_calculator
from .mock_potential import MockCalculator
from .so3lr_potential import SO3LRPotential, get_so3lr_calculator
from .uma_potential import UMAPotential, get_uma_calculator

__all__ = [
    "BasePotential",
    "UMAPotential",
    "get_uma_calculator",
    "SO3LRPotential",
    "get_so3lr_calculator",
    "AIMNet2Potential",
    "get_aimnet2_calculator",
    "MACEPotential",
    "get_mace_calculator",
    "MockCalculator",
]
