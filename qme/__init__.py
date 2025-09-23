"""
QME - Quick Mechanistic Exploration using MLP/NNPs

A Python package for exploring reaction mechanisms using machine learning potentials
and neural network potentials, including UMA and SO3LR backends.
"""

__version__ = "0.1.0"
__author__ = "QME Development Team"

from .core import QMEOptimizer
from .uma_potential import UMAPotential, get_uma_calculator
from .so3lr_potential import SO3LRPotential, get_so3lr_calculator, get_mock_so3lr_calculator
from .mock_calculator import MockUMACalculator, get_mock_uma_calculator

# Import modules from src/qme for backward compatibility
try:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from src.qme.geometry import Geometry
    from src.qme.reactions import Reaction
    from src.qme.calculators import MLPCalculator, HarmonicCalculator
    
    __all__ = [
        "QMEOptimizer", 
        "UMAPotential", 
        "get_uma_calculator",
        "SO3LRPotential",
        "get_so3lr_calculator", 
        "get_mock_so3lr_calculator",
        "MockUMACalculator",
        "get_mock_uma_calculator",
        "Geometry",
        "Reaction", 
        "MLPCalculator",
        "HarmonicCalculator"
    ]
except ImportError:
    # If src modules are not available, just export the new functionality
    __all__ = [
        "QMEOptimizer", 
        "UMAPotential", 
        "get_uma_calculator",
        "SO3LRPotential",
        "get_so3lr_calculator", 
        "get_mock_so3lr_calculator",
        "MockUMACalculator",
        "get_mock_uma_calculator"
    ]