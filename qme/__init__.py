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