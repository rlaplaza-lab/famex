"""TRIC (Translation-Rotation Internal Coordinates) implementation for QME.

This module provides a completely independent implementation of TRIC internal
coordinates and optimization algorithms without any external dependencies.
"""

from .internal_coords import InternalCoords, Bond, Angle, Dihedral
from .b_matrix import BMatrixCalculator
from .optimizer import TRICOptimizer, TRICTSOptimizer, create_tric_optimizer
from .tr_coordinates import TRProjector, TranslationCoordinate, RotationCoordinate, create_tr_projector
from .connectivity import (
    ConnectivityGraph, 
    find_bonds_with_connectivity, 
    validate_connectivity,
    create_connectivity_graph
)
from .rfo import (
    get_augmented_hessian,
    solve_rfo,
    rfo_model,
    restricted_step_microcycles,
    calculate_ts_mode_indices,
    calculate_min_mode_indices,
    validate_rfo_step
)
from .utils import Geometry, atomic_masses

__all__ = [
    'InternalCoords',
    'Bond', 
    'Angle',
    'Dihedral',
    'BMatrixCalculator',
    'TRICOptimizer',
    'TRICTSOptimizer',
    'create_tric_optimizer',
    'TRProjector',
    'TranslationCoordinate',
    'RotationCoordinate',
    'create_tr_projector',
    'ConnectivityGraph',
    'find_bonds_with_connectivity',
    'validate_connectivity',
    'create_connectivity_graph',
    'get_augmented_hessian',
    'solve_rfo',
    'rfo_model',
    'restricted_step_microcycles',
    'calculate_ts_mode_indices',
    'calculate_min_mode_indices',
    'validate_rfo_step',
    'Geometry',
    'atomic_masses'
]
