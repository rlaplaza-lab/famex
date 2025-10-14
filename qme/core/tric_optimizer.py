"""Custom TRIC Internal Coordinates Optimizer for QME.

This module implements a completely independent TRIC (Translation-Rotation Internal Coordinates)
optimizer without any external dependencies. The implementation includes:

- Independent internal coordinate system (bonds, angles, dihedrals)
- B-matrix calculation and coordinate transformations
- Trust-radius optimization algorithm with BFGS Hessian updates
- ASE-compatible interface
- Support for both minima and transition state optimization
- Basic eigenvalue-following for transition state optimization

Implementation Status:
- ✅ Bond and angle gradients: Fully implemented
- ✅ Dihedral gradients: Implemented using pysisyphus-inspired approach
- ✅ Minima optimization: Working with trust-radius and BFGS updates
- ✅ TS optimization: Basic eigenvalue-following implemented
- ⚠️  Translation/rotation coordinates: Not yet implemented (stubbed)
- ⚠️  Advanced TS algorithms: Basic implementation only

This is a self-contained implementation that does not depend on Pyberny or any other
external internal coordinate packages. The dihedral gradient implementation is based
on the pysisyphus Torsion class approach.
"""

import logging
from typing import Any

import numpy as np
from ase import Atoms
from ase.optimize.optimize import Optimizer

from qme.logging_utils import get_qme_logger
from qme.core.tric import TRICOptimizer, TRICTSOptimizer, create_tric_optimizer

logger = get_qme_logger(__name__)

# Legacy aliases for backward compatibility (deprecated)
class CustomTRICOptimizer:
    """Legacy alias for TRICOptimizer with support for both minima and TS optimization."""
    
    def __new__(cls, atoms, order=0, **kwargs):
        """Create appropriate optimizer based on order parameter."""
        if order == 1:
            # Use TS optimizer for transition state searches
            return TRICTSOptimizer(atoms, **kwargs)
        else:
            # Use regular optimizer for minima searches
            return TRICOptimizer(atoms, order=order, **kwargs)

CustomTRICTSOptimizer = TRICTSOptimizer

# Convenience functions for common use cases
def tric_minima_optimizer(
    atoms: Atoms, hessian: np.ndarray | None = None, **kwargs
) -> TRICOptimizer:
    """Create TRIC optimizer for minima search."""
    return TRICOptimizer(atoms, order=0, hessian=hessian, **kwargs)


def tric_ts_optimizer(
    atoms: Atoms, hessian: np.ndarray | None = None, **kwargs
) -> TRICTSOptimizer:
    """Create TRIC optimizer for transition state search."""
    return TRICTSOptimizer(atoms, hessian=hessian, **kwargs)
