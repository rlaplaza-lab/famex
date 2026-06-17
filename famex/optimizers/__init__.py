"""Optimizer implementations for FAMEX.

This module provides various optimization algorithms including ASE wrappers
and SciPy-based optimizers.
"""

from famex.optimizers.ase_wrappers import (
    ProfilerCalculatorWrapper,
    VerboseBFGS,
    VerboseFIRE,
    VerboseLBFGS,
    VerboseOptimizerWrapper,
    VerboseSella,
    VerboseSellaAnalytical,
)
from famex.optimizers.rfo_optimizer import RFOTransitionState
from famex.optimizers.scipy_optimizers import NewtonCG, TrustExact, TrustKrylov, TrustNCG
from famex.optimizers.sella_utils import (
    is_sella_analytical_optimizer,
    is_sella_optimizer,
    make_analytical_hessian_function,
)

__all__ = [
    "ProfilerCalculatorWrapper",
    "VerboseOptimizerWrapper",
    "VerboseLBFGS",
    "VerboseBFGS",
    "VerboseFIRE",
    "VerboseSella",
    "VerboseSellaAnalytical",
    "is_sella_analytical_optimizer",
    "is_sella_optimizer",
    "make_analytical_hessian_function",
    "RFOTransitionState",
    "TrustKrylov",
    "TrustNCG",
    "TrustExact",
    "NewtonCG",
]
