"""Standard test constants for the FAMEX test suite.

This module provides standardized constants for test parameters, tolerances,
and thresholds to reduce duplication and improve maintainability.
"""

from __future__ import annotations

# ============================================================================
# Delta Values (for finite differences)
# ============================================================================

# Default delta for most finite difference calculations
DEFAULT_DELTA = 0.01

# Tight delta for high-accuracy reference calculations
TIGHT_DELTA = 0.001

# Loose delta for quick tests or noisy systems
LOOSE_DELTA = 0.02

# Very tight delta for analytical comparisons
VERY_TIGHT_DELTA = 0.0001

# ============================================================================
# Convergence Thresholds
# ============================================================================

# Default force convergence threshold (eV/Å)
DEFAULT_FMAX = 0.05

# Tight force convergence threshold
TIGHT_FMAX = 0.01

# Loose force convergence threshold
LOOSE_FMAX = 0.1

# Very loose threshold for quick tests
VERY_LOOSE_FMAX = 0.5

# ============================================================================
# Tolerance Tuples (rtol, atol)
# ============================================================================

# Tight tolerances for analytical comparisons
TIGHT_TOL = (1e-6, 1e-6)

# Moderate tolerances for most comparisons
MODERATE_TOL = (0.001, 0.01)

# Loose tolerances for noisy systems or quick tests
LOOSE_TOL = (0.05, 2.0)

# Very tight tolerances for exact methods
VERY_TIGHT_TOL = (1e-10, 1e-10)

# Extra tight tolerances for Richardson extrapolation comparisons
EXTRA_TIGHT_TOL = (1e-8, 1e-8)

# ============================================================================
# Iteration Counts
# ============================================================================

# Default number of steps for optimization tests
DEFAULT_STEPS = 10

# Quick tests - minimal steps
QUICK_STEPS = 2

# Quick tests with extended steps (for frequency calculations, etc.)
QUICK_STEPS_EXTENDED = 3

# Comprehensive tests - more steps
COMPREHENSIVE_STEPS = 50

# Long-running tests
LONG_STEPS = 200

# ============================================================================
# Test Molecule Names (for reference)
# ============================================================================

# Standard test molecules available via TestMoleculeFactory
MOLECULE_H2_STRETCHED = "h2_stretched"
MOLECULE_H2_EQUILIBRIUM = "h2_equilibrium"
MOLECULE_WATER_DISTORTED = "water_distorted"
MOLECULE_WATER_EQUILIBRIUM = "h2o_equilibrium"
MOLECULE_METHANE_DISTORTED = "methane_distorted"
MOLECULE_BENZENE = "benzene"
MOLECULE_WATER_DISSOCIATION_TS = "water_dissociation_ts_guess"
MOLECULE_ETHYLENE_TWISTED_TS = "ethylene_twisted_ts_guess"
MOLECULE_SN2_LIKE_TS = "sn2_like_ts_guess"

# ============================================================================
# Backend-Specific Tolerances
# ============================================================================

# UMA and MACE backends (analytical Hessians) - same tight tolerances
UMA_MACE_HESSIAN_TOL = (0.001, 0.01)
UMA_MACE_FREQUENCY_TOL = (0.01, 5.0)  # rtol, atol in cm^-1

# Finite difference method tolerances
FD_FORWARD_TOL = (0.05, 2.0)  # Less accurate, first-order
FD_CENTRAL_TOL = (0.002, 0.02)  # Second-order
FD_5POINT_TOL = (0.001, 0.01)  # Fourth-order, most accurate

# Fairchem loop reproducibility tolerance (slightly relaxed due to loop-based implementation)
# Loop-based methods may have slight numerical variations between calls
FAIRCHEM_LOOP_REPRO_TOL = (1e-5, 1e-5)  # rtol=1e-5, atol=1e-5

# Harmonic calculator (analytical reference) - extremely tight
HARMONIC_TOL = (1e-6, 1e-6)

# ============================================================================
# Interpolation Tolerances
# ============================================================================

# Exact methods (linear, quadratic, spline)
INTERP_EXACT_TOL = 1e-6

# Iterative methods (geodesic, IDPP)
INTERP_ITERATIVE_TOL = 1e-3

# ============================================================================
# Frequency Comparison Tolerances
# ============================================================================

# Mode matching tolerance (cm^-1)
FREQUENCY_MODE_TOL = 5.0

# Frequency comparison tolerance
FREQUENCY_COMPARE_TOL = (0.01, 5.0)  # rtol, atol in cm^-1

# Adaptive Hessian tolerance (for adaptive delta tests)
ADAPTIVE_HESSIAN_TOL = (0.01, 0.1)

# ASE comparison tolerance (for comparing against ASE implementations)
ASE_COMPARISON_TOL = (5e-3, 5e-3)

# ============================================================================
# Symmetry and Error Tolerances (single values)
# ============================================================================

# Symmetry tolerance for Hessian matrices (max asymmetry)
HESSIAN_SYMMETRY_TOL = 1e-10

# Near-symmetry tolerance (for non-symmetrized Hessians)
HESSIAN_NEAR_SYMMETRY_TOL = 0.01

# Error tolerance for Richardson extrapolation accuracy checks
RICHARDSON_ERROR_TOL = 1e-10

# Asymmetry tolerance for adaptive Hessian calculations
ADAPTIVE_HESSIAN_ASYMMETRY_TOL = 0.001
