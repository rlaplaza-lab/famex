#!/usr/bin/env python
"""Diagnostic script to compare UMA Hessian behavior between Python versions.

This script computes Hessians using different methods and compares them
to identify numerical differences.
"""

import numpy as np
from ase import Atoms

import qme
from qme.analysis.hessian import HessianCalculator

# Create a simple water molecule
water = Atoms(
    symbols="H2O",
    positions=[
        [0.0, 0.0, 0.0],
        [0.757, 0.587, 0.0],
        [-0.757, 0.587, 0.0],
    ],
)

print("=" * 80)
print("UMA Hessian Diagnostic")
print("=" * 80)
print(f"Python version: {__import__('sys').version}")
print(f"NumPy version: {np.__version__}")
try:
    import torch

    print(f"PyTorch version: {torch.__version__}")
except ImportError:
    print("PyTorch: Not available")

# Check if UMA is available
try:
    calc = qme.get_uma_calculator(model_name="uma-s-1p1")
    calc.ensure_loaded()
    print("\n✓ UMA calculator loaded successfully")
except Exception as e:
    print(f"\n✗ Failed to load UMA calculator: {e}")
    exit(1)

# Set calculator
water.calc = calc

print("\n" + "=" * 80)
print("Computing Hessians with different methods")
print("=" * 80)

methods = ["vmap", "double_backward", "fairchem", "fairchem_loop"]
hessians = {}

# Compute analytical Hessians
for method in methods:
    try:
        hessian = calc.get_hessian(water, method=method, symmetrize=True)
        hessians[method] = hessian
        print(f"\n✓ {method}: shape={hessian.shape}, max={np.max(np.abs(hessian)):.6f}")
    except Exception as e:
        print(f"\n✗ {method}: Failed - {e}")

# Compute finite difference reference
print("\n" + "=" * 80)
print("Computing Finite Difference Reference")
print("=" * 80)

try:
    hc = HessianCalculator(water, calc, delta=0.001, method="central", verbose=0)
    hessian_fd = hc.calculate_numerical_hessian()
    hessians["fd_central"] = hessian_fd
    print(
        f"✓ FD (central, delta=0.001): shape={hessian_fd.shape}, max={np.max(np.abs(hessian_fd)):.6f}"
    )
except Exception as e:
    print(f"✗ FD failed: {e}")

# Compare methods
print("\n" + "=" * 80)
print("Comparing Methods")
print("=" * 80)

if "fd_central" in hessians:
    fd_ref = hessians["fd_central"]

    for method in methods:
        if method in hessians:
            hessian = hessians[method]
            diff = hessian - fd_ref
            max_diff = np.max(np.abs(diff))
            mean_diff = np.mean(np.abs(diff))
            rel_diff = max_diff / (np.max(np.abs(fd_ref)) + 1e-10)

            print(f"\n{method} vs FD:")
            print(f"  Max absolute difference: {max_diff:.6e}")
            print(f"  Mean absolute difference: {mean_diff:.6e}")
            print(f"  Max relative difference: {rel_diff:.6e}")

            # Check if within tolerance
            rtol, atol = (0.001, 0.01)  # UMA_MACE_HESSIAN_TOL
            within_tol = np.allclose(hessian, fd_ref, rtol=rtol, atol=atol)
            print(f"  Within tolerance (rtol={rtol}, atol={atol}): {within_tol}")

            if not within_tol:
                # Show worst violations
                violations = np.abs(diff) > (atol + rtol * np.abs(fd_ref))
                n_violations = np.sum(violations)
                print(
                    f"  Violations: {n_violations} / {diff.size} ({100 * n_violations / diff.size:.1f}%)"
                )
                if n_violations > 0:
                    worst_idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
                    print(f"  Worst violation at {worst_idx}:")
                    print(f"    Analytical: {hessian[worst_idx]:.6e}")
                    print(f"    FD:         {fd_ref[worst_idx]:.6e}")
                    print(f"    Difference: {diff[worst_idx]:.6e}")

# Compare reproducibility
print("\n" + "=" * 80)
print("Reproducibility Check")
print("=" * 80)

for method in methods:
    if method in hessians:
        try:
            hessian1 = calc.get_hessian(water, method=method, symmetrize=True)
            hessian2 = calc.get_hessian(water, method=method, symmetrize=True)
            diff = hessian1 - hessian2
            max_diff = np.max(np.abs(diff))
            print(f"{method}: Max difference between calls = {max_diff:.6e}")

            rtol, atol = (1e-6, 1e-6)  # HARMONIC_TOL
            reproducible = np.allclose(hessian1, hessian2, rtol=rtol, atol=atol)
            print(f"  Reproducible (rtol={rtol}, atol={atol}): {reproducible}")
        except Exception as e:
            print(f"{method}: Failed reproducibility check - {e}")

print("\n" + "=" * 80)
print("Done")
print("=" * 80)
