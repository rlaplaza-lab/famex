#!/usr/bin/env python
"""Test if MACE installation affects UMA Hessian calculations."""

import numpy as np
from ase import Atoms

import qme
from qme.analysis.hessian import HessianCalculator

# Create water molecule
water = Atoms(
    symbols="H2O",
    positions=[
        [0.0, 0.0, 0.0],
        [0.757, 0.587, 0.0],
        [-0.757, 0.587, 0.0],
    ],
)

print("=" * 80)
print("Testing MACE Conflict Impact on UMA")
print("=" * 80)

# Check if MACE is installed
try:
    import mace

    print(f"✓ MACE is installed: version {mace.__version__}")
    mace_installed = True
except ImportError:
    print("✗ MACE is not installed")
    mace_installed = False

# Check e3nn version
try:
    import e3nn

    print(f"✓ e3nn version: {e3nn.__version__}")
except ImportError:
    print("✗ e3nn not found")

# Load UMA calculator
try:
    calc = qme.get_uma_calculator(model_name="uma-s-1p1")
    calc.ensure_loaded()
    print("✓ UMA calculator loaded")
except Exception as e:
    print(f"✗ Failed to load UMA: {e}")
    exit(1)

water.calc = calc

# Test 1: Compute Hessian without importing MACE
print("\n" + "=" * 80)
print("Test 1: UMA Hessian WITHOUT importing MACE")
print("=" * 80)

hessian_no_mace = calc.get_hessian(water, method="double_backward", symmetrize=True)
print(f"Hessian max value: {np.max(np.abs(hessian_no_mace)):.6f}")

# Test 2: Import MACE and check if it affects e3nn
if mace_installed:
    print("\n" + "=" * 80)
    print("Test 2: Importing MACE...")
    print("=" * 80)

    try:
        import mace.calculators

        print("✓ MACE imported successfully")

        # Check e3nn version again
        import e3nn

        print(f"✓ e3nn version after MACE import: {e3nn.__version__}")

        # Test 3: Compute Hessian AFTER importing MACE
        print("\n" + "=" * 80)
        print("Test 3: UMA Hessian AFTER importing MACE")
        print("=" * 80)

        hessian_with_mace = calc.get_hessian(water, method="double_backward", symmetrize=True)
        print(f"Hessian max value: {np.max(np.abs(hessian_with_mace)):.6f}")

        # Compare
        diff = hessian_no_mace - hessian_with_mace
        max_diff = np.max(np.abs(diff))
        print(f"\nDifference between before/after MACE import: {max_diff:.6e}")

        if max_diff > 1e-6:
            print("⚠️  WARNING: MACE import affected UMA Hessian calculation!")
        else:
            print("✓ MACE import did not affect UMA Hessian")

    except Exception as e:
        print(f"✗ Failed to import MACE: {e}")
        import traceback

        traceback.print_exc()

# Test 4: Compare with FD reference
print("\n" + "=" * 80)
print("Test 4: Comparing with Finite Difference Reference")
print("=" * 80)

try:
    hc = HessianCalculator(water, calc, delta=0.001, method="central", verbose=0)
    hessian_fd = hc.calculate_numerical_hessian()

    if mace_installed and "hessian_with_mace" in locals():
        diff_with_mace = hessian_with_mace - hessian_fd
        max_diff_with_mace = np.max(np.abs(diff_with_mace))
        print(f"UMA (with MACE imported) vs FD: {max_diff_with_mace:.6e}")

    diff_no_mace = hessian_no_mace - hessian_fd
    max_diff_no_mace = np.max(np.abs(diff_no_mace))
    print(f"UMA (no MACE import) vs FD: {max_diff_no_mace:.6e}")

    rtol, atol = (0.001, 0.01)
    within_tol_no_mace = np.allclose(hessian_no_mace, hessian_fd, rtol=rtol, atol=atol)
    print(f"Within tolerance (no MACE): {within_tol_no_mace}")

    if mace_installed and "hessian_with_mace" in locals():
        within_tol_with_mace = np.allclose(hessian_with_mace, hessian_fd, rtol=rtol, atol=atol)
        print(f"Within tolerance (with MACE): {within_tol_with_mace}")

except Exception as e:
    print(f"✗ FD calculation failed: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 80)
print("Done")
print("=" * 80)
