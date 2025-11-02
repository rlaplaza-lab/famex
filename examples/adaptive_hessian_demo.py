#!/usr/bin/env python3
"""Demonstrate adaptive Hessian selection and comparison.

This script shows how to use QME's adaptive Hessian features to automatically
select the best computation method and parameters for optimal accuracy.
"""

import os

# Disable ASE GUI to prevent popup windows
os.environ["DISPLAY"] = ""
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
from ase import Atoms

import qme
from qme.analysis.frequency import FrequencyAnalysis
from qme.analysis.hessian_comparison import HessianComparisonReport, compare_hessian_methods
from qme.backends.availability import is_backend_available


def get_available_calculator():
    """Get an available calculator, preferring MACE over UMA."""
    if is_backend_available("mace"):
        return qme.get_mace_calculator()
    elif is_backend_available("uma"):
        return qme.get_uma_calculator(model_name="uma-s-1p1")
    else:
        raise RuntimeError("No suitable backend available (need MACE or UMA)")


def demo_basic_usage():
    """Show basic usage of adaptive Hessian features."""
    print("\n" + "=" * 80)
    print("DEMO 1: Basic Adaptive Hessian Usage")
    print("=" * 80)

    # Create a test molecule
    water = Atoms(
        symbols="OHH",
        positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0],
        ],
    )

    # Set up calculator
    calc = get_available_calculator()
    calc.ensure_loaded()
    water.calc = calc

    print(f"\nTesting water molecule with {len(water)} atoms")
    print("Using autoselect method to find optimal Hessian computation approach")

    # Use autoselect to get best Hessian
    freq_analysis = FrequencyAnalysis(water, calc, verbose=1)
    hessian = freq_analysis.calculate_hessian(method="autoselect")

    print("\n✓ Hessian computed successfully")
    print(f"  Shape: {hessian.shape}")
    print(f"  Symmetry error: {np.max(np.abs(hessian - hessian.T)):.2e} eV/Å²")


def demo_adaptive_delta():
    """Demonstrate adaptive delta selection."""
    print("\n" + "=" * 80)
    print("DEMO 2: Adaptive Delta Selection")
    print("=" * 80)

    from qme.analysis.hessian import HessianCalculator

    # Create test molecule
    water = Atoms(
        symbols="OHH",
        positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0],
        ],
    )

    calc = get_available_calculator()
    calc.ensure_loaded()
    water.calc = calc

    print("\nComparing fixed vs adaptive delta:")
    print("-" * 80)

    # Fixed delta
    print("\n1. Fixed delta (δ=0.01 Å):")
    hessian_calc_fixed = HessianCalculator(water, calc, delta=0.01, adaptive_delta=False, verbose=0)
    hessian_fixed = hessian_calc_fixed.calculate_numerical_hessian()
    asymmetry_fixed = np.max(np.abs(hessian_fixed - hessian_fixed.T))
    print(f"   Symmetry error: {asymmetry_fixed:.2e} eV/Å²")

    # Adaptive delta
    print("\n2. Adaptive delta selection:")
    hessian_calc_adaptive = HessianCalculator(
        water, calc, delta=0.01, adaptive_delta=True, max_iterations=3, verbose=1
    )
    hessian_adaptive = hessian_calc_adaptive.calculate_numerical_hessian()
    asymmetry_adaptive = np.max(np.abs(hessian_adaptive - hessian_adaptive.T))
    print(f"   Symmetry error: {asymmetry_adaptive:.2e} eV/Å²")


def demo_method_comparison():
    """Compare multiple Hessian methods."""
    print("\n" + "=" * 80)
    print("DEMO 3: Method Comparison")
    print("=" * 80)

    # Create test molecule
    water = Atoms(
        symbols="OHH",
        positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0],
        ],
    )

    calc = get_available_calculator()
    calc.ensure_loaded()
    water.calc = calc

    # Compare methods
    print("\nRunning comparison of available methods...")
    comparison = compare_hessian_methods(
        water, calc, methods=["analytical", "force_fd", "adaptive"], verbose=1
    )

    # Print detailed report
    report = HessianComparisonReport(comparison)
    report.print_summary()

    # Compare frequencies
    report.compare_frequencies(water)


def demo_noise_estimation():
    """Demonstrate noise estimation features."""
    print("\n" + "=" * 80)
    print("DEMO 4: Noise Estimation")
    print("=" * 80)

    from qme.analysis.noise_estimation import estimate_force_noise, estimate_optimal_delta

    # Create test molecule
    water = Atoms(
        symbols="OHH",
        positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0],
        ],
    )

    calc = get_available_calculator()
    calc.ensure_loaded()
    water.calc = calc

    print("\n1. Force noise estimation:")
    force_noise = estimate_force_noise(water, calc, n_samples=5)
    print(f"   Estimated force noise: {force_noise:.2e} eV/Å")

    if force_noise > 1e-4:
        print("   ⚠ High force noise detected - may affect FD Hessians")
    else:
        print("   ✓ Force noise acceptable for FD calculations")

    print("\n2. Optimal delta estimation:")
    optimal_delta, noise = estimate_optimal_delta(
        water, calc, delta_range=(0.001, 0.05), max_iterations=3, verbose=1
    )
    print(f"   Optimal delta: {optimal_delta:.4f} Å")
    print(f"   Expected noise at optimal delta: {noise:.2e} eV/Å²")


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("ADAPTIVE HESSIAN SELECTION DEMONSTRATION")
    print("=" * 80)
    print("\nThis demo showcases QME's adaptive Hessian features:")
    print("- Automatic method selection based on calculator capabilities")
    print("- Adaptive delta selection for optimal accuracy")
    print("- Noise estimation and handling")
    print("- Method comparison and recommendations")

    try:
        demo_basic_usage()
        demo_adaptive_delta()
        demo_method_comparison()
        demo_noise_estimation()

        print("\n" + "=" * 80)
        print("DEMO COMPLETE")
        print("=" * 80)
        print("\nKey takeaways:")
        print("- Use method='autoselect' for automatic best-method selection")
        print("- Enable adaptive_delta=True for optimal finite difference accuracy")
        print("- Check noise estimates if Hessian quality is questionable")
        print("- Compare methods when in doubt about accuracy")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
