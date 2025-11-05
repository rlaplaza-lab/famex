#!/usr/bin/env python3
"""Demonstrate adaptive Hessian selection and comparison."""

import sys

import numpy as np

from qme.analysis.frequency import FrequencyAnalysis
from qme.analysis.hessian_comparison import HessianComparisonReport, compare_hessian_methods
from qme.example_utils import (
    QMEExampleInterface,
    create_standard_epilog,
    create_water_molecule,
    get_calculator_for_backend,
    setup_example_environment,
)


def demo_basic_usage(backend: str | None = None, device: str | None = None):
    """Show basic usage of adaptive Hessian features."""
    print("\n" + "=" * 80)
    print("DEMO 1: Basic Adaptive Hessian Usage")
    print("=" * 80)

    # Create a test molecule
    water = create_water_molecule()

    # Set up calculator
    calc = get_calculator_for_backend(backend, device=device)
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


def demo_adaptive_delta(backend: str | None = None, device: str | None = None):
    """Demonstrate adaptive delta selection."""
    print("\n" + "=" * 80)
    print("DEMO 2: Adaptive Delta Selection")
    print("=" * 80)

    from qme.analysis.hessian import HessianCalculator

    # Create test molecule
    water = create_water_molecule()

    calc = get_calculator_for_backend(backend, device=device)
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


def demo_method_comparison(backend: str | None = None, device: str | None = None):
    """Compare multiple Hessian methods."""
    print("\n" + "=" * 80)
    print("DEMO 3: Method Comparison")
    print("=" * 80)

    # Create test molecule
    water = create_water_molecule()

    calc = get_calculator_for_backend(backend, device=device)
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


def demo_noise_estimation(backend: str | None = None, device: str | None = None):
    """Demonstrate noise estimation features."""
    print("\n" + "=" * 80)
    print("DEMO 4: Noise Estimation")
    print("=" * 80)

    from qme.analysis.noise_estimation import estimate_force_noise, estimate_optimal_delta

    # Create test molecule
    water = create_water_molecule()

    calc = get_calculator_for_backend(backend, device=device)
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


@setup_example_environment
def main() -> int:
    """Run all demos."""
    # Create standardized interface
    interface = QMEExampleInterface(
        name="Adaptive Hessian Demo",
        description="Adaptive Hessian Selection Demonstration",
        epilog=create_standard_epilog("demo"),
    )

    parser = interface.create_parser()
    args = parser.parse_args()

    interface.print_header()
    interface.setup_logging(args.verbose)

    # Backend handling (consistent pattern)
    requested = [b.strip() for b in args.backends.split(",")] if args.backends else None
    backend, available_backends = interface.select_backend(
        requested_backends=requested,
        preferred_backends=["mace", "uma"],
        verbose=args.verbose,
    )
    if backend is None:
        interface.print_error("No suitable backend available (need MACE or UMA)")
        return 1

    interface.print_backend_summary([backend], "Using Backend")

    # Get device info
    device = interface.get_device_info(args.device)

    config = {
        "Backend": backend,
        "Device": device,
        "Verbose": args.verbose,
    }
    interface.print_configuration(config)

    print("\nThis demo showcases QME's adaptive Hessian features:")
    print("- Automatic method selection based on calculator capabilities")
    print("- Adaptive delta selection for optimal accuracy")
    print("- Noise estimation and handling")
    print("- Method comparison and recommendations")

    try:
        demo_basic_usage(backend=backend, device=device)
        demo_adaptive_delta(backend=backend, device=device)
        demo_method_comparison(backend=backend, device=device)
        demo_noise_estimation(backend=backend, device=device)

        print("\n" + "=" * 80)
        print("DEMO COMPLETE")
        print("=" * 80)
        print("\nKey takeaways:")
        print("- Use method='autoselect' for automatic best-method selection")
        print("- Enable adaptive_delta=True for optimal finite difference accuracy")
        print("- Check noise estimates if Hessian quality is questionable")
        print("- Compare methods when in doubt about accuracy")

        interface.print_success()
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        interface.print_error(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
