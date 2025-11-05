#!/usr/bin/env python3
"""Compare UMA Hessian computation methods against finite differences.

NOTE: For practical usage, use FrequencyAnalysis with method='autoselect'
(see examples/adaptive_hessian_demo.py).
"""

import sys

import numpy as np

# Import shared functions from base module
from hessian_method_comparison_base import compare_methods, load_xyz_from_url

from qme.example_utils import (
    QMEExampleInterface,
    create_methane_molecule,
    create_standard_epilog,
    create_water_molecule,
    get_calculator_for_backend,
    setup_example_environment,
)


@setup_example_environment
def main() -> int:
    """Main function to run comparison."""
    # Create standardized interface
    interface = QMEExampleInterface(
        name="UMA Hessian Method Comparison",
        description="UMA Hessian Computation Methods Comparison",
        epilog=create_standard_epilog("demo"),
    )

    parser = interface.create_parser()
    args = parser.parse_args()

    interface.print_header()
    interface.setup_logging(args.verbose)

    # Backend handling: prefer UMA, fallback to MACE
    requested = [b.strip() for b in args.backends.split(",")] if args.backends else None
    backend, available_backends = interface.select_backend(
        requested_backends=requested,
        preferred_backends=["uma", "mace"],
        verbose=args.verbose,
    )
    if backend is None:
        interface.print_error("No suitable backend available (need UMA or MACE)")
        return 1

    # Get device info
    device = interface.get_device_info(args.device)

    # Set up calculator
    if backend == "uma":
        calc = get_calculator_for_backend(backend, device=device, model_name="uma-s-1p1")
        backend_name = "UMA"
    else:
        calc = get_calculator_for_backend(backend, device=device)
        backend_name = "MACE"
        if backend != "uma":
            interface.print_warning(
                "UMA not selected, using MACE instead. Some UMA-specific Hessian methods may not be available."
            )

    interface.print_backend_summary([backend], "Using Backend")

    config = {
        "Backend": backend_name,
        "Device": device,
        "Verbose": args.verbose,
    }
    interface.print_configuration(config)

    print("\nThis script compares different Hessian computation methods")
    print("against finite differences to identify the most accurate approach.\n")

    try:
        calc.ensure_loaded()

        # Create test molecules
        water = create_water_molecule()
        methane = create_methane_molecule()

        # Test water
        print("\n" + "=" * 80)
        print("WATER MOLECULE")
        print("=" * 80)
        water.calc = calc
        water_results = compare_methods(water, verbose=True)

        # Test methane
        print("\n" + "=" * 80)
        print("METHANE MOLECULE")
        print("=" * 80)
        methane.calc = calc
        methane_results = compare_methods(methane, verbose=True)

        # Additional molecules from external XYZ sources
        external_urls = [
            # 1,3-butadiene
            "https://github.com/lvpp/sigma/raw/cf6ef53a5a9ffef0459b7d2dfe495ebd8d6244c8/geometry/bp86-d2svp/1%2C3-BUTADIENE.xyz",
            # 1-heptanol
            "https://github.com/lvpp/sigma/raw/cf6ef53a5a9ffef0459b7d2dfe495ebd8d6244c8/geometry/bp86-d2svp/1-HEPTANOL.xyz",
        ]

        for url in external_urls:
            name = url.split("/")[-1]
            print("\n" + "=" * 80)
            print(f"EXTERNAL: {name}")
            print("=" * 80)
            try:
                ext_atoms = load_xyz_from_url(url)
            except Exception as e:
                print(f"Failed to load {name}: {e}")
                continue

            ext_atoms.calc = calc
            # Ensure charge/spin
            ext_atoms.info.setdefault("charge", 0)
            ext_atoms.info.setdefault("spin", 1)

            # For large systems, skip FD convergence and only do analytical + one FD if feasible
            compare_methods(ext_atoms, verbose=True)

        # Overall recommendation
        print("\n" + "=" * 80)
        print("RECOMMENDATION")
        print("=" * 80)
        print("\nBased on RMS error with finite differences:\n")

        all_methods = set(water_results.keys()) | set(methane_results.keys())
        method_scores = {}
        for method in all_methods:
            scores = []
            if method in water_results and "error" not in water_results[method]:
                scores.append(water_results[method]["metrics"]["rms_error"])
            if method in methane_results and "error" not in methane_results[method]:
                scores.append(methane_results[method]["metrics"]["rms_error"])
            if scores:
                method_scores[method] = np.mean(scores)

        sorted_methods = sorted(method_scores.items(), key=lambda x: x[1])
        for i, (method, avg_rms) in enumerate(sorted_methods, 1):
            print(f"  {i}. {method}: avg RMS = {avg_rms:.6f} eV/Å²")

        if sorted_methods:
            best = sorted_methods[0][0]
            print(
                f"\nRecommended default: method='{best.split('_sym=')[0]}', symmetrize={best.split('=')[-1]}"
            )

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
