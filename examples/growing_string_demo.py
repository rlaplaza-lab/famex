#!/usr/bin/env python3
"""FAMEX Growing String Method Demo - Transition State Search."""

import sys
from pathlib import Path

from ase.io import read, write

import famex
from famex.example_utils import (
    FAMEXExampleInterface,
    create_standard_epilog,
    setup_example_environment,
)

DEFAULT_REACTANT = "A_C_A_B_A_C_reactant.xyz"
DEFAULT_PRODUCT = "A_C_A_B_A_C_product.xyz"


def load_default_reaction() -> tuple:
    """Load the ACAB example reaction from the bundled example files."""
    examples_dir = Path(__file__).parent / "example_files"
    reactant_path = examples_dir / DEFAULT_REACTANT
    product_path = examples_dir / DEFAULT_PRODUCT

    if not reactant_path.exists() or not product_path.exists():
        missing = [str(path) for path in (reactant_path, product_path) if not path.exists()]
        raise FileNotFoundError(
            f"Default example structures not found. Missing file(s): {', '.join(missing)}"
        )

    reactant = read(reactant_path)
    product = read(product_path)
    return reactant, product


@setup_example_environment
def main() -> int:
    """Run growing string method demo."""
    # Create standardized interface
    interface = FAMEXExampleInterface(
        name="Growing String Method Demo",
        description="Transition State Search with Growing String Method",
        epilog=create_standard_epilog("demo"),
    )

    parser = interface.create_parser()

    # Add demo-specific arguments
    parser.add_argument(
        "--reactant",
        help=(
            "Path to reactant XYZ file "
            "(optional, defaults to examples/example_files/"
            f"{DEFAULT_REACTANT})"
        ),
    )
    parser.add_argument(
        "--product",
        help=(
            "Path to product XYZ file "
            "(optional, defaults to examples/example_files/"
            f"{DEFAULT_PRODUCT})"
        ),
    )
    parser.add_argument(
        "--npoints",
        type=int,
        default=15,
        help="Maximum number of images to generate (default: 15)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Maximum growing iterations (default: 200)",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=0.1,
        help="Step size for adding new nodes in Angstroms (default: 0.1)",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=0.05,
        help="Force convergence threshold for growing string optimization (default: 0.05)",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.1,
        help="Distance threshold for string convergence in Angstroms (default: 0.1)",
    )
    parser.add_argument(
        "--strict-ts-validation",
        action="store_true",
        help="Exit with error if frequency analysis does not find exactly 1 imaginary mode",
    )
    parser.add_argument(
        "--optimize-endpoints",
        action="store_true",
        default=True,
        help="Optimize reactant and product before growing (default: True)",
    )
    parser.add_argument(
        "--no-optimize-endpoints",
        dest="optimize_endpoints",
        action="store_false",
        help="Skip endpoint optimization before growing",
    )
    parser.add_argument(
        "--refine-ts",
        action="store_true",
        default=True,
        help="Refine TS with local optimization after finding it (default: True)",
    )
    parser.add_argument(
        "--no-refine-ts",
        dest="refine_ts",
        action="store_false",
        help="Skip TS refinement (use raw highest-energy image as TS)",
    )

    args = parser.parse_args()

    interface.print_header()
    interface.setup_logging(args.verbose)

    # Backend handling
    requested = [b.strip() for b in args.backends.split(",")] if args.backends else None
    backend, available_backends = interface.select_backend(
        requested_backends=requested,
        preferred_backends=["uma"],
        verbose=args.verbose,
    )
    if backend is None:
        interface.print_error("No suitable backend available (UMA preferred)")
        return 1

    interface.print_backend_summary([backend], "Using Backend")

    # Get device info
    device = interface.get_device_info(args.device)

    # Set default output if not provided
    output_file = args.output or "growing_string_result.xyz"

    config = {
        "Backend": backend,
        "Device": device,
        "Output": output_file,
        "Verbose": args.verbose,
    }
    interface.print_configuration(config)

    try:
        # Load or create structures
        if args.reactant and args.product:
            reactant = read(args.reactant)
            product = read(args.product)
            print("\nLoaded structures from files:")
            print(f"  Reactant: {args.reactant}")
            print(f"  Product: {args.product}")
        else:
            reactant, product = load_default_reaction()
            default_dir = Path(__file__).parent / "example_files"
            print("\nUsing default ACAB reaction from example_files/")
            print(f"  Reactant: {default_dir / DEFAULT_REACTANT}")
            print(f"  Product:  {default_dir / DEFAULT_PRODUCT}")

        # Setup Explorer
        print("\nSetting up Explorer with growing string method...")
        explorer = famex.Explorer(
            atoms=[reactant, product],
            backend=backend,
            target="ts",
            strategy="growing_string",
            device=device,
        )

        # Run growing string method
        print("\nRunning growing string method...")
        print(f"  Max images: {args.npoints}")
        print(f"  Max steps: {args.steps}")
        print(f"  Step size: {args.step_size} Å")
        print(f"  Force threshold: {args.fmax} eV/Å")

        result = explorer.run(
            npoints=args.npoints,
            fmax=args.fmax,
            steps=args.steps,
            step_size=args.step_size,
            distance_threshold=args.distance_threshold,
            optimize_endpoints=args.optimize_endpoints,
            refine_ts=args.refine_ts,
            ts_refinement_steps=500,
            ts_refinement_fmax=0.01,
        )

        # Display results
        trajectory = result["trajectory"]
        print(f"\n✓ Generated trajectory with {len(trajectory)} images")

        # Calculate energies along path
        energies = []
        for atoms in trajectory:
            try:
                energy = atoms.get_potential_energy()
                energies.append(energy)
            except Exception:
                energies.append(None)

        if any(e is not None for e in energies):
            valid_energies = [e for e in energies if e is not None]
            if valid_energies:
                min_e = min(valid_energies)
                max_e = max(valid_energies)
                ts_idx = energies.index(max_e)
                print("\nEnergy profile:")
                print(f"  Minimum energy: {min_e:.6f} eV")
                print(f"  Maximum energy: {max_e:.6f} eV")
                print(f"  TS found at image: {ts_idx + 1}/{len(trajectory)}")

        # Save trajectory
        output_path = Path(output_file)
        write(str(output_path), trajectory)
        print(f"\n✓ Saved trajectory to: {output_path}")

        # Save TS structure separately
        ts_path = output_path.parent / f"{output_path.stem}_ts.xyz"
        write(str(ts_path), result["optimized_atoms"])
        print(f"✓ Saved TS structure to: {ts_path}")

        # Validate TS with frequency analysis
        frequency_analysis = result.get("frequency_analysis")
        if frequency_analysis:
            ts_analysis = frequency_analysis.get("ts_analysis", {})
            n_imaginary = ts_analysis.get("n_imaginary_frequencies", 0)
            imaginary_freqs = ts_analysis.get("imaginary_frequencies", [])

            print("\n" + "=" * 60)
            print("Transition State Frequency Validation")
            print("=" * 60)

            if n_imaginary == 1:
                print(f"✓ Valid transition state: {n_imaginary} imaginary frequency found")
                if imaginary_freqs:
                    print(f"  Imaginary frequency: {imaginary_freqs[0]:.2f} cm⁻¹")
            else:
                print(f"⚠ TS frequency check: {n_imaginary} imaginary frequencies found")
                if imaginary_freqs:
                    freq_str = ", ".join(f"{f:.2f}" for f in imaginary_freqs)
                    print(f"  Imaginary frequencies: {freq_str} cm⁻¹")
                else:
                    print("  No imaginary frequencies found")
                print(
                    "  Note: ML backends may not reproduce reference saddle-point character "
                    "for this reaction; trajectory and TS structure were still generated."
                )
                if args.strict_ts_validation:
                    interface.print_error(
                        f"TS validation failed: expected 1 imaginary frequency, found {n_imaginary}"
                    )
                    return 1
        else:
            print("\n⚠ Warning: Frequency analysis not available for TS validation")
            print("  TS structure saved but not validated")

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
