#!/usr/bin/env python3
"""QME Growing String Method Demo - Transition State Search."""

import sys
from pathlib import Path

from ase import Atoms
from ase.io import read, write

import qme
from qme.example_utils import QMEExampleInterface, create_standard_epilog, setup_example_environment


def create_h2_reaction():
    """Create H2 dissociation reaction (reactant and product)."""
    # Reactant: Compressed H2
    reactant = Atoms("H2", positions=[(0, 0, 0), (0.6, 0, 0)])

    # Product: Stretched H2
    product = Atoms("H2", positions=[(0, 0, 0), (2.0, 0, 0)])

    return reactant, product


@setup_example_environment
def main() -> int:
    """Run growing string method demo."""
    # Create standardized interface
    interface = QMEExampleInterface(
        name="Growing String Method Demo",
        description="Transition State Search with Growing String Method",
        epilog=create_standard_epilog("demo"),
    )

    parser = interface.create_parser()

    # Add demo-specific arguments
    parser.add_argument(
        "--reactant",
        help="Path to reactant XYZ file (optional, uses H2 dissociation if not provided)",
    )
    parser.add_argument(
        "--product",
        help="Path to product XYZ file (optional, uses H2 dissociation if not provided)",
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
        default=50,
        help="Maximum growing iterations (default: 50)",
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
        default=0.5,
        help="Force convergence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--optimize-endpoints",
        action="store_true",
        help="Optimize reactant and product before growing (default: False)",
    )
    parser.add_argument(
        "--refine-ts",
        action="store_true",
        help="Refine TS with local optimization after finding it (default: False)",
    )

    args = parser.parse_args()

    interface.print_header()
    interface.setup_logging(args.verbose)

    # Backend handling
    requested = [b.strip() for b in args.backends.split(",")] if args.backends else None
    backend, available_backends = interface.select_backend(
        requested_backends=requested,
        preferred_backends=["uma", "mace"],
        verbose=args.verbose,
    )
    if backend is None:
        interface.print_error("No suitable backend available (need UMA or MACE)")
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
            reactant, product = create_h2_reaction()
            print("\nUsing default H2 dissociation reaction")

        # Setup Explorer
        print("\nSetting up Explorer with growing string method...")
        explorer = qme.Explorer(
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
            optimize_endpoints=args.optimize_endpoints,
            refine_ts=args.refine_ts,
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
