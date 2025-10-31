#!/usr/bin/env python3
"""QME IRC Demo - Intrinsic Reaction Coordinate Calculation.

This example demonstrates how to calculate an IRC (Intrinsic Reaction Coordinate)
path from a transition state. The IRC follows the gradient downhill in both
forward and backward directions to generate the minimum energy path connecting
reactants and products.

Features:
    - Calculates IRC path from a single transition state structure
    - Follows gradient in both forward and backward directions
    - Generates complete reaction pathway through the TS
    - Energy profile analysis along the IRC path
    - Saves trajectory for visualization
"""

import os
import sys
from pathlib import Path

from ase.io import read, write

# Disable ASE GUI to prevent popup windows
os.environ["DISPLAY"] = ""
os.environ["MPLBACKEND"] = "Agg"

import qme
from qme.example_utils import QMEExampleInterface, create_standard_epilog


def main() -> int:
    """Run IRC calculation demo."""
    # Create standardized interface
    interface = QMEExampleInterface(
        name="IRC Demo",
        description="Intrinsic Reaction Coordinate Calculation",
        epilog=create_standard_epilog("demo"),
    )

    parser = interface.create_parser()

    # Add demo-specific arguments
    parser.add_argument(
        "ts_file",
        nargs="?",
        default="example_files/A_C_A_B_A_C_ts.xyz",
        help="Path to transition state XYZ file (default: example_files/A_C_A_B_A_C_ts.xyz)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Maximum IRC steps per direction (default: 50)",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=0.1,
        help="IRC step size in amu^1/2 * Angstrom (default: 0.1)",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=0.05,
        help="Force convergence threshold (default: 0.05)",
    )
    parser.add_argument(
        "--direction",
        choices=["forward", "backward", "both"],
        default="both",
        help="Direction to follow from TS (default: both)",
    )

    args = parser.parse_args()

    # Set default output if not provided
    if args.output is None:
        args.output = None  # Will be set later based on input filename

    interface.print_header()
    interface.setup_logging(args.verbose)

    # Parse backends if provided
    backend = "uma"  # Default
    if args.backends:
        backends_list = [b.strip() for b in args.backends.split(",")]
        if backends_list:
            backend = backends_list[0]
            if len(backends_list) > 1:
                interface.print_warning(f"Multiple backends specified, using first: {backend}")

    # Check backend availability
    from qme.backends.availability import is_backend_available

    if not is_backend_available(backend):
        interface.print_error(f"Backend '{backend}' not available")
        return 1

    interface.print_backend_summary([backend], "Using Backend")

    # Get device info
    device = interface.get_device_info(args.device)

    config = {
        "Backend": backend,
        "Device": device,
        "Output": args.output or "auto",
        "Verbose": args.verbose,
        "Direction": args.direction,
    }
    interface.print_configuration(config)

    try:
        # Load transition state structure
        ts_file = Path(args.ts_file)
        if not ts_file.exists():
            interface.print_error(f"Transition state file not found: {ts_file}")
            return 1

        print(f"\nLoading transition state from: {ts_file}")
        ts_atoms = read(ts_file)
        print(f"✓ Loaded structure with {len(ts_atoms)} atoms")

        # Setup Explorer for IRC calculation
        print("\nSetting up Explorer for IRC calculation...")
        explorer = qme.Explorer(
            atoms=ts_atoms,
            backend=backend,
            target="path",
            strategy="irc",
            device=device,
        )

        # Run IRC
        print("\nRunning IRC calculation...")
        print(f"  Steps per direction: {args.steps}")
        print(f"  Step size: {args.step_size} amu^1/2 * Å")
        print(f"  Force threshold: {args.fmax} eV/Å")
        print(f"  Direction: {args.direction}")

        result = explorer.run(
            steps=args.steps,
            step_size=args.step_size,
            fmax=args.fmax,
            direction=args.direction,
        )

        # Extract trajectory
        trajectory = result["trajectory"]
        print(f"\n✓ Generated IRC trajectory with {len(trajectory)} images")

        # Calculate and display energies along path
        energies = []
        for _i, atoms in enumerate(trajectory):
            try:
                energy = atoms.get_potential_energy()
                energies.append(energy)
            except Exception:
                energies.append(None)

        if energies:
            valid_energies = [e for e in energies if e is not None]
            if valid_energies:
                min_e = min(valid_energies)
                max_e = max(valid_energies)
                print("\nEnergy profile along IRC:")
                print(f"  Minimum energy: {min_e:.6f} eV")
                print(f"  Maximum energy: {max_e:.6f} eV")
                print(f"  Energy range: {max_e - min_e:.6f} eV")

        # Save trajectory
        output_file = args.output or ts_file.stem + "_irc.xyz"
        write(output_file, trajectory)
        print(f"\n✓ Saved IRC trajectory to: {output_file}")

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
