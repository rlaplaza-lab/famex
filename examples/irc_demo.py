#!/usr/bin/env python
"""IRC (Intrinsic Reaction Coordinate) Demo.

This example demonstrates how to calculate an IRC path from a transition state.
The IRC follows the gradient downhill in both forward and backward directions
to generate the minimum energy path connecting reactants and products.

Usage:
    python irc_demo.py [ts_structure.xyz] [--backend BACKEND] [--steps STEPS]

Example:
    python irc_demo.py example_files/A_C_A_B_A_C_ts.xyz --backend uma --steps 50

"""

import argparse
import sys
from pathlib import Path

from ase.io import read

import qme


def main() -> int:
    """Run IRC calculation demo."""
    parser = argparse.ArgumentParser(description="IRC path calculation from transition state")
    parser.add_argument(
        "ts_file",
        nargs="?",
        default="example_files/A_C_A_B_A_C_ts.xyz",
        help="Path to transition state XYZ file (default: example_files/A_C_A_B_A_C_ts.xyz)",
    )
    parser.add_argument(
        "--backend",
        default="uma",
        help="Backend to use: uma|aimnet2|mace|mock (default: uma)",
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
    parser.add_argument(
        "--output",
        help="Output trajectory file (default: based on input filename)",
    )
    args = parser.parse_args()

    # Load transition state structure
    ts_file = Path(args.ts_file)
    if not ts_file.exists():
        return 1

    ts_atoms = read(ts_file)

    # Setup Explorer for IRC calculation
    explorer = qme.Explorer(
        atoms=ts_atoms,
        backend=args.backend,
        target="path",
        strategy="irc",
    )

    # Run IRC

    result = explorer.run(
        steps=args.steps,
        step_size=args.step_size,
        fmax=args.fmax,
        direction=args.direction,
    )

    # Extract trajectory
    trajectory = result["trajectory"]
    result.get("forward_path", [])
    result.get("backward_path", [])

    # Calculate and display energies along path
    energies = []
    for i, atoms in enumerate(trajectory):
        try:
            energy = atoms.get_potential_energy()
            energies.append(energy)
            if i % 5 == 0 or i == len(trajectory) - 1:
                pass
        except Exception:
            pass

    if energies:
        min(energies)
        max(energies)

    # Save trajectory
    output_file = args.output or ts_file.stem + "_irc.xyz"

    from ase.io import write

    write(output_file, trajectory)

    return 0


if __name__ == "__main__":
    sys.exit(main())
