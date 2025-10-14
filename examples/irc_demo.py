#!/usr/bin/env python
"""
IRC (Intrinsic Reaction Coordinate) Demo
=========================================

This example demonstrates how to calculate an IRC path from a transition state.
The IRC follows the gradient downhill in both forward and backward directions
to generate the minimum energy path connecting reactants and products.

Usage:
    python irc_demo.py [ts_structure.xyz] [--backend BACKEND] [--steps STEPS]

Example:
    python irc_demo.py example_files/reaction_001_ts.xyz --backend uma --steps 50
"""

import argparse
import os
from pathlib import Path

from ase.io import read

import qme


def main():
    """Run IRC calculation demo."""
    parser = argparse.ArgumentParser(
        description="IRC path calculation from transition state"
    )
    parser.add_argument(
        "ts_file",
        nargs="?",
        default="example_files/reaction_001_ts.xyz",
        help="Path to transition state XYZ file (default: example_files/reaction_001_ts.xyz)",
    )
    parser.add_argument(
        "--backend",
        default="mock",
        help="Backend to use: uma|aimnet2|mace|mock (default: mock)",
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
        print(f"Error: File not found: {ts_file}")
        return 1

    print(f"Loading TS structure from: {ts_file}")
    ts_atoms = read(ts_file)
    print(f"  Atoms: {ts_atoms.get_chemical_formula()}")
    print(f"  Number of atoms: {len(ts_atoms)}")

    # Setup Explorer for IRC calculation
    print(f"\nSetting up IRC calculation with backend: {args.backend}")
    explorer = qme.Explorer(
        atoms=ts_atoms,
        backend=args.backend,
        target="path",
        strategy="irc",
    )

    # Run IRC
    print(f"\nRunning IRC calculation...")
    print(f"  Direction: {args.direction}")
    print(f"  Max steps per direction: {args.steps}")
    print(f"  Step size: {args.step_size} amu^1/2 * Angstrom")
    print(f"  Force threshold: {args.fmax} eV/Angstrom")

    result = explorer.run(
        mode="irc",
        steps=args.steps,
        step_size=args.step_size,
        fmax=args.fmax,
        direction=args.direction,
    )

    # Extract trajectory
    trajectory = result["trajectory"]
    forward_path = result.get("forward_path", [])
    backward_path = result.get("backward_path", [])

    print(f"\nIRC calculation completed!")
    print(f"  Total images: {len(trajectory)}")
    print(f"  Forward path images: {len(forward_path)}")
    print(f"  Backward path images: {len(backward_path)}")

    # Calculate and display energies along path
    print("\nEnergy profile along IRC path:")
    energies = []
    for i, atoms in enumerate(trajectory):
        try:
            energy = atoms.get_potential_energy()
            energies.append(energy)
            if i % 5 == 0 or i == len(trajectory) - 1:
                print(f"  Image {i:3d}: {energy:10.6f} eV")
        except Exception:
            pass

    if energies:
        min_energy = min(energies)
        max_energy = max(energies)
        print(f"\nEnergy statistics:")
        print(f"  Min energy: {min_energy:.6f} eV")
        print(f"  Max energy: {max_energy:.6f} eV")
        print(f"  Energy range: {max_energy - min_energy:.6f} eV")

    # Save trajectory
    if args.output:
        output_file = args.output
    else:
        output_file = ts_file.stem + "_irc.xyz"

    from ase.io import write
    write(output_file, trajectory)
    print(f"\nTrajectory saved to: {output_file}")

    return 0


if __name__ == "__main__":
    exit(main())
