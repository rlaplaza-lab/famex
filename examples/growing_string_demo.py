#!/usr/bin/env python
"""Demo script showing the growing string method for TS search.

This example demonstrates the growing string method (DE-GSM) for finding
transition states between a reactant and product configuration.

Usage:
    python growing_string_demo.py [--backend BACKEND] [--reactant REACTANT.xyz] [--product PRODUCT.xyz]

Examples:
    python growing_string_demo.py --backend uma
    python growing_string_demo.py --backend uma --reactant reactant.xyz --product product.xyz
    python growing_string_demo.py --backend uma --npoints 20 --steps 100
"""

import argparse
import sys
from pathlib import Path

from ase import Atoms
from ase.io import write

import qme


def create_h2_reaction():
    """Create a simple H2 dissociation reaction for demo purposes.

    Returns reactant (compressed H2) and product (stretched H2).
    """
    # Reactant: Compressed H2
    reactant = Atoms("H2", positions=[(0, 0, 0), (0.6, 0, 0)])

    # Product: Stretched H2
    product = Atoms("H2", positions=[(0, 0, 0), (2.0, 0, 0)])

    return reactant, product


def main() -> int:
    """Run growing string method demo."""
    parser = argparse.ArgumentParser(
        description="Growing String Method (DE-GSM) demo for TS search",
    )
    parser.add_argument(
        "--reactant",
        help="Path to reactant XYZ file (optional)",
    )
    parser.add_argument(
        "--product",
        help="Path to product XYZ file (optional)",
    )
    parser.add_argument(
        "--backend",
        default="uma",
        help="Backend to use: uma|aimnet2|mace|mock (default: uma)",
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
    parser.add_argument(
        "--output",
        default="growing_string_result.xyz",
        help="Output trajectory file (default: growing_string_result.xyz)",
    )

    args = parser.parse_args()

    # Load or create structures
    if args.reactant and args.product:
        from ase.io import read

        reactant = read(args.reactant)
        product = read(args.product)
    else:
        reactant, product = create_h2_reaction()

    # Setup Explorer
    explorer = qme.Explorer(
        atoms=[reactant, product],
        backend=args.backend,
        target="ts",
        strategy="growing_string",
    )

    # Run growing string method

    result = explorer.run(
        npoints=args.npoints,
        fmax=args.fmax,
        steps=args.steps,
        step_size=args.step_size,
        optimize_endpoints=args.optimize_endpoints,
        refine_ts=args.refine_ts,
    )

    # Display results

    # Calculate energies along path
    trajectory = result["trajectory"]
    energies = []
    for _i, atoms in enumerate(trajectory):
        try:
            energy = atoms.get_potential_energy()
            energies.append(energy)
        except Exception:
            energies.append(None)

    if any(e is not None for e in energies):
        valid_energies = [e for e in energies if e is not None]

        # Find TS index
        if energies:
            max_e = max(valid_energies)
            energies.index(max_e)

    # Save trajectory
    output_path = Path(args.output)
    write(str(output_path), trajectory)

    # Save TS structure separately
    ts_path = output_path.parent / f"{output_path.stem}_ts.xyz"
    write(str(ts_path), result["optimized_atoms"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
