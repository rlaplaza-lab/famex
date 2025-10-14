#!/usr/bin/env python
"""Demo script showing the growing string method for TS search.

This example demonstrates the growing string method (DE-GSM) for finding
transition states between a reactant and product configuration.
"""

import argparse
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


def main():
    """Run growing string method demo."""
    parser = argparse.ArgumentParser(
        description="Growing String Method (DE-GSM) demo for TS search"
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
        default="mock",
        help="Backend to use: uma|aimnet2|mace|mock (default: mock)",
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
        print(f"Loading reactant from: {args.reactant}")
        print(f"Loading product from: {args.product}")
        from ase.io import read
        reactant = read(args.reactant)
        product = read(args.product)
    else:
        print("No input files provided, using H2 dissociation demo...")
        reactant, product = create_h2_reaction()

    print(f"\nReactant: {reactant.get_chemical_formula()}")
    print(f"Product: {product.get_chemical_formula()}")

    # Setup Explorer
    print(f"\nSetting up Growing String Method with backend: {args.backend}")
    explorer = qme.Explorer(
        [reactant, product],
        backend=args.backend,
    )

    # Run growing string method
    print("\nRunning Growing String Method...")
    print(f"  Max images: {args.npoints}")
    print(f"  Max iterations: {args.steps}")
    print(f"  Step size: {args.step_size} Å")
    print(f"  Force threshold: {args.fmax} eV/Å")
    print(f"  Optimize endpoints: {args.optimize_endpoints}")
    print(f"  Refine TS: {args.refine_ts}")

    from qme.core.twoended_strategies import twoended_growing_string_runner

    result = twoended_growing_string_runner(
        [reactant, product],
        npoints=args.npoints,
        explorer=explorer,
        fmax=args.fmax,
        steps=args.steps,
        step_size=args.step_size,
        optimize_endpoints=args.optimize_endpoints,
        refine_ts=args.refine_ts,
    )

    # Display results
    print(f"\n{'=' * 60}")
    print("Growing String Method Results")
    print(f"{'=' * 60}")

    print("\nConvergence:")
    print(f"  Strings met: {result.get('strings_met', False)}")
    print(f"  TS converged: {result.get('converged', False)}")

    print("\nString statistics:")
    print(f"  Forward string images: {len(result['forward_string'])}")
    print(f"  Backward string images: {len(result['backward_string'])}")
    print(f"  Total trajectory images: {len(result['trajectory'])}")

    # Calculate energies along path
    trajectory = result["trajectory"]
    energies = []
    for i, atoms in enumerate(trajectory):
        try:
            energy = atoms.get_potential_energy()
            energies.append(energy)
        except Exception:
            energies.append(None)

    if any(e is not None for e in energies):
        valid_energies = [e for e in energies if e is not None]
        print("\nEnergy profile:")
        print(f"  Min energy: {min(valid_energies):.6f} eV")
        print(f"  Max energy: {max(valid_energies):.6f} eV")
        print(f"  Energy range: {max(valid_energies) - min(valid_energies):.6f} eV")

        # Find TS index
        if energies:
            max_e = max(valid_energies)
            ts_idx = energies.index(max_e)
            print(f"  TS at image {ts_idx}")

    # Save trajectory
    output_path = Path(args.output)
    print(f"\nSaving trajectory to: {output_path}")
    write(str(output_path), trajectory)

    # Save TS structure separately
    ts_path = output_path.parent / f"{output_path.stem}_ts.xyz"
    print(f"Saving TS structure to: {ts_path}")
    write(str(ts_path), result["optimized_atoms"])

    print(f"\n{'=' * 60}")
    print("✓ Growing String Method completed successfully!")
    print(f"{'=' * 60}\n")

    return 0


if __name__ == "__main__":
    exit(main())
