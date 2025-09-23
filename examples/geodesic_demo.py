#!/usr/bin/env python3
"""
Demonstration of geodesic interpolation for transition state searches in QME.

This example shows how to use the new geodesic interpolation feature
for better transition state guesses, inspired by nudged elastic band methods.
"""
import sys

sys.path.insert(0, "src")

import numpy as np

from qme import Geometry, MLPCalculator, Reaction


def demo_geodesic_interpolation():
    """Demonstrate geodesic vs linear interpolation capabilities."""

    print("=" * 60)
    print("QME Geodesic Interpolation Demo")
    print("=" * 60)

    # Example 1: Simple H2 dissociation
    print("\n1. H2 Dissociation Example")
    print("-" * 30)

    atoms = ["H", "H"]
    reactant_coords = np.array([0.0, 0.0, 0.0, 0.74, 0.0, 0.0])  # H2 bond
    product_coords = np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0])  # Dissociated

    reactant = Geometry(atoms, reactant_coords, charge=0, mult=1)
    product = Geometry(atoms, product_coords, charge=0, mult=3)
    reaction = Reaction(reactant, product, name="H2_dissociation")

    # Compare linear vs geodesic interpolation
    linear_path = reaction.interpolate(npoints=7, method="linear")
    geodesic_path = reaction.interpolate(npoints=7, method="geodesic")

    print("H-H distances along reaction coordinate:")
    print("Point  Linear   Geodesic")
    for i in range(7):
        lin_dist = np.linalg.norm(
            linear_path[i].coords3d[0] - linear_path[i].coords3d[1]
        )
        geo_dist = np.linalg.norm(
            geodesic_path[i].coords3d[0] - geodesic_path[i].coords3d[1]
        )
        print(f"{i:3d}    {lin_dist:.3f}    {geo_dist:.3f}")

    # Example 2: Water molecule rotation
    print("\n2. Water Rotation Example (showing bond preservation)")
    print("-" * 50)

    atoms = ["O", "H", "H"]
    coords1 = np.array([0.0, 0.0, 0.0, 0.8, 0.6, 0.0, -0.8, 0.6, 0.0])  # Initial
    coords2 = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0])  # Rotated

    geom1 = Geometry(atoms, coords1)
    geom2 = Geometry(atoms, coords2)
    water_reaction = Reaction(geom1, geom2, name="water_rotation")

    linear_water = water_reaction.interpolate(npoints=5, method="linear")
    geodesic_water = water_reaction.interpolate(npoints=5, method="geodesic")

    print("O-H bond lengths during rotation:")
    print("Point  Linear(OH1) Linear(OH2)  Geodesic(OH1) Geodesic(OH2)")

    for i in range(5):
        # Linear interpolation bond lengths
        lin_coords = linear_water[i].coords3d
        lin_oh1 = np.linalg.norm(lin_coords[0] - lin_coords[1])
        lin_oh2 = np.linalg.norm(lin_coords[0] - lin_coords[2])

        # Geodesic interpolation bond lengths
        geo_coords = geodesic_water[i].coords3d
        geo_oh1 = np.linalg.norm(geo_coords[0] - geo_coords[1])
        geo_oh2 = np.linalg.norm(geo_coords[0] - geo_coords[2])

        print(
            f"{i:3d}       {lin_oh1:.3f}      {lin_oh2:.3f}          {geo_oh1:.3f}        {geo_oh2:.3f}"
        )

    print("\nObservation: Geodesic interpolation preserves O-H bond lengths better!")

    # Example 3: Transition state guess finding
    print("\n3. Automatic Transition State Guess Finding")
    print("-" * 40)

    ts_guess = reaction.find_transition_state_guess(npoints=10, method="geodesic")
    if ts_guess:
        h_h_dist = np.linalg.norm(ts_guess.coords3d[0] - ts_guess.coords3d[1])
        print(f"Found TS guess:")
        print(f"  Energy: {ts_guess.energy:.6f} Hartree")
        print(f"  H-H distance: {h_h_dist:.3f} Å")

    # Example 4: Path optimization (experimental feature)
    print("\n4. Path Optimization with NEB-like Forces")
    print("-" * 40)

    calculator = MLPCalculator(model_type="mock")
    try:
        optimized_path = reaction.interpolate(
            npoints=5, method="geodesic", optimize_path=True, calculator=calculator
        )
        print(f"Successfully optimized path with {len(optimized_path)} geometries")
        print("Note: This uses a simplified NEB-like algorithm")
    except Exception as e:
        print(f"Path optimization demo: {e}")

    print("\n" + "=" * 60)
    print("Summary: Geodesic Interpolation Benefits")
    print("=" * 60)
    print("✓ Better preservation of bond lengths")
    print("✓ More chemically reasonable intermediate structures")
    print("✓ Improved initial guesses for transition state searches")
    print("✓ Compatible with existing QME workflow")
    print("✓ Inspired by nudged elastic band (NEB) methods")
    print("✓ Handles optional Hessian calculations via NNP backends")

    print("\nUsage:")
    print("  reaction.interpolate(npoints=10, method='geodesic')")
    print("  reaction.find_transition_state_guess(method='geodesic')")


if __name__ == "__main__":
    demo_geodesic_interpolation()
