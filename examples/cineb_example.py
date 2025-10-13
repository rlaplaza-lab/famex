#!/usr/bin/env python3
"""
Example script demonstrating CI-NEB (Climbing Image Nudged Elastic Band) usage in QME.

This script shows how to use the newly implemented CI-NEB two-ended strategy
for finding transition states and reaction pathways.
"""

import numpy as np
from ase import Atoms
from ase.build import molecule


def create_h2_dissociation_reaction():
    """Create a simple H2 dissociation reaction for demonstration."""
    # H2 molecule
    h2 = molecule("H2")
    h2.set_cell([10, 10, 10])
    h2.center()

    # Separated H atoms
    h_atoms = Atoms("H2", positions=[[0, 0, 0], [2.5, 0, 0]])
    h_atoms.set_cell([10, 10, 10])
    h_atoms.center()

    return h2, h_atoms


def create_water_formation_reaction():
    """Create a water formation reaction for demonstration."""
    # H2 + O -> H2O
    h2 = molecule("H2")
    o = molecule("O")

    # Separate molecules
    h2.set_cell([15, 15, 15])
    h2.center()
    h2.translate([0, 0, 0])

    o.set_cell([15, 15, 15])
    o.center()
    o.translate([3, 0, 0])

    # Combined reactant
    reactant = h2 + o
    reactant.set_cell([15, 15, 15])
    reactant.center()

    # Product (H2O)
    h2o = molecule("H2O")
    h2o.set_cell([15, 15, 15])
    h2o.center()

    return reactant, h2o


def run_cineb_example(reactant, product, backend="mock", title="CI-NEB Example"):
    """Run CI-NEB optimization on given reactant and product."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

    try:
        from qme import Explorer

        print(f"Reactant: {len(reactant)} atoms")
        print(f"Product: {len(product)} atoms")
        print(f"Backend: {backend}")

        # Create Explorer with CI-NEB strategy
        explorer = Explorer(
            atoms=[reactant, product],
            backend=backend,
            strategy="two-ended",
            target="cineb",
            mode="interpolate",
        )

        print("\nRunning CI-NEB optimization...")
        print("Parameters:")
        print("  - npoints: 7")
        print("  - steps: 50")
        print("  - fmax: 0.05")
        print("  - climb: True")
        print("  - spring_constant: 5.0")

        # Run CI-NEB
        result = explorer.run(
            mode="cineb",
            npoints=7,
            steps=50,
            fmax=0.05,
            climb=True,
            spring_constant=5.0,
        )

        print("\n✅ CI-NEB completed successfully!")
        print(f"   Returned {len(result['trajectory'])} images along the reaction path")

        # Analyze the results
        print("\nPath Analysis:")
        for i, atoms in enumerate(result["trajectory"]):
            try:
                energy = atoms.get_potential_energy()
                max_force = np.max(np.abs(atoms.get_forces()))
                print(
                    f"   Image {i:2d}: Energy = {energy:8.4f} eV, Max Force = {max_force:.4f} eV/Å"
                )
            except Exception:
                print(f"   Image {i:2d}: Energy calculation failed")

        # Find the highest energy image (potential TS)
        energies = []
        for atoms in result["trajectory"]:
            try:
                energies.append(atoms.get_potential_energy())
            except Exception:
                energies.append(float("nan"))

        if energies and not all(np.isnan(energies)):
            valid_energies = [(i, e) for i, e in enumerate(energies) if not np.isnan(e)]
            if valid_energies:
                max_idx, max_energy = max(valid_energies, key=lambda x: x[1])
                print(f"\n🎯 Highest energy image: {max_idx} (Energy = {max_energy:.4f} eV)")
                print("   This is likely the transition state!")

        return result

    except Exception as e:
        print(f"❌ CI-NEB failed: {e}")
        return None


def compare_cineb_vs_neb():
    """Compare CI-NEB with regular NEB on the same reaction."""
    print(f"\n{'='*60}")
    print("CI-NEB vs NEB Comparison")
    print(f"{'='*60}")

    # Create test reaction
    reactant, product = create_h2_dissociation_reaction()

    try:
        from qme import Explorer

        # Run NEB
        print("\n1. Running regular NEB...")
        neb_explorer = Explorer(
            atoms=[reactant, product],
            backend="mock",
            strategy="two-ended",
            target="neb",
        )

        neb_result = neb_explorer.run(
            mode="neb",
            npoints=7,
            steps=50,
            fmax=0.05,
            spring_constant=5.0,
        )

        # Run CI-NEB
        print("\n2. Running CI-NEB...")
        cineb_explorer = Explorer(
            atoms=[reactant, product],
            backend="mock",
            strategy="two-ended",
            target="cineb",
        )

        cineb_result = cineb_explorer.run(
            mode="cineb",
            npoints=7,
            steps=50,
            fmax=0.05,
            climb=True,
            spring_constant=5.0,
        )

        print("\n📊 Comparison Results:")

        # Compare energies
        neb_energies = []
        cineb_energies = []

        for atoms in neb_result["trajectory"]:
            try:
                neb_energies.append(atoms.get_potential_energy())
            except Exception:
                neb_energies.append(float("nan"))

        for atoms in cineb_result["trajectory"]:
            try:
                cineb_energies.append(atoms.get_potential_energy())
            except Exception:
                cineb_energies.append(float("nan"))

        print("   Image | NEB Energy | CI-NEB Energy | Difference")
        print("   ------|------------|---------------|-----------")
        for i in range(min(len(neb_energies), len(cineb_energies))):
            neb_e = neb_energies[i]
            cineb_e = cineb_energies[i]
            if not (np.isnan(neb_e) or np.isnan(cineb_e)):
                diff = cineb_e - neb_e
                print(f"   {i:5d} | {neb_e:10.4f} | {cineb_e:13.4f} | {diff:+9.4f}")
            else:
                print(f"   {i:5d} | {'N/A':>10} | {'N/A':>13} | {'N/A':>9}")

        # Find highest energy images
        if neb_energies and not all(np.isnan(neb_energies)):
            valid_energies = [(i, e) for i, e in enumerate(neb_energies) if not np.isnan(e)]
            if valid_energies:
                neb_max_idx, neb_max_energy = max(valid_energies, key=lambda x: x[1])
                print(f"\n   NEB highest energy: Image {neb_max_idx} ({neb_max_energy:.4f} eV)")

        if cineb_energies and not all(np.isnan(cineb_energies)):
            valid_energies = [(i, e) for i, e in enumerate(cineb_energies) if not np.isnan(e)]
            if valid_energies:
                cineb_max_idx, cineb_max_energy = max(valid_energies, key=lambda x: x[1])
                print(
                    f"   CI-NEB highest energy: Image {cineb_max_idx} ({cineb_max_energy:.4f} eV)"
                )

        print("\n💡 Note: CI-NEB should typically find a higher energy transition state")
        print("   because the climbing image actively moves uphill along the reaction coordinate.")

    except Exception as e:
        print(f"❌ Comparison failed: {e}")


def demonstrate_multiple_atoms_support():
    """Demonstrate CI-NEB with multiple intermediate structures."""
    print(f"\n{'='*60}")
    print("CI-NEB with Multiple Intermediate Structures")
    print(f"{'='*60}")

    # Create a reaction path with multiple waypoints
    reactant, product = create_h2_dissociation_reaction()

    # Create an intermediate structure (H2 with stretched bond)
    intermediate = reactant.copy()
    intermediate.positions[1] = intermediate.positions[1] + [0.5, 0, 0]  # Stretch H-H bond

    print(f"Reactant: {len(reactant)} atoms")
    print(f"Intermediate: {len(intermediate)} atoms")
    print(f"Product: {len(product)} atoms")
    print("\nThis demonstrates CI-NEB's ability to handle multiple waypoints")
    print("along the reaction path, stitching them together into a continuous pathway.")

    try:
        from qme import Explorer

        explorer = Explorer(
            atoms=[reactant, intermediate, product],  # Multiple waypoints!
            backend="mock",
            strategy="two-ended",
            target="cineb",
        )

        result = explorer.run(
            mode="cineb",
            npoints=9,  # More points to show the stitched path
            steps=30,
            fmax=0.1,
            climb=True,
        )

        print("\n✅ CI-NEB with multiple waypoints completed!")
        print(f"   Generated {len(result)} images along the stitched pathway")

        return result

    except Exception as e:
        print(f"❌ Multi-waypoint CI-NEB failed: {e}")
        return None


if __name__ == "__main__":
    print("CI-NEB (Climbing Image Nudged Elastic Band) Examples")
    print("=" * 60)
    print("This script demonstrates the newly implemented CI-NEB strategy in QME.")
    print("CI-NEB is an improved version of NEB that helps locate transition states")
    print("more accurately by having one image 'climb' uphill along the reaction coordinate.")

    # Example 1: Simple H2 dissociation
    reactant1, product1 = create_h2_dissociation_reaction()
    run_cineb_example(reactant1, product1, backend="mock", title="Example 1: H2 Dissociation")

    # Example 2: Water formation (more complex)
    reactant2, product2 = create_water_formation_reaction()
    run_cineb_example(reactant2, product2, backend="mock", title="Example 2: Water Formation")

    # Example 3: Compare CI-NEB vs NEB
    compare_cineb_vs_neb()

    # Example 4: Multiple waypoints
    demonstrate_multiple_atoms_support()

    print(f"\n{'='*60}")
    print("🎉 All CI-NEB examples completed!")
    print("=" * 60)
    print("\nKey Features of the CI-NEB Implementation:")
    print("✅ Compatible with 2+ input atom objects")
    print("✅ Supports batch evaluation for compatible calculators")
    print("✅ Implements proper climbing image behavior")
    print("✅ Integrates seamlessly with QME's Explorer API")
    print("✅ Provides energy-weighted tangent calculations")
    print("✅ Supports spring forces and force projection")
    print("\nUsage:")
    print("  explorer = Explorer(atoms=[reactant, product], target='cineb')")
    print("  result = explorer.run(mode='cineb', climb=True)")
