"""SN2 reaction example: CH3Cl + OH- → CH3OH + Cl-

This example demonstrates how to study a classic SN2 nucleophilic
substitution reaction using QME, including mechanism validation.
"""

import matplotlib.pyplot as plt
import numpy as np

from qme import Geometry, MLPCalculator, Reaction


def create_sn2_reactants():
    """Create SN2 reactant geometry: CH3Cl + OH-"""
    atoms = ["C", "H", "H", "H", "Cl", "O", "H"]
    coords = np.array(
        [
            # CH3Cl part
            0.0,
            0.0,
            0.0,  # C
            1.09,
            0.0,
            0.0,  # H
            -0.36,
            1.03,
            0.0,  # H
            -0.36,
            -0.51,
            0.89,  # H
            -0.37,
            -0.52,
            -1.76,  # Cl
            # OH- approaching
            3.0,
            0.0,
            0.0,  # O (far away initially)
            3.97,
            0.0,
            0.0,  # H
        ]
    )
    return Geometry(atoms=atoms, coords=coords, charge=-1, mult=1)


def create_sn2_products():
    """Create SN2 product geometry: CH3OH + Cl-"""
    atoms = ["C", "H", "H", "H", "Cl", "O", "H"]
    coords = np.array(
        [
            # CH3OH part
            0.0,
            0.0,
            0.0,  # C
            1.09,
            0.0,
            0.0,  # H
            -0.36,
            1.03,
            0.0,  # H
            -0.36,
            -0.51,
            0.89,  # H
            -3.0,
            0.0,
            0.0,  # Cl (far away)
            # OH bound to carbon
            -1.43,
            -0.52,
            -0.87,  # O
            -1.8,
            -1.42,
            -0.87,  # H
        ]
    )
    return Geometry(atoms=atoms, coords=coords, charge=-1, mult=1)


def analyze_sn2_mechanism(path_geoms):
    """Analyze SN2 mechanism by monitoring key bond distances."""
    c_cl_distances = []
    c_o_distances = []

    for geom in path_geoms:
        coords = geom.coords3d
        # C is atom 0, Cl is atom 4, O is atom 5
        c_cl_dist = np.linalg.norm(coords[0] - coords[4])
        c_o_dist = np.linalg.norm(coords[0] - coords[5])

        c_cl_distances.append(c_cl_dist)
        c_o_distances.append(c_o_dist)

    return np.array(c_cl_distances), np.array(c_o_distances)


def main():
    """Run SN2 reaction example."""
    print("=== QME Example: SN2 Reaction (CH₃Cl + OH⁻ → CH₃OH + Cl⁻) ===\n")

    # Create reactants and products
    reactants = create_sn2_reactants()
    products = create_sn2_products()

    print(
        f"Reactants created: CH₃Cl + OH⁻ ({reactants.natoms} atoms, charge = {reactants.charge})"
    )
    print(
        f"Products created: CH₃OH + Cl⁻ ({products.natoms} atoms, charge = {products.charge})"
    )

    # Create reaction
    reaction = Reaction(reactant=reactants, product=products, name="SN2_CH3Cl_OH")

    print(f"\nReaction: {reaction}")

    # Set up calculator
    calculator = MLPCalculator(model_type="SN2_demo")

    # Calculate endpoint energies
    calculator.calculate(reactants)
    calculator.calculate(products)

    print(f"\nReactant energy: {reactants.energy:.6f} Hartree")
    print(f"Product energy: {products.energy:.6f} Hartree")
    print(f"Reaction energy: {reaction.reaction_energy:.6f} Hartree")

    # Generate reaction pathway
    print(f"\nGenerating SN2 reaction pathway...")
    npoints = 20
    path_geoms = reaction.interpolate(npoints=npoints)

    # Calculate energies along path
    energies = []
    for i, geom in enumerate(path_geoms):
        calculator.calculate(geom)
        energies.append(geom.energy)

    # Analyze mechanism
    c_cl_distances, c_o_distances = analyze_sn2_mechanism(path_geoms)

    print(f"\nMechanism analysis:")
    print(f"Initial C-Cl distance: {c_cl_distances[0]:.3f} Å")
    print(f"Final C-Cl distance: {c_cl_distances[-1]:.3f} Å")
    print(f"Initial C-O distance: {c_o_distances[0]:.3f} Å")
    print(f"Final C-O distance: {c_o_distances[-1]:.3f} Å")

    # Check for concerted mechanism
    if (
        c_cl_distances[-1] > c_cl_distances[0] + 0.5
        and c_o_distances[0] > c_o_distances[-1] + 0.5
    ):
        print(
            "✓ Mechanism shows expected SN2 behavior: C-Cl bond breaks while C-O bond forms"
        )
    else:
        print("⚠ Unusual mechanism behavior detected")

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Energy profile
    axes[0, 0].plot(range(npoints), energies, "b-o", markersize=4)
    axes[0, 0].set_xlabel("Reaction Coordinate")
    axes[0, 0].set_ylabel("Energy (Hartree)")
    axes[0, 0].set_title("SN2 Energy Profile")
    axes[0, 0].grid(True, alpha=0.3)

    # Bond distances
    axes[0, 1].plot(range(npoints), c_cl_distances, "r-o", markersize=4, label="C-Cl")
    axes[0, 1].plot(range(npoints), c_o_distances, "g-o", markersize=4, label="C-O")
    axes[0, 1].set_xlabel("Reaction Coordinate")
    axes[0, 1].set_ylabel("Distance (Å)")
    axes[0, 1].set_title("Key Bond Distances")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Energy vs C-Cl distance
    axes[1, 0].plot(c_cl_distances, energies, "purple", marker="o", markersize=4)
    axes[1, 0].set_xlabel("C-Cl Distance (Å)")
    axes[1, 0].set_ylabel("Energy (Hartree)")
    axes[1, 0].set_title("Energy vs C-Cl Distance")
    axes[1, 0].grid(True, alpha=0.3)

    # Energy vs C-O distance
    axes[1, 1].plot(c_o_distances, energies, "orange", marker="o", markersize=4)
    axes[1, 1].set_xlabel("C-O Distance (Å)")
    axes[1, 1].set_ylabel("Energy (Hartree)")
    axes[1, 1].set_title("Energy vs C-O Distance")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("sn2_reaction_analysis.png", dpi=300, bbox_inches="tight")
    print(f"\nAnalysis plots saved as 'sn2_reaction_analysis.png'")

    # Export trajectory
    xyz_trajectory = reaction.to_xyz_trajectory(path_geoms)
    with open("sn2_reaction.xyz", "w") as f:
        f.write(xyz_trajectory)
    print(f"XYZ trajectory saved as 'sn2_reaction.xyz'")

    # RMSD analysis
    rmsd_from_reactant, rmsd_from_product = reaction.get_rmsd_profile(path_geoms)
    print(f"\nRMSD analysis:")
    print(f"Max RMSD from reactants: {max(rmsd_from_reactant):.3f} Å")
    print(f"Max RMSD from products: {max(rmsd_from_product):.3f} Å")

    # Find approximate transition state (highest energy point)
    ts_idx = np.argmax(energies)
    print(f"\nApproximate transition state:")
    print(f"Point {ts_idx}/{npoints-1} along pathway")
    print(f"C-Cl distance: {c_cl_distances[ts_idx]:.3f} Å")
    print(f"C-O distance: {c_o_distances[ts_idx]:.3f} Å")
    print(f"Energy: {energies[ts_idx]:.6f} Hartree")

    print(f"\nSN2 reaction example completed successfully!")


if __name__ == "__main__":
    main()
