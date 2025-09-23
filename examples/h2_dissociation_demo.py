"""Basic example: H2 dissociation reaction using QME.

This example demonstrates the basic workflow for studying a simple
bond dissociation reaction using the QME package.
"""

import numpy as np
import matplotlib.pyplot as plt
from qme import Geometry, Reaction, MLPCalculator


def main():
    """Run H2 dissociation example."""
    print("=== QME Example: H2 Dissociation ===\n")
    
    # Create H2 molecule (reactant)
    atoms = ["H", "H"]
    h2_coords = np.array([0.0, 0.0, 0.0, 0.74, 0.0, 0.0])  # Equilibrium bond length ~0.74 Å
    h2_molecule = Geometry(atoms=atoms, coords=h2_coords, charge=0, mult=1)
    
    print(f"H2 molecule created with {h2_molecule.natoms} atoms")
    print(f"Initial H-H distance: {np.linalg.norm(h2_molecule.coords3d[1] - h2_molecule.coords3d[0]):.3f} Å")
    
    # Create separated H atoms (product)  
    h_separated_coords = np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])  # Large separation
    h_atoms_separated = Geometry(atoms=atoms, coords=h_separated_coords, charge=0, mult=3)  # Triplet state
    
    print(f"Separated H atoms with distance: {np.linalg.norm(h_atoms_separated.coords3d[1] - h_atoms_separated.coords3d[0]):.3f} Å")
    
    # Create reaction pathway
    reaction = Reaction(
        reactant=h2_molecule,
        product=h_atoms_separated, 
        name="H2_dissociation"
    )
    
    print(f"\nReaction created: {reaction}")
    
    # Set up MLP calculator (using mock for demonstration)
    calculator = MLPCalculator(model_type="H2_dissociation_demo")
    
    # Calculate energies for endpoints
    calculator.calculate(h2_molecule)
    calculator.calculate(h_atoms_separated)
    
    print(f"\nH2 molecule energy: {h2_molecule.energy:.6f} Hartree")
    print(f"Separated atoms energy: {h_atoms_separated.energy:.6f} Hartree") 
    print(f"Dissociation energy: {reaction.reaction_energy:.6f} Hartree")
    
    # Generate reaction pathway
    print(f"\nGenerating reaction pathway...")
    npoints = 15
    path_geoms = reaction.interpolate(npoints=npoints)
    
    # Calculate energies and distances along path
    energies = []
    distances = []
    
    for i, geom in enumerate(path_geoms):
        calculator.calculate(geom)
        energies.append(geom.energy)
        
        # Calculate H-H distance
        h_h_dist = np.linalg.norm(geom.coords3d[1] - geom.coords3d[0])
        distances.append(h_h_dist)
        
        print(f"Point {i:2d}: H-H = {h_h_dist:.3f} Å, E = {geom.energy:.6f} Hartree")
    
    # Create energy profile plot
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(distances, energies, 'b-o', markersize=4)
    plt.xlabel('H-H Distance (Å)')
    plt.ylabel('Energy (Hartree)')
    plt.title('H₂ Dissociation Energy Profile')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(npoints), distances, 'r-o', markersize=4)
    plt.xlabel('Reaction Coordinate')
    plt.ylabel('H-H Distance (Å)')
    plt.title('H-H Distance Along Path')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('h2_dissociation_profile.png', dpi=300, bbox_inches='tight')
    print(f"\nEnergy profile saved as 'h2_dissociation_profile.png'")
    
    # Export XYZ trajectory
    xyz_trajectory = reaction.to_xyz_trajectory(path_geoms)
    
    with open('h2_dissociation.xyz', 'w') as f:
        f.write(xyz_trajectory)
    
    print(f"XYZ trajectory saved as 'h2_dissociation.xyz'")
    
    # Calculate some properties
    rmsd_from_reactant, rmsd_from_product = reaction.get_rmsd_profile(path_geoms)
    
    print(f"\nRMSD analysis:")
    print(f"Max RMSD from reactant: {max(rmsd_from_reactant):.3f} Å")
    print(f"Max RMSD from product: {max(rmsd_from_product):.3f} Å")
    
    print(f"\nExample completed successfully!")


if __name__ == "__main__":
    main()