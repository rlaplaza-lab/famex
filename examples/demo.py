"""
Example script demonstrating QME capabilities.
"""

from ase.build import molecule
from ase.io import write

from qme import QMEOptimizer


def main():
    print("QME Example: Molecular Optimization")
    print("=" * 40)

    # Create optimizer (will use mock calculator if UMA not available)
    try:
        qme = QMEOptimizer(model_name="uma-4m")
        print("Using UMA machine learning potential")
    except ImportError:
        qme = QMEOptimizer(use_mock=True)
        print("Using mock calculator for demonstration")

    # Create a slightly distorted water molecule
    atoms = molecule("H2O")
    atoms.positions[1] += [0.3, 0.0, 0.0]  # Move H atom
    atoms.positions[2] += [0.0, 0.2, 0.0]  # Move other H atom

    print(f"\nInitial structure: {len(atoms)} atoms")
    print(f"Initial energy: {atoms.calc is None}")

    # Set calculator and get initial energy
    atoms.calc = qme.calculator
    initial_energy = atoms.get_potential_energy()
    print(f"Initial energy: {initial_energy:.4f} eV")

    # Save initial structure
    write("initial_structure.xyz", atoms)

    # Optimize structure
    print("\nRunning geometry optimization...")
    results = qme.optimize_minimum(atoms=atoms, optimizer="BFGS", fmax=0.05, steps=100)

    # Print results
    print("\nOptimization Results:")
    print(f"Converged: {results['converged']}")
    print(f"Steps taken: {results['steps_taken']}")
    print(f"Energy change: {results['energy_change']:.4f} eV")
    print(f"Final max force: {results['final_max_force']:.4f} eV/Å")

    # Save optimized structure
    write("optimized_structure.xyz", results["optimized_atoms"])

    # Show summary
    print("\n" + qme.get_optimization_summary())

    print("\nFiles created:")
    print("- initial_structure.xyz: Starting geometry")
    print("- optimized_structure.xyz: Optimized geometry")

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
