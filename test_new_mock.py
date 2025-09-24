"""Test script for the new unified mock calculators."""

from ase.build import molecule

from qme.mock_calculator import (
    MockAIMNet2Calculator,
    MockSO3LRCalculator,
    MockUMACalculator,
)


def test_calculator(calc_class, name):
    """Test a calculator with H2 molecule."""
    print(f"\n=== Testing {name} ===")

    # Create H2 molecule
    h2 = molecule("H2")
    print(f"H2 positions: {h2.positions}")
    print(f"H2 symbols: {h2.get_chemical_symbols()}")

    # Create calculator
    calc = calc_class()
    h2.calc = calc

    # Get energy and forces
    energy = h2.get_potential_energy()
    forces = h2.get_forces()

    print(f"Energy: {energy:.6f} eV")
    print(f"Forces shape: {forces.shape}")
    print(f"Forces:\n{forces}")
    print(
        f"Force magnitudes: {[f'{mag:.6f}' for mag in [abs(f).sum() for f in forces]]}"
    )

    return energy, forces


def test_water_molecule(calc_class, name):
    """Test calculator with H2O molecule."""
    print(f"\n=== Testing {name} with H2O ===")

    # Create water molecule
    water = molecule("H2O")
    print(f"H2O positions: {water.positions}")
    print(f"H2O symbols: {water.get_chemical_symbols()}")

    # Create calculator
    calc = calc_class()
    water.calc = calc

    # Get energy and forces
    energy = water.get_potential_energy()
    forces = water.get_forces()

    print(f"Energy: {energy:.6f} eV")
    print(f"Forces shape: {forces.shape}")
    print(f"Forces:\n{forces}")
    print(
        f"Force magnitudes: {[f'{mag:.6f}' for mag in [abs(f).sum() for f in forces]]}"
    )

    return energy, forces


if __name__ == "__main__":
    print("Testing unified mock calculators...")

    calculators = [
        (MockUMACalculator, "UMA"),
        (MockAIMNet2Calculator, "AIMNet2"),
        (MockSO3LRCalculator, "SO3LR"),
    ]

    # Test with H2
    for calc_class, name in calculators:
        test_calculator(calc_class, name)

    # Test with H2O to see covalent bond detection
    for calc_class, name in calculators:
        test_water_molecule(calc_class, name)
