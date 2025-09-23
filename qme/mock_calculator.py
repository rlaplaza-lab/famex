"""
Mock calculator for testing QME functionality without UMA dependencies.
"""

import numpy as np
from ase.calculators.calculator import Calculator, all_changes


class MockUMACalculator(Calculator):
    """
    Mock calculator that simulates UMA behavior for testing.

    This calculator uses simple harmonic oscillator potentials to simulate
    molecular interactions for testing purposes.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, **kwargs):
        """Initialize mock calculator."""
        Calculator.__init__(self, **kwargs)

        # Simple parameters for harmonic oscillator model
        self.bond_length = 1.0  # Equilibrium bond length (Å)
        self.force_constant = 1.0  # Force constant (eV/Å²)

    def calculate(
        self, atoms=None, properties=["energy", "forces"], system_changes=all_changes
    ):
        """Calculate energy and forces using simple harmonic model."""

        Calculator.calculate(self, atoms, properties, system_changes)

        positions = atoms.positions
        n_atoms = len(atoms)

        # Simple harmonic oscillator model between all atom pairs
        energy = 0.0
        forces = np.zeros_like(positions)

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                # Vector between atoms
                r_vec = positions[j] - positions[i]
                r = np.linalg.norm(r_vec)

                # Harmonic potential: E = 0.5 * k * (r - r0)²
                energy += 0.5 * self.force_constant * (r - self.bond_length) ** 2

                # Force: F = -k * (r - r0) * r_unit
                if r > 1e-6:  # Avoid division by zero
                    r_unit = r_vec / r
                    force_magnitude = -self.force_constant * (r - self.bond_length)
                    force_vec = force_magnitude * r_unit

                    forces[i] -= force_vec
                    forces[j] += force_vec

        # Store results
        if "energy" in properties:
            self.results["energy"] = energy

        if "forces" in properties:
            self.results["forces"] = forces


def get_mock_uma_calculator(**kwargs):
    """
    Get mock UMA calculator for testing.

    Returns:
    --------
    MockUMACalculator
        Mock calculator instance
    """
    return MockUMACalculator(**kwargs)
