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


class MockAIMNet2Calculator(Calculator):
    """
    Mock calculator that simulates AIMNET2 behavior for testing.

    This calculator uses a slightly different harmonic oscillator model to 
    simulate AIMNET2-like molecular interactions for testing purposes.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, charge=0, mult=1, **kwargs):
        """Initialize mock AIMNET2 calculator."""
        Calculator.__init__(self, **kwargs)

        # Simple parameters for harmonic oscillator model
        self.bond_length = 1.2  # Equilibrium bond length (Å) - slightly different from UMA
        self.force_constant = 0.8  # Force constant (eV/Å²) - slightly different from UMA
        self.charge = charge
        self.mult = mult

    def set_charge(self, charge):
        """Set molecular charge."""
        self.charge = charge

    def set_mult(self, mult):
        """Set spin multiplicity."""
        self.mult = mult

    def calculate(
        self, atoms=None, properties=["energy", "forces"], system_changes=all_changes
    ):
        """Calculate energy and forces using simple harmonic model."""

        Calculator.calculate(self, atoms, properties, system_changes)

        positions = atoms.positions
        n_atoms = len(atoms)

        # Simple harmonic oscillator model between all atom pairs with charge correction
        energy = 0.0
        forces = np.zeros_like(positions)

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                # Vector between atoms
                r_vec = positions[j] - positions[i]
                r = np.linalg.norm(r_vec)

                # Harmonic potential with charge modification: E = 0.5 * k * (r - r0)²
                # Add small charge correction to simulate charge-dependent behavior
                charge_factor = 1.0 + 0.1 * abs(self.charge) / 10.0
                energy += 0.5 * self.force_constant * charge_factor * (r - self.bond_length) ** 2

                # Force: F = -k * (r - r0) * r_unit
                if r > 1e-6:  # Avoid division by zero
                    r_unit = r_vec / r
                    force_magnitude = -self.force_constant * charge_factor * (r - self.bond_length)
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


def get_mock_aimnet2_calculator(**kwargs):
    """
    Get mock AIMNET2 calculator for testing.

    Returns:
    --------
    MockAIMNet2Calculator
        Mock calculator instance
    """
    return MockAIMNet2Calculator(**kwargs)
