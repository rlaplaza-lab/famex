"""
Mock calculators for testing QME functionality without ML dependencies.

This module provides standardized mock calculators that simulate the behavior
of various quantum mechanical calculators (UMA, AIMNet2, SO3LR) for testing
purposes without requiring the actual dependencies.

All mock calculators use simple harmonic oscillator potentials between
covalently bonded atoms to ensure numerical stability.
"""

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.data import atomic_numbers, covalent_radii


class BaseMockCalculator(Calculator):
    """
    Base mock calculator with common functionality.

    This class provides a standardized interface and common methods
    for all mock calculators used in testing. All calculators use
    harmonic oscillator potentials between covalently bonded atoms only.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, bond_length=1.0, force_constant=1.0, charge=0, mult=1, **kwargs):
        """Initialize base mock calculator.

        Parameters
        ----------
        bond_length : float, optional
            Equilibrium bond length in Å (default: 1.0)
        force_constant : float, optional
            Force constant in eV/Å² (default: 1.0)
        charge : int, optional
            Molecular charge (default: 0)
        mult : int, optional
            Spin multiplicity (default: 1)
        **kwargs
            Additional keyword arguments passed to ASE Calculator
        """
        Calculator.__init__(self, **kwargs)

        self.bond_length = bond_length
        self.force_constant = force_constant
        self.charge = charge
        self.mult = mult

    def set_charge(self, charge):
        """Set molecular charge."""
        self.charge = charge

    def set_mult(self, mult):
        """Set spin multiplicity."""
        self.mult = mult

    def set_atoms(self, atoms):
        """Set atoms object for the calculator."""
        self.atoms = atoms

    def _detect_covalent_bonds(self, atoms):
        """
        Detect covalent bonds based on interatomic distances and covalent radii.

        Parameters
        ----------
        atoms : ase.Atoms
            Atoms object

        Returns
        -------
        list of tuples
            List of (i, j) pairs representing bonded atoms
        """
        positions = atoms.positions
        symbols = atoms.get_chemical_symbols()
        n_atoms = len(atoms)
        bonds = []

        # Get atomic numbers for covalent radii lookup
        atom_numbers = [atomic_numbers[symbol] for symbol in symbols]

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                # Calculate distance between atoms
                r_vec = positions[j] - positions[i]
                r = np.linalg.norm(r_vec)

                # Sum of covalent radii with tolerance
                r_cov_i = covalent_radii[atom_numbers[i]]
                r_cov_j = covalent_radii[atom_numbers[j]]
                bond_threshold = (r_cov_i + r_cov_j) * 1.3  # 30% tolerance

                # Consider atoms bonded if distance is within threshold
                if r < bond_threshold:
                    bonds.append((i, j))

        return bonds

    def _calculate_harmonic_potential(self, atoms):
        """
        Calculate energy and forces using harmonic oscillator model between bonded atoms only.

        Parameters
        ----------
        atoms : ase.Atoms
            Atoms object

        Returns
        -------
        tuple
            (energy, forces) where energy is float and forces is numpy array
        """
        positions = atoms.positions
        n_atoms = len(atoms)
        energy = 0.0
        forces = np.zeros_like(positions)

        # Get list of covalent bonds
        bonds = self._detect_covalent_bonds(atoms)

        # Apply harmonic potential only between bonded atoms
        for i, j in bonds:
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

        return energy, forces

    def calculate(
        self, atoms=None, properties=["energy", "forces"], system_changes=all_changes
    ):
        """Calculate energy and forces using harmonic model between bonded atoms."""
        Calculator.calculate(self, atoms, properties, system_changes)

        if atoms is None:
            raise ValueError("Atoms object must be provided")

        energy, forces = self._calculate_harmonic_potential(atoms)

        # Store results
        if "energy" in properties:
            self.results["energy"] = energy

        if "forces" in properties:
            self.results["forces"] = forces


class UnifiedMockCalculator(BaseMockCalculator):
    """
    Unified mock calculator that can simulate different backend behaviors.

    This replaces the individual mock calculators with a single configurable one.
    """

    # Pre-defined configurations for different backends
    BACKEND_CONFIGS = {
        "uma": {"bond_length": 1.0, "force_constant": 1.0, "name": "MockUMA"},
        "aimnet2": {"bond_length": 1.2, "force_constant": 0.8, "name": "MockAIMNet2"},
        "so3lr": {"bond_length": 1.1, "force_constant": 0.9, "name": "MockSO3LR"},
        "generic": {"bond_length": 1.0, "force_constant": 1.0, "name": "MockGeneric"},
    }

    def __init__(self, backend="generic", charge=0, mult=1, **kwargs):
        """Initialize unified mock calculator.

        Parameters
        ----------
        backend : str, default "generic"
            Backend to simulate ("uma", "aimnet2", "so3lr", "generic")
        charge : int, default 0
            Molecular charge
        mult : int, default 1
            Spin multiplicity
        **kwargs
            Override default parameters
        """
        # Get base configuration for backend
        if backend not in self.BACKEND_CONFIGS:
            backend = "generic"

        config = self.BACKEND_CONFIGS[backend].copy()
        config.update({"charge": charge, "mult": mult})
        config.update(kwargs)  # Allow overriding defaults

        self.backend = backend
        super().__init__(**config)

    def __repr__(self):
        return f"{self.BACKEND_CONFIGS[self.backend]['name']}(charge={self.charge}, mult={self.mult})"


# Backward compatibility classes
class MockUMACalculator(UnifiedMockCalculator):
    """Mock calculator that simulates UMA behavior for testing."""

    def __init__(self, **kwargs):
        super().__init__(backend="uma", **kwargs)


class MockAIMNet2Calculator(UnifiedMockCalculator):
    """Mock calculator that simulates AIMNET2 behavior for testing."""

    def __init__(self, charge=0, mult=1, **kwargs):
        super().__init__(backend="aimnet2", charge=charge, mult=mult, **kwargs)


class MockSO3LRCalculator(UnifiedMockCalculator):
    """Mock calculator that simulates SO3LR behavior for testing."""

    def __init__(self, **kwargs):
        super().__init__(backend="so3lr", **kwargs)

        # Add SO3LR-like attributes for compatibility
        self.device = "cpu"  # Mock device
        self.model = self  # Self-reference for compatibility


def get_mock_uma_calculator(**kwargs):
    """Get mock UMA calculator for testing.

    Parameters
    ----------
    **kwargs
        Keyword arguments passed to UnifiedMockCalculator

    Returns
    -------
    UnifiedMockCalculator
        Mock calculator configured for UMA backend
    """
    return UnifiedMockCalculator(backend="uma", **kwargs)


def get_mock_aimnet2_calculator(**kwargs):
    """Get mock AIMNET2 calculator for testing.

    Parameters
    ----------
    **kwargs
        Keyword arguments passed to UnifiedMockCalculator

    Returns
    -------
    UnifiedMockCalculator
        Mock calculator configured for AIMNET2 backend
    """
    return UnifiedMockCalculator(backend="aimnet2", **kwargs)


def get_mock_so3lr_calculator(**kwargs):
    """Get mock SO3LR calculator for testing.

    Parameters
    ----------
    **kwargs
        Keyword arguments passed to UnifiedMockCalculator

    Returns
    -------
    UnifiedMockCalculator
        Mock calculator configured for SO3LR backend
    """
    return UnifiedMockCalculator(backend="so3lr", **kwargs)


# Convenience alias for backward compatibility and general testing
MockCalculator = UnifiedMockCalculator
