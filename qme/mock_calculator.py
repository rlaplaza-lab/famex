"""
Mock calculators for testing QME functionality without ML dependencies.

This module provides a standardized mock calculator that simulates the behavior
of various quantum mechanical calculators (UMA, AIMNet2, SO3LR) for testing
purposes without requiring the actual dependencies.

The mock calculator uses simple harmonic oscillator potentials between
covalently bonded atoms to ensure numerical stability.
"""

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.data import atomic_numbers, covalent_radii


class MockCalculator(Calculator):
    """
    Mock calculator for testing purposes.

    This class provides a standardized interface and common methods
    for all mock calculators used in testing. It uses harmonic oscillator
    potentials between covalently bonded atoms only, with equilibrium bond
    lengths determined from covalent radii.
    """

    implemented_properties = ["energy", "forces", "hessian"]

    # Pre-defined configurations for different backends
    BACKEND_CONFIGS = {
        "uma": {"force_constant": 5.0, "name": "MockUMA"},
        "aimnet2": {"force_constant": 5.0, "name": "MockAIMNet2"},
        "so3lr": {"force_constant": 5.0, "name": "MockSO3LR"},
        "mace": {"force_constant": 5.0, "name": "MockMACE"},
        "generic": {"force_constant": 5.0, "name": "MockGeneric"},
    }

    def __init__(self, backend="generic", charge=0, mult=1, **kwargs):
        """
        Initialize mock calculator.

        Parameters
        ----------
        backend : str, default "generic"
            Backend to simulate ("uma", "aimnet2", "so3lr", "mace", "generic")
        charge : int, default 0
            Molecular charge
        mult : int, default 1
            Spin multiplicity
        **kwargs
            Override default parameters
        """
        Calculator.__init__(self, **kwargs)

        # Get base configuration for backend
        if backend not in self.BACKEND_CONFIGS:
            backend = "generic"

        config = self.BACKEND_CONFIGS[backend].copy()
        config.update({"charge": charge, "mult": mult})
        config.update(kwargs)  # Allow overriding defaults

        self.backend = backend
        self.force_constant = config["force_constant"]
        self.charge = config["charge"]
        self.mult = config["mult"]

        # Add SO3LR-like attributes for compatibility
        if self.backend == "so3lr":
            self.device = "cpu"  # Mock device
            self.model = self  # Self-reference for compatibility

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
                bond_threshold = (r_cov_i + r_cov_j) * 1.2  # 20% tolerance

                # Consider atoms bonded if distance is within threshold
                if r < bond_threshold:
                    bonds.append((i, j))

        return bonds

    def _calculate_harmonic_potential(self, atoms):
        """
        Calculate energy and forces using harmonic oscillator model between
        bonded atoms only.

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
        symbols = atoms.get_chemical_symbols()
        atom_numbers = [atomic_numbers[symbol] for symbol in symbols]

        energy = 0.0
        forces = np.zeros_like(positions)

        # Get list of covalent bonds
        bonds = self._detect_covalent_bonds(atoms)

        # Apply harmonic potential only between bonded atoms
        for i, j in bonds:
            # Vector between atoms
            r_vec = positions[j] - positions[i]
            r = np.linalg.norm(r_vec)

            # Equilibrium bond length from sum of covalent radii
            r0 = covalent_radii[atom_numbers[i]] + covalent_radii[atom_numbers[j]]

            # Harmonic potential: E = 0.5 * k * (r - r0)²
            energy += 0.5 * self.force_constant * (r - r0) ** 2

            # Force: F = -k * (r - r0) * r_unit
            if r > 1e-6:  # Avoid division by zero
                r_unit = r_vec / r
                force_magnitude = -self.force_constant * (r - r0)
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

    def get_hessian(self, atoms=None):
        """
        Calculate analytical Hessian matrix for the harmonic potential model.

        Parameters
        ----------
        atoms : ase.Atoms, optional
            Atoms object. If None, uses self.atoms

        Returns
        -------
        numpy.ndarray
            Hessian matrix of shape (3*n_atoms, 3*n_atoms)
        """
        if atoms is None:
            atoms = self.atoms

        if atoms is None:
            raise ValueError("No atoms object available for Hessian calculation")

        positions = atoms.positions
        symbols = atoms.get_chemical_symbols()
        atom_numbers = [atomic_numbers[symbol] for symbol in symbols]
        n_atoms = len(atoms)

        # Initialize Hessian matrix (3N x 3N)
        hessian = np.zeros((3 * n_atoms, 3 * n_atoms))

        # Get list of covalent bonds
        bonds = self._detect_covalent_bonds(atoms)

        # Calculate Hessian contributions from each bond
        for i, j in bonds:
            # Vector between atoms
            r_vec = positions[j] - positions[i]
            r = np.linalg.norm(r_vec)

            # Equilibrium bond length from sum of covalent radii
            r0 = covalent_radii[atom_numbers[i]] + covalent_radii[atom_numbers[j]]

            if r > 1e-6:  # Avoid division by zero
                r_unit = r_vec / r

                # For harmonic potential V = 0.5 * k * (r - r0)²
                # The Hessian for a harmonic bond is simpler:
                # Second derivative is just k along the bond direction
                k = self.force_constant

                # Identity matrix for 3D
                identity = np.eye(3)

                # Outer product of unit vector
                rr = np.outer(r_unit, r_unit)

                # For a simple harmonic bond, the Hessian block is:
                # H = k * rr (only along bond direction)
                # This gives the correct second derivative for V = 0.5*k*(r-r0)²
                bond_hessian = k * rr

                # Add contributions to the full Hessian matrix
                # Diagonal blocks (on-site terms)
                hessian[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] += bond_hessian
                hessian[3 * j : 3 * j + 3, 3 * j : 3 * j + 3] += bond_hessian

                # Off-diagonal blocks (coupling terms)
                hessian[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] -= bond_hessian
                hessian[3 * j : 3 * j + 3, 3 * i : 3 * i + 3] -= bond_hessian

        return hessian

    def __repr__(self):
        return (
            f'{self.BACKEND_CONFIGS[self.backend]["name"]}('
            f"charge={self.charge}, mult={self.mult})"
        )
