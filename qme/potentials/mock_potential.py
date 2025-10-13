"""A lightweight MockCalculator for tests and CI.

Implements a minimal ASE Calculator-like interface to return simple
energies and harmonic forces so tests can run without heavy ML deps.
Enhanced with TinyFF-inspired pairwise interactions for more realistic behavior.
"""

from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

# Constants for mock potential calculations
from ase.units import Ang, eV

BOND_TOLERANCE_FACTOR = 1.25  # Multiplier for covalent radius cutoff
MIN_BOND_CUTOFF = 2.0 * Ang  # Minimum bond cutoff distance
DEFAULT_FORCE_CONSTANT = 1.0 * eV / Ang**2  # Default harmonic force constant
DEFAULT_ANGLE_FORCE_RATIO = 0.5  # Angle force constant as fraction of bond force constant

# Non-bonded interaction parameters (TinyFF-inspired)
DEFAULT_EPSILON = 0.1 * eV  # LJ well depth
DEFAULT_SIGMA = 3.4 * Ang  # LJ size parameter
LJ_CUTOFF = 10.0 * Ang  # Cutoff for non-bonded interactions


class MockCalculator(Calculator):
    """Simple mock calculator for testing and CI.

    This calculator implements a minimal ASE Calculator-like interface to return
    simple energies and harmonic forces so tests can run without heavy ML dependencies.

    Parameters
    ----------
    backend : str, default "generic"
        Backend identifier for the mock calculator
    force_constant : float, default 1.0
        Force constant for harmonic potential calculations
    charge : int, default 0
        Total charge of the system
    mult : int, default 1
        Spin multiplicity (2S + 1)
    **kwargs
        Additional arguments passed to ASE Calculator
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        backend: str = "generic",
        force_constant: float = DEFAULT_FORCE_CONSTANT,
        **kwargs: Any,
    ) -> None:
        """Initialize the mock calculator.

        Parameters
        ----------
        backend : str, default "generic"
            Backend identifier for the mock calculator
        force_constant : float, default 1.0
            Force constant for harmonic potential calculations
        **kwargs
            Additional arguments passed to ASE Calculator
        """
        super().__init__(**kwargs)
        self.backend = backend
        self.force_constant = float(force_constant)
        # simple parameters
        self.charge = kwargs.get("charge", 0)
        self.mult = kwargs.get("mult", 1)
        # angle force constant (harmonic in angle space). Default is
        # moderately weaker than bond force to avoid over-constraining
        # small test systems.
        self.k_angle = float(kwargs.get("k_angle", DEFAULT_ANGLE_FORCE_RATIO * self.force_constant))

        # Non-bonded interaction parameters (TinyFF-inspired)
        self.epsilon = float(kwargs.get("epsilon", DEFAULT_EPSILON))
        self.sigma = float(kwargs.get("sigma", DEFAULT_SIGMA))
        self.lj_cutoff = float(kwargs.get("lj_cutoff", LJ_CUTOFF))
        self.use_nonbonded = kwargs.get("use_nonbonded", True)

    def _lennard_jones_energy_force(
        self, dist: float, epsilon: float, sigma: float
    ) -> tuple[float, float]:
        """Calculate Lennard-Jones energy and force for a given distance.

        Inspired by TinyFF's clean pairwise implementation.

        Parameters
        ----------
        dist : float
            Interatomic distance
        epsilon : float
            LJ well depth parameter
        sigma : float
            LJ size parameter

        Returns
        -------
        tuple
            (energy, force_magnitude)
        """
        if dist > self.lj_cutoff:
            return 0.0, 0.0

        # TinyFF-style calculation: (sigma/r)^6
        x = sigma / dist
        x3 = x * x * x
        x6 = x3 * x3
        x12 = x6 * x6

        # Energy: 4 * epsilon * [(sigma/r)^12 - (sigma/r)^6]
        energy = 4 * epsilon * (x12 - x6)

        # Force magnitude: -dU/dr
        force_mag = 4 * epsilon * (12 * x12 - 6 * x6) / dist

        return energy, force_mag

    def _harmonic_bond_energy_force(self, dist: float, r0: float, k: float) -> tuple[float, float]:
        """Calculate harmonic bond energy and force.

        Parameters
        ----------
        dist : float
            Current bond distance
        r0 : float
            Equilibrium bond distance
        k : float
            Force constant

        Returns
        -------
        tuple
            (energy, force_magnitude)
        """
        dr = dist - r0
        energy = 0.5 * k * dr * dr
        force_mag = -k * dr
        return energy, force_mag

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] | None = None,
    ) -> None:
        """Calculate energy and forces using harmonic potential.

        This method implements a simple harmonic potential based on covalent radii
        to provide realistic energies and forces for testing purposes.

        Parameters
        ----------
        atoms : ase.Atoms, optional
            Atoms object to calculate properties for
        properties : list of str, optional
            Properties to calculate (energy, forces)
        system_changes : list, optional
            System changes since last calculation

        Notes
        -----
        The calculation uses pairwise harmonic bonds based on covalent radii
        and applies harmonic potential around equilibrium bond lengths.
        """
        # Pairwise harmonic bond mock: find bonded pairs using covalent radii
        # and apply harmonic potential around equilibrium bond lengths.
        from ase.data import covalent_radii

        positions = atoms.get_positions()
        numbers = atoms.get_atomic_numbers()
        natoms = len(atoms)

        # Build neighbor list by distance cutoff based on covalent radii
        pairs = []  # list of (i, j, r0)
        for i in range(natoms):
            for j in range(i + 1, natoms):
                r0 = covalent_radii[numbers[i]] + covalent_radii[numbers[j]]
                # Use a tolerance factor; allow a generous minimum cutoff so
                # stretched bonds (e.g., starting geometries) are still
                # recognized by the mock calculator.
                cutoff = max(r0 * BOND_TOLERANCE_FACTOR if r0 > 0 else 1.6, MIN_BOND_CUTOFF)
                rij = positions[i] - positions[j]
                dist = np.linalg.norm(rij)
                if dist <= cutoff:
                    # Treat as bonded pair
                    # key = (int(numbers[i]), int(numbers[j]))  # Unused for now
                    # key_rev = (int(numbers[j]), int(numbers[i]))  # Unused for now
                    r_eq = r0 if r0 > 0 else dist
                    pairs.append((i, j, r_eq))

        forces = np.zeros_like(positions)
        energy = 0.0

        k = float(self.force_constant)
        k_angle = float(self.k_angle)

        # If no bonds found, fall back to weak positional springs to preserve shape
        if not pairs:
            # positional springs to initial positions (no centering)
            ref = positions.copy()
            for i in range(natoms):
                disp = positions[i] - ref[i]
                energy += 0.5 * k * np.dot(disp, disp)
                forces[i] -= k * disp

            # Add non-bonded interactions even when no bonds are found
            if self.use_nonbonded:
                for i in range(natoms):
                    for j in range(i + 1, natoms):
                        rij = positions[i] - positions[j]
                        dist = np.linalg.norm(rij)

                        if dist <= self.lj_cutoff:
                            # Use Lennard-Jones potential for non-bonded interactions
                            lj_energy, lj_force_mag = self._lennard_jones_energy_force(
                                dist, self.epsilon, self.sigma
                            )
                            energy += lj_energy

                            # Force along interatomic direction
                            fij = lj_force_mag * (rij / dist)
                            forces[i] += fij
                            forces[j] -= fij

        else:
            # Track bonded pairs for non-bonded exclusions
            bonded_pairs = set()

            for i, j, r0 in pairs:
                bonded_pairs.add((i, j))
                bonded_pairs.add((j, i))  # Add reverse pair

                rij = positions[i] - positions[j]
                dist = np.linalg.norm(rij)
                if dist == 0:
                    # small random perturbation to avoid singularity
                    rij = np.random.RandomState(0).normal(scale=1e-6, size=3)
                    dist = np.linalg.norm(rij)

                # Use new modular harmonic bond function
                bond_energy, bond_force_mag = self._harmonic_bond_energy_force(dist, r0, k)
                energy += bond_energy

                # force along bond direction
                fij = bond_force_mag * (rij / dist)
                forces[i] += fij
                forces[j] -= fij

            # Simplified angle calculation - only if we have bonds
            if len(pairs) > 1:  # Only calculate angles if we have multiple bonds
                # Build neighbor lists for angle terms
                neighbors: dict[int, list[int]] = {i: [] for i in range(natoms)}
                for i, j, _ in pairs:
                    neighbors[i].append(j)
                    neighbors[j].append(i)

                # Calculate angles for each central atom
                for j in range(natoms):
                    neigh = neighbors[j]
                    if len(neigh) < 2:
                        continue

                    # Calculate angles for all pairs of neighbors
                    for a in range(len(neigh)):
                        for b in range(a + 1, len(neigh)):
                            i, k = neigh[a], neigh[b]
                            r1 = positions[i] - positions[j]
                            r2 = positions[k] - positions[j]

                            # Normalize vectors
                            n1 = np.linalg.norm(r1)
                            n2 = np.linalg.norm(r2)
                            if n1 < 1e-8 or n2 < 1e-8:
                                continue

                            r1_norm = r1 / n1
                            r2_norm = r2 / n2
                            cos_angle = np.dot(r1_norm, r2_norm)

                            # Clamp to avoid numerical issues
                            cos_angle = max(-1.0, min(1.0, cos_angle))

                            # Use cos-based potential instead of angle-based (faster)
                            # E = 0.5 * k_angle * (cos_angle - cos_equilibrium)^2
                            # For simplicity, use cos_equilibrium = 0 (90 degrees)
                            cos_eq = 0.0
                            dcos = cos_angle - cos_eq
                            energy += 0.5 * k_angle * dcos * dcos

                            # Forces (simplified cos-based)
                            force_mag = -k_angle * dcos

                            # Force components
                            fi = force_mag * (r2_norm - cos_angle * r1_norm) / n1
                            fk = force_mag * (r1_norm - cos_angle * r2_norm) / n2
                            fj = -(fi + fk)

                            forces[i] += fi
                            forces[j] += fj
                            forces[k] += fk

            # Add non-bonded interactions (TinyFF-inspired)
            if self.use_nonbonded:
                for i in range(natoms):
                    for j in range(i + 1, natoms):
                        # Skip bonded pairs (1-2 interactions)
                        if (i, j) in bonded_pairs:
                            continue

                        rij = positions[i] - positions[j]
                        dist = np.linalg.norm(rij)

                        if dist <= self.lj_cutoff:
                            # Use Lennard-Jones potential for non-bonded interactions
                            lj_energy, lj_force_mag = self._lennard_jones_energy_force(
                                dist, self.epsilon, self.sigma
                            )
                            energy += lj_energy

                            # Force along interatomic direction
                            fij = lj_force_mag * (rij / dist)
                            forces[i] += fij
                            forces[j] -= fij

        # Remove any net translational force (prevents global drift in tests)
        forces = forces - np.mean(forces, axis=0)

        self.results = {"energy": float(energy), "forces": forces}
        return None


__all__ = ["MockCalculator"]
