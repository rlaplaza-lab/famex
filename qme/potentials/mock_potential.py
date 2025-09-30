"""A lightweight MockCalculator for tests and CI.

Implements a minimal ASE Calculator-like interface to return simple
energies and harmonic forces so tests can run without heavy ML deps.
"""

from typing import Any, Dict

import numpy as np
from ase.calculators.calculator import Calculator, all_changes


class MockCalculator(Calculator):
    """Simple mock calculator.

    Parameters (allowed): backend, force_constant, bond_length, charge, mult
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, backend: str = "generic", force_constant: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.backend = backend
        self.force_constant = float(force_constant)
        # simple parameters
        self.charge = kwargs.get("charge", 0)
        self.mult = kwargs.get("mult", 1)

    def calculate(self, atoms=None, properties=None, system_changes=None):
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
                cutoff = max(r0 * 1.5 if r0 > 0 else 1.6, 2.0)
                rij = positions[i] - positions[j]
                dist = np.linalg.norm(rij)
                if dist <= cutoff:
                    # Treat as bonded pair
                    key = (int(numbers[i]), int(numbers[j]))
                    key_rev = (int(numbers[j]), int(numbers[i]))
                    r_eq = r0 if r0 > 0 else dist
                    pairs.append((i, j, r_eq))

        forces = np.zeros_like(positions)
        energy = 0.0

        k = float(self.force_constant)

        # If no bonds found, fall back to weak positional springs to preserve shape
        if not pairs:
            # positional springs to initial positions (no centering)
            ref = positions.copy()
            for i in range(natoms):
                disp = positions[i] - ref[i]
                energy += 0.5 * k * np.dot(disp, disp)
                forces[i] -= k * disp

        else:
            for i, j, r0 in pairs:
                rij = positions[i] - positions[j]
                dist = np.linalg.norm(rij)
                if dist == 0:
                    # small random perturbation to avoid singularity
                    rij = np.random.RandomState(0).normal(scale=1e-6, size=3)
                    dist = np.linalg.norm(rij)

                dr = dist - r0
                energy += 0.5 * k * dr * dr

                # force magnitude along bond
                fmag = -k * dr
                fij = fmag * (rij / dist)
                forces[i] += fij
                forces[j] -= fij

        self.results = {"energy": float(energy), "forces": forces}
        return None


__all__ = ["MockCalculator"]
