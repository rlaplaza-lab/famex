"""A lightweight MockCalculator for tests and CI.

Implements a minimal ASE Calculator-like interface to return simple
energies and harmonic forces so tests can run without heavy ML deps.
"""

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
        # angle force constant (harmonic in angle space). Default is
        # moderately weaker than bond force to avoid over-constraining
        # small test systems.
        self.k_angle = float(kwargs.get("k_angle", 0.5 * self.force_constant))

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
        k_angle = float(self.k_angle)

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

            # Build neighbor lists to find bonded triplets for angle terms
            neighbors = {i: set() for i in range(natoms)}
            for i, j, _ in pairs:
                neighbors[i].add(j)
                neighbors[j].add(i)

            # Reference positions (initial geometry) used to compute theta0
            ref = positions.copy()
            triplets = []  # list of (i, j, k, theta0)
            for j in range(natoms):
                neigh = sorted(neighbors[j])
                # form unique i < k triplets
                for a in range(len(neigh)):
                    for b in range(a + 1, len(neigh)):
                        i = neigh[a]
                        k_idx = neigh[b]
                        r1 = ref[i] - ref[j]
                        r2 = ref[k_idx] - ref[j]
                        n1 = np.linalg.norm(r1)
                        n2 = np.linalg.norm(r2)
                        if n1 == 0 or n2 == 0:
                            continue
                        cos0 = np.dot(r1, r2) / (n1 * n2)
                        cos0 = np.clip(cos0, -1.0, 1.0)
                        theta0 = np.arccos(cos0)
                        triplets.append((i, j, k_idx, float(theta0)))

            # Angle harmonic potentials (θ - θ0)^2
            for i, j, k_idx, theta0 in triplets:
                r1 = positions[i] - positions[j]
                r2 = positions[k_idx] - positions[j]
                n1 = np.linalg.norm(r1)
                n2 = np.linalg.norm(r2)
                if n1 == 0 or n2 == 0:
                    continue
                cosang = np.dot(r1, r2) / (n1 * n2)
                cosang = np.clip(cosang, -1.0, 1.0)
                theta = np.arccos(cosang)

                # energy contribution
                dtheta = theta - theta0
                energy += 0.5 * k_angle * (dtheta * dtheta)

                # derivative dU/dtheta = k * (theta - theta0)
                dU_dtheta = k_angle * dtheta

                # avoid division by very small sin(theta)
                sin_theta = max(1e-8, np.sqrt(max(0.0, 1.0 - cosang * cosang)))

                A = r1 / n1
                B = r2 / n2

                coeff_i = -dU_dtheta / (n1 * sin_theta)
                coeff_k = -dU_dtheta / (n2 * sin_theta)

                fi = coeff_i * (B - A * cosang)
                fk = coeff_k * (A - B * cosang)
                fj = -(fi + fk)

                forces[i] += fi
                forces[j] += fj
                forces[k_idx] += fk

        # Remove any net translational force (prevents global drift in tests)
        forces = forces - np.mean(forces, axis=0)

        self.results = {"energy": float(energy), "forces": forces}
        return None


__all__ = ["MockCalculator"]
