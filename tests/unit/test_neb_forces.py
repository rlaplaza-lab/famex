"""Tests for NEB forces against ASE improvedtangent reference."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.mep import NEB

from famex.strategies.neb_optimizer import (
    NEBOptimizer,
    compute_neb_forces,
    energy_weighted_spring_constants,
)


def _make_band(
    n: int = 5,
    natoms: int = 3,
    seed: int = 0,
    energies: list[float] | None = None,
) -> tuple[list[Atoms], list[float], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    if energies is None:
        energies = [float(0.1 * i + 0.05 * (i - (n - 1) / 2) ** 2) for i in range(n)]
    images: list[Atoms] = []
    forces: list[np.ndarray] = []
    for i in range(n):
        pos = rng.normal(size=(natoms, 3)) + np.array([0.4 * i, 0.0, 0.0])
        f = rng.normal(size=(natoms, 3))
        atoms = Atoms("H" * natoms, positions=pos)
        atoms.calc = SinglePointCalculator(atoms, energy=energies[i], forces=f)
        images.append(atoms)
        forces.append(f.copy())
    return images, list(energies), forces


class TestNEBForcesMatchASE:
    def test_uniform_k_no_climb(self):
        images, energies, forces = _make_band()
        k = 0.1
        ase_neb = NEB(
            images,
            k=k,
            climb=False,
            method="improvedtangent",
            allow_shared_calculator=True,
        )
        ase_forces = ase_neb.get_forces()
        positions = [a.positions for a in images]
        our_forces, climbing = compute_neb_forces(positions, energies, forces, k=k, climb=False)
        our_interior = np.concatenate([f.ravel() for f in our_forces[1:-1]]).reshape(-1, 3)
        assert climbing is None
        np.testing.assert_allclose(our_interior, ase_forces, atol=1e-12)

    def test_climb_matches_ase(self):
        energies = [0.0, 0.4, 1.2, 0.5, 0.1]
        images, _, forces = _make_band(energies=energies)
        for atoms, e, f in zip(images, energies, forces, strict=True):
            atoms.calc = SinglePointCalculator(atoms, energy=e, forces=f)
        k = 0.1
        ase_neb = NEB(
            images,
            k=k,
            climb=True,
            method="improvedtangent",
            allow_shared_calculator=True,
        )
        ase_forces = ase_neb.get_forces()
        positions = [a.positions for a in images]
        our_forces, climbing = compute_neb_forces(positions, energies, forces, k=k, climb=True)
        our_interior = np.concatenate([f.ravel() for f in our_forces[1:-1]]).reshape(-1, 3)
        assert climbing == ase_neb.imax
        np.testing.assert_allclose(our_interior, ase_forces, atol=1e-12)

    def test_per_spring_k_array(self):
        images, energies, forces = _make_band()
        k_list = [0.05, 0.1, 0.2, 0.15]
        ase_neb = NEB(
            images,
            k=k_list,
            climb=False,
            method="improvedtangent",
            allow_shared_calculator=True,
        )
        ase_forces = ase_neb.get_forces()
        positions = [a.positions for a in images]
        our_forces, _ = compute_neb_forces(positions, energies, forces, k=k_list, climb=False)
        our_interior = np.concatenate([f.ravel() for f in our_forces[1:-1]]).reshape(-1, 3)
        np.testing.assert_allclose(our_interior, ase_forces, atol=1e-12)

    def test_endpoints_fixed_zero_force(self):
        images, energies, forces = _make_band()
        positions = [a.positions for a in images]
        our_forces, _ = compute_neb_forces(positions, energies, forces, k=0.1, climb=False)
        np.testing.assert_allclose(our_forces[0], 0.0)
        np.testing.assert_allclose(our_forces[-1], 0.0)

    def test_energy_weighted_spring_constants(self):
        energies = [0.0, 0.5, 1.5, 0.6, 0.1]
        k_u, k_l = 5.0, 0.5
        k = energy_weighted_spring_constants(energies, k_u=k_u, k_l=k_l)
        assert len(k) == 4
        # Highest-energy spring (around image 2) should be stiffest
        assert k[1] == pytest.approx(k_u)
        assert k[2] == pytest.approx(k_u)
        assert k[0] < k_u
        assert k[3] < k_u

    def test_neb_optimizer_uses_energy_weighted_k_when_climb(self):
        energies = [0.0, 0.5, 1.5, 0.6, 0.1]
        images, _, forces = _make_band(energies=energies)
        for atoms, e, f in zip(images, energies, forces, strict=True):
            atoms.calc = SinglePointCalculator(atoms, energy=e, forces=f)
        opt = NEBOptimizer(images, spring_constant=5.0, climb=True, fmax=1.0, steps=1)
        k = opt.get_spring_constants(energies)
        expected = energy_weighted_spring_constants(energies, k_u=5.0, k_l=0.5)
        np.testing.assert_allclose(k, expected)
        # Uniform when climb is False
        opt_plain = NEBOptimizer(images, spring_constant=5.0, climb=False, fmax=1.0, steps=1)
        k_plain = opt_plain.get_spring_constants(energies)
        np.testing.assert_allclose(k_plain, 5.0)
