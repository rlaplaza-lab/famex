"""Tests for IDPP (ASE match) and DHS strategy."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase.mep.neb import IDPP

from famex.core.explorer import Explorer
from famex.interpolation.strategies import IDPPInterpolation, idpp_energy_and_forces
from famex.strategies.ts_dhs import (
    MultiStructureTSDHSStrategy,
    _euclidean_distance,
    distance_constrained_minimize,
)


class TestIDPPMatchesASE:
    def test_energy_and_forces(self):
        rng = np.random.default_rng(42)
        positions = rng.normal(size=(5, 3))
        target = rng.uniform(0.8, 2.5, size=(5, 5))
        target = 0.5 * (target + target.T)
        np.fill_diagonal(target, 0.0)

        atoms = Atoms("H" * 5, positions=positions)
        atoms.calc = IDPP(target, mic=False)
        energy, forces = idpp_energy_and_forces(positions, target)
        assert energy == pytest.approx(atoms.get_potential_energy())
        np.testing.assert_allclose(forces, atoms.get_forces(), atol=1e-12)

    def test_interpolation_preserves_endpoints(self):
        start = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        end = np.array([[0.0, 0.0, 0.1], [1.2, 0.1, 0.0], [0.1, 1.1, 0.0]])
        path = IDPPInterpolation(steps=20).interpolate(start, end, 5)
        assert len(path) == 5
        np.testing.assert_allclose(path[0], start)
        np.testing.assert_allclose(path[-1], end)


class TestDHS:
    def test_strategy_metadata(self):
        assert MultiStructureTSDHSStrategy.metadata.name == "ts:dhs"
        assert MultiStructureTSDHSStrategy.metadata.strategy == "dhs"
        assert "dhs" in MultiStructureTSDHSStrategy.metadata.aliases

    def test_requires_two_structures(self, water_molecule):
        explorer = Explorer(
            [water_molecule, water_molecule.copy()],
            backend="mock",
            target="ts",
            strategy="dhs",
        )
        strategy = MultiStructureTSDHSStrategy(explorer)
        with pytest.raises(ValueError, match="multiple structures"):
            strategy.run([water_molecule], steps=2)

    def test_distance_decreases_on_macro_step(self):
        left = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
        right = Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        left.calc = EMT()
        right.calc = EMT()
        dist0 = _euclidean_distance(left, right)

        moving, pivot = (
            (left, right)
            if left.get_potential_energy() <= right.get_potential_energy()
            else (right, left)
        )
        direction = pivot.positions - moving.positions
        moving.positions = moving.positions + 0.2 * direction / np.linalg.norm(direction)
        distance_constrained_minimize(moving, pivot, fmax=0.5, max_steps=10, trust_radius=0.1)
        assert _euclidean_distance(left, right) < dist0

    def test_run_returns_refined_structure(self, water_molecule):
        reactant = water_molecule.copy()
        product = water_molecule.copy()
        pos = product.get_positions()
        pos[:, 0] += 0.8
        product.set_positions(pos)

        explorer = Explorer([reactant, product], backend="mock", target="ts", strategy="dhs")
        strategy = MultiStructureTSDHSStrategy(explorer)
        fake_result = {
            "optimized_atoms": reactant.copy(),
            "converged": True,
            "steps_taken": 3,
        }

        def _attach(_explorer, images):
            for img in images:
                img.calc = EMT()

        with (
            patch("famex.strategies.ts_dhs.LocalTSStrategy.run", return_value=fake_result),
            patch("famex.strategies.ts_dhs.PathManager.attach_calculators", side_effect=_attach),
        ):
            result = strategy.run(
                [reactant, product],
                fmax=1.0,
                steps=5,
                dist_tol=0.4,
                constrain_steps=5,
                calculate_frequencies=False,
            )
        assert isinstance(result["optimized_atoms"], Atoms)
