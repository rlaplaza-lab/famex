"""Tests for libmolsymm 2.0 force/Hessian projection wiring."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT

from famex.analysis.symmetry import (
    MolecularSymmetry,
    MolecularSymmetryPolicy,
    endpoints_share_point_group_axis,
)
from famex.core.explorer import Explorer
from famex.optimizers.ase_wrappers import SymmetryCalculatorWrapper


@pytest.fixture
def water_c2v() -> Atoms:
    return Atoms(
        "OHH",
        positions=[
            [0.0, 0.0, 0.1173],
            [0.0, 0.7572, -0.4692],
            [0.0, -0.7572, -0.4692],
        ],
    )


@pytest.fixture
def near_c2v_water() -> Atoms:
    return Atoms(
        "OHH",
        positions=[
            [0.0, 0.0, 0.1173],
            [0.001, 0.7572, -0.4692],
            [-0.001, -0.7572, -0.4692],
        ],
    )


@pytest.fixture
def methane_td() -> Atoms:
    return Atoms(
        "CHHHH",
        positions=[
            [0.0, 0.0, 0.0],
            [0.6276, 0.6276, 0.6276],
            [-0.6276, -0.6276, 0.6276],
            [-0.6276, 0.6276, -0.6276],
            [0.6276, -0.6276, -0.6276],
        ],
    )


def test_auto_detects_c2v_without_moving_atoms(water_c2v: Atoms) -> None:
    original = water_c2v.get_positions().copy()
    sym = MolecularSymmetry(MolecularSymmetryPolicy(), atoms_list=[water_c2v])
    assert sym.resolved_point_group == "C2v"
    assert sym.should_project is True
    np.testing.assert_allclose(water_c2v.get_positions(), original)


def test_auto_projects_asymmetric_forces(water_c2v: Atoms) -> None:
    sym = MolecularSymmetry(MolecularSymmetryPolicy(), atoms_list=[water_c2v])
    forces = np.array(
        [[0.0, 0.0, 0.1], [0.05, 0.02, -0.05], [0.0, -0.02, -0.05]],
        dtype=np.float64,
    )
    projected = sym.project_forces(water_c2v, forces)
    assert abs(float(projected[1, 0] + projected[2, 0])) < 1e-8
    assert abs(float(projected[1, 0] - projected[2, 0])) < 1e-8
    assert np.linalg.norm(projected - forces) > 1e-6


def test_user_point_group_projects_geometry(near_c2v_water: Atoms) -> None:
    before = near_c2v_water.get_positions().copy()
    sym = MolecularSymmetry(
        MolecularSymmetryPolicy(point_group="C2v", primary=1e-2),
        atoms_list=[near_c2v_water],
    )
    assert sym.resolved_point_group == "C2v"
    assert sym.should_project is True
    assert np.linalg.norm(near_c2v_water.get_positions() - before) > 0.0


def test_user_point_group_mismatch_raises(water_c2v: Atoms) -> None:
    with pytest.raises(ValueError, match="not compatible with point group 'Td'"):
        MolecularSymmetry(
            MolecularSymmetryPolicy(point_group="Td"),
            atoms_list=[water_c2v],
        )


def test_symmetry_disabled_passthrough(water_c2v: Atoms) -> None:
    sym = MolecularSymmetry(
        MolecularSymmetryPolicy(enabled=False),
        atoms_list=[water_c2v],
    )
    assert sym.should_project is False
    forces = np.arange(9, dtype=np.float64).reshape(3, 3)
    np.testing.assert_allclose(sym.project_forces(water_c2v, forces), forces)


def test_endpoints_compatible_same_water(water_c2v: Atoms) -> None:
    other = water_c2v.copy()
    other.translate([1.0, 0.0, 0.0])
    assert endpoints_share_point_group_axis(water_c2v, other) is True


def test_endpoints_incompatible_water_and_methane(water_c2v: Atoms, methane_td: Atoms) -> None:
    assert endpoints_share_point_group_axis(water_c2v, methane_td) is False


def test_path_symmetry_requires_shared_axis(water_c2v: Atoms) -> None:
    aligned = MolecularSymmetry(
        MolecularSymmetryPolicy(),
        atoms_list=[water_c2v, water_c2v.copy()],
        is_path=True,
    )
    assert aligned.should_project is True

    rotated = water_c2v.copy()
    rotated.rotate(90, "x")
    assert endpoints_share_point_group_axis(water_c2v, rotated) is False
    mismatched = MolecularSymmetry(
        MolecularSymmetryPolicy(),
        atoms_list=[water_c2v, rotated],
        is_path=True,
    )
    assert mismatched.should_project is False


def test_explorer_wrapper_projects_forces(water_c2v: Atoms) -> None:
    explorer = Explorer(water_c2v.copy(), backend="mock", symmetry=True, verbose=0)
    atoms = explorer.atoms_list[0]
    explorer._create_and_attach_calculator(atoms)
    assert isinstance(atoms.calc, SymmetryCalculatorWrapper)
    assert explorer.molecular_symmetry.resolved_point_group == "C2v"

    atoms.calc = SymmetryCalculatorWrapper(EMT(), explorer.molecular_symmetry)
    forces = atoms.get_forces()
    assert abs(float(forces[1, 0] + forces[2, 0])) < 1e-5


def test_explorer_symmetry_false_skips_wrapper(water_c2v: Atoms) -> None:
    explorer = Explorer(water_c2v.copy(), backend="mock", symmetry=False, verbose=0)
    atoms = explorer.atoms_list[0]
    explorer._create_and_attach_calculator(atoms)
    assert not isinstance(atoms.calc, SymmetryCalculatorWrapper)
    assert explorer.molecular_symmetry.should_project is False
