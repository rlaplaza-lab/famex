"""Tests for the PySCF DFT backend."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from famex.potentials.pyscf_potential import (
    DEFAULT_BASIS,
    DEFAULT_XC,
    PySCFPotential,
    parse_pyscf_model_name,
    xc_needs_dispersion,
)
from tests.test_utils import requires_backend


@pytest.mark.parametrize(
    ("model_name", "expected_xc", "expected_basis"),
    [
        (None, DEFAULT_XC, DEFAULT_BASIS),
        ("pbe0-d3bj/def2-tzvp", "pbe0-d3bj", "def2-tzvp"),
        ("wb97x-d3bj", "wb97x-d3bj", DEFAULT_BASIS),
    ],
)
def test_parse_pyscf_model_name(model_name, expected_xc, expected_basis):
    assert parse_pyscf_model_name(model_name) == (expected_xc, expected_basis)


def test_xc_needs_dispersion():
    assert xc_needs_dispersion("b3lyp-d3bj")
    assert not xc_needs_dispersion("b3lyp")


@requires_backend("pyscf")
def test_h2_energy_and_forces():
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    calc = PySCFPotential(verbose=0)
    energy = calc.get_potential_energy(atoms)
    forces = calc.get_forces(atoms)
    assert np.isfinite(energy)
    assert forces.shape == (2, 3)
    assert np.linalg.norm(forces) > 0.0
