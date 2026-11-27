"""Unit tests for famex.utils.hessian_ops."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from famex.utils.hessian_ops import (
    MAX_CONDITION_NUMBER,
    bond_key,
    build_translation_rotation_basis,
    condition_number_and_vibrational_eigenvalues,
    frequencies_from_hessian,
    optional_int,
    project_gradient,
    project_hessian,
    symmetrize_hessian,
)


def test_optional_int():
    assert optional_int(3) == 3
    assert optional_int("7") == 7
    assert optional_int(None) is None
    assert optional_int("x") is None


def test_bond_key():
    assert bond_key(2, 0) == (0, 2)
    assert bond_key(1, 1) == (1, 1)


def test_max_condition_number_constant():
    assert MAX_CONDITION_NUMBER == 1e18


def test_build_translation_rotation_basis_orthonormal():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    masses = np.ones(3)
    basis = build_translation_rotation_basis(positions, masses)
    assert basis.ndim == 2
    assert basis.shape[0] == 9
    gram = basis.T @ basis
    np.testing.assert_allclose(gram, np.eye(gram.shape[0]), atol=1e-10)


def test_project_gradient_removes_translation():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    masses = np.ones(3)
    basis = build_translation_rotation_basis(positions, masses)
    gradient = np.ones(9)
    projected = project_gradient(gradient, basis)
    for col in range(basis.shape[1]):
        assert abs(float(projected @ basis[:, col])) < 1e-10


def test_project_hessian_symmetry():
    rng = np.random.default_rng(0)
    hessian = rng.normal(size=(9, 9))
    hessian = symmetrize_hessian(hessian)
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    masses = np.ones(3)
    basis = build_translation_rotation_basis(positions, masses)
    projected = project_hessian(hessian, basis)
    np.testing.assert_allclose(projected, projected.T, atol=1e-12)


def test_condition_number_and_vibrational_eigenvalues():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    masses = np.ones(3)
    hessian = project_hessian(np.eye(9) * 2.0, build_translation_rotation_basis(positions, masses))
    cond, vibrational, n_null = condition_number_and_vibrational_eigenvalues(
        hessian, positions=positions, masses=masses
    )
    assert n_null > 0
    assert len(vibrational) > 0
    assert cond < 10.0


def test_frequencies_from_hessian():
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    hessian = np.eye(6) * 10.0
    freqs, modes = frequencies_from_hessian(hessian, atoms, project_tr=True)
    assert freqs.ndim == 1
    assert modes.ndim == 2
    assert len(freqs) == modes.shape[1] or modes.shape[0] == 6
