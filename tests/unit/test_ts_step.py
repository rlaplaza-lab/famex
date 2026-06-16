"""Unit tests for shared TS step utilities."""

from __future__ import annotations

import numpy as np
import pytest

from famex.optimizers.ts_step import (
    NULL_MODE_THRESHOLD,
    adjust_trust_radius,
    build_translation_rotation_basis,
    compute_dense_prfo_step,
    compute_matrix_free_ts_step,
    compute_step_quality,
    make_atoms_hessp,
    project_gradient,
    project_hessian,
    select_transition_mode,
)


class TestTSStep:
    def test_translation_rotation_basis_shape(self):
        positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        masses = np.array([16.0, 1.0])
        basis = build_translation_rotation_basis(positions, masses)
        assert basis.shape[0] == 6
        assert basis.shape[1] >= 5

    def test_project_gradient_removes_translation(self):
        n = 9
        basis = np.zeros((n, 3))
        basis[0::3, 0] = 1.0 / np.sqrt(3)
        basis[1::3, 1] = 1.0 / np.sqrt(3)
        basis[2::3, 2] = 1.0 / np.sqrt(3)
        q, _ = np.linalg.qr(basis)
        gradient = np.ones(n)
        projected = project_gradient(gradient, q)
        assert np.allclose(q.T @ projected, 0.0, atol=1e-10)

    def test_select_transition_mode_follows_previous(self):
        eigenvalues = np.array([-0.1, 0.2, 0.5])
        eigenvectors = np.eye(3)
        idx, mode, omega = select_transition_mode(eigenvalues, eigenvectors, eigenvectors[:, 1])
        assert idx == 1
        assert omega == 0.2

    def test_select_transition_mode_ignores_null_eigenvalues(self):
        eigenvalues = np.array([0.0, 0.0, -0.5, 1.0])
        eigenvectors = np.eye(4)
        idx, _mode, omega = select_transition_mode(eigenvalues, eigenvectors, None)
        assert idx == 2
        assert omega == -0.5

    def test_select_transition_mode_raises_without_vibrational_modes(self):
        eigenvalues = np.array([0.0, 1e-6, -1e-6])
        eigenvectors = np.eye(3)
        with pytest.raises(RuntimeError, match="no vibrational eigenvalues"):
            select_transition_mode(
                eigenvalues, eigenvectors, None, null_threshold=NULL_MODE_THRESHOLD
            )

    def test_matrix_free_predicted_uses_real_quadratic_model(self):
        gradient = np.array([0.1, -0.2, 0.05, 0.0])
        hessian = np.diag([-0.5, 0.3, 0.4, 0.5])

        def hessp(vec: np.ndarray) -> np.ndarray:
            return hessian @ vec

        result = compute_matrix_free_ts_step(
            gradient=gradient,
            hessp=hessp,
            trust_radius=0.2,
            subproblem="ncg",
        )
        expected_predicted = float(
            np.dot(result.step, gradient) + 0.5 * np.dot(result.step, hessp(result.step)),
        )
        assert result.predicted_energy_change == pytest.approx(expected_predicted)

    def test_step_quality_increases_trust_radius_on_good_match(self):
        predicted = 0.05
        actual = 0.05
        quality = compute_step_quality(actual, predicted)
        assert quality >= 0.75
        new_trust = adjust_trust_radius(0.1, quality, 0.05, 0.001, 0.3)
        assert new_trust > 0.1

    def test_make_atoms_hessp_caches_base_forces(self):
        from ase import Atoms

        from famex.potentials.mock_potential import MockCalculator

        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        atoms.calc = MockCalculator()
        x = atoms.get_positions().ravel()
        call_count = 0
        original_get_forces = atoms.get_forces

        def counting_get_forces():
            nonlocal call_count
            call_count += 1
            return original_get_forces()

        atoms.get_forces = counting_get_forces  # type: ignore[method-assign]
        hessp = make_atoms_hessp(atoms, x, lambda arr: arr.reshape(-1, 3))
        vec = np.ones(6)
        hessp(vec)
        hessp(vec)
        assert call_count == 3

    def test_make_atoms_hessp_invalidates_cache_on_position_change(self):
        from ase import Atoms

        from famex.potentials.mock_potential import MockCalculator

        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        atoms.calc = MockCalculator()
        x0 = atoms.get_positions().ravel()
        hessp0 = make_atoms_hessp(atoms, x0, lambda arr: arr.reshape(-1, 3))
        call_count = 0
        original_get_forces = atoms.get_forces

        def counting_get_forces():
            nonlocal call_count
            call_count += 1
            return original_get_forces()

        atoms.get_forces = counting_get_forces  # type: ignore[method-assign]
        vec = np.ones(6)
        hessp0(vec)
        x1 = x0.copy()
        x1[2] += 0.01
        hessp1 = make_atoms_hessp(atoms, x1, lambda arr: arr.reshape(-1, 3))
        hessp1(vec)
        assert call_count == 4

    def test_dense_prfo_step_respects_trust_radius(self):
        n = 6
        rng = np.random.default_rng(0)
        hessian = rng.standard_normal((n, n))
        hessian = hessian + hessian.T
        gradient = rng.standard_normal(n)
        trust_radius = 0.1
        result = compute_dense_prfo_step(
            gradient=gradient,
            hessian=hessian,
            trust_radius=trust_radius,
        )
        assert result.step.shape == (n,)
        assert np.linalg.norm(result.step) <= trust_radius * 1.01

    def test_step_quality_perfect_match(self):
        assert compute_step_quality(1.0, 1.0) == 1.0

    def test_adjust_trust_radius_increases_on_good_step(self):
        new = adjust_trust_radius(0.1, 0.8, 0.05, 0.001, 0.3)
        assert new > 0.1

    def test_project_hessian_symmetry(self):
        positions = np.array([[0, 0, 0], [0, 0, 1.0], [1.0, 0, 0.0]])
        masses = np.array([12.0, 1.0, 1.0])
        n = len(positions) * 3
        hessian = np.random.default_rng(1).standard_normal((n, n))
        hessian = hessian + hessian.T
        basis = build_translation_rotation_basis(positions, masses)
        projected = project_hessian(hessian, basis)
        assert projected.shape == (n, n)
        assert np.allclose(projected, projected.T, atol=1e-10)
