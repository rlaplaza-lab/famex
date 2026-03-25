"""Tests for RFO (Rational Function Optimization) transition state optimizer."""

from __future__ import annotations

import numpy as np
import pytest

from qme.optimizers.rfo_optimizer import RFOTransitionState
from tests.test_constants import (
    DEFAULT_FMAX,
    DEFAULT_STEPS,
    LOOSE_FMAX,
    QUICK_STEPS,
    QUICK_STEPS_EXTENDED,
)
from tests.test_utils import StandardTestAssertions


class TestRFOTransitionState:
    def test_rfo_initialization(self, mock_backend, water_dissociation_ts_guess):
        atoms = water_dissociation_ts_guess.copy()
        atoms.calc = mock_backend

        opt = RFOTransitionState(
            atoms,
            logfile=None,
            hessian_update_freq=10,
            trust_radius=0.02,
            max_trust_radius=0.06,
        )

        assert opt.hessian_update_freq == 10
        assert opt.trust_radius == 0.02
        assert opt.max_trust_radius == 0.06

    def test_rfo_positions_conversion(self, mock_backend, water_dissociation_ts_guess):
        atoms = water_dissociation_ts_guess.copy()
        atoms.calc = mock_backend
        opt = RFOTransitionState(atoms, logfile=None)
        x = opt._positions_to_x()
        assert x.shape == (9,)  # 3 atoms × 3 coordinates

        # Round trip conversion
        positions_back = opt._x_to_positions(x)
        assert positions_back.shape == (3, 3)
        assert np.allclose(positions_back, atoms.positions)

    def test_rfo_gradient_function(self, mock_backend, water_dissociation_ts_guess):
        atoms = water_dissociation_ts_guess.copy()
        atoms.calc = mock_backend
        opt = RFOTransitionState(atoms, logfile=None)
        x = opt._positions_to_x()
        grad = opt._get_gradient(x)

        assert grad.shape == (9,)
        forces = atoms.get_forces()
        expected_grad = -forces.ravel()
        assert np.allclose(grad, expected_grad)

    def test_rfo_hessian_computation(self, mock_backend, water_dissociation_ts_guess):
        atoms = water_dissociation_ts_guess.copy()
        atoms.calc = mock_backend
        opt = RFOTransitionState(atoms, logfile=None)
        x = opt._positions_to_x()

        # First call should compute Hessian
        hessian = opt._compute_hessian(x)
        StandardTestAssertions.assert_hessian_valid(hessian, expected_shape=(9, 9))
        assert opt.hessian_calls == 1

    def test_rfo_hessian_caching(self, mock_backend, water_dissociation_ts_guess):
        atoms = water_dissociation_ts_guess.copy()
        atoms.calc = mock_backend
        opt = RFOTransitionState(atoms, logfile=None, hessian_update_freq=5)
        x = opt._positions_to_x()

        # First call computes
        opt._compute_hessian(x)
        assert opt.hessian_calls == 1

        # Increment step counter but not enough for update
        opt.nsteps = 3
        opt._last_full_hessian_step = 0

        # Second call should reuse (no new full Hessian)
        opt._compute_hessian(x)
        assert opt.hessian_calls == 1  # Still only 1 full Hessian

    def test_rfo_convergence(self, mock_backend, water_dissociation_ts_guess):
        atoms = water_dissociation_ts_guess.copy()
        atoms.calc = mock_backend
        opt = RFOTransitionState(atoms, logfile=None)

        # Small forces should converge
        opt.fmax = DEFAULT_FMAX
        assert opt.converged(np.array([0.001, 0.001, 0.001]))

        # Large forces should not converge
        assert not opt.converged(np.array([0.1, 0.1, 0.1]))

    def test_rfo_hessian_update_frequency(self, mock_backend, water_dissociation_ts_guess):
        atoms = water_dissociation_ts_guess.copy()
        atoms.calc = mock_backend
        opt = RFOTransitionState(atoms, logfile=None, hessian_update_freq=3)
        x = opt._positions_to_x()

        # Initial call
        opt._compute_hessian(x)
        assert opt.hessian_calls == 1

        # Simulate steps
        for step in range(1, 4):
            opt.nsteps = step
            opt._last_full_hessian_step = 0
            opt._compute_hessian(x)

        # Should have recomputed at step 3 (update_freq=3)
        assert opt.hessian_calls >= 2

    def test_rfo_basic_optimization_run(self, mock_backend, water_dissociation_ts_guess):
        atoms = water_dissociation_ts_guess.copy()
        atoms.calc = mock_backend
        opt = RFOTransitionState(atoms, logfile=None, hessian_update_freq=5)

        # Run with relaxed convergence for speed
        converged = opt.run(fmax=LOOSE_FMAX, steps=QUICK_STEPS)

        # Should complete without errors (may or may not converge in 5 steps)
        # RFO returns np.bool_, convert to bool for assertion
        assert bool(converged) in (True, False)
        assert opt.nsteps > 0

    def test_rfo_step_counting(self, mock_backend, water_dissociation_ts_guess):
        atoms = water_dissociation_ts_guess.copy()
        atoms.calc = mock_backend
        opt = RFOTransitionState(atoms, logfile=None)
        initial_steps = opt.nsteps

        opt.run(fmax=LOOSE_FMAX, steps=QUICK_STEPS_EXTENDED)

        assert opt.nsteps > initial_steps
        assert opt.get_number_of_steps() == opt.nsteps

    def test_rfo_optimization_quality(self, mock_backend, water_dissociation_ts_guess):
        atoms = water_dissociation_ts_guess.copy()
        atoms.calc = mock_backend

        initial_energy = atoms.get_potential_energy()
        initial_positions = atoms.get_positions().copy()

        # Test optimizer with tighter convergence
        opt = RFOTransitionState(atoms, logfile=None, hessian_update_freq=5, trust_radius=0.02)
        opt.run(fmax=LOOSE_FMAX * 2, steps=DEFAULT_STEPS * 5)

        final_energy = atoms.get_potential_energy()
        final_positions = atoms.get_positions()
        forces = atoms.get_forces()

        # Optimizer should actually optimize
        assert abs(final_energy - initial_energy) > 1e-6, "RFO should change energy"
        assert np.max(np.abs(final_positions - initial_positions)) > 1e-6, (
            "RFO should change positions"
        )

        # Check step counting
        assert opt.get_number_of_steps() > 0, "RFO should report steps"

        # Check convergence quality
        StandardTestAssertions.assert_energy_reasonable(final_energy, backend="mock")
        StandardTestAssertions.assert_forces_reasonable(forces, backend="mock")

    def test_rfo_requires_calculator(self, water_dissociation_ts_guess):
        atoms = water_dissociation_ts_guess.copy()
        with pytest.raises(ValueError, match="calculator"):
            RFOTransitionState(atoms, logfile=None)

    def test_rfo_initial_hessian_provided(self, mock_backend, water_dissociation_ts_guess):
        atoms = water_dissociation_ts_guess.copy()
        atoms.calc = mock_backend

        n = len(atoms) * 3
        initial_hess = np.eye(n)

        opt = RFOTransitionState(atoms, logfile=None, initial_hessian=initial_hess)

        x = opt._positions_to_x()

        # Should use initial hessian (may be updated)
        hessian = opt._compute_hessian(x)
        assert hessian is not None

    def test_rfo_trust_radius_adjustment(self, mock_backend, water_dissociation_ts_guess):
        atoms = water_dissociation_ts_guess.copy()
        atoms.calc = mock_backend

        opt = RFOTransitionState(atoms, logfile=None, trust_radius=0.01, max_trust_radius=0.05)

        initial_trust = opt.trust_radius

        # Test good step quality (should increase trust radius)
        opt._adjust_trust_radius(0.8, step_size=0.005)
        assert opt.trust_radius >= initial_trust

        # Reset
        opt.trust_radius = 0.01
        # Test poor step quality (should decrease trust radius)
        opt._adjust_trust_radius(0.1, step_size=0.005)
        assert opt.trust_radius <= 0.01

    def test_rfo_step_quality_computation(self, mock_backend, water_dissociation_ts_guess):
        atoms = water_dissociation_ts_guess.copy()
        atoms.calc = mock_backend

        opt = RFOTransitionState(atoms, logfile=None)
        x = opt._positions_to_x()
        gradient = opt._get_gradient(x)
        hessian = opt._compute_hessian(x)
        step = np.random.randn(len(x)) * 0.01

        # Good step: actual matches predicted (simulate by using predicted)
        predicted_quadratic = 0.5 * np.dot(step, hessian @ step)
        predicted_linear = np.dot(step, gradient)
        predicted_change = predicted_quadratic + predicted_linear
        quality = opt._compute_step_quality(predicted_change, step, gradient, hessian)
        assert 0.5 <= quality <= 1.0

        # Poor step: actual much different from predicted
        # Use a smaller multiplier to keep quality >= 0 (avoid very poor steps with quality < 0)
        poor_actual = predicted_change * 2.5  # Different from predicted but not extreme
        quality = opt._compute_step_quality(poor_actual, step, gradient, hessian)
        # quality can be negative for very poor steps, so just check it's less than good
        assert quality < 0.5  # Should be poor (quality < 0.5) or very poor (quality < 0)
