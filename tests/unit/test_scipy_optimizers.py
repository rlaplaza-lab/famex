"""Test SciPy Hessian-based optimizers.

This module tests the SciPy optimizer wrappers (TrustKrylov, TrustNCG,
TrustExact, NewtonCG) including Hessian computation, BFGS updates,
and convergence behavior.
"""

import numpy as np
import pytest
from ase import Atoms

import qme
from qme.optimizers.scipy_optimizers import NewtonCG, TrustExact, TrustKrylov, TrustNCG
from tests.test_utils import StandardTestAssertions, TestMoleculeFactory


@pytest.fixture
def h2o_perturbed():
    """Water molecule with small perturbation for optimization tests."""
    return TestMoleculeFactory.get_perturbed_molecule(
        TestMoleculeFactory.get_h2o_equilibrium(), seed=42, magnitude=0.05
    )


@pytest.fixture
def h2o_distorted():
    """Water molecule with larger distortion for convergence tests."""
    return Atoms("H2O", positions=[[0, 0, 0], [0.8, 0.6, 0], [-0.8, 0.6, 0]])


class TestSciPyOptimizers:
    """Comprehensive tests for all SciPy optimizer variants."""

    @pytest.mark.parametrize(
        ("optimizer_class", "method"),
        [
            (TrustKrylov, "trust-krylov"),
            (TrustNCG, "trust-ncg"),
            (TrustExact, "trust-exact"),
            (NewtonCG, "Newton-CG"),
        ],
    )
    def test_optimizer_initialization(self, optimizer_class, method, h2o_perturbed):
        """Test initialization of all optimizer types."""
        h2o = h2o_perturbed.copy()
        h2o.calc = qme.MockCalculator(backend="mock")

        opt = optimizer_class(h2o, logfile=None)
        assert opt.method == method

    @pytest.mark.parametrize("optimizer_class", [TrustKrylov, TrustNCG, TrustExact, NewtonCG])
    def test_optimizer_accepts_all_parameters(self, optimizer_class, h2o_perturbed):
        """Test that all optimizers accept standard parameters."""
        h2o = h2o_perturbed.copy()
        h2o.calc = qme.MockCalculator(backend="mock")

        opt = optimizer_class(
            h2o,
            logfile=None,
            hessian_update_freq=5,
            hessian_method="finite_differences",
            hessian_delta=0.02,
            use_bfgs_update=True,
            adaptive_hessian=True,
        )

        assert opt.hessian_update_freq == 5
        assert opt.hessian_method == "finite_differences"
        assert opt.use_bfgs_update is True
        assert opt.adaptive_hessian is True

    def test_optimizer_with_periodic_updates(self, h2o_perturbed):
        """Test optimizer with periodic Hessian updates."""
        h2o = h2o_perturbed.copy()
        h2o.calc = qme.MockCalculator(backend="mock")
        opt = TrustKrylov(h2o, logfile=None, hessian_update_freq=5)
        assert opt.hessian_update_freq == 5

    def test_optimizer_with_adaptive_mode(self, h2o_perturbed):
        """Test optimizer with adaptive Hessian updates."""
        h2o = h2o_perturbed.copy()
        h2o.calc = qme.MockCalculator(backend="mock")
        opt = TrustKrylov(
            h2o,
            logfile=None,
            hessian_update_freq=10,
            adaptive_hessian=True,
            force_threshold_ratio=2.5,
        )
        assert opt.adaptive_hessian is True
        assert opt.force_threshold_ratio == 2.5

    def test_positions_conversion(self, h2o_perturbed):
        """Test position array conversion methods."""
        h2o = h2o_perturbed.copy()
        h2o.calc = qme.MockCalculator(backend="mock")
        opt = TrustKrylov(h2o, logfile=None)
        x = opt._positions_to_x()
        assert x.shape == (9,)  # 3 atoms × 3 coordinates

        # Round trip conversion
        positions_back = opt._x_to_positions(x)
        assert positions_back.shape == (3, 3)
        assert np.allclose(positions_back, h2o.positions)

    def test_objective_function(self, h2o_perturbed):
        """Test objective function returns energy."""
        h2o = h2o_perturbed.copy()
        h2o.calc = qme.MockCalculator(backend="mock")
        opt = TrustKrylov(h2o, logfile=None)
        x = opt._positions_to_x()
        energy = opt.objective(x)
        assert isinstance(energy, float)
        assert np.isclose(energy, h2o.get_potential_energy())

    def test_gradient_function(self, h2o_perturbed):
        """Test gradient function returns negative forces."""
        h2o = h2o_perturbed.copy()
        h2o.calc = qme.MockCalculator(backend="mock")
        opt = TrustKrylov(h2o, logfile=None)
        x = opt._positions_to_x()
        grad = opt.gradient(x)

        assert grad.shape == (9,)
        forces = h2o.get_forces()
        expected_grad = -forces.ravel()
        assert np.allclose(grad, expected_grad)

    def test_hessian_computation(self, h2o_perturbed):
        """Test Hessian computation and symmetry."""
        h2o = h2o_perturbed.copy()
        h2o.calc = qme.MockCalculator(backend="mock")
        opt = TrustKrylov(h2o, logfile=None)
        x = opt._positions_to_x()

        # First call should compute Hessian
        hessian = opt.hessian_func(x)
        StandardTestAssertions.assert_hessian_valid(hessian, expected_shape=(9, 9))
        assert opt.hessian_calls == 1

    def test_hessian_caching(self, h2o_perturbed):
        """Test that Hessian is cached when update_freq is None."""
        h2o = h2o_perturbed.copy()
        h2o.calc = qme.MockCalculator(backend="mock")
        opt = TrustKrylov(h2o, logfile=None, hessian_update_freq=None)
        x = opt._positions_to_x()

        # First call computes
        opt.hessian_func(x)
        assert opt.hessian_calls == 1

        # Increment step counter
        opt.nsteps = 1
        opt._last_hessian_step = 0
        opt._last_full_hessian_step = 0

        # Second call should use BFGS or cache (no new full Hessian)
        opt.hessian_func(x)
        assert opt.hessian_calls == 1  # Still only 1 full Hessian

    def test_convergence(self, h2o_perturbed):
        """Test basic convergence check."""
        h2o = h2o_perturbed.copy()
        h2o.calc = qme.MockCalculator(backend="mock")
        opt = TrustKrylov(h2o, logfile=None)

        # Small forces should converge
        opt.fmax = 0.05
        assert opt.converged(np.array([0.001, 0.001, 0.001]))

        # Large forces should not converge
        assert not opt.converged(np.array([0.1, 0.1, 0.1]))

    @pytest.mark.parametrize(
        ("update_freq", "expected_total_calls"),
        [(None, 1), (3, 2)],
    )
    def test_hessian_update_frequency(self, update_freq, expected_total_calls, h2o_perturbed):
        """Test Hessian update frequency modes."""
        h2o = h2o_perturbed.copy()
        h2o.calc = qme.MockCalculator(backend="mock")
        opt = TrustKrylov(h2o, logfile=None, hessian_update_freq=update_freq)
        x = opt._positions_to_x()

        # Initial call
        opt.hessian_func(x)
        assert opt.hessian_calls == 1

        # Simulate steps
        num_steps = 5 if update_freq is None else 3
        for step in range(1, num_steps + 1):
            opt.nsteps = step
            if update_freq:
                opt._last_full_hessian_step = 0
            opt.hessian_func(x)

        # Check expected number of full hessian calls
        assert opt.hessian_calls == expected_total_calls

    def test_basic_optimization_run(self, h2o_perturbed):
        """Test that optimizer can run without errors."""
        h2o = h2o_perturbed.copy()
        h2o.calc = qme.MockCalculator(backend="mock")
        opt = TrustKrylov(h2o, logfile=None)

        # Run with relaxed convergence for speed
        converged = opt.run(fmax=0.5, steps=5)

        # Should complete without errors (may or may not converge in 5 steps)
        assert isinstance(converged, bool)
        assert opt.nsteps > 0

    def test_step_counting(self, h2o_perturbed):
        """Test that step counter increments correctly."""
        h2o = h2o_perturbed.copy()
        h2o.calc = qme.MockCalculator(backend="mock")
        opt = TrustKrylov(h2o, logfile=None)
        initial_steps = opt.nsteps

        opt.run(fmax=0.5, steps=3)

        assert opt.nsteps > initial_steps
        assert opt.get_number_of_steps() == opt.nsteps

    @pytest.mark.parametrize("optimizer_class", [TrustKrylov, TrustNCG, TrustExact, NewtonCG])
    def test_optimizer_actually_optimizes(self, optimizer_class, h2o_distorted):
        """Test that optimizers actually optimize energy, positions, and report steps."""
        atoms = h2o_distorted.copy()
        atoms.calc = qme.MockCalculator(backend="mock")

        initial_energy = atoms.get_potential_energy()
        initial_positions = atoms.get_positions().copy()

        # Test optimizer
        opt = optimizer_class(atoms, logfile=None)
        opt.run(fmax=0.01, steps=50)

        final_energy = atoms.get_potential_energy()
        final_positions = atoms.get_positions()

        # Optimizer should actually optimize
        assert abs(final_energy - initial_energy) > 1e-6, (
            f"{optimizer_class.__name__} should change energy"
        )
        assert np.max(np.abs(final_positions - initial_positions)) > 1e-6, (
            f"{optimizer_class.__name__} should change positions"
        )

        # Check step counting
        assert opt.get_number_of_steps() > 0, f"{optimizer_class.__name__} should report steps"

        # Check convergence consistency
        if hasattr(opt, "converged"):
            forces = atoms.get_forces().flatten()
            assert isinstance(opt.converged(forces), (bool, np.bool_))

    @pytest.mark.parametrize("optimizer_class", [TrustKrylov, TrustNCG, TrustExact, NewtonCG])
    def test_optimizer_convergence_quality(self, optimizer_class, h2o_distorted):
        """Test that optimizers achieve reasonable convergence."""
        atoms = h2o_distorted.copy()
        atoms.calc = qme.MockCalculator(backend="mock")

        opt = optimizer_class(atoms, logfile=None)
        opt.run(fmax=0.05, steps=200)

        forces = atoms.get_forces()
        final_energy = atoms.get_potential_energy()

        # Energy should be reasonable
        assert isinstance(final_energy, (float, int))
        assert not np.isnan(final_energy)
        assert not np.isinf(final_energy)

        # If optimizer claims convergence, forces should be low
        if hasattr(opt, "converged"):
            max_force = np.max(np.abs(forces))
            converged = opt.converged(forces.flatten())
            if converged:
                assert max_force < 0.05, (
                    f"{optimizer_class.__name__} claims convergence but max force is {max_force:.6f} eV/Å"
                )


class TestOptimizerErrors:
    """Test error handling and invalid configurations."""

    def test_requires_calculator(self):
        """Test that optimizer raises error without calculator."""
        h2o = TestMoleculeFactory.get_h2o_equilibrium()
        with pytest.raises(ValueError, match="calculator"):
            TrustKrylov(h2o, logfile=None)

    def test_invalid_method(self):
        """Test that invalid method raises error."""
        h2o = TestMoleculeFactory.get_h2o_equilibrium()
        h2o.calc = qme.MockCalculator(backend="mock")

        with pytest.raises(ValueError, match="Invalid method"):
            from qme.optimizers.scipy_optimizers import SciPyHessianOptimizer

            SciPyHessianOptimizer(h2o, method="invalid-method", logfile=None)

    @pytest.mark.parametrize("update_freq", [-1, 0])
    def test_invalid_update_frequency(self, update_freq):
        """Test that invalid update frequencies are handled correctly."""
        h2o = TestMoleculeFactory.get_h2o_equilibrium()
        h2o.calc = qme.MockCalculator(backend="mock")

        # Should be converted to None (disable periodic updates)
        opt = TrustKrylov(h2o, logfile=None, hessian_update_freq=update_freq)
        assert opt.hessian_update_freq is None
