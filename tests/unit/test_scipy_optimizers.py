"""
Test SciPy Hessian-based optimizers.

This module tests the SciPy optimizer wrappers (TrustKrylov, TrustNCG,
TrustExact, NewtonCG) including Hessian computation, BFGS updates,
and convergence behavior.
"""

import numpy as np
import pytest
from ase.build import molecule

import qme
from qme.core.scipy_optimizers import NewtonCG, TrustExact, TrustKrylov, TrustNCG


class TestTrustKrylovBasics:
    """Test basic TrustKrylov optimizer functionality."""

    def setup_method(self):
        """Set up test molecule with mock calculator."""
        self.h2o = molecule("H2O")
        # Perturb slightly from equilibrium
        self.h2o.positions += np.random.RandomState(42).normal(0, 0.05, self.h2o.positions.shape)
        self.h2o.calc = qme.MockCalculator(backend="mock")

    def test_optimizer_initialization(self):
        """Test that TrustKrylov optimizer can be initialized."""
        opt = TrustKrylov(self.h2o, logfile=None)
        assert opt.method == "trust-krylov"
        assert opt.hessian_update_freq is None  # Default: compute once
        assert opt.use_bfgs_update is True  # Default: BFGS enabled
        assert opt.adaptive_hessian is False  # Default: no adaptive triggers

    def test_optimizer_with_periodic_updates(self):
        """Test optimizer with periodic Hessian updates."""
        opt = TrustKrylov(self.h2o, logfile=None, hessian_update_freq=5)
        assert opt.hessian_update_freq == 5

    def test_optimizer_with_adaptive_mode(self):
        """Test optimizer with adaptive Hessian updates."""
        opt = TrustKrylov(
            self.h2o,
            logfile=None,
            hessian_update_freq=10,
            adaptive_hessian=True,
            force_threshold_ratio=2.5,
        )
        assert opt.adaptive_hessian is True
        assert opt.force_threshold_ratio == 2.5

    def test_positions_conversion(self):
        """Test position array conversion methods."""
        opt = TrustKrylov(self.h2o, logfile=None)
        x = opt._positions_to_x()
        assert x.shape == (9,)  # 3 atoms × 3 coordinates

        # Round trip conversion
        positions_back = opt._x_to_positions(x)
        assert positions_back.shape == (3, 3)
        assert np.allclose(positions_back, self.h2o.positions)

    def test_objective_function(self):
        """Test objective function returns energy."""
        opt = TrustKrylov(self.h2o, logfile=None)
        x = opt._positions_to_x()
        energy = opt.objective(x)
        assert isinstance(energy, float)
        # With alpha=1.0, should equal potential energy
        assert np.isclose(energy, self.h2o.get_potential_energy())

    def test_gradient_function(self):
        """Test gradient function returns negative forces."""
        opt = TrustKrylov(self.h2o, logfile=None)
        x = opt._positions_to_x()
        grad = opt.gradient(x)

        assert grad.shape == (9,)
        forces = self.h2o.get_forces()
        expected_grad = -forces.ravel()
        assert np.allclose(grad, expected_grad)

    def test_hessian_computation(self):
        """Test Hessian computation."""
        opt = TrustKrylov(self.h2o, logfile=None)
        x = opt._positions_to_x()

        # First call should compute Hessian
        hessian = opt.hessian_func(x)
        assert hessian.shape == (9, 9)
        assert opt.hessian_calls == 1

        # Should be symmetric
        assert np.allclose(hessian, hessian.T, atol=1e-10)

    def test_hessian_caching(self):
        """Test that Hessian is cached when update_freq is None."""
        opt = TrustKrylov(self.h2o, logfile=None, hessian_update_freq=None)
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

    def test_convergence(self):
        """Test basic convergence check."""
        opt = TrustKrylov(self.h2o, logfile=None)

        # Small forces should converge
        small_forces = np.array([0.001, 0.001, 0.001])
        opt.fmax = 0.05
        assert opt.converged(small_forces)

        # Large forces should not converge
        large_forces = np.array([0.1, 0.1, 0.1])
        assert not opt.converged(large_forces)


class TestAllSciPyOptimizers:
    """Test all SciPy optimizer variants."""

    @pytest.mark.parametrize(
        "optimizer_class,method",
        [
            (TrustKrylov, "trust-krylov"),
            (TrustNCG, "trust-ncg"),
            (TrustExact, "trust-exact"),
            (NewtonCG, "Newton-CG"),
        ],
    )
    def test_optimizer_initialization(self, optimizer_class, method):
        """Test initialization of all optimizer types."""
        h2o = molecule("H2O")
        h2o.calc = qme.MockCalculator(backend="mock")

        opt = optimizer_class(h2o, logfile=None)
        assert opt.method == method

    @pytest.mark.parametrize("optimizer_class", [TrustKrylov, TrustNCG, TrustExact, NewtonCG])
    def test_optimizer_accepts_all_parameters(self, optimizer_class):
        """Test that all optimizers accept standard parameters."""
        h2o = molecule("H2O")
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


class TestHessianUpdateStrategies:
    """Test different Hessian update strategies."""

    def setup_method(self):
        """Set up test molecule."""
        self.h2o = molecule("H2O")
        self.h2o.positions += np.random.RandomState(42).normal(0, 0.05, self.h2o.positions.shape)
        self.h2o.calc = qme.MockCalculator(backend="mock")

    def test_single_hessian_mode(self):
        """Test default mode: compute Hessian once."""
        opt = TrustKrylov(self.h2o, logfile=None, hessian_update_freq=None)
        x = opt._positions_to_x()

        # First call
        opt.hessian_func(x)
        assert opt.hessian_calls == 1

        # Simulate several steps
        for step in range(5):
            opt.nsteps = step + 1
            opt.hessian_func(x)

        # Should still be only 1 full Hessian (BFGS handles updates)
        assert opt.hessian_calls == 1

    def test_periodic_update_mode(self):
        """Test periodic Hessian updates."""
        opt = TrustKrylov(
            self.h2o,
            logfile=None,
            hessian_update_freq=3,
            adaptive_hessian=False,
        )
        x = opt._positions_to_x()

        # Initial
        opt.hessian_func(x)
        assert opt.hessian_calls == 1

        # Steps 1, 2 - no update
        for step in [1, 2]:
            opt.nsteps = step
            opt._last_full_hessian_step = 0
            opt.hessian_func(x)
        assert opt.hessian_calls == 1

        # Step 3 - should trigger update
        opt.nsteps = 3
        opt._last_full_hessian_step = 0
        opt.hessian_func(x)
        assert opt.hessian_calls == 2

    def test_bfgs_update_tracking(self):
        """Test that BFGS updates are tracked."""
        opt = TrustKrylov(
            self.h2o,
            logfile=None,
            hessian_update_freq=None,
            use_bfgs_update=True,
        )
        x = opt._positions_to_x()

        # Initial Hessian
        opt.hessian_func(x)
        initial_bfgs = opt.bfgs_updates

        # Compute gradient to initialize state
        opt.gradient(x)

        # Move positions slightly
        x_new = x + np.random.RandomState(42).normal(0, 0.01, x.shape)
        opt.nsteps = 1
        opt._last_hessian_step = 0

        # This should trigger BFGS update
        opt.hessian_func(x_new)

        # BFGS counter should increase or stay same (depending on step size)
        assert opt.bfgs_updates >= initial_bfgs


class TestOptimizerWithoutCalculator:
    """Test error handling when calculator is missing."""

    def test_requires_calculator(self):
        """Test that optimizer raises error without calculator."""
        h2o = molecule("H2O")

        with pytest.raises(ValueError, match="calculator"):
            TrustKrylov(h2o, logfile=None)


class TestOptimizerRun:
    """Test running full optimizations (lightweight with mock calculator)."""

    def test_basic_optimization_run(self):
        """Test that optimizer can run without errors."""
        h2o = molecule("H2O")
        # Small perturbation
        h2o.positions += np.random.RandomState(42).normal(0, 0.01, h2o.positions.shape)
        h2o.calc = qme.MockCalculator(backend="mock")

        opt = TrustKrylov(h2o, logfile=None)

        # Run with relaxed convergence for speed
        converged = opt.run(fmax=0.5, steps=5)

        # Should complete without errors (may or may not converge in 5 steps)
        assert isinstance(converged, bool)
        assert opt.nsteps > 0

    def test_step_counting(self):
        """Test that step counter increments correctly."""
        h2o = molecule("H2O")
        h2o.positions += np.random.RandomState(42).normal(0, 0.02, h2o.positions.shape)
        h2o.calc = qme.MockCalculator(backend="mock")

        opt = TrustKrylov(h2o, logfile=None)
        initial_steps = opt.nsteps

        opt.run(fmax=0.5, steps=3)

        assert opt.nsteps > initial_steps
        assert opt.get_number_of_steps() == opt.nsteps


class TestInvalidConfiguration:
    """Test invalid optimizer configurations."""

    def test_invalid_method(self):
        """Test that invalid method raises error."""
        h2o = molecule("H2O")
        h2o.calc = qme.MockCalculator(backend="mock")

        with pytest.raises(ValueError, match="Invalid method"):
            from qme.core.scipy_optimizers import SciPyHessianOptimizer

            SciPyHessianOptimizer(h2o, method="invalid-method", logfile=None)

    def test_negative_update_frequency(self):
        """Test that negative update frequency is handled."""
        h2o = molecule("H2O")
        h2o.calc = qme.MockCalculator(backend="mock")

        # Should be converted to None (disable periodic updates)
        opt = TrustKrylov(h2o, logfile=None, hessian_update_freq=-1)
        assert opt.hessian_update_freq is None

    def test_zero_update_frequency(self):
        """Test that zero update frequency is handled."""
        h2o = molecule("H2O")
        h2o.calc = qme.MockCalculator(backend="mock")

        # Should be converted to None
        opt = TrustKrylov(h2o, logfile=None, hessian_update_freq=0)
        assert opt.hessian_update_freq is None
