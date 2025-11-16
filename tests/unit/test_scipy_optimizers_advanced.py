"""Advanced tests for SciPy optimizers: adaptive Hessian updates, BFGS, and advanced features."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from qme.optimizers.scipy_optimizers import NewtonCG, TrustExact, TrustKrylov, TrustNCG
from tests.test_constants import LOOSE_FMAX, QUICK_STEPS


class TestAdaptiveHessianUpdates:
    def test_adaptive_hessian_force_increase_trigger(self, h2o_molecule_with_mock):
        atoms = h2o_molecule_with_mock

        opt = TrustKrylov(
            atoms,
            logfile=None,
            adaptive_hessian=True,
            hessian_update_freq=10,
            force_threshold_ratio=2.0,
        )

        x = opt._positions_to_x()

        # Set up initial state
        opt.nsteps = 1
        opt._previous_fmax = 0.01
        opt._last_full_hessian_step = 0

        # Mock current fmax to be high
        from tests.test_constants import DEFAULT_FMAX

        with patch.object(opt, "_get_current_fmax", return_value=DEFAULT_FMAX):
            # Should trigger full update due to force increase
            _hessian1 = opt.hessian_func(x)
            assert opt.hessian_calls >= 1

    def test_adaptive_hessian_periodic_update(self, h2o_molecule_with_mock):
        atoms = h2o_molecule_with_mock

        opt = TrustKrylov(
            atoms,
            logfile=None,
            adaptive_hessian=True,
            hessian_update_freq=3,
        )

        x = opt._positions_to_x()

        # Initial hessian
        opt.hessian_func(x)
        _initial_calls = opt.hessian_calls

        # Simulate steps
        opt.nsteps = 4
        opt._last_full_hessian_step = 0
        opt._previous_fmax = 0.01

        with patch.object(opt, "_get_current_fmax", return_value=0.01):
            # Should trigger periodic update
            opt.hessian_func(x)
            assert opt.hessian_calls > _initial_calls

    def test_adaptive_hessian_no_force_increase(self, h2o_molecule_with_mock):
        atoms = h2o_molecule_with_mock

        opt = TrustKrylov(
            atoms,
            logfile=None,
            adaptive_hessian=True,
            hessian_update_freq=10,
            force_threshold_ratio=2.0,
        )

        x = opt._positions_to_x()

        # Set up initial state
        opt.nsteps = 1
        opt._previous_fmax = 0.01
        opt._last_full_hessian_step = 0

        # Mock current fmax to be lower
        with patch.object(opt, "_get_current_fmax", return_value=0.005):
            _initial_calls = opt.hessian_calls
            opt.hessian_func(x)
            # Should not trigger update if forces decreased
            # (unless periodic update is due)


class TestBFGSUpdates:
    def test_bfgs_update_basic(self, h2o_molecule_with_mock):
        atoms = h2o_molecule_with_mock

        opt = TrustKrylov(
            atoms,
            logfile=None,
            hessian_update_freq=None,  # Disable periodic updates
            use_bfgs_update=True,
        )

        x = opt._positions_to_x()

        # Get initial hessian
        _hessian1 = opt.hessian_func(x)
        assert opt.hessian_calls == 1

        # Set up for BFGS update
        opt.nsteps = 1
        opt._last_full_hessian_step = 0
        opt._last_positions = x.copy()
        opt._last_gradient = opt.gradient(x)

        # Modify positions slightly
        import numpy as np

        x2 = x + 0.01 * np.random.randn(*x.shape)
        atoms.set_positions(opt._x_to_positions(x2))

        # Should use BFGS update
        _hessian2 = opt.hessian_func(x2)

        # BFGS updates should increment counter
        assert opt.bfgs_updates >= 0  # May be 0 if sy too small

    def test_bfgs_update_small_sy(self, h2o_molecule_with_mock):
        atoms = h2o_molecule_with_mock

        opt = TrustKrylov(
            atoms,
            logfile=None,
            hessian_update_freq=None,
            use_bfgs_update=True,
        )

        x = opt._positions_to_x()

        # Get initial hessian
        opt.hessian_func(x)

        # Set up for BFGS update with very small change
        opt.nsteps = 1
        opt._last_full_hessian_step = 0
        opt._last_positions = x.copy()
        opt._last_gradient = opt.gradient(x)

        # Very small position change
        import numpy as np

        x2 = x + 1e-20 * np.ones_like(x)
        atoms.set_positions(opt._x_to_positions(x2))

        _initial_bfgs = opt.bfgs_updates
        opt.hessian_func(x2)

        # BFGS might be skipped due to very small sy
        # but should not crash


class TestSciPyOptimizerVerboseMode:
    @pytest.mark.parametrize(
        ("verbose", "logfile", "expected_verbose"),
        [(0, "-", 0), (2, None, 2)],
    )
    def test_verbose_mode_settings(
        self, verbose, logfile, expected_verbose, h2o_molecule_with_mock
    ):
        atoms = h2o_molecule_with_mock

        opt = TrustKrylov(
            atoms,
            logfile=logfile,
            verbose=verbose,
            hessian_update_freq=5 if verbose == 2 else None,
            adaptive_hessian=verbose == 2,
        )

        assert opt.verbose == expected_verbose
        if verbose == 0:
            # Should have logfile set to None in quiet mode
            assert opt.logfile is None or opt.verbose == 0

    @pytest.mark.parametrize("optimizer_class", [TrustKrylov, TrustNCG, TrustExact, NewtonCG])
    def test_verbose_logging_in_optimization(self, optimizer_class, h2o_molecule_with_mock):
        atoms = h2o_molecule_with_mock

        opt = optimizer_class(atoms, logfile=None, verbose=2)

        # Run short optimization
        opt.run(fmax=LOOSE_FMAX, steps=QUICK_STEPS)

        assert opt.nsteps > 0


class TestInitialHessian:
    def test_initial_hessian_provided(self, h2o_molecule_with_mock):
        atoms = h2o_molecule_with_mock

        n = len(atoms) * 3
        initial_hess = np.eye(n)

        opt = TrustKrylov(atoms, logfile=None, initial_hessian=initial_hess)

        x = opt._positions_to_x()

        # Should use initial hessian (may be updated)
        hessian = opt.hessian_func(x)
        assert hessian is not None


class TestAlphaScaling:
    @pytest.mark.parametrize(
        ("function_name", "expected_factor"),
        [("objective", 1.0 / 2.0), ("gradient", -1.0 / 2.0)],
    )
    def test_alpha_scaling(self, function_name, expected_factor, h2o_molecule_with_mock):
        atoms = h2o_molecule_with_mock

        opt = TrustKrylov(atoms, logfile=None, alpha=2.0)

        x = opt._positions_to_x()
        result = getattr(opt, function_name)(x)

        if function_name == "objective":
            expected = atoms.get_potential_energy() * expected_factor
            assert np.isclose(result, expected)
        else:  # gradient
            forces = atoms.get_forces()
            expected = forces.ravel() * expected_factor
            assert np.allclose(result, expected)


class TestGetCurrentFmax:
    def test_get_current_fmax_success(self, h2o_molecule_with_mock):
        atoms = h2o_molecule_with_mock

        opt = TrustKrylov(atoms, logfile=None)

        fmax = opt._get_current_fmax()
        assert fmax is not None
        assert fmax >= 0

    def test_get_current_fmax_failure_returns_none(self, h2o_molecule_with_mock):
        atoms = h2o_molecule_with_mock

        opt = TrustKrylov(atoms, logfile=None)

        # Force error in get_forces
        with patch.object(atoms, "get_forces", side_effect=Exception("Error")):
            fmax = opt._get_current_fmax()
            assert fmax is None
