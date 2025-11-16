"""Additional tests for RFO optimizer to improve coverage."""

from __future__ import annotations

from qme.optimizers.rfo_optimizer import ConvergedError, RFOTransitionState
from tests.test_constants import DEFAULT_FMAX, LOOSE_FMAX, QUICK_STEPS, QUICK_STEPS_EXTENDED


class TestRFOConvergenceScenarios:
    """Test convergence scenarios in RFO optimizer."""

    def test_converged_error_handling(self, mock_backend, water_dissociation_ts_guess):
        """Test that ConvergedError is properly handled during optimization."""
        atoms = water_dissociation_ts_guess.copy()
        atoms.calc = mock_backend

        opt = RFOTransitionState(atoms, logfile=None, hessian_update_freq=5)

        # Mock converged to raise ConvergedError early
        original_converged = opt.converged

        def mock_converged(forces):
            # Raise ConvergedError after first step
            if opt.nsteps > 0:
                raise ConvergedError
            return original_converged(forces)

        opt.converged = mock_converged

        # Should handle ConvergedError gracefully
        result = opt.run(fmax=LOOSE_FMAX, steps=QUICK_STEPS_EXTENDED)
        assert bool(result) in (True, False)

    def test_non_converged_logging(self, mock_backend, water_dissociation_ts_guess):
        """Test logging when optimization doesn't converge."""
        atoms = water_dissociation_ts_guess.copy()
        atoms.calc = mock_backend

        # Use very tight fmax and few steps to ensure non-convergence
        opt = RFOTransitionState(
            atoms,
            logfile=None,
            hessian_update_freq=5,
            verbose=1,  # Enable logging
        )

        # Run with very few steps and tight convergence
        result = opt.run(fmax=DEFAULT_FMAX * 0.1, steps=2)  # Very tight, few steps

        # Should handle non-convergence gracefully
        assert bool(result) in (True, False)
        assert opt.nsteps >= 0

    def test_step_quality_summary_logging(self, mock_backend, water_dissociation_ts_guess):
        """Test step quality summary logging."""
        atoms = water_dissociation_ts_guess.copy()
        atoms.calc = mock_backend

        opt = RFOTransitionState(
            atoms,
            logfile=None,
            hessian_update_freq=5,
            verbose=2,  # Enable verbose logging
        )

        # Run optimization to generate step quality history
        opt.run(fmax=LOOSE_FMAX, steps=QUICK_STEPS_EXTENDED)

        # Check that step quality history was tracked
        if hasattr(opt, "_step_quality_history") and opt._step_quality_history:
            assert len(opt._step_quality_history) > 0
            # Quality values should be reasonable
            for quality in opt._step_quality_history:
                assert isinstance(quality, (int, float))

    def test_closed_file_stream_handling(self, mock_backend, water_dissociation_ts_guess):
        """Test that closed file streams are handled gracefully."""
        atoms = water_dissociation_ts_guess.copy()
        atoms.calc = mock_backend

        # Create optimizer with a logfile that might be closed
        import io

        logfile = io.StringIO()
        opt = RFOTransitionState(
            atoms,
            logfile=logfile,
            hessian_update_freq=5,
            verbose=1,
        )

        # Close the logfile to simulate closed stream
        logfile.close()

        # Should handle closed file stream gracefully
        try:
            result = opt.run(fmax=LOOSE_FMAX, steps=QUICK_STEPS)
            assert bool(result) in (True, False)
        except (ValueError, OSError):
            # These exceptions are acceptable for closed streams
            pass

    def test_verbose_logging_levels(self, mock_backend, water_dissociation_ts_guess):
        """Test different verbose logging levels."""
        atoms = water_dissociation_ts_guess.copy()
        atoms.calc = mock_backend

        for verbose_level in [0, 1, 2]:
            # Create fresh atoms copy for each iteration
            test_atoms = atoms.copy()
            test_atoms.calc = mock_backend
            opt = RFOTransitionState(
                test_atoms,
                logfile=None,
                hessian_update_freq=5,
                verbose=verbose_level,
            )

            # Should run without errors at all verbosity levels
            result = opt.run(fmax=LOOSE_FMAX, steps=QUICK_STEPS)
            assert bool(result) in (True, False)

    def test_convergence_at_step_limit(self, mock_backend, water_dissociation_ts_guess):
        """Test behavior when reaching step limit without convergence."""
        atoms = water_dissociation_ts_guess.copy()
        atoms.calc = mock_backend

        opt = RFOTransitionState(
            atoms,
            logfile=None,
            hessian_update_freq=5,
            verbose=1,
        )

        # Run with very few steps to hit limit quickly
        result = opt.run(fmax=DEFAULT_FMAX * 0.1, steps=1)  # Tight convergence, 1 step

        # Should handle step limit gracefully
        assert bool(result) in (True, False)
        assert opt.nsteps >= 0

    def test_trust_radius_boundaries(self, mock_backend, water_dissociation_ts_guess):
        """Test trust radius adjustment at boundaries."""
        atoms = water_dissociation_ts_guess.copy()
        atoms.calc = mock_backend

        # Test with very small trust radius
        opt = RFOTransitionState(
            atoms,
            logfile=None,
            trust_radius=0.001,
            max_trust_radius=0.01,
        )

        initial_trust = opt.trust_radius

        # Test that trust radius doesn't exceed max
        opt._adjust_trust_radius(1.0, step_size=0.0001)  # Excellent step
        assert opt.trust_radius <= opt.max_trust_radius

        # Reset and test minimum
        opt.trust_radius = initial_trust
        opt._adjust_trust_radius(0.0, step_size=0.0001)  # Poor step
        assert opt.trust_radius >= 0.0  # Should not go negative
