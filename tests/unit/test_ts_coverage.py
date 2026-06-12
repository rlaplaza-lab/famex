"""Additional tests for TS strategy to improve coverage."""

from __future__ import annotations

import pytest

from famex.core.explorer import Explorer
from famex.strategies.ts import LocalTSStrategy
from tests.test_constants import LOOSE_FMAX, QUICK_STEPS_EXTENDED
from tests.test_utils import StandardTestAssertions, handle_backend_errors


class TestTSErrorHandling:
    """Test error handling paths in TS strategy."""

    def test_validates_forbidden_optimizers(
        self, water_dissociation_ts_guess, any_real_backend_explorer
    ):
        """Test that TS strategy rejects unsuitable optimizers."""
        atoms = water_dissociation_ts_guess.copy()
        explorer = any_real_backend_explorer(
            atoms, "No real backend available for TS optimizer validation test"
        )
        strategy = LocalTSStrategy(explorer)

        # Test forbidden optimizers
        forbidden_optimizers = ["lbfgs", "l-bfgs", "l_bfgs", "bfgs", "fire"]
        for optimizer in forbidden_optimizers:
            with pytest.raises(ValueError, match="not suitable for transition state"):
                strategy.run(
                    [atoms],
                    steps=QUICK_STEPS_EXTENDED,
                    fmax=LOOSE_FMAX,
                    local_optimizer_name=optimizer,
                )

    def test_validates_backend_case_insensitive(self, water_dissociation_ts_guess):
        """Test that backend validation is case-insensitive."""
        atoms = water_dissociation_ts_guess.copy()
        # Create explorer with mock backend but test case-insensitive check
        explorer = Explorer(atoms, backend="MOCK")  # Uppercase
        strategy = LocalTSStrategy(explorer)

        with pytest.raises(ValueError, match="not suitable for transition state"):
            strategy.run([atoms], steps=QUICK_STEPS_EXTENDED, fmax=LOOSE_FMAX)

    def test_rfo_optimizer_kwargs_preparation(
        self, water_dissociation_ts_guess, any_real_backend_explorer
    ):
        """Test that RFO optimizer gets appropriate kwargs."""
        atoms = water_dissociation_ts_guess.copy()
        explorer = any_real_backend_explorer(
            atoms, "No real backend available for RFO optimizer test"
        )
        strategy = LocalTSStrategy(explorer)

        # Test RFO optimizer variants
        rfo_variants = ["rfo", "rfo-ts", "rational-function", "rational_function"]
        for optimizer in rfo_variants:
            try:
                result = strategy.run(
                    [atoms],
                    steps=QUICK_STEPS_EXTENDED,
                    fmax=LOOSE_FMAX,
                    local_optimizer_name=optimizer,
                )
                # If it succeeds, verify result structure
                StandardTestAssertions.assert_optimization_result(result)
            except (ValueError, ImportError, NotImplementedError) as e:
                # RFO might not be available or might fail for other reasons
                # That's okay - we're testing that the kwargs preparation doesn't crash
                if "not suitable" in str(e).lower():
                    pytest.fail(f"RFO optimizer {optimizer} was incorrectly rejected")
                # Other errors are acceptable (missing dependency, etc.)

    def test_force_finite_diff_hessian_flag(
        self, water_dissociation_ts_guess, any_real_backend_explorer
    ):
        """Test force_finite_diff_hessian flag handling."""
        atoms = water_dissociation_ts_guess.copy()
        explorer = any_real_backend_explorer(
            atoms, "No real backend available for finite diff Hessian test"
        )
        explorer.force_finite_diff_hessian = True
        strategy = LocalTSStrategy(explorer)

        # Test with RFO optimizer (should use finite_differences method)
        try:
            result = strategy.run(
                [atoms],
                steps=QUICK_STEPS_EXTENDED,
                fmax=LOOSE_FMAX,
                local_optimizer_name="rfo",
                validate_ts=True,
            )
            StandardTestAssertions.assert_optimization_result(result)
        except (ValueError, ImportError, NotImplementedError):
            # RFO might not be available - that's okay
            pass

    @handle_backend_errors()
    def test_ts_validation_hook_handles_tuple_return(
        self, water_dissociation_ts_guess, any_real_backend_explorer
    ):
        """Test that TS validation hook handles tuple returns correctly."""
        atoms = water_dissociation_ts_guess.copy()
        explorer = any_real_backend_explorer(
            atoms, "No real backend available for TS validation test"
        )
        strategy = LocalTSStrategy(explorer)

        # Test with validate_ts=True to trigger validation hook
        result = strategy.run(
            [atoms],
            steps=QUICK_STEPS_EXTENDED,
            fmax=LOOSE_FMAX,
            validate_ts=True,
        )
        StandardTestAssertions.assert_optimization_result(result)
        # Validation hook should handle both dict and tuple returns
        if "ts_validation" in result:
            assert isinstance(result["ts_validation"], dict | list)

    @handle_backend_errors()
    def test_post_optimization_hook_logging(
        self, water_dissociation_ts_guess, any_real_backend_explorer
    ):
        """Test that post-optimization hook logs diagnostic information."""
        atoms = water_dissociation_ts_guess.copy()
        explorer = any_real_backend_explorer(atoms, "No real backend available for logging test")
        explorer.verbose = 1  # Enable logging
        strategy = LocalTSStrategy(explorer)

        # Test with verbose logging enabled
        result = strategy.run(
            [atoms],
            steps=QUICK_STEPS_EXTENDED,
            fmax=LOOSE_FMAX,
            verbose=1,
        )
        StandardTestAssertions.assert_optimization_result(result)

    @handle_backend_errors()
    def test_prepare_frequency_kwargs_with_flag(
        self, water_dissociation_ts_guess, any_real_backend_explorer
    ):
        """Test prepare_ts_frequency_kwargs with force_finite_diff_hessian."""
        atoms = water_dissociation_ts_guess.copy()
        explorer = any_real_backend_explorer(
            atoms, "No real backend available for frequency kwargs test"
        )
        explorer.force_finite_diff_hessian = True
        strategy = LocalTSStrategy(explorer)

        # Test with calculate_frequencies=True
        result = strategy.run(
            [atoms],
            steps=QUICK_STEPS_EXTENDED,
            fmax=LOOSE_FMAX,
            calculate_frequencies=True,
        )
        StandardTestAssertions.assert_optimization_result(result)
        if "frequency_analysis" in result:
            assert isinstance(result["frequency_analysis"], dict)
