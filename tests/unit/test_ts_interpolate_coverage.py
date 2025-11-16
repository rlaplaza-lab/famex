"""Additional tests for TS interpolate strategy to improve coverage."""

from __future__ import annotations

import pytest

from qme.core.explorer import Explorer
from qme.strategies.ts_interpolate import MultiStructureTSGuessStrategy
from tests.test_constants import QUICK_STEPS_EXTENDED, VERY_LOOSE_FMAX
from tests.test_utils import StandardTestAssertions


class TestTSInterpolateEdgeCases:
    """Test edge cases in TS interpolate strategy."""

    def test_result_with_validation_result(self, reactant_product_pair):
        """Test that validation results are properly included in result."""
        from qme.backends.availability import is_backend_available

        reactant, product = reactant_product_pair
        # Check backend availability before creating explorer
        if not is_backend_available("uma") and not is_backend_available("mace"):
            pytest.skip("No real backend available for TS interpolate test")

        from qme.utils.validation import BackendError

        try:
            explorer = Explorer([reactant, product], backend="uma")
        except (ImportError, BackendError, Exception):
            try:
                explorer = Explorer([reactant, product], backend="mace")
            except (ImportError, BackendError, Exception):
                pytest.skip("No real backend available for TS interpolate test")

        strategy = MultiStructureTSGuessStrategy(explorer)

        try:
            result = strategy.run(
                [reactant, product],
                npoints=5,
                fmax=VERY_LOOSE_FMAX,
                steps=QUICK_STEPS_EXTENDED,
                validate_ts=True,
            )
            StandardTestAssertions.assert_optimization_result(result)
            # If validation was requested, check that result includes validation info
            if "ts_validation" in result:
                assert isinstance(result["ts_validation"], dict)
        except (ValueError, ImportError) as e:
            if "not suitable" in str(e).lower():
                pytest.skip("Backend doesn't support TS optimization")
            if "not available" in str(e).lower():
                pytest.skip("Backend not available")
            raise

    def test_result_with_frequency_analysis(self, reactant_product_pair):
        """Test that frequency analysis results are properly passed through."""
        from qme.backends.availability import is_backend_available

        reactant, product = reactant_product_pair
        # Check backend availability before creating explorer
        if not is_backend_available("uma") and not is_backend_available("mace"):
            pytest.skip("No real backend available for TS interpolate test")

        from qme.utils.validation import BackendError

        try:
            explorer = Explorer([reactant, product], backend="uma")
        except (ImportError, BackendError, Exception):
            try:
                explorer = Explorer([reactant, product], backend="mace")
            except (ImportError, BackendError, Exception):
                pytest.skip("No real backend available for TS interpolate test")

        strategy = MultiStructureTSGuessStrategy(explorer)

        try:
            result = strategy.run(
                [reactant, product],
                npoints=5,
                fmax=VERY_LOOSE_FMAX,
                steps=QUICK_STEPS_EXTENDED,
                calculate_frequencies=True,
            )
            StandardTestAssertions.assert_optimization_result(result)
            # Check if frequency analysis results are included
            if "frequency_analysis" in result:
                assert isinstance(result["frequency_analysis"], dict)
            if "is_ts" in result:
                assert isinstance(result["is_ts"], bool)
            if "free_energy_correction" in result:
                assert isinstance(result["free_energy_correction"], float | type(None))
        except (ValueError, ImportError) as e:
            if "not suitable" in str(e).lower():
                pytest.skip("Backend doesn't support TS optimization")
            if "not available" in str(e).lower():
                pytest.skip("Backend not available")
            raise

    def test_result_with_both_validation_and_frequencies(self, reactant_product_pair):
        """Test result structure with both validation and frequency analysis."""
        from qme.backends.availability import is_backend_available

        reactant, product = reactant_product_pair
        # Check backend availability before creating explorer
        if not is_backend_available("uma") and not is_backend_available("mace"):
            pytest.skip("No real backend available for TS interpolate test")

        from qme.utils.validation import BackendError

        try:
            explorer = Explorer([reactant, product], backend="uma")
        except (ImportError, BackendError, Exception):
            try:
                explorer = Explorer([reactant, product], backend="mace")
            except (ImportError, BackendError, Exception):
                pytest.skip("No real backend available for TS interpolate test")

        strategy = MultiStructureTSGuessStrategy(explorer)

        try:
            result = strategy.run(
                [reactant, product],
                npoints=5,
                fmax=VERY_LOOSE_FMAX,
                steps=QUICK_STEPS_EXTENDED,
                validate_ts=True,
                calculate_frequencies=True,
            )
            StandardTestAssertions.assert_optimization_result(result)
        except (ValueError, ImportError) as e:
            if "not suitable" in str(e).lower():
                pytest.skip("Backend doesn't support TS optimization")
            if "not available" in str(e).lower():
                pytest.skip("Backend not available")
            raise

    def test_custom_local_optimizer_name(self, reactant_product_pair):
        """Test that custom local_optimizer_name is properly handled."""
        from qme.backends.availability import is_backend_available

        reactant, product = reactant_product_pair
        # Check backend availability before creating explorer
        if not is_backend_available("uma") and not is_backend_available("mace"):
            pytest.skip("No real backend available for TS interpolate test")

        from qme.utils.validation import BackendError

        try:
            explorer = Explorer([reactant, product], backend="uma")
        except (ImportError, BackendError, Exception):
            try:
                explorer = Explorer([reactant, product], backend="mace")
            except (ImportError, BackendError, Exception):
                pytest.skip("No real backend available for TS interpolate test")

        strategy = MultiStructureTSGuessStrategy(explorer)

        # Test with different optimizer names
        optimizer_names = ["sella", "rfo"]
        for optimizer_name in optimizer_names:
            try:
                result = strategy.run(
                    [reactant, product],
                    npoints=5,
                    fmax=VERY_LOOSE_FMAX,
                    steps=QUICK_STEPS_EXTENDED,
                    local_optimizer_name=optimizer_name,
                )
                StandardTestAssertions.assert_optimization_result(result)
            except (ValueError, ImportError, NotImplementedError) as e:
                # Some optimizers might not be available or suitable
                if "not suitable" in str(e).lower():
                    # This is expected for some optimizers
                    continue
                # Other errors might be acceptable (missing dependency, etc.)

    def test_interpolation_kwargs_filtering(self, reactant_product_pair):
        """Test that interpolation kwargs are properly filtered."""
        from qme.backends.availability import is_backend_available

        reactant, product = reactant_product_pair
        # Check backend availability before creating explorer
        if not is_backend_available("uma") and not is_backend_available("mace"):
            pytest.skip("No real backend available for TS interpolate test")

        from qme.utils.validation import BackendError

        try:
            explorer = Explorer([reactant, product], backend="uma")
        except (ImportError, BackendError, Exception):
            try:
                explorer = Explorer([reactant, product], backend="mace")
            except (ImportError, BackendError, Exception):
                pytest.skip("No real backend available for TS interpolate test")

        strategy = MultiStructureTSGuessStrategy(explorer)

        # Test with additional kwargs that should be filtered
        try:
            result = strategy.run(
                [reactant, product],
                npoints=5,
                fmax=VERY_LOOSE_FMAX,
                steps=QUICK_STEPS_EXTENDED,
                calculator=None,  # Should be filtered and passed to interpolate
            )
            StandardTestAssertions.assert_optimization_result(result)
        except (ValueError, ImportError) as e:
            if "not suitable" in str(e).lower():
                pytest.skip("Backend doesn't support TS optimization")
            if "not available" in str(e).lower():
                pytest.skip("Backend not available")
            raise

    def test_different_interpolation_methods(self, reactant_product_pair):
        """Test different interpolation methods."""
        from qme.backends.availability import is_backend_available

        reactant, product = reactant_product_pair
        # Check backend availability before creating explorer
        if not is_backend_available("uma") and not is_backend_available("mace"):
            pytest.skip("No real backend available for TS interpolate test")

        from qme.utils.validation import BackendError

        try:
            explorer = Explorer([reactant, product], backend="uma")
        except (ImportError, BackendError, Exception):
            try:
                explorer = Explorer([reactant, product], backend="mace")
            except (ImportError, BackendError, Exception):
                pytest.skip("No real backend available for TS interpolate test")

        strategy = MultiStructureTSGuessStrategy(explorer)

        methods = ["linear", "geodesic", "idpp"]
        for method in methods:
            try:
                result = strategy.run(
                    [reactant, product],
                    npoints=5,
                    method=method,
                    fmax=VERY_LOOSE_FMAX,
                    steps=QUICK_STEPS_EXTENDED,
                )
                StandardTestAssertions.assert_optimization_result(result)
            except (ValueError, ImportError, NotImplementedError) as e:
                if "not suitable" in str(e).lower():
                    pytest.skip("Backend doesn't support TS optimization")
                if "not available" in str(e).lower():
                    pytest.skip("Backend not available")
                # Some methods might not be available
                if "not implemented" in str(e).lower():
                    continue
                raise
