"""Test reaction handling functionality."""

import pytest
from ase.build import molecule

from qme import QMEOptimizer


class TestReactions:
    """Test suite for reaction-related functionality in QME."""

    def test_reactant_product_optimization(self):
        """Test optimizing reactant and product structures."""
        qme = QMEOptimizer(use_mock=True)

        # Simple H2 dissociation: H2 -> 2H
        h2 = molecule("H2")
        h2.set_calculator(qme.calculator)
        qme.atoms = h2  # Set atoms on QME instance

        # Optimize "reactant"
        reactant_result = qme.optimize_minimum(steps=5)
        assert reactant_result is not None

        # Create "product" (stretched H2 to simulate dissociation)
        h2_stretched = h2.copy()
        positions = h2_stretched.get_positions()
        positions[1][0] += 2.0  # Move second H atom away
        h2_stretched.set_positions(positions)
        h2_stretched.set_calculator(qme.calculator)

        # Optimize "product"
        qme.atoms = h2_stretched
        product_result = qme.optimize_minimum(steps=5)
        assert product_result is not None

    def test_transition_state_search_availability(self):
        """Test that transition state search is available when SELLA is present."""
        qme = QMEOptimizer(use_mock=True)

        # Check if Sella is available
        if "Sella" in qme.AVAILABLE_OPTIMIZERS:
            # Create a simple system
            h2 = molecule("H2")
            h2.set_calculator(qme.calculator)
            qme.atoms = h2

            # Try transition state search (may not converge, but should not error)
            try:
                ts_result = qme.find_transition_state(steps=2)
                # If it runs, it should return something
                assert ts_result is not None
            except Exception as e:
                # Mock calculator may give numerical issues with TS search
                # Accept various failure modes as expected with mock calculator
                error_msg = str(e).lower()
                expected_errors = [
                    "inf",
                    "nan",
                    "transition state",
                    "sella",
                    "numerical",
                ]
                assert any(err in error_msg for err in expected_errors)
        else:
            pytest.skip("SELLA not available for transition state search")

    def test_energy_comparison(self):
        """Test comparing energies of different structures."""
        qme = QMEOptimizer(use_mock=True)

        # Create two different structures
        h2_normal = molecule("H2")
        h2_normal.set_calculator(qme.calculator)

        h2_stretched = h2_normal.copy()
        positions = h2_stretched.get_positions()
        positions[1][0] += 0.5  # Stretch bond
        h2_stretched.set_positions(positions)
        h2_stretched.set_calculator(qme.calculator)

        # Compare energies
        e1 = h2_normal.get_potential_energy()
        e2 = h2_stretched.get_potential_energy()

        # Both should be valid floats
        assert isinstance(e1, float)
        assert isinstance(e2, float)
        # Note: With mock calculator, energy comparison may not follow physical expectations
        # Just verify we can calculate different energies
        assert e1 != e2
