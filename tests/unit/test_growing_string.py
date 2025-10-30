"""Test growing string method implementation."""

import pytest
from ase import Atoms

from qme.core.explorer import Explorer

# Function runner removed - use Explorer class instead


class TestGrowingStringMethod:
    """Test suite for growing string method."""

    def test_growing_string_basic(self) -> None:
        """Test basic growing string functionality with mock backend."""
        # Create simple reactant and product structures (H2 molecule)
        reactant = Atoms("H2", positions=[(0, 0, 0), (0.7, 0, 0)])
        product = Atoms("H2", positions=[(0, 0, 0), (1.5, 0, 0)])

        # Create explorer with mock backend
        explorer = Explorer(
            [reactant, product],
            backend="mock",
            target="ts",
            strategy="growing_string",
        )

        # Run growing string method
        result = explorer.run(
            npoints=10,
            fmax=0.5,
            steps=20,
            step_size=0.1,
            optimize_endpoints=False,  # Skip endpoint optimization for speed
            refine_ts=False,  # Skip TS refinement for speed
        )

        # Verify result structure
        assert isinstance(result, dict)
        assert "optimized_atoms" in result
        assert "trajectory" in result
        assert "converged" in result
        assert "strategy" in result
        assert "forward_string" in result
        assert "backward_string" in result

        # Verify strategy name
        assert result["strategy"] == "ts:growing_string"

        # Verify trajectory is a list
        assert isinstance(result["trajectory"], list)
        assert len(result["trajectory"]) >= 2

        # Verify forward and backward strings
        assert isinstance(result["forward_string"], list)
        assert isinstance(result["backward_string"], list)
        assert len(result["forward_string"]) >= 1
        assert len(result["backward_string"]) >= 1

        # Verify optimized_atoms is an Atoms object
        assert isinstance(result["optimized_atoms"], Atoms)

    def test_growing_string_requires_two_atoms(self) -> None:
        """Test that growing string requires exactly two Atoms objects."""
        single_atoms = Atoms("H2", positions=[(0, 0, 0), (0.7, 0, 0)])
        explorer = Explorer(single_atoms, backend="mock", target="ts", strategy="growing_string")

        # Should raise ValueError for single Atoms
        with pytest.raises(ValueError, match="exactly 2 Atoms objects"):
            explorer.run()

        # Should raise ValueError for more than two Atoms
        reactant = Atoms("H2", positions=[(0, 0, 0), (0.7, 0, 0)])
        product = Atoms("H2", positions=[(0, 0, 0), (1.5, 0, 0)])
        intermediate = Atoms("H2", positions=[(0, 0, 0), (1.0, 0, 0)])

        explorer_multi = Explorer(
            [reactant, intermediate, product],
            backend="mock",
            target="ts",
            strategy="growing_string",
        )
        with pytest.raises(ValueError, match="exactly 2 Atoms objects"):
            explorer_multi.run()

    def test_growing_string_with_ts_refinement_fails(self) -> None:
        """Test growing string with TS refinement fails with mock backend."""
        reactant = Atoms("H2", positions=[(0, 0, 0), (0.7, 0, 0)])
        product = Atoms("H2", positions=[(0, 0, 0), (1.5, 0, 0)])

        explorer = Explorer(
            [reactant, product],
            backend="mock",
            target="ts",
            strategy="growing_string",
        )

        # Mock backend doesn't support TS optimization, so this should fail
        with pytest.raises(ValueError, match="not suitable for transition state"):
            explorer.run(
                npoints=8,
                fmax=0.5,
                steps=10,
                optimize_endpoints=False,
                refine_ts=True,  # This will fail with mock backend
            )

    def test_growing_string_strategy_registration(self) -> None:
        """Test that growing string strategy is properly registered with correct metadata."""
        # Create reactant and product
        reactant = Atoms("H2", positions=[(0, 0, 0), (0.7, 0, 0)])
        product = Atoms("H2", positions=[(0, 0, 0), (1.5, 0, 0)])

        explorer = Explorer([reactant, product], backend="mock")

        # List all strategies and verify growing_string is present
        all_strategies = explorer.list_strategies()
        assert "ts:growing_string" in all_strategies
        assert "growing_string" in all_strategies
        assert "gsm" in all_strategies

        # Verify it's registered as multi-structure strategy for TS
        assert all_strategies["ts:growing_string"]["type"] == "multi-structure"
        assert "growing string" in all_strategies["ts:growing_string"]["description"].lower()

    def test_growing_string_limits_and_thresholds(self) -> None:
        """Test that growing string respects limits and uses thresholds correctly."""
        reactant = Atoms("H2", positions=[(0, 0, 0), (0.7, 0, 0)])
        product = Atoms("H2", positions=[(0, 0, 0), (1.5, 0, 0)])

        explorer = Explorer(
            [reactant, product],
            backend="mock",
            target="ts",
            strategy="growing_string",
        )
        result = explorer.run(
            npoints=10,
            fmax=0.5,
            steps=50,
            optimize_endpoints=False,
            refine_ts=False,
        )

        # Total images should not exceed npoints
        total_images = len(result["forward_string"]) + len(result["backward_string"])
        assert total_images <= 10
        assert len(result["trajectory"]) >= 2
