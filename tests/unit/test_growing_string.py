"""Test growing string method implementation."""

import pytest
from ase import Atoms

from qme.core.explorer import Explorer
from qme.core.twoended_strategies import twoended_growing_string_runner


class TestGrowingStringMethod:
    """Test suite for growing string method."""

    def test_growing_string_basic(self):
        """Test basic growing string functionality with mock backend."""
        # Create simple reactant and product structures (H2 molecule)
        reactant = Atoms("H2", positions=[(0, 0, 0), (0.7, 0, 0)])
        product = Atoms("H2", positions=[(0, 0, 0), (1.5, 0, 0)])

        # Create explorer with mock backend
        explorer = Explorer(reactant, backend="mock")

        # Run growing string method
        result = twoended_growing_string_runner(
            [reactant, product],
            npoints=10,
            explorer=explorer,
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
        assert result["strategy"] == "twoended_growing_string_runner"

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

    def test_growing_string_requires_two_atoms(self):
        """Test that growing string requires exactly two Atoms objects."""
        single_atoms = Atoms("H2", positions=[(0, 0, 0), (0.7, 0, 0)])
        explorer = Explorer(single_atoms, backend="mock")

        # Should raise ValueError for single Atoms
        with pytest.raises(ValueError, match="requires two Atoms objects"):
            twoended_growing_string_runner(single_atoms, explorer=explorer)

        # Should raise ValueError for more than two Atoms
        reactant = Atoms("H2", positions=[(0, 0, 0), (0.7, 0, 0)])
        product = Atoms("H2", positions=[(0, 0, 0), (1.5, 0, 0)])
        intermediate = Atoms("H2", positions=[(0, 0, 0), (1.0, 0, 0)])

        with pytest.raises(ValueError, match="exactly 2 Atoms objects"):
            twoended_growing_string_runner(
                [reactant, intermediate, product], explorer=explorer
            )

    def test_growing_string_with_endpoint_optimization(self):
        """Test growing string with endpoint optimization enabled."""
        reactant = Atoms("H2", positions=[(0, 0, 0), (0.7, 0, 0)])
        product = Atoms("H2", positions=[(0, 0, 0), (1.5, 0, 0)])

        explorer = Explorer(reactant, backend="mock")

        result = twoended_growing_string_runner(
            [reactant, product],
            npoints=8,
            explorer=explorer,
            fmax=0.5,
            steps=10,
            optimize_endpoints=True,  # Enable endpoint optimization
            refine_ts=False,
        )

        # Should still complete successfully
        assert result["strategy"] == "twoended_growing_string_runner"
        assert len(result["trajectory"]) >= 2

    def test_growing_string_with_ts_refinement(self):
        """Test growing string with TS refinement enabled."""
        reactant = Atoms("H2", positions=[(0, 0, 0), (0.7, 0, 0)])
        product = Atoms("H2", positions=[(0, 0, 0), (1.5, 0, 0)])

        explorer = Explorer(reactant, backend="mock")

        # Mock backend doesn't support TS optimization, so this should fail
        # when refine_ts=True. Test that it raises the appropriate error.
        with pytest.raises(ValueError, match="not suitable for transition state"):
            twoended_growing_string_runner(
                [reactant, product],
                npoints=8,
                explorer=explorer,
                fmax=0.5,
                steps=10,
                optimize_endpoints=False,
                refine_ts=True,  # This will fail with mock backend
            )

    def test_growing_string_strategy_registered(self):
        """Test that growing string strategy is properly registered."""
        atoms = Atoms("H2", positions=[(0, 0, 0), (0.7, 0, 0)])
        explorer = Explorer(atoms, backend="mock")

        # Check that strategy is registered
        strategies = explorer.list_strategies()
        assert "ts:growing_string" in strategies
        assert "growing_string" in strategies
        assert "gsm" in strategies

        # Verify strategy type
        assert strategies["ts:growing_string"]["type"] == "two-ended"

    def test_growing_string_via_explorer_run(self):
        """Test growing string method registration and basic interface."""
        # Create reactant and product
        reactant = Atoms("H2", positions=[(0, 0, 0), (0.7, 0, 0)])
        product = Atoms("H2", positions=[(0, 0, 0), (1.5, 0, 0)])

        # Verify strategy is properly registered with correct key
        explorer = Explorer([reactant, product], backend="mock")

        # List all strategies and verify growing_string is present
        all_strategies = explorer.list_strategies()
        assert "ts:growing_string" in all_strategies
        assert "growing_string" in all_strategies
        assert "gsm" in all_strategies

        # Verify it's registered as two-ended strategy for TS
        assert all_strategies["ts:growing_string"]["type"] == "two-ended"
        assert "growing string" in all_strategies["ts:growing_string"]["description"].lower()

    def test_growing_string_max_images_limit(self):
        """Test that growing string respects maximum images limit."""
        reactant = Atoms("H2", positions=[(0, 0, 0), (0.7, 0, 0)])
        product = Atoms("H2", positions=[(0, 0, 0), (1.5, 0, 0)])

        explorer = Explorer(reactant, backend="mock")

        result = twoended_growing_string_runner(
            [reactant, product],
            npoints=5,  # Small limit
            explorer=explorer,
            fmax=0.5,
            steps=100,  # Many steps but should stop at npoints
            optimize_endpoints=False,
            refine_ts=False,
        )

        # Total images should not exceed npoints
        total_images = len(result["forward_string"]) + len(result["backward_string"])
        assert total_images <= 5

    def test_growing_string_distance_threshold(self):
        """Test that growing string uses distance threshold for convergence."""
        reactant = Atoms("H2", positions=[(0, 0, 0), (0.7, 0, 0)])
        product = Atoms("H2", positions=[(0, 0, 0), (0.8, 0, 0)])  # Very close

        explorer = Explorer(reactant, backend="mock")

        result = twoended_growing_string_runner(
            [reactant, product],
            npoints=20,
            explorer=explorer,
            fmax=0.5,
            steps=50,
            distance_threshold=1.0,  # Large threshold
            optimize_endpoints=False,
            refine_ts=False,
        )

        # Should complete with strings_met flag
        assert "strings_met" in result
        # Trajectory should exist
        assert len(result["trajectory"]) >= 2
