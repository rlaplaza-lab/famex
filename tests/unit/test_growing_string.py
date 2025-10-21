"""Test growing string method implementation."""

import pytest
from ase import Atoms

from qme.core.explorer import Explorer

# Function runner removed - use Explorer class instead


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
        # Set the atoms_list on the explorer first
        explorer.atoms_list = [reactant, product]
        result = explorer.run(
            mode="ts:growing_string",
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

    def test_growing_string_requires_two_atoms(self):
        """Test that growing string requires exactly two Atoms objects."""
        single_atoms = Atoms("H2", positions=[(0, 0, 0), (0.7, 0, 0)])
        explorer = Explorer(single_atoms, backend="mock")

        # Should raise ValueError for single Atoms
        explorer.atoms_list = single_atoms
        with pytest.raises(ValueError, match="requires two Atoms objects"):
            explorer.run(mode="ts:growing_string")

        # Should raise ValueError for more than two Atoms
        reactant = Atoms("H2", positions=[(0, 0, 0), (0.7, 0, 0)])
        product = Atoms("H2", positions=[(0, 0, 0), (1.5, 0, 0)])
        intermediate = Atoms("H2", positions=[(0, 0, 0), (1.0, 0, 0)])

        explorer.atoms_list = [reactant, intermediate, product]
        with pytest.raises(ValueError, match="exactly 2 Atoms objects"):
            explorer.run(mode="ts:growing_string")

    @pytest.mark.parametrize(
        "optimize_endpoints,refine_ts",
        [
            (True, False),
            (False, False),
        ],
    )
    def test_growing_string_optimization_options(self, optimize_endpoints, refine_ts):
        """Test growing string with different optimization options."""
        reactant = Atoms("H2", positions=[(0, 0, 0), (0.7, 0, 0)])
        product = Atoms("H2", positions=[(0, 0, 0), (1.5, 0, 0)])

        explorer = Explorer(reactant, backend="mock")
        explorer.atoms_list = [reactant, product]

        if refine_ts:
            # Mock backend doesn't support TS optimization, so this should fail
            with pytest.raises(ValueError, match="not suitable for transition state"):
                explorer.run(
                    mode="ts:growing_string",
                    npoints=8,
                    fmax=0.5,
                    steps=10,
                    optimize_endpoints=optimize_endpoints,
                    refine_ts=refine_ts,
                )
        else:
            result = explorer.run(
                mode="ts:growing_string",
                npoints=8,
                fmax=0.5,
                steps=10,
                optimize_endpoints=optimize_endpoints,
                refine_ts=refine_ts,
            )
            # Should complete successfully
            assert result["strategy"] == "ts:growing_string"
            assert len(result["trajectory"]) >= 2

    def test_growing_string_with_ts_refinement_fails(self):
        """Test growing string with TS refinement fails with mock backend."""
        reactant = Atoms("H2", positions=[(0, 0, 0), (0.7, 0, 0)])
        product = Atoms("H2", positions=[(0, 0, 0), (1.5, 0, 0)])

        explorer = Explorer(reactant, backend="mock")
        explorer.atoms_list = [reactant, product]

        # Mock backend doesn't support TS optimization, so this should fail
        with pytest.raises(ValueError, match="not suitable for transition state"):
            explorer.run(
                mode="ts:growing_string",
                npoints=8,
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
        assert strategies["ts:growing_string"]["type"] == "multi-structure"

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
        assert all_strategies["ts:growing_string"]["type"] == "multi-structure"
        assert "growing string" in all_strategies["ts:growing_string"]["description"].lower()

    @pytest.mark.parametrize("npoints,steps", [(5, 100), (10, 50), (3, 20)])
    def test_growing_string_max_images_limit(self, npoints, steps):
        """Test that growing string respects maximum images limit."""
        reactant = Atoms("H2", positions=[(0, 0, 0), (0.7, 0, 0)])
        product = Atoms("H2", positions=[(0, 0, 0), (1.5, 0, 0)])

        explorer = Explorer(reactant, backend="mock")
        explorer.atoms_list = [reactant, product]
        result = explorer.run(
            mode="ts:growing_string",
            npoints=npoints,
            fmax=0.5,
            steps=steps,
            optimize_endpoints=False,
            refine_ts=False,
        )

        # Total images should not exceed npoints
        total_images = len(result["forward_string"]) + len(result["backward_string"])
        assert total_images <= npoints

    @pytest.mark.parametrize("distance_threshold", [0.5, 1.0, 2.0])
    def test_growing_string_distance_threshold(self, distance_threshold):
        """Test that growing string uses distance threshold for convergence."""
        reactant = Atoms("H2", positions=[(0, 0, 0), (0.7, 0, 0)])
        product = Atoms("H2", positions=[(0, 0, 0), (0.8, 0, 0)])  # Very close

        explorer = Explorer(reactant, backend="mock")
        explorer.atoms_list = [reactant, product]
        result = explorer.run(
            mode="ts:growing_string",
            npoints=20,
            fmax=0.5,
            steps=50,
            distance_threshold=distance_threshold,
            optimize_endpoints=False,
            refine_ts=False,
        )

        # Should complete with strings_met flag
        assert "strings_met" in result
        # Trajectory should exist
        assert len(result["trajectory"]) >= 2
