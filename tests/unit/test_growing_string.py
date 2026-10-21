from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from famex.core.explorer import Explorer
from famex.strategies.growing_string import MultiStructureGrowingStringStrategy
from tests.test_constants import (
    COMPREHENSIVE_STEPS,
    DEFAULT_STEPS,
    QUICK_STEPS_EXTENDED,
    VERY_LOOSE_FMAX,
)


class TestGrowingStringMethod:
    def test_growing_string_basic(self):
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
            fmax=VERY_LOOSE_FMAX,
            steps=DEFAULT_STEPS,
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
        single_atoms = Atoms("H2", positions=[(0, 0, 0), (0.7, 0, 0)])
        explorer = Explorer(single_atoms, backend="mock", target="ts", strategy="growing_string")

        # Should raise ValueError for single Atoms
        # BaseStrategy validates first, so error message is from BaseStrategy
        with pytest.raises(ValueError, match=r"requires multiple structures.*at least 2"):
            explorer.run()

        # Should raise ValueError for more than two Atoms
        # Growing string strategy validates for exactly 2 after BaseStrategy check passes
        reactant = Atoms("H2", positions=[(0, 0, 0), (0.7, 0, 0)])
        product = Atoms("H2", positions=[(0, 0, 0), (1.5, 0, 0)])
        intermediate = Atoms("H2", positions=[(0, 0, 0), (1.0, 0, 0)])

        explorer_multi = Explorer(
            [reactant, intermediate, product],
            backend="mock",
            target="ts",
            strategy="growing_string",
        )
        # With 3 atoms, BaseStrategy check passes, so we get the specific growing_string error
        with pytest.raises(ValueError, match="exactly 2 Atoms objects"):
            explorer_multi.run()

    def test_growing_string_with_ts_refinement_fails(self):
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
                fmax=VERY_LOOSE_FMAX,
                steps=QUICK_STEPS_EXTENDED,
                optimize_endpoints=False,
                refine_ts=True,  # This will fail with mock backend
            )

    def test_growing_string_strategy_registration(self):
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

    def test_growing_string_limits_and_thresholds(self):
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
            fmax=VERY_LOOSE_FMAX,
            steps=COMPREHENSIVE_STEPS,
            optimize_endpoints=False,
            refine_ts=False,
        )

        # Total images should not exceed npoints
        total_images = len(result["forward_string"]) + len(result["backward_string"])
        assert total_images <= 10
        assert len(result["trajectory"]) >= 2


def _bare_string() -> MultiStructureGrowingStringStrategy:
    """Strategy object with string geometry only; no explorer or calculator."""
    gsm = MultiStructureGrowingStringStrategy.__new__(MultiStructureGrowingStringStrategy)
    gsm.explorer = None
    gsm.max_nodes = 9
    gsm.sk = 1.0 / (gsm.max_nodes - 1)
    gsm.all_energies = []
    gsm.all_forces = []
    return gsm


class TestGrowingStringGeometry:
    def test_right_frontier_advances_away_from_the_product(self):
        reactant = Atoms("H3", positions=[[0.0, 0.0, 0.0], [0.9, 0.2, 0.0], [1.6, 0.8, 0.1]])
        product = Atoms("H3", positions=[[0.2, 1.5, 0.0], [1.1, 1.6, 0.2], [2.0, 1.2, 0.4]])
        gsm = _bare_string()
        gsm.left_string = [reactant]
        gsm.right_string = [product]
        gsm._set_images_from_strings()

        gsm.left_string.append(gsm.get_new_image(gsm.lf_ind))
        gsm._set_images_from_strings()
        gsm.right_string.append(gsm.get_new_image(gsm.rf_ind))
        gsm._set_images_from_strings()
        first_right = gsm.right_string[-1]
        gsm.right_string.append(gsm.get_new_image(gsm.rf_ind))
        gsm._set_images_from_strings()
        newest = gsm.right_string[-1]
        product_node = gsm.right_string[0]

        newest_gap = float(np.linalg.norm(newest.positions - product_node.positions))
        first_gap = float(np.linalg.norm(first_right.positions - product_node.positions))
        assert newest_gap > first_gap

        assert gsm.images[gsm.rf_ind] is newest
        frontier_gap = float(
            np.linalg.norm(gsm.left_string[-1].positions - gsm.right_string[-1].positions)
        )
        product_gap = float(
            np.linalg.norm(gsm.left_string[-1].positions - gsm.right_string[0].positions)
        )
        neighbor_gap = float(
            np.linalg.norm(gsm.images[gsm.lf_ind].positions - gsm.images[gsm.rf_ind].positions)
        )
        assert frontier_gap == pytest.approx(neighbor_gap)
        assert frontier_gap < product_gap

    def test_interior_tangent_blends_both_chords_at_equal_energy_drops(self):
        gsm = _bare_string()
        reactant = Atoms("H", positions=[[0.0, 0.0, 0.0]])
        left = Atoms("H", positions=[[1.0, 0.0, 0.0]])
        middle = Atoms("H", positions=[[1.0, 1.0, 0.0]])
        right = Atoms("H", positions=[[2.0, 1.0, 0.0]])
        product = Atoms("H", positions=[[3.0, 0.0, 0.0]])
        gsm.left_string = [reactant, left, middle]
        gsm.right_string = [product, right]
        gsm._set_images_from_strings()
        gsm.all_energies = [[0.0, 1.0, 0.0, 0.0, 0.0]]

        tangent = gsm.get_tangent(1)
        v_prev = (left.positions - reactant.positions).ravel()
        v_next = (middle.positions - left.positions).ravel()
        expected = v_prev + v_next
        expected = expected / np.linalg.norm(expected)
        assert tangent == pytest.approx(expected)
        crossed = v_next - v_prev
        crossed = crossed / np.linalg.norm(crossed)
        assert abs(float(np.dot(tangent, crossed))) < 0.2

    def test_reparametrization_anchors_the_product_end(self):
        gsm = _bare_string()
        gsm.max_nodes = 11
        gsm.sk = 1.0 / (gsm.max_nodes - 1)
        reactant = Atoms("H", positions=[[0.0, 0.0, 0.0]])
        left = Atoms("H", positions=[[1.0, 0.0, 0.0]])
        right = Atoms("H", positions=[[8.0, 0.0, 0.0]])
        product = Atoms("H", positions=[[10.0, 0.0, 0.0]])
        gsm.left_string = [reactant, left]
        gsm.right_string = [product, right]
        gsm._set_images_from_strings()

        assert list(gsm.full_string_image_inds) == [0, 1, 9, 10]
        gsm.reparam_cart(gsm.sk * gsm.full_string_image_inds)

        assert float(left.positions[0, 0]) == pytest.approx(1.0, abs=1e-6)
        assert float(right.positions[0, 0]) == pytest.approx(9.0, abs=1e-6)
        assert float(product.positions[0, 0]) == pytest.approx(10.0)
