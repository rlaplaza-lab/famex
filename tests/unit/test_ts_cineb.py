"""Tests for TS CI-NEB strategy."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from ase import Atoms

from famex.core.explorer import Explorer
from famex.strategies.neb_optimizer import NEBOptimizer
from famex.strategies.ts_cineb import MultiStructureTSCINEBStrategy
from tests.test_constants import DEFAULT_STEPS, VERY_LOOSE_FMAX


class TestMultiStructureTSCINEBStrategy:
    """Tests for MultiStructureTSCINEBStrategy."""

    def test_strategy_metadata(self):
        assert MultiStructureTSCINEBStrategy.metadata.name == "ts:cineb"
        assert MultiStructureTSCINEBStrategy.metadata.target == "ts"
        assert MultiStructureTSCINEBStrategy.metadata.strategy == "cineb"
        assert MultiStructureTSCINEBStrategy.metadata.requires_multiple_structures is True

    def test_run_basic(self, water_molecule):
        reactant = water_molecule.copy()
        product = water_molecule.copy()
        pos = product.get_positions()
        pos[1, 0] += 0.2
        product.set_positions(pos)

        explorer = Explorer([reactant, product], backend="mock", target="ts", strategy="cineb")
        strategy = MultiStructureTSCINEBStrategy(explorer)

        with pytest.raises(ValueError, match="not suitable for transition state optimization"):
            strategy.run([reactant, product], npoints=5, fmax=VERY_LOOSE_FMAX, steps=DEFAULT_STEPS)

    def test_run_requires_three_images(self, water_molecule):
        reactant = water_molecule.copy()
        product = water_molecule.copy()

        explorer = Explorer([reactant, product], backend="mock", target="ts", strategy="cineb")
        strategy = MultiStructureTSCINEBStrategy(explorer)

        with pytest.raises(ValueError, match="at least 3 images"):
            strategy.run([reactant, product], npoints=2, fmax=VERY_LOOSE_FMAX, steps=DEFAULT_STEPS)

    def test_explorer_explain_run(self, water_molecule):
        reactant = water_molecule.copy()
        product = water_molecule.copy()
        explorer = Explorer([reactant, product], backend="mock", target="ts", strategy="cineb")
        explanation = explorer.explain_run()
        assert explanation["strategy_key"] == "ts:cineb"
        assert explanation["valid"] is True

    def test_run_happy_path_with_rfo_local_optimizer(self, water_molecule):
        reactant = water_molecule.copy()
        product = water_molecule.copy()
        pos = product.get_positions()
        pos[1, 0] += 0.2
        product.set_positions(pos)

        explorer = Explorer(
            [reactant, product],
            backend="mock",
            target="ts",
            strategy="cineb",
            local_optimizer="rfo",
        )
        strategy = MultiStructureTSCINEBStrategy(explorer)

        def fake_neb_optimize(self: NEBOptimizer) -> list[Atoms]:
            self.climbing_image = len(self.images) // 2
            return self.images

        with (
            patch.object(NEBOptimizer, "optimize", fake_neb_optimize),
            patch("famex.strategies.ts._validate_ts_optimization_setup"),
        ):
            result = strategy.run(
                [reactant, product],
                npoints=5,
                fmax=VERY_LOOSE_FMAX,
                steps=5,
                local_optimizer_name="rfo",
            )

        assert "optimized_atoms" in result
        assert result.get("climb") is True
        assert result.get("ts_guess_index") is not None
        assert "converged" in result
