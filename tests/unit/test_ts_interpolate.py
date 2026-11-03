"""Tests for TS interpolate strategy."""

from __future__ import annotations

from typing import Any

import pytest

from famex.core.explorer import Explorer
from famex.strategies.ts_interpolate import MultiStructureTSGuessStrategy
from tests.test_constants import DEFAULT_STEPS, VERY_LOOSE_FMAX


class TestMultiStructureTSGuessStrategy:
    """Tests for MultiStructureTSGuessStrategy."""

    def test_strategy_metadata(self):
        assert MultiStructureTSGuessStrategy.metadata.name == "ts:interpolate"
        assert MultiStructureTSGuessStrategy.metadata.target == "ts"
        assert MultiStructureTSGuessStrategy.metadata.strategy == "interpolate"
        assert MultiStructureTSGuessStrategy.metadata.requires_multiple_structures is True

    @pytest.mark.parametrize(
        "run_kwargs",
        [
            {},
            {"validate_ts": True},
            {"calculate_frequencies": True},
            {"validate_ts": True, "calculate_frequencies": True},
            {"local_optimizer_name": "sella"},
            {"method": "linear"},
            {"method": "geodesic"},
            {"npoints": 3},
            {"npoints": 11},
        ],
        ids=[
            "basic",
            "validate_ts",
            "frequencies",
            "both_options",
            "custom_optimizer",
            "method_linear",
            "method_geodesic",
            "npoints_3",
            "npoints_11",
        ],
    )
    def test_mock_backend_rejected(self, water_molecule, run_kwargs: dict[str, Any]):
        reactant = water_molecule.copy()
        product = water_molecule.copy()
        pos = product.get_positions()
        pos[1, 0] += 0.2
        product.set_positions(pos)

        explorer = Explorer(
            [reactant, product], backend="mock", target="ts", strategy="interpolate"
        )
        strategy = MultiStructureTSGuessStrategy(explorer)

        kwargs: dict[str, Any] = {
            "npoints": 5,
            "fmax": VERY_LOOSE_FMAX,
            "steps": DEFAULT_STEPS,
            **run_kwargs,
        }
        with pytest.raises(ValueError, match="not suitable for transition state optimization"):
            strategy.run([reactant, product], **kwargs)

    def test_run_requires_multiple_structures(self, water_molecule):
        single_atoms = water_molecule.copy()
        explorer = Explorer(single_atoms, backend="mock", target="ts", strategy="interpolate")
        strategy = MultiStructureTSGuessStrategy(explorer)

        with pytest.raises(ValueError, match="requires multiple structures"):
            strategy.run([single_atoms])

    def test_run_without_explorer(self, water_molecule):
        reactant = water_molecule.copy()
        product = water_molecule.copy()
        pos = product.get_positions()
        pos[1, 0] += 0.2
        product.set_positions(pos)

        strategy = MultiStructureTSGuessStrategy(None)

        with pytest.raises(RuntimeError, match="Failed to evaluate energy"):
            strategy.run([reactant, product], npoints=5, fmax=VERY_LOOSE_FMAX, steps=DEFAULT_STEPS)
