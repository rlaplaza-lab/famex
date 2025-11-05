"""Tests for TS interpolate strategy."""

from __future__ import annotations

import pytest

from qme.core.explorer import Explorer
from qme.strategies.ts_interpolate import MultiStructureTSGuessStrategy
from tests.test_utils import TestMoleculeFactory


class TestMultiStructureTSGuessStrategy:
    """Tests for MultiStructureTSGuessStrategy."""

    def test_strategy_metadata(self):
        """Test strategy metadata is correct."""
        assert MultiStructureTSGuessStrategy.metadata.name == "ts:interpolate"
        assert MultiStructureTSGuessStrategy.metadata.target == "ts"
        assert MultiStructureTSGuessStrategy.metadata.strategy == "interpolate"
        assert MultiStructureTSGuessStrategy.metadata.requires_multiple_structures is True

    def test_run_basic(self):
        """Test basic run functionality - expects ValueError with mock backend."""
        reactant = TestMoleculeFactory.get_water_distorted()
        product = TestMoleculeFactory.get_water_distorted()
        # Slightly modify product
        pos = product.get_positions()
        pos[1, 0] += 0.2
        product.set_positions(pos)

        explorer = Explorer(
            [reactant, product], backend="mock", target="ts", strategy="interpolate"
        )
        strategy = MultiStructureTSGuessStrategy(explorer)

        # Mock backend is not allowed for TS optimization
        with pytest.raises(ValueError, match="not suitable for transition state optimization"):
            strategy.run([reactant, product], npoints=5, fmax=0.5, steps=10)

    def test_run_with_validate_ts(self):
        """Test run with validate_ts=True option - expects ValueError with mock backend."""
        reactant = TestMoleculeFactory.get_water_distorted()
        product = TestMoleculeFactory.get_water_distorted()
        pos = product.get_positions()
        pos[1, 0] += 0.2
        product.set_positions(pos)

        explorer = Explorer(
            [reactant, product], backend="mock", target="ts", strategy="interpolate"
        )
        strategy = MultiStructureTSGuessStrategy(explorer)

        # Mock backend is not allowed for TS optimization
        with pytest.raises(ValueError, match="not suitable for transition state optimization"):
            strategy.run([reactant, product], npoints=5, fmax=0.5, steps=10, validate_ts=True)

    def test_run_with_calculate_frequencies(self):
        """Test run with calculate_frequencies=True - expects ValueError with mock backend."""
        reactant = TestMoleculeFactory.get_water_distorted()
        product = TestMoleculeFactory.get_water_distorted()
        pos = product.get_positions()
        pos[1, 0] += 0.2
        product.set_positions(pos)

        explorer = Explorer(
            [reactant, product], backend="mock", target="ts", strategy="interpolate"
        )
        strategy = MultiStructureTSGuessStrategy(explorer)

        # Mock backend is not allowed for TS optimization
        with pytest.raises(ValueError, match="not suitable for transition state optimization"):
            strategy.run(
                [reactant, product], npoints=5, fmax=0.5, steps=10, calculate_frequencies=True
            )

    def test_run_with_both_options(self):
        """Test run with both validate_ts and calculate_frequencies - expects ValueError."""
        reactant = TestMoleculeFactory.get_water_distorted()
        product = TestMoleculeFactory.get_water_distorted()
        pos = product.get_positions()
        pos[1, 0] += 0.2
        product.set_positions(pos)

        explorer = Explorer(
            [reactant, product], backend="mock", target="ts", strategy="interpolate"
        )
        strategy = MultiStructureTSGuessStrategy(explorer)

        # Mock backend is not allowed for TS optimization
        with pytest.raises(ValueError, match="not suitable for transition state optimization"):
            strategy.run(
                [reactant, product],
                npoints=5,
                fmax=0.5,
                steps=10,
                validate_ts=True,
                calculate_frequencies=True,
            )

    def test_run_with_custom_optimizer(self):
        """Test run with custom local_optimizer_name - expects ValueError with mock backend."""
        reactant = TestMoleculeFactory.get_water_distorted()
        product = TestMoleculeFactory.get_water_distorted()
        pos = product.get_positions()
        pos[1, 0] += 0.2
        product.set_positions(pos)

        explorer = Explorer(
            [reactant, product], backend="mock", target="ts", strategy="interpolate"
        )
        strategy = MultiStructureTSGuessStrategy(explorer)

        # Mock backend is not allowed for TS optimization
        with pytest.raises(ValueError, match="not suitable for transition state optimization"):
            strategy.run(
                [reactant, product],
                npoints=5,
                fmax=0.5,
                steps=10,
                local_optimizer_name="sella",
            )

    def test_run_with_different_methods(self):
        """Test run with different interpolation methods - expects ValueError with mock backend."""
        reactant = TestMoleculeFactory.get_water_distorted()
        product = TestMoleculeFactory.get_water_distorted()
        pos = product.get_positions()
        pos[1, 0] += 0.2
        product.set_positions(pos)

        explorer = Explorer(
            [reactant, product], backend="mock", target="ts", strategy="interpolate"
        )
        strategy = MultiStructureTSGuessStrategy(explorer)

        # Mock backend is not allowed for TS optimization
        for method in ["linear", "geodesic"]:
            with pytest.raises(ValueError, match="not suitable for transition state optimization"):
                strategy.run([reactant, product], npoints=5, method=method, fmax=0.5, steps=10)

    def test_run_with_different_npoints(self):
        """Test run with different npoints values - expects ValueError with mock backend."""
        reactant = TestMoleculeFactory.get_water_distorted()
        product = TestMoleculeFactory.get_water_distorted()
        pos = product.get_positions()
        pos[1, 0] += 0.2
        product.set_positions(pos)

        explorer = Explorer(
            [reactant, product], backend="mock", target="ts", strategy="interpolate"
        )
        strategy = MultiStructureTSGuessStrategy(explorer)

        # Mock backend is not allowed for TS optimization
        for npoints in [3, 5, 11]:
            with pytest.raises(ValueError, match="not suitable for transition state optimization"):
                strategy.run([reactant, product], npoints=npoints, fmax=0.5, steps=10)

    def test_run_requires_multiple_structures(self):
        """Test that strategy requires multiple structures."""
        single_atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(single_atoms, backend="mock", target="ts", strategy="interpolate")
        strategy = MultiStructureTSGuessStrategy(explorer)

        # Should raise ValueError for single structure
        with pytest.raises((ValueError, TypeError)):
            strategy.run([single_atoms])

    def test_run_result_structure(self):
        """Test that result has expected structure - expects ValueError with mock backend."""
        reactant = TestMoleculeFactory.get_water_distorted()
        product = TestMoleculeFactory.get_water_distorted()
        pos = product.get_positions()
        pos[1, 0] += 0.2
        product.set_positions(pos)

        explorer = Explorer(
            [reactant, product], backend="mock", target="ts", strategy="interpolate"
        )
        strategy = MultiStructureTSGuessStrategy(explorer)

        # Mock backend is not allowed for TS optimization
        with pytest.raises(ValueError, match="not suitable for transition state optimization"):
            strategy.run([reactant, product], npoints=5, fmax=0.5, steps=10)

    def test_run_without_explorer(self):
        """Test run without explorer (should still work but may have limitations)."""
        reactant = TestMoleculeFactory.get_water_distorted()
        product = TestMoleculeFactory.get_water_distorted()
        pos = product.get_positions()
        pos[1, 0] += 0.2
        product.set_positions(pos)

        strategy = MultiStructureTSGuessStrategy(None)

        # Should raise error or handle gracefully
        with pytest.raises((AttributeError, RuntimeError)):
            strategy.run([reactant, product], npoints=5, fmax=0.5, steps=10)
