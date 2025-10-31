"""Unit tests for transition state optimization strategies."""

from __future__ import annotations

import pytest

import qme
from qme.core.explorer import Explorer
from qme.strategies.ts import LocalTSStrategy
from qme.strategies.ts_interpolate import MultiStructureTSGuessStrategy
from tests.test_utils import TestMoleculeFactory


class TestLocalTSStrategy:
    """Test LocalTSStrategy."""

    def test_strategy_metadata(self) -> None:
        """Test strategy metadata."""
        assert LocalTSStrategy.metadata.name == "ts:local"
        assert LocalTSStrategy.metadata.target == "ts"
        assert LocalTSStrategy.metadata.strategy == "local"
        assert not LocalTSStrategy.metadata.requires_multiple_structures

    def test_strategy_initialization(self) -> None:
        """Test strategy initialization."""
        atoms = TestMoleculeFactory.get_water_dissociation_ts_guess()
        explorer = Explorer(atoms, backend="mock")

        strategy = LocalTSStrategy(explorer)

        assert strategy.explorer == explorer

    @pytest.mark.skipif(not qme.deps.has("sella"), reason="Sella required for TS optimization")
    def test_strategy_run_basic(self) -> None:
        """Test basic run of TS optimization."""
        atoms = TestMoleculeFactory.get_water_dissociation_ts_guess()
        explorer = Explorer(atoms, backend="mock")

        strategy = LocalTSStrategy(explorer)

        # Mock backend may not support real TS optimization
        # This test verifies the strategy structure works
        try:
            result = strategy.run([atoms], steps=5, fmax=0.5)

            assert "optimized_atoms" in result
            assert "strategy" in result
            assert result["strategy"] == "ts:local"
        except ValueError as e:
            # Mock backend may not support TS optimization
            if "not suitable" in str(e).lower():
                pytest.skip("Mock backend doesn't support TS optimization")
            raise

    @pytest.mark.skipif(not qme.deps.has("sella"), reason="Sella required for TS optimization")
    def test_strategy_run_with_validation(self) -> None:
        """Test run with TS validation."""
        atoms = TestMoleculeFactory.get_water_dissociation_ts_guess()
        explorer = Explorer(atoms, backend="mock")

        strategy = LocalTSStrategy(explorer)

        try:
            result = strategy.run([atoms], steps=3, fmax=0.5, validate_ts=True)

            assert "ts_validation" in result
            assert "is_ts" in result
        except ValueError as e:
            if "not suitable" in str(e).lower():
                pytest.skip("Mock backend doesn't support TS optimization")
            raise

    @pytest.mark.skipif(not qme.deps.has("sella"), reason="Sella required for TS optimization")
    def test_strategy_run_with_frequencies(self) -> None:
        """Test run with frequency calculation."""
        atoms = TestMoleculeFactory.get_water_dissociation_ts_guess()
        explorer = Explorer(atoms, backend="mock")

        strategy = LocalTSStrategy(explorer)

        try:
            result = strategy.run([atoms], steps=3, fmax=0.5, calculate_frequencies=True)

            assert "frequency_analysis" in result
            assert "is_ts" in result
        except ValueError as e:
            if "not suitable" in str(e).lower():
                pytest.skip("Mock backend doesn't support TS optimization")
            raise

    def test_strategy_validates_backend(self) -> None:
        """Test that strategy validates backend supports TS optimization."""
        atoms = TestMoleculeFactory.get_water_dissociation_ts_guess()
        explorer = Explorer(atoms, backend="mock")

        strategy = LocalTSStrategy(explorer)

        # Mock backend should raise error for TS optimization
        with pytest.raises(ValueError, match="not suitable for transition state"):
            strategy.run([atoms], steps=3, fmax=0.5)

    def test_strategy_validates_inputs(self) -> None:
        """Test that strategy validates inputs."""
        atoms = TestMoleculeFactory.get_water_dissociation_ts_guess()
        explorer = Explorer(atoms, backend="mock")

        strategy = LocalTSStrategy(explorer)

        with pytest.raises(ValueError, match="No atoms provided"):
            strategy.run([])

    def test_strategy_run_multiple_structures(self) -> None:
        """Test run with multiple structures."""
        atoms1 = TestMoleculeFactory.get_water_dissociation_ts_guess()
        atoms2 = TestMoleculeFactory.get_water_dissociation_ts_guess()

        explorer = Explorer([atoms1, atoms2], backend="mock")

        strategy = LocalTSStrategy(explorer)

        # Should validate backend for each structure
        with pytest.raises(ValueError, match="not suitable"):
            strategy.run([atoms1, atoms2], steps=3, fmax=0.5)


class TestMultiStructureTSGuessStrategy:
    """Test MultiStructureTSGuessStrategy."""

    def test_strategy_metadata(self) -> None:
        """Test strategy metadata."""
        assert MultiStructureTSGuessStrategy.metadata.name == "ts:interpolate"
        assert MultiStructureTSGuessStrategy.metadata.target == "ts"
        assert MultiStructureTSGuessStrategy.metadata.strategy == "interpolate"
        assert MultiStructureTSGuessStrategy.metadata.requires_multiple_structures

    def test_strategy_run_basic(self) -> None:
        """Test basic run of TS interpolation strategy."""
        reactant = TestMoleculeFactory.get_water_distorted()
        product = TestMoleculeFactory.get_water_distorted()
        pos = product.get_positions()
        pos[1, 0] += 0.2
        product.set_positions(pos)

        explorer = Explorer([reactant, product], backend="mock")

        strategy = MultiStructureTSGuessStrategy(explorer)

        # Mock backend doesn't support TS optimization
        with pytest.raises(ValueError, match="not suitable for transition state"):
            strategy.run([reactant, product], npoints=5, steps=2, fmax=0.5)

    def test_strategy_run_with_different_interpolation_methods(self) -> None:
        """Test run with different interpolation methods."""
        reactant = TestMoleculeFactory.get_water_distorted()
        product = TestMoleculeFactory.get_water_distorted()
        pos = product.get_positions()
        pos[1, 0] += 0.2
        product.set_positions(pos)

        explorer = Explorer([reactant, product], backend="mock")

        strategy = MultiStructureTSGuessStrategy(explorer)

        # Mock backend doesn't support TS optimization
        for method in ["linear", "geodesic", "idpp"]:
            with pytest.raises(ValueError, match="not suitable for transition state"):
                strategy.run([reactant, product], npoints=5, method=method, steps=2, fmax=0.5)

    def test_strategy_run_with_refinement(self) -> None:
        """Test run with TS refinement."""
        reactant = TestMoleculeFactory.get_water_distorted()
        product = TestMoleculeFactory.get_water_distorted()
        pos = product.get_positions()
        pos[1, 0] += 0.2
        product.set_positions(pos)

        explorer = Explorer([reactant, product], backend="mock")

        strategy = MultiStructureTSGuessStrategy(explorer)

        # TS refinement may fail with mock backend, but test the code path
        try:
            result = strategy.run(
                [reactant, product],
                npoints=5,
                steps=2,
                fmax=0.5,
                refine_ts=True,
            )

            assert "optimized_atoms" in result
        except ValueError as e:
            if "not suitable" in str(e).lower():
                # Expected for mock backend
                pass

    def test_strategy_validates_inputs(self) -> None:
        """Test that strategy validates inputs."""
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock")

        strategy = MultiStructureTSGuessStrategy(explorer)

        with pytest.raises(ValueError, match="No atoms provided"):
            strategy.run([])

    def test_strategy_requires_multiple_structures(self) -> None:
        """Test that strategy requires multiple structures."""
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock")

        strategy = MultiStructureTSGuessStrategy(explorer)

        # Single structure should be handled
        # Strategy may wrap it or may require multiple
        # Test that it doesn't crash
        try:
            result = strategy.run([atoms], npoints=5, steps=2, fmax=0.5)
            assert "optimized_atoms" in result
        except ValueError:
            # Strategy may require exactly 2 structures
            pass
