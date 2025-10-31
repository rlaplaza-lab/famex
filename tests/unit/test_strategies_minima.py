"""Unit tests for minima optimization strategies."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from qme.core.explorer import Explorer
from qme.strategies.minima import LocalMinimaStrategy
from qme.strategies.minima_interpolate import MultiStructureMinimaInterpolateStrategy
from tests.test_utils import TestMoleculeFactory


class TestLocalMinimaStrategy:
    """Test LocalMinimaStrategy."""

    def test_strategy_metadata(self) -> None:
        """Test strategy metadata."""
        assert LocalMinimaStrategy.metadata.name == "minima:local"
        assert LocalMinimaStrategy.metadata.target == "minima"
        assert LocalMinimaStrategy.metadata.strategy == "local"
        assert not LocalMinimaStrategy.metadata.requires_multiple_structures

    def test_strategy_initialization(self) -> None:
        """Test strategy initialization."""
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock")

        strategy = LocalMinimaStrategy(explorer)

        assert strategy.explorer == explorer
        assert strategy.profiler is None

    def test_strategy_run_basic(self) -> None:
        """Test basic run of minima optimization."""
        atoms = TestMoleculeFactory.get_h2_stretched()
        explorer = Explorer(atoms, backend="mock")

        strategy = LocalMinimaStrategy(explorer)

        result = strategy.run([atoms], steps=5, fmax=0.5)

        assert "optimized_atoms" in result
        assert "strategy" in result
        assert "converged" in result
        assert "steps_taken" in result
        assert result["strategy"] == "minima:local"

    def test_strategy_run_with_different_optimizers(self) -> None:
        """Test run with different local optimizers."""
        atoms = TestMoleculeFactory.get_h2_stretched()
        explorer = Explorer(atoms, backend="mock")

        strategy = LocalMinimaStrategy(explorer)

        # Test with LBFGS
        result_lbfgs = strategy.run([atoms], steps=3, fmax=0.5, local_optimizer_name="LBFGS")

        assert result_lbfgs["optimized_atoms"] is not None

        # Test with BFGS
        result_bfgs = strategy.run([atoms], steps=3, fmax=0.5, local_optimizer_name="BFGS")

        assert result_bfgs["optimized_atoms"] is not None

    def test_strategy_run_with_frequencies(self) -> None:
        """Test run with frequency calculation."""
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock")

        strategy = LocalMinimaStrategy(explorer)

        result = strategy.run(
            [atoms],
            steps=3,
            fmax=0.5,
            calculate_frequencies=True,
        )

        assert "frequency_analysis" in result
        assert "is_minimum" in result
        assert "free_energy_correction" in result

    def test_strategy_run_multiple_structures(self) -> None:
        """Test run with multiple structures."""
        atoms1 = TestMoleculeFactory.get_water_distorted()
        atoms2 = TestMoleculeFactory.get_water_distorted()

        explorer = Explorer([atoms1, atoms2], backend="mock")

        strategy = LocalMinimaStrategy(explorer)

        result = strategy.run([atoms1, atoms2], steps=3, fmax=0.5)

        assert isinstance(result["optimized_atoms"], list)
        assert len(result["optimized_atoms"]) == 2
        assert isinstance(result["converged"], list)
        assert len(result["converged"]) == 2

    def test_strategy_run_single_structure_list(self) -> None:
        """Test run with single-element list."""
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock")

        strategy = LocalMinimaStrategy(explorer)

        result = strategy.run([atoms], steps=3, fmax=0.5)

        # Single-element list should return single Atoms, not list
        assert isinstance(result["optimized_atoms"], type(atoms))
        assert isinstance(result["converged"], bool)

    def test_strategy_run_with_initial_hessian(self) -> None:
        """Test run with initial Hessian (when supported)."""
        atoms = TestMoleculeFactory.get_water_distorted()
        # Create hessian with correct size (3N x 3N)
        n = len(atoms) * 3
        explorer = Explorer(atoms, backend="mock", initial_hessian=np.eye(n))

        strategy = LocalMinimaStrategy(explorer)

        # Note: Sella optimizer may not accept hessian directly in PES wrapper
        # Test that strategy can handle initial_hessian attribute without error
        # Use LBFGS which doesn't require hessian
        result = strategy.run([atoms], steps=3, fmax=0.5, local_optimizer_name="LBFGS")

        assert result["optimized_atoms"] is not None

    def test_strategy_run_with_profiler(self) -> None:
        """Test run with profiler enabled."""
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock", profile=True)

        strategy = LocalMinimaStrategy(explorer, profiler=explorer.profiler)

        result = strategy.run([atoms], steps=3, fmax=0.5)

        assert "performance" in result
        assert result["optimized_atoms"] is not None

    def test_strategy_validates_inputs(self) -> None:
        """Test that strategy validates inputs."""
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock")

        strategy = LocalMinimaStrategy(explorer)

        with pytest.raises(ValueError, match="No atoms provided"):
            strategy.run([])

    def test_strategy_with_constraints(self) -> None:
        """Test strategy respects constraints."""
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock", constraints="fix 0")

        strategy = LocalMinimaStrategy(explorer)

        result = strategy.run([atoms], steps=3, fmax=0.5)

        assert result["optimized_atoms"] is not None

    def test_strategy_handles_optimizer_failure(self) -> None:
        """Test strategy handles optimizer failures gracefully."""
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock")

        strategy = LocalMinimaStrategy(explorer)

        # Run with very strict convergence that might not be reached
        result = strategy.run([atoms], steps=2, fmax=1e-10)

        # Should still return a result, even if not converged
        assert "optimized_atoms" in result
        assert "converged" in result


class TestMultiStructureMinimaInterpolateStrategy:
    """Test MultiStructureMinimaInterpolateStrategy."""

    def test_strategy_metadata(self) -> None:
        """Test strategy metadata."""
        assert MultiStructureMinimaInterpolateStrategy.metadata.name == "minima:interpolate"
        assert MultiStructureMinimaInterpolateStrategy.metadata.target == "minima"
        assert MultiStructureMinimaInterpolateStrategy.metadata.strategy == "interpolate"
        assert MultiStructureMinimaInterpolateStrategy.metadata.requires_multiple_structures

    def test_strategy_run_basic(self) -> None:
        """Test basic run of interpolate minima optimization."""
        atoms1 = TestMoleculeFactory.get_water_distorted()
        atoms2 = TestMoleculeFactory.get_water_distorted()
        # Modify second structure slightly
        pos = atoms2.get_positions()
        pos[1, 0] += 0.1
        atoms2.set_positions(pos)

        explorer = Explorer([atoms1, atoms2], backend="mock")

        strategy = MultiStructureMinimaInterpolateStrategy(explorer)

        result = strategy.run([atoms1, atoms2], npoints=5, steps=2, fmax=0.5)

        assert "optimized_atoms" in result
        assert "trajectory" in result
        assert "strategy" in result
        assert result["strategy"] == "minima:interpolate"
        assert isinstance(result["optimized_atoms"], list)

    def test_strategy_run_with_different_interpolation_methods(self) -> None:
        """Test run with different interpolation methods."""
        atoms1 = TestMoleculeFactory.get_water_distorted()
        atoms2 = TestMoleculeFactory.get_water_distorted()
        pos = atoms2.get_positions()
        pos[1, 0] += 0.1
        atoms2.set_positions(pos)

        explorer = Explorer([atoms1, atoms2], backend="mock")

        strategy = MultiStructureMinimaInterpolateStrategy(explorer)

        for method in ["linear", "geodesic", "idpp"]:
            result = strategy.run([atoms1, atoms2], npoints=5, method=method, steps=2, fmax=0.5)

            assert result["optimized_atoms"] is not None
            assert isinstance(result["optimized_atoms"], list)

    def test_strategy_run_with_frequencies(self) -> None:
        """Test run with frequency calculation."""
        atoms1 = TestMoleculeFactory.get_water_distorted()
        atoms2 = TestMoleculeFactory.get_water_distorted()
        pos = atoms2.get_positions()
        pos[1, 0] += 0.1
        atoms2.set_positions(pos)

        explorer = Explorer([atoms1, atoms2], backend="mock")

        strategy = MultiStructureMinimaInterpolateStrategy(explorer)

        result = strategy.run(
            [atoms1, atoms2],
            npoints=5,
            steps=2,
            fmax=0.5,
            calculate_frequencies=True,
        )

        assert "frequency_analysis" in result
        assert "is_minimum" in result

    def test_strategy_handles_optimization_failure(self) -> None:
        """Test strategy handles individual structure optimization failures."""
        atoms1 = TestMoleculeFactory.get_water_distorted()
        atoms2 = TestMoleculeFactory.get_water_distorted()
        pos = atoms2.get_positions()
        pos[1, 0] += 0.1
        atoms2.set_positions(pos)

        explorer = Explorer([atoms1, atoms2], backend="mock")

        strategy = MultiStructureMinimaInterpolateStrategy(explorer)

        # Mock LocalMinimaStrategy to raise exception for one structure
        # The strategy interpolates first, creating npoints structures
        # Then optimizes each one
        with patch.object(LocalMinimaStrategy, "run") as mock_run:
            # First call succeeds, second raises error
            mock_run.side_effect = [
                {"optimized_atoms": atoms1, "converged": True},
                ValueError("Error"),
            ]

            result = strategy.run([atoms1, atoms2], npoints=3, steps=2, fmax=0.5)

            # Should still return result with structures (some may have failed)
            assert "optimized_atoms" in result
            assert isinstance(result["optimized_atoms"], list)
            # May have npoints optimized structures (interpolated path)
            assert len(result["optimized_atoms"]) >= 2

    def test_strategy_validates_inputs(self) -> None:
        """Test that strategy validates inputs."""
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock")

        strategy = MultiStructureMinimaInterpolateStrategy(explorer)

        with pytest.raises(ValueError, match="No atoms provided"):
            strategy.run([])

    def test_strategy_requires_multiple_structures(self) -> None:
        """Test that strategy requires multiple structures."""
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock")

        strategy = MultiStructureMinimaInterpolateStrategy(explorer)

        # Strategy requires at least 2 structures for interpolation
        # Single structure should raise error
        with pytest.raises(ValueError, match="at least 2"):
            strategy.run([atoms], npoints=5, steps=2, fmax=0.5)
