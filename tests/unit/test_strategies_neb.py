"""Unit tests for NEB and CI-NEB optimization strategies."""

from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest

from qme.core.explorer import Explorer
from qme.strategies.cineb import MultiStructureCINEBStrategy
from qme.strategies.neb import MultiStructureNEBStrategy
from tests.test_utils import TestMoleculeFactory


class TestMultiStructureNEBStrategy:
    """Test MultiStructureNEBStrategy."""

    def test_strategy_metadata(self) -> None:
        """Test strategy metadata."""
        assert MultiStructureNEBStrategy.metadata.name == "path:neb"
        assert MultiStructureNEBStrategy.metadata.target == "path"
        assert MultiStructureNEBStrategy.metadata.strategy == "neb"
        assert MultiStructureNEBStrategy.metadata.requires_multiple_structures

    def test_strategy_initialization(self) -> None:
        """Test strategy initialization."""
        reactant = TestMoleculeFactory.get_water_distorted()
        product = TestMoleculeFactory.get_water_distorted()
        pos = product.get_positions()
        pos[1, 0] += 0.2
        product.set_positions(pos)

        explorer = Explorer([reactant, product], backend="mock")

        strategy = MultiStructureNEBStrategy(explorer)

        assert strategy.explorer == explorer

    def test_strategy_run_basic(self) -> None:
        """Test basic run of NEB optimization."""
        reactant = TestMoleculeFactory.get_water_distorted()
        product = TestMoleculeFactory.get_water_distorted()
        pos = product.get_positions()
        pos[1, 0] += 0.2
        product.set_positions(pos)

        explorer = Explorer([reactant, product], backend="mock")

        strategy = MultiStructureNEBStrategy(explorer)

        result = strategy.run([reactant, product], npoints=5, steps=5, fmax=0.5)

        assert "optimized_atoms" in result
        assert "strategy" in result
        assert "trajectory" in result
        assert result["strategy"] == "path:neb"
        assert isinstance(result["optimized_atoms"], list)

    def test_strategy_run_with_different_interpolation_methods(self) -> None:
        """Test run with different interpolation methods."""
        reactant = TestMoleculeFactory.get_water_distorted()
        product = TestMoleculeFactory.get_water_distorted()
        pos = product.get_positions()
        pos[1, 0] += 0.2
        product.set_positions(pos)

        explorer = Explorer([reactant, product], backend="mock")

        strategy = MultiStructureNEBStrategy(explorer)

        for method in ["linear", "geodesic", "idpp"]:
            result = strategy.run([reactant, product], npoints=5, method=method, steps=3, fmax=0.5)

            assert result["optimized_atoms"] is not None
            assert isinstance(result["optimized_atoms"], list)

    def test_strategy_run_with_spring_constant(self) -> None:
        """Test run with custom spring constant."""
        reactant = TestMoleculeFactory.get_water_distorted()
        product = TestMoleculeFactory.get_water_distorted()
        pos = product.get_positions()
        pos[1, 0] += 0.2
        product.set_positions(pos)

        explorer = Explorer([reactant, product], backend="mock")

        strategy = MultiStructureNEBStrategy(explorer)

        result = strategy.run(
            [reactant, product], npoints=5, spring_constant=10.0, steps=3, fmax=0.5
        )

        assert result["optimized_atoms"] is not None

    def test_strategy_validates_minimum_images(self) -> None:
        """Test that strategy validates minimum number of images."""
        reactant = TestMoleculeFactory.get_water_distorted()
        product = TestMoleculeFactory.get_water_distorted()
        pos = product.get_positions()
        pos[1, 0] += 0.2
        product.set_positions(pos)

        explorer = Explorer([reactant, product], backend="mock")

        strategy = MultiStructureNEBStrategy(explorer)

        # npoints=2 gives only 2 images after interpolation, need at least 3
        with pytest.raises(ValueError, match="at least 3 images"):
            strategy.run([reactant, product], npoints=2, steps=3, fmax=0.5)

    def test_strategy_validates_inputs(self) -> None:
        """Test that strategy validates inputs."""
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock")

        strategy = MultiStructureNEBStrategy(explorer)

        with pytest.raises(ValueError, match="No atoms provided"):
            strategy.run([])

    def test_strategy_filters_redundant_structures(self) -> None:
        """Test that strategy filters redundant structures."""
        reactant = TestMoleculeFactory.get_water_distorted()
        product = TestMoleculeFactory.get_water_distorted()
        pos = product.get_positions()
        pos[1, 0] += 0.2
        product.set_positions(pos)

        explorer = Explorer([reactant, product], backend="mock")

        strategy = MultiStructureNEBStrategy(explorer)

        # Capture warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")

            result = strategy.run(
                [reactant, product],
                npoints=5,
                steps=3,
                fmax=0.5,
                rmsd_threshold=0.1,
                energy_threshold=0.001,
            )

            # Should have optimized path
            assert result["optimized_atoms"] is not None

    def test_strategy_handles_calculator_attachment_failure(self) -> None:
        """Test strategy handles calculator attachment failures."""
        reactant = TestMoleculeFactory.get_water_distorted()
        product = TestMoleculeFactory.get_water_distorted()
        pos = product.get_positions()
        pos[1, 0] += 0.2
        product.set_positions(pos)

        explorer = Explorer([reactant, product], backend="mock")

        strategy = MultiStructureNEBStrategy(explorer)

        # Mock PathManager.attach_calculators to leave some images without calculators
        with patch("qme.strategies.neb.PathManager.attach_calculators") as mock_attach:
            # Make attach_calculators not attach calculators properly
            def side_effect(explorer, path):
                # Don't attach calculators - leave them as None
                pass

            mock_attach.side_effect = side_effect

            # Should raise error when calculator attachment fails
            with pytest.raises(RuntimeError, match="Failed to attach calculators"):
                strategy.run([reactant, product], npoints=5, steps=3, fmax=0.5)


class TestMultiStructureCINEBStrategy:
    """Test MultiStructureCINEBStrategy."""

    def test_strategy_metadata(self) -> None:
        """Test strategy metadata."""
        assert MultiStructureCINEBStrategy.metadata.name == "path:cineb"
        assert MultiStructureCINEBStrategy.metadata.target == "path"
        assert MultiStructureCINEBStrategy.metadata.strategy == "cineb"
        assert MultiStructureCINEBStrategy.metadata.requires_multiple_structures

    def test_strategy_initialization(self) -> None:
        """Test strategy initialization."""
        reactant = TestMoleculeFactory.get_water_distorted()
        product = TestMoleculeFactory.get_water_distorted()
        pos = product.get_positions()
        pos[1, 0] += 0.2
        product.set_positions(pos)

        explorer = Explorer([reactant, product], backend="mock")

        strategy = MultiStructureCINEBStrategy(explorer)

        assert strategy.explorer == explorer

    def test_strategy_run_basic(self) -> None:
        """Test basic run of CI-NEB optimization."""
        reactant = TestMoleculeFactory.get_water_distorted()
        product = TestMoleculeFactory.get_water_distorted()
        pos = product.get_positions()
        pos[1, 0] += 0.2
        product.set_positions(pos)

        explorer = Explorer([reactant, product], backend="mock")

        strategy = MultiStructureCINEBStrategy(explorer)

        result = strategy.run([reactant, product], npoints=5, steps=5, fmax=0.5)

        assert "optimized_atoms" in result
        assert "strategy" in result
        assert "trajectory" in result
        assert result["strategy"] == "path:cineb"
        assert isinstance(result["optimized_atoms"], list)

    def test_strategy_run_with_climb_false(self) -> None:
        """Test run with climb=False (should still use CI-NEB but with climb disabled)."""
        reactant = TestMoleculeFactory.get_water_distorted()
        product = TestMoleculeFactory.get_water_distorted()
        pos = product.get_positions()
        pos[1, 0] += 0.2
        product.set_positions(pos)

        explorer = Explorer([reactant, product], backend="mock")

        strategy = MultiStructureCINEBStrategy(explorer)

        result = strategy.run([reactant, product], npoints=5, climb=False, steps=3, fmax=0.5)

        assert result["optimized_atoms"] is not None

    def test_strategy_run_with_different_interpolation_methods(self) -> None:
        """Test run with different interpolation methods."""
        reactant = TestMoleculeFactory.get_water_distorted()
        product = TestMoleculeFactory.get_water_distorted()
        pos = product.get_positions()
        pos[1, 0] += 0.2
        product.set_positions(pos)

        explorer = Explorer([reactant, product], backend="mock")

        strategy = MultiStructureCINEBStrategy(explorer)

        for method in ["linear", "geodesic", "idpp"]:
            result = strategy.run([reactant, product], npoints=5, method=method, steps=3, fmax=0.5)

            assert result["optimized_atoms"] is not None
            assert isinstance(result["optimized_atoms"], list)

    def test_strategy_run_with_spring_constant(self) -> None:
        """Test run with custom spring constant."""
        reactant = TestMoleculeFactory.get_water_distorted()
        product = TestMoleculeFactory.get_water_distorted()
        pos = product.get_positions()
        pos[1, 0] += 0.2
        product.set_positions(pos)

        explorer = Explorer([reactant, product], backend="mock")

        strategy = MultiStructureCINEBStrategy(explorer)

        result = strategy.run(
            [reactant, product], npoints=5, spring_constant=10.0, steps=3, fmax=0.5
        )

        assert result["optimized_atoms"] is not None

    def test_strategy_validates_minimum_images(self) -> None:
        """Test that strategy validates minimum number of images."""
        reactant = TestMoleculeFactory.get_water_distorted()
        product = TestMoleculeFactory.get_water_distorted()
        pos = product.get_positions()
        pos[1, 0] += 0.2
        product.set_positions(pos)

        explorer = Explorer([reactant, product], backend="mock")

        strategy = MultiStructureCINEBStrategy(explorer)

        # npoints=2 gives only 2 images after interpolation, need at least 3
        with pytest.raises(ValueError, match="at least 3 images"):
            strategy.run([reactant, product], npoints=2, steps=3, fmax=0.5)

    def test_strategy_validates_inputs(self) -> None:
        """Test that strategy validates inputs."""
        atoms = TestMoleculeFactory.get_water_distorted()
        explorer = Explorer(atoms, backend="mock")

        strategy = MultiStructureCINEBStrategy(explorer)

        with pytest.raises(ValueError, match="No atoms provided"):
            strategy.run([])

    def test_strategy_filters_redundant_structures(self) -> None:
        """Test that strategy filters redundant structures."""
        reactant = TestMoleculeFactory.get_water_distorted()
        product = TestMoleculeFactory.get_water_distorted()
        pos = product.get_positions()
        pos[1, 0] += 0.2
        product.set_positions(pos)

        explorer = Explorer([reactant, product], backend="mock")

        strategy = MultiStructureCINEBStrategy(explorer)

        # Capture warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")

            result = strategy.run(
                [reactant, product],
                npoints=5,
                steps=3,
                fmax=0.5,
                rmsd_threshold=0.1,
                energy_threshold=0.001,
            )

            # Should have optimized path
            assert result["optimized_atoms"] is not None

    def test_strategy_handles_calculator_attachment_failure(self) -> None:
        """Test strategy handles calculator attachment failures."""
        reactant = TestMoleculeFactory.get_water_distorted()
        product = TestMoleculeFactory.get_water_distorted()
        pos = product.get_positions()
        pos[1, 0] += 0.2
        product.set_positions(pos)

        explorer = Explorer([reactant, product], backend="mock")

        strategy = MultiStructureCINEBStrategy(explorer)

        # Mock PathManager.attach_calculators to leave some images without calculators
        with patch("qme.strategies.cineb.PathManager.attach_calculators") as mock_attach:
            # Make attach_calculators not attach calculators properly
            def side_effect(explorer, path):
                # Don't attach calculators
                pass

            mock_attach.side_effect = side_effect

            # Should raise error when calculator attachment fails
            with pytest.raises(RuntimeError, match="Failed to attach calculators"):
                strategy.run([reactant, product], npoints=5, steps=3, fmax=0.5)
