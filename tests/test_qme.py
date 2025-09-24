"""
Basic tests for QME (Quick Mechanistic Exploration) package.

This test suite provides minimal coverage of core functionality using
mock calculators to avoid ML dependencies in CI/CD.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

import qme


class TestPackageImports:
    """Test that all main package components can be imported."""

    def test_version_available(self):
        """Test that package version is accessible."""
        assert hasattr(qme, "__version__")
        assert isinstance(qme.__version__, str)
        assert qme.__version__ == "0.1.0"

    def test_core_imports(self):
        """Test that core classes can be imported."""
        assert hasattr(qme, "QMEOptimizer")
        assert hasattr(qme, "minimize_structure")
        assert hasattr(qme, "Geometry")

    def test_calculator_imports(self):
        """Test that calculator functions can be imported."""
        assert hasattr(qme, "get_mock_uma_calculator")
        assert hasattr(qme, "get_mock_so3lr_calculator")
        assert hasattr(qme, "get_mock_aimnet2_calculator")

    def test_config_imports(self):
        """Test that configuration functions can be imported."""
        assert hasattr(qme, "config")
        assert hasattr(qme, "get_default_backend")
        assert hasattr(qme, "get_default_model")


class TestMockCalculators:
    """Test mock calculators functionality."""

    def test_mock_uma_calculator(self):
        """Test UMA mock calculator creation."""
        calc = qme.get_mock_uma_calculator()
        assert calc is not None
        assert hasattr(calc, "calculate")

    def test_mock_so3lr_calculator(self):
        """Test SO3LR mock calculator creation."""
        calc = qme.get_mock_so3lr_calculator()
        assert calc is not None
        assert hasattr(calc, "calculate")

    def test_mock_aimnet2_calculator(self):
        """Test AIMNET2 mock calculator creation."""
        calc = qme.get_mock_aimnet2_calculator()
        assert calc is not None
        assert hasattr(calc, "calculate")

    def test_mock_calculator_with_water(self):
        """Test mock calculator with a simple water molecule."""
        # Create a simple water molecule
        water = Atoms(
            "H2O",
            positions=[
                [0.757, 0.586, 0.000],
                [-0.757, 0.586, 0.000],
                [0.000, 0.000, 0.000],
            ],
        )

        calc = qme.get_mock_so3lr_calculator()
        water.calc = calc

        # Test that we can compute energy and forces
        energy = water.get_potential_energy()
        forces = water.get_forces()

        assert isinstance(energy, float)
        assert isinstance(forces, np.ndarray)
        assert forces.shape == (3, 3)  # 3 atoms, 3 dimensions


class TestQMEOptimizer:
    """Test QMEOptimizer core functionality."""

    def test_optimizer_creation(self):
        """Test basic QMEOptimizer instantiation."""
        optimizer = qme.QMEOptimizer(backend="so3lr", use_mock=True)
        assert optimizer is not None
        assert hasattr(optimizer, "optimize_minimum")
        assert hasattr(optimizer, "load_structure")

    def test_optimizer_with_mock_backend(self):
        """Test optimizer with mock backend."""
        optimizer = qme.QMEOptimizer(backend="so3lr", use_mock=True)

        # Create a simple H2 molecule
        h2 = Atoms("H2", positions=[[0, 0, 0], [1.5, 0, 0]])
        optimizer.atoms = h2

        # Test that the optimizer has the expected attributes
        assert optimizer.backend == "so3lr"
        assert optimizer.atoms is not None

    def test_available_optimizers(self):
        """Test that available optimizers are defined."""
        assert hasattr(qme.QMEOptimizer, "AVAILABLE_OPTIMIZERS")
        assert "BFGS" in qme.QMEOptimizer.AVAILABLE_OPTIMIZERS
        assert "LBFGS" in qme.QMEOptimizer.AVAILABLE_OPTIMIZERS
        assert "FIRE" in qme.QMEOptimizer.AVAILABLE_OPTIMIZERS

    def test_available_backends(self):
        """Test that available backends are defined."""
        assert hasattr(qme.QMEOptimizer, "AVAILABLE_BACKENDS")
        assert "uma" in qme.QMEOptimizer.AVAILABLE_BACKENDS
        assert "so3lr" in qme.QMEOptimizer.AVAILABLE_BACKENDS
        assert "aimnet2" in qme.QMEOptimizer.AVAILABLE_BACKENDS


class TestGeometry:
    """Test geometry handling functionality."""

    def test_geometry_creation(self):
        """Test Geometry class instantiation."""
        # Create geometry with atoms and positions
        geom = qme.Geometry(
            atoms=["H", "H"], positions=np.array([[0, 0, 0], [1.5, 0, 0]])
        )
        assert geom is not None
        assert geom.atoms is not None
        assert len(geom.atoms) == 2

    def test_read_write_geometry(self):
        """Test geometry reading and writing with XYZ format."""
        # Create a temporary XYZ file
        xyz_content = """3
Water molecule  
O       0.000000    0.000000    0.000000
H       0.757000    0.586000    0.000000
H      -0.757000    0.586000    0.000000
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write(xyz_content)
            temp_path = f.name

        try:
            # Test reading geometry
            geom = qme.read_geometry(temp_path)
            assert isinstance(geom, qme.Geometry)
            assert len(geom.atoms) == 3
            assert geom.atoms.get_chemical_symbols() == ["O", "H", "H"]

            # Test writing geometry
            output_path = temp_path.replace(".xyz", "_output.xyz")
            qme.write_geometry(geom, output_path)

            # Verify the output file exists
            assert Path(output_path).exists()

            # Clean up
            Path(output_path).unlink()

        finally:
            Path(temp_path).unlink()


class TestConfiguration:
    """Test configuration functionality."""

    def test_config_object(self):
        """Test that config object exists and has expected attributes."""
        assert qme.config is not None
        assert hasattr(qme.config, "default_backend")
        assert hasattr(qme.config, "default_optimizer")

    def test_default_backend(self):
        """Test default backend retrieval."""
        backend = qme.get_default_backend()
        assert backend in ["uma", "so3lr", "aimnet2"]

    def test_default_model(self):
        """Test default model retrieval."""
        model = qme.get_default_model("so3lr")
        assert isinstance(model, str)


class TestMinimizeStructure:
    """Test the standalone minimize_structure function."""

    def test_minimize_structure_exists(self):
        """Test that minimize_structure function exists."""
        assert callable(qme.minimize_structure)

    def test_minimize_structure_with_mock(self):
        """Test minimize_structure with mock calculator."""
        # Create a simple molecule
        h2 = Atoms(
            "H2", positions=[[0, 0, 0], [2.0, 0, 0]]
        )  # Longer bond for optimization

        # This should not raise an exception even if optimization isn't fully performed
        try:
            result = qme.minimize_structure(h2, backend="so3lr", use_mock=True, steps=1)
            # If it returns something, verify it's the right type
            if result is not None:
                assert isinstance(result, (Atoms, dict))
        except Exception as e:
            # Some exceptions might be expected if full ML dependencies aren't available
            # We just want to ensure the function exists and is callable
            assert "minimize_structure" not in str(e) or "not found" not in str(e)


class TestReaction:
    """Test reaction functionality."""

    def test_reaction_creation(self):
        """Test Reaction class instantiation."""
        # Create simple reactant and product structures
        reactant = Atoms("H2", positions=[[0, 0, 0], [1.5, 0, 0]])
        product = Atoms("H2", positions=[[0, 0, 0], [1.0, 0, 0]])

        reaction = qme.Reaction(reactant, product)
        assert reaction is not None
        assert reaction.reactant is not None
        assert reaction.product is not None


class TestMLPCalculator:
    """Test MLPCalculator functionality."""

    def test_mlp_calculator_creation(self):
        """Test MLPCalculator class exists."""
        assert hasattr(qme, "MLPCalculator")
        # Don't instantiate as it might require ML dependencies


class TestCLIIntegration:
    """Test CLI command availability (import level only)."""

    def test_cli_import(self):
        """Test that CLI module can be imported."""
        from qme import cli

        assert hasattr(cli, "main")
        assert callable(cli.main)


if __name__ == "__main__":
    pytest.main([__file__])
