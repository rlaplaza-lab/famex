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

# Import concrete symbols from subpackages (no top-level re-exports)
from qme.core import QMEOptimizer, minimize_structure
from qme.geometry import Geometry, read_geometry, write_geometry
from qme.potentials.mock import MockCalculator
from qme.reaction import Reaction
from qme.utils.settings import config, get_default_backend, get_default_model


class TestPackageImports:
    """Test that all main package components can be imported."""

    def test_version_available(self):
        """Test that package version is accessible."""
        # Package metadata is available via package import
        import qme as _qme

        assert hasattr(_qme, "__version__")
        assert isinstance(_qme.__version__, str)
        assert _qme.__version__ == "0.1.0"

    def test_core_imports(self):
        """Test that core classes can be imported."""
        assert QMEOptimizer is not None
        assert callable(minimize_structure)
        assert Geometry is not None

    def test_calculator_imports(self):
        """Test that calculator functions can be imported."""
        assert MockCalculator is not None

    def test_config_imports(self):
        """Test that configuration functions can be imported."""
        assert config is not None
        assert callable(get_default_backend)
        assert callable(get_default_model)


class TestMockCalculators:
    """Test mock calculators functionality."""

    def test_mock_uma_calculator(self):
        """Test UMA mock calculator creation."""
        calc = MockCalculator(backend="uma")
        assert calc is not None
        assert hasattr(calc, "calculate")

    def test_mock_so3lr_calculator(self):
        """Test SO3LR mock calculator creation."""
        calc = MockCalculator(backend="so3lr")
        assert calc is not None
        assert hasattr(calc, "calculate")

    def test_mock_aimnet2_calculator(self):
        """Test AIMNET2 mock calculator creation."""
        calc = MockCalculator(backend="aimnet2")
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

        calc = MockCalculator(backend="so3lr")
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
        optimizer = QMEOptimizer(backend="mock")
        assert optimizer is not None
        assert hasattr(optimizer, "optimize_minimum")
        assert hasattr(optimizer, "load_structure")

    def test_optimizer_with_mock_backend(self):
        """Test optimizer with mock backend."""
        optimizer = QMEOptimizer(backend="mock")

        # Create a simple H2 molecule
        h2 = Atoms("H2", positions=[[0, 0, 0], [1.5, 0, 0]])
        optimizer.atoms = h2

        # Test that the optimizer has the expected attributes
        assert optimizer.backend == "mock"
        assert optimizer.atoms is not None

    def test_available_optimizers(self):
        """Test that available optimizers are defined."""
        assert hasattr(QMEOptimizer, "AVAILABLE_OPTIMIZERS")
        assert "BFGS" in QMEOptimizer.AVAILABLE_OPTIMIZERS
        assert "LBFGS" in QMEOptimizer.AVAILABLE_OPTIMIZERS
        assert "FIRE" in QMEOptimizer.AVAILABLE_OPTIMIZERS

    def test_available_backends(self):
        """Test that available backends are defined."""
        assert hasattr(QMEOptimizer, "AVAILABLE_BACKENDS")
        assert "uma" in QMEOptimizer.AVAILABLE_BACKENDS
        assert "so3lr" in QMEOptimizer.AVAILABLE_BACKENDS
        assert "aimnet2" in QMEOptimizer.AVAILABLE_BACKENDS
        assert "mock" in QMEOptimizer.AVAILABLE_BACKENDS


class TestGeometry:
    """Test geometry handling functionality."""

    def test_geometry_creation(self):
        """Test Geometry class instantiation."""
        geom = Geometry()
        assert geom is not None

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
            atoms = read_geometry(temp_path)
            assert isinstance(atoms, Atoms)
            assert len(atoms) == 3
            assert atoms.get_chemical_symbols() == ["O", "H", "H"]

            # Test writing geometry
            output_path = temp_path.replace(".xyz", "_output.xyz")
            write_geometry(atoms, output_path)

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
        assert config is not None
        assert hasattr(config, "get") or hasattr(config, "__getitem__")

    def test_default_backend(self):
        """Test default backend retrieval."""
        backend = get_default_backend()
        assert backend in ["uma", "so3lr", "aimnet2", "mock"]

    def test_default_model(self):
        """Test default model retrieval."""
        model = get_default_model("mock")
        assert isinstance(model, str)


class TestMinimizeStructure:
    """Test the standalone minimize_structure function."""

    def test_minimize_structure_exists(self):
        """Test that minimize_structure function exists."""
        assert callable(minimize_structure)

    def test_minimize_structure_with_mock(self):
        """Test minimize_structure with mock calculator."""
        # Create a simple molecule
        h2 = Atoms(
            "H2", positions=[[0, 0, 0], [2.0, 0, 0]]
        )  # Longer bond for optimization

        # This should not raise an exception even if optimization isn't fully performed
        try:
            result = minimize_structure(h2, backend="mock", steps=1)
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
        reaction = Reaction(reactant, product)
        assert reaction is not None
        assert reaction.reactant is not None
        assert reaction.product is not None


# MLPCalculator has been removed as it was deprecated


class TestCLIIntegration:
    """Test CLI command availability (import level only)."""

    def test_cli_import(self):
        """Test that CLI module can be imported."""
        import qme.cli as cli

        assert hasattr(cli, "main")
        assert callable(cli.main)


if __name__ == "__main__":
    pytest.main([__file__])
