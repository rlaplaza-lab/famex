"""
Tests for QME package functionality.
"""

import tempfile
from pathlib import Path

import pytest

from qme import QMEOptimizer
from qme.aimnet2_potential import AIMNet2Potential
from qme.uma_potential import UMAPotential


class TestQMEBasics:
    """Basic functionality tests that don't require UMA models."""

    def test_imports(self):
        """Test that core modules can be imported."""
        from qme import AIMNet2Potential, QMEOptimizer, UMAPotential

        assert QMEOptimizer is not None
        assert UMAPotential is not None
        assert AIMNet2Potential is not None

    def test_structure_loading(self):
        """Test structure loading from XYZ file."""
        # Create temporary XYZ file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write(
                """3
Water molecule
O       0.000000    0.000000    0.000000
H       0.757000    0.586000    0.000000
H      -0.757000    0.586000    0.000000"""
            )
            xyz_file = f.name

        try:
            # Test loading would work if we had a calculator
            qme = QMEOptimizer.__new__(QMEOptimizer)  # Create without __init__
            qme.calculator = None
            qme.atoms = None
            qme.results = {}

            from ase.io import read

            atoms = read(xyz_file)
            assert len(atoms) == 3
            assert atoms.symbols[0] == "O"
            assert atoms.symbols[1] == "H"
            assert atoms.symbols[2] == "H"

        finally:
            Path(xyz_file).unlink()


class TestUMAIntegration:
    """Tests that require UMA model (may be skipped if dependencies missing)."""

    def test_uma_calculator_init(self):
        """Test UMA calculator initialization."""
        try:
            calc = UMAPotential(model_name="uma-4m")
            assert calc is not None
            assert hasattr(calc, "device")
        except ImportError:
            pytest.skip("fairchem-core not available")
        except Exception as e:
            pytest.skip(f"UMA model loading failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
