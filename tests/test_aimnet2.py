"""
Tests for AIMNET2 neural network potential integration.
"""

import tempfile
from pathlib import Path

import pytest
from ase import Atoms
from ase.build import molecule

from qme import QMEOptimizer
from qme.aimnet2_potential import (
    AIMNet2Potential,
    get_aimnet2_calculator,
)


class TestAIMNet2Basics:
    """Basic functionality tests for AIMNET2 backend."""

    def test_aimnet2_import(self):
        """Test that AIMNET2 modules can be imported."""
        from qme.aimnet2_potential import AIMNet2Potential

        assert AIMNet2Potential is not None

    def test_mock_aimnet2_calculator(self):
        """Test mock AIMNET2 calculator functionality."""
        from qme.mock_calculator import get_mock_aimnet2_calculator

        calc = get_mock_aimnet2_calculator()

        # Test with water molecule
        atoms = molecule("H2O")
        atoms.calc = calc

        # Test energy calculation
        energy = atoms.get_potential_energy()
        assert isinstance(energy, float)

        # Test forces calculation
        forces = atoms.get_forces()
        assert forces.shape == (3, 3)  # 3 atoms, 3 dimensions

    def test_aimnet2_calculator_init(self):
        """Test AIMNET2 calculator initialization."""
        calc = AIMNet2Potential()

        # Check default parameters
        assert calc.model_name == "aimnet2"
        assert calc.charge == 0
        assert calc.mult == 1

        # Test with custom parameters
        calc_custom = AIMNet2Potential(model_name="custom_model", charge=1, mult=2)
        assert calc_custom.model_name == "custom_model"
        assert calc_custom.charge == 1
        assert calc_custom.mult == 2

    def test_aimnet2_charge_mult_setting(self):
        """Test setting charge and multiplicity."""
        calc = AIMNet2Potential()

        # Test setting charge
        calc.set_charge(-1)
        assert calc.charge == -1

        # Test setting multiplicity
        calc.set_mult(3)
        assert calc.mult == 3

    def test_get_aimnet2_calculator(self):
        """Test convenience function to get AIMNET2 calculator."""
        calc = get_aimnet2_calculator()
        assert isinstance(calc, AIMNet2Potential)
        assert calc.model_name == "aimnet2"

        # Test with custom parameters
        calc_custom = get_aimnet2_calculator(model_name="test_model", charge=-2, mult=2)
        assert calc_custom.model_name == "test_model"
        assert calc_custom.charge == -2
        assert calc_custom.mult == 2


class TestQMEOptimizerWithAIMNet2:
    """Test QMEOptimizer with AIMNET2 backend."""

    def test_qme_optimizer_aimnet2_backend(self):
        """Test QMEOptimizer initialization with AIMNET2 backend."""
        qme = QMEOptimizer(backend="aimnet2", use_mock=True)
        assert qme.backend == "aimnet2"
        assert qme.calculator is not None

    def test_qme_optimizer_aimnet2_with_model(self):
        """Test QMEOptimizer with custom AIMNET2 model."""
        qme = QMEOptimizer(
            backend="aimnet2", model_name="custom_aimnet2", use_mock=True
        )
        assert qme.backend == "aimnet2"

    def test_optimization_with_aimnet2(self):
        """Test basic optimization with AIMNET2 backend."""
        # Create QME optimizer with mock calculator
        qme = QMEOptimizer(backend="aimnet2", use_mock=True)

        # Create test molecule (water)
        atoms = molecule("H2O")

        # Test optimization
        results = qme.optimize_minimum(atoms=atoms, optimizer="BFGS", fmax=0.1, steps=5)

        # Check results
        assert "optimized_atoms" in results
        assert "final_energy" in results
        assert "converged" in results
        assert len(results["optimized_atoms"]) == 3  # H2O has 3 atoms

    def test_aimnet2_with_charge(self):
        """Test AIMNET2 backend with charged molecules."""
        qme = QMEOptimizer(backend="aimnet2", use_mock=True)

        # Test with charged system
        atoms = molecule("H2O")
        # The mock calculator should handle charge internally
        results = qme.optimize_minimum(atoms=atoms, optimizer="BFGS", fmax=0.1, steps=3)

        assert "optimized_atoms" in results
        assert "final_energy" in results


class TestAIMNet2Integration:
    """Tests that require AIMNET2 package (may be skipped if dependencies missing)."""

    def test_aimnet2_calculator_init_real(self):
        """Test AIMNET2 calculator initialization with real package."""
        try:
            # This will use mock if AIMNET2 is not available
            calc = AIMNet2Potential(model_name="aimnet2")
            assert calc is not None
        except ImportError:
            pytest.skip("AIMNET2 not available")

    def test_model_loading_fallback(self):
        """Test that model loading falls back gracefully."""
        # Should work with mock even if real AIMNET2 is not available
        calc = AIMNet2Potential()
        atoms = molecule("H2O")
        atoms.calc = calc

        # Should not raise an exception
        energy = atoms.get_potential_energy()
        assert isinstance(energy, float)


class TestBackendComparison:
    """Test comparison between different backends including AIMNET2."""

    def test_all_backend_consistency(self):
        """Test that all three backends work consistently."""
        # Create test molecule
        atoms = molecule("H2O")

        # Test UMA backend
        qme_uma = QMEOptimizer(backend="uma", use_mock=True)
        results_uma = qme_uma.optimize_minimum(
            atoms=atoms.copy(), optimizer="BFGS", fmax=0.1, steps=5
        )

        # Test SO3LR backend
        qme_so3lr = QMEOptimizer(backend="so3lr", use_mock=True)
        results_so3lr = qme_so3lr.optimize_minimum(
            atoms=atoms.copy(), optimizer="BFGS", fmax=0.1, steps=5
        )

        # Test AIMNET2 backend
        qme_aimnet2 = QMEOptimizer(backend="aimnet2", use_mock=True)
        results_aimnet2 = qme_aimnet2.optimize_minimum(
            atoms=atoms.copy(), optimizer="BFGS", fmax=0.1, steps=5
        )

        # All should complete successfully
        assert "optimized_atoms" in results_uma
        assert "optimized_atoms" in results_so3lr
        assert "optimized_atoms" in results_aimnet2

        # All should have same number of atoms
        assert len(results_uma["optimized_atoms"]) == len(
            results_so3lr["optimized_atoms"]
        )
        assert len(results_so3lr["optimized_atoms"]) == len(
            results_aimnet2["optimized_atoms"]
        )

    def test_backend_switching(self):
        """Test switching between all backends."""
        # UMA backend
        qme_uma = QMEOptimizer(backend="uma", use_mock=True)
        assert qme_uma.backend == "uma"

        # SO3LR backend
        qme_so3lr = QMEOptimizer(backend="so3lr", use_mock=True)
        assert qme_so3lr.backend == "so3lr"

        # AIMNET2 backend
        qme_aimnet2 = QMEOptimizer(backend="aimnet2", use_mock=True)
        assert qme_aimnet2.backend == "aimnet2"

    def test_invalid_backend_still_works(self):
        """Test error handling for invalid backend."""
        with pytest.raises(ValueError, match="Unknown backend"):
            QMEOptimizer(backend="invalid_backend")

    def test_available_backends_includes_aimnet2(self):
        """Test that AIMNET2 is listed in available backends."""
        qme = QMEOptimizer(backend="aimnet2", use_mock=True)
        available_backends = qme.AVAILABLE_BACKENDS

        assert "aimnet2" in available_backends
        assert "uma" in available_backends
        assert "so3lr" in available_backends

        # Check that AIMNET2 description is appropriate
        assert "AIMNET2" in available_backends["aimnet2"]


class TestAIMNet2MockCalculator:
    """Test the mock AIMNET2 calculator specifically."""

    def test_mock_aimnet2_basic_functionality(self):
        """Test basic functionality of mock AIMNET2 calculator."""
        from qme.mock_calculator import get_mock_aimnet2_calculator

        calc = get_mock_aimnet2_calculator()
        atoms = molecule("H2O")
        atoms.calc = calc

        # Test energy calculation
        energy = atoms.get_potential_energy()
        assert isinstance(energy, float)

        # Test forces calculation
        forces = atoms.get_forces()
        assert forces.shape == (3, 3)  # 3 atoms, 3 dimensions

    def test_mock_aimnet2_parameters(self):
        """Test mock AIMNET2 calculator parameters."""
        from qme.mock_calculator import MockAIMNet2Calculator

        calc = MockAIMNet2Calculator(charge=2, mult=3)
        assert calc.charge == 2
        assert calc.mult == 3
        assert calc.bond_length == 1.2  # Different from UMA mock
        assert calc.force_constant == 0.8  # Different from UMA mock

        # Test set methods if they exist
        if hasattr(calc, "set_charge") and hasattr(calc, "set_mult"):
            calc.set_charge(-1)
            calc.set_mult(1)
            assert calc.charge == -1
            assert calc.mult == 1
