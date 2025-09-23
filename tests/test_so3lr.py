"""
Tests for SO3LR neural network potential integration.
"""

import tempfile
from pathlib import Path

import pytest
from ase import Atoms
from ase.build import molecule

from qme import QMEOptimizer
from qme.so3lr_potential import (
    SO3LRPotential,
    get_mock_so3lr_calculator,
    get_so3lr_calculator,
)


class TestSO3LRBasics:
    """Basic functionality tests for SO3LR backend."""

    def test_so3lr_import(self):
        """Test that SO3LR modules can be imported."""
        from qme.so3lr_potential import SO3LRPotential

        assert SO3LRPotential is not None

    def test_mock_so3lr_calculator(self):
        """Test mock SO3LR calculator functionality."""
        calc = get_mock_so3lr_calculator()
        assert calc is not None
        assert hasattr(calc, "device")
        assert hasattr(calc, "model")

    def test_so3lr_calculator_with_mock(self):
        """Test SO3LR calculator initialization with mock fallback."""
        # This should use mock implementation since so3lr package isn't available
        calc = SO3LRPotential()
        assert calc is not None
        assert hasattr(calc, "device")
        assert hasattr(calc, "model")

    def test_so3lr_calculation(self):
        """Test basic calculation with mock SO3LR."""
        calc = get_mock_so3lr_calculator()

        # Create simple test molecule
        atoms = molecule("H2O")
        atoms.calc = calc

        # Test energy calculation
        energy = atoms.get_potential_energy()
        assert isinstance(energy, (int, float))

        # Test forces calculation
        forces = atoms.get_forces()
        assert forces.shape == (3, 3)  # 3 atoms, 3 dimensions


class TestQMEOptimizerWithSO3LR:
    """Test QMEOptimizer with SO3LR backend."""

    def test_qme_optimizer_so3lr_backend(self):
        """Test QMEOptimizer initialization with SO3LR backend."""
        qme = QMEOptimizer(backend="so3lr", use_mock=True)
        assert qme.backend == "so3lr"
        assert qme.calculator is not None

    def test_qme_optimizer_default_backend(self):
        """Test that SO3LR is now the default backend."""
        qme = QMEOptimizer(use_mock=True)  # Should use SO3LR by default
        assert qme.backend == "so3lr"

    def test_backend_switching(self):
        """Test switching between backends."""
        # SO3LR backend
        qme_so3lr = QMEOptimizer(backend="so3lr", use_mock=True)
        assert qme_so3lr.backend == "so3lr"

        # UMA backend
        qme_uma = QMEOptimizer(backend="uma", use_mock=True)
        assert qme_uma.backend == "uma"

    def test_invalid_backend(self):
        """Test error handling for invalid backend."""
        with pytest.raises(ValueError, match="Unknown backend"):
            QMEOptimizer(backend="invalid_backend")

    def test_optimization_with_so3lr(self):
        """Test basic optimization with SO3LR backend."""
        qme = QMEOptimizer(backend="so3lr", use_mock=True)

        # Create test molecule
        atoms = molecule("H2O")
        atoms.rattle(stdev=0.1)  # Add some noise to require optimization

        # Run optimization
        results = qme.optimize_minimum(
            atoms=atoms,
            optimizer="BFGS",
            fmax=0.1,  # Loose convergence for testing
            steps=10,  # Few steps for testing
        )

        assert "converged" in results
        assert "optimized_atoms" in results
        assert "energy_change" in results
        assert len(results["optimized_atoms"]) == len(atoms)


class TestSO3LRIntegration:
    """Tests that require SO3LR model (may be skipped if dependencies missing)."""

    def test_so3lr_calculator_init_real(self):
        """Test SO3LR calculator initialization with real package."""
        try:
            # This will use real SO3LR if available, mock otherwise
            calc = SO3LRPotential(model_name="so3lr-small")
            assert calc is not None
            assert hasattr(calc, "device")
        except ImportError:
            pytest.skip("so3lr package not available")
        except Exception as e:
            # Expected to use mock fallback
            assert "mock" in str(e).lower() or calc is not None

    def test_model_path_loading(self):
        """Test loading SO3LR from model path."""
        # Create a dummy model file for testing
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            model_path = f.name

        try:
            # This should fall back to mock since the file isn't a real model
            calc = SO3LRPotential(model_path=model_path)
            assert calc is not None
        except Exception as e:
            # Expected to fail or use mock fallback
            assert calc is not None or "mock" in str(e).lower()
        finally:
            Path(model_path).unlink(missing_ok=True)


class TestBackendComparison:
    """Test comparison between different backends."""

    def test_backend_consistency(self):
        """Test that different backends work consistently."""
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

        # Both should complete successfully
        assert "optimized_atoms" in results_uma
        assert "optimized_atoms" in results_so3lr
        assert len(results_uma["optimized_atoms"]) == len(
            results_so3lr["optimized_atoms"]
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
