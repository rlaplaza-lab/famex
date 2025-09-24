"""
Integration tests for all backends working together.
"""

import pytest
from ase.build import molecule

from qme import QMEOptimizer


class TestAllBackendsIntegration:
    """Test that all backends work consistently together."""

    @pytest.mark.parametrize("backend", ["so3lr", "uma", "aimnet2"])
    def test_backend_initialization(self, backend):
        """Test that all backends initialize correctly."""
        qme = QMEOptimizer(backend=backend, use_mock=True)
        assert qme.backend == backend
        assert qme.calculator is not None

    @pytest.mark.parametrize("backend", ["so3lr", "uma", "aimnet2"])
    def test_backend_optimization(self, backend):
        """Test that all backends can perform optimization."""
        qme = QMEOptimizer(backend=backend, use_mock=True)

        # Create test molecule
        atoms = molecule("H2O")

        # Run optimization
        results = qme.optimize_minimum(atoms=atoms, optimizer="BFGS", fmax=0.1, steps=5)

        # Check results
        assert "optimized_atoms" in results
        assert "final_energy" in results
        assert "converged" in results
        assert len(results["optimized_atoms"]) == 3  # H2O has 3 atoms

    def test_all_backends_available(self):
        """Test that all three backends are available."""
        qme = QMEOptimizer(backend="so3lr", use_mock=True)
        available = list(qme.AVAILABLE_BACKENDS.keys())

        assert "so3lr" in available
        assert "uma" in available
        assert "aimnet2" in available
        assert len(available) >= 3

    def test_backend_descriptions(self):
        """Test that all backends have proper descriptions."""
        qme = QMEOptimizer(backend="so3lr", use_mock=True)
        backends = qme.AVAILABLE_BACKENDS

        assert "SO3LR" in backends["so3lr"]
        assert "UMA" in backends["uma"]
        assert "AIMNET2" in backends["aimnet2"]

    def test_backend_switching(self):
        """Test switching between all backends works."""
        backends = ["so3lr", "uma", "aimnet2"]

        for backend in backends:
            qme = QMEOptimizer(backend=backend, use_mock=True)
            assert qme.backend == backend

            # Quick test that calculator works
            atoms = molecule("H2")
            results = qme.optimize_minimum(atoms=atoms, steps=1)
            assert "optimized_atoms" in results

    def test_invalid_backend_error(self):
        """Test that invalid backend raises proper error."""
        with pytest.raises(ValueError) as excinfo:
            QMEOptimizer(backend="nonexistent_backend")

        error_msg = str(excinfo.value)
        assert "Unknown backend" in error_msg
        assert "nonexistent_backend" in error_msg
        # Should mention all available backends
        assert "so3lr" in error_msg
        assert "uma" in error_msg
        assert "aimnet2" in error_msg

    @pytest.mark.parametrize("backend", ["so3lr", "uma", "aimnet2"])
    def test_backend_energy_calculation(self, backend):
        """Test that all backends can calculate energies."""
        qme = QMEOptimizer(backend=backend, use_mock=True)

        atoms = molecule("H2")
        atoms.calc = qme.calculator

        energy = atoms.get_potential_energy()
        assert isinstance(energy, float)

        forces = atoms.get_forces()
        assert forces.shape == (2, 3)  # 2 atoms, 3 dimensions

    @pytest.mark.parametrize("backend", ["so3lr", "uma", "aimnet2"])
    def test_backend_with_constraints(self, backend):
        """Test that all backends work with geometric constraints."""
        qme = QMEOptimizer(backend=backend, use_mock=True)

        # Create water molecule
        atoms = molecule("H2O")

        # Add constraint (fix one atom)
        from ase.constraints import FixAtoms

        constraint = FixAtoms(indices=[0])  # Fix oxygen
        atoms.set_constraint(constraint)

        # Test optimization with constraint
        results = qme.optimize_minimum(atoms=atoms, optimizer="BFGS", fmax=0.1, steps=3)

        assert "optimized_atoms" in results
        # The constraint should be preserved
        optimized = results["optimized_atoms"]
        assert len(optimized.constraints) == 1
