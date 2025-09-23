"""Test calculator interfaces and implementations."""

import numpy as np
import pytest

from qme import QMEOptimizer

"""Test calculator interfaces and implementations."""

import numpy as np
import pytest
from ase.build import molecule

from qme import MockUMACalculator, QMEOptimizer


class TestCalculators:
    """Test suite for calculator implementations."""

    def test_mock_uma_calculator_basic(self):
        """Test basic functionality of MockUMACalculator."""
        calc = MockUMACalculator()

        # Create a simple H2 molecule
        h2 = molecule("H2")
        h2.set_calculator(calc)

        # Test energy calculation
        energy = h2.get_potential_energy()
        assert isinstance(energy, float)

        # Test forces calculation
        forces = h2.get_forces()
        assert forces.shape == (2, 3)  # 2 atoms, 3 dimensions
        assert isinstance(forces, np.ndarray)

    def test_qme_optimizer_with_mock(self):
        """Test QMEOptimizer with mock calculator."""
        qme = QMEOptimizer(use_mock=True, backend="uma")

        # Load a simple molecule
        h2 = molecule("H2")
        qme.atoms = h2
        qme.atoms.set_calculator(qme.calculator)

        # Test that calculator is properly set
        assert qme.calculator is not None
        assert hasattr(qme.calculator, "calculate")

        # Test energy calculation through QME
        energy = qme.atoms.get_potential_energy()
        assert isinstance(energy, float)

    def test_qme_optimizer_so3lr_mock(self):
        """Test QMEOptimizer with SO3LR mock calculator."""
        qme = QMEOptimizer(use_mock=True, backend="so3lr")

        # Load a simple molecule
        h2o = molecule("H2O")
        qme.atoms = h2o
        qme.atoms.set_calculator(qme.calculator)

        # Test that calculator is properly set
        assert qme.calculator is not None

        # Test energy calculation
        energy = qme.atoms.get_potential_energy()
        assert isinstance(energy, float)

        # Test forces calculation
        forces = qme.atoms.get_forces()
        assert forces.shape == (3, 3)  # 3 atoms, 3 dimensions

    def test_calculator_properties(self):
        """Test calculator implemented properties."""
        calc = MockUMACalculator()

        expected_properties = ["energy", "forces"]
        for prop in expected_properties:
            assert prop in calc.implemented_properties
