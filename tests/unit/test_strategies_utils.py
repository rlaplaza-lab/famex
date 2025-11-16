"""Tests for strategy utility functions."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from ase import Atoms

from qme.strategies.utils import StrategyUtils


class TestStrategyUtils:
    """Tests for StrategyUtils class."""

    def test_ensure_charge_spin_info_with_info(self):
        """Test ensure_charge_spin_info when atoms.info exists."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        atoms.info = {}

        StrategyUtils.ensure_charge_spin_info(atoms, charge=0, spin=1)

        assert atoms.info["charge"] == 0
        assert atoms.info["spin"] == 1

    def test_ensure_charge_spin_info_preserves_existing(self):
        """Test ensure_charge_spin_info preserves existing values."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        atoms.info = {"charge": 2, "spin": 3}

        StrategyUtils.ensure_charge_spin_info(atoms, charge=0, spin=1)

        assert atoms.info["charge"] == 2  # Preserved
        assert atoms.info["spin"] == 3  # Preserved

    def test_ensure_charge_spin_info_no_info(self):
        """Test ensure_charge_spin_info when atoms.info doesn't exist."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        # Remove info attribute if it exists
        if hasattr(atoms, "info"):
            delattr(atoms, "info")

        # Should not raise error
        StrategyUtils.ensure_charge_spin_info(atoms, charge=0, spin=1)

    def test_get_step_count_from_step_count_attr(self):
        """Test get_step_count when optimizer has step_count attribute."""
        optimizer = MagicMock()
        optimizer.step_count = 5

        result = StrategyUtils.get_step_count(optimizer)

        assert result == 5

    def test_get_step_count_from_get_number_of_steps(self):
        """Test get_step_count when optimizer has get_number_of_steps method."""
        optimizer = MagicMock()
        optimizer.step_count = None
        optimizer.get_number_of_steps.return_value = 10

        result = StrategyUtils.get_step_count(optimizer)

        assert result == 10
        optimizer.get_number_of_steps.assert_called_once()

    def test_get_step_count_float_conversion(self):
        """Test get_step_count converts float to int."""
        optimizer = MagicMock()
        optimizer.step_count = 5.0

        result = StrategyUtils.get_step_count(optimizer)

        assert result == 5
        assert isinstance(result, int)

    def test_get_step_count_none(self):
        """Test get_step_count returns None when not available."""
        optimizer = MagicMock()
        del optimizer.step_count
        del optimizer.get_number_of_steps

        result = StrategyUtils.get_step_count(optimizer)

        assert result is None

    def test_get_convergence_status_callable_no_args(self):
        """Test get_convergence_status with callable converged that takes no args."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        optimizer = MagicMock()
        optimizer.converged = MagicMock(return_value=True)

        result = StrategyUtils.get_convergence_status(optimizer, atoms)

        assert result is True
        optimizer.converged.assert_called_once()

    def test_get_convergence_status_callable_with_args(self):
        """Test get_convergence_status with callable converged that needs forces."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        atoms.calc = MagicMock()
        atoms.calc.get_forces.return_value = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        optimizer = MagicMock()

        def converged_func(*args):
            if len(args) == 0:
                raise TypeError("needs argument")
            return True

        optimizer.converged = converged_func

        result = StrategyUtils.get_convergence_status(optimizer, atoms)

        assert isinstance(result, bool)
        assert result is True

    def test_get_convergence_status_non_callable(self):
        """Test get_convergence_status with non-callable converged attribute."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        optimizer = MagicMock()
        optimizer.converged = True

        result = StrategyUtils.get_convergence_status(optimizer, atoms)

        assert result is True

    def test_get_convergence_status_no_converged(self):
        """Test get_convergence_status when converged attribute doesn't exist."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        optimizer = MagicMock()
        del optimizer.converged

        result = StrategyUtils.get_convergence_status(optimizer, atoms)

        assert result is False

    @pytest.mark.parametrize(
        ("scenario", "expected_result"),
        [
            ("supports_batch", True),
            ("no_calculate_batch", False),
            ("no_supports_attr", False),
            ("supports_false", False),
            ("none_calculator", False),
        ],
        ids=[
            "supports_batch",
            "no_calculate_batch",
            "no_supports_attr",
            "supports_false",
            "none_calculator",
        ],
    )
    def test_check_batch_support_scenarios(self, scenario, expected_result):
        """Test check_batch_support with various calculator configurations."""
        if scenario == "supports_batch":
            calculator = MagicMock()
            calculator.supports_batch_evaluation = True
            calculator.calculate_batch = MagicMock()
        elif scenario == "no_calculate_batch":
            calculator = MagicMock()
            calculator.supports_batch_evaluation = True
            del calculator.calculate_batch
        elif scenario == "no_supports_attr":
            calculator = MagicMock()
            calculator.calculate_batch = MagicMock()
            del calculator.supports_batch_evaluation
        elif scenario == "supports_false":
            calculator = MagicMock()
            calculator.supports_batch_evaluation = False
            calculator.calculate_batch = MagicMock()
        elif scenario == "none_calculator":
            calculator = None
        else:
            pytest.fail(f"Unknown scenario: {scenario}")

        result = StrategyUtils.check_batch_support(calculator)
        assert result is expected_result

    def test_calculate_batch_energies_forces_batch_success(self, mock_backend):
        """Test calculate_batch_energies_forces with successful batch calculation."""
        atoms1 = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        atoms2 = Atoms("H2", positions=[[0, 0, 0], [0.8, 0, 0]])
        atoms1.calc = mock_backend
        atoms2.calc = mock_backend

        path = [atoms1, atoms2]
        calculator = MagicMock()
        calculator.calculate_batch.return_value = [
            {"energy": 1.0, "forces": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])},
            {"energy": 1.1, "forces": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])},
        ]

        energies, forces_list = StrategyUtils.calculate_batch_energies_forces(
            path, calculator, True
        )

        assert len(energies) == 2
        assert len(forces_list) == 2
        assert energies[0] == 1.0
        assert energies[1] == 1.1

    def test_calculate_batch_energies_forces_batch_fallback(self, mock_backend):
        """Test calculate_batch_energies_forces falls back to individual calculations."""
        atoms1 = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        atoms2 = Atoms("H2", positions=[[0, 0, 0], [0.8, 0, 0]])
        atoms1.calc = mock_backend
        atoms2.calc = mock_backend

        path = [atoms1, atoms2]
        calculator = MagicMock()
        calculator.calculate_batch.side_effect = RuntimeError("Batch failed")

        energies, forces_list = StrategyUtils.calculate_batch_energies_forces(
            path, calculator, True
        )

        assert len(energies) == 2
        assert len(forces_list) == 2

    def test_calculate_batch_energies_forces_no_batch(self, mock_backend):
        """Test calculate_batch_energies_forces without batch support."""
        atoms1 = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        atoms2 = Atoms("H2", positions=[[0, 0, 0], [0.8, 0, 0]])
        atoms1.calc = mock_backend
        atoms2.calc = mock_backend

        path = [atoms1, atoms2]
        calculator = MagicMock()

        energies, forces_list = StrategyUtils.calculate_batch_energies_forces(
            path, calculator, False
        )

        assert len(energies) == 2
        assert len(forces_list) == 2

    def test_check_convergence_true(self):
        """Test check_convergence when converged."""
        forces_list = [
            np.array([[0.01, 0.0, 0.0], [0.01, 0.0, 0.0]]),
            np.array([[0.02, 0.0, 0.0], [0.02, 0.0, 0.0]]),
        ]
        fmax = 0.05

        result = StrategyUtils.check_convergence(forces_list, fmax, step=0)

        assert result is True

    def test_check_convergence_false(self):
        """Test check_convergence when not converged."""
        forces_list = [
            np.array([[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]),
            np.array([[0.2, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        ]
        fmax = 0.05

        result = StrategyUtils.check_convergence(forces_list, fmax, step=0)

        assert result is False
