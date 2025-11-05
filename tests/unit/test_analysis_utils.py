"""Tests for analysis utility functions."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from ase import Atoms

from qme.analysis.utils import get_calculator_property, has_calculator_property, validate_indices


class TestValidateIndices:
    """Tests for validate_indices function."""

    def test_none_returns_all_indices(self):
        """Test that None returns all atom indices."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 0.96], [0.82, 0, -0.26]])
        result = validate_indices(atoms, None)
        assert result == [0, 1, 2]

    def test_valid_indices(self):
        """Test that valid indices are returned as-is."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 0.96], [0.82, 0, -0.26]])
        result = validate_indices(atoms, [0, 2])
        assert result == [0, 2]

    def test_empty_list_raises_error(self):
        """Test that empty list raises ValueError."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 0.96], [0.82, 0, -0.26]])
        with pytest.raises(ValueError, match="non-empty list"):
            validate_indices(atoms, [])

    def test_not_list_raises_error(self):
        """Test that non-list input raises ValueError."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 0.96], [0.82, 0, -0.26]])
        with pytest.raises(ValueError, match="non-empty list"):
            validate_indices(atoms, "not_a_list")  # type: ignore[arg-type]

    def test_duplicate_indices_raise_error(self):
        """Test that duplicate indices raise ValueError."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 0.96], [0.82, 0, -0.26]])
        with pytest.raises(ValueError, match="must be unique"):
            validate_indices(atoms, [0, 1, 0])

    def test_out_of_bounds_negative_raises_error(self):
        """Test that negative indices raise ValueError."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 0.96], [0.82, 0, -0.26]])
        with pytest.raises(ValueError, match="out of bounds"):
            validate_indices(atoms, [-1])

    def test_out_of_bounds_positive_raises_error(self):
        """Test that indices beyond array length raise ValueError."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 0.96], [0.82, 0, -0.26]])
        with pytest.raises(ValueError, match="out of bounds"):
            validate_indices(atoms, [3])

    def test_error_message_contains_invalid_indices(self):
        """Test that error message contains invalid indices."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 0.96], [0.82, 0, -0.26]])
        with pytest.raises(ValueError) as exc_info:
            validate_indices(atoms, [0, 5, 10])
        assert "5" in str(exc_info.value)
        assert "10" in str(exc_info.value)
        assert "3 atoms" in str(exc_info.value)


class TestGetCalculatorProperty:
    """Tests for get_calculator_property function."""

    def test_implemented_properties_interface(self):
        """Test calculator with implemented_properties interface."""
        calculator = MagicMock()
        calculator.implemented_properties = ["hessian", "energy"]
        calculator.get_property = MagicMock(return_value="test_hessian")

        result = get_calculator_property(calculator, "hessian")
        assert result == "test_hessian"
        calculator.get_property.assert_called_once_with("hessian")

    def test_implemented_properties_with_atoms(self):
        """Test implemented_properties interface with atoms parameter."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        calculator = MagicMock()
        calculator.implemented_properties = ["hessian"]
        calculator.get_property = MagicMock(return_value="test_hessian")

        result = get_calculator_property(calculator, "hessian", atoms=atoms)
        assert result == "test_hessian"
        calculator.get_property.assert_called_once_with("hessian", atoms)

    def test_get_property_name_pattern(self):
        """Test calculator with get_{property_name} method pattern."""
        calculator = MagicMock()
        calculator.get_hessian = MagicMock(return_value="test_hessian")

        result = get_calculator_property(calculator, "hessian")
        assert result == "test_hessian"
        calculator.get_hessian.assert_called_once()

    def test_get_property_name_pattern_with_atoms(self):
        """Test get_{property_name} pattern with atoms parameter."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        calculator = MagicMock()
        calculator.get_hessian = MagicMock(return_value="test_hessian")

        result = get_calculator_property(calculator, "hessian", atoms=atoms)
        assert result == "test_hessian"
        calculator.get_hessian.assert_called_once_with(atoms)

    def test_calculate_property_name_pattern(self):
        """Test calculator with calculate_{property_name} method pattern."""
        calculator = MagicMock(spec=[])
        calculator.calculate_hessian = MagicMock(return_value="test_hessian")

        result = get_calculator_property(calculator, "hessian")
        assert result == "test_hessian"
        calculator.calculate_hessian.assert_called_once()

    def test_calculate_property_name_pattern_with_atoms(self):
        """Test calculate_{property_name} pattern with atoms parameter."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        calculator = MagicMock(spec=[])
        calculator.calculate_hessian = MagicMock(return_value="test_hessian")

        result = get_calculator_property(calculator, "hessian", atoms=atoms)
        assert result == "test_hessian"
        calculator.calculate_hessian.assert_called_once_with(atoms)

    def test_priority_order(self):
        """Test that implemented_properties takes priority over method patterns."""
        calculator = MagicMock()
        calculator.implemented_properties = ["hessian"]
        calculator.get_property = MagicMock(return_value="from_implemented")
        calculator.get_hessian = MagicMock(return_value="from_get_method")

        result = get_calculator_property(calculator, "hessian")
        assert result == "from_implemented"
        calculator.get_property.assert_called_once()
        calculator.get_hessian.assert_not_called()

    def test_not_found_without_default_raises_error(self):
        """Test that missing property raises AttributeError when no default."""
        # Create a calculator with no methods or properties
        calculator = MagicMock(spec=[])
        # Don't set any attributes

        with pytest.raises(AttributeError, match="does not support property"):
            get_calculator_property(calculator, "hessian")

    def test_not_found_with_default_returns_default(self):
        """Test that missing property returns default when provided."""
        calculator = MagicMock(spec=[])

        result = get_calculator_property(calculator, "hessian", default="default_value")
        assert result == "default_value"


class TestHasCalculatorProperty:
    """Tests for has_calculator_property function."""

    def test_implemented_properties_interface(self):
        """Test detection via implemented_properties."""
        calculator = MagicMock(spec=[])
        calculator.implemented_properties = ["hessian", "energy"]

        assert has_calculator_property(calculator, "hessian") is True
        assert has_calculator_property(calculator, "energy") is True
        assert has_calculator_property(calculator, "forces") is False

    def test_get_property_name_pattern(self):
        """Test detection via get_{property_name} method."""
        calculator = MagicMock(spec=[])
        calculator.get_hessian = MagicMock()

        assert has_calculator_property(calculator, "hessian") is True
        assert has_calculator_property(calculator, "energy") is False

    def test_calculate_property_name_pattern(self):
        """Test detection via calculate_{property_name} method."""
        calculator = MagicMock(spec=[])
        calculator.calculate_hessian = MagicMock()

        assert has_calculator_property(calculator, "hessian") is True
        assert has_calculator_property(calculator, "energy") is False

    def test_priority_order(self):
        """Test that implemented_properties is checked first."""
        calculator = MagicMock()
        calculator.implemented_properties = ["hessian"]
        calculator.get_hessian = MagicMock()

        # Should return True from implemented_properties, not from get_hessian
        assert has_calculator_property(calculator, "hessian") is True

    def test_no_property_support(self):
        """Test calculator with no property support."""
        calculator = MagicMock(spec=[])

        assert has_calculator_property(calculator, "hessian") is False
        assert has_calculator_property(calculator, "energy") is False

    def test_calculator_without_implemented_properties(self):
        """Test calculator without implemented_properties attribute."""
        calculator = MagicMock(spec=[])
        calculator.get_hessian = MagicMock()

        assert has_calculator_property(calculator, "hessian") is True

    def test_multiple_interfaces(self):
        """Test calculator that supports multiple interfaces."""
        calculator = MagicMock(spec=[])
        calculator.implemented_properties = ["hessian"]
        calculator.get_energy = MagicMock()
        calculator.calculate_forces = MagicMock()

        assert has_calculator_property(calculator, "hessian") is True
        assert has_calculator_property(calculator, "energy") is True
        assert has_calculator_property(calculator, "forces") is True
