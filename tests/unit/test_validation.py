"""
Tests for QME validation functions.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from qme.core.validation import (
    BackendError,
    DependencyError,
    QMEError,
    ValidationError,
    validate_atoms_compatibility,
    validate_atoms_structure,
    validate_charge_and_spin,
    validate_device_parameter,
    validate_file_exists,
    validate_file_format,
    validate_model_parameters,
    validate_optimization_parameters,
)


class TestQMEError:
    """Test QME error classes."""

    def test_qme_error_basic(self):
        """Test basic QMEError functionality."""
        error = QMEError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.suggestion is None

    def test_qme_error_with_suggestion(self):
        """Test QMEError with suggestion."""
        error = QMEError("Test error", "Try this instead")
        expected = "Test error\n\n💡 Suggestion: Try this instead"
        assert str(error) == expected
        assert error.message == "Test error"
        assert error.suggestion == "Try this instead"

    def test_dependency_error(self):
        """Test DependencyError."""
        error = DependencyError("torch", "calculations", "pip install torch")
        assert "torch" in str(error)
        assert "calculations" in str(error)
        assert "pip install torch" in str(error)
        assert error.dependency == "torch"
        assert error.purpose == "calculations"

    def test_backend_error(self):
        """Test BackendError."""
        error = BackendError("unknown", ["uma", "aimnet2"], "calculation")
        assert "unknown" in str(error)
        assert "uma" in str(error)
        assert "aimnet2" in str(error)
        assert error.backend == "unknown"
        assert error.available_backends == ["uma", "aimnet2"]


class TestAtomsValidation:
    """Test atoms structure validation."""

    def test_validate_atoms_structure_none(self):
        """Test validation with None atoms."""
        with pytest.raises(ValidationError) as exc_info:
            validate_atoms_structure(None)
        assert "No atoms provided" in str(exc_info.value)
        assert "load_structure" in str(exc_info.value)

    def test_validate_atoms_structure_empty(self):
        """Test validation with empty atoms."""
        empty_atoms = Atoms()
        with pytest.raises(ValidationError) as exc_info:
            validate_atoms_structure(empty_atoms)
        assert "Empty structure" in str(exc_info.value)

    def test_validate_atoms_structure_valid(self):
        """Test validation with valid atoms."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        # Should not raise
        validate_atoms_structure(atoms)

    def test_validate_atoms_structure_overlapping(self):
        """Test validation with overlapping atoms."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.05, 0, 0]])  # Very close
        with pytest.raises(ValidationError) as exc_info:
            validate_atoms_structure(atoms)
        assert "too close" in str(exc_info.value)

    def test_validate_atoms_structure_invalid_numbers(self):
        """Test validation with invalid atomic numbers."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms.numbers[0] = 0  # Invalid atomic number
        with pytest.raises(ValidationError) as exc_info:
            validate_atoms_structure(atoms)
        assert "Invalid atomic numbers" in str(exc_info.value)


class TestOptimizationValidation:
    """Test optimization parameter validation."""

    def test_validate_optimization_parameters_valid(self):
        """Test validation with valid parameters."""
        # Should not raise
        validate_optimization_parameters(0.01, 1000, "sella")
        validate_optimization_parameters(0.05, 500, "lbfgs")

    def test_validate_optimization_parameters_invalid_fmax(self):
        """Test validation with invalid fmax."""
        with pytest.raises(ValidationError) as exc_info:
            validate_optimization_parameters(0, 1000, "sella")
        assert "Invalid force convergence threshold" in str(exc_info.value)
        assert "fmax must be positive" in str(exc_info.value)

    def test_validate_optimization_parameters_invalid_steps(self):
        """Test validation with invalid steps."""
        with pytest.raises(ValidationError) as exc_info:
            validate_optimization_parameters(0.01, 0, "sella")
        assert "Invalid maximum steps" in str(exc_info.value)
        assert "steps must be positive" in str(exc_info.value)

    def test_validate_optimization_parameters_invalid_optimizer(self):
        """Test validation with invalid optimizer."""
        with pytest.raises(ValidationError) as exc_info:
            validate_optimization_parameters(0.01, 1000, "unknown")
        assert "Unknown optimizer" in str(exc_info.value)
        assert "sella" in str(exc_info.value)


class TestDeviceValidation:
    """Test device parameter validation."""

    def test_validate_device_parameter_valid(self):
        """Test validation with valid devices."""
        # Should not raise
        validate_device_parameter("cpu", "aimnet2")
        validate_device_parameter(None, "aimnet2")
        # Only test CUDA if it's available
        try:
            validate_device_parameter("cuda", "aimnet2")
        except ValidationError:
            # CUDA not available, which is expected in some environments
            pass

    def test_validate_device_parameter_invalid(self):
        """Test validation with invalid device."""
        with pytest.raises(ValidationError) as exc_info:
            validate_device_parameter("invalid", "aimnet2")
        assert "Invalid device" in str(exc_info.value)
        assert "cpu" in str(exc_info.value)

    def test_validate_device_parameter_cuda_unavailable(self):
        """Test validation with CUDA requested but unavailable."""
        # Mock the deps system to simulate torch unavailable
        from unittest.mock import patch

        with patch("qme.dependencies.deps") as mock_deps:
            mock_deps.has.return_value = False
            with pytest.raises(ValidationError) as exc_info:
                validate_device_parameter("cuda", "aimnet2")
            assert "PyTorch not available to check CUDA" in str(exc_info.value)


class TestFileValidation:
    """Test file validation functions."""

    def test_validate_file_format_valid(self):
        """Test validation with valid file formats."""
        # Should not raise
        validate_file_format("test.xyz", [".xyz", ".pdb"])
        validate_file_format("test.pdb", [".xyz", ".pdb"])

    def test_validate_file_format_invalid(self):
        """Test validation with invalid file format."""
        with pytest.raises(ValidationError) as exc_info:
            validate_file_format("test.txt", [".xyz", ".pdb"])
        assert "Unsupported file format" in str(exc_info.value)
        assert ".xyz" in str(exc_info.value)

    def test_validate_file_exists_valid(self):
        """Test validation with existing file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            temp_path = f.name

        try:
            # Should not raise
            validate_file_exists(temp_path)
        finally:
            os.unlink(temp_path)

    def test_validate_file_exists_missing(self):
        """Test validation with missing file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            validate_file_exists("nonexistent.xyz")
        assert "File not found" in str(exc_info.value)

    def test_validate_file_exists_empty(self):
        """Test validation with empty file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            # Empty file
            pass
        temp_path = f.name

        try:
            with pytest.raises(ValidationError) as exc_info:
                validate_file_exists(temp_path)
            assert "Empty file" in str(exc_info.value)
        finally:
            os.unlink(temp_path)


class TestChargeSpinValidation:
    """Test charge and spin validation."""

    def test_validate_charge_and_spin_valid(self):
        """Test validation with valid charge and spin."""
        # Should not raise
        validate_charge_and_spin(0, 1, 10)  # Neutral, singlet
        validate_charge_and_spin(1, 2, 10)  # Cation, doublet
        validate_charge_and_spin(-1, 2, 10)  # Anion, doublet

    def test_validate_charge_and_spin_invalid_spin(self):
        """Test validation with invalid spin."""
        with pytest.raises(ValidationError) as exc_info:
            validate_charge_and_spin(0, 0, 10)
        assert "Invalid spin multiplicity" in str(exc_info.value)
        assert "≥ 1" in str(exc_info.value)

    def test_validate_charge_and_spin_invalid_combination_even_even(self):
        """Test validation with invalid even charge + even spin."""
        with pytest.raises(ValidationError) as exc_info:
            validate_charge_and_spin(2, 2, 10)
        assert "even spin" in str(exc_info.value)
        assert "even charge" in str(exc_info.value)

    def test_validate_charge_and_spin_invalid_combination_odd_odd(self):
        """Test validation with invalid odd charge + odd spin."""
        with pytest.raises(ValidationError) as exc_info:
            validate_charge_and_spin(1, 1, 10)
        assert "odd spin" in str(exc_info.value)
        assert "odd charge" in str(exc_info.value)

    def test_validate_charge_and_spin_too_many_charges(self):
        """Test validation with too many positive charges."""
        with pytest.raises(ValidationError) as exc_info:
            validate_charge_and_spin(15, 2, 10)  # 15 charges > 10 electrons, even spin
        assert "Too many positive charges" in str(exc_info.value)


class TestAtomsCompatibility:
    """Test atoms compatibility validation."""

    def test_validate_atoms_compatibility_valid(self):
        """Test validation with compatible atoms."""
        atoms1 = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms2 = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        # Should not raise
        validate_atoms_compatibility(atoms1, atoms2)

    def test_validate_atoms_compatibility_different_length(self):
        """Test validation with different number of atoms."""
        atoms1 = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms2 = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
        with pytest.raises(ValidationError) as exc_info:
            validate_atoms_compatibility(atoms1, atoms2)
        assert "different number of atoms" in str(exc_info.value)
        assert "3 vs 2" in str(exc_info.value)

    def test_validate_atoms_compatibility_different_symbols(self):
        """Test validation with different atomic symbols."""
        atoms1 = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms2 = Atoms("H2S", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        with pytest.raises(ValidationError) as exc_info:
            validate_atoms_compatibility(atoms1, atoms2)
        assert "different atomic symbols" in str(exc_info.value)


class TestModelValidation:
    """Test model parameter validation."""

    def test_validate_model_parameters_valid(self):
        """Test validation with valid parameters."""
        # Should not raise
        validate_model_parameters("uma-s-1p1", None, "uma")
        validate_model_parameters(None, "/path/to/model.jpt", "so3lr")

    def test_validate_model_parameters_uma_with_path(self):
        """Test validation with UMA and model_path."""
        with pytest.raises(ValidationError) as exc_info:
            validate_model_parameters("uma-s-1p1", "/path/to/model.jpt", "uma")
        assert "model_path is not supported" in str(exc_info.value)
        assert "Use model_name instead" in str(exc_info.value)

    def test_validate_model_parameters_aimnet2_with_path(self):
        """Test validation with AIMNet2 and model_path."""
        with pytest.raises(ValidationError) as exc_info:
            validate_model_parameters("aimnet2", "/path/to/model.jpt", "aimnet2")
        assert "model_path is not supported" in str(exc_info.value)
        assert "Use model_name instead" in str(exc_info.value)
