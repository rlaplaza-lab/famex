"""
Tests for QME validation functions.
"""

import pytest
from ase import Atoms

from qme.core.validation import (
    BackendError,
    DependencyError,
    QMEError,
    validate_atoms_compatibility,
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
        assert error.install_command == "pip install torch"

    def test_backend_error(self):
        """Test BackendError."""
        available = ["uma", "aimnet2", "mace"]
        error = BackendError("so3lr", available, "optimization")
        assert "so3lr" in str(error)
        assert "optimization" in str(error)
        assert "uma" in str(error)
        assert "aimnet2" in str(error)
        assert "mace" in str(error)
        assert error.backend == "so3lr"
        assert error.available_backends == available
        assert error.operation == "optimization"


class TestValidateAtomsCompatibility:
    """Test validate_atoms_compatibility function."""

    def test_compatible_atoms(self):
        """Test validation with compatible atoms."""
        atoms1 = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
        atoms2 = Atoms("H2", positions=[[0, 0, 0], [1.1, 0, 0]])

        # Should not raise any exception
        validate_atoms_compatibility(atoms1, atoms2)

    def test_different_number_of_atoms(self):
        """Test validation with different number of atoms."""
        atoms1 = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
        atoms2 = Atoms("H2S", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])

        with pytest.raises(ValueError) as exc_info:
            validate_atoms_compatibility(atoms1, atoms2)

        assert "different number of atoms" in str(exc_info.value)
        assert "2 vs 3" in str(exc_info.value)

    def test_different_atomic_symbols(self):
        """Test validation with different atomic symbols."""
        atoms1 = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
        atoms2 = Atoms(
            "He2", positions=[[0, 0, 0], [1, 0, 0]]
        )  # Same number of atoms, different symbols

        with pytest.raises(ValueError) as exc_info:
            validate_atoms_compatibility(atoms1, atoms2)

        assert "different atomic symbols" in str(exc_info.value)

    def test_custom_context(self):
        """Test validation with custom context."""
        atoms1 = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
        atoms2 = Atoms("H2S", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])

        with pytest.raises(ValueError) as exc_info:
            validate_atoms_compatibility(atoms1, atoms2, "path segment 0")

        assert "path segment 0" in str(exc_info.value)
