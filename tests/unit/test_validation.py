from __future__ import annotations

import pytest
from ase import Atoms

from famex.utils.validation import (
    BackendError,
    DependencyError,
    FAMEXError,
    validate_atoms_compatibility,
)
from tests.test_utils import assert_error_contains


class TestFAMEXError:
    def test_famex_error_basic(self):
        error = FAMEXError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.suggestion is None

    def test_famex_error_with_suggestion(self):
        error = FAMEXError("Test error", "Try this instead")
        expected = "Test error\n\n💡 Suggestion: Try this instead"
        assert str(error) == expected
        assert error.message == "Test error"
        assert error.suggestion == "Try this instead"

    def test_dependency_error(self):
        error = DependencyError("torch", "calculations", "pip install torch")
        assert "torch" in str(error)
        assert "calculations" in str(error)
        assert "pip install torch" in str(error)
        assert error.dependency == "torch"
        assert error.purpose == "calculations"
        assert error.install_command == "pip install torch"

    def test_backend_error(self):
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

    def test_backend_error_no_available_backends(self):
        """Test BackendError when no backends are available."""
        error = BackendError("so3lr", [], "optimization")
        assert "so3lr" in str(error)
        assert "optimization" in str(error)
        assert "No backends are currently available" in str(error)
        assert "Install at least one backend" in str(error)
        assert error.backend == "so3lr"
        assert error.available_backends == []
        assert error.operation == "optimization"


class TestValidateAtomsCompatibility:
    def test_compatible_atoms(self):
        atoms1 = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
        atoms2 = Atoms("H2", positions=[[0, 0, 0], [1.1, 0, 0]])

        # Should not raise any exception
        validate_atoms_compatibility(atoms1, atoms2)

    def test_different_number_of_atoms(self):
        atoms1 = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
        atoms2 = Atoms("H2S", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])

        with pytest.raises(ValueError) as exc_info:
            validate_atoms_compatibility(atoms1, atoms2)

        assert_error_contains(exc_info.value, "different number of atoms")
        assert_error_contains(exc_info.value, "2 vs 3")

    def test_different_atomic_symbols(self):
        atoms1 = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
        atoms2 = Atoms(
            "He2",
            positions=[[0, 0, 0], [1, 0, 0]],
        )  # Same number of atoms, different symbols

        with pytest.raises(ValueError) as exc_info:
            validate_atoms_compatibility(atoms1, atoms2)

        assert_error_contains(exc_info.value, "different atomic symbols")

    def test_custom_context(self):
        atoms1 = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
        atoms2 = Atoms("H2S", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])

        with pytest.raises(ValueError) as exc_info:
            validate_atoms_compatibility(atoms1, atoms2, "path segment 0")

        assert_error_contains(exc_info.value, "path segment 0")
