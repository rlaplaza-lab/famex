"""Validation functions for QME Explorer runs.

This module provides input validation for different types of optimization runs,
checking atoms, optimizer compatibility, backend requirements, and other constraints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase import Atoms


class QMEError(Exception):
    """Base exception for QME errors."""

    def __init__(self, message: str, suggestion: str | None = None) -> None:
        """Initialize QME error with message and optional suggestion."""
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion

    def __str__(self) -> str:
        """Return formatted error message with optional suggestion."""
        if self.suggestion:
            return f"{self.message}\n\n💡 Suggestion: {self.suggestion}"
        return self.message


class BackendError(Exception):
    """Raised when backend is unavailable or incompatible."""

    def __init__(self, backend: str, available: list[str], operation: str) -> None:
        """Initialize backend error with backend name and available alternatives."""
        message = (
            f"Backend '{backend}' is not available for {operation}. "
            f"Available backends: {', '.join(available)}"
        )
        super().__init__(message)
        self.backend = backend
        self.available_backends = available
        self.operation = operation


class DependencyError(Exception):
    """Raised when required dependencies are missing."""

    def __init__(self, dependency: str, purpose: str, install_command: str) -> None:
        """Initialize dependency error with dependency name and install command."""
        message = (
            f"Missing dependency '{dependency}' required for {purpose}. "
            f"Install with: {install_command}"
        )
        super().__init__(message)
        self.dependency = dependency
        self.purpose = purpose
        self.install_command = install_command


def validate_atoms_compatibility(atoms1: Atoms, atoms2: Atoms, context: str = "operation") -> None:
    """Validate that two Atoms objects are compatible for operations.

    Parameters
    ----------
    atoms1, atoms2 : Atoms
        The two structures to compare
    context : str
        Context for error messages

    Raises:
    ------
    ValueError
        If atoms are incompatible

    """
    if len(atoms1) != len(atoms2):
        msg = (
            f"Incompatible atoms for {context}: different number of atoms "
            f"({len(atoms1)} vs {len(atoms2)})"
        )
        raise ValueError(
            msg,
        )

    if list(atoms1.symbols) != list(atoms2.symbols):
        msg = (
            f"Incompatible atoms for {context}: different atomic symbols "
            f"({list(atoms1.symbols)} vs {list(atoms2.symbols)})"
        )
        raise ValueError(
            msg,
        )
