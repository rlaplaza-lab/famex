"""Validation functions for FAMEX Explorer runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase import Atoms


class FAMEXError(Exception):
    """General FAMEX error with an optional suggestion for the user.

    Parameters
    ----------
    message : str
        Description of the problem.
    suggestion : str, optional
        How to fix it; appended to the message when printed.

    """

    def __init__(self, message: str, suggestion: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion

    def __str__(self) -> str:
        if self.suggestion:
            return f"{self.message}\n\n💡 Suggestion: {self.suggestion}"
        return self.message


class BackendError(Exception):
    """Raised when a requested backend is not available.

    Parameters
    ----------
    backend : str
        Backend that was requested.
    available : list[str]
        Backends that are available; listed in the message.
    operation : str
        What the backend was needed for.

    """

    def __init__(self, backend: str, available: list[str], operation: str) -> None:
        if available:
            available_str = ", ".join(available)
            suggestion = f"Available backends: {available_str}"
        else:
            suggestion = (
                "No backends are currently available. "
                "Install at least one backend (e.g., pip install torch for aimnet2)."
            )
        message = f"Backend '{backend}' is not available for {operation}. {suggestion}"
        super().__init__(message)
        self.backend = backend
        self.available_backends = available
        self.operation = operation


class DependencyError(Exception):
    """Raised when an optional dependency is missing.

    Parameters
    ----------
    dependency : str
        Name of the missing package.
    purpose : str
        What the package is needed for.
    install_command : str
        Command that installs it; shown in the message.

    """

    def __init__(self, dependency: str, purpose: str, install_command: str) -> None:
        message = (
            f"Missing dependency '{dependency}' required for {purpose}. "
            f"Install with: {install_command}"
        )
        super().__init__(message)
        self.dependency = dependency
        self.purpose = purpose
        self.install_command = install_command


def validate_atoms_compatibility(atoms1: Atoms, atoms2: Atoms, context: str = "operation") -> None:
    if len(atoms1) != len(atoms2):
        raise ValueError(
            f"Incompatible atoms for {context}: different number of atoms "
            f"({len(atoms1)} vs {len(atoms2)})"
        )
    if list(atoms1.symbols) != list(atoms2.symbols):
        raise ValueError(
            f"Incompatible atoms for {context}: different atomic symbols "
            f"({list(atoms1.symbols)} vs {list(atoms2.symbols)})"
        )
