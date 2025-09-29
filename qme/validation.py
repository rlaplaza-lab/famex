"""
Common error messages and validation patterns for QME.

This module provides standardized error messages and validation functions
to ensure consistency across the codebase.
"""

from typing import List, Optional


class QMEError(Exception):
    """Base exception class for QME-specific errors."""

    pass


class DependencyError(QMEError):
    """Raised when required dependencies are not available."""

    pass


class BackendError(QMEError):
    """Raised when backend-related errors occur."""

    pass


class ValidationError(QMEError):
    """Raised when validation fails."""

    pass


def get_dependency_error_message(dependency: str, purpose: str) -> str:
    """Get standardized dependency error message."""
    install_commands = {
        "torch": "pip install torch",
        "fairchem-core": "pip install fairchem-core",
        "sella": "pip install sella",
        "so3lr": (
            "git clone https://github.com/general-molecular-simulations/so3lr.git && "
            "cd so3lr && pip install ."
        ),
    }

    command = install_commands.get(dependency, f"pip install {dependency}")

    return f"{dependency} is required for {purpose}. " f"Install with: {command}"


def get_backend_error_message(backend: str, available_backends: List[str]) -> str:
    """Get standardized backend error message."""
    return (
        f"Unknown backend: {backend}. "
        f"Available backends: {', '.join(available_backends)}"
    )


def validate_atoms_compatibility(atoms1, atoms2, context: str = "operation"):
    """Validate that two Atoms objects are compatible.

    Parameters:
    -----------
    atoms1, atoms2 : Atoms
        Atoms objects to compare
    context : str
        Context for error message (e.g., "reaction", "interpolation")

    Raises:
    -------
    ValidationError
        If atoms are not compatible
    """
    if len(atoms1) != len(atoms2):
        raise ValidationError(
            f"Incompatible structures for {context}: "
            f"different number of atoms ({len(atoms1)} vs {len(atoms2)})"
        )

    symbols1 = atoms1.get_chemical_symbols()
    symbols2 = atoms2.get_chemical_symbols()

    if symbols1 != symbols2:
        raise ValidationError(
            f"Incompatible structures for {context}: " f"different atomic symbols"
        )


def validate_file_exists(filepath, purpose: str = "operation"):
    """Validate that a file exists.

    Parameters:
    -----------
    filepath : str or Path
        Path to file to check
    purpose : str
        Purpose for error message

    Raises:
    -------
    FileNotFoundError
        If file does not exist
    ValidationError
        If file is empty or invalid
    """
    from pathlib import Path

    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found for {purpose}: {filepath}")

    if path.stat().st_size == 0:
        raise ValidationError(f"Empty file for {purpose}: {filepath}")


def validate_model_parameters(
    model_name: Optional[str], model_path: Optional[str], backend: str
):
    """Validate model parameters for a given backend.

    Parameters:
    -----------
    model_name : str, optional
        Model name
    model_path : str, optional
        Model path
    backend : str
        Backend name

    Raises:
    -------
    ValidationError
        If parameters are invalid for the backend
    """
    if backend == "so3lr" and model_path and model_name:
        # Both specified - could be confusing
        pass  # Allow for now

    if backend in ["uma", "aimnet2"] and model_path:
        raise ValidationError(
            f"model_path is not supported for {backend} backend. "
            f"Use model_name instead."
        )
