"""
Common error messages and validation patterns for QME.

This module provides standardized error messages and validation functions
to ensure consistency across the codebase.
"""

from pathlib import Path
from typing import List, Optional, Union


class QMEError(Exception):
    """Base exception class for QME-specific errors."""

    def __init__(self, message: str, suggestion: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion

    def __str__(self):
        if self.suggestion:
            return f"{self.message}\n\n💡 Suggestion: {self.suggestion}"
        return self.message


class DependencyError(QMEError):
    """Raised when required dependencies are not available."""

    def __init__(
        self, dependency: str, purpose: str, install_command: Optional[str] = None
    ):
        message = f"Missing dependency '{dependency}' required for {purpose}."
        suggestion = f"Install with: {install_command or f'pip install {dependency}'}"
        super().__init__(message, suggestion)
        self.dependency = dependency
        self.purpose = purpose


class BackendError(QMEError):
    """Raised when backend-related errors occur."""

    def __init__(
        self, backend: str, available_backends: List[str], context: str = "calculation"
    ):
        message = f"Backend '{backend}' is not available for {context}."
        suggestion = f"Available backends: {', '.join(available_backends)}"
        super().__init__(message, suggestion)
        self.backend = backend
        self.available_backends = available_backends


class ValidationError(QMEError):
    """Raised when validation fails."""

    def __init__(self, message: str, suggestion: Optional[str] = None):
        super().__init__(message, suggestion)


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
            f"Use model_name instead.",
            suggestion="Remove the model_path parameter and use model_name instead.",
        )


def validate_atoms_structure(atoms, context: str = "calculation"):
    """Validate that an Atoms object is suitable for calculations.

    Parameters:
    -----------
    atoms : Atoms
        Atoms object to validate
    context : str
        Context for error message

    Raises:
    -------
    ValidationError
        If atoms structure is invalid
    """
    if atoms is None:
        raise ValidationError(
            f"No atoms provided for {context}.",
            suggestion="Load a molecular structure using load_structure() or pass atoms parameter.",
        )

    if len(atoms) == 0:
        raise ValidationError(
            f"Empty structure provided for {context}.",
            suggestion="Ensure the structure contains at least one atom.",
        )

    # Check for overlapping atoms (very close positions)
    positions = atoms.get_positions()
    if len(positions) > 1:
        from scipy.spatial.distance import pdist

        distances = pdist(positions)
        min_distance = distances.min()
        if min_distance < 0.1:  # Less than 0.1 Å
            raise ValidationError(
                f"Atoms are too close (minimum distance: {min_distance:.3f} Å) for {context}.",
                suggestion="Check for overlapping atoms or adjust the structure geometry.",
            )

    # Check for reasonable atomic numbers
    numbers = atoms.get_atomic_numbers()
    if any(num <= 0 or num > 118 for num in numbers):
        raise ValidationError(
            f"Invalid atomic numbers found for {context}.",
            suggestion="Ensure all atomic numbers are between 1 and 118.",
        )


def validate_optimization_parameters(fmax: float, steps: int, optimizer: str):
    """Validate optimization parameters.

    Parameters:
    -----------
    fmax : float
        Force convergence threshold
    steps : int
        Maximum number of steps
    optimizer : str
        Optimizer name

    Raises:
    -------
    ValidationError
        If parameters are invalid
    """
    if fmax <= 0:
        raise ValidationError(
            f"Invalid force convergence threshold: {fmax}",
            suggestion="fmax must be positive (e.g., 0.01 eV/Å).",
        )

    if steps <= 0:
        raise ValidationError(
            f"Invalid maximum steps: {steps}",
            suggestion="steps must be positive (e.g., 1000).",
        )

    valid_optimizers = ["sella", "lbfgs", "bfgs", "fire"]
    if optimizer.lower() not in valid_optimizers:
        raise ValidationError(
            f"Unknown optimizer: {optimizer}",
            suggestion=f"Use one of: {', '.join(valid_optimizers)}",
        )


def validate_device_parameter(device: Optional[str], backend: str):
    """Validate device parameter for a given backend.

    Parameters:
    -----------
    device : str, optional
        Device specification
    backend : str
        Backend name

    Raises:
    -------
    ValidationError
        If device parameter is invalid
    """
    if device is None:
        return  # None is valid (auto-detect)

    valid_devices = ["cpu", "cuda", "gpu"]
    if device.lower() not in valid_devices:
        raise ValidationError(
            f"Invalid device: {device}",
            suggestion=f"Use one of: {', '.join(valid_devices)} or None for auto-detection.",
        )

    if device.lower() in ["cuda", "gpu"]:
        try:
            import torch

            if not torch.cuda.is_available():
                raise ValidationError(
                    f"CUDA device requested but CUDA is not available for {backend}.",
                    suggestion="Use device='cpu' or install CUDA-enabled PyTorch.",
                )
        except ImportError:
            raise ValidationError(
                f"PyTorch not available to check CUDA for {backend}.",
                suggestion="Install PyTorch or use device='cpu'.",
            )


def validate_file_format(filepath: Union[str, Path], supported_formats: List[str]):
    """Validate that a file has a supported format.

    Parameters:
    -----------
    filepath : str or Path
        Path to file to validate
    supported_formats : List[str]
        List of supported file extensions

    Raises:
    -------
    ValidationError
        If file format is not supported
    """
    path = Path(filepath)
    suffix = path.suffix.lower()

    if suffix not in supported_formats:
        raise ValidationError(
            f"Unsupported file format: {suffix}",
            suggestion=f"Use one of: {', '.join(supported_formats)}",
        )


def validate_charge_and_spin(charge: int, spin: int, n_electrons: int):
    """Validate charge and spin parameters.

    Parameters:
    -----------
    charge : int
        Molecular charge
    spin : int
        Spin multiplicity (2S + 1)
    n_electrons : int
        Number of electrons in neutral system

    Raises:
    -------
    ValidationError
        If charge or spin are invalid
    """
    if spin < 1:
        raise ValidationError(
            f"Invalid spin multiplicity: {spin}",
            suggestion="Spin multiplicity must be ≥ 1 (2S + 1).",
        )

    if spin % 2 == 0 and charge % 2 == 0:
        raise ValidationError(
            f"Invalid combination: even spin ({spin}) with even charge ({charge})",
            suggestion="For even charge, use odd spin multiplicity.",
        )

    if spin % 2 == 1 and charge % 2 == 1:
        raise ValidationError(
            f"Invalid combination: odd spin ({spin}) with odd charge ({charge})",
            suggestion="For odd charge, use even spin multiplicity.",
        )

    total_electrons = n_electrons - charge
    if total_electrons < 0:
        raise ValidationError(
            f"Too many positive charges: {charge} > {n_electrons} electrons",
            suggestion="Reduce the charge or check the molecular formula.",
        )
