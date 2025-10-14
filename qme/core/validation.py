"""Validation functions for QME Explorer runs.

This module provides input validation for different types of optimization runs,
checking atoms, optimizer compatibility, backend requirements, and other constraints.
"""

import os
from pathlib import Path

import numpy as np
from ase import Atoms

from qme.backend_availability import get_available_backends, is_backend_available

# =============================================================================
# Constants for Target and Strategy Parameters
# =============================================================================

# Targets (what you want to obtain)
TARGET_MINIMA = "minima"
TARGET_TS = "ts"
TARGET_PATH = "path"

# Strategies (how to get there)
STRATEGY_LOCAL = "local"
STRATEGY_NEB = "neb"
STRATEGY_CINEB = "cineb"
STRATEGY_INTERPOLATE = "interpolate"

# Aliases for normalization (backward compatibility)
TARGET_ALIASES = {
    "minimum": TARGET_MINIMA,
    "min": TARGET_MINIMA,
    "local_minima": TARGET_MINIMA,
    "transition_state": TARGET_TS,
    "transition-state": TARGET_TS,
    "transition": TARGET_TS,
    "ts": TARGET_TS,
    "reaction_path": TARGET_PATH,
    "reaction-path": TARGET_PATH,
    "path": TARGET_PATH,
    "trajectory": TARGET_PATH,
    "neb": TARGET_PATH,  # For backward compatibility
    "cineb": TARGET_PATH,  # For backward compatibility
}

STRATEGY_ALIASES = {
    # Backward compatibility for old "two-ended" concept
    "two-ended": STRATEGY_INTERPOLATE,
    "twoended": STRATEGY_INTERPOLATE,
    "two_ended": STRATEGY_INTERPOLATE,
    "two": STRATEGY_INTERPOLATE,
    # Other aliases
    "interpolate": STRATEGY_INTERPOLATE,
    "interpolation": STRATEGY_INTERPOLATE,
    "local": STRATEGY_LOCAL,
    "one-ended": STRATEGY_LOCAL,
    "oneended": STRATEGY_LOCAL,
    "one_ended": STRATEGY_LOCAL,
    "neb": STRATEGY_NEB,
    "nudged-elastic-band": STRATEGY_NEB,
    "nudged_elastic_band": STRATEGY_NEB,
    "cineb": STRATEGY_CINEB,
    "climbing-image-neb": STRATEGY_CINEB,
    "climbing_image_neb": STRATEGY_CINEB,
    "ci-neb": STRATEGY_CINEB,
    "ci_neb": STRATEGY_CINEB,
}


# =============================================================================
# Helper Functions for Target/Strategy Normalization
# =============================================================================


def normalize_target(target: str) -> str:
    """Normalize target string using aliases.

    Parameters
    ----------
    target : str
        Target string to normalize

    Returns
    -------
    str
        Normalized target string

    Raises
    ------
    QMEValidationError
        If target is not recognized
    """
    if not target:
        return TARGET_MINIMA  # Default target

    normalized = target.strip().lower()
    return TARGET_ALIASES.get(normalized, normalized)


def normalize_strategy(strategy: str) -> str:
    """Normalize strategy string using aliases.

    Parameters
    ----------
    strategy : str
        Strategy string to normalize

    Returns
    -------
    str
        Normalized strategy string

    Raises
    ------
    QMEValidationError
        If strategy is not recognized
    """
    if not strategy:
        return STRATEGY_LOCAL  # Default strategy

    normalized = strategy.strip().lower()
    return STRATEGY_ALIASES.get(normalized, normalized)


def prepare_atoms_list(
    atoms_list: Atoms | list[Atoms], product_kwarg: Atoms | None = None
) -> list[Atoms]:
    """Prepare atoms list with product handling.

    Parameters
    ----------
    atoms_list : Atoms or List[Atoms]
        Initial atoms list
    product_kwarg : Atoms, optional
        Product structure to append

    Returns
    -------
    List[Atoms]
        Final atoms list

    Raises
    ------
    QMEValidationError
        If atoms list is invalid
    """
    # Convert single Atoms to list
    if isinstance(atoms_list, Atoms):
        atoms_list = [atoms_list]
    else:
        atoms_list = list(atoms_list)

    # Add product if provided
    if product_kwarg is not None:
        atoms_list.append(product_kwarg)

    # Validate the final list
    if not atoms_list:
        raise QMEValidationError("No atoms provided")

    return atoms_list


def validate_target_strategy_combination(target: str, strategy: str, num_atoms: int) -> None:
    """Validate target/strategy combination.

    Parameters
    ----------
    target : str
        Normalized target
    strategy : str
        Normalized strategy
    num_atoms : int
        Number of atoms structures provided

    Raises
    ------
    QMEValidationError
        If combination is invalid
    """
    # Path strategies require multiple structures
    if target == TARGET_PATH and strategy in [STRATEGY_NEB, STRATEGY_CINEB]:
        if num_atoms < 2:
            raise QMEValidationError(
                f"Path optimization with {strategy} requires at least 2 structures, got {num_atoms}"
            )

    # Interpolate strategies require multiple structures
    if strategy == STRATEGY_INTERPOLATE:
        if num_atoms < 2:
            raise QMEValidationError(
                f"Interpolation strategy requires at least 2 structures, got {num_atoms}"
            )

    # Local strategies work with single structure
    if strategy == STRATEGY_LOCAL and num_atoms > 1:
        # This is not an error, but we might want to warn
        pass


class QMEValidationError(Exception):
    """Raised when input validation fails."""


# Alias for backward compatibility
ValidationError = QMEValidationError


class QMEBackendError(Exception):
    """Raised when backend is incompatible with the requested operation."""


class QMEStrategyError(Exception):
    """Raised when strategy selection fails."""


class QMEError(Exception):
    """Base exception for QME errors."""

    def __init__(self, message: str, suggestion: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion

    def __str__(self) -> str:
        if self.suggestion:
            return f"{self.message}\n\n💡 Suggestion: {self.suggestion}"
        return self.message


class BackendError(Exception):
    """Raised when backend is unavailable or incompatible."""

    def __init__(self, backend: str, available: list[str], operation: str) -> None:
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
        message = (
            f"Missing dependency '{dependency}' required for {purpose}. "
            f"Install with: {install_command}"
        )
        super().__init__(message)
        self.dependency = dependency
        self.purpose = purpose
        self.install_command = install_command


def validate_atoms_structure(atoms: Atoms) -> None:
    """Validate atoms structure for basic requirements.

    Parameters
    ----------
    atoms : Atoms
        Structure to validate

    Raises
    ------
    QMEValidationError
        If atoms structure is invalid
    """
    if atoms is None:
        raise QMEValidationError("No atoms provided. Use load_structure to load from file.")

    if len(atoms) == 0:
        raise QMEValidationError("Empty structure")

    # Check for overlapping atoms (basic check)
    positions = atoms.get_positions()
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            distance = np.linalg.norm(positions[i] - positions[j])
            if distance < 0.1:  # Very small distance threshold
                raise QMEValidationError(f"Atoms {i} and {j} are too close ({distance:.3f} Å)")

    # Check for invalid atomic numbers
    numbers = atoms.get_atomic_numbers()
    for i, num in enumerate(numbers):
        if num <= 0 or num > 118:  # Valid atomic numbers 1-118
            raise QMEValidationError(f"Invalid atomic numbers found at position {i}: {num}")


def validate_charge_and_spin(charge: int, spin: int, n_electrons: int | None = None) -> None:
    """Validate charge and spin parameters."""
    if not isinstance(charge, int):
        raise QMEValidationError(f"Charge must be integer, got {type(charge)}")
    if not isinstance(spin, int):
        raise QMEValidationError(f"Spin must be integer, got {type(spin)}")
    if spin < 0:
        raise QMEValidationError(f"Spin must be non-negative, got {spin}")

    # Check charge/spin parity
    if (charge % 2) == (spin % 2):
        if charge % 2 == 0:
            if spin == 0:
                raise QMEValidationError(f"Invalid spin multiplicity: spin must be ≥ 1, got {spin}")
            else:
                raise QMEValidationError(
                    f"Invalid spin multiplicity: even charge requires odd spin, "
                    f"got even charge={charge}, even spin={spin}"
                )
        else:
            raise QMEValidationError(
                f"Invalid spin multiplicity: odd charge requires even spin, "
                f"got odd charge={charge}, odd spin={spin}"
            )

    # Check electron count if provided
    if n_electrons is not None:
        if charge > n_electrons:
            raise QMEValidationError(
                f"Too many positive charges: Charge ({charge}) cannot exceed "
                f"number of electrons ({n_electrons})"
            )
        # Check if the number of unpaired electrons is consistent with spin
        unpaired_electrons = spin
        total_electrons = n_electrons - charge
        if total_electrons < unpaired_electrons:
            raise QMEValidationError(
                f"Invalid electron configuration: {n_electrons} electrons, "
                f"charge={charge}, spin={spin}"
            )


def validate_device_parameter(device: str, backend: str | None = None) -> None:
    """Validate device parameter."""
    if device is not None and device not in ["cpu", "cuda", "auto"]:
        raise QMEValidationError(
            f"Invalid device: Device must be 'cpu', 'cuda', or 'auto', got '{device}'"
        )
    if device == "cuda" and backend == "aimnet2":
        # Check if CUDA is actually available for AIMNet2
        try:
            import torch

            if not torch.cuda.is_available():
                raise QMEValidationError("CUDA requested but not available on this system")
        except ImportError as e:
            raise QMEValidationError("PyTorch not available to check CUDA availability") from e


def validate_file_exists(filepath: str) -> None:
    """Validate that file exists."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    if os.path.getsize(filepath) == 0:
        raise QMEValidationError(f"Empty file: {filepath}")


def validate_file_format(filepath: str, allowed_formats: list[str]) -> None:
    """Validate file format."""
    ext = Path(filepath).suffix.lower()
    if ext not in allowed_formats:
        raise QMEValidationError(f"Unsupported file format '{ext}'. Allowed: {allowed_formats}")


def validate_model_parameters(
    model_name: str | None = None,
    model_path: str | None = None,
    backend: str | None = None,
) -> None:
    """Validate model parameters."""
    # Backend-specific validation first
    if backend == "uma" and model_path is not None:
        raise QMEValidationError("UMA backend model_path is not supported. Use model_name instead")
    if backend == "aimnet2" and model_path is not None:
        raise QMEValidationError(
            "AIMNet2 backend model_path is not supported. Use model_name instead"
        )

    # General validation
    if model_name is not None and model_path is not None:
        raise QMEValidationError("Cannot specify both model_name and model_path")


def validate_optimization_parameters(fmax: float, steps: int, optimizer: str | None = None) -> None:
    """Validate optimization parameters."""
    if fmax <= 0:
        raise QMEValidationError(
            f"Invalid force convergence threshold: fmax must be positive, got {fmax}"
        )
    if steps <= 0:
        raise QMEValidationError(f"Invalid maximum steps: steps must be positive, got {steps}")
    if optimizer is not None:
        valid_optimizers = ["lbfgs", "bfgs", "sella", "tric", "fmin"]
        if optimizer.lower() not in valid_optimizers:
            raise QMEValidationError(
                f"Unknown optimizer '{optimizer}'. Valid optimizers: {valid_optimizers}"
            )


def validate_minima_run(atoms: Atoms | list[Atoms], backend: str, optimizer: str) -> None:
    """Validate inputs for minima optimization.

    Parameters
    ----------
    atoms : Atoms or List[Atoms]
        Structure(s) to optimize
    backend : str
        Calculator backend name
    optimizer : str
        Optimizer name

    Raises
    ------
    QMEValidationError
        If atoms are invalid or missing
    QMEBackendError
        If backend is incompatible
    """
    if not atoms:
        raise QMEValidationError("No atoms provided for optimization")

    if isinstance(atoms, list) and len(atoms) == 0:
        raise QMEValidationError("Empty atoms list provided")

    # Check if backend is available
    if not is_backend_available(backend):
        available = ", ".join(get_available_backends())
        raise QMEBackendError(
            f"Backend '{backend}' is not available. Available backends: {available}"
        )


def validate_ts_run(atoms: Atoms | list[Atoms], backend: str, optimizer: str) -> None:
    """Validate inputs for transition state optimization.

    Parameters
    ----------
    atoms : Atoms or List[Atoms]
        Structure(s) to optimize
    backend : str
        Calculator backend name
    optimizer : str
        Optimizer name

    Raises
    ------
    QMEValidationError
        If atoms are invalid or missing
    QMEBackendError
        If backend is incompatible with TS optimization
    """
    # Basic validation same as minima
    validate_minima_run(atoms, backend, optimizer)

    # TS-specific restrictions
    forbidden_backends_for_ts = {"mock"}
    forbidden_optimizers_for_ts = {"lbfgs", "l-bfgs", "l_bfgs", "bfgs", "fire"}

    if backend.lower() in forbidden_backends_for_ts:
        raise QMEBackendError(
            f"Backend '{backend}' is not suitable for transition state optimization. "
            f"Use a real ML potential backend (uma, aimnet2, mace, so3lr) instead."
        )

    if optimizer.lower() in forbidden_optimizers_for_ts:
        raise QMEBackendError(
            f"Optimizer '{optimizer}' is not suitable for transition state "
            f"optimization. Use 'sella' or 'tric' optimizers for TS searches."
        )


def validate_path_run(
    atoms: Atoms | list[Atoms], backend: str, optimizer: str, require_two_ended: bool = True
) -> None:
    """Validate inputs for path optimization (NEB/CI-NEB).

    Parameters
    ----------
    atoms : Atoms or List[Atoms]
        Structure(s) to optimize
    backend : str
        Calculator backend name
    optimizer : str
        Optimizer name
    require_two_ended : bool
        Whether to require exactly two structures for two-ended methods

    Raises
    ------
    QMEValidationError
        If atoms are invalid or missing
    QMEBackendError
        If backend is incompatible with path optimization
    """
    # Basic validation same as minima
    validate_minima_run(atoms, backend, optimizer)

    if require_two_ended:
        if not isinstance(atoms, (list, tuple)):
            raise QMEValidationError(
                "Path optimization requires two or more structures (reactant and product)"
            )

        if len(atoms) < 2:
            raise QMEValidationError(
                f"Path optimization requires at least 2 structures, got {len(atoms)}"
            )

        # Check that all structures have the same number of atoms
        n_atoms = len(atoms[0])
        for i, struct in enumerate(atoms):
            if len(struct) != n_atoms:
                raise QMEValidationError(
                    f"All structures must have the same number of atoms. "
                    f"Structure 0 has {n_atoms} atoms, structure {i} has {len(struct)} atoms"
                )


def validate_atoms_compatibility(atoms1: Atoms, atoms2: Atoms, context: str = "operation") -> None:
    """Validate that two Atoms objects are compatible for operations.

    Parameters
    ----------
    atoms1, atoms2 : Atoms
        The two structures to compare
    context : str
        Context for error messages

    Raises
    ------
    QMEValidationError
        If atoms are incompatible
    """
    if len(atoms1) != len(atoms2):
        raise QMEValidationError(
            f"Incompatible atoms for {context}: different number of atoms "
            f"({len(atoms1)} vs {len(atoms2)})"
        )

    if list(atoms1.symbols) != list(atoms2.symbols):
        raise QMEValidationError(
            f"Incompatible atoms for {context}: different atomic symbols "
            f"({list(atoms1.symbols)} vs {list(atoms2.symbols)})"
        )


def validate_backend_compatibility(backend: str, operation: str) -> None:
    """Validate that backend supports the requested operation.

    Parameters
    ----------
    backend : str
        Backend name
    operation : str
        Operation type ("minima", "ts", "path")

    Raises
    ------
    QMEBackendError
        If backend is incompatible with operation
    """
    if not is_backend_available(backend):
        available = ", ".join(get_available_backends())
        raise QMEBackendError(
            f"Backend '{backend}' is not available for {operation} optimization. "
            f"Available backends: {available}"
        )

    # TS-specific restrictions
    if operation == "ts" and backend.lower() == "mock":
        raise QMEBackendError(
            f"Backend '{backend}' cannot be used for transition state optimization. "
            f"Use a real ML potential backend (uma, aimnet2, mace, so3lr) instead."
        )
