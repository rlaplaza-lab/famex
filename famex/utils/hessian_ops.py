"""Hessian helpers shared across optimizers and analysis."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from ase import Atoms
from numpy.typing import NDArray

MAX_CONDITION_NUMBER = 1e18


def optional_int(value: Any) -> int | None:
    """Coerce ``value`` to ``int``, or return ``None`` if that fails."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def bond_key(i: int, j: int) -> tuple[int, int]:
    """Return a canonical unordered bond key ``(min, max)``."""
    return (i, j) if i < j else (j, i)


def symmetrize_hessian(hessian: NDArray[np.floating]) -> NDArray[np.float64]:
    """Return (H + Hᵀ) / 2."""
    return np.asarray(0.5 * (hessian + hessian.T), dtype=np.float64)


def coerce_square_hessian_shape(hessian: np.ndarray, n_atoms: int) -> np.ndarray:
    """Reshape ``hessian`` to ``(3N, 3N)`` when the element count matches."""
    expected_shape = (3 * n_atoms, 3 * n_atoms)
    if hessian.shape == expected_shape:
        return hessian
    if hessian.size == expected_shape[0] * expected_shape[1]:
        return hessian.reshape(expected_shape)
    msg = f"Hessian has unexpected shape {hessian.shape}, expected {expected_shape}"
    raise ValueError(msg)


def autograd_hessian_rows_from_forces(forces: Any, positions: Any) -> list[Any]:
    """Build Hessian rows from force components via ``torch.autograd.grad``."""
    import torch

    return [
        (-torch.autograd.grad(component, positions, retain_graph=True)[0]).reshape(-1)
        for component in forces.reshape(-1)
    ]


def build_translation_rotation_basis(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Build an orthonormal Cartesian translation/rotation basis."""
    n_atoms = positions.shape[0]
    n_coords = 3 * n_atoms
    if n_atoms == 0:
        return np.zeros((0, 0))

    total_mass = float(np.sum(masses))
    com = np.sum(positions * masses[:, None], axis=0) / total_mass
    rel = positions - com

    modes: list[np.ndarray] = []

    for axis in range(3):
        mode = np.zeros(n_coords)
        mode[axis::3] = 1.0
        norm = np.linalg.norm(mode)
        if norm > 1e-12:
            modes.append(mode / norm)

    if n_atoms >= 2:
        for axis in range(3):
            rot = np.zeros(n_coords)
            axis_vec = np.zeros(3)
            axis_vec[axis] = 1.0
            for i in range(n_atoms):
                disp = np.cross(axis_vec, rel[i])
                rot[3 * i : 3 * i + 3] = disp
            norm = np.linalg.norm(rot)
            if norm > 1e-12:
                modes.append(rot / norm)

    if not modes:
        return np.zeros((n_coords, 0))

    basis = np.column_stack(modes)
    q, _ = np.linalg.qr(basis)
    return q


def project_gradient(gradient: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project gradient onto subspace orthogonal to translation/rotation."""
    if basis.size == 0:
        return gradient
    return cast(np.ndarray, gradient - basis @ (basis.T @ gradient))


def project_hessian(hessian: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project Hessian onto subspace orthogonal to translation/rotation."""
    if basis.size == 0:
        return hessian
    projector = np.eye(hessian.shape[0]) - basis @ basis.T
    return cast(np.ndarray, projector @ hessian @ projector)


def condition_number_and_vibrational_eigenvalues(
    hessian: np.ndarray,
    positions: np.ndarray | None = None,
    masses: np.ndarray | None = None,
) -> tuple[float, np.ndarray, int]:
    """Return condition number, vibrational eigenvalues, and null-mode count.

    When ``positions`` and ``masses`` are provided for a matching 3N×3N Hessian,
    translations/rotations are projected out before the vibrational block is
    used for the condition number.
    """
    matrix = symmetrize_hessian(hessian)
    n_null = 0
    if positions is not None and masses is not None:
        positions_arr = np.asarray(positions, dtype=np.float64)
        masses_arr = np.asarray(masses, dtype=np.float64)
        n_atoms = positions_arr.shape[0]
        if (
            positions_arr.ndim == 2
            and positions_arr.shape[1] == 3
            and masses_arr.shape == (n_atoms,)
            and matrix.shape == (3 * n_atoms, 3 * n_atoms)
        ):
            basis = build_translation_rotation_basis(positions_arr, masses_arr)
            n_null = int(basis.shape[1]) if basis.size else 0
            matrix = project_hessian(matrix, basis)

    if matrix.shape[0] <= n_null:
        return 0.0, np.asarray([], dtype=np.float64), n_null

    eigenvalues = np.linalg.eigvalsh(matrix)
    eigenvalues = eigenvalues[np.isfinite(eigenvalues)]
    if n_null > 0 and len(eigenvalues) > n_null:
        vibrational = eigenvalues[np.argsort(np.abs(eigenvalues))[n_null:]]
    else:
        vibrational = eigenvalues

    eigenvalues_abs = np.abs(vibrational[vibrational != 0])
    if len(eigenvalues_abs) == 0:
        return float("inf"), vibrational, n_null
    condition_number = float(np.max(eigenvalues_abs) / np.min(eigenvalues_abs))
    return condition_number, vibrational, n_null


def frequencies_from_hessian(
    hessian: np.ndarray,
    atoms: Atoms,
    indices: list[int] | None = None,
    *,
    project_tr: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Project (optional) and diagonalize a Hessian into frequencies and modes."""
    # Deferred to avoid analysis ↔ utils import cycles at module load.
    from famex.analysis.normal_modes import diagonalize_mass_weighted_hessian
    from famex.analysis.physics_constants import normalize_frequencies_cm1

    if indices is None:
        indices = list(range(len(atoms)))

    working = np.asarray(hessian, dtype=np.float64)
    if project_tr:
        positions = atoms.get_positions()[indices]
        masses = atoms.get_masses()[indices]
        basis = build_translation_rotation_basis(positions, masses)
        working = project_hessian(working, basis)

    frequencies, eigenvectors = diagonalize_mass_weighted_hessian(working, atoms, indices)
    return normalize_frequencies_cm1(frequencies), eigenvectors
