"""Molecular geometry property utilities for frequency analysis.

This module provides utility functions for determining molecular properties
such as linearity and degrees of freedom, which are needed for proper
frequency analysis.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

__all__ = ["determine_degrees_of_freedom", "is_linear_molecule"]


def is_linear_molecule(atoms: Atoms, indices: list[int]) -> bool:
    """Check if molecule is linear based on geometry.

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object
    indices : list[int]
        Indices of atoms to consider

    Returns:
    -------
    bool
        True if molecule is linear, False otherwise

    """
    if len(indices) <= 2:
        return True

    positions = atoms.positions[indices]
    if len(positions) == 3:
        # For 3 atoms, check if they are collinear
        v1 = positions[1] - positions[0]
        v2 = positions[2] - positions[0]
        cross = np.cross(v1, v2)
        return bool(np.linalg.norm(cross) < 1e-3)

    # For more atoms, use moment of inertia approach
    return _check_linearity_inertia(atoms, indices)


def _check_linearity_inertia(atoms: Atoms, indices: list[int]) -> bool:
    """Check linearity using moment of inertia tensor.

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object
    indices : list[int]
        Indices of atoms to consider

    Returns:
    -------
    bool
        True if molecule is linear (smallest eigenvalue of inertia tensor is near zero)

    """
    atoms_subset = atoms[indices]
    masses = atoms_subset.get_masses()
    positions = atoms_subset.positions

    # Center of mass
    com = atoms_subset.get_center_of_mass()
    positions_centered = positions - com

    # Moment of inertia tensor
    inertia_tensor = np.zeros((3, 3))
    for _i, (pos, mass) in enumerate(zip(positions_centered, masses, strict=False)):
        inertia_tensor += mass * (np.dot(pos, pos) * np.eye(3) - np.outer(pos, pos))

    # Eigenvalues of moment of inertia tensor
    eigenvalues = np.linalg.eigvals(inertia_tensor)
    eigenvalues = np.sort(eigenvalues)

    # Linear if smallest eigenvalue is essentially zero
    return bool(eigenvalues[0] < 1e-6)


def determine_degrees_of_freedom(atoms: Atoms, indices: list[int]) -> int:
    """Determine number of degrees of freedom to remove (translation + rotation).

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object
    indices : list[int]
        Indices of atoms to consider

    Returns:
    -------
    int
        Number of degrees of freedom to remove:
        - 3 for single atom (translation only)
        - 5 for 2-atom or linear molecules (3 translation + 2 rotation)
        - 6 for non-linear molecules (3 translation + 3 rotation)

    """
    if len(indices) == 1:
        return 3  # Only translation for single atom
    if len(indices) == 2:
        return 5  # 3 translation + 2 rotation for 2-atom molecules (always linear)
    if is_linear_molecule(atoms, indices):
        return 5  # 3 translation + 2 rotation for linear molecules
    return 6  # 3 translation + 3 rotation for non-linear molecules
