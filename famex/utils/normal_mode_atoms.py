"""Helpers for mapping normal modes to atom indices and fixed atoms."""

from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms


def fixed_atom_indices(atoms: Atoms) -> np.ndarray:
    """Atom indices held fixed by any ``FixAtoms`` constraint on ``atoms``."""
    indices: list[int] = []
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            indices.extend(int(index) for index in constraint.get_indices())
    return np.array(sorted(set(indices)), dtype=int)


def mode_atom_indices(
    n_atoms: int,
    normal_modes: np.ndarray,
    frequency_result: dict[str, Any],
) -> list[int] | None:
    """Map normal-mode rows to atom indices, including partial Hessian analyses."""
    if normal_modes.shape[0] == 3 * n_atoms:
        return list(range(n_atoms))

    indices = frequency_result.get("indices")
    if indices is not None and normal_modes.shape[0] == 3 * len(indices):
        return [int(index) for index in indices]
    return None
