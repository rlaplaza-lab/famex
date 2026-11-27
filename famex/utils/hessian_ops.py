"""Small Hessian helpers shared across optimizers and analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


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
