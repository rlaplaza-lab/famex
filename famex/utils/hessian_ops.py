"""Small Hessian helpers shared across optimizers and analysis."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def symmetrize_hessian(hessian: NDArray[np.floating]) -> NDArray[np.float64]:
    """Return (H + Hᵀ) / 2."""
    return np.asarray(0.5 * (hessian + hessian.T), dtype=np.float64)
