"""Normal mode calculations for frequency analysis.

This module provides functions for diagonalizing Hessians, computing normal
modes, and converting between frequency units.

Uses ASE's VibrationsData for consistency with ASE conventions.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms, units
from numpy.typing import NDArray

from famex.analysis.ase_integration import frequencies_via_ase
from famex.utils.logging import get_famex_logger

logger = get_famex_logger(__name__)

__all__ = ["diagonalize_mass_weighted_hessian", "convert_frequency_unit"]


def diagonalize_mass_weighted_hessian(
    hessian: np.ndarray,
    atoms: Atoms,
    indices: list[int],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Diagonalize mass-weighted Hessian to get normal modes and frequencies.

    This function uses ASE's VibrationsData to ensure consistency with ASE's
    conventions for frequency and mode calculations.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix (3N x 3N for N atoms)
    atoms : Atoms
        ASE Atoms object
    indices : list[int]
        Indices of atoms included in Hessian

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        Frequencies in cm^-1 and normal mode eigenvectors (Cartesian coordinates).
        Frequencies are signed: positive for real modes, negative for imaginary.

    Notes
    -----
    - Uses ASE's VibrationsData.from_2d() for consistency
    - Frequencies follow ASE's convention (signed, cm^-1)
    - Normal modes are in Cartesian coordinates, normalized

    """
    # Use ASE's VibrationsData for consistency
    frequencies, modes = frequencies_via_ase(hessian, atoms, indices)

    return frequencies, modes


def convert_frequency_unit(frequencies: np.ndarray, unit: str) -> np.ndarray:
    """Convert frequencies between different units.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequencies in cm^-1
    unit : str
        Target unit: 'cm-1', 'meV', or 'THz'

    Returns
    -------
    np.ndarray
        Frequencies in requested unit

    Raises
    ------
    ValueError
        If unit is not recognized

    """
    if unit == "cm-1":
        return frequencies
    if unit == "meV":
        # Convert cm^-1 to meV using ASE units
        return frequencies * units.invcm * 1000  # Convert to meV
    if unit == "THz":
        # Convert cm^-1 to THz using ASE units
        return frequencies * units._c * 100 / 1e12
    msg = f"Unknown frequency unit: {unit}"
    logger.error("%s (supported units: cm-1, meV, THz)", msg)
    raise ValueError(msg)
