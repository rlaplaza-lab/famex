"""Normal mode calculations for frequency analysis.

This module provides functions for diagonalizing Hessians, computing normal
modes, and converting between frequency units.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms, units
from scipy.linalg import eigh

from qme.utils.logging import get_qme_logger

logger = get_qme_logger(__name__)

__all__ = ["diagonalize_mass_weighted_hessian", "convert_frequency_unit"]


def diagonalize_mass_weighted_hessian(
    hessian: np.ndarray,
    atoms: Atoms,
    indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize mass-weighted Hessian to get normal modes and frequencies.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix (3N x 3N for N atoms)
    atoms : Atoms
        ASE Atoms object
    indices : list[int]
        Indices of atoms included in Hessian

    Returns:
    -------
    tuple[np.ndarray, np.ndarray]
        Frequencies in cm^-1 and normal mode eigenvectors

    """
    # Mass-weight the Hessian
    masses = atoms.get_masses()[indices]
    if np.any(masses == 0):
        msg = "Zero mass encountered. Use Atoms.set_masses() to set all masses to non-zero values."
        logger.error(msg)
        raise ValueError(msg)

    mass_sqrt = np.repeat(np.sqrt(masses), 3)
    mass_inv_sqrt = mass_sqrt**-1
    mass_weighted_hessian = hessian * np.outer(mass_inv_sqrt, mass_inv_sqrt)

    # Diagonalize mass-weighted Hessian
    omega2, mw_modes = eigh(mass_weighted_hessian)

    # ASE conversion factor: h*nu (eV) from omega^2 (in mass-weighted units)
    # s = ħ * 1e10 / sqrt(e * amu)
    s = units._hbar * 1e10 / np.sqrt(units._e * units._amu)
    hnu = s * np.sqrt(omega2.astype(complex))  # eV (complex for negative omega2)

    # Convert to frequencies: use absolute value to get magnitude
    # For negative omega2, sqrt gives imaginary numbers - we want the magnitude
    frequencies = np.abs(hnu / units.invcm)  # cm^-1 (magnitude)

    # Mark imaginary frequencies as negative (omega2 < 0 means saddle point)
    frequencies[omega2 < 0] *= -1

    # Convert eigenvectors back to Cartesian coordinates
    cart_modes = mw_modes / mass_sqrt[:, None]

    # Normalize Cartesian mode vectors column-wise
    for i in range(cart_modes.shape[1]):
        norm = np.linalg.norm(cart_modes[:, i])
        if norm > 0:
            cart_modes[:, i] /= norm

    return frequencies, cart_modes


def convert_frequency_unit(frequencies: np.ndarray, unit: str) -> np.ndarray:
    """Convert frequencies between different units.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequencies in cm^-1
    unit : str
        Target unit: 'cm-1', 'meV', or 'THz'

    Returns:
    -------
    np.ndarray
        Frequencies in requested unit

    Raises:
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
