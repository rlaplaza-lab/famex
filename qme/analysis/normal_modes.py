"""Normal mode calculations for frequency analysis.

This module provides functions for diagonalizing Hessians, computing normal
modes, and converting between frequency units.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms, units
from scipy.linalg import eigh

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
    mass_weights = np.repeat(masses, 3) ** -0.5
    mass_weighted_hessian = hessian * np.outer(mass_weights, mass_weights)

    # Diagonalize
    eigenvalues, eigenvectors = eigh(mass_weighted_hessian)

    # Convert eigenvalues to frequencies (in cm^-1)
    # Following ASE's implementation:
    # 1. Convert eigenvalues to energies in eV using ASE's unit conversion
    # 2. Convert energies to frequencies using units.invcm
    unit_conversion = units._hbar * units.m / np.sqrt(units._e * units._amu)
    energies = unit_conversion * np.sqrt(np.abs(eigenvalues))
    frequencies = energies / units.invcm

    # Handle imaginary frequencies
    imaginary_mask = eigenvalues < 0
    frequencies[imaginary_mask] *= -1  # Make imaginary frequencies negative

    # Sort by frequency (most negative first)
    sort_indices = np.argsort(frequencies)
    frequencies = frequencies[sort_indices]
    eigenvectors = eigenvectors[:, sort_indices]

    # Normalize eigenvectors in mass-weighted coordinates
    # The eigenvectors are already in mass-weighted coordinates from diagonalization
    for i in range(len(eigenvectors[0])):
        eigenvectors[:, i] /= np.linalg.norm(eigenvectors[:, i])

    return frequencies, eigenvectors


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
    raise ValueError(msg)
