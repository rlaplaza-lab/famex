"""Shared physical constants and frequency helpers for analysis."""

from __future__ import annotations

import numpy as np
from ase import units
from numpy.typing import NDArray

GAS_CONSTANT = units.kB * units._Nav / units.J  # J/(mol·K)
PLANCK_CONSTANT = 6.62606957e-34  # J·s
HBAR = PLANCK_CONSTANT / (2 * np.pi)
BOLTZMANN_CONSTANT = units.kB / units.J  # J/K
AVOGADRO_CONSTANT = units._Nav
AMU_TO_KG = 1.66053886e-27  # kg/amu
SPEED_OF_LIGHT = 2.99792458e10  # cm/s
J_PER_MOL_TO_EV = units.J / units.mol


def filter_positive_frequencies(frequencies: NDArray[np.floating]) -> NDArray[np.float64]:
    """Return real, strictly positive vibrational frequencies in cm^-1."""
    signed = normalize_frequencies_cm1(frequencies)
    positive = signed[signed > 0]
    return np.asarray(positive, dtype=np.float64)


def normalize_frequencies_cm1(
    frequencies: NDArray[np.complexfloating] | NDArray[np.floating],
) -> NDArray[np.float64]:
    """Convert harmonic frequencies to signed real values in cm^-1.

    ASE represents imaginary normal modes as complex frequencies with a
    predominantly imaginary component (e.g. ``0 + 774j`` cm^-1) when the
    mass-weighted Hessian has negative eigenvalues. FAMEX follows ASE's
    export convention: imaginary modes are reported as negative real
    frequencies.

    Parameters
    ----------
    frequencies
        Frequencies in cm^-1 from ASE or another backend. May be real or complex.

    Returns
    -------
    np.ndarray
        Signed real frequencies in cm^-1 (negative for imaginary modes).
    """
    values = np.asarray(frequencies)
    if not np.iscomplexobj(values):
        return np.asarray(values, dtype=np.float64)

    real = np.real(values)
    imag = np.imag(values)
    signed = np.where(np.abs(imag) > np.abs(real), -np.abs(imag), real)
    return np.asarray(signed, dtype=np.float64)


__all__ = [
    "AMU_TO_KG",
    "AVOGADRO_CONSTANT",
    "BOLTZMANN_CONSTANT",
    "GAS_CONSTANT",
    "HBAR",
    "J_PER_MOL_TO_EV",
    "PLANCK_CONSTANT",
    "SPEED_OF_LIGHT",
    "filter_positive_frequencies",
    "normalize_frequencies_cm1",
]
