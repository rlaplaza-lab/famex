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
    real_frequencies = np.real(frequencies)
    positive = real_frequencies[real_frequencies > 0]
    return np.asarray(positive, dtype=np.float64)


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
]
