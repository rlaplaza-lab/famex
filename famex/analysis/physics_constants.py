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
# amu·Å² → kg·m² (moments of inertia)
AMU_ANGSTROM2_TO_KG_M2 = AMU_TO_KG * 1e-20
MOMENT_OF_INERTIA_FLOOR_KG_M2 = 1e-4 * AMU_ANGSTROM2_TO_KG_M2
# Rotational constants: 1 GHz → cm⁻¹
GHZ_TO_CM1 = 1e9 / SPEED_OF_LIGHT
# Imaginary-mode cutoff shared by frequency analysis, cleanup, and mode I/O
IMAGINARY_MODE_THRESHOLD_CM = 50.0
J_PER_MOL_TO_EV = units.J / units.mol
ATMOS = 101.325  # kPa; GoodVibes 1 atm concentration is ATMOS/(R T) mol/L
ATMOS_PA = ATMOS * 1000.0  # 1 atm in pascal


def ideal_gas_concentration(temperature: float, pressure: float = ATMOS_PA) -> float:
    """Ideal-gas concentration in mol/L.

    ``pressure`` is in pascal. Dividing by 1000 converts it to kPa so that
    ``P / (R T)`` has units of mol/L when ``R`` is in J/(mol·K).
    """
    return (pressure / 1000.0) / (GAS_CONSTANT * temperature)


def gas_phase_concentration_1atm(temperature: float) -> float:
    """GoodVibes 1 atm ideal-gas concentration in mol/L."""
    return ideal_gas_concentration(temperature)


def filter_positive_frequencies(frequencies: NDArray[np.floating]) -> NDArray[np.float64]:
    """Return real, strictly positive vibrational frequencies in cm⁻¹."""
    signed = normalize_frequencies_cm1(frequencies)
    positive = signed[signed > 0]
    return np.asarray(positive, dtype=np.float64)


def normalize_frequencies_cm1(
    frequencies: NDArray[np.complexfloating] | NDArray[np.floating],
) -> NDArray[np.float64]:
    """Convert harmonic frequencies to signed real values in cm⁻¹.

    ASE represents imaginary normal modes as complex frequencies with a
    predominantly imaginary component (e.g. ``0 + 774j`` cm⁻¹) when the
    mass-weighted Hessian has negative eigenvalues. FAMEX follows ASE's
    export convention: imaginary modes are reported as negative real
    frequencies.

    Parameters
    ----------
    frequencies
        Frequencies in cm⁻¹ from ASE or another backend. May be real or complex.

    Returns
    -------
    np.ndarray
        Signed real frequencies in cm⁻¹ (negative for imaginary modes).
    """
    values = np.asarray(frequencies)
    if not np.iscomplexobj(values):
        return np.asarray(values, dtype=np.float64)

    real = np.real(values)
    imag = np.imag(values)
    signed = np.where(np.abs(imag) > np.abs(real), -np.abs(imag), real)
    return np.asarray(signed, dtype=np.float64)


__all__ = [
    "AMU_ANGSTROM2_TO_KG_M2",
    "AMU_TO_KG",
    "ATMOS",
    "ATMOS_PA",
    "AVOGADRO_CONSTANT",
    "BOLTZMANN_CONSTANT",
    "GAS_CONSTANT",
    "GHZ_TO_CM1",
    "HBAR",
    "IMAGINARY_MODE_THRESHOLD_CM",
    "J_PER_MOL_TO_EV",
    "MOMENT_OF_INERTIA_FLOOR_KG_M2",
    "PLANCK_CONSTANT",
    "SPEED_OF_LIGHT",
    "filter_positive_frequencies",
    "gas_phase_concentration_1atm",
    "ideal_gas_concentration",
    "normalize_frequencies_cm1",
]
