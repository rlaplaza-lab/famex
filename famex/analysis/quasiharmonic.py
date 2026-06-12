"""Quasi-harmonic corrections for low-frequency vibrational modes.

This module implements Grimme's and Truhlar's methods for correcting
harmonic oscillator entropies and energies for low-frequency modes that
behave more like free rotors at typical reaction temperatures.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
from ase import units
from numpy.typing import NDArray

from famex.utils.logging import get_famex_logger

logger = get_famex_logger(__name__)

__all__ = [
    "QuasiHarmonicHandler",
    "calculate_damping_function",
    "calculate_rrho_entropy",
    "calculate_rrho_energy",
    "calculate_free_rotor_entropy",
    "calculate_qRRHO_entropy",
    "calculate_qRRHO_energy",
]

# Physical constants from ASE units where available
GAS_CONSTANT = units.kB * units._Nav / units.J  # J/(mol·K) = R (gas constant)
PLANCK_CONSTANT = 6.62606957e-34  # J·s (not in ASE, keep manual)
BOLTZMANN_CONSTANT = units.kB / units.J  # J/K = kB in SI units
SPEED_OF_LIGHT = 2.99792458e10  # cm/s (not in ASE, keep manual)


def calculate_damping_function(
    frequencies: np.ndarray, freq_cutoff: float, alpha: float = 4
) -> np.ndarray:
    """Calculate damping function for interpolating RRHO and free rotor entropies.

    Damping function d = 1 / (1 + (freq_cutoff / freq)^alpha)
    - d ≈ 1 for high frequencies (RRHO behavior)
    - d ≈ 0 for low frequencies (free rotor behavior)

    Parameters
    ----------
    frequencies : np.ndarray
        Vibrational frequencies in cm⁻¹
    freq_cutoff : float
        Cutoff frequency in cm⁻¹
    alpha : float
        Damping parameter (default: 4)

    Returns
    -------
    np.ndarray
        Damping factors (0 to 1)
    """
    # Avoid division by zero for zero frequencies
    # Handle complex frequencies by taking real part for comparison
    frequencies_safe = np.where(np.real(frequencies) > 0, np.real(frequencies), 1e-10)
    damp = 1.0 / (1.0 + (freq_cutoff / frequencies_safe) ** alpha)
    return damp


def calculate_rrho_entropy(
    frequencies: np.ndarray,
    temperature: float,
    freq_scale_factor: float = 1.0,
) -> np.ndarray:
    """Calculate rigid rotor harmonic oscillator (RRHO) entropy per mode.

    Sv = R [hv/(kT(e^(hv/kT)-1)) - ln(1-e^(-hv/kT))]

    Parameters
    ----------
    frequencies : np.ndarray
        Vibrational frequencies in cm⁻¹
    temperature : float
        Temperature in Kelvin
    freq_scale_factor : float
        Frequency scaling factor (default: 1.0)

    Returns
    -------
    np.ndarray
        RRHO entropy in J/(mol·K) for each mode
    """
    # Scale frequencies
    frequencies_scaled = frequencies * freq_scale_factor

    # Calculate x = hν/(kT)
    factor = (
        PLANCK_CONSTANT * frequencies_scaled * SPEED_OF_LIGHT / (BOLTZMANN_CONSTANT * temperature)
    )

    # Sv = R [x/(exp(x)-1) - ln(1-exp(-x))]
    entropy = np.where(
        factor < 50,  # Avoid overflow
        factor * GAS_CONSTANT / (np.exp(factor) - 1) - GAS_CONSTANT * np.log(1 - np.exp(-factor)),
        GAS_CONSTANT * factor / (np.exp(factor) - 1),  # High frequency limit
    )

    return entropy


def calculate_rrho_energy(
    frequencies: np.ndarray,
    temperature: float,
    freq_scale_factor: float = 1.0,
) -> np.ndarray:
    """Calculate rigid rotor harmonic oscillator (RRHO) energy per mode.

    Evib = RT [0.5 + 1/(e^(hν/kT)-1)]

    Parameters
    ----------
    frequencies : np.ndarray
        Vibrational frequencies in cm⁻¹
    temperature : float
        Temperature in Kelvin
    freq_scale_factor : float
        Frequency scaling factor (default: 1.0)

    Returns
    -------
    np.ndarray
        RRHO energy in J/mol for each mode
    """
    # Scale frequencies
    frequencies_scaled = frequencies * freq_scale_factor

    # Calculate x = hν/(kT)
    factor = (
        PLANCK_CONSTANT * frequencies_scaled * SPEED_OF_LIGHT / (BOLTZMANN_CONSTANT * temperature)
    )

    # Evib = RT [0.5 + 1/(exp(x)-1)]
    energy = np.where(
        factor < 50,  # Avoid overflow
        factor * GAS_CONSTANT * temperature * (0.5 + 1.0 / (np.exp(factor) - 1.0)),
        factor * GAS_CONSTANT * temperature * 0.5,  # High frequency limit
    )

    return energy


def calculate_free_rotor_entropy(
    frequencies: np.ndarray,
    temperature: float,
    freq_scale_factor: float = 1.0,
    average_moment_of_inertia: float = 1.0e-44,
) -> np.ndarray:
    """Calculate free rotor entropy per mode.

    Sr = R [1/2 + 1/2 ln(8π³μ'kT/h²)]
    where μ' is the reduced mass μb/(μ+b) with b the average moment of inertia

    Parameters
    ----------
    frequencies : np.ndarray
        Vibrational frequencies in cm⁻¹
    temperature : float
        Temperature in Kelvin
    freq_scale_factor : float
        Frequency scaling factor (default: 1.0)
    average_moment_of_inertia : float
        Average moment of inertia in kg·m² (default: 1e-44 from Grimme)

    Returns
    -------
    np.ndarray
        Free rotor entropy in J/(mol·K) for each mode
    """
    # Scale frequencies
    frequencies_scaled = frequencies * freq_scale_factor

    # Calculate reduced mass μ = h/(8π²ν)
    mu = PLANCK_CONSTANT / (8 * math.pi**2 * frequencies_scaled * SPEED_OF_LIGHT)

    # Calculate modified reduced mass μ' = μb/(μ+b)
    mu_primed = mu * average_moment_of_inertia / (mu + average_moment_of_inertia)

    # Calculate entropy Sr = R [1/2 + 1/2 ln(8π³μ'kT/h²)]
    factor = 8 * math.pi**3 * mu_primed * BOLTZMANN_CONSTANT * temperature / PLANCK_CONSTANT**2
    entropy = (0.5 + np.log(np.sqrt(factor))) * GAS_CONSTANT

    return cast(NDArray[np.float64], entropy)


def calculate_qRRHO_entropy(  # noqa: N802
    frequencies: np.ndarray,
    temperature: float,
    freq_cutoff: float,
    freq_scale_factor: float = 1.0,
) -> np.ndarray:
    """Calculate Truhlar's quasi-RRHO entropy.

    For frequencies > freq_cutoff: use RRHO entropy
    For frequencies ≤ freq_cutoff: use quasi-RRHO entropy with freq_cutoff

    Parameters
    ----------
    frequencies : np.ndarray
        Vibrational frequencies in cm⁻¹
    temperature : float
        Temperature in Kelvin
    freq_cutoff : float
        Cutoff frequency in cm⁻¹
    freq_scale_factor : float
        Frequency scaling factor (default: 1.0)

    Returns
    -------
    np.ndarray
        Quasi-RRHO entropy in J/(mol·K) for each mode
    """
    # Calculate full RRHO entropy
    entropy_full = calculate_rrho_entropy(frequencies, temperature, freq_scale_factor)

    # For modes below cutoff, use cutoff frequency in RRHO calculation
    entropy_cutoff = calculate_rrho_entropy(
        np.full_like(frequencies, freq_cutoff), temperature, freq_scale_factor
    )

    # Use cutoff entropy for frequencies below cutoff
    entropy = np.where(frequencies > freq_cutoff, entropy_full, entropy_cutoff)

    return entropy


def calculate_qRRHO_energy(  # noqa: N802
    frequencies: np.ndarray,
    temperature: float,
    freq_cutoff: float,
    freq_scale_factor: float = 1.0,
) -> np.ndarray:
    """Calculate Head-Gordon quasi-RRHO energy.

    E_qRRHO = 1/2 Nhν + RT (hν/kT) e^(-hν/kT) / (1 - e^(-hν/kT))

    Parameters
    ----------
    frequencies : np.ndarray
        Vibrational frequencies in cm⁻¹
    temperature : float
        Temperature in Kelvin
    freq_cutoff : float
        Cutoff frequency in cm⁻¹ (not used in qRRHO energy, for API consistency)
    freq_scale_factor : float
        Frequency scaling factor (default: 1.0)

    Returns
    -------
    np.ndarray
        Quasi-RRHO energy in J/mol for each mode

    Notes
    -----
    The freq_cutoff parameter is included for API consistency but is not
    used in this qRRHO energy formulation.
    """
    # Scale frequencies
    frequencies_scaled = frequencies * freq_scale_factor

    # Convert to energy
    hnu = PLANCK_CONSTANT * frequencies_scaled * SPEED_OF_LIGHT  # J/mol equivalent
    kT = BOLTZMANN_CONSTANT * temperature

    # E_qRRHO = 0.5 Nhν + RT (hν/kT) e^(-hν/kT) / (1 - e^(-hν/kT))
    factor = hnu / kT
    energy = np.where(
        factor < 50,
        0.5 * hnu + GAS_CONSTANT * temperature * factor * np.exp(-factor) / (1 - np.exp(-factor)),
        0.5 * hnu,  # High frequency limit
    )

    return energy


class QuasiHarmonicHandler:
    """Handles quasi-harmonic corrections for vibrational thermodynamics."""

    def __init__(
        self,
        method: str = "rrho",
        freq_cutoff: float = 100.0,
        freq_scale_factor: float = 1.0,
        alpha: float = 4.0,
        average_moment_of_inertia: float = 1.0e-44,
    ):
        """Initialize quasi-harmonic handler.

        Parameters
        ----------
        method : str
            Method to use: 'rrho', 'grimme', or 'truhlar'
        freq_cutoff : float
            Cutoff frequency in cm⁻¹ for low-frequency corrections
        freq_scale_factor : float
            Frequency scaling factor (default: 1.0)
        alpha : float
            Damping parameter for Grimme method (default: 4.0)
        average_moment_of_inertia : float
            Average moment of inertia for free rotor calculations (default: 1e-44)
        """
        if method not in ["rrho", "grimme", "truhlar"]:
            msg = f"Unknown method: {method}. Must be 'rrho', 'grimme', or 'truhlar'"
            logger.error(msg)
            raise ValueError(msg)

        self.method = method
        self.freq_cutoff = freq_cutoff
        self.freq_scale_factor = freq_scale_factor
        self.alpha = alpha
        self.average_moment_of_inertia = average_moment_of_inertia

    def vibrational_entropy(
        self,
        frequencies: np.ndarray,
        temperature: float,
    ) -> tuple[float, np.ndarray]:
        """Calculate vibrational entropy with quasi-harmonic corrections.

        Parameters
        ----------
        frequencies : np.ndarray
            Vibrational frequencies in cm⁻¹
        temperature : float
            Temperature in Kelvin

        Returns
        -------
        tuple[float, np.ndarray]
            Total entropy and per-mode entropies in J/(mol·K)
        """
        # Remove zero or negative frequencies
        # Handle complex frequencies (imaginary frequencies for TS)
        # Take real part and filter out non-positive values
        real_freqs = frequencies[np.real(frequencies) > 0]
        # Ensure we have real values (take real part if complex)
        real_freqs = np.real(real_freqs)

        if self.method == "rrho":
            per_mode_entropy = calculate_rrho_entropy(
                real_freqs,
                temperature,
                self.freq_scale_factor,
            )
        elif self.method == "grimme":
            # Grimme's method: interpolate between RRHO and free rotor
            s_rrho = calculate_rrho_entropy(
                real_freqs,
                temperature,
                self.freq_scale_factor,
            )
            s_free_rot = calculate_free_rotor_entropy(
                real_freqs,
                temperature,
                self.freq_scale_factor,
                self.average_moment_of_inertia,
            )
            damp = calculate_damping_function(real_freqs, self.freq_cutoff, self.alpha)
            per_mode_entropy = s_rrho * damp + (1 - damp) * s_free_rot
        elif self.method == "truhlar":
            # Truhlar's method: use quasi-RRHO below cutoff
            per_mode_entropy = calculate_qRRHO_entropy(
                real_freqs,
                temperature,
                self.freq_cutoff,
                self.freq_scale_factor,
            )
        else:
            msg = f"Unknown method: {self.method}"
            logger.error(msg)
            raise RuntimeError(msg)

        total_entropy = float(np.sum(per_mode_entropy))

        return total_entropy, per_mode_entropy

    def vibrational_energy(
        self,
        frequencies: np.ndarray,
        temperature: float,
    ) -> tuple[float, np.ndarray]:
        """Calculate vibrational energy with quasi-harmonic corrections.

        Parameters
        ----------
        frequencies : np.ndarray
            Vibrational frequencies in cm⁻¹
        temperature : float
            Temperature in Kelvin

        Returns
        -------
        tuple[float, np.ndarray]
            Total energy and per-mode energies in J/mol
        """
        # Remove zero or negative frequencies
        # Handle complex frequencies (imaginary frequencies for TS)
        # Take real part and filter out non-positive values
        real_freqs = frequencies[np.real(frequencies) > 0]
        # Ensure we have real values (take real part if complex)
        real_freqs = np.real(real_freqs)

        if self.method == "grimme":
            # Use qRRHO energy with Grimme's damping
            e_qRRHO, _ = self._vibrational_energy_qRRHO(real_freqs, temperature)
            damp = calculate_damping_function(real_freqs, self.freq_cutoff, self.alpha)
            # Interpolate between qRRHO and classical limit (0.5 RT)
            per_mode_energy = e_qRRHO * damp + (1 - damp) * 0.5 * GAS_CONSTANT * temperature
            total_energy = float(np.sum(per_mode_energy))
        elif self.method == "truhlar":
            # Truhlar method doesn't apply to energy, use standard RRHO
            per_mode_energy = calculate_rrho_energy(
                real_freqs,
                temperature,
                self.freq_scale_factor,
            )
            total_energy = float(np.sum(per_mode_energy))
        else:  # rrho
            per_mode_energy = calculate_rrho_energy(
                real_freqs,
                temperature,
                self.freq_scale_factor,
            )
            total_energy = float(np.sum(per_mode_energy))

        return total_energy, per_mode_energy

    def _vibrational_energy_qRRHO(  # noqa: N802
        self,
        frequencies: np.ndarray,
        temperature: float,
    ) -> tuple[float, np.ndarray]:
        """Calculate qRRHO energy (helper method).

        Parameters
        ----------
        frequencies : np.ndarray
            Vibrational frequencies in cm⁻¹
        temperature : float
            Temperature in Kelvin

        Returns
        -------
        tuple[float, np.ndarray]
            Total energy and per-mode energies in J/mol
        """
        per_mode_energy = calculate_qRRHO_energy(
            frequencies,
            temperature,
            self.freq_cutoff,
            self.freq_scale_factor,
        )
        total_energy = float(np.sum(per_mode_energy))
        return total_energy, per_mode_energy

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"QuasiHarmonicHandler(method='{self.method}', freq_cutoff={self.freq_cutoff:.1f} cm⁻¹)"
        )
