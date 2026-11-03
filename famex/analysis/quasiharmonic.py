"""Quasi-harmonic corrections for low-frequency vibrational modes.

This module implements Grimme's and Truhlar's methods for correcting
harmonic oscillator entropies and energies for low-frequency modes that
behave more like free rotors at typical reaction temperatures.
"""

from __future__ import annotations

import math

import numpy as np

from famex.analysis.physics_constants import (
    AVOGADRO_CONSTANT,
    BOLTZMANN_CONSTANT,
    GAS_CONSTANT,
    J_PER_MOL_TO_EV,
    PLANCK_CONSTANT,
    SPEED_OF_LIGHT,
    filter_positive_frequencies,
    normalize_frequencies_cm1,
)
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
    "harmonic_zero_point_energy_ev",
]

# Above this reduced frequency, exp(x) overflows and the thermal terms vanish.
_X_STABLE = 40.0


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
    signed_freqs = normalize_frequencies_cm1(frequencies)
    frequencies_safe = np.where(signed_freqs > 0, signed_freqs, 1e-10)
    damp = 1.0 / (1.0 + (freq_cutoff / frequencies_safe) ** alpha)
    return damp


def _as_frequency_array(frequencies: np.ndarray) -> np.ndarray:
    return np.atleast_1d(np.asarray(frequencies, dtype=np.float64))


def _scale_frequencies(frequencies: np.ndarray, freq_scale_factor: float = 1.0) -> np.ndarray:
    return _as_frequency_array(frequencies) * freq_scale_factor


def vibrational_thermal_factor(frequencies_cm1: np.ndarray, temperature: float) -> np.ndarray:
    """Return x = hν/kT for frequencies in cm^-1."""
    return (PLANCK_CONSTANT * np.asarray(frequencies_cm1) * SPEED_OF_LIGHT) / (
        BOLTZMANN_CONSTANT * temperature
    )


def _rrho_entropy_from_x(x: np.ndarray) -> np.ndarray:
    """RRHO entropy in J/(mol·K). Large x uses the zero thermal-entropy limit."""
    entropy = np.zeros(x.shape, dtype=np.float64)
    stable = x < _X_STABLE
    if np.any(stable):
        xs = x[stable]
        entropy[stable] = GAS_CONSTANT * (xs / np.expm1(xs) - np.log(-np.expm1(-xs)))
    return entropy


def _rrho_energy_from_x(x: np.ndarray, temperature: float) -> np.ndarray:
    """RRHO energy in J/mol, including the zero-point term."""
    thermal = np.zeros(x.shape, dtype=np.float64)
    stable = x < _X_STABLE
    if np.any(stable):
        thermal[stable] = x[stable] / np.expm1(x[stable])
    return GAS_CONSTANT * temperature * (0.5 * x + thermal)


def harmonic_heat_capacity_factor(x: np.ndarray) -> np.ndarray:
    """Return x² e^x / (e^x − 1)². The high-frequency limit is 0."""
    values = np.asarray(x, dtype=np.float64)
    factor = np.zeros_like(values)
    stable = values < _X_STABLE
    if np.any(stable):
        xs = values[stable]
        factor[stable] = xs**2 * np.exp(xs) / np.expm1(xs) ** 2
    return factor


def harmonic_zero_point_energy_ev(frequencies_cm1: np.ndarray) -> float:
    """Harmonic zero-point energy in eV from the same hν as the RRHO energy."""
    freqs = _as_frequency_array(frequencies_cm1)
    zpe_j_per_mol = (
        0.5 * PLANCK_CONSTANT * float(np.sum(freqs)) * SPEED_OF_LIGHT * AVOGADRO_CONSTANT
    )
    return float(zpe_j_per_mol * J_PER_MOL_TO_EV)


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
    scaled = _scale_frequencies(frequencies, freq_scale_factor)
    return _rrho_entropy_from_x(vibrational_thermal_factor(scaled, temperature))


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
    scaled = _scale_frequencies(frequencies, freq_scale_factor)
    return _rrho_energy_from_x(vibrational_thermal_factor(scaled, temperature), temperature)


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
    frequencies_scaled = _scale_frequencies(frequencies, freq_scale_factor)
    mu = PLANCK_CONSTANT / (8 * math.pi**2 * frequencies_scaled * SPEED_OF_LIGHT)
    mu_primed = mu * average_moment_of_inertia / (mu + average_moment_of_inertia)
    factor = 8 * math.pi**3 * mu_primed * BOLTZMANN_CONSTANT * temperature / PLANCK_CONSTANT**2
    return (0.5 + np.log(np.sqrt(factor))) * GAS_CONSTANT


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
    scaled = _scale_frequencies(frequencies, freq_scale_factor)
    entropy_full = _rrho_entropy_from_x(vibrational_thermal_factor(scaled, temperature))
    entropy_cutoff = _rrho_entropy_from_x(
        vibrational_thermal_factor(np.full_like(scaled, freq_cutoff), temperature)
    )
    return np.where(scaled > freq_cutoff, entropy_full, entropy_cutoff)


def calculate_qRRHO_energy(  # noqa: N802
    frequencies: np.ndarray,
    temperature: float,
    freq_scale_factor: float = 1.0,
) -> np.ndarray:
    """Harmonic vibrational energy, including the zero-point term."""
    return calculate_rrho_energy(frequencies, temperature, freq_scale_factor)


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

    def _positive_scaled_frequencies(self, frequencies: np.ndarray) -> np.ndarray:
        return _scale_frequencies(filter_positive_frequencies(frequencies), self.freq_scale_factor)

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
        real_freqs = self._positive_scaled_frequencies(frequencies)

        if self.method == "grimme":
            s_rrho = calculate_rrho_entropy(real_freqs, temperature)
            s_free_rot = calculate_free_rotor_entropy(
                real_freqs,
                temperature,
                average_moment_of_inertia=self.average_moment_of_inertia,
            )
            damp = calculate_damping_function(real_freqs, self.freq_cutoff, self.alpha)
            per_mode_entropy = s_rrho * damp + (1 - damp) * s_free_rot
        elif self.method == "truhlar":
            per_mode_entropy = calculate_qRRHO_entropy(real_freqs, temperature, self.freq_cutoff)
        else:
            per_mode_entropy = calculate_rrho_entropy(real_freqs, temperature)

        return float(np.sum(per_mode_entropy)), per_mode_entropy

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
        real_freqs = self._positive_scaled_frequencies(frequencies)
        per_mode_energy = calculate_rrho_energy(real_freqs, temperature)
        if self.method == "grimme":
            damp = calculate_damping_function(real_freqs, self.freq_cutoff, self.alpha)
            per_mode_energy = per_mode_energy * damp + (1 - damp) * 0.5 * GAS_CONSTANT * temperature
        return float(np.sum(per_mode_energy)), per_mode_energy

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"QuasiHarmonicHandler(method='{self.method}', freq_cutoff={self.freq_cutoff:.1f} cm⁻¹)"
        )
