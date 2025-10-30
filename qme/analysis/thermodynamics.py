"""Thermodynamic property calculations from vibrational frequencies.

This module provides the ThermodynamicProperties class for calculating
vibrational partition functions, heat capacities, and entropies.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms, units

__all__ = ["ThermodynamicProperties"]


class ThermodynamicProperties:
    """Calculate thermodynamic properties from vibrational frequencies."""

    def __init__(
        self,
        frequencies: np.ndarray,
        atoms: Atoms,
        temperature: float = 298.15,
        pressure: float = 101325,
    ) -> None:
        """Initialize thermodynamic property calculator.

        Parameters
        ----------
        frequencies : np.ndarray
            Vibrational frequencies in cm^-1
        atoms : Atoms
            Molecular structure
        temperature : float
            Temperature in Kelvin
        pressure : float
            Pressure in Pascal

        """
        self.frequencies = frequencies[frequencies > 0]  # Only real frequencies
        self.atoms = atoms
        self.temperature = temperature
        self.pressure = pressure

    def partition_function_vibrational(self) -> float:
        """Calculate vibrational partition function.

        Returns:
        -------
        float
            Vibrational partition function

        """
        # Convert frequencies from cm^-1 to eV
        freq_eV = self.frequencies * units.invcm  # Convert cm^-1 to eV
        kT = units.kB * self.temperature

        # q_vib = ∏ 1/(1 - exp(-hν/kT))
        q_vib = 1.0
        for freq in freq_eV:
            if freq / kT < 50:  # Avoid overflow
                q_vib *= 1.0 / (1.0 - np.exp(-freq / kT))
            else:
                # High frequency limit: q ≈ exp(-hν/2kT)
                q_vib *= np.exp(-freq / (2 * kT))

        return q_vib

    def heat_capacity_vibrational(self) -> float:
        """Calculate vibrational heat capacity.

        Returns:
        -------
        float
            Heat capacity in eV/K

        """
        # Convert frequencies from cm^-1 to eV
        freq_eV = self.frequencies * units.invcm  # Convert cm^-1 to eV
        kT = units.kB * self.temperature

        cv_vib = 0.0
        for freq in freq_eV:
            x = freq / kT
            if x < 50:  # Avoid overflow
                exp_x = np.exp(x)
                cv_vib += units.kB * x**2 * exp_x / (exp_x - 1) ** 2

        return cv_vib

    def entropy_vibrational(self) -> float:
        """Calculate vibrational entropy.

        Returns:
        -------
        float
            Entropy in eV/K

        """
        # Convert frequencies from cm^-1 to eV
        freq_eV = self.frequencies * units.invcm  # Convert cm^-1 to eV
        kT = units.kB * self.temperature

        s_vib = 0.0
        for freq in freq_eV:
            x = freq / kT
            if x < 50:  # Avoid overflow
                exp_x = np.exp(x)
                s_vib += units.kB * (x / (exp_x - 1) - np.log(1 - np.exp(-x)))

        return s_vib
