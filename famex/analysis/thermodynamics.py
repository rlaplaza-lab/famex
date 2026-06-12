"""Thermodynamic property calculations from vibrational frequencies.

This module provides the ThermodynamicProperties class for calculating
complete thermodynamic properties including vibrational, translational,
rotational, and electronic contributions with quasi-harmonic corrections.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms, units

from famex.analysis.quasiharmonic import QuasiHarmonicHandler
from famex.analysis.solvation import SolvationHandler
from famex.analysis.statistical_thermo import StatisticalThermodynamics
from famex.analysis.symmetry import SymmetryHandler

# Unit conversions using ASE units
# ASE units convention: energy in eV, lengths in Å
# For J/mol <-> eV: use units.J / units.mol (J/mol in eV units)
J_PER_MOL_TO_EV = units.J / units.mol  # J/mol to eV conversion from ASE
EV_TO_KCAL_PER_MOL = 23.06035  # eV to kcal/mol conversion
J_TO_AU = 4.184 * 627.509541 * 1000.0  # J to atomic units (GoodVibes convention)

__all__ = ["ThermodynamicProperties"]


class ThermodynamicProperties:
    """Calculate complete thermodynamic properties from vibrational frequencies.

    This class integrates vibrational, translational, rotational, and electronic
    contributions with support for quasi-harmonic corrections, solvation effects,
    and symmetry corrections.
    """

    def __init__(
        self,
        frequencies: np.ndarray,
        atoms: Atoms,
        temperature: float = 298.15,
        pressure: float = 101325,
        # Quasi-harmonic parameters
        method: str = "rrho",
        freq_cutoff: float = 100.0,
        freq_scale_factor: float = 1.0,
        # Statistical thermodynamics parameters
        rotational_temperatures: np.ndarray | None = None,
        rotational_constants: np.ndarray | None = None,
        symmetry_number: int = 1,
        point_group: str | None = None,
        linear: bool | None = None,
        multiplicity: int = 1,
        # Solvation parameters
        solvent: str = "none",
        concentration: float = 1.0,
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
        method : str
            Method for vibrational corrections: 'rrho', 'grimme', or 'truhlar'
        freq_cutoff : float
            Cutoff frequency in cm^-1 for quasi-harmonic corrections
        freq_scale_factor : float
            Frequency scaling factor (default: 1.0)
        rotational_temperatures : array-like, optional
            Rotational temperatures in Kelvin
        rotational_constants : array-like, optional
            Rotational constants in GHz
        symmetry_number : int
            Symmetry number (default: 1 for C1)
        point_group : str, optional
            Point group symbol for automatic symmetry determination
        linear : bool, optional
            Whether molecule is linear (auto-detected if None)
        multiplicity : int
            Spin multiplicity 2S+1 (default: 1)
        solvent : str
            Solvent name (default: 'none' for gas phase)
        concentration : float
            Concentration in mol/L

        """
        # Handle complex frequencies (imaginary frequencies for TS)
        # Take real part and filter out non-positive values
        self.frequencies = frequencies[np.real(frequencies) > 0]
        # Ensure we have real values (take real part if complex)
        self.frequencies = np.real(self.frequencies)
        self.all_frequencies = frequencies  # Keep all for reference
        self.atoms = atoms
        self.temperature = temperature
        self.pressure = pressure

        # Initialize handlers
        self.qh_handler = QuasiHarmonicHandler(method, freq_cutoff, freq_scale_factor)
        self.solvation_handler = SolvationHandler(solvent, concentration)
        self.symmetry_handler = SymmetryHandler(symmetry_number, point_group)
        self.stat_thermo = StatisticalThermodynamics(
            atoms,
            rotational_temperatures,
            rotational_constants,
            self.symmetry_handler.symmetry_number,
            linear,
            multiplicity,
        )

    def partition_function_vibrational(self) -> float:
        """Calculate vibrational partition function.

        Returns
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

        Returns
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
        """Calculate vibrational entropy with quasi-harmonic corrections.

        Returns
        -------
        float
            Vibrational entropy in eV/K

        """
        # Use quasi-harmonic handler for entropy
        total_entropy, _ = self.qh_handler.vibrational_entropy(self.frequencies, self.temperature)
        # Convert from J/(mol·K) to eV/K
        return total_entropy * J_PER_MOL_TO_EV

    def calculate_complete_thermodynamics(
        self,
        energy: float | None = None,
    ) -> dict[str, Any]:
        """Calculate complete thermodynamic properties.

        Parameters
        ----------
        energy : float, optional
            Electronic energy in eV. If None, extracts from atoms.calc

        Returns
        -------
        dict
            Dictionary with all thermodynamic contributions:
            - energy: Electronic energy (eV)
            - zpe: Zero-point energy (eV)
            - enthalpy_trans: Translational enthalpy (eV)
            - enthalpy_rot: Rotational enthalpy (eV)
            - enthalpy_vib: Vibrational enthalpy (eV)
            - enthalpy_total: Total enthalpy H = E + ZPE + H_trans + H_rot + H_vib + RT (eV)
            - entropy_trans: Translational entropy (eV/K)
            - entropy_rot: Rotational entropy (eV/K)
            - entropy_vib: Vibrational entropy with QH corrections (eV/K)
            - entropy_elec: Electronic entropy (eV/K)
            - entropy_total: Total entropy (eV/K)
            - gibbs_free_energy: Gibbs free energy G = H - TS (eV)
            - temperature: Temperature (K)
            - method: QH method used
            - contributions: Breakdown of all contributions
        """
        # Get electronic energy
        if energy is None:
            if hasattr(self.atoms, "calc") and self.atoms.calc is not None:
                if hasattr(self.atoms.calc, "get_potential_energy"):
                    energy = self.atoms.calc.get_potential_energy()
                elif hasattr(self.atoms.calc, "results"):
                    energy = self.atoms.calc.results.get("energy", 0.0)
                else:
                    energy = 0.0
            else:
                energy = 0.0

        # Convert energy from eV to J/mol for calculations
        energy_J_per_mol = energy / J_PER_MOL_TO_EV

        # Calculate zero-point energy in eV
        zpe = self.calculate_zero_point_energy()

        # Get quasi-harmonic vibrational contributions (J/mol and J/(mol·K))
        u_vib, _ = self.qh_handler.vibrational_energy(self.frequencies, self.temperature)
        s_vib_actual, _ = self.qh_handler.vibrational_entropy(self.frequencies, self.temperature)

        # Calculate vibrational enthalpy: H_vib = U_vib (no PV term for vibrational)
        enthalpy_vib = u_vib  # J/mol

        # Statistical thermodynamics contributions
        enthalpy_trans = self.stat_thermo.translational_energy(self.temperature)  # J/mol
        enthalpy_rot = self.stat_thermo.rotational_energy(self.temperature)  # J/mol

        # Entropies (J/(mol·K))
        entropy_trans = self.stat_thermo.translational_entropy(
            self.temperature,
            self.solvation_handler.concentration,
            self.solvation_handler.free_space_ml_per_l,
        )
        entropy_rot = self.stat_thermo.rotational_entropy(self.temperature)
        entropy_elec = self.stat_thermo.electronic_entropy()

        # Total enthalpy: H = E + ZPE + U_trans + U_rot + U_vib + RT
        H_total_J_per_mol = (
            energy_J_per_mol
            + self.calculate_zero_point_energy_in_J_per_mol()
            + enthalpy_trans
            + enthalpy_rot
            + enthalpy_vib
            + 8.3144621 * self.temperature  # RT term
        )

        # Total entropy
        S_total_J_per_K = entropy_trans + entropy_rot + s_vib_actual + entropy_elec

        # Convert to eV and eV/K
        enthalpy_total = H_total_J_per_mol * J_PER_MOL_TO_EV  # eV
        entropy_total = S_total_J_per_K * J_PER_MOL_TO_EV  # eV/K

        # Gibbs free energy: G = H - TS
        gibbs_free_energy = enthalpy_total - self.temperature * entropy_total

        return {
            "energy": energy,
            "zpe": zpe,
            "enthalpy_trans": enthalpy_trans * J_PER_MOL_TO_EV,
            "enthalpy_rot": enthalpy_rot * J_PER_MOL_TO_EV,
            "enthalpy_vib": enthalpy_vib * J_PER_MOL_TO_EV,
            "enthalpy_total": enthalpy_total,
            "entropy_trans": entropy_trans * J_PER_MOL_TO_EV,
            "entropy_rot": entropy_rot * J_PER_MOL_TO_EV,
            "entropy_vib": s_vib_actual * J_PER_MOL_TO_EV,
            "entropy_elec": entropy_elec * J_PER_MOL_TO_EV,
            "entropy_total": entropy_total,
            "gibbs_free_energy": gibbs_free_energy,
            "temperature": self.temperature,
            "method": self.qh_handler.method,
            "contributions": {
                "translational": {
                    "enthalpy": enthalpy_trans * J_PER_MOL_TO_EV,
                    "entropy": entropy_trans * J_PER_MOL_TO_EV,
                },
                "rotational": {
                    "enthalpy": enthalpy_rot * J_PER_MOL_TO_EV,
                    "entropy": entropy_rot * J_PER_MOL_TO_EV,
                },
                "vibrational": {
                    "enthalpy": enthalpy_vib * J_PER_MOL_TO_EV,
                    "entropy": s_vib_actual * J_PER_MOL_TO_EV,
                },
                "electronic": {
                    "enthalpy": 0.0,
                    "entropy": entropy_elec * J_PER_MOL_TO_EV,
                },
                "zero_point": zpe,
            },
        }

    def calculate_zero_point_energy(self) -> float:
        """Calculate zero-point vibrational energy.

        Returns
        -------
        float
            Zero-point energy in eV

        """
        freq_eV = self.frequencies * units.invcm  # Convert cm^-1 to eV
        zpe = 0.5 * np.sum(freq_eV)
        return zpe

    def calculate_zero_point_energy_in_J_per_mol(self) -> float:  # noqa: N802
        """Calculate zero-point energy in J/mol.

        Returns
        -------
        float
            Zero-point energy in J/mol

        """
        # Convert frequencies from cm^-1 to energy per mode
        # E = 0.5 * h * ν in J per quantum
        freq_invcm = self.frequencies
        h = 6.62606957e-34  # J·s
        c = 2.99792458e10  # cm/s
        zpe_per_mode = 0.5 * h * freq_invcm * c  # J per quantum
        zpe_J_per_mol = np.sum(zpe_per_mode) * 6.0221415e23  # J/mol
        return zpe_J_per_mol

    def internal_energy_vibrational(self) -> float:
        """Calculate vibrational internal energy (including thermal contributions).

        Returns
        -------
        float
            Vibrational internal energy in eV

        """
        total_energy, _ = self.qh_handler.vibrational_energy(self.frequencies, self.temperature)
        # Convert from J/mol to eV
        return total_energy * J_PER_MOL_TO_EV
