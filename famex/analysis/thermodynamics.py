"""Thermodynamic property calculations from vibrational frequencies.

This module provides the ThermodynamicProperties class for calculating
complete thermodynamic properties including vibrational, translational,
rotational, and electronic contributions with quasi-harmonic corrections.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms, units

from famex.analysis.physics_constants import (
    GAS_CONSTANT,
    J_PER_MOL_TO_EV,
    filter_positive_frequencies,
)
from famex.analysis.quasiharmonic import QuasiHarmonicHandler
from famex.analysis.solvation import SolvationHandler
from famex.analysis.statistical_thermo import StatisticalThermodynamics
from famex.analysis.symmetry import SymmetryHandler

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
        """Initialize thermodynamic property calculator."""
        self.frequencies = filter_positive_frequencies(frequencies)
        self.all_frequencies = frequencies
        self.atoms = atoms
        self.temperature = temperature
        self.pressure = pressure

        self.qh_handler = QuasiHarmonicHandler(method, freq_cutoff, freq_scale_factor)
        self.solvation_handler = SolvationHandler(solvent, concentration)
        self.symmetry_handler = SymmetryHandler(symmetry_number, point_group)
        self.stat_thermo = StatisticalThermodynamics(
            atoms,
            rotational_temperatures,
            rotational_constants,
            symmetry_number,
            linear,
            multiplicity,
        )
        self.stat_thermo.symmetry_number = self.symmetry_handler.get_rotational_symmetry_number(
            self.stat_thermo.linear,
        )

    def heat_capacity_vibrational(self) -> float:
        """Calculate vibrational heat capacity."""
        freq_eV = self.frequencies * units.invcm
        kT = units.kB * self.temperature

        cv_vib = 0.0
        for freq in freq_eV:
            x = freq / kT
            if x < 50:
                exp_x = np.exp(x)
                cv_vib += units.kB * x**2 * exp_x / (exp_x - 1) ** 2

        return cv_vib

    def entropy_vibrational(self) -> float:
        """Calculate vibrational entropy with quasi-harmonic corrections."""
        total_entropy, _ = self.qh_handler.vibrational_entropy(self.frequencies, self.temperature)
        return total_entropy * J_PER_MOL_TO_EV

    def calculate_complete_thermodynamics(
        self,
        energy: float | None = None,
    ) -> dict[str, Any]:
        """Calculate complete thermodynamic properties."""
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

        energy_J_per_mol = energy / J_PER_MOL_TO_EV
        zpe = self.calculate_zero_point_energy()

        u_vib, _ = self.qh_handler.vibrational_energy(self.frequencies, self.temperature)
        s_vib_actual, _ = self.qh_handler.vibrational_entropy(self.frequencies, self.temperature)

        enthalpy_vib = u_vib
        enthalpy_trans = self.stat_thermo.translational_energy(self.temperature)
        enthalpy_rot = self.stat_thermo.rotational_energy(self.temperature)

        entropy_trans = self.stat_thermo.translational_entropy(
            self.temperature,
            self.solvation_handler.concentration,
            self.solvation_handler.free_space_ml_per_l,
        )
        entropy_rot = self.stat_thermo.rotational_entropy(self.temperature)
        entropy_elec = self.stat_thermo.electronic_entropy()

        H_total_J_per_mol = (
            energy_J_per_mol
            + self.calculate_zero_point_energy_in_J_per_mol()
            + enthalpy_trans
            + enthalpy_rot
            + enthalpy_vib
            + GAS_CONSTANT * self.temperature
        )

        S_total_J_per_K = entropy_trans + entropy_rot + s_vib_actual + entropy_elec

        enthalpy_total = H_total_J_per_mol * J_PER_MOL_TO_EV
        entropy_total = S_total_J_per_K * J_PER_MOL_TO_EV
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
        """Calculate zero-point vibrational energy in eV."""
        freq_eV = self.frequencies * units.invcm
        return float(0.5 * np.sum(freq_eV))

    def calculate_zero_point_energy_in_J_per_mol(self) -> float:  # noqa: N802
        """Calculate zero-point energy in J/mol."""
        return self.calculate_zero_point_energy() / J_PER_MOL_TO_EV

    def internal_energy_vibrational(self) -> float:
        """Calculate vibrational internal energy in eV."""
        total_energy, _ = self.qh_handler.vibrational_energy(self.frequencies, self.temperature)
        return total_energy * J_PER_MOL_TO_EV
