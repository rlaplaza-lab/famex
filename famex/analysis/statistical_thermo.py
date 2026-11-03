"""Statistical thermodynamics contributions for thermochemistry.

This module provides translational, rotational, and electronic contributions
to energy and entropy for statistical thermodynamics calculations.
"""

from __future__ import annotations

import math

import numpy as np
from ase import Atoms

from famex.analysis.molecular_properties import (
    is_linear_molecule,
    rotational_temperatures_from_atoms,
)
from famex.analysis.physics_constants import (
    AMU_TO_KG,
    AVOGADRO_CONSTANT,
    BOLTZMANN_CONSTANT,
    GAS_CONSTANT,
    PLANCK_CONSTANT,
    SPEED_OF_LIGHT,
)

__all__ = [
    "StatisticalThermodynamics",
    "calculate_translational_energy",
    "calculate_translational_entropy",
    "calculate_rotational_energy",
    "calculate_rotational_entropy",
    "calculate_electronic_entropy",
]

_ROTATIONAL_TEMPERATURE_FLOOR = 1e-8


def calculate_translational_energy(temperature: float) -> float:
    """Calculate translational energy contribution.

    Parameters
    ----------
    temperature : float
        Temperature in Kelvin

    Returns
    -------
    float
        Translational energy in J/mol
    """
    return 1.5 * GAS_CONSTANT * temperature


def calculate_translational_entropy(
    molecular_mass: float,
    temperature: float,
    concentration: float = 1.0,
    free_space_ml_per_l: float = 1000.0,
) -> float:
    """Calculate translational entropy contribution.

    Uses the Sackur-Tetrode equation for ideal gas entropy.

    Parameters
    ----------
    molecular_mass : float
        Molecular mass in atomic mass units (amu)
    temperature : float
        Temperature in Kelvin
    concentration : float
        Concentration in mol/L (default: 1.0 for 1M standard state)
    free_space_ml_per_l : float
        Free space in mL per L of solution (default: 1000 for gas phase)

    Returns
    -------
    float
        Translational entropy in J/(mol·K)
    """
    inverse_thermal_wavelength = (
        math.sqrt(2.0 * math.pi * molecular_mass * AMU_TO_KG * BOLTZMANN_CONSTANT * temperature)
        / PLANCK_CONSTANT
    )

    # mol/L -> molecules/m³, reduced by the accessible volume fraction.
    number_density = concentration * 1000.0 * AVOGADRO_CONSTANT / (free_space_ml_per_l / 1000.0)
    return GAS_CONSTANT * (2.5 + math.log(inverse_thermal_wavelength**3 / number_density))


def calculate_rotational_energy(
    temperature: float,
    linear: bool = False,
    is_atom: bool = False,
) -> float:
    """Calculate rotational energy contribution.

    Parameters
    ----------
    temperature : float
        Temperature in Kelvin
    linear : bool
        Whether the molecule is linear (default: False for non-linear)
    is_atom : bool
        Whether this is a single atom (default: False)

    Returns
    -------
    float
        Rotational energy in J/mol
    """
    if is_atom:
        return 0.0
    if linear:
        return GAS_CONSTANT * temperature
    return 1.5 * GAS_CONSTANT * temperature


def calculate_rotational_entropy(
    rotational_temperatures: np.ndarray | list[float],
    symmetry_number: int,
    temperature: float,
    linear: bool = False,
    is_atom: bool = False,
) -> float:
    """Calculate rotational entropy contribution.

    Uses rotational partition function: q_rot = π^(1/2) (T^3 / (Θ_A Θ_B Θ_C))^(1/2) / σ
    for non-linear molecules, or q_rot = T / (Θ_rot σ) for linear molecules.

    Parameters
    ----------
    rotational_temperatures : array-like
        Rotational temperatures in Kelvin [Θ_A, Θ_B, Θ_C] for non-linear,
        or [Θ_rot] for linear molecules
    symmetry_number : int
        Symmetry number for rotational degeneracy
    temperature : float
        Temperature in Kelvin
    linear : bool
        Whether the molecule is linear (default: False)
    is_atom : bool
        Whether this is a single atom (default: False)

    Returns
    -------
    float
        Rotational entropy in J/(mol·K)

    Notes
    -----
    Atoms and an empty temperature list contribute nothing. A linear molecule
    uses one rotational temperature: a degenerate pair is averaged, and a third
    near-zero or outlying axis temperature is dropped. A non-linear molecule
    requires three positive rotational temperatures.
    """
    rotational_temps = np.asarray(rotational_temperatures, dtype=np.float64).reshape(-1)
    if is_atom or bool(np.allclose(rotational_temps, 0.0)):
        return 0.0

    if linear:
        theta = _linear_rotational_temperature(rotational_temps)
        if theta == 0.0:
            return 0.0
        q_rot = temperature / theta
        return GAS_CONSTANT * (math.log(q_rot / symmetry_number) + 1.0)

    positive = rotational_temps[rotational_temps > _ROTATIONAL_TEMPERATURE_FLOOR]
    if positive.size != 3:
        msg = (
            "Non-linear rotational entropy expects 3 positive rotational temperatures, "
            f"got {positive.size}"
        )
        raise ValueError(msg)

    q_rot = math.sqrt(math.pi * temperature**3 / float(np.prod(positive)))
    return GAS_CONSTANT * (math.log(q_rot / symmetry_number) + 1.5)


def _linear_rotational_temperature(rotational_temperatures: np.ndarray) -> float:
    """Return the unique linear-rotor temperature in kelvin.

    Three positive values mean a near-zero axis temperature is still present.
    That smallest value is dropped and the degenerate pair is averaged.
    """
    positive = rotational_temperatures[rotational_temperatures > _ROTATIONAL_TEMPERATURE_FLOOR]
    if positive.size == 0:
        return 0.0
    if positive.size > 3:
        msg = (
            "Linear rotational entropy expects at most 3 rotational temperatures, "
            f"got {positive.size} positive values"
        )
        raise ValueError(msg)
    if positive.size == 3:
        positive = np.sort(positive)[1:]
    return float(np.mean(positive))


def calculate_electronic_entropy(multiplicity: int) -> float:
    """Calculate electronic entropy contribution.

    Parameters
    ----------
    multiplicity : int
        Spin multiplicity (2S+1)

    Returns
    -------
    float
        Electronic entropy in J/(mol·K)
    """
    if multiplicity <= 0:
        raise ValueError("Multiplicity must be positive")
    if multiplicity == 1:
        # For singlet, entropy is 0
        return 0.0
    return GAS_CONSTANT * math.log(multiplicity)


class StatisticalThermodynamics:
    """Calculate statistical thermodynamics contributions for molecules."""

    def __init__(
        self,
        atoms: Atoms,
        rotational_temperatures: np.ndarray | None = None,
        rotational_constants: np.ndarray | None = None,
        symmetry_number: int = 1,
        linear: bool | None = None,
        multiplicity: int = 1,
    ):
        """Initialize statistical thermodynamics calculator.

        Parameters
        ----------
        atoms : Atoms
            Molecular structure
        rotational_temperatures : array-like, optional
            Rotational temperatures in Kelvin [Θ_A, Θ_B, Θ_C]
        rotational_constants : array-like, optional
            Rotational constants in GHz (alternative to temperatures)
        symmetry_number : int
            Symmetry number for rotational degeneracy (default: 1 for C1)
        linear : bool, optional
            Whether molecule is linear. If None, auto-detect.
        multiplicity : int
            Spin multiplicity 2S+1 (default: 1 for singlet)
        """
        self.atoms = atoms
        self.multiplicity = multiplicity
        self.symmetry_number = symmetry_number

        # Determine molecular mass
        self.molecular_mass = np.sum(atoms.get_masses())

        # Determine linearity if not specified
        if linear is None:
            self.linear = is_linear_molecule(atoms, list(range(len(atoms))))
        else:
            self.linear = linear

        # Determine if atom
        self.is_atom = len(atoms) == 1

        # Handle rotational temperatures/constants
        self.rotational_temperatures = self._process_rotational_data(
            rotational_temperatures,
            rotational_constants,
        )

    def _process_rotational_data(
        self,
        rotational_temperatures: np.ndarray | None,
        rotational_constants: np.ndarray | None,
    ) -> np.ndarray:
        """Process rotational temperatures or constants.

        Parameters
        ----------
        rotational_temperatures : array-like, optional
            Rotational temperatures in Kelvin
        rotational_constants : array-like, optional
            Rotational constants in GHz

        Returns
        -------
        np.ndarray
            Rotational temperatures in Kelvin
        """
        if rotational_temperatures is not None:
            return np.asarray(rotational_temperatures)

        if rotational_constants is not None:
            # Θ = hcB/k_B. 1 GHz = 1e9 / c(cm/s) cm^-1.
            rotational_constants_cm1 = np.asarray(rotational_constants, dtype=np.float64) * (
                1e9 / SPEED_OF_LIGHT
            )
            return PLANCK_CONSTANT * SPEED_OF_LIGHT * rotational_constants_cm1 / BOLTZMANN_CONSTANT

        return rotational_temperatures_from_atoms(self.atoms, self.linear)

    def translational_energy(self, temperature: float) -> float:
        """Calculate translational energy.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin

        Returns
        -------
        float
            Translational energy in J/mol
        """
        return calculate_translational_energy(temperature)

    def translational_entropy(
        self,
        temperature: float,
        concentration: float = 1.0,
        free_space_ml_per_l: float = 1000.0,
    ) -> float:
        """Calculate translational entropy.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin
        concentration : float
            Concentration in mol/L
        free_space_ml_per_l : float
            Free space in mL per L (1000 for gas phase)

        Returns
        -------
        float
            Translational entropy in J/(mol·K)
        """
        return calculate_translational_entropy(
            self.molecular_mass,
            temperature,
            concentration,
            free_space_ml_per_l,
        )

    def rotational_energy(self, temperature: float) -> float:
        """Calculate rotational energy.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin

        Returns
        -------
        float
            Rotational energy in J/mol
        """
        return calculate_rotational_energy(temperature, self.linear, self.is_atom)

    def rotational_entropy(self, temperature: float) -> float:
        """Calculate rotational entropy.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin

        Returns
        -------
        float
            Rotational entropy in J/(mol·K)
        """
        return calculate_rotational_entropy(
            self.rotational_temperatures,
            self.symmetry_number,
            temperature,
            self.linear,
            self.is_atom,
        )

    def electronic_entropy(self) -> float:
        """Calculate electronic entropy.

        Returns
        -------
        float
            Electronic entropy in J/(mol·K)
        """
        return calculate_electronic_entropy(self.multiplicity)
