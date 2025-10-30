"""Statistical thermodynamics contributions for thermochemistry.

This module provides translational, rotational, and electronic contributions
to energy and entropy for statistical thermodynamics calculations.
"""

from __future__ import annotations

import math

import numpy as np
from ase import Atoms

__all__ = [
    "StatisticalThermodynamics",
    "calculate_translational_energy",
    "calculate_translational_entropy",
    "calculate_rotational_energy",
    "calculate_rotational_entropy",
    "calculate_electronic_entropy",
]

# Physical constants
GAS_CONSTANT = 8.3144621  # J/(mol·K)
PLANCK_CONSTANT = 6.62606957e-34  # J·s
BOLTZMANN_CONSTANT = 1.3806488e-23  # J/K
AVOGADRO_CONSTANT = 6.0221415e23  # 1/mol
AMU_to_KG = 1.66053886e-27  # kg/amu


def calculate_translational_energy(temperature: float) -> float:
    """Calculate translational energy contribution.

    Parameters
    ----------
    temperature : float
        Temperature in Kelvin

    Returns:
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

    Returns:
    -------
    float
        Translational entropy in J/(mol·K)
    """
    # Calculate de Broglie wavelength
    # λ = h / sqrt(2πm kT)
    de_broglie = (
        (2.0 * math.pi * molecular_mass * AMU_to_KG * BOLTZMANN_CONSTANT * temperature) ** 0.5
    ) / PLANCK_CONSTANT

    # Convert concentration from mol/L to number density (molecules/m³)
    # For gas phase or solution phase, use free space to calculate accessible volume
    ndens = concentration * 1000.0 * AVOGADRO_CONSTANT / (free_space_ml_per_l / 1000.0)

    # Sackur-Tetrode equation: S = R (ln((2πmkT/h²)^(3/2) / n) + 5/2)
    entropy = GAS_CONSTANT * (2.5 + math.log(de_broglie**3 / ndens))

    return entropy


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

    Returns:
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

    Returns:
    -------
    float
        Rotational entropy in J/(mol·K)

    Notes:
    -----
    If rotational_temperatures is empty or all zero, returns 0.0 (atoms).
    For linear molecules with 2 rotational temperatures (rare), returns 0.0
    to avoid errors.
    """
    rotational_temps = np.asarray(rotational_temperatures)

    # Check for atoms
    if is_atom or len(rotational_temps) == 0 or np.allclose(rotational_temps, 0.0):
        return 0.0

    # Handle edge case of 2 rotational temperatures
    if len(rotational_temps) == 2:
        # Gaussian sometimes reports only 2 temps for linear molecules
        return 0.0

    if linear:
        # Linear molecule: q_rot = T / Θ_rot
        if len(rotational_temps) == 1:
            q_rot = temperature / rotational_temps[0]
            entropy = GAS_CONSTANT * (math.log(q_rot / symmetry_number) + 1)
        else:
            # Shouldn't happen, but fallback
            entropy = 0.0
    else:
        # Non-linear molecule: q_rot = π^(1/2) (T^3 / (Θ_A Θ_B Θ_C))^(1/2)
        if len(rotational_temps) == 3:
            q_rot = math.sqrt(math.pi * temperature**3 / np.prod(rotational_temps))
            entropy = GAS_CONSTANT * (math.log(q_rot / symmetry_number) + 1.5)
        else:
            # Fallback for non-standard cases
            entropy = 0.0

    return entropy


def calculate_electronic_entropy(multiplicity: int) -> float:
    """Calculate electronic entropy contribution.

    Parameters
    ----------
    multiplicity : int
        Spin multiplicity (2S+1)

    Returns:
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
            self.linear = self._check_linearity()
        else:
            self.linear = linear

        # Determine if atom
        self.is_atom = len(atoms) == 1

        # Handle rotational temperatures/constants
        self.rotational_temperatures = self._process_rotational_data(
            rotational_temperatures,
            rotational_constants,
        )

    def _check_linearity(self) -> bool:
        """Check if molecule is linear using moment of inertia."""
        if len(self.atoms) <= 2:
            return True

        # Calculate moment of inertia tensor
        positions = self.atoms.positions
        masses = self.atoms.get_masses()
        com = self.atoms.get_center_of_mass()
        positions_centered = positions - com

        inertia_tensor = np.zeros((3, 3))
        for pos, mass in zip(positions_centered, masses, strict=False):
            inertia_tensor += mass * (np.dot(pos, pos) * np.eye(3) - np.outer(pos, pos))

        # Eigenvalues of moment of inertia tensor
        eigenvalues = np.linalg.eigvals(inertia_tensor)
        eigenvalues = np.sort(eigenvalues)

        # Linear if smallest eigenvalue is essentially zero
        return eigenvalues[0] < 1e-6

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

        Returns:
        -------
        np.ndarray
            Rotational temperatures in Kelvin
        """
        if rotational_temperatures is not None:
            return np.asarray(rotational_temperatures)

        if rotational_constants is not None:
            # Convert rotational constants (GHz) to rotational temperatures (K)
            # Θ = hcB / k_B, where B is in cm^-1
            # 1 GHz = 0.033356 cm^-1
            GHz_to_cm1 = 0.033356
            rotational_constants_cm1 = np.asarray(rotational_constants) * GHz_to_cm1
            rotational_temps = (
                PLANCK_CONSTANT
                * 2.99792458e10  # speed of light in cm/s
                * rotational_constants_cm1
                / BOLTZMANN_CONSTANT
            )
            return rotational_temps

        # Default: try to calculate from geometry
        if self.is_atom or len(self.atoms) == 1:
            return np.array([0.0, 0.0, 0.0])

        # For small molecules, calculate from moment of inertia
        return np.array([1.0, 1.0, 1.0])  # Placeholder - could implement calculation

    def translational_energy(self, temperature: float) -> float:
        """Calculate translational energy.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin

        Returns:
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

        Returns:
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

        Returns:
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

        Returns:
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

        Returns:
        -------
        float
            Electronic entropy in J/(mol·K)
        """
        return calculate_electronic_entropy(self.multiplicity)
