"""Vibrational frequency analysis for QME.

This module provides comprehensive vibrational frequency analysis capabilities including:
- Hessian matrix calculations using finite differences or direct methods
- Normal mode analysis and vibrational frequencies
- Transition state verification
- Thermodynamic property calculations

Based on ASE's Vibrations class but extended for QME functionality with automatic
method selection and enhanced calculator integration.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from ase import Atoms, units
from numpy.typing import NDArray

from qme.analysis.hessian import HessianCalculator
from qme.analysis.molecular_properties import determine_degrees_of_freedom
from qme.analysis.normal_modes import convert_frequency_unit, diagonalize_mass_weighted_hessian
from qme.analysis.thermodynamics import ThermodynamicProperties
from qme.analysis.validation import validate_hessian
from qme.utils.logging import get_qme_logger
from qme.utils.validation import QMEError

logger = get_qme_logger(__name__)


class FrequencyAnalysis:
    """Vibrational frequency analysis with automatic Hessian method selection.

    Supports both direct Hessian calculation (when available from calculator)
    and finite difference Hessian calculation as fallback.
    """

    def __init__(
        self,
        atoms: Atoms,
        calculator: Any,
        delta: float = 0.01,
        richardson: bool = False,
        delta2: float | None = None,
        nfree: int | None = None,
        indices: list[int] | None = None,
        verbose: int = 1,
    ) -> None:
        """Initialize frequency analysis.

        Parameters
        ----------
        atoms : Atoms
            ASE Atoms object for the structure
        calculator : Calculator
            QME calculator (UMA, SO3LR, AIMNET2, etc.)
        delta : float
            Displacement for finite differences (Å)
        nfree : int, optional
            Number of degrees of freedom to remove (3 for translation + 3 for rotation).
            If None, automatically determined (6 for non-linear, 5 for linear molecules)
        indices : List[int], optional
            Indices of atoms to include in frequency analysis.
            If None, all atoms are included.
        verbose : int
            Verbosity level for frequency analysis output:
            - 0: Quiet (minimal output)
            - 1: Normal (default, shows progress)
            - 2: Verbose (detailed information)

        """
        self.atoms = atoms
        self.calculator = calculator
        # Only set calculator if atoms doesn't have one or it's different
        if self.atoms.calc is None or self.atoms.calc is not calculator:
            self.atoms.calc = calculator
        self.delta = delta
        self.richardson = richardson
        self.delta2 = delta2
        self.indices = indices if indices is not None else list(range(len(atoms)))
        self.verbose = verbose

        # Determine degrees of freedom to remove
        if nfree is None:
            self.nfree = determine_degrees_of_freedom(atoms, self.indices)
        else:
            self.nfree = nfree

        # Initialize result storage
        self._hessian: np.ndarray | None = None
        self._frequencies: np.ndarray | None = None
        self._normal_modes: np.ndarray | None = None
        self._zero_point_energy: float | None = None
        self._is_calculated = False
        self._direct_frequencies: np.ndarray | None = None  # For direct frequency calculation
        self._keep_indices: np.ndarray | None = None  # indices kept after removing trans/rot

    def calculate_hessian(self, method: str = "auto") -> np.ndarray:
        """Calculate Hessian matrix using the most appropriate method.

        Parameters
        ----------
        method : str
            Method to use: 'auto', 'direct_frequencies', 'direct', 'batch', or 'finite_differences'

        Returns:
        -------
        np.ndarray
            Hessian matrix (3N x 3N for N atoms)

        """
        if method == "auto":
            if self._supports_direct_frequencies():
                method = "direct_frequencies"
            elif (
                hasattr(self.calculator, "supports_batch_evaluation")
                and self.calculator.supports_batch_evaluation
            ):
                method = "batch"
            elif self._supports_direct_hessian():
                # Try direct method but be prepared to fall back
                method = "direct"
            else:
                method = "finite_differences"

        if method == "direct_frequencies":
            self._hessian = np.eye(3 * len(self.indices))  # Dummy hessian
            self._direct_frequencies = self._calculate_direct_frequencies()
        elif method == "direct":
            self._hessian = self._calculate_direct_hessian()
        elif method == "batch":
            self._hessian = self._calculate_hessian_batch()
        elif method == "finite_differences":
            hessian_calc = HessianCalculator(
                self.atoms,
                self.calculator,
                self.delta,
                method="central",
                richardson=self.richardson,
                delta2=self.delta2,
                indices=self.indices,
                verbose=self.verbose,
            )
            self._hessian = hessian_calc.calculate_numerical_hessian()
        else:
            msg = f"Unknown Hessian method: {method}"
            raise ValueError(msg)

        if self._hessian is None:
            msg = "Hessian calculation failed"
            raise RuntimeError(msg)

        # Validate Hessian and warn if issues detected
        validation_results = validate_hessian(self._hessian, warn_on_issues=True)
        if not validation_results["is_valid"]:
            logger.warning(
                "Hessian validation detected issues. Results may be unreliable. "
                f"Validation results: {validation_results}"
            )

        return self._hessian

    def _calculate_hessian_batch(self) -> np.ndarray:
        """Calculate Hessian using batch evaluation for improved performance."""
        if not (
            hasattr(self.calculator, "supports_batch_evaluation")
            and self.calculator.supports_batch_evaluation
        ):
            msg = "Calculator does not support batch evaluation"
            raise RuntimeError(msg)

        if self.verbose >= 1:
            logger.info("Using batch evaluation for Hessian calculation...")

        # Generate all displaced structures
        displaced_structures = self._generate_displaced_structures()

        # Calculate energies and forces for all structures in one batch
        batch_results = self.calculator.calculate_batch(
            displaced_structures,
            properties=["energy", "forces"],
        )

        # Construct Hessian matrix from batch results
        return self._construct_hessian_from_batch(batch_results)

    def _generate_displaced_structures(self) -> list[Atoms]:
        """Generate all displaced structures for finite differences."""
        displaced_structures = []

        # Add the original structure
        displaced_structures.append(self.atoms.copy())

        # Generate displaced structures
        for i in self.indices:
            for j in range(3):  # x, y, z directions
                # Positive displacement
                atoms_pos = self.atoms.copy()
                atoms_pos.positions[i, j] += self.delta
                displaced_structures.append(atoms_pos)

                # Negative displacement
                atoms_neg = self.atoms.copy()
                atoms_neg.positions[i, j] -= self.delta
                displaced_structures.append(atoms_neg)

        return displaced_structures

    def _construct_hessian_from_batch(self, batch_results: list[dict[str, Any]]) -> np.ndarray:
        """Construct Hessian matrix from batch calculation results."""
        n_atoms = len(self.indices)
        hessian = np.zeros((3 * n_atoms, 3 * n_atoms))

        # Calculate Hessian using finite differences
        result_idx = 1  # Start from index 1 (skip reference structure)

        for i in range(n_atoms):
            for j in range(3):  # x, y, z directions
                # Positive displacement
                pos_forces = batch_results[result_idx]["forces"]
                result_idx += 1

                # Negative displacement
                neg_forces = batch_results[result_idx]["forces"]
                result_idx += 1

                # Calculate second derivative using finite differences
                # d²E/dx² ≈ (F_neg - F_pos) / (2 * delta)
                hessian_row = 3 * i + j
                for k in range(n_atoms):
                    atom_k = self.indices[k]
                    for coord in range(3):
                        hessian_col = 3 * k + coord
                        hessian[hessian_row, hessian_col] = (
                            neg_forces[atom_k, coord] - pos_forces[atom_k, coord]
                        ) / (2 * self.delta)

        # Symmetrize for numerical stability
        hessian = 0.5 * (hessian + hessian.T)

        return cast(NDArray[np.float64], hessian)

    def _supports_direct_hessian(self) -> bool:
        """Check if calculator supports direct Hessian calculation."""
        # First check implemented_properties if available
        if hasattr(self.calculator, "implemented_properties"):
            return "hessian" in self.calculator.implemented_properties

        # Fallback: check for methods only if implemented_properties is not available
        return hasattr(self.calculator, "get_hessian") or hasattr(
            self.calculator,
            "calculate_hessian",
        )

    def _supports_direct_frequencies(self) -> bool:
        """Check if calculator supports direct frequency calculation."""
        if (
            hasattr(self.calculator, "implemented_properties")
            and "frequencies" in self.calculator.implemented_properties
        ):
            return True

        return hasattr(self.calculator, "get_frequencies") or hasattr(
            self.calculator,
            "calculate_frequencies",
        )

    def _calculate_direct_frequencies(self) -> np.ndarray:
        """Calculate frequencies directly from calculator (when supported)."""
        if (
            hasattr(self.calculator, "implemented_properties")
            and "frequencies" in self.calculator.implemented_properties
        ):
            return cast(
                NDArray[np.float64], self.calculator.get_property("frequencies", self.atoms)
            )

        if hasattr(self.calculator, "get_frequencies"):
            return cast(NDArray[np.float64], self.calculator.get_frequencies(self.atoms))
        return cast(NDArray[np.float64], self.calculator.calculate_frequencies(self.atoms))

    def _calculate_direct_hessian(self) -> np.ndarray:
        """Calculate Hessian directly from calculator (when supported)."""
        if (
            hasattr(self.calculator, "implemented_properties")
            and "hessian" in self.calculator.implemented_properties
        ):
            return cast(NDArray[np.float64], self.calculator.get_property("hessian", self.atoms))

        if hasattr(self.calculator, "get_hessian"):
            return cast(NDArray[np.float64], self.calculator.get_hessian(self.atoms))
        return cast(NDArray[np.float64], self.calculator.calculate_hessian(self.atoms))

    def diagonalize_hessian(self) -> tuple[np.ndarray, np.ndarray]:
        """Diagonalize mass-weighted Hessian to get normal modes and frequencies.

        Returns:
        -------
        Tuple[np.ndarray, np.ndarray]
            frequencies (in cm^-1) and normal mode eigenvectors

        """
        if self._hessian is None:
            msg = "Hessian not calculated. Call calculate_hessian() first."
            raise QMEError(msg)

        frequencies, eigenvectors = diagonalize_mass_weighted_hessian(
            self._hessian, self.atoms, self.indices
        )

        self._frequencies = frequencies
        self._normal_modes = eigenvectors
        self._is_calculated = True

        return frequencies, eigenvectors

    def get_frequencies(self, unit: str = "cm-1") -> np.ndarray:
        """Get vibrational frequencies in specified units.

        Parameters
        ----------
        unit : str
            Unit for frequencies: 'cm-1', 'meV', 'THz'

        Returns:
        -------
        np.ndarray
            Vibrational frequencies, excluding translational and rotational modes

        """
        # Gather all frequencies
        if self._direct_frequencies is not None:
            freq_all = self._direct_frequencies
        else:
            if not self._is_calculated:
                self.calculate_hessian()
                self.diagonalize_hessian()

            if self._frequencies is None:
                msg = "Frequencies not calculated"
                raise QMEError(msg)

            freq_all = self._frequencies

        # Remove translational and rotational modes: drop nfree frequencies closest to zero by |freq|
        idx_sorted = np.argsort(np.abs(freq_all))
        keep_idx = idx_sorted[self.nfree :]
        self._keep_indices = keep_idx
        frequencies = freq_all[keep_idx]

        return convert_frequency_unit(frequencies, unit)

    def get_normal_modes(self) -> np.ndarray:
        """Get normal mode vectors.

        Returns:
        -------
        np.ndarray
            Normal mode vectors (3N x 3N-6/5 for vibrational modes only)

        """
        if not self._is_calculated:
            self.calculate_hessian()
            self.diagonalize_hessian()

        if self._normal_modes is None:
            msg = "Normal modes not calculated"
            raise QMEError(msg)

        # Ensure keep indices exist (computed in get_frequencies)
        if self._keep_indices is None:
            _ = self.get_frequencies()  # sets _keep_indices

        return self._normal_modes[:, self._keep_indices]

    def is_transition_state(
        self,
        threshold: float = 50.0,
    ) -> dict[str, bool | int | list[float] | str]:
        """Check if structure is a transition state (exactly one imaginary frequency).

        Parameters
        ----------
        threshold : float
            Minimum frequency magnitude in cm^-1 to consider significant

        Returns:
        -------
        dict[str, Union[bool, int, list[float], str]]
            Dictionary with TS verification results containing:
            - is_transition_state: Whether structure is a TS (bool)
            - n_imaginary_frequencies: Number of imaginary frequencies (int)
            - imaginary_frequencies: List of imaginary frequencies (list[float])
            - n_near_zero_frequencies: Number of near-zero frequencies (int)
            - all_frequencies: All frequencies (list[float])
            - threshold: Threshold used (float)
            - assessment: Assessment string (str)

        """
        frequencies = self.get_frequencies()

        # Count imaginary frequencies above threshold
        imaginary_freqs = frequencies[frequencies < -threshold]
        n_imaginary = len(imaginary_freqs)

        # Count near-zero frequencies (should be excluded)
        near_zero = np.abs(frequencies) < threshold
        n_near_zero = int(np.sum(near_zero))

        is_ts = n_imaginary == 1

        return {
            "is_transition_state": is_ts,
            "n_imaginary_frequencies": n_imaginary,
            "imaginary_frequencies": imaginary_freqs.tolist(),
            "n_near_zero_frequencies": n_near_zero,
            "all_frequencies": frequencies.tolist(),
            "threshold": threshold,
            "assessment": self._assess_stationary_point(n_imaginary, n_near_zero, threshold),
        }

    def _assess_stationary_point(self, n_imaginary: int, n_near_zero: int, threshold: float) -> str:
        """Assess type of stationary point based on frequency analysis."""
        if n_imaginary == 0:
            return "Minimum (no imaginary frequencies)"
        if n_imaginary == 1:
            return "First-order transition state (one imaginary frequency)"
        if n_imaginary > 1:
            return f"Higher-order saddle point ({n_imaginary} imaginary frequencies)"
        return "Undetermined stationary point type"

    def is_minima(
        self,
        threshold: float = 50.0,
        small_negative_cutoff: float = -10.0,
    ) -> dict[str, bool | int | list[float] | str]:
        """Check if structure is a minimum (no significant imaginary frequencies).

        Parameters
        ----------
        threshold : float
            Minimum frequency magnitude in cm^-1 to consider significant
        small_negative_cutoff : float
            Maximum negative frequency in cm^-1 to consider as "small negative"
            (likely numerical noise, not a true imaginary frequency)

        Returns:
        -------
        dict[str, Union[bool, int, list[float], str]]
            Dictionary with minima verification results containing:
            - is_minimum: Whether structure is a minimum (bool)
            - n_significant_imaginary_frequencies: Number of significant imaginary frequencies (int)
            - n_small_negative_frequencies: Number of small negative frequencies (int)
            - significant_imaginary_frequencies: List of significant imaginary frequencies (list[float])
            - small_negative_frequencies: List of small negative frequencies (list[float])
            - n_near_zero_frequencies: Number of near-zero frequencies (int)
            - all_frequencies: All frequencies (list[float])
            - threshold: Threshold used (float)
            - small_negative_cutoff: Small negative cutoff used (float)
            - assessment: Assessment string (str)

        """
        frequencies = self.get_frequencies()

        # Count significant imaginary frequencies above threshold
        significant_imaginary = frequencies[frequencies < -threshold]
        n_significant_imaginary = len(significant_imaginary)

        # Count small negative frequencies (likely numerical noise)
        small_negative = frequencies[(frequencies < 0) & (frequencies > small_negative_cutoff)]
        n_small_negative = len(small_negative)

        # Count near-zero frequencies (should be excluded)
        near_zero = np.abs(frequencies) < threshold
        n_near_zero = int(np.sum(near_zero))

        # A structure is a minimum if it has no significant imaginary frequencies
        # Small negative frequencies (above cutoff) are considered acceptable
        is_minimum = n_significant_imaginary == 0

        return {
            "is_minimum": is_minimum,
            "n_significant_imaginary_frequencies": n_significant_imaginary,
            "n_small_negative_frequencies": n_small_negative,
            "significant_imaginary_frequencies": significant_imaginary.tolist(),
            "small_negative_frequencies": small_negative.tolist(),
            "n_near_zero_frequencies": n_near_zero,
            "all_frequencies": frequencies.tolist(),
            "threshold": threshold,
            "small_negative_cutoff": small_negative_cutoff,
            "assessment": self._assess_stationary_point_minima(
                n_significant_imaginary,
                n_small_negative,
                n_near_zero,
                threshold,
                small_negative_cutoff,
            ),
        }

    def _assess_stationary_point_minima(
        self,
        n_significant_imaginary: int,
        n_small_negative: int,
        n_near_zero: int,
        threshold: float,
        small_negative_cutoff: float,
    ) -> str:
        """Assess type of stationary point for minima validation."""
        if n_significant_imaginary == 0:
            if n_small_negative > 0:
                return (
                    f"Minimum (no significant imaginary frequencies, "
                    f"{n_small_negative} small negative frequencies likely numerical noise)"
                )
            return "Minimum (no imaginary frequencies)"
        if n_significant_imaginary == 1:
            return "First-order transition state (one significant imaginary frequency)"
        if n_significant_imaginary > 1:
            return (
                f"Higher-order saddle point "
                f"({n_significant_imaginary} significant imaginary frequencies)"
            )
        return "Undetermined stationary point type"

    def get_zero_point_energy(self) -> float:
        """Calculate zero-point vibrational energy.

        Returns:
        -------
        float
            Zero-point energy in eV

        """
        if self._zero_point_energy is not None:
            return self._zero_point_energy

        frequencies = self.get_frequencies()  # in cm^-1
        # Only include real (positive) frequencies
        real_frequencies = frequencies[frequencies > 0]

        # Convert frequencies from cm^-1 to eV, then calculate ZPE
        # ZPE = (1/2) * Σ hνᵢ where νᵢ are frequencies in eV
        freq_eV = real_frequencies * units.invcm  # Convert cm^-1 to eV
        zpe = 0.5 * np.sum(freq_eV)  # ZPE in eV

        self._zero_point_energy = zpe
        return zpe

    def get_thermodynamic_properties(
        self,
        temperature: float = 298.15,
        *,
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
        # Full thermodynamics option
        complete: bool = False,
    ) -> dict[str, float | int | list[float]]:
        """Calculate thermodynamic properties at given temperature.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin
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
        complete : bool
            If True, calculate complete thermodynamics with all contributions.
            If False, return vibrational-only properties (backward compatible).

        Returns:
        -------
        dict[str, Union[float, int, list[float]]]
            Dictionary with thermodynamic properties.
            If complete=True, includes all contributions (trans, rot, vib, elec).
            If complete=False, returns vibrational-only (backward compatible).

        """
        frequencies = self.get_frequencies()  # in cm^-1
        # Only include real frequencies
        real_frequencies = frequencies[frequencies > 0]

        # Initialize comprehensive thermodynamics calculator
        thermo_props = ThermodynamicProperties(
            real_frequencies,
            self.atoms,
            temperature=temperature,
            method=method,
            freq_cutoff=freq_cutoff,
            freq_scale_factor=freq_scale_factor,
            rotational_temperatures=rotational_temperatures,
            rotational_constants=rotational_constants,
            symmetry_number=symmetry_number,
            point_group=point_group,
            linear=linear,
            multiplicity=multiplicity,
            solvent=solvent,
            concentration=concentration,
        )

        if complete:
            # Return complete thermodynamics with all contributions
            complete_thermo = thermo_props.calculate_complete_thermodynamics()

            # Merge with vibrational-only data for compatibility
            vibrational_only = {
                "temperature": temperature,
                "zero_point_energy": thermo_props.calculate_zero_point_energy(),
                "internal_energy": thermo_props.internal_energy_vibrational(),
                "heat_capacity": thermo_props.heat_capacity_vibrational(),
                "entropy": thermo_props.entropy_vibrational(),
                "n_vibrational_modes": len(real_frequencies),
                "frequencies_cm_1": real_frequencies.tolist(),
            }

            # Combine both dictionaries
            return {**vibrational_only, **complete_thermo}
        else:
            # Backward compatible: return vibrational-only properties
            zpe = thermo_props.calculate_zero_point_energy()
            entropy = thermo_props.entropy_vibrational()
            internal_energy = thermo_props.internal_energy_vibrational()
            heat_capacity = thermo_props.heat_capacity_vibrational()

            return {
                "temperature": temperature,
                "zero_point_energy": zpe,
                "internal_energy": internal_energy,
                "heat_capacity": heat_capacity,
                "entropy": entropy,
                "n_vibrational_modes": len(real_frequencies),
                "frequencies_cm_1": real_frequencies.tolist(),
            }

    def write_mode_trajectory(
        self,
        mode_index: int,
        filename: str | None = None,
        amplitude: float = 1.0,
        nframes: int = 20,
    ) -> None:
        """Write trajectory showing normal mode motion.

        Parameters
        ----------
        mode_index : int
            Index of normal mode (0-based, after removing trans/rot modes)
        filename : str, optional
            Output filename. If None, uses mode_{index}.traj
        amplitude : float
            Amplitude of motion in Å
        nframes : int
            Number of frames in trajectory

        """
        if not self._is_calculated:
            self.calculate_hessian()
            self.diagonalize_hessian()

        if filename is None:
            filename = f"mode_{mode_index}.traj"

        normal_modes = self.get_normal_modes()
        if mode_index >= normal_modes.shape[1]:
            msg = f"Mode index {mode_index} out of range"
            raise ValueError(msg)

        mode_vector = normal_modes[:, mode_index].reshape(-1, 3)

        # Create trajectory
        from ase.io import write

        trajectory = []
        for i in range(nframes):
            phase = 2 * np.pi * i / nframes
            displacement = amplitude * np.sin(phase) * mode_vector

            atoms_displaced = self.atoms.copy()
            atoms_displaced.positions[self.indices] += displacement
            trajectory.append(atoms_displaced)

        write(filename, trajectory)
        if self.verbose >= 1:
            logger.info(f"Normal mode trajectory written to {filename}")


__all__ = [
    "FrequencyAnalysis",
    "HessianCalculator",
    "ThermodynamicProperties",
]
