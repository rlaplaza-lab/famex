"""Vibrational frequency analysis for FAMEX."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from ase import Atoms, units
from numpy.typing import NDArray

from famex.analysis.hessian import HessianCalculator
from famex.analysis.molecular_properties import determine_degrees_of_freedom
from famex.analysis.normal_modes import convert_frequency_unit, diagonalize_mass_weighted_hessian
from famex.analysis.thermodynamics import ThermodynamicProperties
from famex.analysis.utils import has_calculator_property, validate_indices
from famex.analysis.validation import validate_hessian
from famex.utils.logging import get_famex_logger
from famex.utils.validation import FAMEXError

logger = get_famex_logger(__name__)


class FrequencyAnalysis:
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
        self.atoms = atoms
        self.calculator = calculator
        self.atoms.calc = calculator
        self.delta = delta
        self.richardson = richardson
        self.delta2 = delta2
        self.indices = validate_indices(atoms, indices)
        self.verbose = verbose
        self.nfree = (
            nfree if nfree is not None else determine_degrees_of_freedom(atoms, self.indices)
        )

        self._hessian: np.ndarray | None = None
        self._frequencies: np.ndarray | None = None
        self._normal_modes: np.ndarray | None = None
        self._zero_point_energy: float | None = None
        self._is_calculated = False
        self._direct_frequencies: np.ndarray | None = None
        self._keep_indices: np.ndarray | None = None

    def calculate_hessian(self, method: str = "auto") -> np.ndarray:
        if method == "autoselect":
            return self._calculate_hessian_autoselect()

        if method == "auto":
            if has_calculator_property(self.calculator, "frequencies"):
                method = "direct_frequencies"
            elif (
                hasattr(self.calculator, "supports_batch_evaluation")
                and self.calculator.supports_batch_evaluation
            ):
                method = "batch"
            elif has_calculator_property(self.calculator, "hessian"):
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

        displaced_structures = self._generate_displaced_structures()

        batch_results = self.calculator.calculate_batch(
            displaced_structures,
            properties=["energy", "forces"],
        )

        return self._construct_hessian_from_batch(batch_results)

    def _generate_displaced_structures(self) -> list[Atoms]:
        displaced_structures = []

        displaced_structures.append(self.atoms.copy())

        for i in self.indices:
            for j in range(3):
                atoms_pos = self.atoms.copy()
                atoms_pos.positions[i, j] += self.delta
                displaced_structures.append(atoms_pos)

                atoms_neg = self.atoms.copy()
                atoms_neg.positions[i, j] -= self.delta
                displaced_structures.append(atoms_neg)

        if self.verbose >= 2:
            logger.debug(f"Generated {len(displaced_structures)} structures for batch calculation")

        return displaced_structures

    def _construct_hessian_from_batch(self, batch_results: list[dict[str, Any]]) -> np.ndarray:
        n_atoms = len(self.indices)
        n_coords = 3 * n_atoms
        hessian = np.zeros((n_coords, n_coords))

        expected_results = 1 + 2 * n_coords
        if len(batch_results) < expected_results:
            msg = (
                f"Insufficient batch results: expected {expected_results}, got {len(batch_results)}"
            )
            raise RuntimeError(msg)

        result_idx = 1

        for i in range(n_atoms):
            for j in range(3):
                pos_forces = batch_results[result_idx].get("forces")
                if pos_forces is None:
                    msg = f"Missing forces in batch result {result_idx}"
                    raise RuntimeError(msg)
                result_idx += 1

                neg_forces = batch_results[result_idx].get("forces")
                if neg_forces is None:
                    msg = f"Missing forces in batch result {result_idx}"
                    raise RuntimeError(msg)
                result_idx += 1

                hessian_row = 3 * i + j
                for k in range(n_atoms):
                    atom_k = self.indices[k]
                    for coord in range(3):
                        hessian_col = 3 * k + coord
                        hessian[hessian_row, hessian_col] = (
                            neg_forces[atom_k, coord] - pos_forces[atom_k, coord]
                        ) / (2 * self.delta)

        hessian = cast(NDArray[np.float64], 0.5 * (hessian + hessian.T))

        if self.verbose >= 2:
            logger.debug("Hessian constructed from batch results")

        return cast(NDArray[np.float64], hessian)

    def _calculate_direct_frequencies(self) -> np.ndarray:
        from famex.analysis.utils import get_calculator_property

        return cast(
            NDArray[np.float64],
            get_calculator_property(self.calculator, "frequencies", self.atoms),
        )

    def _calculate_direct_hessian(self) -> np.ndarray:
        from famex.analysis.utils import get_calculator_property

        return cast(
            NDArray[np.float64],
            get_calculator_property(self.calculator, "hessian", self.atoms),
        )

    def _calculate_hessian_autoselect(self) -> np.ndarray:
        if self.verbose >= 1:
            logger.info("Hessian autoselect: choosing optimal method...")

        if has_calculator_property(self.calculator, "hessian"):
            if self.verbose >= 1:
                logger.info("  Trying analytical Hessian...")
            try:
                analytical_hessian = self._calculate_direct_hessian()
                if np.any(np.isnan(analytical_hessian)) or np.any(np.isinf(analytical_hessian)):
                    logger.warning(
                        "  Analytical Hessian contains NaN/Inf, falling back to finite differences"
                    )
                else:
                    asymmetry = np.max(np.abs(analytical_hessian - analytical_hessian.T))
                    if asymmetry >= 1e-4:
                        logger.warning(
                            f"  Analytical Hessian has high asymmetry ({asymmetry:.2e}), "
                            "falling back to finite differences"
                        )
                    else:
                        if self.verbose >= 1:
                            logger.info("  ✓ Using analytical Hessian")
                        self._hessian = analytical_hessian
                        return analytical_hessian
            except (AttributeError, RuntimeError) as e:
                logger.warning(f"  Analytical Hessian failed: {e}, falling back to FD")

        # Estimate force noise
        if self.verbose >= 1:
            logger.info("  Estimating force noise...")
        try:
            from famex.analysis.noise_estimation import estimate_force_noise

            force_noise = estimate_force_noise(
                self.atoms, self.calculator, n_samples=5, indices=self.indices
            )
            if self.verbose >= 1:
                logger.info(f"  Force noise: {force_noise:.2e} eV/Å")
        except Exception as e:
            logger.warning(f"  Force noise estimation failed: {e}, assuming moderate noise")
            force_noise = 1e-5  # Default moderate noise

        # Step 3: Try energy-based FD if force noise is very high
        if force_noise > 1e-4:
            if self.verbose >= 1:
                logger.info("  High force noise detected, trying energy-based FD...")
            try:
                from famex.analysis.hessian_energy import EnergyBasedHessianCalculator

                energy_calc = EnergyBasedHessianCalculator(
                    self.atoms,
                    self.calculator,
                    delta=self.delta,
                    indices=self.indices,
                    verbose=0 if self.verbose < 2 else self.verbose,
                )
                energy_hessian = energy_calc.calculate_energy_hessian()
                if np.any(np.isnan(energy_hessian)) or np.any(np.isinf(energy_hessian)):
                    logger.warning("  Energy-based FD failed: invalid Hessian, falling back")
                else:
                    if self.verbose >= 1:
                        logger.info("  ✓ Using energy-based FD Hessian")
                    self._hessian = energy_hessian
                    return energy_hessian
            except (RuntimeError, ValueError) as e:
                logger.warning(f"  Energy-based FD failed: {e}, falling back to force-based FD")
        elif self.verbose >= 1:
            logger.info("  Force noise acceptable, using force-based FD")

        if self.verbose >= 1:
            logger.info("  Using high-quality force-based FD (5-point + Richardson)...")
        hessian_calc = HessianCalculator(
            self.atoms,
            self.calculator,
            self.delta,
            method="5point",  # Higher order for better accuracy
            richardson=True,
            delta2=self.delta / 2.0 if self.delta2 is None else self.delta2,
            indices=self.indices,
            verbose=self.verbose,
            adaptive_delta=False,  # Fixed delta with Richardson is optimal cost/accuracy
        )
        self._hessian = hessian_calc.calculate_numerical_hessian()

        if self.verbose >= 1:
            logger.info("  ✓ Force-based FD Hessian computed")
        return self._hessian

    def diagonalize_hessian(self) -> tuple[np.ndarray, np.ndarray]:
        """Diagonalize mass-weighted Hessian to get normal modes and frequencies.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            frequencies (in cm^-1) and normal mode eigenvectors.
            Frequencies are always real numbers (complex parts are discarded).

        """
        if self._hessian is None:
            msg = "Hessian not calculated. Call calculate_hessian() first."
            raise FAMEXError(msg)

        frequencies, eigenvectors = diagonalize_mass_weighted_hessian(
            self._hessian, self.atoms, self.indices
        )

        # Ensure frequencies are real (discard any imaginary parts)
        # In normal mode analysis, frequencies should be real numbers.
        # The sign convention (positive/negative) handles imaginary modes.
        frequencies = np.real(frequencies)

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

        Returns
        -------
        np.ndarray
            Vibrational frequencies, excluding translational and rotational modes.
            Frequencies are always real numbers (complex parts are discarded).

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
                raise FAMEXError(msg)

            freq_all = self._frequencies

        # Ensure frequencies are real (discard any imaginary parts)
        # In normal mode analysis, frequencies should be real numbers.
        # The sign convention (positive/negative) handles imaginary modes.
        freq_all = np.real(freq_all)

        idx_sorted = np.argsort(np.abs(freq_all))
        keep_idx = idx_sorted[self.nfree :]
        self._keep_indices = keep_idx
        frequencies = freq_all[keep_idx]

        return convert_frequency_unit(frequencies, unit)

    def get_normal_modes(self) -> np.ndarray:
        """Get normal mode vectors.

        Returns
        -------
        np.ndarray
            Normal mode vectors (3N x 3N-6/5 for vibrational modes only)

        """
        if not self._is_calculated:
            self.calculate_hessian()
            self.diagonalize_hessian()

        if self._normal_modes is None:
            msg = "Normal modes not calculated"
            raise FAMEXError(msg)

        # Ensure keep indices exist (computed in get_frequencies)
        if self._keep_indices is None:
            _ = self.get_frequencies()

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

        Returns
        -------
        dict[str, bool | int | list[float] | str]
            Dictionary with TS verification results containing:
            - is_transition_state: Whether structure is a TS (bool)
            - n_imaginary_frequencies: Number of imaginary frequencies (int)
            - imaginary_frequencies: List of imaginary frequencies (list[float])
            - n_near_zero_frequencies: Number of near-zero frequencies (int)
            - all_frequencies: Vibrational frequencies (list[float])
                Note: This contains vibrational frequencies (trans/rot modes removed).
                For all frequencies including trans/rot, use the top-level all_frequencies
                from explorer.calculate_frequencies().
            - threshold: Threshold used (float)
            - assessment: Assessment string (str)

        """
        frequencies = self.get_frequencies()

        # Frequencies are now guaranteed to be real (from get_frequencies)
        # Negative frequencies indicate imaginary modes
        imaginary_freqs_list = []
        for freq in frequencies:
            if freq < -threshold:
                imaginary_freqs_list.append(freq)

        imaginary_freqs = np.array(imaginary_freqs_list)
        n_imaginary = len(imaginary_freqs)

        freq_magnitudes = np.abs(frequencies)
        near_zero = freq_magnitudes < threshold
        n_near_zero = int(np.sum(near_zero))

        is_ts = n_imaginary == 1

        return {
            "is_transition_state": is_ts,
            "n_imaginary_frequencies": n_imaginary,
            "imaginary_frequencies": imaginary_freqs.tolist(),
            "n_near_zero_frequencies": n_near_zero,
            "all_frequencies": frequencies.tolist(),  # Note: vibrational frequencies (trans/rot removed)
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

        Returns
        -------
        dict[str, bool | int | list[float] | str]
            Dictionary with minima verification results containing:
            - is_minimum: Whether structure is a minimum (bool)
            - n_significant_imaginary_frequencies: Number of significant imaginary frequencies (int)
            - n_small_negative_frequencies: Number of small negative frequencies (int)
            - significant_imaginary_frequencies: List of significant imaginary frequencies (list[float])
            - small_negative_frequencies: List of small negative frequencies (list[float])
            - n_near_zero_frequencies: Number of near-zero frequencies (int)
            - all_frequencies: Vibrational frequencies (list[float])
                Note: This contains vibrational frequencies (trans/rot modes removed).
                For all frequencies including trans/rot, use the top-level all_frequencies
                from explorer.calculate_frequencies().
            - threshold: Threshold used (float)
            - small_negative_cutoff: Small negative cutoff used (float)
            - assessment: Assessment string (str)

        """
        frequencies = self.get_frequencies()

        # Frequencies are now guaranteed to be real (from get_frequencies)
        # Negative frequencies indicate imaginary modes
        significant_imaginary = frequencies[frequencies < -threshold]
        n_significant_imaginary = len(significant_imaginary)

        small_negative = frequencies[(frequencies < 0) & (frequencies > small_negative_cutoff)]
        n_small_negative = len(small_negative)

        near_zero = np.abs(frequencies) < threshold
        n_near_zero = int(np.sum(near_zero))

        is_minimum = n_significant_imaginary == 0

        return {
            "is_minimum": is_minimum,
            "n_significant_imaginary_frequencies": n_significant_imaginary,
            "n_small_negative_frequencies": n_small_negative,
            "significant_imaginary_frequencies": significant_imaginary.tolist(),
            "small_negative_frequencies": small_negative.tolist(),
            "n_near_zero_frequencies": n_near_zero,
            "all_frequencies": frequencies.tolist(),  # Note: vibrational frequencies (trans/rot removed)
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

        Returns
        -------
        float
            Zero-point energy in eV

        """
        if self._zero_point_energy is not None:
            return self._zero_point_energy

        frequencies = self.get_frequencies()
        real_frequencies = frequencies[np.real(frequencies) > 0]
        real_frequencies = np.real(real_frequencies)

        freq_eV = real_frequencies * units.invcm
        zpe = float(0.5 * np.sum(freq_eV))  # ZPE in eV, ensure float type

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

        Returns
        -------
        dict[str, float | int | list[float]]
            Dictionary with thermodynamic properties.
            If complete=True, includes all contributions (trans, rot, vib, elec).
            If complete=False, returns vibrational-only (backward compatible).

        """
        frequencies = self.get_frequencies()
        real_frequencies = frequencies[np.real(frequencies) > 0]
        real_frequencies = np.real(real_frequencies)

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
