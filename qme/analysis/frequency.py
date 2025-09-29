"""
Vibrational frequency analysis for QME.

This module provides comprehensive vibrational frequency analysis capabilities including:
- Hessian matrix calculations using finite differences or direct methods
- Normal mode analysis and vibrational frequencies
- Transition state verification
- Thermodynamic property calculations
- IR spectrum prediction

Based on ASE's Vibrations class but extended for QME functionality with automatic
method selection and enhanced calculator integration.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ase import Atoms, units
from ase.parallel import world
from ase.thermochemistry import HarmonicThermo
from scipy.linalg import eigh

from .validation import QMEError


class FrequencyAnalysis:
    """
    Vibrational frequency analysis with automatic Hessian method selection.

    Supports both direct Hessian calculation (when available from calculator)
    and finite difference Hessian calculation as fallback.
    """

    def __init__(
        self,
        atoms: Atoms,
        calculator,
        delta: float = 0.01,
        nfree: Optional[int] = None,
        indices: Optional[List[int]] = None,
    ):
        """
        Initialize frequency analysis.

        Parameters:
        -----------
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
        """
        self.atoms = atoms.copy()
        self.calculator = calculator
        self.atoms.calc = calculator
        self.delta = delta
        self.indices = indices if indices is not None else list(range(len(atoms)))

        # Determine degrees of freedom to remove
        if nfree is None:
            self.nfree = self._determine_nfree()
        else:
            self.nfree = nfree

        # Initialize result storage
        self._hessian = None
        self._frequencies = None
        self._normal_modes = None
        self._zero_point_energy = None
        self._is_calculated = False

    def _determine_nfree(self) -> int:
        """Determine number of degrees of freedom to remove (translation + rotation)."""
        if len(self.indices) == 1:
            return 3  # Only translation for single atom
        elif len(self.indices) == 2:
            return 5  # 3 translation + 2 rotation for 2-atom molecules (always linear)
        elif self._is_linear():
            return 5  # 3 translation + 2 rotation for linear molecules
        else:
            return 6  # 3 translation + 3 rotation for non-linear molecules

    def _is_linear(self) -> bool:
        """Check if molecule is linear."""
        if len(self.indices) <= 2:
            return True

        positions = self.atoms.positions[self.indices]
        if len(positions) == 3:
            # For 3 atoms, check if they are collinear
            v1 = positions[1] - positions[0]
            v2 = positions[2] - positions[0]
            cross = np.cross(v1, v2)
            return bool(np.linalg.norm(cross) < 1e-3)

        # For more atoms, use moment of inertia approach
        return self._check_linearity_inertia()

    def _check_linearity_inertia(self) -> bool:
        """Check linearity using moment of inertia tensor."""
        atoms_subset = self.atoms[self.indices]
        masses = atoms_subset.get_masses()
        positions = atoms_subset.positions

        # Center of mass
        com = atoms_subset.get_center_of_mass()
        positions_centered = positions - com

        # Moment of inertia tensor
        inertia_tensor = np.zeros((3, 3))
        for i, (pos, mass) in enumerate(zip(positions_centered, masses)):
            inertia_tensor += mass * (np.dot(pos, pos) * np.eye(3) - np.outer(pos, pos))

        # Eigenvalues of moment of inertia tensor
        eigenvalues = np.linalg.eigvals(inertia_tensor)
        eigenvalues = np.sort(eigenvalues)

        # Linear if smallest eigenvalue is essentially zero
        return eigenvalues[0] < 1e-6

    def calculate_hessian(self, method: str = "auto") -> np.ndarray:
        """
        Calculate Hessian matrix using the most appropriate method.

        Parameters:
        -----------
        method : str
            Method to use: 'auto', 'direct', or 'finite_differences'

        Returns:
        --------
        np.ndarray
            Hessian matrix (3N x 3N for N atoms)
        """
        if method == "auto":
            # Check if calculator supports direct Hessian
            if self._supports_direct_hessian():
                method = "direct"
            else:
                method = "finite_differences"

        if method == "direct":
            self._hessian = self._calculate_direct_hessian()
        elif method == "finite_differences":
            hessian_calc = HessianCalculator(
                self.atoms, self.calculator, self.delta, indices=self.indices
            )
            self._hessian = hessian_calc.calculate_numerical_hessian()
        else:
            raise ValueError(f"Unknown Hessian method: {method}")

        return self._hessian

    def _supports_direct_hessian(self) -> bool:
        """Check if calculator supports direct Hessian calculation."""
        # Check if calculator has hessian method or supports analytical second derivatives
        if hasattr(self.calculator, "get_hessian"):
            return True
        if hasattr(self.calculator, "calculate_hessian"):
            return True
        # For now, assume no direct Hessian support for ML potentials
        # This can be extended when ML calculators implement analytical Hessians
        return False

    def _calculate_direct_hessian(self) -> np.ndarray:
        """Calculate Hessian directly from calculator (when supported)."""
        if hasattr(self.calculator, "get_hessian"):
            return self.calculator.get_hessian(self.atoms)
        elif hasattr(self.calculator, "calculate_hessian"):
            return self.calculator.calculate_hessian(self.atoms)
        else:
            raise QMEError("Calculator does not support direct Hessian calculation")

    def diagonalize_hessian(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonalize mass-weighted Hessian to get normal modes and frequencies.

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            eigenvalues (frequencies^2) and eigenvectors (normal modes)
        """
        if self._hessian is None:
            raise QMEError("Hessian not calculated. Call calculate_hessian() first.")

        # Mass-weight the Hessian
        masses = self.atoms.get_masses()[self.indices]
        mass_weights = np.repeat(masses, 3) ** -0.5
        mass_weighted_hessian = self._hessian * np.outer(mass_weights, mass_weights)

        # Diagonalize
        eigenvalues, eigenvectors = eigh(mass_weighted_hessian)

        # Convert eigenvalues to frequencies (in cm^-1)
        # eigenvalues are in eV/Å^2/amu, convert to cm^-1
        # Convert eigenvalues to frequencies (in cm^-1)
        # eigenvalues are in eV/Å^2/amu, convert to cm^-1
        conversion = np.sqrt(units._e / units._amu) / (2 * np.pi * units._c * 100)
        frequencies = np.sqrt(np.abs(eigenvalues)) * conversion

        # Handle imaginary frequencies
        imaginary_mask = eigenvalues < 0
        frequencies[imaginary_mask] *= -1  # Make imaginary frequencies negative

        # Sort by frequency (most negative first)
        sort_indices = np.argsort(frequencies)
        frequencies = frequencies[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices]

        # Normalize eigenvectors properly (mass-weighted)
        for i in range(len(eigenvectors[0])):
            eigenvectors[:, i] /= mass_weights
            eigenvectors[:, i] /= np.linalg.norm(eigenvectors[:, i])

        self._frequencies = frequencies
        self._normal_modes = eigenvectors
        self._is_calculated = True

        return frequencies, eigenvectors

    def get_frequencies(self, unit: str = "cm-1") -> np.ndarray:
        """
        Get vibrational frequencies in specified units.

        Parameters:
        -----------
        unit : str
            Unit for frequencies: 'cm-1', 'meV', 'THz'

        Returns:
        --------
        np.ndarray
            Vibrational frequencies, excluding translational and rotational modes
        """
        if not self._is_calculated:
            self.calculate_hessian()
            self.diagonalize_hessian()

        if self._frequencies is None:
            raise QMEError("Frequencies not calculated")

        frequencies = self._frequencies[self.nfree :].copy()

        if unit == "cm-1":
            return frequencies
        elif unit == "meV":
            return frequencies * units._hplanck * units._c * 100 / units._e * 1000
        elif unit == "THz":
            return frequencies * units._c * 100 / 1e12
        else:
            raise ValueError(f"Unknown frequency unit: {unit}")

    def get_normal_modes(self) -> np.ndarray:
        """
        Get normal mode vectors.

        Returns:
        --------
        np.ndarray
            Normal mode vectors (3N x 3N-6/5 for vibrational modes only)
        """
        if not self._is_calculated:
            self.calculate_hessian()
            self.diagonalize_hessian()

        if self._normal_modes is None:
            raise QMEError("Normal modes not calculated")

        return self._normal_modes[:, self.nfree :]

    def is_transition_state(self, threshold: float = 50.0) -> Dict[str, Any]:
        """
        Check if structure is a transition state (exactly one imaginary frequency).

        Parameters:
        -----------
        threshold : float
            Minimum frequency magnitude in cm^-1 to consider significant

        Returns:
        --------
        Dict[str, Any]
            Dictionary with TS verification results
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
            "assessment": self._assess_stationary_point(
                n_imaginary, n_near_zero, threshold
            ),
        }

    def _assess_stationary_point(
        self, n_imaginary: int, n_near_zero: int, threshold: float
    ) -> str:
        """Assess type of stationary point based on frequency analysis."""
        if n_imaginary == 0:
            return "Minimum (no imaginary frequencies)"
        elif n_imaginary == 1:
            return "First-order transition state (one imaginary frequency)"
        elif n_imaginary > 1:
            return f"Higher-order saddle point ({n_imaginary} imaginary frequencies)"
        else:
            return "Undetermined stationary point type"

    def get_zero_point_energy(self) -> float:
        """
        Calculate zero-point vibrational energy.

        Returns:
        --------
        float
            Zero-point energy in eV
        """
        if self._zero_point_energy is not None:
            return self._zero_point_energy

        frequencies = self.get_frequencies()  # in cm^-1
        # Only include real (positive) frequencies
        real_frequencies = frequencies[frequencies > 0]

        # Convert to eV: E = 0.5 * h * c * nu (in cm^-1)
        zpe = (
            0.5 * np.sum(real_frequencies) * units._hplanck * units._c * 100 / units._e
        )

        self._zero_point_energy = zpe
        return zpe

    def get_thermodynamic_properties(
        self, temperature: float = 298.15
    ) -> Dict[str, float]:
        """
        Calculate thermodynamic properties at given temperature.

        Parameters:
        -----------
        temperature : float
            Temperature in Kelvin

        Returns:
        --------
        Dict[str, float]
            Dictionary with thermodynamic properties (all in eV except entropy in eV/K)
        """
        frequencies = self.get_frequencies()  # in cm^-1
        # Only include real frequencies
        real_frequencies = frequencies[frequencies > 0]

        # freq_eV not used here since we use ASE's HarmonicThermo

        # Use ASE's HarmonicThermo for consistency
        thermo = HarmonicThermo(real_frequencies * units._hplanck * units._c * 100)

        # Get thermodynamic contributions
        zpe = self.get_zero_point_energy()
        entropy = thermo.get_entropy(temperature, verbose=False)
        internal_energy = thermo.get_internal_energy(temperature, verbose=False)

        # Calculate heat capacity manually since ASE doesn't provide it
        thermo_props = ThermodynamicProperties(
            real_frequencies, self.atoms, temperature
        )
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
        filename: Optional[str] = None,
        amplitude: float = 1.0,
        nframes: int = 20,
    ) -> None:
        """
        Write trajectory showing normal mode motion.

        Parameters:
        -----------
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
            raise ValueError(f"Mode index {mode_index} out of range")

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
        print(f"Normal mode trajectory written to {filename}")


class HessianCalculator:
    """
    Numerical Hessian calculation using finite differences.
    """

    def __init__(
        self,
        atoms: Atoms,
        calculator,
        delta: float = 0.01,
        method: str = "central",
        indices: Optional[List[int]] = None,
    ):
        """
        Initialize Hessian calculator.

        Parameters:
        -----------
        atoms : Atoms
            ASE Atoms object
        calculator : Calculator
            QME calculator
        delta : float
            Displacement for finite differences (Å)
        method : str
            'forward' or 'central' differences
        indices : List[int], optional
            Indices of atoms to include. If None, all atoms included.
        """
        self.atoms = atoms.copy()
        self.calculator = calculator
        self.atoms.calc = calculator
        self.delta = delta
        self.method = method
        self.indices = indices if indices is not None else list(range(len(atoms)))

    def calculate_numerical_hessian(self) -> np.ndarray:
        """
        Calculate Hessian matrix using finite differences.

        Returns:
        --------
        np.ndarray
            Hessian matrix (3N x 3N for N atoms in indices)
        """
        n_atoms = len(self.indices)
        n_coords = 3 * n_atoms
        hessian = np.zeros((n_coords, n_coords))

        print(
            f"Calculating Hessian for {n_atoms} atoms using {self.method} differences..."
        )

        if self.method == "central":
            # Central differences: H_ij = (F_i(+δj) - F_i(-δj)) / (2δ)
            for j in range(n_coords):
                atom_j = self.indices[j // 3]
                coord_j = j % 3

                # Positive displacement
                forces_plus = self._get_forces_displaced(atom_j, coord_j, self.delta)

                # Negative displacement
                forces_minus = self._get_forces_displaced(atom_j, coord_j, -self.delta)

                # Central difference
                hessian[:, j] = -(forces_plus - forces_minus) / (2 * self.delta)

                if world.rank == 0:
                    print(f"  Completed coordinate {j+1}/{n_coords}")

        elif self.method == "forward":
            # Forward differences: H_ij = (F_i(+δj) - F_i(0)) / δ
            forces_ref = self._get_reference_forces()

            for j in range(n_coords):
                atom_j = self.indices[j // 3]
                coord_j = j % 3

                forces_displaced = self._get_forces_displaced(
                    atom_j, coord_j, self.delta
                )
                hessian[:, j] = -(forces_displaced - forces_ref) / self.delta

                if world.rank == 0:
                    print(f"  Completed coordinate {j+1}/{n_coords}")
        else:
            raise ValueError(f"Unknown finite difference method: {self.method}")

        # Symmetrize Hessian
        hessian = 0.5 * (hessian + hessian.T)

        print("Hessian calculation completed.")
        return hessian

    def _get_reference_forces(self) -> np.ndarray:
        """Get forces at reference geometry."""
        # Ensure calculator is properly set
        if not hasattr(self.atoms, "calc") or self.atoms.calc is None:
            self.atoms.calc = self.calculator
        forces = self.atoms.get_forces()
        return forces[self.indices].flatten()

    def _get_forces_displaced(
        self, atom_index: int, direction: int, displacement: float
    ) -> np.ndarray:
        """
        Get forces for displaced geometry.

        Parameters:
        -----------
        atom_index : int
            Index of atom to displace
        direction : int
            Coordinate direction (0=x, 1=y, 2=z)
        displacement : float
            Displacement in Å

        Returns:
        --------
        np.ndarray
            Forces on atoms in indices, flattened
        """
        atoms_displaced = self.atoms.copy()
        atoms_displaced.positions[atom_index, direction] += displacement

        # Create a fresh calculator instance to avoid caching issues
        if hasattr(self.calculator, "backend"):
            # For MockCalculator, create a new instance with same parameters
            from .potentials import MockCalculator

            calc = MockCalculator(
                backend=self.calculator.backend,
                force_constant=self.calculator.force_constant,
                charge=self.calculator.charge,
                mult=self.calculator.mult,
            )
        else:
            # For other calculators, try to create a copy
            calc = self.calculator

        atoms_displaced.calc = calc
        forces = atoms_displaced.get_forces()
        return forces[self.indices].flatten()


class ThermodynamicProperties:
    """Calculate thermodynamic properties from vibrational frequencies."""

    def __init__(
        self,
        frequencies: np.ndarray,
        atoms: Atoms,
        temperature: float = 298.15,
        pressure: float = 101325,
    ):
        """
        Initialize thermodynamic property calculator.

        Parameters:
        -----------
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
        """Calculate vibrational partition function."""
        freq_eV = self.frequencies * units._hplanck * units._c * 100 / units._e
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
        """Calculate vibrational heat capacity."""
        freq_eV = self.frequencies * units._hplanck * units._c * 100 / units._e
        kT = units.kB * self.temperature

        cv_vib = 0.0
        for freq in freq_eV:
            x = freq / kT
            if x < 50:  # Avoid overflow
                exp_x = np.exp(x)
                cv_vib += units.kB * x**2 * exp_x / (exp_x - 1) ** 2

        return cv_vib

    def entropy_vibrational(self) -> float:
        """Calculate vibrational entropy."""
        freq_eV = self.frequencies * units._hplanck * units._c * 100 / units._e
        kT = units.kB * self.temperature

        s_vib = 0.0
        for freq in freq_eV:
            x = freq / kT
            if x < 50:  # Avoid overflow
                exp_x = np.exp(x)
                s_vib += units.kB * (x / (exp_x - 1) - np.log(1 - np.exp(-x)))

        return s_vib
