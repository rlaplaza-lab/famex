"""Hessian calculation using finite differences.

This module provides the HessianCalculator class for numerical Hessian
calculation using finite difference methods.

The finite difference schemes are implemented in qme.analysis.finite_differences.
Richardson extrapolation can be combined with 3-point or 5-point schemes
for additional accuracy improvements.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from ase import Atoms

from qme.analysis.finite_differences import (
    CentralDifferenceScheme,
    FiniteDifferenceScheme,
    FivePointCentralDifferenceScheme,
    ForwardDifferenceScheme,
)
from qme.utils.logging import get_qme_logger

logger = get_qme_logger(__name__)


class CalculatorProtocol(Protocol):
    """Protocol for calculator objects compatible with ASE Atoms.

    Any object that can be assigned to ``atoms.calc`` and provides
    ``get_forces()`` method is compatible. This includes:
    - QME calculators (BasePotential subclasses)
    - ASE Calculator instances
    - Any object with compatible interface
    """

    def get_forces(self, atoms: Atoms | None = None) -> np.ndarray:
        """Compute forces for the given atoms structure.

        Parameters
        ----------
        atoms : Atoms, optional
            Atoms structure. If None, uses previously set atoms.

        Returns:
        -------
        np.ndarray
            Forces array of shape (N, 3) where N is number of atoms.

        """
        ...


class HessianCalculator:
    """Numerical Hessian calculation using finite differences.

    This class computes the Hessian matrix (second derivative of energy with
    respect to atomic coordinates) using finite difference methods. The Hessian
    is essential for vibrational frequency analysis, transition state verification,
    and thermodynamic property calculations.

    The Hessian matrix H has elements H_ij = ∂²E/(∂x_i ∂x_j), where E is the
    potential energy and x_i, x_j are atomic coordinates (flattened: atom 0 x,y,z,
    atom 1 x,y,z, ...).

    Available finite difference methods:
    - 'forward' (2-point): Fast, 1st order accuracy, requires N+1 calculations
    - 'central' (3-point): Standard choice, 2nd order accuracy, requires 2N+1 calculations
    - '5point': High accuracy, 4th order accuracy, requires 4N+1 calculations

    Richardson extrapolation can improve accuracy by combining results from two
    different step sizes, effectively canceling leading error terms.

    Examples:
    --------
    >>> from ase import Atoms
    >>> from qme.potentials.mock_potential import MockCalculator
    >>> # Create a simple system
    >>> atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    >>> calc = MockCalculator()
    >>> # Standard central difference Hessian
    >>> hessian_calc = HessianCalculator(atoms, calc, delta=0.01)
    >>> hessian = hessian_calc.calculate_numerical_hessian()
    >>> # High-accuracy with Richardson extrapolation
    >>> hessian_calc_rich = HessianCalculator(
    ...     atoms, calc, delta=0.02, richardson=True, delta2=0.01, method='central'
    ... )
    >>> hessian = hessian_calc_rich.calculate_numerical_hessian()

    Notes:
    -----
    - The Hessian is automatically symmetrized: H = (H + H^T) / 2
    - For N atoms, the Hessian has shape (3N, 3N)
    - When using `indices`, only specified atoms contribute to the Hessian
    - Large `delta` values (>10% of typical bond length) may cause inaccurate results
    - Very small `delta` values (<1e-5 Å) may suffer from numerical precision issues

    See Also:
    --------
    FrequencyAnalysis : Higher-level interface for frequency calculations
    validate_hessian : Function to validate Hessian properties

    """

    def __init__(
        self,
        atoms: Atoms,
        calculator: CalculatorProtocol,
        delta: float = 0.01,
        method: str = "central",
        richardson: bool = False,
        delta2: float | None = None,
        indices: list[int] | None = None,
        verbose: int = 1,
    ) -> None:
        """Initialize Hessian calculator.

        Parameters
        ----------
        atoms : Atoms
            ASE Atoms object
        calculator : CalculatorProtocol
            Calculator compatible with ASE (QME calculator, ASE Calculator, etc.)
        delta : float
            Displacement for finite differences (Å). Must be positive.
        method : str
            'forward', 'central', or '5point' differences
        richardson : bool, default False
            Whether to use Richardson extrapolation for improved accuracy
        delta2 : float, optional
            Second displacement for Richardson extrapolation (Å).
            If None and richardson=True, defaults to delta/2.
            Must be positive and differ from delta.
        indices : List[int], optional
            Indices of atoms to include. If None, all atoms included.
            Indices must be unique, within bounds, and valid atom indices.
        verbose : int
            Verbosity level for Hessian calculation output:
            - 0: Quiet (minimal output)
            - 1: Normal (default, shows progress)
            - 2: Verbose (detailed information)

        Raises:
        ------
        ValueError
            If delta <= 0, indices are invalid, or other parameters are inconsistent

        Examples:
        --------
        >>> from ase import Atoms
        >>> from qme.analysis.hessian import HessianCalculator
        >>> # Basic usage with central differences
        >>> atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
        >>> calc = SomeCalculator()  # Your calculator here
        >>> hessian_calc = HessianCalculator(atoms, calc, delta=0.01)
        >>> hessian = hessian_calc.calculate_numerical_hessian()
        >>> # High-accuracy with Richardson extrapolation
        >>> hessian_calc_rich = HessianCalculator(
        ...     atoms, calc, delta=0.02, richardson=True, delta2=0.01
        ... )
        >>> hessian = hessian_calc_rich.calculate_numerical_hessian()

        """
        # Validate atoms
        if len(atoms) == 0:
            msg = "Atoms object must contain at least one atom"
            raise ValueError(msg)

        # Validate delta
        if delta <= 0:
            msg = f"delta must be positive, got {delta}"
            raise ValueError(msg)

        if indices is not None:
            if not isinstance(indices, list) or len(indices) == 0:
                msg = "indices must be a non-empty list"
                raise ValueError(msg)
            if len(set(indices)) != len(indices):
                msg = "indices must be unique"
                raise ValueError(msg)
            if not all(0 <= idx < len(atoms) for idx in indices):
                invalid = [idx for idx in indices if not (0 <= idx < len(atoms))]
                msg = f"indices out of bounds: {invalid} (system has {len(atoms)} atoms)"
                raise ValueError(msg)
            self.indices = indices
        else:
            self.indices = list(range(len(atoms)))

        self.atoms = atoms
        self.calculator = calculator
        if self.atoms.calc is None or self.atoms.calc is not calculator:
            self.atoms.calc = calculator
        try:
            test_forces = self.atoms.get_forces()
            if test_forces is None:
                msg = "Calculator does not provide forces (get_forces returned None)"
                raise ValueError(msg)
            if not isinstance(test_forces, np.ndarray):
                msg = f"Calculator forces must be numpy array, got {type(test_forces)}"
                raise ValueError(msg)
            if np.any(np.isnan(test_forces)) or np.any(np.isinf(test_forces)):
                logger.warning(
                    "Calculator returned NaN or Inf forces at reference geometry. "
                    "Hessian calculation may fail."
                )
        except Exception as e:
            msg = f"Calculator validation failed: {e}. Ensure calculator is properly initialized."
            raise ValueError(msg) from e

        self.delta = delta
        self.richardson = richardson
        self.delta2 = delta2
        self.verbose = verbose

        if len(self.indices) > 0:
            positions = atoms.positions[self.indices]
            max_distance = np.max(np.linalg.norm(positions, axis=1))
            min_distance = np.min(
                [
                    np.linalg.norm(pos - positions[j])
                    for i, pos in enumerate(positions)
                    for j in range(i + 1, len(positions))
                ]
                if len(positions) > 1
                else [max_distance],
            )
            if delta > 0.1 * min_distance:
                logger.warning(
                    f"delta ({delta:.4f} Å) is large compared to minimum interatomic distance "
                    f"({min_distance:.4f} Å). Consider using delta < {0.1 * min_distance:.4f} Å"
                )
            if delta < 1e-5:
                logger.warning(
                    f"delta ({delta:.4f} Å) is very small. Numerical precision may be limited."
                )

        if method == "central":
            self.scheme: FiniteDifferenceScheme = CentralDifferenceScheme()
        elif method == "forward":
            self.scheme = ForwardDifferenceScheme()
        elif method == "5point":
            self.scheme = FivePointCentralDifferenceScheme()
        else:
            msg = f"Unknown finite difference method: {method}. Use 'forward', 'central', or '5point'."
            raise ValueError(msg)

        if self.richardson:
            if isinstance(self.scheme, FivePointCentralDifferenceScheme):
                self._richardson_order = 4
            elif isinstance(self.scheme, CentralDifferenceScheme):
                self._richardson_order = 2
            else:
                msg = "Richardson extrapolation currently supported only for 'central' or '5point' methods."
                raise ValueError(msg)
            if self.delta2 is None:
                self.delta2 = self.delta / 2.0
            if self.delta2 <= 0:
                msg = f"delta2 must be positive when using Richardson extrapolation, got {self.delta2}"
                raise ValueError(msg)
            if np.isclose(self.delta2, self.delta, rtol=1e-10):
                msg = (
                    f"delta2 ({self.delta2}) must differ from delta ({self.delta}) "
                    "when using Richardson extrapolation."
                )
                raise ValueError(msg)
            if self.delta2 >= self.delta:
                logger.warning(
                    "For Richardson extrapolation, typically delta2 < delta. "
                    f"Got delta={self.delta}, delta2={self.delta2}"
                )
        else:
            self._richardson_order = None
            if delta2 is not None:
                logger.warning("delta2 provided but richardson=False. delta2 will be ignored.")

    def calculate_numerical_hessian(self) -> np.ndarray:
        """Calculate Hessian matrix using finite differences.

        This method computes the complete Hessian matrix by displacing each
        coordinate and computing the derivative of forces with respect to that
        displacement. The computation is done column by column for efficiency.

        Returns:
        -------
        np.ndarray
            Hessian matrix of shape (3N, 3N) where N is the number of atoms
            in `indices`. Units are eV/Å². The matrix is symmetric and includes
            all selected atoms.

        Raises:
        ------
        RuntimeError
            If force calculations fail at any displacement step

        Notes:
        -----
        - For central differences with N atoms: performs 2*3*N force calculations
        - For 5-point scheme: performs 4*3*N force calculations
        - For Richardson extrapolation: doubles the number of calculations
        - The final Hessian is symmetrized to ensure H = H^T (required by theory)
        - Progress information is logged if verbose >= 2

        Examples:
        --------
        >>> from ase import Atoms
        >>> from qme.potentials.mock_potential import MockCalculator
        >>> atoms = Atoms('H2O', positions=[[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0]])
        >>> calc = MockCalculator()
        >>> hessian_calc = HessianCalculator(atoms, calc, delta=0.01, verbose=1)
        >>> hessian = hessian_calc.calculate_numerical_hessian()
        >>> print(f"Hessian shape: {hessian.shape}")  # (9, 9) for 3 atoms
        >>> # Check symmetry (should be very close to symmetric)
        >>> symmetry_error = np.max(np.abs(hessian - hessian.T))
        >>> print(f"Symmetry error: {symmetry_error:.2e}")  # Should be ~0

        """
        n_atoms = len(self.indices)
        n_coords = 3 * n_atoms
        hessian = np.zeros((n_coords, n_coords))

        if self.verbose >= 2:
            logger.info(
                f"Calculating Hessian for {n_atoms} atoms using {type(self.scheme).__name__}"
            )

        forces_ref = None
        if isinstance(self.scheme, ForwardDifferenceScheme):
            forces_ref = self._get_reference_forces()

        for j in range(n_coords):
            atom_j = self.indices[j // 3]
            coord_j = j % 3

            try:
                if self.richardson:
                    hessian[:, j] = self._compute_richardson_extrapolated_derivative(
                        atom_j, coord_j
                    )
                else:
                    hessian[:, j] = self._compute_derivative_at_delta(
                        atom_j, coord_j, self.delta, forces_ref
                    )
            except Exception as e:
                coord_name = ["x", "y", "z"][coord_j]
                msg = (
                    f"Failed to compute Hessian column for atom {atom_j}, "
                    f"coordinate {coord_name} (index {j}/{n_coords - 1}): {e}"
                )
                raise RuntimeError(msg) from e

            if self.verbose >= 2:
                logger.debug(f"Completed coordinate {j + 1}/{n_coords}")

        # Symmetrize Hessian: H_sym = (H + H^T) / 2
        # This ensures exact symmetry (required by theory) and removes small
        # numerical asymmetries that can arise from finite precision arithmetic
        HESSIAN_SYMMETRIZATION_FACTOR = 0.5
        hessian = HESSIAN_SYMMETRIZATION_FACTOR * (hessian + hessian.T)

        if self.verbose >= 2:
            logger.info("Hessian calculation completed")
        return hessian

    def _compute_derivative_at_delta(
        self,
        atom_index: int,
        direction: int,
        delta: float,
        forces_ref: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute derivative at a given delta using the configured scheme.

        This is a helper method that handles the common logic for computing
        a Hessian column using the configured finite difference scheme.

        Parameters
        ----------
        atom_index : int
            Index of atom to displace
        direction : int
            Coordinate direction (0=x, 1=y, 2=z)
        delta : float
            Displacement step size
        forces_ref : np.ndarray, optional
            Forces at reference geometry (for forward differences)

        Returns:
        -------
        np.ndarray
            Hessian column

        """
        if isinstance(self.scheme, FivePointCentralDifferenceScheme):
            return self._compute_five_point_derivative(atom_index, direction, delta)
        else:
            forces_plus = self._get_forces_displaced(atom_index, direction, delta)
            forces_minus = None
            if isinstance(self.scheme, CentralDifferenceScheme):
                forces_minus = self._get_forces_displaced(atom_index, direction, -delta)
            return self.scheme.compute_derivative(forces_plus, forces_minus, forces_ref, delta)

    def _compute_richardson_extrapolated_derivative(
        self,
        atom_index: int,
        direction: int,
    ) -> np.ndarray:
        """Compute Richardson extrapolated derivative for improved accuracy.

        Richardson extrapolation combines derivatives computed at two different
        step sizes to cancel leading error terms, resulting in higher accuracy.

        Formula: D_extrapolated ≈ (2^p * D(delta2) - D(delta)) / (2^p - 1)
        where p is the order of the underlying finite difference scheme.

        Parameters
        ----------
        atom_index : int
            Index of atom to displace
        direction : int
            Coordinate direction (0=x, 1=y, 2=z)

        Returns:
        -------
        np.ndarray
            Richardson extrapolated Hessian column

        """
        if self._richardson_order is None:
            msg = "Richardson order not set. Cannot perform extrapolation."
            raise RuntimeError(msg)

        richardson_order = self._richardson_order
        derivative_at_delta = self._compute_derivative_at_delta(
            atom_index, direction, self.delta, forces_ref=None
        )
        d2 = float(self.delta2)  # type: ignore[arg-type]
        derivative_at_delta2 = self._compute_derivative_at_delta(
            atom_index, direction, d2, forces_ref=None
        )
        RICHARDSON_EXTRAPOLATION_BASE = 2.0
        extrapolation_factor = RICHARDSON_EXTRAPOLATION_BASE**richardson_order
        extrapolated_derivative = (
            extrapolation_factor * derivative_at_delta2 - derivative_at_delta
        ) / (extrapolation_factor - 1.0)

        return extrapolated_derivative

    def _get_reference_forces(self) -> np.ndarray:
        """Get forces at reference geometry.

        Returns:
        -------
        np.ndarray
            Forces on atoms in indices, flattened

        """
        if not hasattr(self.atoms, "calc") or self.atoms.calc is None:
            self.atoms.calc = self.calculator
        forces = self.atoms.get_forces()
        return forces[self.indices].flatten()

    def _get_forces_displaced(
        self,
        atom_index: int,
        direction: int,
        displacement: float,
    ) -> np.ndarray:
        """Get forces for displaced geometry.

        Parameters
        ----------
        atom_index : int
            Index of atom to displace
        direction : int
            Coordinate direction (0=x, 1=y, 2=z)
        displacement : float
            Displacement in Å

        Returns:
        -------
        np.ndarray
            Forces on atoms in indices, flattened

        Raises:
        ------
        RuntimeError
            If force calculation fails or returns invalid results

        """
        coord_names = ["x", "y", "z"]
        coord_name = coord_names[direction] if 0 <= direction < 3 else f"unknown({direction})"

        atoms_displaced = self.atoms.copy()
        atoms_displaced.positions[atom_index, direction] += displacement

        # MockCalculator requires fresh instances to avoid state contamination.
        # Real calculators (UMA, AIMNet2, etc.) are reused to avoid model reloading.
        try:
            from qme.potentials.mock_potential import MockCalculator
        except ImportError:
            MockCalculator = None

        if MockCalculator is not None and isinstance(self.calculator, MockCalculator):
            calc = MockCalculator(
                backend=self.calculator.backend,
                force_constant=getattr(self.calculator, "force_constant", 1.0),
                charge=getattr(self.calculator, "charge", 0),
                mult=getattr(self.calculator, "mult", 1),
            )
        else:
            calc = self.calculator

        atoms_displaced.calc = calc

        try:
            forces = atoms_displaced.get_forces()
        except Exception as e:
            msg = (
                f"Force calculation failed for atom {atom_index}, "
                f"{coord_name}-displacement {displacement:+.4f} Å: {e}"
            )
            raise RuntimeError(msg) from e

        if forces is None:
            msg = (
                f"Calculator returned None forces for atom {atom_index}, "
                f"{coord_name}-displacement {displacement:+.4f} Å"
            )
            raise RuntimeError(msg)
        if not isinstance(forces, np.ndarray):
            msg = (
                f"Calculator forces must be numpy array, got {type(forces)} "
                f"for atom {atom_index}, {coord_name}-displacement {displacement:+.4f} Å"
            )
            raise RuntimeError(msg)
        if np.any(np.isnan(forces)) or np.any(np.isinf(forces)):
            nan_count = np.sum(np.isnan(forces))
            inf_count = np.sum(np.isinf(forces))
            msg = (
                f"Calculator returned invalid forces for atom {atom_index}, "
                f"{coord_name}-displacement {displacement:+.4f} Å: "
                f"{nan_count} NaN, {inf_count} Inf values"
            )
            raise RuntimeError(msg)

        return forces[self.indices].flatten()

    def _compute_five_point_derivative(
        self,
        atom_index: int,
        direction: int,
        delta: float,
    ) -> np.ndarray:
        """Compute 5-point finite difference derivative.

        The 5-point scheme requires forces at ±delta and ±2delta displacements,
        but not at the reference geometry. This method computes all required
        forces and uses the scheme's compute_derivative method.

        Parameters
        ----------
        atom_index : int
            Index of atom to displace
        direction : int
            Coordinate direction (0=x, 1=y, 2=z)
        delta : float
            Displacement step size

        Returns:
        -------
        np.ndarray
            Hessian column using 5-point stencil

        """
        forces_minus2 = self._get_forces_displaced(atom_index, direction, -2 * delta)
        forces_minus = self._get_forces_displaced(atom_index, direction, -delta)
        forces_plus = self._get_forces_displaced(atom_index, direction, delta)
        forces_plus2 = self._get_forces_displaced(atom_index, direction, 2 * delta)

        return self.scheme.compute_derivative(
            forces_plus,
            forces_minus,
            None,  # forces_ref not used for 5-point
            delta,
            forces_plus2=forces_plus2,
            forces_minus2=forces_minus2,
        )


__all__ = [
    "CalculatorProtocol",
    "HessianCalculator",
    # Re-export FD schemes for backward compatibility
    "FiniteDifferenceScheme",
    "CentralDifferenceScheme",
    "ForwardDifferenceScheme",
    "FivePointCentralDifferenceScheme",
]
