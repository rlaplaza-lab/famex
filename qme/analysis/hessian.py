"""Hessian calculation using finite differences.

This module provides the HessianCalculator class for numerical Hessian
calculation and an extensible interface for finite difference schemes.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from ase import Atoms

from qme.utils.logging import get_qme_logger

logger = get_qme_logger(__name__)


class FiniteDifferenceScheme(Protocol):
    """Protocol for finite difference schemes.

    This protocol allows for extensible finite difference implementations.
    Future schemes could include 4-point, 5-point, or Richardson extrapolation.
    """

    def compute_derivative(
        self,
        forces_plus: np.ndarray,
        forces_minus: np.ndarray | None,
        forces_ref: np.ndarray | None,
        delta: float,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute Hessian column using finite differences.

        Parameters
        ----------
        forces_plus : np.ndarray
            Forces at +delta displacement
        forces_minus : np.ndarray, optional
            Forces at -delta displacement (for central differences)
        forces_ref : np.ndarray, optional
            Forces at reference geometry (for forward differences)
        delta : float
            Displacement step size
        **kwargs : Any
            Additional force arrays for higher-order schemes

        Returns:
        -------
        np.ndarray
            Hessian column: -∂F/∂x

        """
        ...


class CentralDifferenceScheme:
    """Central difference finite difference scheme (2nd order accuracy)."""

    def compute_derivative(
        self,
        forces_plus: np.ndarray,
        forces_minus: np.ndarray | None,
        forces_ref: np.ndarray | None,
        delta: float,
    ) -> np.ndarray:
        """Compute Hessian column using central differences.

        H_ij = (F_i(+δj) - F_i(-δj)) / (2δ)

        """
        if forces_minus is None:
            msg = "Central difference requires forces_minus"
            raise ValueError(msg)
        return -(forces_plus - forces_minus) / (2 * delta)


class ForwardDifferenceScheme:
    """Forward difference finite difference scheme (1st order accuracy)."""

    def compute_derivative(
        self,
        forces_plus: np.ndarray,
        forces_minus: np.ndarray | None,
        forces_ref: np.ndarray | None,
        delta: float,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute Hessian column using forward differences.

        H_ij = (F_i(+δj) - F_i(0)) / δ

        """
        if forces_ref is None:
            msg = "Forward difference requires forces_ref"
            raise ValueError(msg)
        return -(forces_plus - forces_ref) / delta


class FivePointCentralDifferenceScheme:
    """5-point central difference finite difference scheme (4th order accuracy).

    Uses the stencil:
    f''(x) ≈ [-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)] / (12h²)

    For Hessian elements:
    H_ij ≈ [-F_i(x+2δ·e_j) + 16F_i(x+δ·e_j) - 30F_i(x) + 16F_i(x-δ·e_j) - F_i(x-2δ·e_j)] / (12δ²)
    """

    def compute_derivative(
        self,
        forces_plus: np.ndarray,
        forces_minus: np.ndarray | None,
        forces_ref: np.ndarray | None,
        delta: float,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute Hessian column using 5-point central differences.

        Parameters
        ----------
        forces_plus : np.ndarray
            Forces at +delta displacement
        forces_minus : np.ndarray
            Forces at -delta displacement
        forces_ref : np.ndarray
            Forces at reference geometry (required for 5-point scheme)
        delta : float
            Displacement step size
        **kwargs : Any
            Additional forces at +2delta and -2delta:
            - forces_plus2: Forces at +2*delta displacement
            - forces_minus2: Forces at -2*delta displacement

        Returns:
        -------
        np.ndarray
            Hessian column: -∂F/∂x

        """
        if forces_minus is None:
            msg = "5-point central difference requires forces_minus"
            raise ValueError(msg)
        if forces_ref is None:
            msg = "5-point central difference requires forces_ref"
            raise ValueError(msg)
        if "forces_plus2" not in kwargs or "forces_minus2" not in kwargs:
            msg = "5-point central difference requires forces_plus2 and forces_minus2"
            raise ValueError(msg)

        forces_plus2 = kwargs["forces_plus2"]
        forces_minus2 = kwargs["forces_minus2"]

        # 5-point stencil coefficients: [-1, 16, -30, 16, -1] / (12h²)
        # For second derivative of energy: H = ∂²E/∂x² = -∂F/∂x
        hessian_col = (
            -forces_minus2 + 16 * forces_minus - 30 * forces_ref + 16 * forces_plus - forces_plus2
        ) / (12 * delta**2)

        # Note: No negative sign needed here because the stencil already computes ∂²E/∂x²
        # and we want H = -∂F/∂x = -∂(-∇E)/∂x = ∂²E/∂x²
        return -hessian_col


class HessianCalculator:
    """Numerical Hessian calculation using finite differences."""

    def __init__(
        self,
        atoms: Atoms,
        calculator: Any,
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
        calculator : Calculator
            QME calculator
        delta : float
            Displacement for finite differences (Å)
        method : str
            'forward', 'central', or '5point' differences
        indices : List[int], optional
            Indices of atoms to include. If None, all atoms included.
        verbose : int
            Verbosity level for Hessian calculation output:
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
        self.verbose = verbose
        self.indices = indices if indices is not None else list(range(len(atoms)))

        # Set up finite difference scheme
        if method == "central":
            self.scheme: FiniteDifferenceScheme = CentralDifferenceScheme()
        elif method == "forward":
            self.scheme = ForwardDifferenceScheme()
        elif method == "5point":
            self.scheme = FivePointCentralDifferenceScheme()
        else:
            msg = f"Unknown finite difference method: {method}. Use 'forward', 'central', or '5point'."
            raise ValueError(msg)

        # Validate Richardson settings
        if self.richardson:
            # Richardson extrapolation is defined for both central (p=2) and 5-point (p=4) schemes
            if isinstance(self.scheme, FivePointCentralDifferenceScheme):
                self._richardson_order = 4  # 5-point scheme is 4th order
            elif isinstance(self.scheme, CentralDifferenceScheme):
                self._richardson_order = 2  # 3-point scheme is 2nd order
            else:
                msg = "Richardson extrapolation currently supported only for 'central' or '5point' methods."
                raise ValueError(msg)
            # Default delta2 = delta/2 if not provided
            if self.delta2 is None:
                self.delta2 = self.delta / 2.0
            if self.delta2 <= 0:
                msg = "delta2 must be positive when using Richardson extrapolation."
                raise ValueError(msg)
            if np.isclose(self.delta2, self.delta):
                msg = "delta2 must differ from delta when using Richardson extrapolation."
                raise ValueError(msg)
        else:
            self._richardson_order = None

    def calculate_numerical_hessian(self) -> np.ndarray:
        """Calculate Hessian matrix using finite differences.

        Returns:
        -------
        np.ndarray
            Hessian matrix (3N x 3N for N atoms in indices)

        """
        n_atoms = len(self.indices)
        n_coords = 3 * n_atoms
        hessian = np.zeros((n_coords, n_coords))

        if self.verbose >= 2:
            logger.info(
                f"Calculating Hessian for {n_atoms} atoms using {type(self.scheme).__name__}"
            )

        # Get reference forces for schemes that need them (forward and 5-point)
        forces_ref = None
        if isinstance(self.scheme, ForwardDifferenceScheme | FivePointCentralDifferenceScheme):
            forces_ref = self._get_reference_forces()

        # Compute Hessian column by column
        for j in range(n_coords):
            atom_j = self.indices[j // 3]
            coord_j = j % 3

            if self.richardson:
                # Richardson extrapolation works for both 3-point and 5-point schemes
                p = self._richardson_order  # Order of the underlying scheme

                # Compute derivative at delta
                if isinstance(self.scheme, FivePointCentralDifferenceScheme):
                    D1 = self._compute_five_point_derivative(
                        atom_j, coord_j, self.delta, forces_ref
                    )
                else:
                    forces_plus_d1 = self._get_forces_displaced(atom_j, coord_j, self.delta)
                    forces_minus_d1 = self._get_forces_displaced(atom_j, coord_j, -self.delta)
                    D1 = self.scheme.compute_derivative(
                        forces_plus_d1, forces_minus_d1, None, self.delta
                    )

                # Compute derivative at delta2
                d2 = float(self.delta2)  # type: ignore[arg-type]
                if isinstance(self.scheme, FivePointCentralDifferenceScheme):
                    D2 = self._compute_five_point_derivative(atom_j, coord_j, d2, forces_ref)
                else:
                    forces_plus_d2 = self._get_forces_displaced(atom_j, coord_j, d2)
                    forces_minus_d2 = self._get_forces_displaced(atom_j, coord_j, -d2)
                    D2 = self.scheme.compute_derivative(forces_plus_d2, forces_minus_d2, None, d2)

                # Richardson extrapolation formula: D ≈ (2^p·D(h/2) - D(h)) / (2^p - 1)
                # For p=2 (3-point): (4*D2 - D1) / 3
                # For p=4 (5-point): (16*D2 - D1) / 15
                factor = 2.0**p
                hessian[:, j] = (factor * D2 - D1) / (factor - 1.0)
            else:
                # No Richardson extrapolation
                if isinstance(self.scheme, FivePointCentralDifferenceScheme):
                    hessian[:, j] = self._compute_five_point_derivative(
                        atom_j, coord_j, self.delta, forces_ref
                    )
                else:
                    # Positive displacement
                    forces_plus = self._get_forces_displaced(atom_j, coord_j, self.delta)

                    # Negative displacement (for central differences)
                    forces_minus = None
                    if isinstance(self.scheme, CentralDifferenceScheme):
                        forces_minus = self._get_forces_displaced(atom_j, coord_j, -self.delta)

                    # Compute derivative using the scheme
                    hessian[:, j] = self.scheme.compute_derivative(
                        forces_plus, forces_minus, forces_ref, self.delta
                    )

            if self.verbose >= 2:
                logger.debug(f"Completed coordinate {j + 1}/{n_coords}")

        # Symmetrize Hessian
        hessian = 0.5 * (hessian + hessian.T)

        if self.verbose >= 2:
            logger.info("Hessian calculation completed")
        return hessian

    def _get_reference_forces(self) -> np.ndarray:
        """Get forces at reference geometry.

        Returns:
        -------
        np.ndarray
            Forces on atoms in indices, flattened

        """
        # Ensure calculator is properly set
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

        """
        atoms_displaced = self.atoms.copy()
        atoms_displaced.positions[atom_index, direction] += displacement

        # Reuse the same calculator instance for efficiency
        # Only create a new MockCalculator if absolutely necessary (for state isolation)
        try:
            from qme.potentials.mock_potential import MockCalculator
        except ImportError:
            MockCalculator = None

        if MockCalculator is not None and isinstance(self.calculator, MockCalculator):
            # For MockCalculator, we need a fresh instance to avoid state contamination
            # but we can reuse the same parameters
            calc = MockCalculator(
                backend=self.calculator.backend,
                force_constant=getattr(self.calculator, "force_constant", 1.0),
                charge=getattr(self.calculator, "charge", 0),
                mult=getattr(self.calculator, "mult", 1),
            )
        else:
            # For real calculators (UMA, AIMNet2, etc.), reuse the same instance
            # This is much more efficient and avoids repeated model loading
            calc = self.calculator

        atoms_displaced.calc = calc
        forces = atoms_displaced.get_forces()
        return forces[self.indices].flatten()

    def _compute_five_point_derivative(
        self,
        atom_index: int,
        direction: int,
        delta: float,
        forces_ref: np.ndarray,
    ) -> np.ndarray:
        """Compute 5-point finite difference derivative.

        Parameters
        ----------
        atom_index : int
            Index of atom to displace
        direction : int
            Coordinate direction (0=x, 1=y, 2=z)
        delta : float
            Displacement step size
        forces_ref : np.ndarray
            Forces at reference geometry

        Returns:
        -------
        np.ndarray
            Hessian column using 5-point stencil

        """
        # Compute forces at all 5 points: -2δ, -δ, 0, +δ, +2δ
        forces_minus2 = self._get_forces_displaced(atom_index, direction, -2 * delta)
        forces_minus = self._get_forces_displaced(atom_index, direction, -delta)
        forces_plus = self._get_forces_displaced(atom_index, direction, delta)
        forces_plus2 = self._get_forces_displaced(atom_index, direction, 2 * delta)

        # Use the scheme's compute_derivative method
        return self.scheme.compute_derivative(
            forces_plus,
            forces_minus,
            forces_ref,
            delta,
            forces_plus2=forces_plus2,
            forces_minus2=forces_minus2,
        )


__all__ = [
    "HessianCalculator",
    "FiniteDifferenceScheme",
    "CentralDifferenceScheme",
    "ForwardDifferenceScheme",
    "FivePointCentralDifferenceScheme",
]
