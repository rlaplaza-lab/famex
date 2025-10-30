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

__all__ = [
    "HessianCalculator",
    "FiniteDifferenceScheme",
    "CentralDifferenceScheme",
    "ForwardDifferenceScheme",
]


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
    ) -> np.ndarray:
        """Compute Hessian column using forward differences.

        H_ij = (F_i(+δj) - F_i(0)) / δ

        """
        if forces_ref is None:
            msg = "Forward difference requires forces_ref"
            raise ValueError(msg)
        return -(forces_plus - forces_ref) / delta


class HessianCalculator:
    """Numerical Hessian calculation using finite differences."""

    def __init__(
        self,
        atoms: Atoms,
        calculator: Any,
        delta: float = 0.01,
        method: str = "central",
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
            'forward' or 'central' differences
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
        self.verbose = verbose
        self.indices = indices if indices is not None else list(range(len(atoms)))

        # Set up finite difference scheme
        if method == "central":
            self.scheme: FiniteDifferenceScheme = CentralDifferenceScheme()
        elif method == "forward":
            self.scheme = ForwardDifferenceScheme()
        else:
            msg = f"Unknown finite difference method: {method}. Use 'central' or 'forward'."
            raise ValueError(msg)

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

        # Get reference forces for forward differences
        forces_ref = None
        if isinstance(self.scheme, ForwardDifferenceScheme):
            forces_ref = self._get_reference_forces()

        # Compute Hessian column by column
        for j in range(n_coords):
            atom_j = self.indices[j // 3]
            coord_j = j % 3

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
