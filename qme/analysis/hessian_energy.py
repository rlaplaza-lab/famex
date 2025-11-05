"""Energy-based Hessian calculation using finite differences.

This module implements Hessian calculations using energy-based finite differences
instead of force-based. This can be more stable when energy evaluations are more
accurate than force evaluations, or when forces have systematic errors.

Energy-based FD computes second derivatives directly:
    H_ij = [E(r + δ_i + δ_j) - E(r + δ_i - δ_j) - E(r - δ_i + δ_j) + E(r - δ_i - δ_j)] / (4δ²)
"""

from __future__ import annotations

from typing import Protocol, cast

import numpy as np
from ase import Atoms
from numpy.typing import NDArray

from qme.analysis.utils import validate_indices
from qme.utils.logging import get_qme_logger

logger = get_qme_logger(__name__)


class CalculatorProtocol(Protocol):
    """Protocol for calculator objects compatible with ASE Atoms.

    Any object that can be assigned to ``atoms.calc`` and provides
    ``get_potential_energy()`` method is compatible.
    """

    def get_potential_energy(
        self, atoms: Atoms | None = None, force_consistent: bool = False
    ) -> float:
        """Compute potential energy for the given atoms structure."""


class EnergyBasedHessianCalculator:
    """Hessian calculation using energy-based finite differences.

    This class computes the Hessian matrix using energy evaluations instead of
    force evaluations. This can be more accurate when:

    - Energy evaluations have higher precision than force evaluations
    - Forces have systematic errors or discontinuities
    - Computational cost is acceptable (4N² energy evals vs 6N force evals)

    The Hessian matrix H has elements:
        H_ij = ∂²E/(∂x_i ∂x_j)
    where E is potential energy and x_i, x_j are atomic coordinates.

    Note:
    -----
    Energy-based FD is **much slower** than force-based FD for large systems.
    For a system with N atoms:
    - Force-based: O(6N) force evaluations
    - Energy-based: O(4N²) energy evaluations

    Use only when force-based methods fail or produce inaccurate results.

    Examples:
    --------
    >>> from ase import Atoms
    >>> from qme.potentials.mock_potential import MockCalculator
    >>> atoms = Atoms('H2O', positions=[[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0]])
    >>> calc = MockCalculator()
    >>> hessian_calc = EnergyBasedHessianCalculator(atoms, calc, delta=0.01)
    >>> hessian = hessian_calc.calculate_energy_hessian()
    """

    def __init__(
        self,
        atoms: Atoms,
        calculator: CalculatorProtocol,
        delta: float = 0.01,
        indices: list[int] | None = None,
        verbose: int = 1,
    ) -> None:
        """Initialize energy-based Hessian calculator.

        Parameters
        ----------
        atoms : Atoms
            ASE Atoms object
        calculator : CalculatorProtocol
            Calculator compatible with ASE (QME calculator, ASE Calculator, etc.)
        delta : float, default 0.01
            Displacement for finite differences (Å). Must be positive.
        indices : list[int], optional
            Indices of atoms to include. If None, all atoms included.
        verbose : int, default 1
            Verbosity level for output:
            - 0: Quiet (minimal output)
            - 1: Normal (default, shows progress)
            - 2: Verbose (detailed information)

        Raises:
        ------
        ValueError
            If delta <= 0 or indices are invalid

        Examples:
        --------
        >>> from ase import Atoms
        >>> atoms = Atoms('H2O', positions=[[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0]])
        >>> calc = SomeCalculator()  # Your calculator here
        >>> hessian_calc = EnergyBasedHessianCalculator(atoms, calc, delta=0.01)
        >>> hessian = hessian_calc.calculate_energy_hessian()
        """
        # Validate atoms
        if len(atoms) == 0:
            msg = "Atoms object must contain at least one atom"
            raise ValueError(msg)

        # Validate delta
        if delta <= 0:
            msg = f"delta must be positive, got {delta}"
            raise ValueError(msg)

        self.indices = validate_indices(atoms, indices)

        self.atoms = atoms
        self.calculator = calculator
        self.atoms.calc = calculator

        self.delta = delta
        self.verbose = verbose

    def calculate_energy_hessian(self) -> np.ndarray:
        """Calculate Hessian matrix using energy-based finite differences.

        Computes second derivatives directly from energy evaluations using
        a 4-point cross-derivative stencil.

        Returns:
        -------
        np.ndarray
            Hessian matrix of shape (3N, 3N) where N is the number of atoms
            in `indices`. Units are eV/Å². The matrix is symmetric and includes
            all selected atoms.

        Raises:
        ------
        RuntimeError
            If energy calculations fail at any displacement step

        Notes:
        -----
        - For N atoms: performs 4N² energy calculations
        - Progress information is logged if verbose >= 2
        - The final Hessian is symmetrized to ensure H = H^T

        Examples:
        --------
        >>> from ase import Atoms
        >>> atoms = Atoms('H2O', positions=[[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0]])
        >>> calc = SomeCalculator()  # Your calculator here
        >>> hessian_calc = EnergyBasedHessianCalculator(atoms, calc, delta=0.01)
        >>> hessian = hessian_calc.calculate_energy_hessian()
        >>> print(f"Hessian shape: {hessian.shape}")  # (9, 9) for 3 atoms
        """
        n_atoms = len(self.indices)
        n_coords = 3 * n_atoms
        hessian = np.zeros((n_coords, n_coords))

        # Warn about computational cost for large systems
        if n_atoms > 20:
            logger.warning(
                f"Energy-based Hessian for {n_atoms} atoms will require "
                f"~{4 * n_atoms**2:.0f} energy evaluations. This may be slow."
            )

        if self.verbose >= 2:
            logger.info(f"Calculating energy-based Hessian for {n_atoms} atoms")
            total_calculations = 4 * n_coords * (n_coords + 1) // 2
            logger.info(f"Total energy calculations: {total_calculations}")

        # Compute upper triangle (Hessian is symmetric)
        total_elements = n_coords * (n_coords + 1) // 2
        completed = 0

        for i in range(n_coords):
            for j in range(i, n_coords):
                try:
                    hessian[i, j] = self._compute_cross_derivative(i, j)
                    completed += 1

                    if self.verbose >= 2 and completed % 10 == 0:
                        logger.debug(f"Completed {completed}/{total_elements} elements")

                except Exception as e:
                    coord_i_name = f"atom {self.indices[i // 3]}, coord {i % 3}"
                    coord_j_name = f"atom {self.indices[j // 3]}, coord {j % 3}"
                    msg = (
                        f"Failed to compute Hessian element H[{i},{j}] "
                        f"({coord_i_name} <-> {coord_j_name}): {e}"
                    )
                    raise RuntimeError(msg) from e

        # Fill lower triangle (Hessian is symmetric)
        for i in range(n_coords):
            for j in range(i + 1, n_coords):
                hessian[j, i] = hessian[i, j]

        # Final symmetrization for numerical stability
        hessian = 0.5 * (hessian + hessian.T)

        if self.verbose >= 2:
            logger.info("Energy-based Hessian calculation completed")

        return cast(NDArray[np.float64], hessian)

    def _compute_cross_derivative(self, i: int, j: int) -> float:
        """Compute cross-derivative H_ij using 4-point energy stencil.

        For second derivative ∂²E/(∂x_i ∂x_j):
            H_ij = [E(++) - E(+-) - E(-+) + E(--)] / (4δ²)

        where:
        - E(++): energy at r + δ_i + δ_j
        - E(+-): energy at r + δ_i - δ_j
        - E(-+): energy at r - δ_i + δ_j
        - E(--): energy at r - δ_i - δ_j

        Parameters
        ----------
        i : int
            Coordinate index in flattened array (0 to 3N-1)
        j : int
            Coordinate index in flattened array (0 to 3N-1)

        Returns:
        -------
        float
            Hessian element H_ij in eV/Å²

        Raises:
        ------
        RuntimeError
            If energy calculation fails
        """
        # Map from flattened index to (atom, coord)
        atom_i = self.indices[i // 3]
        coord_i = i % 3
        atom_j = self.indices[j // 3]
        coord_j = j % 3

        # Four displacements for second-order cross derivative
        energy_pp = self._get_energy_displaced(atom_i, coord_i, atom_j, coord_j, +1, +1)
        energy_pm = self._get_energy_displaced(atom_i, coord_i, atom_j, coord_j, +1, -1)
        energy_mp = self._get_energy_displaced(atom_i, coord_i, atom_j, coord_j, -1, +1)
        energy_mm = self._get_energy_displaced(atom_i, coord_i, atom_j, coord_j, -1, -1)

        # Second derivative: ∂²E/∂x_i∂x_j
        # Using central difference formula for cross-derivative
        cross_derivative = (energy_pp - energy_pm - energy_mp + energy_mm) / (
            4 * self.delta * self.delta
        )

        return cross_derivative

    def _get_energy_displaced(
        self,
        atom_i: int,
        coord_i: int,
        atom_j: int,
        coord_j: int,
        disp_i: int,
        disp_j: int,
    ) -> float:
        """Get energy at displaced geometry.

        Parameters
        ----------
        atom_i : int
            Index of first atom to displace
        coord_i : int
            Coordinate index (0=x, 1=y, 2=z) for first atom
        atom_j : int
            Index of second atom to displace
        coord_j : int
            Coordinate index (0=x, 1=y, 2=z) for second atom
        disp_i : int
            Displacement sign for first atom (-1 or +1)
        disp_j : int
            Displacement sign for second atom (-1 or +1)

        Returns:
        -------
        float
            Energy at displaced geometry in eV
        """
        # Create displaced structure
        atoms_displaced = self.atoms.copy()

        # Apply displacements
        atoms_displaced.positions[atom_i, coord_i] += disp_i * self.delta
        atoms_displaced.positions[atom_j, coord_j] += disp_j * self.delta

        # Attach calculator
        atoms_displaced.calc = self.calculator

        # Compute energy
        energy = atoms_displaced.get_potential_energy()
        if not np.isfinite(energy):
            msg = "Calculator returned non-finite energy at displacement"
            raise RuntimeError(msg)
        return cast(float, energy)

    def _get_energy_at_ref(self) -> float:
        """Get energy at reference geometry.

        Returns:
        -------
        float
            Reference energy in eV
        """
        return cast(float, self.atoms.get_potential_energy())


__all__ = [
    "EnergyBasedHessianCalculator",
    "CalculatorProtocol",
]
