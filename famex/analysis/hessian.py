"""Hessian calculation using finite differences.

This module provides the HessianCalculator class for numerical Hessian
calculation using finite difference methods.

The finite difference schemes are implemented in famex.analysis.finite_differences.
Richardson extrapolation can be combined with 3-point or 5-point schemes
for additional accuracy improvements.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import numpy as np
from ase import Atoms
from numpy.typing import NDArray

from famex.analysis.ase_integration import can_use_ase, hessian_via_ase
from famex.analysis.finite_differences import (
    CentralDifferenceScheme,
    FiniteDifferenceScheme,
    FivePointCentralDifferenceScheme,
    ForwardDifferenceScheme,
    SevenPointCentralDifferenceScheme,
)
from famex.analysis.utils import validate_indices
from famex.utils.logging import get_famex_logger

# Optional progress bar support
HAS_TQDM: bool
if TYPE_CHECKING:
    from tqdm import tqdm

    HAS_TQDM = True
else:
    try:
        from tqdm import tqdm

        HAS_TQDM = True
    except ImportError:
        HAS_TQDM = False

        # Dummy tqdm for when not available
        def tqdm(iterable: Any, *args: Any, **kwargs: Any) -> Any:
            return iterable


logger = get_famex_logger(__name__)


class CalculatorProtocol(Protocol):
    """Protocol for calculator objects compatible with ASE Atoms.

    Any object that can be assigned to ``atoms.calc`` and provides
    ``get_forces()`` / ``get_potential_energy()`` is compatible. This includes:
    - FAMEX calculators (BasePotential subclasses)
    - ASE Calculator instances
    - Any object with compatible interface
    """

    def get_forces(self, atoms: Atoms | None = None) -> np.ndarray:
        """Compute forces for the given atoms structure.

        Parameters
        ----------
        atoms : Atoms, optional
            Atoms structure. If None, uses previously set atoms.

        Returns
        -------
        np.ndarray
            Forces array of shape (N, 3) where N is number of atoms.

        """
        ...

    def get_potential_energy(
        self, atoms: Atoms | None = None, force_consistent: bool = False
    ) -> float:
        """Compute potential energy for the given atoms structure."""
        ...


@runtime_checkable
class CopyableCalculator(CalculatorProtocol, Protocol):
    """Calculator that can produce an independent instance for parallel FD."""

    def copy(self) -> CalculatorProtocol: ...


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
    - '7point': Very high accuracy, 6th order accuracy, requires 6N+1 calculations

    Richardson extrapolation can improve accuracy by combining results from two
    different step sizes, effectively canceling leading error terms.

    Examples
    --------
    >>> from ase import Atoms
    >>> from famex.potentials.mock_potential import MockCalculator
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

    Notes
    -----
    - The Hessian is automatically symmetrized: H = (H + H^T) / 2
    - For N atoms, the Hessian has shape (3N, 3N)
    - When using `indices`, only specified atoms contribute to the Hessian
    - Large `delta` values (>10% of typical bond length) may cause inaccurate results
    - Very small `delta` values (<1e-5 Å) may suffer from numerical precision issues

    See Also
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
        adaptive_delta: bool = False,
        delta_range: tuple[float, float] = (0.001, 0.05),
        target_noise: float = 1e-5,
        max_iterations: int = 5,
        n_workers: int | None = None,
    ) -> None:
        """Initialize Hessian calculator.

        Parameters
        ----------
        atoms : Atoms
            ASE Atoms object
        calculator : CalculatorProtocol
            Calculator compatible with ASE (FAMEX calculator, ASE Calculator, etc.)
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
            - 3: Very verbose (includes progress bar if tqdm available)
        adaptive_delta : bool, default False
            Whether to automatically select optimal delta based on noise estimation.
            Uses Richardson extrapolation error to find best delta.
        delta_range : tuple[float, float], default (0.001, 0.05)
            (min_delta, max_delta) search range for adaptive delta selection.
        target_noise : float, default 1e-5
            Target noise level in eV/Å² for adaptive delta selection.
        max_iterations : int, default 5
            Maximum iterations for adaptive delta selection.
        n_workers : int, optional
            Number of thread workers for force calculations.
            If None or 1, uses sequential computation. If > 1, parallelizes
            independent displacements with a ThreadPoolExecutor. Each worker
            needs an independent calculator with ``copy()``;
            otherwise evaluation stays serial.

        Raises
        ------
        ValueError
            If delta <= 0, indices are invalid, or other parameters are inconsistent

        Examples
        --------
        >>> from ase import Atoms
        >>> from famex.analysis.hessian import HessianCalculator
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
        >>> # Adaptive delta selection for optimal accuracy
        >>> hessian_calc_adaptive = HessianCalculator(
        ...     atoms, calc, delta=0.01, adaptive_delta=True, max_iterations=5
        ... )
        >>> hessian = hessian_calc_adaptive.calculate_numerical_hessian()

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
        self.richardson = richardson
        self.delta2 = delta2
        self.verbose = verbose
        self.adaptive_delta = adaptive_delta
        self.delta_range = delta_range
        self.target_noise = target_noise
        self.max_iterations = max_iterations

        # Parallelization settings
        if n_workers is not None and n_workers < 1:
            msg = f"n_workers must be >= 1, got {n_workers}"
            raise ValueError(msg)
        self.n_workers = n_workers

        # Cache for reference forces (computed once, reused)
        self._reference_forces: np.ndarray | None = None

        # Richardson extrapolation order (None if not using Richardson)
        self._richardson_order: int | None = None

        # Error recovery settings
        self.max_retries = 3

        if len(self.indices) > 0:
            positions = atoms.positions[self.indices]
            max_distance = np.max(np.linalg.norm(positions, axis=1))
            min_distance = np.min(
                (
                    [
                        np.linalg.norm(pos - positions[j])
                        for i, pos in enumerate(positions)
                        for j in range(i + 1, len(positions))
                    ]
                    if len(positions) > 1
                    else [max_distance]
                ),
            )
            if delta > 0.1 * min_distance:
                logger.warning(
                    "delta (%.4f Å) is large compared to minimum interatomic distance (%.4f Å). Consider using delta < %.4f Å",
                    delta,
                    min_distance,
                    0.1 * min_distance,
                )
            if delta < 1e-5:
                logger.warning(
                    "delta (%.4f Å) is very small. Numerical precision may be limited.", delta
                )

        if method == "central":
            self.scheme: FiniteDifferenceScheme = cast(
                FiniteDifferenceScheme, CentralDifferenceScheme()
            )
        elif method == "forward":
            self.scheme = cast(FiniteDifferenceScheme, ForwardDifferenceScheme())
        elif method == "5point":
            self.scheme = cast(FiniteDifferenceScheme, FivePointCentralDifferenceScheme())
        elif method == "7point":
            self.scheme = cast(FiniteDifferenceScheme, SevenPointCentralDifferenceScheme())
        else:
            msg = f"Unknown finite difference method: {method}. Use 'forward', 'central', '5point', or '7point'."
            raise ValueError(msg)

        if self.richardson:
            if isinstance(self.scheme, SevenPointCentralDifferenceScheme):
                self._richardson_order = 6
            elif isinstance(self.scheme, FivePointCentralDifferenceScheme):
                self._richardson_order = 4
            elif isinstance(self.scheme, CentralDifferenceScheme):
                self._richardson_order = 2
            else:
                msg = "Richardson extrapolation currently supported only for 'central', '5point', or '7point' methods."
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
                    "For Richardson extrapolation, typically delta2 < delta. Got delta=%s, delta2=%s",
                    self.delta,
                    self.delta2,
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

        For basic cases (central differences, no advanced features), this method
        delegates to ASE's Vibrations class. For advanced cases (5-point, 7-point,
        Richardson extrapolation, adaptive delta), uses custom implementation.

        Returns
        -------
        np.ndarray
            Hessian matrix of shape (3N, 3N) where N is the number of atoms
            in `indices`. Units are eV/Å². The matrix is symmetric and includes
            all selected atoms.

        Raises
        ------
        RuntimeError
            If force calculations fail at any displacement step

        Notes
        -----
        - For basic central differences: delegates to ASE's Vibrations class
        - For central differences with N atoms: performs 2*3*N force calculations
        - For 5-point scheme: performs 4*3*N force calculations
        - For Richardson extrapolation: doubles the number of calculations
        - The final Hessian is symmetrized to ensure H = H^T (required by theory)
        - Progress information is logged if verbose >= 2
        - With adaptive_delta=True, performs additional calculations to optimize delta

        Examples
        --------
        >>> from ase import Atoms
        >>> from famex.potentials.mock_potential import MockCalculator
        >>> atoms = Atoms('H2O', positions=[[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0]])
        >>> calc = MockCalculator()
        >>> hessian_calc = HessianCalculator(atoms, calc, delta=0.01, verbose=1)
        >>> hessian = hessian_calc.calculate_numerical_hessian()
        >>> print(f"Hessian shape: {hessian.shape}")  # (9, 9) for 3 atoms
        >>> # Check symmetry (should be very close to symmetric)
        >>> symmetry_error = np.max(np.abs(hessian - hessian.T))
        >>> print(f"Symmetry error: {symmetry_error:.2e}")  # Should be ~0

        """
        # Check if ASE can handle this case
        # Determine method string from scheme type
        method_str = "central"
        if isinstance(self.scheme, ForwardDifferenceScheme):
            method_str = "forward"
        elif isinstance(self.scheme, CentralDifferenceScheme):
            method_str = "central"
        else:
            # 5-point, 7-point, etc. - not supported by ASE
            method_str = "unknown"

        if can_use_ase(
            method=method_str,
            richardson=self.richardson,
            adaptive_delta=self.adaptive_delta,
            n_workers=self.n_workers,
        ):
            # Only use ASE for central differences (ASE doesn't support forward well)
            if method_str == "central":
                if self.verbose >= 1:
                    logger.info("Using ASE Vibrations for Hessian calculation")
                return hessian_via_ase(
                    atoms=self.atoms,
                    calculator=self.calculator,
                    delta=self.delta,
                    indices=self.indices,
                    verbose=self.verbose,
                )

        # Reset cache at start of calculation
        self._reference_forces = None

        if self.adaptive_delta:
            return self._calculate_hessian_adaptive()
        return self._calculate_hessian_fixed()

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

        Returns
        -------
        np.ndarray
            Hessian column

        """
        if isinstance(self.scheme, SevenPointCentralDifferenceScheme):
            return self._compute_seven_point_derivative(atom_index, direction, delta)
        elif isinstance(self.scheme, FivePointCentralDifferenceScheme):
            return self._compute_five_point_derivative(atom_index, direction, delta)
        else:
            forces_plus = self._get_forces_displaced_with_retry(atom_index, direction, delta)
            forces_minus = None
            if isinstance(self.scheme, CentralDifferenceScheme):
                forces_minus = self._get_forces_displaced_with_retry(atom_index, direction, -delta)
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

        Returns
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

        return cast(NDArray[np.float64], extrapolated_derivative)

    def _get_reference_forces(self) -> np.ndarray:
        """Get forces at reference geometry.

        Caches the result after first computation to avoid redundant calculations.

        Returns
        -------
        np.ndarray
            Forces on atoms in indices, flattened

        """
        if self._reference_forces is not None:
            return self._reference_forces

        forces = self.atoms.get_forces()
        self._reference_forces = cast(NDArray[np.float64], forces[self.indices].flatten())
        return self._reference_forces

    def _get_forces_displaced_with_retry(
        self,
        atom_index: int,
        direction: int,
        displacement: float,
    ) -> np.ndarray:
        """Get forces for displaced geometry with retry logic.

        Parameters
        ----------
        atom_index : int
            Index of atom to displace
        direction : int
            Coordinate direction (0=x, 1=y, 2=z)
        displacement : float
            Displacement in Å

        Returns
        -------
        np.ndarray
            Forces on atoms in indices, flattened

        Raises
        ------
        RuntimeError
            If force calculation fails after all retries

        """
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return self._get_forces_displaced(atom_index, direction, displacement)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    # Simple retry without backoff
                    if self.verbose >= 2:
                        coord_name = ["x", "y", "z"][direction]
                        logger.debug(
                            "Retry %s/%s for atom %s, %s-displacement %s Å: %s",
                            attempt + 1,
                            self.max_retries,
                            atom_index,
                            coord_name,
                            displacement,
                            e,
                        )
                    continue
                # Final attempt failed
                coord_name = ["x", "y", "z"][direction]
                msg = (
                    f"Force calculation failed after {self.max_retries} attempts "
                    f"for atom {atom_index}, {coord_name}-displacement {displacement:+.4f} Å: {last_error}"
                )
                raise RuntimeError(msg) from last_error
        # This should never be reached, but mypy needs it
        assert last_error is not None, "All retries failed but no error was captured"
        coord_name = ["x", "y", "z"][direction]
        msg = (
            f"Force calculation failed after {self.max_retries} attempts "
            f"for atom {atom_index}, {coord_name}-displacement {displacement:+.4f} Å: {last_error}"
        )
        raise RuntimeError(msg) from last_error

    def _supports_parallel_workers(self) -> bool:
        return isinstance(self.calculator, CopyableCalculator)

    def _calculator_for_displacement(self) -> CalculatorProtocol:
        """Return a calculator instance safe for one force evaluation."""
        if (
            self.n_workers is not None
            and self.n_workers > 1
            and isinstance(self.calculator, CopyableCalculator)
        ):
            return self.calculator.copy()
        return self.calculator

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

        Returns
        -------
        np.ndarray
            Forces on atoms in indices, flattened

        Raises
        ------
        RuntimeError
            If force calculation fails or returns invalid results

        """
        coord_name = ["x", "y", "z"][direction]

        atoms_displaced = self.atoms.copy()
        atoms_displaced.positions[atom_index, direction] += displacement

        atoms_displaced.calc = self._calculator_for_displacement()

        try:
            forces = atoms_displaced.get_forces()
        except Exception as e:
            msg = (
                f"Force calculation failed for atom {atom_index}, "
                f"{coord_name}-displacement {displacement:+.4f} Å: {e}"
            )
            raise RuntimeError(msg) from e

        if np.any(np.isnan(forces)) or np.any(np.isinf(forces)):
            nan_count = np.sum(np.isnan(forces))
            inf_count = np.sum(np.isinf(forces))
            msg = (
                f"Calculator returned invalid forces for atom {atom_index}, "
                f"{coord_name}-displacement {displacement:+.4f} Å: "
                f"{nan_count} NaN, {inf_count} Inf values"
            )
            raise RuntimeError(msg)

        result = forces[self.indices].flatten()
        return np.asarray(result)

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

        Returns
        -------
        np.ndarray
            Hessian column using 5-point stencil

        """
        forces_minus2 = self._get_forces_displaced_with_retry(atom_index, direction, -2 * delta)
        forces_minus = self._get_forces_displaced_with_retry(atom_index, direction, -delta)
        forces_plus = self._get_forces_displaced_with_retry(atom_index, direction, delta)
        forces_plus2 = self._get_forces_displaced_with_retry(atom_index, direction, 2 * delta)

        return self.scheme.compute_derivative(
            forces_plus,
            forces_minus,
            None,  # forces_ref not used for 5-point
            delta,
            forces_plus2=forces_plus2,
            forces_minus2=forces_minus2,
        )

    def _compute_seven_point_derivative(
        self,
        atom_index: int,
        direction: int,
        delta: float,
    ) -> np.ndarray:
        """Compute 7-point finite difference derivative.

        The 7-point scheme requires forces at ±delta, ±2delta, and ±3delta displacements.

        Parameters
        ----------
        atom_index : int
            Index of atom to displace
        direction : int
            Coordinate direction (0=x, 1=y, 2=z)
        delta : float
            Displacement step size

        Returns
        -------
        np.ndarray
            Hessian column using 7-point stencil

        """
        forces_minus3 = self._get_forces_displaced_with_retry(atom_index, direction, -3 * delta)
        forces_minus2 = self._get_forces_displaced_with_retry(atom_index, direction, -2 * delta)
        forces_minus = self._get_forces_displaced_with_retry(atom_index, direction, -delta)
        forces_plus = self._get_forces_displaced_with_retry(atom_index, direction, delta)
        forces_plus2 = self._get_forces_displaced_with_retry(atom_index, direction, 2 * delta)
        forces_plus3 = self._get_forces_displaced_with_retry(atom_index, direction, 3 * delta)

        return self.scheme.compute_derivative(
            forces_plus,
            forces_minus,
            None,  # forces_ref not used for 7-point
            delta,
            forces_plus2=forces_plus2,
            forces_minus2=forces_minus2,
            forces_plus3=forces_plus3,
            forces_minus3=forces_minus3,
        )

    def _calculate_hessian_adaptive(self) -> np.ndarray:
        """Calculate Hessian with adaptive delta selection.

        Uses iterative refinement to find optimal delta that balances truncation
        and roundoff errors based on Richardson extrapolation error.

        Returns
        -------
        np.ndarray
            Hessian matrix optimized for low noise

        """
        from famex.analysis.noise_estimation import estimate_richardson_noise

        min_delta, max_delta = self.delta_range
        current_delta = self.delta

        # Track best result
        best_hessian: np.ndarray | None = None
        best_noise = float("inf")
        best_delta = current_delta

        for iteration in range(self.max_iterations):
            if self.verbose >= 1:
                logger.info(
                    "Adaptive delta iteration %s: trying δ=%.4f Å", iteration + 1, current_delta
                )

            # Compute Hessian at current delta
            try:
                hessian_current = self._calculate_hessian_at_delta(current_delta)
            except Exception as e:
                if self.verbose >= 1:
                    logger.warning("  Failed at delta=%.4f: %s", current_delta, e)
                # Try smaller delta
                current_delta = current_delta / 2.0
                if current_delta < min_delta:
                    break
                continue

            # Also compute at half-delta for noise estimation
            half_delta = current_delta / 2.0
            try:
                hessian_half = self._calculate_hessian_at_delta(half_delta)
            except Exception as e:
                if self.verbose >= 1:
                    logger.warning("  Failed at half-delta=%.4f: %s", half_delta, e)
                # Use current result
                return hessian_current

            # Estimate noise level
            noise_estimate = estimate_richardson_noise(hessian_current, hessian_half)

            if self.verbose >= 1:
                logger.info("  Noise estimate: %.2e eV/Å²", noise_estimate)

            # Track best result
            if noise_estimate < best_noise:
                best_hessian = hessian_current
                best_noise = noise_estimate
                best_delta = current_delta

            # Check convergence
            if noise_estimate < self.target_noise:
                if self.verbose >= 1:
                    logger.info("  Converged! Noise below threshold %.2e", self.target_noise)
                return hessian_current

            if noise_estimate > self.target_noise * 100:  # Way too noisy
                # Try smaller delta
                current_delta = half_delta
                if current_delta < min_delta:
                    if self.verbose >= 1:
                        logger.warning(
                            "  Delta %.4f below minimum, using best result.", current_delta
                        )
                    break
            else:
                # Acceptable noise level
                break

        if self.verbose >= 1:
            logger.info("Final delta: %.4f Å, noise: %.2e eV/Å²", best_delta, best_noise)

        if best_hessian is None:
            msg = "Adaptive delta selection failed to produce any valid Hessian"
            raise RuntimeError(msg)

        return best_hessian

    def _calculate_hessian_at_delta(self, delta: float) -> np.ndarray:
        """Calculate Hessian using specified delta (temporarily override).

        Parameters
        ----------
        delta : float
            Step size to use

        Returns
        -------
        np.ndarray
            Hessian matrix computed at specified delta
        """
        # Save original delta
        original_delta = self.delta
        try:
            self.delta = delta
            # Use the original non-adaptive calculation
            return self._calculate_hessian_fixed()
        finally:
            self.delta = original_delta

    def _compute_hessian_column(
        self, j: int, forces_ref: np.ndarray | None
    ) -> tuple[int, np.ndarray]:
        """Compute a single Hessian column (helper for parallelization).

        Parameters
        ----------
        j : int
            Column index (0 to n_coords-1)
        forces_ref : np.ndarray | None
            Reference forces (for forward differences)

        Returns
        -------
        tuple[int, np.ndarray]
            (column_index, hessian_column)

        """
        atom_j = self.indices[j // 3]
        coord_j = j % 3

        if self.richardson:
            column = self._compute_richardson_extrapolated_derivative(atom_j, coord_j)
        else:
            column = self._compute_derivative_at_delta(atom_j, coord_j, self.delta, forces_ref)

        return (j, column)

    def _calculate_hessian_fixed(self) -> np.ndarray:
        """Original fixed-delta Hessian calculation (non-adaptive path).

        This is the core Hessian computation logic extracted from
        calculate_numerical_hessian to support adaptive delta selection.

        Returns
        -------
        np.ndarray
            Hessian matrix
        """
        n_atoms = len(self.indices)
        n_coords = 3 * n_atoms
        hessian = np.zeros((n_coords, n_coords))

        if self.verbose >= 2:
            logger.info(
                "Calculating Hessian for %s atoms using %s", n_atoms, type(self.scheme).__name__
            )
            if self.n_workers and self.n_workers > 1:
                logger.info("Using %s parallel workers", self.n_workers)

        forces_ref = None
        if isinstance(self.scheme, ForwardDifferenceScheme):
            forces_ref = self._get_reference_forces()

        want_parallel = self.n_workers is not None and self.n_workers > 1 and not self.richardson
        use_parallel = want_parallel and self._supports_parallel_workers()
        if want_parallel and not use_parallel and self.verbose >= 1:
            logger.warning("Calculator has no copy() method; computing Hessian serially")

        use_progress_bar = self.verbose >= 3 and HAS_TQDM and not use_parallel
        start_time = time.time() if self.verbose >= 1 else None

        if use_parallel:
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                # Submit all column computations
                futures = {
                    executor.submit(self._compute_hessian_column, j, forces_ref): j
                    for j in range(n_coords)
                }

                # Collect results as they complete with progress tracking
                completed = 0
                if self.verbose >= 1:
                    logger.info("Computing %s columns in parallel...", n_coords)

                for future in as_completed(futures):
                    j = futures[future]  # Get column index from future mapping
                    try:
                        _, column = future.result()
                        hessian[:, j] = column
                        completed += 1
                        if self.verbose >= 2:
                            elapsed = time.time() - (start_time or 0)
                            rate = completed / elapsed if elapsed > 0 else 0
                            eta = (n_coords - completed) / rate if rate > 0 else 0
                            logger.debug(
                                "Completed %s/%s columns (%.1f cols/s, ETA: %.1fs)",
                                completed,
                                n_coords,
                                rate,
                                eta,
                            )
                    except Exception as e:
                        coord_j = j % 3
                        coord_name = ["x", "y", "z"][coord_j]
                        atom_j = self.indices[j // 3]
                        msg = (
                            f"Failed to compute Hessian column for atom {atom_j}, "
                            f"coordinate {coord_name} (index {j}/{n_coords - 1}): {e}"
                        )
                        raise RuntimeError(msg) from e
        else:
            # Sequential computation (original approach)
            coord_iter = tqdm(
                range(n_coords), desc="Computing Hessian", disable=not use_progress_bar
            )
            for j in coord_iter:
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

                if self.verbose >= 2 and not use_progress_bar:
                    elapsed = time.time() - (start_time or 0) if start_time else 0
                    rate = (j + 1) / elapsed if elapsed > 0 else 0
                    eta = (n_coords - j - 1) / rate if rate > 0 else 0
                    logger.debug(
                        "Completed coordinate %s/%s (%.2f cols/s, ETA: %.1fs)",
                        j + 1,
                        n_coords,
                        rate,
                        eta,
                    )

        # Symmetrize Hessian: H_sym = (H + H^T) / 2
        HESSIAN_SYMMETRIZATION_FACTOR = 0.5
        symmetrized = HESSIAN_SYMMETRIZATION_FACTOR * (hessian + hessian.T)

        if self.verbose >= 1:
            elapsed = time.time() - (start_time or 0) if start_time else 0
            logger.info("Hessian calculation completed in %.2f seconds", elapsed)
        elif self.verbose >= 2:
            logger.info("Hessian calculation completed")
        return symmetrized


__all__ = [
    "CalculatorProtocol",
    "HessianCalculator",
]
