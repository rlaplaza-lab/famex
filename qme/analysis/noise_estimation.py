"""Noise estimation utilities for Hessian calculations.

This module provides functions to measure and analyze numerical noise in
force and energy calculations. This information is used by adaptive Hessian
calculation strategies to select optimal methods and parameters.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from ase import Atoms
from numpy.typing import NDArray

from qme.analysis.utils import validate_indices
from qme.utils.logging import get_qme_logger

logger = get_qme_logger(__name__)


class CalculatorProtocol(Protocol):
    """Protocol for calculator objects compatible with ASE Atoms.

    Any object that can be assigned to ``atoms.calc`` and provides
    ``get_forces()`` and ``get_potential_energy()`` methods is compatible.
    """

    def get_forces(self, atoms: Atoms | None = None) -> np.ndarray:
        """Compute forces for the given atoms structure."""
        ...

    def get_potential_energy(
        self, atoms: Atoms | None = None, force_consistent: bool = False
    ) -> float:
        """Compute potential energy for the given atoms structure."""
        ...


def estimate_richardson_noise(
    hessian1: NDArray[np.float64], hessian2: NDArray[np.float64]
) -> float:
    """Estimate noise level by comparing Hessians at different deltas.

    Uses Richardson extrapolation error as a proxy for noise. For second-order
    schemes, the error should scale as δ² if there's no noise. Any deviation
    from this scaling indicates noise contamination.

    Parameters
    ----------
    hessian1 : np.ndarray
        Hessian computed at larger delta
    hessian2 : np.ndarray
        Hessian computed at smaller delta (typically delta/2)

    Returns
    -------
    float
        Estimated RMS noise level in eV/Å²

    Examples
    --------
    >>> # Compute Hessians at different deltas
    >>> H_01 = compute_hessian(atoms, calc, delta=0.01)
    >>> H_005 = compute_hessian(atoms, calc, delta=0.005)
    >>> noise = estimate_richardson_noise(H_01, H_005)
    >>> print(f"Estimated noise: {noise:.2e} eV/Å²")
    """
    # Compute Richardson extrapolated estimate (4th order)
    # For 2nd order scheme: extrap = (4*H(delta/2) - H(delta)) / 3
    extrapolated = (4.0 * hessian2 - hessian1) / 3.0

    # Estimate truncation error from Richardson
    truncation_error = np.abs(hessian2 - extrapolated)

    # RMS of truncation error across matrix
    noise_estimate = np.sqrt(np.mean(truncation_error**2))

    return float(noise_estimate)


def estimate_force_noise(
    atoms: Atoms,
    calculator: CalculatorProtocol,
    n_samples: int = 5,
    perturbation_size: float = 1e-5,
    indices: list[int] | None = None,
) -> float:
    """Estimate noise level in force calculations from small perturbations.

    This helps detect if the calculator has inherent noise that will
    contaminate finite difference Hessians. The method makes multiple
    tiny random perturbations and measures force consistency.

    Parameters
    ----------
    atoms : Atoms
        Structure to test
    calculator : CalculatorProtocol
        Calculator compatible with ASE
    n_samples : int, default 5
        Number of random perturbations to test
    perturbation_size : float, default 1e-5
        Size of random perturbations in Å
    indices : list[int], optional
        Indices of atoms to include in noise estimation
        If None, all atoms are included

    Returns
    -------
    float
        Estimated RMS noise in forces (eV/Å)

    Examples
    --------
    >>> noise = estimate_force_noise(atoms, calc, n_samples=10)
    >>> if noise > 1e-4:
    ...     print("High force noise detected, may affect FD Hessians")
    """
    indices = validate_indices(atoms, indices)

    # Get reference forces
    atoms_ref = atoms.copy()
    atoms_ref.calc = calculator
    forces_ref = atoms_ref.get_forces()[indices].flatten()

    # Make small perturbations and recompute
    noise_samples = []

    for i in range(n_samples):
        # Random small perturbation
        perturbed_atoms = atoms.copy()
        perturb = np.random.normal(0, perturbation_size, perturbed_atoms.positions.shape)
        perturbed_atoms.positions += perturb
        perturbed_atoms.calc = calculator

        try:
            forces_perturbed = perturbed_atoms.get_forces()
            forces_perturbed_flat = forces_perturbed[indices].flatten()

            # Estimate noise from difference
            noise = np.abs(forces_perturbed_flat - forces_ref)
            noise_samples.append(noise)
        except Exception:
            # If calculation fails, skip this sample
            logger.debug(f"Force calculation failed for perturbation sample {i}, skipping")
            continue

    if len(noise_samples) == 0:
        msg = "All force noise samples failed. Cannot estimate noise."
        raise RuntimeError(msg)

    # RMS noise across all samples
    noise_array = np.array(noise_samples)
    rms_noise = np.sqrt(np.mean(noise_array**2))

    return float(rms_noise)


def estimate_optimal_delta(
    atoms: Atoms,
    calculator: CalculatorProtocol,
    delta_range: tuple[float, float] = (0.001, 0.05),
    method: str = "central",
    target_noise: float = 1e-5,
    max_iterations: int = 5,
    indices: list[int] | None = None,
    verbose: int = 0,
) -> tuple[float, float]:
    """Estimate optimal delta for finite difference Hessian calculation.

    Uses binary search with Richardson extrapolation error to find the delta
    that minimizes noise while avoiding roundoff errors.

    Parameters
    ----------
    atoms : Atoms
        Structure to test
    calculator : CalculatorProtocol
        Calculator compatible with ASE
    delta_range : tuple[float, float], default (0.001, 0.05)
        (min_delta, max_delta) search range
    method : str, default "central"
        FD method to use ('central' or '5point')
    target_noise : float, default 1e-5
        Target noise level in eV/Å²
    max_iterations : int, default 5
        Maximum binary search iterations
    indices : list[int], optional
        Indices of atoms to include
        If None, all atoms are included
    verbose : int, default 0
        Verbosity level

    Returns
    -------
    tuple[float, float]
        (optimal_delta, estimated_noise)

    Examples
    --------
    >>> opt_delta, noise = estimate_optimal_delta(atoms, calc, delta_range=(0.001, 0.05))
    >>> hessian_calc = HessianCalculator(atoms, calc, delta=opt_delta)
    >>> hessian = hessian_calc.calculate_numerical_hessian()
    """
    # Import locally to avoid circular imports
    from qme.analysis.finite_differences import (
        CentralDifferenceScheme,
        FivePointCentralDifferenceScheme,
    )

    min_delta, max_delta = delta_range

    # Validate range
    if min_delta >= max_delta:
        msg = f"Invalid delta_range: min_delta ({min_delta}) >= max_delta ({max_delta})"
        raise ValueError(msg)

    # Select finite difference scheme class
    if method == "central":
        scheme_class: type = CentralDifferenceScheme
    elif method == "5point":
        scheme_class = FivePointCentralDifferenceScheme
    else:
        msg = f"Unknown method '{method}', use 'central' or '5point'"
        raise ValueError(msg)

    indices = validate_indices(atoms, indices)

    best_delta = 0.01  # Reasonable default
    best_noise = float("inf")

    # Binary search for optimal delta
    current_low = min_delta
    current_high = max_delta

    for iteration in range(max_iterations):
        # Try delta in middle of range
        current_delta = (current_low + current_high) / 2.0
        half_delta = current_delta / 2.0

        if verbose >= 1:
            logger.debug(
                f"Iteration {iteration + 1}: trying delta={current_delta:.4f}, "
                f"half={half_delta:.4f}"
            )

        # Compute Hessians at both deltas
        try:
            hessian_current = _compute_hessian_at_delta(
                atoms, calculator, current_delta, scheme_class, indices
            )
            hessian_half = _compute_hessian_at_delta(
                atoms, calculator, half_delta, scheme_class, indices
            )

            # Estimate noise
            noise_estimate = estimate_richardson_noise(hessian_current, hessian_half)

            if verbose >= 1:
                logger.debug(f"  Noise estimate: {noise_estimate:.2e} eV/Å²")

            # Track best result
            if noise_estimate < best_noise:
                best_noise = noise_estimate
                best_delta = current_delta

            # Check convergence
            if noise_estimate < target_noise:
                if verbose >= 1:
                    logger.debug(f"  Converged! Noise below target {target_noise:.2e}")
                return current_delta, noise_estimate

            # Binary search: if too noisy, try smaller delta
            # If noise is acceptable, we can stop here
            if noise_estimate > target_noise * 100:  # Way too noisy
                current_high = current_delta
            elif noise_estimate > target_noise * 10:  # Somewhat noisy
                current_high = (current_low + current_delta) / 2.0
            else:
                # Acceptable noise, stop searching
                break

        except Exception as e:
            if verbose >= 1:
                logger.warning(f"  Failed to compute at delta={current_delta:.4f}: {e}")
            # If calculation fails, try smaller delta
            current_high = current_delta

    if verbose >= 1:
        logger.debug(f"Final delta: {best_delta:.4f}, noise: {best_noise:.2e} eV/Å²")

    return best_delta, best_noise


def _compute_hessian_at_delta(
    atoms: Atoms,
    calculator: CalculatorProtocol,
    delta: float,
    scheme: type,
    indices: list[int],
) -> NDArray[np.float64]:
    """Compute Hessian at a specific delta.

    Parameters
    ----------
    atoms : Atoms
        Structure
    calculator : CalculatorProtocol
        Calculator
    delta : float
        Step size
    scheme : type
        Finite difference scheme class
    indices : list[int]
        Atom indices to include

    Returns
    -------
    np.ndarray
        Hessian matrix
    """
    from qme.analysis.hessian import HessianCalculator

    hessian_calc = HessianCalculator(
        atoms=atoms,
        calculator=calculator,
        delta=delta,
        method="central" if "Central" in scheme.__name__ else "5point",
        indices=indices,
        verbose=0,
    )
    return hessian_calc.calculate_numerical_hessian()


__all__ = [
    "estimate_richardson_noise",
    "estimate_force_noise",
    "estimate_optimal_delta",
    "CalculatorProtocol",
]
