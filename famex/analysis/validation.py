"""Hessian validation utilities for frequency analysis.

This module provides functions to validate Hessian matrices and warn
about potential issues such as asymmetry, NaN/Inf values, or poor conditioning.
"""

from __future__ import annotations

import numpy as np

from famex.utils.logging import get_famex_logger

logger = get_famex_logger(__name__)

__all__ = ["validate_hessian"]


def validate_hessian(
    hessian: np.ndarray,
    tolerance_symmetry: float = 1e-6,
    max_condition_number: float = 1e18,
    warn_on_issues: bool = True,
    estimated_noise: float | None = None,
    force_noise_estimate: float | None = None,
    positions: np.ndarray | None = None,
    masses: np.ndarray | None = None,
) -> dict[str, bool | float | tuple[int, int]]:
    """Validate Hessian matrix and warn about potential issues.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix to validate
    tolerance_symmetry : float
        Tolerance for symmetry check (default: 1e-6)
    max_condition_number : float
        Maximum acceptable condition number before warning (default: 1e18)
        Note: ML potentials often have higher condition numbers (1e16-1e18)
        that are still numerically stable for frequency calculations.
        Only condition numbers > 1e18 are considered truly problematic.
    warn_on_issues : bool
        If True, log warnings when issues are detected (default: True)
    estimated_noise : float, optional
        Estimated noise level from Richardson extrapolation (eV/Å²)
    force_noise_estimate : float, optional
        Estimated noise in force calculations (eV/Å)
    positions : np.ndarray, optional
        Atomic positions (N, 3). With matching ``masses`` and a 3N x 3N Hessian,
        the condition number uses the vibrational block after T/R projection.
    masses : np.ndarray, optional
        Atomic masses (N,). Required together with ``positions``.

    Returns
    -------
    dict[str, bool | float]
        Validation results containing:
        - is_valid: Overall validity (bool)
        - is_symmetric: Whether matrix is symmetric (bool)
        - has_nan: Whether matrix contains NaN values (bool)
        - has_inf: Whether matrix contains Inf values (bool)
        - condition_number: Condition number of the matrix (float)
        - max_asymmetry: Maximum asymmetry (float)
        - shape: Shape of the matrix (tuple)
        - estimated_noise: Estimated noise level from Richardson (float, optional)
        - force_noise_estimate: Estimated force noise (float, optional)

    """
    results: dict[str, bool | float | tuple[int, int]] = {
        "is_valid": True,
        "is_symmetric": True,
        "has_nan": False,
        "has_inf": False,
        "condition_number": 0.0,
        "max_asymmetry": 0.0,
        "shape": hessian.shape,
    }
    if estimated_noise is not None:
        results["estimated_noise"] = estimated_noise
    if force_noise_estimate is not None:
        results["force_noise_estimate"] = force_noise_estimate

    if hessian.shape[0] != hessian.shape[1]:
        if warn_on_issues:
            logger.warning(
                f"Hessian is not square: shape {hessian.shape}. Expected square matrix (3N x 3N)."
            )
        results["is_valid"] = False
        return results

    has_nan = bool(np.any(np.isnan(hessian)))
    has_inf = bool(np.any(np.isinf(hessian)))
    results["has_nan"] = has_nan
    results["has_inf"] = has_inf

    if has_nan and warn_on_issues:
        logger.warning("Hessian contains NaN values. This indicates a calculation error.")
    if has_inf and warn_on_issues:
        logger.warning("Hessian contains infinite values. This indicates a calculation error.")

    if has_nan or has_inf:
        results["is_valid"] = False
    else:
        asymmetry = np.abs(hessian - hessian.T)
        max_asymmetry = float(np.max(asymmetry))
        results["max_asymmetry"] = max_asymmetry
        is_symmetric = max_asymmetry < tolerance_symmetry
        results["is_symmetric"] = is_symmetric
        if not is_symmetric and warn_on_issues:
            logger.warning(
                f"Hessian is not symmetric. Maximum asymmetry: {max_asymmetry:.2e}. "
                f"Tolerance: {tolerance_symmetry:.2e}. "
                "This may indicate numerical errors or non-stationary geometry."
            )

        matrix = 0.5 * (hessian + hessian.T)
        n_null = 0
        if positions is not None and masses is not None:
            positions = np.asarray(positions, dtype=np.float64)
            masses = np.asarray(masses, dtype=np.float64)
            n_atoms = positions.shape[0]
            if (
                positions.ndim == 2
                and positions.shape[1] == 3
                and masses.shape == (n_atoms,)
                and matrix.shape == (3 * n_atoms, 3 * n_atoms)
            ):
                # Lazy import avoids analysis ↔ optimizers cycle
                from famex.optimizers.ts_step import (
                    build_translation_rotation_basis,
                    project_hessian,
                )

                basis = build_translation_rotation_basis(positions, masses)
                n_null = int(basis.shape[1]) if basis.size else 0
                matrix = project_hessian(matrix, basis)

        if matrix.shape[0] > n_null:
            eigenvalues = np.linalg.eigvalsh(matrix)
            eigenvalues = eigenvalues[np.isfinite(eigenvalues)]
            if n_null > 0 and len(eigenvalues) > n_null:
                vibrational = eigenvalues[np.argsort(np.abs(eigenvalues))[n_null:]]
            else:
                vibrational = eigenvalues
            eigenvalues_abs = np.abs(vibrational[vibrational != 0])

            if len(eigenvalues_abs) == 0:
                if warn_on_issues:
                    logger.warning(
                        "Hessian has no finite eigenvalues. This indicates a severe problem."
                    )
                results["is_valid"] = False
            else:
                condition_number = float(np.max(eigenvalues_abs) / np.min(eigenvalues_abs))
                results["condition_number"] = condition_number
                if condition_number > max_condition_number and warn_on_issues:
                    logger.warning(
                        f"Hessian is ill-conditioned. Condition number: {condition_number:.2e}. "
                        f"Maximum acceptable: {max_condition_number:.2e}. "
                        "This may lead to numerical instability in frequency calculations."
                    )

    HIGH_NOISE_THRESHOLD = 0.01  # eV/Å²
    HIGH_FORCE_NOISE_THRESHOLD = 1e-3  # eV/Å

    if estimated_noise is not None and estimated_noise > HIGH_NOISE_THRESHOLD and warn_on_issues:
        logger.warning(
            f"Hessian has high estimated noise: {estimated_noise:.2e} eV/Å². "
            f"Threshold: {HIGH_NOISE_THRESHOLD:.2e}. "
            "This may indicate numerical errors or PES instability."
        )

    if (
        force_noise_estimate is not None
        and force_noise_estimate > HIGH_FORCE_NOISE_THRESHOLD
        and warn_on_issues
    ):
        logger.warning(
            f"High force noise detected: {force_noise_estimate:.2e} eV/Å. "
            f"Threshold: {HIGH_FORCE_NOISE_THRESHOLD:.2e}. "
            "This may contaminate finite difference Hessians."
        )

    if not results["is_symmetric"] or has_nan or has_inf:
        results["is_valid"] = False

    return results
