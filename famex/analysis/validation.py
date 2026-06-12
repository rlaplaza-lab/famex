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
    # Initialize results with required fields
    results: dict[str, bool | float | tuple[int, int]] = {
        "is_valid": True,
        "is_symmetric": True,
        "has_nan": False,
        "has_inf": False,
        "condition_number": 0.0,
        "max_asymmetry": 0.0,
        "shape": hessian.shape,
    }
    # Add optional noise metrics
    if estimated_noise is not None:
        results["estimated_noise"] = estimated_noise
    if force_noise_estimate is not None:
        results["force_noise_estimate"] = force_noise_estimate

    # Check shape
    if hessian.shape[0] != hessian.shape[1]:
        if warn_on_issues:
            logger.warning(
                f"Hessian is not square: shape {hessian.shape}. Expected square matrix (3N x 3N)."
            )
        results["is_valid"] = False
        return results

    # Check for NaN and Inf values
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

    # Check symmetry
    asymmetry = np.abs(hessian - hessian.T)
    max_asymmetry = np.max(asymmetry)
    results["max_asymmetry"] = float(max_asymmetry)
    is_symmetric = max_asymmetry < tolerance_symmetry
    results["is_symmetric"] = is_symmetric
    if not is_symmetric:
        if warn_on_issues:
            logger.warning(
                f"Hessian is not symmetric. Maximum asymmetry: {max_asymmetry:.2e}. "
                f"Tolerance: {tolerance_symmetry:.2e}. "
                "This may indicate numerical errors or non-stationary geometry."
            )

    # Check condition number
    eigenvalues = np.linalg.eigvals(hessian)
    eigenvalues = eigenvalues[np.isfinite(eigenvalues)]
    eigenvalues_abs = np.abs(eigenvalues[eigenvalues != 0])

    if len(eigenvalues_abs) == 0:
        if warn_on_issues:
            logger.warning("Hessian has no finite eigenvalues. This indicates a severe problem.")
        results["is_valid"] = False
    else:
        condition_number = np.max(eigenvalues_abs) / np.min(eigenvalues_abs)
        results["condition_number"] = float(condition_number)
        if condition_number > max_condition_number and warn_on_issues:
            logger.warning(
                f"Hessian is ill-conditioned. Condition number: {condition_number:.2e}. "
                f"Maximum acceptable: {max_condition_number:.2e}. "
                "This may lead to numerical instability in frequency calculations."
            )

    # Check noise levels if provided
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

    # Overall validity
    if not is_symmetric or has_nan or has_inf:
        results["is_valid"] = False

    return results
