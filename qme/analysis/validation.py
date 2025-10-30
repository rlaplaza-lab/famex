"""Hessian validation utilities for frequency analysis.

This module provides functions to validate Hessian matrices and warn
about potential issues such as asymmetry, NaN/Inf values, or poor conditioning.
"""

from __future__ import annotations

import numpy as np

from qme.utils.logging import get_qme_logger

logger = get_qme_logger(__name__)

__all__ = ["validate_hessian"]


def validate_hessian(
    hessian: np.ndarray,
    tolerance_symmetry: float = 1e-6,
    max_condition_number: float = 1e12,
    warn_on_issues: bool = True,
) -> dict[str, bool | float]:
    """Validate Hessian matrix and warn about potential issues.

    Parameters
    ----------
    hessian : np.ndarray
        Hessian matrix to validate
    tolerance_symmetry : float
        Tolerance for symmetry check (default: 1e-6)
    max_condition_number : float
        Maximum acceptable condition number before warning (default: 1e12)
    warn_on_issues : bool
        If True, log warnings when issues are detected (default: True)

    Returns:
    -------
    dict[str, Union[bool, float]]
        Validation results containing:
        - is_valid: Overall validity (bool)
        - is_symmetric: Whether matrix is symmetric (bool)
        - has_nan: Whether matrix contains NaN values (bool)
        - has_inf: Whether matrix contains Inf values (bool)
        - condition_number: Condition number of the matrix (float)
        - max_asymmetry: Maximum asymmetry (float)
        - shape: Shape of the matrix (tuple)

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

    # Check shape
    if hessian.shape[0] != hessian.shape[1]:
        if warn_on_issues:
            logger.warning(
                f"Hessian is not square: shape {hessian.shape}. Expected square matrix (3N x 3N)."
            )
        results["is_valid"] = False
        return results

    # Check for NaN values
    has_nan = np.any(np.isnan(hessian))
    results["has_nan"] = has_nan
    if has_nan:
        if warn_on_issues:
            logger.warning("Hessian contains NaN values. This indicates a calculation error.")
        results["is_valid"] = False

    # Check for Inf values
    has_inf = np.any(np.isinf(hessian))
    results["has_inf"] = has_inf
    if has_inf:
        if warn_on_issues:
            logger.warning("Hessian contains infinite values. This indicates a calculation error.")
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
    try:
        # Use eigenvalues to compute condition number more robustly
        eigenvalues = np.linalg.eigvals(hessian)
        eigenvalues = eigenvalues[np.isfinite(eigenvalues)]
        if len(eigenvalues) > 0:
            eigenvalues_abs = np.abs(eigenvalues)
            eigenvalues_abs = eigenvalues_abs[eigenvalues_abs > 0]
            if len(eigenvalues_abs) > 0:
                condition_number = np.max(eigenvalues_abs) / np.min(eigenvalues_abs)
                results["condition_number"] = float(condition_number)
                if condition_number > max_condition_number:
                    if warn_on_issues:
                        logger.warning(
                            f"Hessian is ill-conditioned. Condition number: {condition_number:.2e}. "
                            f"Maximum acceptable: {max_condition_number:.2e}. "
                            "This may lead to numerical instability in frequency calculations."
                        )
        else:
            if warn_on_issues:
                logger.warning(
                    "Hessian has no finite eigenvalues. This indicates a severe problem."
                )
            results["is_valid"] = False
    except Exception as e:
        if warn_on_issues:
            logger.warning(f"Could not compute condition number: {e}")

    # Overall validity
    if not is_symmetric or has_nan or has_inf:
        results["is_valid"] = False

    return results
