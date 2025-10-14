"""Rational Function Optimization (RFO) and Partitioned RFO (P-RFO) algorithms.

This module implements RFO and P-RFO algorithms for transition state optimization,
inspired by the pysisyphus implementation.
"""

import numpy as np
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


def get_augmented_hessian(
    eigenvals: np.ndarray, 
    gradient: np.ndarray, 
    alpha: float
) -> np.ndarray:
    """Build augmented Hessian matrix for RFO.
    
    The augmented Hessian has the form:
    H_aug = [ H/α    g/α ]
            [ g^T      0  ]
    
    Parameters
    ----------
    eigenvals : np.ndarray
        Eigenvalues of the Hessian matrix
    gradient : np.ndarray
        Gradient vector
    alpha : float
        Step scaling parameter
        
    Returns
    -------
    np.ndarray
        Augmented Hessian matrix (n+1 × n+1)
    """
    n = len(eigenvals)
    
    # Build augmented Hessian
    H_aug = np.zeros((n + 1, n + 1))
    
    # Diagonal block: H/α
    H_aug[:n, :n] = np.diag(eigenvals / alpha)
    
    # Off-diagonal blocks: g and g^T
    # The augmented Hessian must be symmetric
    H_aug[:n, n] = gradient  # g
    H_aug[n, :n] = gradient  # g^T (not g^T/α!)
    
    # Bottom-right element is 0
    H_aug[n, n] = 0.0
    
    return H_aug


def solve_rfo(
    H_aug: np.ndarray, 
    mode: str = 'min',
    prev_eigvec: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """Solve RFO eigenvalue problem.
    
    Parameters
    ----------
    H_aug : np.ndarray
        Augmented Hessian matrix
    mode : str, default 'min'
        'min' for minimization, 'max' for maximization
    prev_eigvec : np.ndarray, optional
        Previous eigenvector for continuity
        
    Returns
    -------
    Tuple[np.ndarray, float, float, np.ndarray]
        (step, eigenvalue, nu, eigenvector)
    """
    try:
        # Solve eigenvalue problem
        eigenvals, eigenvecs = np.linalg.eigh(H_aug)
        
        # Find appropriate eigenvalue based on mode (matching pysisyphus)
        sorted_inds = np.argsort(eigenvals)
        
        if mode == 'min':
            # For minimization, use smallest (most negative) eigenvalue
            naive_idx = sorted_inds[0]
        else:  # mode == 'max'
            # For maximization, use largest (most positive) eigenvalue  
            naive_idx = sorted_inds[-1]
        
        # Ensure eigenvector continuity using overlap (matching pysisyphus)
        if prev_eigvec is not None:
            overlaps = np.array([np.abs(np.dot(prev_eigvec, ev)) for ev in eigenvecs.T])
            target_idx = np.argmax(overlaps)
        else:
            target_idx = naive_idx
        
        target_eigval = eigenvals[target_idx]
        target_eigvec = eigenvecs[:, target_idx]
        
        # Extract step and nu
        step = target_eigvec[:-1] / target_eigvec[-1]
        nu = target_eigvec[-1]
        
        return step, target_eigval, nu, target_eigvec
        
    except np.linalg.LinAlgError as e:
        logger.warning(f"RFO eigenvalue solution failed: {e}")
        # Fallback: return zero step
        n = H_aug.shape[0] - 1
        return np.zeros(n), 0.0, 1.0, np.zeros(n + 1)


def rfo_model(gradient: np.ndarray, hessian: np.ndarray, step: np.ndarray) -> float:
    """Calculate RFO model energy change.
    
    The RFO model predicts the energy change as:
    ΔE = g^T * s + 0.5 * s^T * H * s
    
    Parameters
    ----------
    gradient : np.ndarray
        Gradient vector
    hessian : np.ndarray
        Hessian matrix
    step : np.ndarray
        Step vector
        
    Returns
    -------
    float
        Predicted energy change
    """
    linear_term = np.dot(gradient, step)
    quadratic_term = 0.5 * np.dot(step, hessian @ step)
    return linear_term + quadratic_term


def restricted_step_microcycles(
    eigenvals: np.ndarray,
    eigenvecs: np.ndarray,
    gradient: np.ndarray,
    trust_radius: float,
    max_micro_cycles: int = 25,
    alpha0: float = 1.0,
    min_indices: Optional[List[int]] = None,
    max_indices: Optional[List[int]] = None
) -> Tuple[np.ndarray, List[float]]:
    """Perform restricted-step RFO micro-cycles (simplified for testing).
    
    This is a simplified version that ensures the trust radius constraint
    is satisfied by scaling the final step if necessary.
    
    Parameters
    ----------
    eigenvals : np.ndarray
        Eigenvalues of the Hessian
    eigenvecs : np.ndarray
        Eigenvectors of the Hessian
    gradient : np.ndarray
        Gradient vector
    trust_radius : float
        Trust radius constraint
    max_micro_cycles : int, default 25
        Maximum number of micro-cycles (unused in simplified version)
    alpha0 : float, default 1.0
        Initial alpha value
    min_indices : List[int], optional
        Indices for minimization subspace (unused)
    max_indices : List[int], optional
        Indices for maximization subspace (unused)
        
    Returns
    -------
    Tuple[np.ndarray, List[float]]
        (final_step, list_of_alpha_values)
    """
    n = len(eigenvals)
    
    # Transform gradient to eigensystem
    gradient_trans = eigenvecs.T @ gradient
    
    # Filter out small gradient components
    small_eigval_thresh = 1e-6
    mask = np.abs(gradient_trans) > small_eigval_thresh
    gradient_filtered = gradient_trans[mask]
    eigenvals_filtered = eigenvals[mask]
    eigenvecs_filtered = eigenvecs[:, mask]
    
    alpha = alpha0
    alpha_history = [alpha]
    
    # Simple RFO step with alpha adjustment
    for mu in range(max_micro_cycles):
        logger.debug(f"RS-RFO micro cycle {mu:02d}, alpha={alpha:.6f}")
        
        # Single RFO step
        H_aug = get_augmented_hessian(eigenvals_filtered, gradient_filtered, alpha)
        rfo_step_trans, eigval_min, nu, _ = solve_rfo(H_aug, "min")
        rfo_norm = np.linalg.norm(rfo_step_trans)
        logger.debug(f"norm(rfo step)={rfo_norm:.6f}")
        
        # Check if step satisfies trust radius
        if rfo_norm <= trust_radius * 1.01:  # Allow small tolerance
            logger.debug(f"Trust radius satisfied: {rfo_norm:.6f} <= {trust_radius:.6f}")
            break
        
        # Simple alpha update: increase alpha to reduce step size
        alpha *= 1.5
        alpha_history.append(alpha)
        
        # Safety check
        if alpha > 1e6:
            logger.warning("Alpha became too large, breaking micro-cycles")
            break
    
    # If still too large, scale to trust radius
    if rfo_norm > trust_radius * 1.01:
        logger.debug(f"Scaling step to trust radius: {rfo_norm:.6f} -> {trust_radius:.6f}")
        rfo_step_trans = rfo_step_trans / rfo_norm * trust_radius
    
    # Transform step back to original coordinate system
    final_step = eigenvecs_filtered @ rfo_step_trans
    
    return final_step, alpha_history


def _get_alpha_step(cur_alpha: float, rfo_eigval: float, step_norm: float, 
                   eigenvals: np.ndarray, gradient: np.ndarray, trust_radius: float) -> float:
    """Calculate alpha step using pysisyphus formula.
    
    Parameters
    ----------
    cur_alpha : float
        Current alpha value
    rfo_eigval : float
        RFO eigenvalue
    step_norm : float
        Current step norm
    eigenvals : np.ndarray
        Filtered eigenvalues
    gradient : np.ndarray
        Filtered gradient
    trust_radius : float
        Trust radius constraint
        
    Returns
    -------
    float
        Alpha step size
    """
    # Derivative of the squared step w.r.t. alpha (pysisyphus formula)
    numer = gradient**2
    denom = (eigenvals - rfo_eigval * cur_alpha) ** 3
    quot = np.sum(numer / denom)
    dstep2_dalpha = 2 * rfo_eigval / (1 + step_norm**2 * cur_alpha) * quot
    
    # Update alpha
    alpha_step = 2 * (trust_radius * step_norm - step_norm**2) / dstep2_dalpha
    assert (cur_alpha + alpha_step) > 0, "alpha must not be negative!"
    return alpha_step




def calculate_ts_mode_indices(eigenvals: np.ndarray, n_negative: int = 1) -> List[int]:
    """Calculate indices of transition state modes.
    
    Parameters
    ----------
    eigenvals : np.ndarray
        Eigenvalues of the Hessian
    n_negative : int, default 1
        Number of negative eigenvalues to treat as TS modes
        
    Returns
    -------
    List[int]
        Indices of TS modes (negative eigenvalues)
    """
    # Find indices of negative eigenvalues
    negative_indices = np.where(eigenvals < 0)[0]
    
    # Sort by eigenvalue (most negative first)
    if len(negative_indices) > 0:
        sorted_indices = negative_indices[np.argsort(eigenvals[negative_indices])]
        return sorted_indices[:n_negative].tolist()
    
    return []


def calculate_min_mode_indices(
    eigenvals: np.ndarray, 
    ts_indices: List[int],
    small_eigval_thresh: float = 1e-6
) -> List[int]:
    """Calculate indices of minimization modes.
    
    Parameters
    ----------
    eigenvals : np.ndarray
        Eigenvalues of the Hessian
    ts_indices : List[int]
        Indices of TS modes
    small_eigval_thresh : float, default 1e-6
        Threshold for small eigenvalues
        
    Returns
    -------
    List[int]
        Indices of minimization modes (positive eigenvalues, excluding TS modes)
    """
    # Find all indices except TS modes
    all_indices = set(range(len(eigenvals)))
    ts_indices_set = set(ts_indices)
    candidate_indices = all_indices - ts_indices_set
    
    # Filter by eigenvalue magnitude
    min_indices = []
    for idx in candidate_indices:
        if abs(eigenvals[idx]) > small_eigval_thresh:
            min_indices.append(idx)
    
    return min_indices


def validate_rfo_step(
    step: np.ndarray,
    gradient: np.ndarray,
    hessian: np.ndarray,
    trust_radius: float
) -> Tuple[bool, List[str]]:
    """Validate RFO step.
    
    Parameters
    ----------
    step : np.ndarray
        Proposed step
    gradient : np.ndarray
        Gradient vector
    hessian : np.ndarray
        Hessian matrix
    trust_radius : float
        Trust radius
        
    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_warnings)
    """
    warnings = []
    
    # Check step norm
    step_norm = np.linalg.norm(step)
    if step_norm > trust_radius * 1.01:  # Allow small numerical errors
        warnings.append(f"Step norm {step_norm:.6f} exceeds trust radius {trust_radius:.6f}")
    
    # Check for NaN or infinite values
    if not np.all(np.isfinite(step)):
        warnings.append("Step contains NaN or infinite values")
    
    # Check energy change prediction
    try:
        energy_change = rfo_model(gradient, hessian, step)
        if not np.isfinite(energy_change):
            warnings.append("Predicted energy change is not finite")
    except:
        warnings.append("Could not calculate energy change prediction")
    
    is_valid = len(warnings) == 0
    return is_valid, warnings
