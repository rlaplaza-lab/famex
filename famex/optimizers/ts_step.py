"""Shared transition-state step builders for local TS optimizers.

Provides trans/rot projection, mode-following, RS-P-RFO (dense Hessian), and
matrix-free image-Hessian trust-region steps aligned with each optimizer's design.

Default trust radii (0.1 / 0.3 Å) are intentionally larger than geomeTRIC's TS
defaults (0.01 / 0.03 Å) to suit ML potentials and projected Cartesian coordinates.
Override via optimizer kwargs when tighter steps are needed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from ase import Atoms
from scipy.sparse.linalg import LinearOperator, eigsh

from famex.utils.logging import get_famex_logger

logger = get_famex_logger(__name__)

HessianVectorProduct = Callable[[np.ndarray], np.ndarray]

# Eigenvalues below this (eV/Å²) are treated as trans/rot null modes, not TS modes.
NULL_MODE_THRESHOLD = 1e-3


@dataclass
class TSStepResult:
    """Result of a single TS optimization step."""

    step: np.ndarray
    predicted_energy_change: float
    transition_mode: np.ndarray
    transition_eigenvalue: float
    alpha: float = 1.0


def build_translation_rotation_basis(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Build an orthonormal basis for translation and rotation modes in Cartesian space.

    Parameters
    ----------
    positions : np.ndarray
        Atomic positions, shape (n_atoms, 3).
    masses : np.ndarray
        Atomic masses, shape (n_atoms,).

    Returns
    -------
    np.ndarray
        Orthonormal basis matrix of shape (3 * n_atoms, n_modes).
    """
    n_atoms = positions.shape[0]
    n_coords = 3 * n_atoms
    if n_atoms == 0:
        return np.zeros((0, 0))

    total_mass = float(np.sum(masses))
    com = np.sum(positions * masses[:, None], axis=0) / total_mass
    rel = positions - com

    # Translation modes (always 3 for multi-atom; for single atom, only translations exist)
    modes: list[np.ndarray] = []
    sqrt_masses = np.sqrt(masses)

    for axis in range(3):
        mode = np.zeros(n_coords)
        for i in range(n_atoms):
            mode[3 * i + axis] = sqrt_masses[i]
        norm = np.linalg.norm(mode)
        if norm > 1e-12:
            modes.append(mode / norm)

    if n_atoms >= 2:
        # Rotation modes about COM (mass-weighted)
        for axis in range(3):
            rot = np.zeros(n_coords)
            axis_vec = np.zeros(3)
            axis_vec[axis] = 1.0
            for i in range(n_atoms):
                disp = np.cross(axis_vec, rel[i])
                rot[3 * i : 3 * i + 3] = sqrt_masses[i] * disp
            norm = np.linalg.norm(rot)
            if norm > 1e-12:
                modes.append(rot / norm)

    if not modes:
        return np.zeros((n_coords, 0))

    basis = np.column_stack(modes)
    # Orthonormalize (Gram-Schmidt) for numerical stability
    q, _ = np.linalg.qr(basis)
    return q


def project_gradient(gradient: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project gradient onto subspace orthogonal to translation/rotation."""
    if basis.size == 0:
        return gradient
    return cast(np.ndarray, gradient - basis @ (basis.T @ gradient))


def project_hessian(hessian: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project Hessian onto subspace orthogonal to translation/rotation."""
    if basis.size == 0:
        return hessian
    projector = np.eye(hessian.shape[0]) - basis @ basis.T
    return cast(np.ndarray, projector @ hessian @ projector)


def select_transition_mode(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    previous_mode: np.ndarray | None,
    null_threshold: float = NULL_MODE_THRESHOLD,
) -> tuple[int, np.ndarray, float]:
    """Select the transition mode with optional mode-following."""
    if previous_mode is None:
        vibrational = np.abs(eigenvalues) > null_threshold
        if np.any(vibrational):
            masked = np.where(vibrational, eigenvalues, np.inf)
            tv_index = int(np.argmin(masked))
        else:
            msg = (
                "Cannot determine transition mode: no vibrational eigenvalues "
                f"above null threshold ({null_threshold} eV/Å²)"
            )
            raise RuntimeError(msg)
    else:
        overlaps = np.abs(eigenvectors.T @ previous_mode)
        tv_index = int(np.argmax(overlaps))

    mode = eigenvectors[:, tv_index].copy()
    norm = np.linalg.norm(mode)
    if norm < 1e-12:
        msg = "Cannot determine transition mode: zero eigenvector"
        raise RuntimeError(msg)
    mode /= norm
    return tv_index, mode, float(eigenvalues[tv_index])


def _solve_rfo_2x2(
    gradient_proj: float, omega: float, alpha: float, maximize: bool
) -> tuple[float, float]:
    """Solve 2x2 RFO for a single mode. Returns (step_component, eigenvalue)."""
    rfo = np.array([[0.0, gradient_proj], [gradient_proj, omega]])
    metric = np.array([[1.0, 0.0], [0.0, alpha]])
    from scipy.linalg import eigh as scipy_eigh

    eigvals, eigvecs = scipy_eigh(rfo, metric)
    idx = int(np.argmax(eigvals) if maximize else np.argmin(eigvals))
    lam = float(eigvals[idx])
    v = eigvecs[:, idx]
    if abs(v[0]) > 1e-12:
        step_comp = float(v[1] / v[0])
    else:
        denom = omega - alpha * lam
        step_comp = -gradient_proj / denom if abs(denom) > 1e-12 else 0.0
    return step_comp, lam


def _solve_other_subspace_lambda(
    eigenvalues: np.ndarray,
    gradient_eigen: np.ndarray,
    tv_index: int,
    alpha: float,
) -> float:
    """Solve the augmented RFO problem for the minimization subspace."""
    mask = np.ones(len(eigenvalues), dtype=bool)
    mask[tv_index] = False
    omega_ot = eigenvalues[mask]
    g_ot = gradient_eigen[mask]
    n_ot = len(omega_ot)

    if n_ot == 0:
        return 0.0

    aug = np.zeros((n_ot + 1, n_ot + 1))
    aug[:n_ot, :n_ot] = np.diag(omega_ot)
    aug[:n_ot, n_ot] = g_ot
    aug[n_ot, :n_ot] = g_ot

    metric = np.eye(n_ot + 1)
    metric[n_ot, n_ot] = alpha

    from scipy.linalg import eigh as scipy_eigh

    eigvals, _ = scipy_eigh(aug, metric)
    return float(np.min(eigvals))


def compute_dense_prfo_step(
    gradient: np.ndarray,
    hessian: np.ndarray,
    trust_radius: float,
    previous_mode: np.ndarray | None = None,
    alpha: float = 1.0,
    max_alpha_iter: int = 20,
) -> TSStepResult:
    """Compute a dense RS-P-RFO step for transition-state optimization."""
    sym_hessian = 0.5 * (hessian + hessian.T)
    eigenvalues, eigenvectors = np.linalg.eigh(sym_hessian)

    tv_index, transition_mode, omega_tv = select_transition_mode(
        eigenvalues,
        eigenvectors,
        previous_mode,
    )

    alpha_opt = _find_alpha_for_trust_radius(
        gradient=gradient,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        tv_index=tv_index,
        trust_radius=trust_radius,
        alpha=alpha,
        max_iter=max_alpha_iter,
    )

    step, predicted = _prfo_step_from_eigensystem(
        gradient=gradient,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        tv_index=tv_index,
        alpha=alpha_opt,
        scale_to_radius=False,
    )

    step_norm = float(np.linalg.norm(step))
    if step_norm > 1e-12 and step_norm > trust_radius:
        step = step / step_norm * trust_radius

    predicted = _predict_energy_change(step, gradient, sym_hessian)

    return TSStepResult(
        step=step,
        predicted_energy_change=predicted,
        transition_mode=transition_mode,
        transition_eigenvalue=omega_tv,
        alpha=alpha_opt,
    )


def _prfo_step_from_eigensystem(
    gradient: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    tv_index: int,
    alpha: float,
    scale_to_radius: bool,
    trust_radius: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Build RS-P-RFO step from Hessian eigensystem."""
    n = len(gradient)
    gradient_eigen = eigenvectors.T @ gradient

    g_tv = float(gradient_eigen[tv_index])
    omega_tv = float(eigenvalues[tv_index])
    y_tv, _ = _solve_rfo_2x2(g_tv, omega_tv, alpha, maximize=True)

    lambda_ot = _solve_other_subspace_lambda(eigenvalues, gradient_eigen, tv_index, alpha)

    y_eigen = np.zeros(n)
    for k in range(n):
        if k == tv_index:
            y_eigen[k] = y_tv
        else:
            denom = eigenvalues[k] - alpha * lambda_ot
            y_eigen[k] = -gradient_eigen[k] / denom if abs(denom) > 1e-12 else 0.0

    step = eigenvectors @ y_eigen
    if scale_to_radius:
        step_norm = float(np.linalg.norm(step))
        if step_norm > 1e-12 and step_norm > trust_radius:
            step = step / step_norm * trust_radius

    predicted = _predict_energy_change(
        step, gradient, eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    )
    return step, predicted


def _find_alpha_for_trust_radius(
    gradient: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    tv_index: int,
    trust_radius: float,
    alpha: float,
    max_iter: int,
) -> float:
    """Bisection search for alpha that yields a step near the trust radius."""
    alpha_min = 1e-4
    alpha_max = 1e4
    alpha = float(np.clip(alpha, alpha_min, alpha_max))
    tolerance = 0.05 * trust_radius

    for _ in range(max_iter):
        step, _ = _prfo_step_from_eigensystem(
            gradient=gradient,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            tv_index=tv_index,
            alpha=alpha,
            scale_to_radius=False,
        )
        step_norm = float(np.linalg.norm(step))
        if abs(step_norm - trust_radius) < tolerance:
            break
        if step_norm > trust_radius:
            alpha_min = alpha
            alpha = (alpha + alpha_max) / 2.0 if alpha_max < 1e4 else alpha * 1.5
        else:
            alpha_max = alpha
            alpha = (alpha_min + alpha) / 2.0 if alpha_min > 1e-4 else alpha * 0.8
        alpha = float(np.clip(alpha, alpha_min, alpha_max))
        if alpha_min >= alpha_max - 1e-10:
            break

    return alpha


def _predict_energy_change(step: np.ndarray, gradient: np.ndarray, hessian: np.ndarray) -> float:
    """Quadratic model energy change: g^T s + 0.5 s^T H s."""
    return float(np.dot(step, gradient) + 0.5 * np.dot(step, hessian @ step))


def compute_step_quality(actual_energy_change: float, predicted_energy_change: float) -> float:
    """TS step quality factor Q in [-inf, 1]."""
    if abs(predicted_energy_change) < 1e-12:
        if abs(actual_energy_change) < 1e-12:
            return 1.0
        return -abs(actual_energy_change)
    ratio = actual_energy_change / predicted_energy_change
    return float(1.0 - abs(ratio - 1.0))


def adjust_trust_radius(
    trust_radius: float,
    step_quality: float,
    step_size: float,
    min_trust_radius: float,
    max_trust_radius: float,
) -> float:
    """Update trust radius from step quality (geomeTRIC-style)."""
    if step_quality >= 0.75:
        return float(min(trust_radius * np.sqrt(2.0), max_trust_radius))
    if step_quality >= 0.50:
        return trust_radius
    new_trust = 0.5 * min(trust_radius, step_size)
    return max(new_trust, min_trust_radius)


def lowest_eigenpair(
    hessp: HessianVectorProduct,
    n: int,
    previous_mode: np.ndarray | None = None,
    n_iter: int = 30,
) -> tuple[float, np.ndarray]:
    """Find the lowest Hessian eigenpair via matrix-free Lanczos (eigsh)."""
    op = LinearOperator((n, n), matvec=hessp, dtype=float)

    if previous_mode is not None and np.linalg.norm(previous_mode) > 1e-12:
        v0 = previous_mode / np.linalg.norm(previous_mode)
    else:
        v0 = np.random.default_rng(0).standard_normal(n)
        v0 /= np.linalg.norm(v0)

    try:
        eigenvalues, eigenvectors = eigsh(op, k=1, which="SA", tol=1e-4, maxiter=n_iter, v0=v0)
    except Exception:
        # Fallback: power iteration on inverse-shifted operator is expensive;
        # use a few steps of Rayleigh quotient iteration with diagonal estimate.
        vec = v0
        lam = float(np.dot(vec, hessp(vec)))
        for _ in range(max(n_iter, 10)):
            vec = hessp(vec)
            norm = np.linalg.norm(vec)
            if norm < 1e-12:
                break
            vec /= norm
            lam = float(np.dot(vec, hessp(vec)))
        return lam, vec

    return float(eigenvalues[0]), eigenvectors[:, 0]


def build_image_operators(
    hessp: HessianVectorProduct,
    gradient: np.ndarray,
    lambda_min: float,
    mode: np.ndarray,
) -> tuple[np.ndarray, HessianVectorProduct]:
    """Build image gradient and image Hessian-vector product for TS search."""
    mode = mode / max(np.linalg.norm(mode), 1e-12)
    g_dot = float(np.dot(gradient, mode))
    image_gradient = gradient - 2.0 * g_dot * mode

    def image_hessp(vec: np.ndarray) -> np.ndarray:
        hv = hessp(vec)
        v_dot = float(np.dot(vec, mode))
        return hv - 2.0 * lambda_min * v_dot * mode

    return image_gradient, image_hessp


def _get_trlib_subproblem_class(subproblem: str) -> Any:
    """Return SciPy trlib quadratic subproblem class for matrix-free TR solves."""
    try:
        from scipy.optimize._trlib import get_trlib_quadratic_subproblem
    except ImportError as exc:
        msg = (
            "SciPy trust-region subproblem support is unavailable. "
            "Install scipy>=1.9 for matrix-free TS optimizers."
        )
        raise ImportError(msg) from exc

    # Krylov subproblem uses relaxed inner tolerances (SciPy trust-krylov convention).
    if subproblem == "krylov":
        return get_trlib_quadratic_subproblem(tol_rel_i=-2.0, tol_rel_b=-3.0)
    return get_trlib_quadratic_subproblem(tol_rel_i=1e-8, tol_rel_b=1e-6)


def solve_trust_region_subproblem(
    gradient: np.ndarray,
    hessp: HessianVectorProduct,
    trust_radius: float,
    subproblem: str = "krylov",
) -> np.ndarray:
    """Solve a matrix-free trust-region subproblem for a quadratic model."""
    n = len(gradient)
    x0 = np.zeros(n)

    def fun(_x: np.ndarray) -> float:
        return 0.0

    def jac(_x: np.ndarray) -> np.ndarray:
        return gradient

    def hessp_fn(_x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return hessp(p)

    Subproblem = _get_trlib_subproblem_class(subproblem)
    sub = Subproblem(x0, fun, jac, None, hessp_fn)
    step, _on_boundary = sub.solve(trust_radius)
    return np.asarray(step)


def compute_matrix_free_ts_step(
    gradient: np.ndarray,
    hessp: HessianVectorProduct,
    trust_radius: float,
    previous_mode: np.ndarray | None = None,
    subproblem: str = "krylov",
) -> TSStepResult:
    """Matrix-free TS step via lowest eigenpair + image-Hessian trust-region solve."""
    n = len(gradient)
    lambda_min, mode = lowest_eigenpair(hessp, n, previous_mode=previous_mode)
    mode_norm = np.linalg.norm(mode)
    if mode_norm > 1e-12:
        mode = mode / mode_norm

    image_gradient, image_hessp = build_image_operators(hessp, gradient, lambda_min, mode)
    step = solve_trust_region_subproblem(
        image_gradient,
        hessp=image_hessp,
        trust_radius=trust_radius,
        subproblem=subproblem,
    )

    # Predicted change uses the real (projected) quadratic model, not the image model.
    h_step = hessp(step)
    predicted = float(np.dot(step, gradient) + 0.5 * np.dot(step, h_step))

    return TSStepResult(
        step=step,
        predicted_energy_change=predicted,
        transition_mode=mode,
        transition_eigenvalue=lambda_min,
    )


def prepare_ts_hessian(
    hessian: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    """Project a dense Hessian onto the vibrational subspace."""
    return project_hessian(0.5 * (hessian + hessian.T), basis)


def make_atoms_hessp(
    atoms: Atoms,
    x: np.ndarray,
    x_to_positions: Callable[[np.ndarray], np.ndarray],
    delta: float = 0.01,
    basis: np.ndarray | None = None,
) -> HessianVectorProduct:
    """Finite-difference Hessian-vector product from forces."""
    cached_x: np.ndarray | None = None
    cached_forces0: np.ndarray | None = None

    def _project(vec: np.ndarray) -> np.ndarray:
        if basis is not None and basis.size > 0:
            return cast(np.ndarray, vec - basis @ (basis.T @ vec))
        return vec

    def _ensure_base_forces() -> np.ndarray:
        nonlocal cached_x, cached_forces0
        if cached_x is None or not np.array_equal(cached_x, x):
            atoms.set_positions(x_to_positions(x))
            cached_forces0 = atoms.get_forces().ravel()
            cached_x = x.copy()
        assert cached_forces0 is not None
        return cached_forces0

    def hessp(vec: np.ndarray) -> np.ndarray:
        vec = _project(vec)
        forces0 = _ensure_base_forces()
        step = delta * vec
        atoms.set_positions(x_to_positions(x + step))
        forces1 = atoms.get_forces().ravel()
        atoms.set_positions(x_to_positions(x))
        result = -(forces1 - forces0) / delta
        return _project(result)

    return hessp
