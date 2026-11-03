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
from scipy.linalg import eigh as scipy_eigh
from scipy.sparse.linalg import LinearOperator, eigsh

HessianVectorProduct = Callable[[np.ndarray], np.ndarray]

# Eigenvalues below this (eV/Å²) are treated as trans/rot null modes, not TS modes.
NULL_MODE_THRESHOLD = 1e-3


@dataclass
class TSStepResult:
    """Result of a single TS optimization step."""

    step: np.ndarray
    predicted_energy_change: float
    transition_mode: np.ndarray
    alpha: float = 1.0


def build_translation_rotation_basis(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Build an orthonormal Cartesian basis for translation and rotation.

    The basis is intended for projecting Cartesian gradients and Hessians.
    Translations are uniform Cartesian displacements; rotations are rigid
    displacements ``r × axis`` about the center of mass.

    Parameters
    ----------
    positions : np.ndarray
        Atomic positions, shape (n_atoms, 3).
    masses : np.ndarray
        Atomic masses, shape (n_atoms,). Used only to locate the center of mass.

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

    modes: list[np.ndarray] = []

    for axis in range(3):
        mode = np.zeros(n_coords)
        mode[axis::3] = 1.0
        norm = np.linalg.norm(mode)
        if norm > 1e-12:
            modes.append(mode / norm)

    if n_atoms >= 2:
        for axis in range(3):
            rot = np.zeros(n_coords)
            axis_vec = np.zeros(3)
            axis_vec[axis] = 1.0
            for i in range(n_atoms):
                disp = np.cross(axis_vec, rel[i])
                rot[3 * i : 3 * i + 3] = disp
            norm = np.linalg.norm(rot)
            if norm > 1e-12:
                modes.append(rot / norm)

    if not modes:
        return np.zeros((n_coords, 0))

    basis = np.column_stack(modes)
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


def _solve_restricted_rfo_2x2(
    gradient_proj: float, omega: float, alpha: float, maximize: bool
) -> float:
    """Step component from the 2×2 RFO problem with metric scale ``alpha``."""
    rfo = np.array([[0.0, gradient_proj], [gradient_proj, omega]])
    metric = np.array([[1.0, 0.0], [0.0, alpha]])
    eigvals, eigvecs = scipy_eigh(rfo, metric)
    idx = int(np.argmax(eigvals) if maximize else np.argmin(eigvals))
    lam = float(eigvals[idx])
    v = eigvecs[:, idx]
    if abs(v[0]) > 1e-12:
        return float(v[1] / v[0])
    denom = omega - alpha * lam
    if abs(denom) <= 1e-12:
        return 0.0
    return float(-gradient_proj / denom)


def _restricted_subspace_lambda(
    eigenvalues: np.ndarray,
    gradient_eigen: np.ndarray,
    tv_index: int,
    alpha: float,
) -> float:
    """Lowest eigenvalue of the scaled augmented Hessian.

    ``alpha`` scales the geometry block. The minimization step is
    ``-g / (ω − α λ)``.
    """
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
    metric = np.eye(n_ot + 1) * alpha
    metric[n_ot, n_ot] = 1.0
    eigvals = scipy_eigh(aug, metric, eigvals_only=True)
    return float(np.min(eigvals))


def _restricted_prfo_step(
    gradient: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    tv_index: int,
    alpha: float,
) -> np.ndarray:
    """RS-P-RFO step. Smaller ``alpha`` lengthens the step."""
    n = len(gradient)
    gradient_eigen = eigenvectors.T @ gradient
    y_eigen = np.zeros(n)
    y_eigen[tv_index] = _solve_restricted_rfo_2x2(
        float(gradient_eigen[tv_index]),
        float(eigenvalues[tv_index]),
        alpha,
        maximize=True,
    )
    lambda_ot = _restricted_subspace_lambda(eigenvalues, gradient_eigen, tv_index, alpha)
    for k in range(n):
        if k == tv_index:
            continue
        denom = eigenvalues[k] - alpha * lambda_ot
        y_eigen[k] = -gradient_eigen[k] / denom if abs(denom) > 1e-12 else 0.0
    return eigenvectors @ y_eigen


def _alpha_for_trust_radius(
    gradient: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    tv_index: int,
    trust_radius: float,
    alpha: float,
    max_iter: int,
) -> float:
    """Find ``alpha`` whose RS-P-RFO step lies near the trust radius."""
    alpha_min = 1e-4
    alpha_max = 1e4
    alpha = float(np.clip(alpha, alpha_min, alpha_max))
    tolerance = 0.05 * trust_radius

    for _ in range(max_iter):
        step = _restricted_prfo_step(gradient, eigenvalues, eigenvectors, tv_index, alpha)
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


def compute_restricted_prfo_step(
    gradient: np.ndarray,
    hessian: np.ndarray,
    trust_radius: float,
    previous_mode: np.ndarray | None = None,
    alpha: float = 1.0,
    max_alpha_iter: int = 20,
) -> TSStepResult:
    """RS-P-RFO step whose length matches ``trust_radius``.

    Plain Baker P-RFO (``alpha`` = 1) leaves stiff modes almost unmoved when the
    gradient is large. The metric scale is chosen so the step reaches the trust
    sphere, then capped if the search still overshoots.
    """
    sym_hessian = 0.5 * (hessian + hessian.T)
    eigenvalues, eigenvectors = np.linalg.eigh(sym_hessian)
    tv_index, transition_mode, _omega_tv = select_transition_mode(
        eigenvalues,
        eigenvectors,
        previous_mode,
    )
    alpha_opt = _alpha_for_trust_radius(
        gradient=gradient,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        tv_index=tv_index,
        trust_radius=trust_radius,
        alpha=alpha,
        max_iter=max_alpha_iter,
    )
    step = _restricted_prfo_step(gradient, eigenvalues, eigenvectors, tv_index, alpha_opt)
    step_norm = float(np.linalg.norm(step))
    if step_norm > 1e-12 and step_norm > trust_radius:
        step = step / step_norm * trust_radius
    predicted = _predict_energy_change(step, gradient, sym_hessian)
    return TSStepResult(
        step=step,
        predicted_energy_change=predicted,
        transition_mode=transition_mode,
        alpha=alpha_opt,
    )


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
    null_threshold: float = NULL_MODE_THRESHOLD,
) -> tuple[float, np.ndarray]:
    """Lowest vibrational Hessian eigenpair, or the previous mode by overlap.

    Requests enough Lanczos pairs to see past rigid-body null modes, then drops
    eigenvalues inside ``null_threshold``. With a previous mode, the vibrational
    pair of largest absolute overlap is kept.
    """
    op = LinearOperator((n, n), matvec=hessp, dtype=float)

    if previous_mode is not None and np.linalg.norm(previous_mode) > 1e-12:
        v0 = previous_mode / np.linalg.norm(previous_mode)
    else:
        v0 = np.random.default_rng(0).standard_normal(n)
        v0 /= np.linalg.norm(v0)

    # Six rigid-body modes plus the softest vibrations. eigsh needs k < n.
    n_pairs = min(7, n - 1) if n > 1 else 1
    eigenvalues, eigenvectors = eigsh(op, k=n_pairs, which="SA", tol=1e-4, maxiter=n_iter, v0=v0)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    vibrational = np.abs(eigenvalues) > null_threshold
    if not np.any(vibrational):
        msg = (
            "Cannot determine transition mode: no vibrational eigenvalues "
            f"above null threshold ({null_threshold} eV/Å²)"
        )
        raise RuntimeError(msg)

    candidates = np.flatnonzero(vibrational)
    if previous_mode is not None and np.linalg.norm(previous_mode) > 1e-12:
        previous = previous_mode / np.linalg.norm(previous_mode)
        overlaps = np.abs(eigenvectors[:, candidates].T @ previous)
        chosen = int(candidates[int(np.argmax(overlaps))])
    else:
        chosen = int(candidates[int(np.argmin(eigenvalues[candidates]))])
    return float(eigenvalues[chosen]), eigenvectors[:, chosen]


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
