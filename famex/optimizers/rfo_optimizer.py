"""RFO-based transition state optimizer for FAMEX.

This module implements a Restricted-Step Partitioned Rational Function Optimization (RFO)
transition state optimizer following geomeTRIC's approach. The RFO method is specifically
designed for locating transition states by maximizing along the lowest eigenvalue mode of
the Hessian while minimizing in all other directions.

The implementation uses Cartesian coordinates and leverages FAMEX's efficient Hessian
calculation infrastructure.
"""

from __future__ import annotations

from typing import IO, Any, cast

import numpy as np
from ase import Atoms
from ase.optimize.optimize import Optimizer

from famex.utils.logging import get_famex_logger

logger = get_famex_logger(__name__)


class ConvergedError(Exception):
    """Exception raised when optimizer has converged."""


class RFOTransitionState(Optimizer):
    """Restricted-Step Partitioned Rational Function Optimization for transition states.

    This optimizer implements the RFO method as described in geomeTRIC for locating
    transition states. It solves a generalized eigenvalue problem to find the optimal
    step direction, with one mode treated as maximization (the transition mode) and all
    other modes as minimization.

    Parameters
    ----------
    atoms : Atoms
        The Atoms object to optimize.
    logfile : IO | str
        File object or filename for logging. Use '-' for stdout.
    trajectory : Optional[str]
        Trajectory file to store optimization path.
    hessian_update_freq : Optional[int]
        Optional frequency of full Hessian recalculation (in steps).
        Default is None, which only computes the Hessian once at the beginning.
        Set to an integer to recompute every N steps.
    hessian_method : str
        Method for Hessian calculation: 'auto', 'batch', 'finite_differences', 'direct'
        Default is 'auto' which selects the best available method.
    hessian_delta : float
        Step size for finite difference Hessian calculation (Å). Default is 0.01.
    initial_hessian : Optional[np.ndarray]
        Initial Hessian matrix. If None, computed on first step.
    trust_radius : float
        Initial trust radius (Å). Default is 0.01 (geomeTRIC default for TS).
    max_trust_radius : float
        Maximum trust radius (Å). Default is 0.03 (geomeTRIC default for TS).
    min_trust_radius : float
        Minimum trust radius (Å). Default is 0.001.
    alpha : float
        Initial scaling factor for trust radius control. Default is 1.0.
        Dynamically adjusted during optimization.
    verbose : int
        Verbosity level for optimization output:
        - 0: Quiet (minimal output)
        - 1: Normal (default, shows progress)
        - 2: Verbose (detailed information)
    **kwargs
        Additional arguments passed to Optimizer base class.

    Attributes
    ----------
    freq_analysis : FrequencyAnalysis
        FrequencyAnalysis instance for Hessian computation.
    hessian : Optional[np.ndarray]
        Current Hessian matrix.
    force_calls : int
        Number of force/gradient evaluations.
    hessian_calls : int
        Number of Hessian evaluations.

    References
    ----------
    .. [1] geomeTRIC documentation: https://geometric.readthedocs.io/en/latest/transition.html
    .. [2] Banerjee et al., J. Chem. Phys. 63, 3214 (1975)
    .. [3] Peng et al., J. Chem. Phys. 105, 11042 (1996)

    """

    def __init__(
        self,
        atoms: Atoms,
        logfile: IO | str | None = "-",
        trajectory: str | None = None,
        hessian_update_freq: int | None = None,
        hessian_method: str = "auto",
        hessian_delta: float = 0.01,
        initial_hessian: np.ndarray | None = None,
        trust_radius: float = 0.01,
        max_trust_radius: float = 0.03,
        min_trust_radius: float = 0.001,
        alpha: float = 1.0,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize RFO transition state optimizer."""
        # Store verbosity level
        self.verbose = verbose

        # Set up logging based on verbosity
        if verbose == 0:
            # Quiet mode: suppress ASE logging by using None logfile
            logfile = None
        elif verbose == 1:
            # Normal mode: use provided logfile or default
            pass
        else:
            # Verbose mode: use provided logfile or default
            pass

        # Don't use restart for RFO optimizer
        restart = None
        # Filter out profiler argument as ASE Optimizer doesn't accept it
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "profiler"}
        Optimizer.__init__(self, atoms, restart, logfile, trajectory, **filtered_kwargs)

        # Hessian update settings
        freq = hessian_update_freq
        if freq is not None and freq <= 0:
            logger.warning("hessian_update_freq <= 0 provided; disabling periodic Hessian updates")
            freq = None

        self.hessian_update_freq: int | None = freq
        self.hessian_method = hessian_method

        # Trust radius settings (geomeTRIC defaults for TS)
        self.trust_radius = trust_radius
        self.max_trust_radius = max_trust_radius
        self.min_trust_radius = min_trust_radius
        self.alpha = alpha  # Will be adjusted dynamically

        # Initialize FrequencyAnalysis for Hessian computation
        if atoms.calc is None:
            msg = "Atoms object must have a calculator attached"
            raise ValueError(msg)

        # Import locally to avoid circular imports
        from famex.analysis.frequency import FrequencyAnalysis

        self.freq_analysis = FrequencyAnalysis(
            atoms=atoms,
            calculator=atoms.calc,
            delta=hessian_delta,
            verbose=verbose,
        )

        # Hessian state
        self.hessian = initial_hessian
        self.force_calls = 0
        self.hessian_calls = 0
        self._last_hessian_step = -1
        self._last_full_hessian_step = -1

        # State for Hessian updates
        self._last_positions: np.ndarray | None = None
        self._last_gradient: np.ndarray | None = None
        self._last_hessian: np.ndarray | None = None

        # State for step quality tracking
        self._previous_energy: float | None = None
        self._previous_positions: np.ndarray | None = None
        self._step_quality_history: list[float] = []

        # Track transition mode
        self._transition_mode: np.ndarray | None = None
        self._transition_mode_eigenvalue: float | None = None

        # Initialize fmax (inherited from Optimizer but may not be typed)
        # Use setattr to avoid type checking issues with inherited attributes
        if not hasattr(self, "fmax"):
            self.fmax = 0.05
        # Type annotation for mypy - getattr returns Any, but we know it's float here
        # This is needed because ASE's Optimizer base class doesn't type fmax
        self.fmax: float = getattr(self, "fmax", 0.05)  # type: ignore[assignment]
        self.max_steps: int = 0

        if self.verbose >= 2:
            logger.info("Initialized RFO transition state optimizer")
            if hessian_update_freq is None:
                logger.info("Periodic Hessian updates disabled (compute once and reuse)")
            else:
                logger.info(f"Fixed Hessian update frequency: every {hessian_update_freq} step(s)")
            logger.info(f"Hessian calculation method: {hessian_method}")
            logger.info(f"Initial trust radius: {trust_radius:.4f} Å")

    def _positions_to_x(self, atoms: Atoms | None = None) -> np.ndarray:
        """Convert atoms positions to 1D array."""
        if atoms is None:
            atoms = self.atoms
        if atoms is None:  # type: ignore[unreachable]
            raise RuntimeError("Atoms object is not initialized")
        return cast(np.ndarray, atoms.get_positions().ravel())

    def _x_to_positions(self, x: np.ndarray) -> np.ndarray:
        """Convert 1D array to positions array."""
        return cast(np.ndarray, x.reshape(-1, 3))

    def _get_gradient(self, x: np.ndarray) -> np.ndarray:
        """Get gradient (negative forces) at position x."""
        self.atoms.set_positions(self._x_to_positions(x))
        self.force_calls += 1

        # Forces are negative gradient
        forces = self.atoms.get_forces()
        gradient = -forces.ravel()

        return gradient  # type: ignore[no-any-return]

    def _update_hessian_ms_psb(
        self, current_positions: np.ndarray, current_gradient: np.ndarray
    ) -> None:
        """Update Hessian using MS-PSB hybrid method.

        This should be called after taking a step, when we have both the previous
        and current positions and gradients.

        Parameters
        ----------
        current_positions : np.ndarray
            Current position vector
        current_gradient : np.ndarray
            Current gradient vector

        """
        if (
            self._last_positions is None
            or self._last_gradient is None
            or self._last_hessian is None
            or self.hessian is None
        ):
            return

        # Compute differences
        delta = current_positions - self._last_positions  # position change
        xi = current_gradient - self._last_gradient  # gradient change

        # Murtagh-Sargent (MS) update
        delta_xi = np.dot(delta, xi)
        if abs(delta_xi) > 1e-10:
            hessian_ms = self._last_hessian + np.outer(xi, xi) / delta_xi
        else:
            hessian_ms = self._last_hessian

        # Powell symmetric Broyden (PSB) update
        delta_norm_sq = np.dot(delta, delta)
        if delta_norm_sq > 1e-10:
            delta_xi_norm_sq = delta_xi**2
            delta_norm_fourth = delta_norm_sq**2

            correction = (
                -delta_xi_norm_sq * np.outer(delta, delta) / delta_norm_fourth
                + (np.outer(delta, xi) + np.outer(xi, delta)) / delta_norm_sq
            )
            hessian_psb = self._last_hessian + correction
        else:
            hessian_psb = self._last_hessian

        # Hybrid: phi measures alignment between delta and xi
        if delta_norm_sq > 1e-10 and np.linalg.norm(xi) > 1e-10:
            phi = 1.0 - delta_xi_norm_sq / (delta_norm_sq * np.dot(xi, xi))
        else:
            phi = 0.0

        # Mix MS and PSB updates
        self.hessian = (1.0 - phi) * hessian_ms + phi * hessian_psb
        self._last_hessian = self.hessian.copy()

        if self.verbose >= 2:
            logger.debug(f"MS-PSB Hessian update at step {self.nsteps} (phi={phi:.4f})")

    def _compute_hessian(self, x: np.ndarray) -> np.ndarray:
        """Compute or update Hessian matrix."""
        steps_since_full = self.nsteps - self._last_full_hessian_step

        # Determine if we need a full Hessian update
        need_full_update = False

        if self.hessian is None:
            # Always compute on first step
            need_full_update = True
        elif self.hessian_update_freq is not None and steps_since_full >= self.hessian_update_freq:
            need_full_update = True

        if need_full_update:
            # Compute full Hessian
            self.atoms.set_positions(self._x_to_positions(x))
            self.hessian_calls += 1
            self._last_hessian_step = self.nsteps
            self._last_full_hessian_step = self.nsteps

            if self.verbose >= 1:
                logger.info(
                    f"Computing full Hessian at step {self.nsteps} (call #{self.hessian_calls})",
                )

            # Update FrequencyAnalysis with current atoms state
            self.freq_analysis.atoms = self.atoms
            self.freq_analysis.atoms.calc = self.atoms.calc

            # Compute Hessian using FAMEX's infrastructure
            hessian = self.freq_analysis.calculate_hessian(method=self.hessian_method)

            # Store for potential reuse
            self.hessian = hessian
            self._last_hessian = hessian.copy()

            if self.verbose >= 2:
                logger.info(f"Full Hessian computed (shape: {hessian.shape})")

            # Reset update state after full Hessian
            self._last_positions = x.copy()
            self._last_gradient = None

        else:
            # Reuse cached Hessian
            if self.verbose >= 1:
                logger.info(f"Reusing Hessian (computed {steps_since_full} steps ago)")

        if self.hessian is None:
            raise RuntimeError("Hessian is None after update logic")

        return self.hessian

    def _find_transition_mode(self, hessian: np.ndarray) -> tuple[np.ndarray, float]:
        """Find the transition mode (lowest eigenvalue eigenvector) from Hessian."""
        # Symmetrize Hessian
        sym_hessian = 0.5 * (hessian + hessian.T)

        # Check condition number
        try:
            cond_num = np.linalg.cond(sym_hessian)
            if cond_num > 1e12:
                if self.verbose >= 2:
                    logger.warning(
                        f"Ill-conditioned Hessian (cond={cond_num:.2e}) detected. "
                        "Using regularization."
                    )
                # Add small regularization
                n = sym_hessian.shape[0]
                reg_factor = np.trace(sym_hessian) / n * 1e-10
                sym_hessian = sym_hessian + reg_factor * np.eye(n)
        except (np.linalg.LinAlgError, ValueError):
            pass

        # Diagonalize Hessian
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(sym_hessian)
        except np.linalg.LinAlgError as exc:
            logger.error(f"Failed to diagonalize Hessian: {exc}")
            raise

        # Find lowest eigenvalue (transition mode)
        min_index = int(np.argmin(eigenvalues))
        mode_eigenvalue = float(eigenvalues[min_index])
        mode_vector = eigenvectors[:, min_index].copy()

        # Normalize
        mode_norm = np.linalg.norm(mode_vector)
        if mode_norm < 1e-12:
            # Fallback: use gradient direction if available
            if self._last_gradient is not None:
                grad_norm = np.linalg.norm(self._last_gradient)
                if grad_norm > 1e-12:
                    mode_vector = self._last_gradient / grad_norm
                    mode_eigenvalue = -0.01  # Small negative value
                else:
                    msg = "Cannot determine transition mode: zero gradient and zero eigenvector"
                    raise RuntimeError(msg)
            else:
                msg = "Cannot determine transition mode: zero eigenvector and no gradient"
                raise RuntimeError(msg)
        else:
            mode_vector = mode_vector / mode_norm

        return mode_vector, mode_eigenvalue

    def _compute_rfo_step(
        self, gradient: np.ndarray, hessian: np.ndarray, alpha: float
    ) -> tuple[np.ndarray, float]:
        """Compute RFO step using Partitioned RFO (P-RFO) method for TS optimization.

        Following geomeTRIC's implementation of Restricted-Step Partitioned RFO (RS-P-RFO):
        1. Diagonalize Hessian to find transition mode (lowest eigenvalue)
        2. For transition mode: solve 2x2 RFO problem, select highest eigenvalue for maximization
        3. For other modes: minimize using projected RFO formula
        4. Combine steps: y = w_tv * y~_tv + sum(w_k * y~_k)

        This is the correct approach for TS optimization as described in geomeTRIC documentation.

        Parameters
        ----------
        gradient : np.ndarray
            Gradient vector (negative forces)
        hessian : np.ndarray
            Hessian matrix
        alpha : float
            Trust radius control parameter

        Returns
        -------
        step : np.ndarray
            RFO step vector
        lambda_max : float
            Highest eigenvalue from transition mode RFO problem

        """
        n = len(gradient)

        # Step 1: Diagonalize Hessian to find transition mode
        # Symmetrize Hessian
        sym_hessian = 0.5 * (hessian + hessian.T)

        # Diagonalize Hessian
        try:
            hessian_eigenvalues, hessian_eigenvectors = np.linalg.eigh(sym_hessian)
        except np.linalg.LinAlgError as exc:
            logger.error(f"Failed to diagonalize Hessian: {exc}")
            raise

        # Find transition mode (lowest eigenvalue)
        tv_index = int(np.argmin(hessian_eigenvalues))
        omega_tv = float(hessian_eigenvalues[tv_index])
        w_tv = hessian_eigenvectors[:, tv_index].copy()  # Transition mode eigenvector

        # Normalize transition mode
        w_tv_norm = np.linalg.norm(w_tv)
        if w_tv_norm < 1e-12:
            # Fallback: use gradient direction if transition mode is zero
            grad_norm = np.linalg.norm(gradient)
            if grad_norm > 1e-12:
                w_tv = -gradient / grad_norm  # Negative because we want to climb
            else:
                # No gradient, use first eigenvector
                w_tv = hessian_eigenvectors[:, 0] / np.linalg.norm(hessian_eigenvectors[:, 0])
        else:
            w_tv = w_tv / w_tv_norm

        # Store transition mode for reference
        if self._transition_mode is None:
            self._transition_mode = w_tv.copy()
            self._transition_mode_eigenvalue = omega_tv

        # Step 2: Compute step along transition mode using 2x2 RFO problem
        # Project gradient onto transition mode
        g_tilde_tv = np.dot(gradient, w_tv)

        # Build 2x2 RFO matrix for transition mode:
        # [[0, g~_tv],
        #  [g~_tv, ω_tv]]
        rfo_tv_matrix = np.array([[0.0, g_tilde_tv], [g_tilde_tv, omega_tv]])

        # Build 2x2 metric matrix:
        # [[1, 0],
        #  [0, α]]
        metric_tv = np.array([[1.0, 0.0], [0.0, alpha]])

        # Solve 2x2 generalized eigenvalue problem
        try:
            from scipy.linalg import eigh as scipy_eigh

            tv_eigenvalues, tv_eigenvectors = scipy_eigh(rfo_tv_matrix, metric_tv)
        except (ImportError, np.linalg.LinAlgError):
            # Fallback: use standard eigenvalue problem
            metric_tv_inv = np.linalg.inv(metric_tv)
            standard_tv = metric_tv_inv @ rfo_tv_matrix
            tv_eigenvalues, tv_eigenvectors = np.linalg.eig(standard_tv)
            tv_eigenvalues = np.real(tv_eigenvalues)
            tv_eigenvectors = np.real(tv_eigenvectors)

        # Select HIGHEST eigenvalue for maximization (TS climbing)
        tv_max_idx = int(np.argmax(tv_eigenvalues))
        lambda_tv_max = float(tv_eigenvalues[tv_max_idx])
        v_tv = tv_eigenvectors[:, tv_max_idx]

        # Extract step along transition mode: y~_tv = v_tv[1] / v_tv[0]
        # Formula: y~_tv = -g~_tv / (ω_tv - α*λ_tv_max)
        v0_tv = v_tv[0]
        if abs(v0_tv) > 1e-12:
            y_tilde_tv = v_tv[1] / v0_tv
        else:
            # Fallback formula
            denominator = omega_tv - alpha * lambda_tv_max
            if abs(denominator) > 1e-12:
                y_tilde_tv = -g_tilde_tv / denominator
            else:
                y_tilde_tv = 0.0

        # Step 3: Compute steps along other modes (minimization)
        # For each other mode k: y~_k = -g~_k / (ω_k - α*λ_ot;min)
        # We need to solve the RFO problem for the "other" subspace
        # Simplified: use the minimization eigenvalue from transition mode problem
        # or solve a separate RFO problem for the other modes
        # For now, use the standard formula: y~_k = -g~_k / (ω_k - α*λ_min)

        # Find the minimum eigenvalue from transition mode RFO (for minimization)
        tv_min_idx = int(np.argmin(tv_eigenvalues))
        lambda_tv_min = float(tv_eigenvalues[tv_min_idx])

        # Project gradient onto all Hessian eigenvectors
        g_tilde_all = np.dot(hessian_eigenvectors.T, gradient)  # Projection onto eigenbasis

        # Compute steps in eigenbasis
        y_tilde_all = np.zeros(n)
        for k in range(n):
            omega_k = hessian_eigenvalues[k]
            g_tilde_k = g_tilde_all[k]

            if k == tv_index:
                # Use transition mode step
                y_tilde_all[k] = y_tilde_tv
            else:
                # Minimization step: y~_k = -g~_k / (ω_k - α*λ_min)
                # Use lambda_tv_min or a separate lambda_ot_min
                # Simplified: use same alpha scaling with minimization eigenvalue
                denominator = omega_k - alpha * lambda_tv_min
                if abs(denominator) > 1e-12:
                    y_tilde_all[k] = -g_tilde_k / denominator
                else:
                    y_tilde_all[k] = 0.0

        # Step 4: Transform back from eigenbasis to Cartesian coordinates
        # y = sum_k w_k * y~_k
        step = np.zeros(n)
        for k in range(n):
            step += hessian_eigenvectors[:, k] * y_tilde_all[k]

        # Apply trust radius constraint
        step_norm = float(np.linalg.norm(step))
        if step_norm > 1e-12:
            # Scale to trust radius if needed
            if step_norm > self.trust_radius * 1.01:
                step = step / step_norm * self.trust_radius
                step_norm = float(self.trust_radius)
        else:
            # Step is zero, use small step along transition mode
            if np.linalg.norm(w_tv) > 1e-12:
                step = w_tv * self.trust_radius * 0.1
            else:
                step = np.zeros(n)

        if self.verbose >= 2:
            step_norm_final = np.linalg.norm(step)
            grad_step_dot = np.dot(gradient, step) if step_norm_final > 1e-12 else 0.0
            logger.debug(
                f"P-RFO step: lambda_tv_max={lambda_tv_max:.6f}, omega_tv={omega_tv:.6f}, "
                f"step_norm={step_norm_final:.6f}, grad·step={grad_step_dot:.6f}, alpha={alpha:.6f}"
            )

        return step, lambda_tv_max

    def _compute_step_quality(
        self,
        actual_energy_change: float,
        step_vector: np.ndarray,
        gradient: np.ndarray,
        hessian: np.ndarray,
    ) -> float:
        """Compute step quality factor Q based on geomeTRIC approach.

        For TS optimization, both positive and negative energy deviations
        from the predicted change are considered bad (unlike minimization).

        Q = 1 - |ΔE_actual/ΔE_pred - 1|

        Parameters
        ----------
        actual_energy_change : float
            Actual energy change: E_new - E_old
        step_vector : np.ndarray
            Step vector δ = x_new - x_old
        gradient : np.ndarray
            Gradient at previous point (before step)
        hessian : np.ndarray
            Hessian at previous point (before step)

        Returns
        -------
        float
            Quality factor Q ∈ [-∞, 1]. Higher is better.
            Q ≥ 0.75: Good step
            0.5 ≤ Q < 0.75: Okay step
            0 ≤ Q < 0.5: Poor step
            Q < 0: Very poor step (should be rejected)

        """
        # Predicted energy change: ΔE_pred = 0.5 * δ^T * H * δ + δ^T * g
        H_delta = hessian @ step_vector
        predicted_quadratic = 0.5 * np.dot(step_vector, H_delta)
        predicted_linear = np.dot(step_vector, gradient)
        predicted_change = predicted_quadratic + predicted_linear

        # Handle edge cases
        if abs(predicted_change) < 1e-12:
            # Very small predicted change
            if abs(actual_energy_change) < 1e-12:
                return 1.0  # Both zero, perfect match
            return -abs(actual_energy_change)  # Negative quality for unexpected change

        # Quality factor: Q = 1 - |ΔE_actual/ΔE_pred - 1|
        ratio = actual_energy_change / predicted_change
        quality: float = 1.0 - abs(ratio - 1.0)

        return quality

    def _adjust_trust_radius(self, step_quality: float, step_size: float) -> None:
        """Adjust trust radius based on step quality.

        Following geomeTRIC's approach:
        - Q ≥ 0.75: Good step, increase by √2
        - 0.50 ≤ Q < 0.75: Okay step, unchanged
        - 0 ≤ Q < 0.50: Poor step, decrease
        - Q < 0: Very poor, reject step and decrease

        Parameters
        ----------
        step_quality : float
            Step quality factor Q
        step_size : float
            Norm of the step taken

        """
        if step_quality >= 0.75:
            # Good step
            new_trust = self.trust_radius * np.sqrt(2.0)
            self.trust_radius = min(new_trust, self.max_trust_radius)
            if self.verbose >= 2:
                logger.debug(
                    f"Good step (Q={step_quality:.4f}), increased trust radius to {self.trust_radius:.6f}"
                )
        elif step_quality >= 0.50:
            # Okay step
            if self.verbose >= 2:
                logger.debug(
                    f"Okay step (Q={step_quality:.4f}), trust radius unchanged: {self.trust_radius:.6f}"
                )
            # Trust radius unchanged
        elif step_quality >= 0.0:
            # Poor step
            new_trust = 0.5 * min(self.trust_radius, step_size)
            self.trust_radius = max(new_trust, self.min_trust_radius)
            if self.verbose >= 1:
                try:
                    logger.warning(
                        f"Poor step (Q={step_quality:.4f}), decreased trust radius to {self.trust_radius:.6f}"
                    )
                except (ValueError, OSError):
                    # Handle closed file streams gracefully
                    pass
        else:
            # Very poor step (Q < 0)
            new_trust = 0.5 * min(self.trust_radius, step_size)
            self.trust_radius = max(new_trust, self.min_trust_radius)
            if self.verbose >= 1:
                try:
                    logger.warning(
                        f"Very poor step (Q={step_quality:.4f}), decreased trust radius to {self.trust_radius:.6f}"
                    )
                except (ValueError, OSError):
                    # Handle closed file streams gracefully
                    pass

    def _find_optimal_alpha(self, gradient: np.ndarray, hessian: np.ndarray) -> float:
        """Find optimal alpha parameter for trust radius control.

        Adjusts alpha to ensure step size matches trust radius constraint.
        Uses bisection method for more reliable convergence.

        Parameters
        ----------
        gradient : np.ndarray
            Gradient vector
        hessian : np.ndarray
            Hessian matrix

        Returns
        -------
        float
            Optimal alpha value

        """
        # Use bisection to find alpha that gives step size = trust_radius
        alpha_min = 1e-4
        alpha_max = 1e4

        # Start with current alpha if it's reasonable
        if 1e-4 < self.alpha < 1e4:
            alpha = self.alpha
        else:
            alpha = 1.0

        max_iter = 20
        tolerance = 0.01 * self.trust_radius

        for _iteration in range(max_iter):
            try:
                step, _ = self._compute_rfo_step(gradient, hessian, alpha)
                step_norm = np.linalg.norm(step)

                # If step is close to trust radius, we're done
                error = abs(step_norm - self.trust_radius)
                if error < tolerance:
                    break

                # Use bisection: larger alpha -> smaller step
                if step_norm > self.trust_radius:
                    # Step too large, need larger alpha
                    alpha_min = alpha
                    if alpha_max < 1e4:
                        alpha = (alpha + alpha_max) / 2.0
                    else:
                        alpha *= 1.5
                else:
                    # Step too small, need smaller alpha
                    alpha_max = alpha
                    if alpha_min > 1e-4:
                        alpha = (alpha_min + alpha) / 2.0
                    else:
                        alpha *= 0.8

                # Bounds on alpha
                alpha = max(1e-4, min(alpha, 1e4))

                # Prevent infinite loop
                if alpha_min >= alpha_max - 1e-10:
                    break

            except Exception as e:
                if self.verbose >= 2:
                    logger.debug(f"Alpha optimization error: {e}")
                # If computation fails, use current alpha
                break

        self.alpha = alpha
        return alpha

    def run(self, fmax: float = 0.05, steps: int = 100) -> bool:  # type: ignore[override]
        """Run the RFO optimization.

        Parameters
        ----------
        fmax : float
            Maximum force convergence criterion (eV/Å).
        steps : int
            Maximum number of optimization steps.

        Returns
        -------
        bool
            True if converged, False otherwise.

        """
        self.fmax = float(fmax)
        self.max_steps = int(steps + self.nsteps)

        # Get initial position and energy
        x0 = self._positions_to_x()
        initial_energy = self.atoms.get_potential_energy()

        # Initialize state
        if self.nsteps == 0:
            self._previous_energy = initial_energy
            self._previous_positions = x0.copy()
            forces = self.atoms.get_forces()
            self._previous_fmax = np.max(np.abs(forces))
            self.log(forces)
            self.call_observers()
            self.nsteps += 1

        if self.verbose >= 2:
            logger.info("Starting RFO transition state optimization")
            logger.info(f"Convergence criterion: fmax = {fmax} eV/Å")
            logger.info(f"Maximum steps: {steps}")

        try:
            x = x0.copy()

            while self.nsteps < self.max_steps:
                # Get gradient and Hessian at current position
                gradient = self._get_gradient(x)
                hessian = self._compute_hessian(x)

                # If we have previous state, update Hessian using MS-PSB
                if (
                    self._last_positions is not None
                    and self._last_gradient is not None
                    and not np.array_equal(x, self._last_positions)
                ):
                    self._update_hessian_ms_psb(x, gradient)
                    # Re-get gradient after Hessian update (might have changed atoms)
                    gradient = self._get_gradient(x)

                # Find transition mode if needed
                if self._transition_mode is None:
                    mode_vector, mode_eigenvalue = self._find_transition_mode(hessian)
                    self._transition_mode = mode_vector
                    self._transition_mode_eigenvalue = mode_eigenvalue
                    if self.verbose >= 2:
                        logger.info(
                            f"Transition mode found: eigenvalue = {mode_eigenvalue:.6f} eV/Å²"
                        )

                # Find optimal alpha for trust radius control
                alpha_opt = self._find_optimal_alpha(gradient, hessian)

                # Compute RFO step
                try:
                    step, lambda_max = self._compute_rfo_step(gradient, hessian, alpha_opt)
                except Exception as e:
                    logger.error(f"RFO step computation failed: {e}")
                    return False

                # Check step size
                step_norm: float = float(np.linalg.norm(step))
                if step_norm < 1e-12:
                    if self.verbose >= 1:
                        logger.warning("Step size is zero, stopping optimization")
                    break

                # Take step
                x_new = x + step
                self.atoms.set_positions(self._x_to_positions(x_new))
                new_energy = self.atoms.get_potential_energy()

                # Compute step quality
                energy_change = new_energy - self._previous_energy

                # Debug: Check step direction and predicted energy change
                if self.verbose >= 2:
                    # Predicted energy change for debugging
                    H_delta = hessian @ step
                    predicted_quad = 0.5 * np.dot(step, H_delta)
                    predicted_linear = np.dot(step, gradient)
                    predicted_change = predicted_quad + predicted_linear

                    # Check alignment with transition mode
                    mode_alignment = 0.0
                    if self._transition_mode is not None:
                        step_norm_check = np.linalg.norm(step)
                        mode_norm = np.linalg.norm(self._transition_mode)
                        if step_norm_check > 1e-12 and mode_norm > 1e-12:
                            mode_alignment = np.dot(step, self._transition_mode) / (
                                step_norm_check * mode_norm
                            )

                    logger.debug(
                        f"Step {self.nsteps}: ΔE_actual={energy_change:.6f}, "
                        f"ΔE_pred={predicted_change:.6f}, "
                        f"step_norm={step_norm:.6f}, "
                        f"mode_align={mode_alignment:.4f}"
                    )

                step_quality = self._compute_step_quality(energy_change, step, gradient, hessian)
                self._step_quality_history.append(step_quality)

                # Adjust trust radius based on step quality
                self._adjust_trust_radius(step_quality, step_norm)

                # Check if step should be rejected (very poor quality)
                should_reject = False
                if step_quality < -1.0:  # Very poor prediction
                    # Additional check: are forces getting worse?
                    forces = self.atoms.get_forces()
                    current_fmax = np.max(np.abs(forces))

                    if self._previous_energy is not None and abs(energy_change) > 10.0:
                        # Energy changed by > 10 eV, likely a bad step
                        should_reject = True
                    elif hasattr(self, "_previous_fmax") and self._previous_fmax is not None:
                        if current_fmax > self._previous_fmax * 1.5:
                            # Forces increased significantly
                            should_reject = True
                    else:
                        # First step or no previous fmax - reject only if extremely bad
                        should_reject = step_quality < -5.0

                if should_reject:
                    # Reject step
                    x_new = x  # Revert to previous position
                    self.atoms.set_positions(self._x_to_positions(x_new))

                    # Force Hessian recomputation on next iteration after rejection
                    steps_since_hessian = self.nsteps - self._last_full_hessian_step
                    if steps_since_hessian > 2:
                        if self.verbose >= 1:
                            logger.info(
                                f"Step rejected (Q={step_quality:.4f}), "
                                "will recompute Hessian on next iteration"
                            )
                        self._last_full_hessian_step = self.nsteps - (
                            self.hessian_update_freq or 10
                        )

                    if self.verbose >= 1:
                        try:
                            logger.warning(
                                f"Step rejected due to poor quality (Q={step_quality:.4f})"
                            )
                        except (ValueError, OSError):
                            # Handle closed file streams gracefully
                            pass
                else:
                    # Accept step - update state for next iteration
                    # Store OLD position and gradient for MS-PSB update in next iteration
                    self._last_positions = x.copy()  # Position BEFORE step
                    self._last_gradient = gradient.copy()  # Gradient BEFORE step
                    # Now update x to new position
                    x = x_new
                    self._previous_energy = new_energy
                    self._previous_positions = x.copy()
                    # Store fmax for step rejection logic
                    forces = self.atoms.get_forces()
                    self._previous_fmax = np.max(np.abs(forces))

                # Update step counter
                self.nsteps += 1

                # Log step
                forces = self.atoms.get_forces()
                fmax_current = np.max(np.abs(forces))
                self.log(forces)
                self.call_observers()

                if self.verbose >= 2:
                    logger.info(
                        f"Step {self.nsteps}: energy = {new_energy:.8f} eV, "
                        f"fmax = {fmax_current:.6f} eV/Å, "
                        f"Q = {step_quality:.4f}, "
                        f"trust_radius = {self.trust_radius:.6f} Å"
                    )

                # Check convergence
                forces_flat = forces.ravel()
                if self.converged(forces_flat):
                    raise ConvergedError

        except ConvergedError:
            if self.verbose >= 1:
                logger.info("Optimization converged!")
            return True

        # Check final convergence
        forces = self.atoms.get_forces()
        forces_flat = forces.ravel()
        converged: bool = bool(self.converged(forces_flat))

        if converged:
            if self.verbose >= 1:
                logger.info("Optimization converged!")
        elif self.verbose >= 1:
            try:
                logger.warning(f"Optimization stopped after {steps} steps without converging")
                logger.warning(f"Final max force: {np.max(np.abs(forces)):.6f} eV/Å")
            except (ValueError, OSError):
                # Handle closed file streams gracefully
                pass

        # Log step quality summary
        if self._step_quality_history and self.verbose >= 1:
            avg_quality = np.mean(self._step_quality_history)
            good_steps = sum(1 for q in self._step_quality_history if q >= 0.75)
            poor_steps = sum(1 for q in self._step_quality_history if q < 0.0)
            if self.verbose >= 2:
                logger.info(
                    f"Step quality summary: avg={avg_quality:.3f}, "
                    f"good={good_steps}, poor={poor_steps}"
                )

        return converged

    def get_number_of_steps(self) -> int:
        """Get the number of optimization steps taken."""
        return self.nsteps

    def dump(self, data: Any) -> None:
        """Dump optimizer state (not implemented for RFO optimizer)."""

    def load(self) -> None:
        """Load optimizer state (not implemented for RFO optimizer)."""
