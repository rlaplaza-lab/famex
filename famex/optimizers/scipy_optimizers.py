"""SciPy-based optimizers with Hessian support for FAMEX."""

from __future__ import annotations

from typing import IO, Any, cast

import numpy as np
from ase import Atoms
from ase.optimize.optimize import Optimizer
from scipy.optimize import minimize

from famex.optimizers.ts_step import (
    adjust_trust_radius,
    build_translation_rotation_basis,
    compute_dense_prfo_step,
    compute_matrix_free_ts_step,
    compute_step_quality,
    make_atoms_hessp,
    prepare_ts_hessian,
    project_gradient,
)
from famex.utils.logging import get_famex_logger

logger = get_famex_logger(__name__)


class ConvergedError(Exception):
    """Exception raised when optimizer has converged."""


class SciPyHessianOptimizer(Optimizer):
    """Base class for SciPy optimizers that use Hessian information."""

    def __init__(
        self,
        atoms: Atoms,
        method: str = "trust-krylov",
        logfile: IO | str | None = "-",
        trajectory: str | None = None,
        hessian_update_freq: int | None = None,
        hessian_method: str = "auto",
        hessian_delta: float = 0.01,
        initial_hessian: np.ndarray | None = None,
        alpha: float = 1.0,
        use_bfgs_update: bool = True,
        adaptive_hessian: bool = False,
        force_threshold_ratio: float = 2.0,
        verbose: int = 1,
        ts_search: bool = False,
        trust_radius: float = 0.1,
        max_trust_radius: float = 0.3,
        min_trust_radius: float = 0.001,
        **kwargs: Any,
    ) -> None:
        """Initialize SciPy Hessian-based optimizer."""
        self.verbose = verbose
        self.ts_search = ts_search
        self.hessian_delta = hessian_delta

        if verbose == 0:
            logfile = None

        restart = None
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "profiler"}
        Optimizer.__init__(self, atoms, restart, logfile, trajectory, **filtered_kwargs)

        self.method = method
        freq = hessian_update_freq
        if freq is not None and freq <= 0:
            logger.warning("hessian_update_freq <= 0 provided; disabling periodic Hessian updates")
            freq = None

        self.hessian_update_freq: int | None = freq
        self.hessian_method = hessian_method
        self.alpha = alpha
        self.use_bfgs_update = use_bfgs_update
        self.adaptive_hessian = adaptive_hessian
        self.force_threshold_ratio = force_threshold_ratio

        if atoms.calc is None:
            raise ValueError("Atoms object must have a calculator attached")

        from famex.analysis.frequency import FrequencyAnalysis

        self.freq_analysis = FrequencyAnalysis(
            atoms=atoms,
            calculator=atoms.calc,
            delta=hessian_delta,
            verbose=verbose,
        )

        self.hessian = initial_hessian
        self.force_calls = 0
        self.hessian_calls = 0
        self._last_hessian_step = -1
        self._last_full_hessian_step = -1
        self.bfgs_updates = 0
        self._last_positions: np.ndarray | None = None
        self._last_gradient: np.ndarray | None = None
        self._previous_fmax = None
        self.max_steps: int = 0
        self._scipy_result = None
        if not hasattr(self, "fmax"):
            self.fmax = 0.05
        self.fmax: float = getattr(self, "fmax", 0.05)  # type: ignore[assignment]

        if ts_search:
            if method == "Newton-CG":
                msg = (
                    "Newton-CG is not supported for transition-state searches. "
                    "Use 'trust-krylov', 'trust-ncg', 'trust-exact', 'rfo', or 'sella' instead."
                )
                raise ValueError(msg)
            self._ts_trust_radius = trust_radius
            self.trust_radius = trust_radius
            self.max_trust_radius = max_trust_radius
            self.min_trust_radius = min_trust_radius
            self._transition_mode: np.ndarray | None = None
            self._previous_energy: float | None = None
            if hessian_update_freq is None:
                self.hessian_update_freq = 1
            if method == "trust-exact":
                self._ts_step_mode = "dense"
                self._ts_subproblem = "exact"
            else:
                self._ts_step_mode = "matrix_free"
                self._ts_subproblem = "krylov" if method == "trust-krylov" else "ncg"

        valid_methods = ["trust-krylov", "trust-ncg", "trust-exact", "Newton-CG"]
        if method not in valid_methods:
            msg = f"Invalid method '{method}'. Valid methods: {valid_methods}"
            raise ValueError(msg)

        if self.verbose >= 2:
            logger.info(f"Initialized {method} optimizer with Hessian support")
            if adaptive_hessian:
                if hessian_update_freq is None:
                    logger.info(
                        "Adaptive Hessian updates enabled (force-triggered; no periodic updates)",
                    )
                else:
                    logger.info(
                        f"Adaptive Hessian updates enabled (base frequency: every {hessian_update_freq} step(s))",
                    )
            elif hessian_update_freq is None:
                logger.info("Periodic Hessian updates disabled (compute once and reuse)")
            else:
                logger.info(
                    f"Fixed Hessian update frequency: every {hessian_update_freq} step(s)",
                )
            if use_bfgs_update:
                logger.info("BFGS approximate updates enabled between full Hessians")
            logger.info(f"Hessian calculation method: {hessian_method}")

    def _positions_to_x(self, atoms: Atoms | None = None) -> np.ndarray:
        """Convert positions to 1D array."""
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise RuntimeError("Atoms object is not initialized")
        return cast(np.ndarray, atoms.get_positions().ravel())

    def _x_to_positions(self, x: np.ndarray) -> np.ndarray:
        """Convert 1D array to positions."""
        return cast(np.ndarray, x.reshape(-1, 3))

    def objective(self, x: np.ndarray) -> float:
        """Objective function for minimization (potential energy)."""
        self.atoms.set_positions(self._x_to_positions(x))
        energy = self.atoms.get_potential_energy()
        return energy / self.alpha  # type: ignore[no-any-return]

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Gradient of objective function (negative forces)."""
        self.atoms.set_positions(self._x_to_positions(x))
        self.force_calls += 1

        forces = self.atoms.get_forces()
        gradient = -forces.ravel()

        if self.use_bfgs_update:
            if self._last_gradient is None:
                self._last_gradient = gradient.copy()
            if self._last_positions is None:
                self._last_positions = x.copy()

        return gradient / self.alpha  # type: ignore[no-any-return]

    def hessian_func(self, x: np.ndarray) -> np.ndarray:
        """Compute Hessian matrix with adaptive updates and BFGS approximation."""
        current_positions = x.copy()
        steps_since_full = self.nsteps - self._last_full_hessian_step

        need_full_update = False
        reason = "unknown"

        if self.hessian is None:
            need_full_update = True
            reason = "initial"
        elif self.adaptive_hessian:
            current_fmax = self._get_current_fmax()

            if current_fmax is not None and self._previous_fmax is not None:
                if current_fmax > self._previous_fmax * self.force_threshold_ratio:  # type: ignore[unreachable]
                    need_full_update = True
                    reason = f"force increase ({current_fmax:.4f} > {self._previous_fmax:.4f} × {self.force_threshold_ratio})"

            if (
                not need_full_update
                and self.hessian_update_freq is not None
                and steps_since_full >= self.hessian_update_freq
            ):
                need_full_update = True
                reason = f"periodic (every {self.hessian_update_freq} steps)"
        elif self.hessian_update_freq is not None and steps_since_full >= self.hessian_update_freq:
            need_full_update = True
            reason = f"fixed frequency ({self.hessian_update_freq} steps)"

        if need_full_update:
            self.atoms.set_positions(self._x_to_positions(x))
            self.hessian_calls += 1
            self._last_hessian_step = self.nsteps
            self._last_full_hessian_step = self.nsteps

            if self.verbose >= 1:
                logger.info(
                    f"Computing full Hessian at step {self.nsteps} "
                    f"(call #{self.hessian_calls}, reason: {reason})",
                )

            self.freq_analysis.atoms = self.atoms
            self.freq_analysis.atoms.calc = self.atoms.calc

            hessian = self.freq_analysis.calculate_hessian(method=self.hessian_method)

            self.hessian = hessian

            if self.verbose >= 2:
                logger.info(f"Full Hessian computed (shape: {hessian.shape})")

            self._last_positions = current_positions.copy()
            self._last_gradient = None
            self.bfgs_updates = 0

        elif (
            self.use_bfgs_update
            and self._last_positions is not None
            and self._last_gradient is not None
        ):
            self._last_hessian_step = self.nsteps

            current_gradient = self.gradient(x)
            s = current_positions - self._last_positions
            y = current_gradient - self._last_gradient

            sy = np.dot(s, y)

            if sy > 1e-10:
                if self.hessian is None:
                    raise RuntimeError("Hessian unavailable during BFGS update")
                Hs = np.dot(self.hessian, s)
                sHs = np.dot(s, Hs)

                self.hessian = self.hessian + np.outer(y, y) / sy - np.outer(Hs, Hs) / sHs

                self.bfgs_updates += 1
                logger.debug(
                    f"BFGS Hessian update at step {self.nsteps} "
                    f"(#{self.bfgs_updates} since last full, "
                    f"{steps_since_full} steps since full Hessian)",
                )
            else:
                logger.debug(f"Skipping BFGS update (sy = {sy:.2e} too small)")

            self._last_positions = current_positions.copy()
            self._last_gradient = current_gradient.copy()
        else:
            if self.verbose >= 1:
                logger.info(f"Reusing Hessian (computed {steps_since_full} steps ago)")
            else:
                logger.debug(f"Reusing Hessian (computed {steps_since_full} steps ago)")

        if self.hessian is None:
            raise RuntimeError("Hessian is None after update logic")
        return self.hessian / self.alpha

    def _get_current_fmax(self) -> float | None:
        """Get current max force."""
        try:
            forces = self.atoms.get_forces()
            return np.max(np.abs(forces))  # type: ignore[no-any-return]
        except Exception:
            return None

    def callback(self, x: np.ndarray, **kwargs: Any) -> None:
        """Handle SciPy callback after each iteration."""
        self.atoms.set_positions(self._x_to_positions(x))

        if self.nsteps < self.max_steps:
            self.nsteps += 1

        forces = self.atoms.get_forces()
        fmax = np.max(np.abs(forces))

        self._previous_fmax = fmax

        self.log(forces)
        self.call_observers()

        forces_flat = forces.ravel()
        if self.converged(forces_flat):
            raise ConvergedError

    def _get_ts_gradient(self, x: np.ndarray) -> np.ndarray:
        self.atoms.set_positions(self._x_to_positions(x))
        self.force_calls += 1
        gradient = -self.atoms.get_forces().ravel()
        basis = build_translation_rotation_basis(
            self.atoms.get_positions(),
            self.atoms.get_masses(),
        )
        return project_gradient(gradient, basis)

    def _compute_ts_hessian(self, x: np.ndarray) -> np.ndarray:
        steps_since_full = self.nsteps - self._last_full_hessian_step
        need_full = self.hessian is None or (
            self.hessian_update_freq is not None and steps_since_full >= self.hessian_update_freq
        )
        if need_full:
            self.atoms.set_positions(self._x_to_positions(x))
            self.hessian_calls += 1
            self._last_full_hessian_step = self.nsteps
            if self.verbose >= 1:
                logger.info(
                    f"Computing full Hessian at step {self.nsteps} (call #{self.hessian_calls})",
                )
            self.freq_analysis.atoms = self.atoms
            self.freq_analysis.atoms.calc = self.atoms.calc
            hessian = self.freq_analysis.calculate_hessian(method=self.hessian_method)
            basis = build_translation_rotation_basis(
                self.atoms.get_positions(),
                self.atoms.get_masses(),
            )
            self.hessian = prepare_ts_hessian(hessian, basis)
        if self.hessian is None:
            raise RuntimeError("Hessian is None after update logic")
        return self.hessian

    def _run_ts_search(self, fmax: float, steps: int) -> bool:
        """Run transition-state search with Hessian usage aligned to the SciPy method."""
        x = self._positions_to_x()
        self._previous_energy = float(self.atoms.get_potential_energy())

        if self.verbose >= 2:
            logger.info(f"Starting {self.method} transition-state search ({self._ts_step_mode})")

        try:
            while self.nsteps < self.max_steps:
                gradient = self._get_ts_gradient(x)

                if self._ts_step_mode == "dense":
                    hessian = self._compute_ts_hessian(x)
                    step_result = compute_dense_prfo_step(
                        gradient=gradient,
                        hessian=hessian,
                        trust_radius=self.trust_radius,
                        previous_mode=self._transition_mode,
                        alpha=self.alpha,
                    )
                else:
                    basis = build_translation_rotation_basis(
                        self.atoms.get_positions(),
                        self.atoms.get_masses(),
                    )
                    hessp = make_atoms_hessp(
                        self.atoms,
                        x,
                        self._x_to_positions,
                        delta=self.hessian_delta,
                        basis=basis,
                    )
                    step_result = compute_matrix_free_ts_step(
                        gradient=gradient,
                        hessp=hessp,
                        trust_radius=self.trust_radius,
                        previous_mode=self._transition_mode,
                        subproblem=self._ts_subproblem,
                    )

                self._transition_mode = step_result.transition_mode
                step = step_result.step
                step_norm = float(np.linalg.norm(step))
                if step_norm < 1e-12:
                    if self.verbose >= 1:
                        logger.warning("Step size is zero, stopping optimization")
                    break

                x_new = x + step
                self.atoms.set_positions(self._x_to_positions(x_new))
                new_energy = float(self.atoms.get_potential_energy())
                energy_change = new_energy - (self._previous_energy or new_energy)
                quality = compute_step_quality(energy_change, step_result.predicted_energy_change)

                self.trust_radius = adjust_trust_radius(
                    self.trust_radius,
                    quality,
                    step_norm,
                    self.min_trust_radius,
                    self.max_trust_radius,
                )
                self._ts_trust_radius = self.trust_radius

                reject = quality < -1.0 and abs(energy_change) > 10.0
                if not reject:
                    x = x_new
                    self._previous_energy = new_energy
                else:
                    self.atoms.set_positions(self._x_to_positions(x))
                    if self.verbose >= 1:
                        logger.warning(f"Rejected poor TS step (Q={quality:.4f})")

                self.nsteps += 1
                forces = self.atoms.get_forces()
                self.log(forces)
                self.call_observers()

                if self.converged(forces.ravel()):
                    raise ConvergedError

        except ConvergedError:
            if self.verbose >= 1:
                logger.info("Transition-state optimization converged!")
            return True

        forces = self.atoms.get_forces()
        return bool(self.converged(forces.ravel()))

    def run(self, fmax: float = 0.05, steps: int = 100) -> bool:  # type: ignore[override]
        """Run the optimization."""
        self.fmax = float(fmax)
        self.max_steps = int(steps + self.nsteps)

        x0 = self._positions_to_x()

        if self.nsteps == 0:
            forces = self.atoms.get_forces()
            self.log(forces)
            self.call_observers()
            self.nsteps += 1

        if self.ts_search:
            return self._run_ts_search(fmax, steps)

        if self.verbose >= 2:
            logger.info(f"Starting {self.method} optimization")
            logger.info(f"Convergence criterion: fmax = {fmax} eV/Å")
            logger.info(f"Maximum steps: {steps}")

        try:
            options: dict[str, Any] = {
                "maxiter": steps,
                "disp": False,
            }

            if self.method in ("trust-krylov", "trust-ncg", "trust-exact"):
                is_ts_optimizer = hasattr(self, "_ts_trust_radius")
                if is_ts_optimizer:
                    options["gtol"] = 1e-9
                else:
                    options["gtol"] = 1e-8
                if is_ts_optimizer:
                    options["initial_tr_radius"] = 2.0
                else:
                    options["initial_tr_radius"] = 1.0
                options["max_tr_radius"] = 10.0
                if self.method == "trust-krylov":
                    options["maxiter"] = int(steps)
                    options["inexact"] = True

            self._scipy_result = minimize(  # type: ignore[call-overload]
                fun=self.objective,
                x0=x0,
                method=self.method,
                jac=self.gradient,
                hess=self.hessian_func,
                callback=self.callback,
                options=options,
            )

            if self._scipy_result is not None:
                self.atoms.set_positions(self._x_to_positions(self._scipy_result.x))  # type: ignore[unreachable]
        except ConvergedError:
            if self.verbose >= 1:
                logger.info("Optimization converged!")
            return True

        forces = self.atoms.get_forces()
        forces_flat = forces.ravel()
        converged_result = self.converged(forces_flat)
        converged: bool = bool(converged_result)

        if converged:
            if self.verbose >= 1:
                logger.info("Optimization converged!")
        elif self.verbose >= 1:
            scipy_iterations = (
                self._scipy_result.nit if self._scipy_result is not None else self.nsteps - 1
            )
            callback_steps = self.nsteps - 1

            actual_steps = scipy_iterations if self._scipy_result is not None else callback_steps

            scipy_message = (
                str(getattr(self._scipy_result, "message", "unknown reason"))
                if self._scipy_result is not None
                else "unknown reason"
            )

            scipy_stopped_due_to_maxiter = (
                "Maximum number of iterations" in scipy_message
                or "max iterations" in scipy_message.lower()
            )

            if scipy_stopped_due_to_maxiter:
                logger.warning(
                    f"Optimization stopped after {actual_steps} trust-region steps "
                    f"(reached max outer iterations: {steps})"
                )
            else:
                logger.warning(
                    f"Optimization stopped after {actual_steps} trust-region steps without converging"
                )
                logger.warning(f"  SciPy reason: {scipy_message}")
                if "trust-krylov" in self.method:
                    logger.warning(
                        "  Note: Each trust-region step uses inner Krylov iterations to solve the subproblem. "
                        "Failure may indicate inner Krylov solver struggling with the subproblem "
                        "(each step may perform up to ~1000 inner Krylov iterations)."
                    )
            logger.warning(f"Final max force: {np.max(np.abs(forces)):.6f} eV/Å")

        return converged

    def get_number_of_steps(self) -> int:
        """Get number of optimization steps."""
        return self.nsteps

    def dump(self, data: Any) -> None:
        """Dump optimizer state (not implemented)."""

    def load(self) -> None:
        """Load optimizer state (not implemented)."""


class TrustKrylov(SciPyHessianOptimizer):
    """Trust-Krylov optimizer for ASE."""

    def __init__(
        self,
        atoms: Atoms,
        logfile: IO | str | None = "-",
        trajectory: str | None = None,
        hessian_update_freq: int | None = None,
        hessian_method: str = "auto",
        hessian_delta: float = 0.01,
        initial_hessian: np.ndarray | None = None,
        use_bfgs_update: bool = True,
        adaptive_hessian: bool = False,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize Trust-Krylov optimizer."""
        super().__init__(
            atoms=atoms,
            method="trust-krylov",
            logfile=logfile,
            trajectory=trajectory,
            hessian_update_freq=hessian_update_freq,
            hessian_method=hessian_method,
            hessian_delta=hessian_delta,
            initial_hessian=initial_hessian,
            use_bfgs_update=use_bfgs_update,
            adaptive_hessian=adaptive_hessian,
            verbose=verbose,
            **kwargs,
        )


class TrustNCG(SciPyHessianOptimizer):
    """Trust-NCG optimizer for ASE."""

    def __init__(
        self,
        atoms: Atoms,
        logfile: IO | str | None = "-",
        trajectory: str | None = None,
        hessian_update_freq: int | None = None,
        hessian_method: str = "auto",
        hessian_delta: float = 0.01,
        initial_hessian: np.ndarray | None = None,
        use_bfgs_update: bool = True,
        adaptive_hessian: bool = False,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize Trust-NCG optimizer."""
        super().__init__(
            atoms=atoms,
            method="trust-ncg",
            logfile=logfile,
            trajectory=trajectory,
            hessian_update_freq=hessian_update_freq,
            hessian_method=hessian_method,
            hessian_delta=hessian_delta,
            initial_hessian=initial_hessian,
            use_bfgs_update=use_bfgs_update,
            adaptive_hessian=adaptive_hessian,
            verbose=verbose,
            **kwargs,
        )


class TrustExact(SciPyHessianOptimizer):
    """Trust-Exact optimizer for ASE."""

    def __init__(
        self,
        atoms: Atoms,
        logfile: IO | str | None = "-",
        trajectory: str | None = None,
        hessian_update_freq: int | None = None,
        hessian_method: str = "auto",
        hessian_delta: float = 0.01,
        initial_hessian: np.ndarray | None = None,
        use_bfgs_update: bool = True,
        adaptive_hessian: bool = False,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize Trust-Exact optimizer."""
        super().__init__(
            atoms=atoms,
            method="trust-exact",
            logfile=logfile,
            trajectory=trajectory,
            hessian_update_freq=hessian_update_freq,
            hessian_method=hessian_method,
            hessian_delta=hessian_delta,
            initial_hessian=initial_hessian,
            use_bfgs_update=use_bfgs_update,
            adaptive_hessian=adaptive_hessian,
            verbose=verbose,
            **kwargs,
        )


class NewtonCG(SciPyHessianOptimizer):
    """Newton-CG optimizer for ASE."""

    def __init__(
        self,
        atoms: Atoms,
        logfile: IO | str | None = "-",
        trajectory: str | None = None,
        hessian_update_freq: int | None = None,
        hessian_method: str = "auto",
        hessian_delta: float = 0.01,
        initial_hessian: np.ndarray | None = None,
        use_bfgs_update: bool = True,
        adaptive_hessian: bool = False,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize Newton-CG optimizer."""
        super().__init__(
            atoms=atoms,
            method="Newton-CG",
            logfile=logfile,
            trajectory=trajectory,
            hessian_update_freq=hessian_update_freq,
            hessian_method=hessian_method,
            hessian_delta=hessian_delta,
            initial_hessian=initial_hessian,
            use_bfgs_update=use_bfgs_update,
            adaptive_hessian=adaptive_hessian,
            verbose=verbose,
            **kwargs,
        )
