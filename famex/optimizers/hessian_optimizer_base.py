"""Shared Hessian optimizer infrastructure for SciPy and TS optimizers."""

from __future__ import annotations

from typing import IO, Any, cast

import numpy as np
from ase import Atoms
from ase.optimize.optimize import Optimizer

from famex.analysis.frequency import FrequencyAnalysis
from famex.optimizers._exceptions import ConvergedError
from famex.optimizers.messages import (
    MSG_CONVERGENCE_CRITERION_FMAX,
    MSG_INITIALIZED_SCIPY_HESSIAN,
    MSG_MAXIMUM_STEPS,
    MSG_STARTING_NAMED_OPTIMIZATION,
    MSG_TS_OPTIMIZATION_CONVERGED,
    log_optimization_converged,
    log_scipy_trust_region_not_converged,
)
from famex.optimizers.ts_driver import TSDriverOptions, run_ts_optimization_loop
from famex.optimizers.ts_step import (
    TSStepResult,
    compute_matrix_free_ts_step,
    compute_restricted_prfo_step,
    make_atoms_hessp,
    prepare_ts_hessian,
)
from famex.utils.hessian_ops import (
    build_translation_rotation_basis,
    project_gradient,
)
from famex.utils.logging import get_famex_logger

logger = get_famex_logger(__name__)

__all__ = ["HessianOptimizerBase", "ProjectedHessianOptimizerMixin"]


class ProjectedHessianOptimizerMixin:
    """Position, projected-gradient, and projected-Hessian helpers."""

    atoms: Atoms
    verbose: int
    hessian_update_freq: int | None
    hessian_method: str
    freq_analysis: FrequencyAnalysis
    hessian: np.ndarray | None
    hessian_calls: int
    _last_full_hessian_step: int
    nsteps: int

    def _positions_to_x(self, atoms: Atoms | None = None) -> np.ndarray:
        target = self.atoms if atoms is None else atoms
        return cast(np.ndarray, target.get_positions().ravel())

    def _x_to_positions(self, x: np.ndarray) -> np.ndarray:
        return cast(np.ndarray, x.reshape(-1, 3))

    def _get_ts_gradient(self, x: np.ndarray) -> np.ndarray:
        self.atoms.set_positions(self._x_to_positions(x))
        gradient = -self.atoms.get_forces().ravel()
        basis = build_translation_rotation_basis(
            self.atoms.get_positions(),
            self.atoms.get_masses(),
        )
        return project_gradient(gradient, basis)

    def _refresh_projected_hessian(self, x: np.ndarray, *, log_reuse: bool) -> np.ndarray:
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
                    "Computing full Hessian at step %s (call #%s)",
                    self.nsteps,
                    self.hessian_calls,
                )
            self.freq_analysis.atoms = self.atoms
            self.freq_analysis.atoms.calc = self.atoms.calc
            hessian = self.freq_analysis.calculate_hessian(method=self.hessian_method)
            basis = build_translation_rotation_basis(
                self.atoms.get_positions(),
                self.atoms.get_masses(),
            )
            self.hessian = prepare_ts_hessian(hessian, basis)
        elif log_reuse:
            log = logger.info if self.verbose >= 1 else logger.debug
            log("Reusing Hessian (computed %s steps ago)", steps_since_full)

        if self.hessian is None:
            raise RuntimeError("Hessian is None after update logic")
        return self.hessian


class HessianOptimizerBase(ProjectedHessianOptimizerMixin, Optimizer):
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

        self.hessian_update_freq = freq
        self.hessian_method = hessian_method
        self.alpha = alpha
        self.use_bfgs_update = use_bfgs_update
        self.adaptive_hessian = adaptive_hessian
        self.force_threshold_ratio = force_threshold_ratio

        if atoms.calc is None:
            raise ValueError("Atoms object must have a calculator attached")

        self.freq_analysis = FrequencyAnalysis(
            atoms=atoms,
            calculator=atoms.calc,
            delta=hessian_delta,
            verbose=verbose,
        )

        self.hessian = initial_hessian
        self.hessian_calls = 0
        self._last_full_hessian_step = -1
        self.bfgs_updates = 0
        self._last_positions: np.ndarray | None = None
        self._last_gradient: np.ndarray | None = None
        self._previous_fmax = None
        self.max_steps: int = 0
        self._scipy_result = None
        self.trust_radius = trust_radius
        self.max_trust_radius = max_trust_radius
        self.min_trust_radius = min_trust_radius
        existing_fmax = getattr(self, "fmax", None)
        self.fmax: float = 0.05 if existing_fmax is None else float(existing_fmax)  # type: ignore[assignment]

        if ts_search:
            if method == "Newton-CG":
                msg = (
                    "Newton-CG is not supported for transition-state searches. "
                    "Use 'trust-krylov', 'trust-ncg', 'trust-exact', 'rfo', or 'sella' instead."
                )
                raise ValueError(msg)
            self._transition_mode: np.ndarray | None = None
            self._rfo_alpha = 1.0
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
            logger.info(MSG_INITIALIZED_SCIPY_HESSIAN, method)
            if adaptive_hessian:
                if hessian_update_freq is None:
                    logger.info(
                        "Adaptive Hessian updates enabled (force-triggered; no periodic updates)",
                    )
                else:
                    logger.info(
                        "Adaptive Hessian updates enabled (base frequency: every %s step(s))",
                        hessian_update_freq,
                    )
            elif hessian_update_freq is None:
                logger.info("Periodic Hessian updates disabled (compute once and reuse)")
            else:
                logger.info(
                    "Fixed Hessian update frequency: every %s step(s)",
                    hessian_update_freq,
                )
            if use_bfgs_update:
                logger.info("BFGS approximate updates enabled between full Hessians")
            logger.info("Hessian calculation method: %s", hessian_method)

    def _cartesian_gradient(self, x: np.ndarray) -> np.ndarray:
        """Unscaled Cartesian gradient, ``-forces``."""
        self.atoms.set_positions(self._x_to_positions(x))
        forces = self.atoms.get_forces()
        return cast(np.ndarray, -forces.ravel())

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Gradient of objective function (negative forces)."""
        gradient = self._cartesian_gradient(x)

        if self.use_bfgs_update:
            if self._last_gradient is None:
                self._last_gradient = gradient.copy()
            if self._last_positions is None:
                self._last_positions = x.copy()

        return gradient / self.alpha

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
            self._last_full_hessian_step = self.nsteps

            if self.verbose >= 1:
                logger.info(
                    "Computing full Hessian at step %s (call #%s, reason: %s)",
                    self.nsteps,
                    self.hessian_calls,
                    reason,
                )

            self.freq_analysis.atoms = self.atoms
            self.freq_analysis.atoms.calc = self.atoms.calc

            hessian = self.freq_analysis.calculate_hessian(method=self.hessian_method)

            self.hessian = hessian

            if self.verbose >= 2:
                logger.info("Full Hessian computed (shape: %s)", hessian.shape)

            self._last_positions = current_positions.copy()
            self._last_gradient = None
            self.bfgs_updates = 0

        elif (
            self.use_bfgs_update
            and self._last_positions is not None
            and self._last_gradient is not None
        ):
            current_gradient = self._cartesian_gradient(x)
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
                    "BFGS Hessian update at step %s (#%s since last full, %s steps since full Hessian)",
                    self.nsteps,
                    self.bfgs_updates,
                    steps_since_full,
                )
            else:
                logger.debug("Skipping BFGS update (sy = %.2e too small)", sy)

            self._last_positions = current_positions.copy()
            self._last_gradient = current_gradient.copy()
        else:
            log = logger.info if self.verbose >= 1 else logger.debug
            log("Reusing Hessian (computed %s steps ago)", steps_since_full)

        if self.hessian is None:
            raise RuntimeError("Hessian is None after update logic")
        return self.hessian / self.alpha

    def _get_current_fmax(self) -> float:
        return float(np.max(np.abs(self.atoms.get_forces())))

    def objective(self, x: np.ndarray) -> float:
        self.atoms.set_positions(self._x_to_positions(x))
        return float(self.atoms.get_potential_energy() / self.alpha)

    def callback(self, x: np.ndarray, **kwargs: Any) -> None:
        self.atoms.set_positions(self._x_to_positions(x))

        if self.nsteps < self.max_steps:
            self.nsteps += 1

        forces = self.atoms.get_forces()
        self._previous_fmax = np.max(np.abs(forces))
        self.log(forces)
        self.call_observers()

        if self.converged(forces.ravel()):
            raise ConvergedError

    def _compute_ts_step(self, x: np.ndarray) -> TSStepResult:
        gradient = self._get_ts_gradient(x)

        if self._ts_step_mode == "dense":
            step_result = compute_restricted_prfo_step(
                gradient=gradient,
                hessian=self._refresh_projected_hessian(x, log_reuse=False),
                trust_radius=self.trust_radius,
                previous_mode=self._transition_mode,
                alpha=self._rfo_alpha,
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
        return step_result

    def _run_ts_search(self) -> bool:
        x = self._positions_to_x()
        self._previous_energy = float(self.atoms.get_potential_energy())

        if self.verbose >= 2:
            logger.info("Starting %s transition-state search (%s)", self.method, self._ts_step_mode)

        update_alpha = (
            (lambda alpha: setattr(self, "_rfo_alpha", alpha))
            if self._ts_step_mode == "dense"
            else None
        )

        converged, x, self.trust_radius, self._previous_energy = run_ts_optimization_loop(
            self,
            x,
            compute_step=self._compute_ts_step,
            x_to_positions=self._x_to_positions,
            atoms=self.atoms,
            options=TSDriverOptions(
                converged_error_message=MSG_TS_OPTIMIZATION_CONVERGED,
                log_converged_after_loop=False,
                log_stopped_without_converging=False,
            ),
            max_steps=self.max_steps,
            verbose=self.verbose,
            trust_radius=self.trust_radius,
            min_trust_radius=self.min_trust_radius,
            max_trust_radius=self.max_trust_radius,
            previous_energy=self._previous_energy,
            update_rfo_alpha=update_alpha,
        )
        self.atoms.set_positions(self._x_to_positions(x))
        return converged

    def run(self, fmax: float = 0.05, steps: int = 100) -> bool:  # type: ignore[override]
        self.fmax = float(fmax)
        self.max_steps = int(steps + self.nsteps)

        x0 = self._positions_to_x()

        if self.nsteps == 0:
            forces = self.atoms.get_forces()
            self.log(forces)
            self.call_observers()
            self.nsteps += 1

        if self.ts_search:
            return self._run_ts_search()

        if self.verbose >= 2:
            logger.info(MSG_STARTING_NAMED_OPTIMIZATION, self.method)
            logger.info(MSG_CONVERGENCE_CRITERION_FMAX, fmax)
            logger.info(MSG_MAXIMUM_STEPS, steps)

        try:
            options: dict[str, Any] = {"maxiter": steps, "disp": False}
            if self.method in ("trust-krylov", "trust-ncg", "trust-exact"):
                options["gtol"] = 1e-8
                options["initial_trust_radius"] = self.trust_radius
                options["max_trust_radius"] = self.max_trust_radius
                if self.method == "trust-krylov":
                    options["inexact"] = True

            # Tests patch ``famex.optimizers.scipy_optimizers.minimize``.
            from famex.optimizers import scipy_optimizers as _scipy_optimizers

            self._scipy_result = _scipy_optimizers.minimize(  # type: ignore[call-overload]
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
            log_optimization_converged(self.verbose, log=logger)
            return True

        forces = self.atoms.get_forces()
        converged = bool(self.converged(forces.ravel()))

        if converged:
            log_optimization_converged(self.verbose, log=logger)
        else:
            actual_steps = (
                self._scipy_result.nit if self._scipy_result is not None else self.nsteps - 1
            )
            scipy_message = (
                str(getattr(self._scipy_result, "message", "unknown reason"))
                if self._scipy_result is not None
                else "unknown reason"
            )
            log_scipy_trust_region_not_converged(
                self.verbose,
                log=logger,
                method=self.method,
                actual_steps=actual_steps,
                steps=steps,
                scipy_message=scipy_message,
                forces=forces,
            )

        return converged

    def get_number_of_steps(self) -> int:
        return self.nsteps

    def dump(self, data: Any) -> None:
        return None

    def load(self) -> None:
        return None
