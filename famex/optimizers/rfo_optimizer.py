"""RFO-based transition state optimizer for FAMEX.

Uses Restricted-Step Partitioned Rational Function Optimization (RS-P-RFO)
with trans/rot projection and mode-following via the shared ``ts_step`` module.
"""

from __future__ import annotations

from typing import IO, Any, cast

import numpy as np
from ase import Atoms
from ase.optimize.optimize import Optimizer

from famex.optimizers.ts_step import (
    adjust_trust_radius,
    build_translation_rotation_basis,
    compute_dense_prfo_step,
    compute_step_quality,
    prepare_ts_hessian,
    project_gradient,
)
from famex.utils.logging import get_famex_logger

logger = get_famex_logger(__name__)


class ConvergedError(Exception):
    """Exception raised when optimizer has converged."""


class RFOTransitionState(Optimizer):
    """Restricted-Step Partitioned RFO transition state optimizer.

    Default trust radii (0.1 / 0.3 Å) are larger than geomeTRIC TS defaults to suit
    ML potentials. ``hessian_update_freq=1`` recomputes the projected Hessian each step.
    """

    def __init__(
        self,
        atoms: Atoms,
        logfile: IO | str | None = "-",
        trajectory: str | None = None,
        hessian_update_freq: int | None = 1,
        hessian_method: str = "auto",
        hessian_delta: float = 0.01,
        initial_hessian: np.ndarray | None = None,
        trust_radius: float = 0.1,
        max_trust_radius: float = 0.3,
        min_trust_radius: float = 0.001,
        alpha: float = 1.0,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize RFO transition state optimizer."""
        self.verbose = verbose
        if verbose == 0:
            logfile = None

        restart = None
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "profiler"}
        Optimizer.__init__(self, atoms, restart, logfile, trajectory, **filtered_kwargs)

        freq = hessian_update_freq
        if freq is not None and freq <= 0:
            logger.warning("hessian_update_freq <= 0 provided; disabling periodic Hessian updates")
            freq = None

        self.hessian_update_freq: int | None = freq
        self.hessian_method = hessian_method
        self.trust_radius = trust_radius
        self.max_trust_radius = max_trust_radius
        self.min_trust_radius = min_trust_radius
        self.alpha = alpha

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
        self._last_full_hessian_step = -1
        self._transition_mode: np.ndarray | None = None
        self._previous_energy: float | None = None
        self._step_quality_history: list[float] = []

        if not hasattr(self, "fmax"):
            self.fmax = 0.05
        self.fmax: float = getattr(self, "fmax", 0.05)  # type: ignore[assignment]
        self.max_steps: int = 0

        if self.verbose >= 2:
            logger.info("Initialized RFO transition state optimizer")
            if hessian_update_freq is None:
                logger.info("Periodic Hessian updates disabled (compute once and reuse)")
            else:
                logger.info(f"Hessian update frequency: every {hessian_update_freq} step(s)")
            logger.info(f"Initial trust radius: {trust_radius:.4f} Å")

    def _positions_to_x(self, atoms: Atoms | None = None) -> np.ndarray:
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise RuntimeError("Atoms object is not initialized")
        return cast(np.ndarray, atoms.get_positions().ravel())

    def _x_to_positions(self, x: np.ndarray) -> np.ndarray:
        return cast(np.ndarray, x.reshape(-1, 3))

    def _get_gradient(self, x: np.ndarray) -> np.ndarray:
        self.atoms.set_positions(self._x_to_positions(x))
        self.force_calls += 1
        forces = self.atoms.get_forces()
        gradient = -forces.ravel()
        basis = build_translation_rotation_basis(
            self.atoms.get_positions(),
            self.atoms.get_masses(),
        )
        return project_gradient(gradient, basis)

    def _compute_hessian(self, x: np.ndarray) -> np.ndarray:
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
        elif self.verbose >= 2:
            logger.debug(f"Reusing Hessian (computed {steps_since_full} steps ago)")

        if self.hessian is None:
            raise RuntimeError("Hessian is None after update logic")
        return self.hessian

    def run(self, fmax: float = 0.05, steps: int = 100) -> bool:
        """Run the RFO optimization."""
        self.fmax = float(fmax)
        self.max_steps = int(steps + self.nsteps)
        x = self._positions_to_x()
        initial_energy = self.atoms.get_potential_energy()

        if self.nsteps == 0:
            self._previous_energy = initial_energy
            forces = self.atoms.get_forces()
            self.log(forces)
            self.call_observers()
            self.nsteps += 1

        if self.verbose >= 2:
            logger.info("Starting RFO transition state optimization")
            logger.info(f"Convergence criterion: fmax = {fmax} eV/Å")

        try:
            while self.nsteps < self.max_steps:
                gradient = self._get_gradient(x)
                hessian = self._compute_hessian(x)

                step_result = compute_dense_prfo_step(
                    gradient=gradient,
                    hessian=hessian,
                    trust_radius=self.trust_radius,
                    previous_mode=self._transition_mode,
                    alpha=self.alpha,
                )
                self.alpha = step_result.alpha
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
                self._step_quality_history.append(quality)

                self.trust_radius = adjust_trust_radius(
                    self.trust_radius,
                    quality,
                    step_norm,
                    self.min_trust_radius,
                    self.max_trust_radius,
                )

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
                logger.info("Optimization converged!")
            return True

        forces = self.atoms.get_forces()
        converged: bool = bool(self.converged(forces.ravel()))
        if converged and self.verbose >= 1:
            logger.info("Optimization converged!")
        elif self.verbose >= 1:
            logger.warning(f"Optimization stopped after {steps} steps without converging")
            logger.warning(f"Final max force: {np.max(np.abs(forces)):.6f} eV/Å")

        return converged

    def get_number_of_steps(self) -> int:
        return self.nsteps

    def dump(self, data: Any) -> None:
        pass

    def load(self) -> None:
        pass
