"""RFO-based transition state optimizer for FAMEX.

Uses Restricted-Step Partitioned Rational Function Optimization (RS-P-RFO)
with trans/rot projection and mode-following via the shared ``ts_step`` module.
"""

from __future__ import annotations

from typing import IO, Any

import numpy as np
from ase import Atoms
from ase.optimize.optimize import Optimizer

from famex.analysis.frequency import FrequencyAnalysis
from famex.optimizers._exceptions import ConvergedError
from famex.optimizers.hessian_optimizer_base import ProjectedHessianOptimizerMixin
from famex.optimizers.messages import (
    MSG_CONVERGENCE_CRITERION_FMAX,
    MSG_INITIALIZED_RFO,
    MSG_OPTIMIZATION_CONVERGED,
    MSG_STARTING_RFO,
)
from famex.optimizers.ts_driver import TSDriverOptions, run_ts_optimization_loop
from famex.optimizers.ts_step import TSStepResult, compute_restricted_prfo_step
from famex.utils.logging import get_famex_logger

logger = get_famex_logger(__name__)

__all__ = ["ConvergedError", "RFOTransitionState"]


class RFOTransitionState(ProjectedHessianOptimizerMixin, Optimizer):
    """Restricted-Step Partitioned RFO transition state optimizer.

    Default trust radii (0.02 / 0.06 Å) tuned on full BH28 with uma-s-1p2.
    ``hessian_update_freq=1`` recomputes the projected Hessian each step.
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
        trust_radius: float = 0.02,
        max_trust_radius: float = 0.06,
        min_trust_radius: float = 0.001,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        self.verbose = verbose
        if verbose == 0:
            logfile = None

        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "profiler"}
        Optimizer.__init__(self, atoms, None, logfile, trajectory, **filtered_kwargs)

        freq = hessian_update_freq
        if freq is not None and freq <= 0:
            logger.warning("hessian_update_freq <= 0 provided; disabling periodic Hessian updates")
            freq = None

        self.hessian_update_freq = freq
        self.hessian_method = hessian_method
        self.trust_radius = trust_radius
        self.max_trust_radius = max_trust_radius
        self.min_trust_radius = min_trust_radius

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
        self._transition_mode: np.ndarray | None = None
        self._rfo_alpha = 1.0
        self._previous_energy: float | None = None
        self.fmax: float = getattr(self, "fmax", 0.05)  # type: ignore[assignment]
        self.max_steps: int = 0

        if self.verbose >= 2:
            logger.info(MSG_INITIALIZED_RFO)
            if hessian_update_freq is None:
                logger.info("Periodic Hessian updates disabled (compute once and reuse)")
            else:
                logger.info("Hessian update frequency: every %s step(s)", hessian_update_freq)
            logger.info("Initial trust radius: %.4f Å", trust_radius)

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
            logger.info(MSG_STARTING_RFO)
            logger.info(MSG_CONVERGENCE_CRITERION_FMAX, fmax)

        def compute_step(pos: np.ndarray) -> TSStepResult:
            step_result = compute_restricted_prfo_step(
                gradient=self._get_ts_gradient(pos),
                hessian=self._refresh_projected_hessian(pos, log_reuse=True),
                trust_radius=self.trust_radius,
                previous_mode=self._transition_mode,
                alpha=self._rfo_alpha,
            )
            self._transition_mode = step_result.transition_mode
            return step_result

        converged, x, self.trust_radius, self._previous_energy = run_ts_optimization_loop(
            self,
            x,
            compute_step=compute_step,
            x_to_positions=self._x_to_positions,
            atoms=self.atoms,
            options=TSDriverOptions(
                converged_error_message=MSG_OPTIMIZATION_CONVERGED,
                log_converged_after_loop=True,
                log_stopped_without_converging=True,
            ),
            max_steps=self.max_steps,
            verbose=self.verbose,
            trust_radius=self.trust_radius,
            min_trust_radius=self.min_trust_radius,
            max_trust_radius=self.max_trust_radius,
            previous_energy=self._previous_energy,
            update_rfo_alpha=lambda alpha: setattr(self, "_rfo_alpha", alpha),
            stop_message_steps=steps,
        )
        self.atoms.set_positions(self._x_to_positions(x))
        return converged

    def get_number_of_steps(self) -> int:
        return self.nsteps

    def dump(self, data: Any) -> None:
        return None

    def load(self) -> None:
        return None
