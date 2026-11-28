"""Shared transition-state optimization loop for Hessian-based optimizers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from famex.optimizers._exceptions import ConvergedError
from famex.optimizers.messages import (
    MSG_OPTIMIZATION_CONVERGED,
    MSG_OPTIMIZATION_STOPPED,
    log_converged_error_caught,
    log_final_max_force,
    log_rejected_poor_ts_step,
    log_step_size_zero,
)
from famex.optimizers.ts_step import TSStepResult, adjust_trust_radius, compute_step_quality
from famex.utils.logging import get_famex_logger

if TYPE_CHECKING:
    from ase import Atoms
    from ase.optimize.optimize import Optimizer

logger = get_famex_logger(__name__)

__all__ = ["TSDriverOptions", "run_ts_optimization_loop"]


@dataclass(frozen=True)
class TSDriverOptions:
    """Flags for RFO vs SciPy TS logging differences."""

    converged_error_message: str = MSG_OPTIMIZATION_CONVERGED
    log_converged_after_loop: bool = False
    log_stopped_without_converging: bool = False


def run_ts_optimization_loop(
    optimizer: Optimizer,
    x: np.ndarray,
    *,
    compute_step: Callable[[np.ndarray], TSStepResult],
    x_to_positions: Callable[[np.ndarray], np.ndarray],
    atoms: Atoms,
    options: TSDriverOptions,
    max_steps: int,
    verbose: int,
    trust_radius: float,
    min_trust_radius: float,
    max_trust_radius: float,
    previous_energy: float | None,
    update_rfo_alpha: Callable[[float], None] | None = None,
    stop_message_steps: int | None = None,
) -> tuple[bool, np.ndarray, float, float | None]:
    """Run the TS trust-region loop; returns (converged, x, trust_radius, previous_energy)."""
    prev_energy = previous_energy
    current_trust = trust_radius

    try:
        while optimizer.nsteps < max_steps:
            step_result = compute_step(x)

            if update_rfo_alpha is not None:
                update_rfo_alpha(step_result.alpha)

            step = step_result.step
            step_norm = float(np.linalg.norm(step))
            if step_norm < 1e-12:
                log_step_size_zero(verbose, log=logger)
                break

            x_new = x + step
            atoms.set_positions(x_to_positions(x_new))
            new_energy = float(atoms.get_potential_energy())
            baseline = prev_energy if prev_energy is not None else new_energy
            energy_change = new_energy - baseline
            quality = compute_step_quality(energy_change, step_result.predicted_energy_change)

            current_trust = adjust_trust_radius(
                current_trust,
                quality,
                step_norm,
                min_trust_radius,
                max_trust_radius,
            )

            reject = quality < -1.0 and abs(energy_change) > 10.0
            if not reject:
                x = x_new
                prev_energy = new_energy
            else:
                atoms.set_positions(x_to_positions(x))
                log_rejected_poor_ts_step(verbose, quality, log=logger)

            optimizer.nsteps += 1
            forces = atoms.get_forces()
            optimizer.log(forces)
            optimizer.call_observers()

            if optimizer.converged(forces.ravel()):
                raise ConvergedError

    except ConvergedError:
        log_converged_error_caught(verbose, options.converged_error_message, log=logger)
        return True, x, current_trust, prev_energy

    forces = atoms.get_forces()
    converged = bool(optimizer.converged(forces.ravel()))

    if converged:
        if options.log_converged_after_loop:
            log_converged_error_caught(verbose, options.converged_error_message, log=logger)
        return True, x, current_trust, prev_energy

    if options.log_stopped_without_converging and verbose >= 1:
        reported_steps = stop_message_steps if stop_message_steps is not None else max_steps
        logger.warning(MSG_OPTIMIZATION_STOPPED, reported_steps)
        log_final_max_force(float(np.max(np.abs(forces))), log=logger)

    return converged, x, current_trust, prev_energy
