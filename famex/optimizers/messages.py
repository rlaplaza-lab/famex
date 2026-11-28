"""User-facing optimizer log message strings."""

from __future__ import annotations

import logging

import numpy as np

MSG_OPTIMIZATION_CONVERGED = "Optimization converged!"
MSG_TS_OPTIMIZATION_CONVERGED = "Transition-state optimization converged!"
MSG_STEP_SIZE_ZERO = "Step size is zero, stopping optimization"
MSG_OPTIMIZATION_STOPPED = "Optimization stopped after %s steps without converging"
MSG_STARTING_NAMED_OPTIMIZATION = "Starting %s optimization"
MSG_CONVERGENCE_CRITERION_FMAX = "Convergence criterion: fmax = %s eV/Å"
MSG_MAXIMUM_STEPS = "Maximum steps: %s"
MSG_INITIALIZED_OPTIMIZER_VERBOSITY = "Initialized %s optimizer with verbosity control"
MSG_INITIALIZED_RFO = "Initialized RFO transition state optimizer"
MSG_STARTING_RFO = "Starting RFO transition state optimization"
MSG_INITIALIZED_SCIPY_HESSIAN = "Initialized %s optimizer with Hessian support"


def log_converged_error_caught(verbose: int, message: str, *, log: logging.Logger) -> None:
    if verbose >= 1:
        log.info(message)


def log_optimization_converged(verbose: int, *, log: logging.Logger) -> None:
    if verbose >= 1:
        log.info(MSG_OPTIMIZATION_CONVERGED)


def log_step_size_zero(verbose: int, *, log: logging.Logger) -> None:
    if verbose >= 1:
        log.warning(MSG_STEP_SIZE_ZERO)


def log_rejected_poor_ts_step(verbose: int, quality: float, *, log: logging.Logger) -> None:
    if verbose >= 1:
        log.warning("Rejected poor TS step (Q=%.4f)", quality)


def log_final_max_force(fmax: float, *, log: logging.Logger) -> None:
    log.warning("Final max force: %.6f eV/Å", fmax)


def log_scipy_trust_region_not_converged(
    verbose: int,
    *,
    log: logging.Logger,
    method: str,
    actual_steps: int,
    steps: int,
    scipy_message: str,
    forces: np.ndarray,
) -> None:
    if verbose < 1:
        return

    scipy_stopped_due_to_maxiter = (
        "Maximum number of iterations" in scipy_message or "max iterations" in scipy_message.lower()
    )

    if scipy_stopped_due_to_maxiter:
        log.warning(
            "Optimization stopped after %s trust-region steps (reached max outer iterations: %s)",
            actual_steps,
            steps,
        )
    else:
        log.warning(
            "Optimization stopped after %s trust-region steps without converging",
            actual_steps,
        )
        log.warning("  SciPy reason: %s", scipy_message)
        if "trust-krylov" in method:
            log.warning(
                "  Note: Each trust-region step uses inner Krylov iterations to solve the subproblem. "
                "Failure may indicate inner Krylov solver struggling with the subproblem "
                "(each step may perform up to ~1000 inner Krylov iterations)."
            )
    log_final_max_force(float(np.max(np.abs(forces))), log=log)
