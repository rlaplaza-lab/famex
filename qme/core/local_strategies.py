"""Default runner strategies used by Explorer.

This module contains lightweight default implementations for minima and
transition-state runners and an optimizer lookup helper. They are kept
separate from `explorer.py` to avoid circular imports and make the
strategy implementations easier to test and replace.
"""

import warnings
from typing import Any, List, Optional

from qme.dependencies import deps


def _validate_ts_optimization_setup(backend: str, optimizer_name: str):
    """Validate that TS optimization is using appropriate calculator and optimizer.

    This function hardcodes restrictions to prevent using mock calculators or
    unsuitable optimizers for transition state optimization tasks.

    Parameters
    ----------
    backend : str
        The calculator backend being used
    optimizer_name : str
        The optimizer being used

    Raises
    ------
    ValueError
        If the setup is unsuitable for TS optimization
    """
    # Hardcoded restrictions for TS optimization
    FORBIDDEN_BACKENDS_FOR_TS = {"mock"}
    FORBIDDEN_OPTIMIZERS_FOR_TS = {"lbfgs", "l-bfgs", "l_bfgs", "bfgs", "fire"}

    if backend.lower() in FORBIDDEN_BACKENDS_FOR_TS:
        raise ValueError(
            f"Backend '{backend}' is not suitable for transition state optimization. "
            f"Use a real ML potential backend (uma, aimnet2, mace, so3lr) instead."
        )

    if optimizer_name.lower() in FORBIDDEN_OPTIMIZERS_FOR_TS:
        raise ValueError(
            f"Optimizer '{optimizer_name}' is not suitable for transition state "
            f"optimization. Use 'sella' or 'geometric' optimizers for TS searches."
        )


def _get_local_optimizer_class(name: str):
    """Map a short name to an ASE optimizer class or SELLA's Sella.

    SELLA is preferred when requested and is now a core dependency.
    """
    name = (name or "").lower()

    if name == "sella":
        from sella import Sella

        return Sella

    if name == "geometric":
        from qme.core.geometric_interface import GeometricOptimizer

        return GeometricOptimizer

    try:
        if name in ("lbfgs", "l-bfgs", "l_bfgs"):
            from ase.optimize.lbfgs import LBFGS

            return LBFGS
        if name in ("bfgs",):
            from ase.optimize import BFGS

            return BFGS
        if name in ("fire",):
            from ase.optimize import FIRE

            return FIRE
    except Exception as e:  # pragma: no cover - ASE optional in some envs
        raise ImportError(f"Requested optimizer '{name}' is not available: {e}")

    raise ValueError(f"Unknown optimizer name: {name}")


def _get_step_count(optimizer) -> Optional[int]:
    """Extract step count from various optimizer types.

    Args:
        optimizer: The optimizer instance

    Returns:
        int or None: Number of steps taken
    """
    # For GeometricOptimizer, prioritize step_count attribute over get_number_of_steps()
    # because get_number_of_steps() returns 0 by default from ASE Optimizer base class
    if hasattr(optimizer, "step_count") and optimizer.step_count is not None:
        return optimizer.step_count

    # Fallback to ASE's get_number_of_steps() for other optimizers
    if hasattr(optimizer, "get_number_of_steps"):
        return optimizer.get_number_of_steps()

    return None


def _get_convergence_status(optimizer, atoms) -> bool:
    """Extract convergence status from various optimizer types.

    Args:
        optimizer: The optimizer instance
        atoms: ASE Atoms object

    Returns:
        bool: True if converged, False otherwise
    """
    converged_attr = getattr(optimizer, "converged", None)

    if callable(converged_attr):
        try:
            return converged_attr()
        except TypeError:
            # Some optimizers need gradient argument
            forces = atoms.get_forces()
            return converged_attr(forces.flatten())

    return bool(converged_attr)


def local_minima_runner(
    atoms_list: List[Any],
    fmax=0.05,
    steps=1000,
    explorer=None,
    local_optimizer_name="sella",
    **kwargs,
):
    """Default minima runner.

    The runner uses the explorer helpers to attach calculators and
    constraints. It selects a sensible local optimizer based on
    `explorer.optimizer_name` and falls back to LBFGS/BFGS/FIRE as needed.

    Returns
    -------
    dict
        Dictionary containing optimized atoms and step count information
    """
    if explorer is None:
        raise ValueError("explorer must be provided to default_minima_runner")
    opt_class = _get_local_optimizer_class(local_optimizer_name)
    # Accept re meither a single Atoms instance or a list of them
    single_input = False
    if not isinstance(atoms_list, (list, tuple)):
        single_input = True
        atoms_iter = [atoms_list]
    else:
        atoms_iter = atoms_list

    results = []
    step_counts = []
    converged_flags = []

    for atoms in atoms_iter:
        # CRITICAL FIX: Make a copy of atoms before optimization to prevent in-place modifications
        # that corrupt the coordinate system for subsequent Hessian calculations
        atoms_copy = atoms.copy()

        explorer._create_and_attach_calculator(atoms_copy)
        explorer._apply_constraints(atoms_copy)
        opt_kwargs = getattr(explorer, "optimizer_kwargs", {}) or {}
        if local_optimizer_name.lower() == "sella":
            # Sella-specific kwargs for minima search
            opt_kwargs = dict(opt_kwargs)
            opt_kwargs.setdefault("internal", True)
            opt_kwargs.setdefault("order", 0)
        elif local_optimizer_name.lower() == "geometric":
            # geomeTRIC-specific kwargs for minima search
            opt_kwargs = dict(opt_kwargs)
            opt_kwargs.setdefault("order", 0)
            # Check if Hessian is provided in explorer
            if hasattr(explorer, "initial_hessian") and explorer.initial_hessian is not None:
                opt_kwargs["hessian"] = explorer.initial_hessian

        opt = opt_class(atoms_copy, **opt_kwargs)
        opt.run(fmax=fmax, steps=steps)

        # Get step count and convergence status using helpers
        steps_taken = _get_step_count(opt)
        converged = _get_convergence_status(opt, atoms_copy)

        results.append(atoms_copy)
        step_counts.append(steps_taken)
        converged_flags.append(converged)

    if single_input and results:
        return {
            "optimized_atoms": results[0],
            "steps_taken": step_counts[0],
            "converged": converged_flags[0],
        }
    else:
        return {
            "optimized_atoms": results,
            "steps_taken": step_counts,
            "converged": converged_flags,
        }


def local_ts_runner(
    atoms_list: List[Any],
    fmax=0.05,
    steps=1000,
    explorer=None,
    local_optimizer_name="sella",
    **kwargs,
):
    """Default TS runner.

    Uses the explorer helpers to attach calculators and constraints before
    running the chosen optimizer.

    Returns
    -------
    dict
        Dictionary containing optimized atoms and step count information
    """
    if explorer is None:
        raise ValueError("explorer must be provided to default_ts_runner")

    # Validate TS optimization setup - hardcoded restrictions
    _validate_ts_optimization_setup(explorer.backend, local_optimizer_name)

    opt_class = _get_local_optimizer_class(local_optimizer_name)
    # Accept either a single Atoms instance or a list of them
    single_input = False
    if not isinstance(atoms_list, (list, tuple)):
        single_input = True
        atoms_iter = [atoms_list]
    else:
        atoms_iter = atoms_list

    results = []
    step_counts = []
    converged_flags = []

    for atoms in atoms_iter:
        # CRITICAL FIX: Make a copy of atoms before optimization to prevent in-place modifications
        # that corrupt the coordinate system for subsequent Hessian calculations
        atoms_copy = atoms.copy()

        explorer._create_and_attach_calculator(atoms_copy)
        explorer._apply_constraints(atoms_copy)
        opt_kwargs = getattr(explorer, "ts_kwargs", {}) or {}
        if local_optimizer_name.lower() == "sella":
            # Sella-specific kwargs for TS search
            opt_kwargs = dict(opt_kwargs)
            opt_kwargs.setdefault("internal", True)
            opt_kwargs.setdefault("order", 1)
        elif local_optimizer_name.lower() == "geometric":
            # geomeTRIC-specific kwargs for TS search
            opt_kwargs = dict(opt_kwargs)
            opt_kwargs.setdefault("order", 1)
            # Check if Hessian is provided in explorer
            if hasattr(explorer, "initial_hessian") and explorer.initial_hessian is not None:
                opt_kwargs["hessian"] = explorer.initial_hessian

        opt = opt_class(atoms_copy, **opt_kwargs)
        opt.run(fmax=fmax, steps=steps)

        # Get step count and convergence status using helpers
        steps_taken = _get_step_count(opt)
        converged = _get_convergence_status(opt, atoms_copy)

        results.append(atoms_copy)
        step_counts.append(steps_taken)
        converged_flags.append(converged)

    if single_input and results:
        return {
            "optimized_atoms": results[0],
            "steps_taken": step_counts[0],
            "converged": converged_flags[0],
        }
    else:
        return {
            "optimized_atoms": results,
            "steps_taken": step_counts,
            "converged": converged_flags,
        }
