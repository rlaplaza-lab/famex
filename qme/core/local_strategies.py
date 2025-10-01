"""Default runner strategies used by Explorer.

This module contains lightweight default implementations for minima and
transition-state runners and an optimizer lookup helper. They are kept
separate from `explorer.py` to avoid circular imports and make the
strategy implementations easier to test and replace.
"""

import warnings
from typing import Any, List

from qme.dependencies import deps


def _get_local_optimizer_class(name: str):
    """Map a short name to an ASE optimizer class or SELLA's Sella.

    SELLA is preferred when requested. This function uses the centralized
    `deps` manager to lazily import Sella to avoid import-time failures.
    """
    name = (name or "").lower()

    if name == "sella":
        if deps.has("sella"):
            return deps.get("sella")
        raise ImportError("SELLA is not available in this environment")

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
    """
    if explorer is None:
        raise ValueError("explorer must be provided to default_minima_runner")
    try:
        opt_class = _get_local_optimizer_class(local_optimizer_name)
    except Exception as e:
        warnings.warn(
            f"Could not select requested optimizer '{local_optimizer_name}': {e}"
        )
        opt_class = None
    # Accept re meither a single Atoms instance or a list of them
    single_input = False
    if not isinstance(atoms_list, (list, tuple)):
        single_input = True
        atoms_iter = [atoms_list]
    else:
        atoms_iter = atoms_list

    results = []
    for atoms in atoms_iter:
        try:
            explorer._create_and_attach_calculator(atoms)
        except Exception as e:
            warnings.warn(f"Failed to create calculator for a structure: {e}")
        explorer._apply_constraints(atoms)
        OptClass = opt_class
        opt_kwargs = getattr(explorer, "optimizer_kwargs", {}) or {}
        if local_optimizer_name.lower() == "sella":
            # Sella-specific kwargs for minima search
            opt_kwargs = dict(opt_kwargs)
            opt_kwargs.setdefault("internal", True)
            opt_kwargs.setdefault("order", 0)

        if OptClass is None:
            # Try to fall back to ASE's LBFGS/BFGS/FIRE if available
            try:
                from ase.optimize.lbfgs import LBFGS as _LBFGS  # type: ignore

                OptClass = _LBFGS
            except Exception:
                raise ImportError(
                    "No suitable optimizer available (requested: %s). "
                    "Install 'sella' or ASE optimizers." % local_optimizer_name
                )

        opt = OptClass(atoms, **opt_kwargs)
        opt.run(fmax=fmax, steps=steps)
        results.append(atoms)

    return results[0] if single_input and results else results


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
    """
    if explorer is None:
        raise ValueError("explorer must be provided to default_ts_runner")
    try:
        opt_class = _get_local_optimizer_class(local_optimizer_name)
    except Exception as e:
        warnings.warn(
            f"Could not select requested optimizer '{local_optimizer_name}': {e}"
        )
        opt_class = None
    # Accept either a single Atoms instance or a list of them
    single_input = False
    if not isinstance(atoms_list, (list, tuple)):
        single_input = True
        atoms_iter = [atoms_list]
    else:
        atoms_iter = atoms_list

    results = []
    for atoms in atoms_iter:
        try:
            explorer._create_and_attach_calculator(atoms)
        except Exception as e:
            warnings.warn(f"Failed to create calculator for a structure: {e}")
        explorer._apply_constraints(atoms)
        OptClass = opt_class
        opt_kwargs = getattr(explorer, "ts_kwargs", {}) or {}
        if local_optimizer_name.lower() == "sella":
            # Sella-specific kwargs for TS search
            opt_kwargs = dict(opt_kwargs)
            opt_kwargs.setdefault("internal", True)
            opt_kwargs.setdefault("order", 1)

        if OptClass is None:
            # Try to fall back to ASE LBFGS
            try:
                from ase.optimize.lbfgs import LBFGS as _LBFGS  # type: ignore

                OptClass = _LBFGS
            except Exception:
                raise ImportError(
                    "No suitable optimizer available for TS optimization (requested: %s)."
                    % local_optimizer_name
                )

        opt = OptClass(atoms, **opt_kwargs)
        opt.run(fmax=fmax, steps=steps)
        results.append(atoms)

    return results[0] if single_input and results else results
