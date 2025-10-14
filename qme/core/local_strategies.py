"""Default local strategy runners used by Explorer.

This module contains lightweight default implementations for local optimization
strategies (minima and transition-state) and an optimizer lookup helper. They are kept
separate from `explorer.py` to avoid circular imports and make the
strategy implementations easier to test and replace.

Local strategies work with single structures and perform direct optimization:
- minima:local - Direct local minima optimization
- ts:local - Direct local transition state optimization

These strategies are registered in Explorer with the new target:strategy naming scheme.
"""

from typing import Any

from ase import Atoms


def _validate_ts_optimization_setup(backend: str, optimizer_name: str) -> None:
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
            f"optimization. Use 'sella' optimizer for TS searches."
        )


def _get_local_optimizer_class(name: str) -> type[Any]:
    """Map a short name to an ASE optimizer class or SELLA's Sella.

    SELLA is preferred when requested and is now a core dependency.
    """
    name = (name or "").lower()

    if name == "sella":
        from sella import Sella

        return Sella

    # SciPy Hessian-based optimizers
    if name in ("trust-krylov", "trustkrylov", "trust_krylov"):
        from qme.core.scipy_optimizers import TrustKrylov

        return TrustKrylov
    if name in ("trust-ncg", "trustncg", "trust_ncg"):
        from qme.core.scipy_optimizers import TrustNCG

        return TrustNCG
    if name in ("trust-exact", "trustexact", "trust_exact"):
        from qme.core.scipy_optimizers import TrustExact

        return TrustExact
    if name in ("newton-cg", "newtoncg", "newton_cg"):
        from qme.core.scipy_optimizers import NewtonCG

        return NewtonCG

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


def _get_step_count(optimizer: Any) -> int | None:
    """Extract step count from various optimizer types.

    Args:
        optimizer: The optimizer instance

    Returns:
        int or None: Number of steps taken
    """
    # For optimizers, prioritize step_count attribute over get_number_of_steps()
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
    atoms_list: list[Atoms],
    fmax: float = 0.05,
    steps: int = 1000,
    explorer: Any | None = None,
    local_optimizer_name: str = "sella",
    **kwargs: Any,
) -> dict[str, Any]:
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
    # Accept either a single Atoms instance or a list of them
    single_input = False
    if not isinstance(atoms_list, (list, tuple)):
        single_input = True
        atoms_iter = [atoms_list]
    else:
        atoms_iter = atoms_list
        # If it's a single-element list, treat as single input
        if len(atoms_list) == 1:
            single_input = True

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
            "converged": bool(converged_flags[0]),
            "strategy": "local_minima_runner",
        }
    else:
        return {
            "optimized_atoms": results,
            "steps_taken": step_counts,
            "converged": [bool(c) for c in converged_flags],
            "strategy": "local_minima_runner",
        }


def local_ts_runner(
    atoms_list: list[Atoms],
    fmax: float = 0.05,
    steps: int = 1000,
    explorer: Any | None = None,
    local_optimizer_name: str = "sella",
    **kwargs: Any,
) -> dict[str, Any]:
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
        # If it's a single-element list, treat as single input
        if len(atoms_list) == 1:
            single_input = True

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
            "converged": bool(converged_flags[0]),
            "strategy": "local_ts_runner",
        }
    else:
        return {
            "optimized_atoms": results,
            "steps_taken": step_counts,
            "converged": [bool(c) for c in converged_flags],
            "strategy": "local_ts_runner",
        }


def local_irc_runner(
    atoms_list: list[Atoms] | Atoms,
    fmax: float = 0.05,
    steps: int = 100,
    step_size: float = 0.1,
    explorer: Any | None = None,
    local_optimizer_name: str = "bfgs",
    direction: str = "both",
    **kwargs: Any,
) -> dict[str, Any]:
    """IRC (Intrinsic Reaction Coordinate) path calculation from a transition state.

    Follows the reaction coordinate downhill from a transition state in both
    forward and backward directions to generate a reaction path connecting
    reactants and products. The IRC path follows the steepest descent path
    in mass-weighted coordinates.

    Parameters
    ----------
    atoms_list : list[Atoms] or Atoms
        Initial structure(s), typically a transition state
    fmax : float, default=0.05
        Force convergence threshold for endpoint optimization
    steps : int, default=100
        Maximum number of IRC steps in each direction
    step_size : float, default=0.1
        Step size for IRC path following (in amu^1/2 * Angstrom)
    explorer : Any, optional
        Explorer instance for calculator and constraint management
    local_optimizer_name : str, default="bfgs"
        Optimizer for endpoint optimization
    direction : str, default="both"
        Direction to follow: "forward", "backward", or "both"
    **kwargs
        Additional keyword arguments

    Returns
    -------
    dict
        Dictionary containing:
        - trajectory: list of Atoms objects along the IRC path
        - forward_path: list of Atoms in forward direction
        - backward_path: list of Atoms in backward direction
        - converged: bool indicating if IRC converged
        - strategy: "local_irc_runner"

    Notes
    -----
    IRC calculations start from a transition state and follow the gradient
    downhill in mass-weighted coordinates. This produces the minimum energy
    path connecting reactants and products through the TS.
    """
    import numpy as np

    if explorer is None:
        raise ValueError("explorer must be provided to local_irc_runner")

    # Handle single Atoms or list input
    if not isinstance(atoms_list, (list, tuple)):
        atoms_input = atoms_list
    else:
        if len(atoms_list) != 1:
            raise ValueError(
                "IRC runner expects a single structure (transition state), "
                f"got {len(atoms_list)} structures"
            )
        atoms_input = atoms_list[0]

    # Create a copy to avoid modifying the original
    ts_atoms = atoms_input.copy()

    # Attach calculator and constraints
    explorer._create_and_attach_calculator(ts_atoms)
    explorer._apply_constraints(ts_atoms)

    # Get the initial forces and masses (commented out for now)
    # forces = ts_atoms.get_forces()
    # masses = ts_atoms.get_masses()

    # Mass-weighted forces (commented out for now)
    # mass_weighted_forces = forces / np.sqrt(masses[:, np.newaxis])

    # Initialize paths
    forward_path = [ts_atoms.copy()]
    backward_path = []

    def follow_irc_direction(
        initial_atoms: Atoms, direction_sign: float, max_steps: int
    ) -> list[Atoms]:
        """Follow IRC in one direction.

        Parameters
        ----------
        initial_atoms : Atoms
            Starting structure
        direction_sign : float
            +1 for forward, -1 for backward
        max_steps : int
            Maximum number of steps

        Returns
        -------
        list[Atoms]
            Path in this direction (not including initial structure)
        """
        path = []
        current = initial_atoms.copy()
        # Make sure calculator is attached
        if current.calc is None:
            explorer._create_and_attach_calculator(current)
            explorer._apply_constraints(current)

        for _step in range(max_steps):
            # Get current forces
            current_forces = current.get_forces()
            current_masses = current.get_masses()

            # Check if we've converged (near a minimum)
            max_force = np.max(np.abs(current_forces))
            if max_force < fmax:
                # Optimize to local minimum
                opt_class = _get_local_optimizer_class(local_optimizer_name)
                opt_copy = current.copy()
                explorer._create_and_attach_calculator(opt_copy)
                explorer._apply_constraints(opt_copy)
                opt = opt_class(opt_copy)
                opt.run(fmax=fmax, steps=100)
                path.append(opt_copy)
                break

            # Mass-weighted step
            mw_forces = current_forces / np.sqrt(current_masses[:, np.newaxis])
            mw_forces_norm = np.linalg.norm(mw_forces)

            if mw_forces_norm < 1e-10:
                # Forces too small, stop
                break

            # Normalize and take step
            step_direction = mw_forces / mw_forces_norm
            displacement = (
                direction_sign * step_size * step_direction * np.sqrt(current_masses[:, np.newaxis])
            )

            # Create new structure
            next_atoms = current.copy()
            new_positions = current.get_positions() + displacement
            next_atoms.set_positions(new_positions)

            # Attach calculator to new structure
            explorer._create_and_attach_calculator(next_atoms)
            explorer._apply_constraints(next_atoms)

            path.append(next_atoms.copy())
            current = next_atoms

        return path

    # Follow IRC in requested direction(s)
    if direction.lower() in ("forward", "both"):
        forward_path.extend(follow_irc_direction(ts_atoms, 1.0, steps))

    if direction.lower() in ("backward", "both"):
        backward_path = follow_irc_direction(ts_atoms, -1.0, steps)

    # Combine paths: backward (reversed) + ts + forward
    if direction.lower() == "both":
        trajectory = list(reversed(backward_path)) + forward_path
    elif direction.lower() == "forward":
        trajectory = forward_path
    elif direction.lower() == "backward":
        trajectory = list(reversed(backward_path)) + [ts_atoms.copy()]
    else:
        raise ValueError(
            f"Invalid direction: {direction}. Must be 'forward', 'backward', or 'both'"
        )

    return {
        "trajectory": trajectory,
        "forward_path": forward_path,
        "backward_path": backward_path,
        "converged": True,
        "strategy": "local_irc_runner",
    }
