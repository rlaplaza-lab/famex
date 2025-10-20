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

import numpy as np
from ase import Atoms

from qme.core.strategy import REGISTRY, BaseStrategy, StrategyMetadata
from qme.core.strategy_utils import StrategyUtils
from qme.logging_utils import get_qme_logger

logger = get_qme_logger(__name__)


def validate_ts_structure(atoms, explorer, threshold: float = 50.0) -> dict[str, Any]:
    """Validate that structure is a transition state via frequency analysis.

    Parameters
    ----------
    atoms : Atoms
        The structure to validate
    explorer : Explorer
        Explorer instance for calculator access
    threshold : float, default=50.0
        Minimum frequency magnitude in cm^-1 to consider significant

    Returns
    -------
    dict[str, Any]
        Validation results dictionary from FrequencyAnalysis.is_transition_state()
    """
    from qme.analysis.frequency import FrequencyAnalysis

    # Ensure calculator is attached
    if getattr(atoms, "calc", None) is None:
        explorer._create_and_attach_calculator(atoms)

    # Run frequency analysis
    freq_analysis = FrequencyAnalysis(atoms=atoms, calculator=atoms.calc, verbose=0)
    freq_analysis.calculate_hessian(method="auto")
    freq_analysis.diagonalize_hessian()

    # Check if structure is a TS
    validation_result = freq_analysis.is_transition_state(threshold=threshold)

    # Log warning if validation fails
    if not validation_result["is_transition_state"]:
        logger.warning(
            "TS validation failed: %s. Expected exactly 1 imaginary frequency, "
            "found %d imaginary frequencies.",
            validation_result["assessment"],
            validation_result["n_imaginary_frequencies"]
        )
    else:
        logger.info("TS validation passed: structure has exactly 1 imaginary frequency")

    return validation_result


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

    normalized_name = optimizer_name.lower()
    if normalized_name in FORBIDDEN_OPTIMIZERS_FOR_TS:
        raise ValueError(
            f"Optimizer '{optimizer_name}' is not suitable for transition state "
            "optimization. Use 'sella' or 'trust-krylov-ts' for TS searches."
        )


def _get_local_optimizer_class(name: str) -> type[Any]:
    """Map a short name to an ASE optimizer class or SELLA's Sella.

    SELLA is preferred when requested and is now a core dependency.
    All optimizers now support verbosity control through QME's logging system.
    """
    name = (name or "").lower()

    if name == "sella":
        from qme.core.ase_optimizer_wrappers import VerboseSella

        return VerboseSella

    # SciPy Hessian-based optimizers
    if name in ("trust-krylov", "trustkrylov", "trust_krylov"):
        from qme.core.scipy_optimizers import TrustKrylov

        return TrustKrylov
    if name in (
        "trust-krylov-ts",
        "trustkrylovts",
        "trust_krylov_ts",
        "trust-krylov-transition",
    ):
        from qme.core.scipy_optimizers import TrustKrylovTS

        return TrustKrylovTS
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
            from qme.core.ase_optimizer_wrappers import VerboseLBFGS

            return VerboseLBFGS
        if name in ("bfgs",):
            from qme.core.ase_optimizer_wrappers import VerboseBFGS

            return VerboseBFGS
        if name in ("fire",):
            from qme.core.ase_optimizer_wrappers import VerboseFIRE

            return VerboseFIRE
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
            result = converged_attr()
            return bool(result)
        except TypeError:
            # Some optimizers need gradient argument
            forces = atoms.get_forces()
            result = converged_attr(forces.flatten())
            return bool(result)

    return bool(converged_attr)


# =============================================================================
# Strategy Classes
# =============================================================================

class LocalMinimaStrategy(BaseStrategy):
    """Local minima optimization strategy."""

    metadata = StrategyMetadata(
        name="minima:local",
        target="minima",
        strategy="local",
        description="Local minima optimization (ASE/LBFGS or SELLA)",
        aliases=["minima", "local:minima", "local-minima"],
        requires_multiple_structures=False,
    )

    def run(self, atoms_list: list[Atoms], fmax: float = 0.05, steps: int = 1000, **kwargs) -> dict[str, Any]:
        """Run local minima optimization.

        Parameters
        ----------
        atoms_list : list[Atoms]
            List of structures to optimize
        fmax : float, default=0.05
            Force convergence threshold
        steps : int, default=1000
            Maximum optimization steps
        **kwargs
            Additional keyword arguments

        Returns
        -------
        dict[str, Any]
            Standardized result dictionary
        """
        self.validate_inputs(atoms_list)

        local_optimizer_name = kwargs.get("local_optimizer_name", "sella")
        verbose = kwargs.get("verbose", 1)

        # Log minima optimization start
        if verbose >= 1:
            logger.info(f"Starting local minima optimization with {local_optimizer_name}")
            logger.info(f"Force threshold: {fmax} eV/Å, Max steps: {steps}")

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

            self.explorer._create_and_attach_calculator(atoms_copy)
            self.explorer._apply_constraints(atoms_copy)
            opt_kwargs = getattr(self.explorer, "optimizer_kwargs", {}) or {}
            # Add verbosity control
            opt_kwargs = dict(opt_kwargs)
            opt_kwargs.setdefault("verbose", getattr(self.explorer, "verbose", 1))

            if local_optimizer_name.lower() == "sella":
                # Sella-specific kwargs for minima search
                opt_kwargs.setdefault("internal", True)
                opt_kwargs.setdefault("order", 0)
                # Check if Hessian is provided in explorer
                if hasattr(self.explorer, "initial_hessian") and self.explorer.initial_hessian is not None:
                    opt_kwargs["hessian"] = self.explorer.initial_hessian

            opt = opt_class(atoms_copy, **opt_kwargs)
            opt.run(fmax=fmax, steps=steps)

            # Get step count and convergence status using helpers
            steps_taken = StrategyUtils.get_step_count(opt)
            converged = StrategyUtils.get_convergence_status(opt, atoms_copy)

            results.append(atoms_copy)
            step_counts.append(steps_taken)
            converged_flags.append(converged)

        # Log completion
        if verbose >= 1:
            if single_input and results:
                converged = bool(converged_flags[0])
                steps_taken = step_counts[0]
                logger.info(f"Minima optimization completed: converged={converged}, steps={steps_taken}")
            else:
                total_converged = sum(converged_flags)
                total_structures = len(converged_flags)
                logger.info(f"Minima optimization completed: {total_converged}/{total_structures} structures converged")

        if single_input and results:
            return self.prepare_result(
                results[0],
                steps_taken=step_counts[0],
                converged=bool(converged_flags[0]),
            )
        else:
            return self.prepare_result(
                results,
                steps_taken=step_counts,
                converged=[bool(c) for c in converged_flags],
            )


# Register the strategy
REGISTRY.register(LocalMinimaStrategy)


class LocalTSStrategy(BaseStrategy):
    """Local transition state optimization strategy."""

    metadata = StrategyMetadata(
        name="ts:local",
        target="ts",
        strategy="local",
        description="Local transition-state optimization (SELLA preferred)",
        aliases=["ts", "local:ts", "local-ts"],
        requires_multiple_structures=False,
    )

    def run(self, atoms_list: list[Atoms], fmax: float = 0.05, steps: int = 1000, validate_ts: bool = False, **kwargs) -> dict[str, Any]:
        """Run local transition state optimization.

        Parameters
        ----------
        atoms_list : list[Atoms]
            List of structures to optimize
        fmax : float, default=0.05
            Force convergence threshold
        steps : int, default=1000
            Maximum optimization steps
        **kwargs
            Additional keyword arguments

        Returns
        -------
        dict[str, Any]
            Standardized result dictionary
        """
        self.validate_inputs(atoms_list)

        local_optimizer_name = kwargs.get("local_optimizer_name", "sella")
        verbose = kwargs.get("verbose", 1)

        # Log TS optimization start
        if verbose >= 1:
            logger.info(f"Starting local transition state optimization with {local_optimizer_name}")
            logger.info(f"Force threshold: {fmax} eV/Å, Max steps: {steps}")

        # Validate TS optimization setup - hardcoded restrictions
        _validate_ts_optimization_setup(self.explorer.backend, local_optimizer_name)

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

            self.explorer._create_and_attach_calculator(atoms_copy)
            self.explorer._apply_constraints(atoms_copy)
            opt_kwargs = getattr(self.explorer, "ts_kwargs", {}) or {}
            # Add verbosity control
            opt_kwargs = dict(opt_kwargs)
            opt_kwargs.setdefault("verbose", getattr(self.explorer, "verbose", 1))

            normalized_name = local_optimizer_name.lower()
            if normalized_name == "sella":
                # Sella-specific kwargs for TS search
                opt_kwargs.setdefault("internal", True)
                opt_kwargs.setdefault("order", 1)
                # Check if Hessian is provided in explorer
                if hasattr(self.explorer, "initial_hessian") and self.explorer.initial_hessian is not None:
                    opt_kwargs["hessian"] = self.explorer.initial_hessian
            elif normalized_name in {
                "trust-krylov-ts",
                "trustkrylovts",
                "trust_krylov_ts",
                "trust-krylov-transition",
            }:
                opt_kwargs.setdefault("hessian_update_freq", 1)
                opt_kwargs.setdefault("mode_recompute_interval", 1)
                opt_kwargs.setdefault("index_tolerance", 5e-4)
                opt_kwargs.setdefault("min_positive_eigenvalue", 4e-3)
                opt_kwargs.setdefault("negative_mode_boost", 8e-3)

            opt = opt_class(atoms_copy, **opt_kwargs)
            opt.run(fmax=fmax, steps=steps)

            # Get step count and convergence status using helpers
            steps_taken = StrategyUtils.get_step_count(opt)
            converged = StrategyUtils.get_convergence_status(opt, atoms_copy)

            results.append(atoms_copy)
            step_counts.append(steps_taken)
            converged_flags.append(converged)

        # Validate TS structures if requested
        validation_results = []
        if validate_ts:
            for atoms_copy in results:
                validation_result = validate_ts_structure(atoms_copy, self.explorer)
                validation_results.append(validation_result)
        else:
            validation_results = [None] * len(results)

        # Log completion
        if verbose >= 1:
            if single_input and results:
                converged = bool(converged_flags[0])
                steps_taken = step_counts[0]
                logger.info(f"Transition state optimization completed: converged={converged}, steps={steps_taken}")
            else:
                total_converged = sum(converged_flags)
                total_structures = len(converged_flags)
                logger.info(f"Transition state optimization completed: {total_converged}/{total_structures} structures converged")

        if single_input and results:
            result = self.prepare_result(
                results[0],
                steps_taken=step_counts[0],
                converged=bool(converged_flags[0]),
            )
            if validation_results[0] is not None:
                result["ts_validation"] = validation_results[0]
            return result
        else:
            result = self.prepare_result(
                results,
                steps_taken=step_counts,
                converged=[bool(c) for c in converged_flags],
            )
            if any(v is not None for v in validation_results):
                result["ts_validation"] = validation_results
            return result


# Register the strategy
REGISTRY.register(LocalTSStrategy)


class LocalIRCStrategy(BaseStrategy):
    """IRC (Intrinsic Reaction Coordinate) path calculation strategy."""

    metadata = StrategyMetadata(
        name="path:irc",
        target="path",
        strategy="irc",
        description="IRC (Intrinsic Reaction Coordinate) path from transition state",
        aliases=["irc", "local:irc", "local-irc"],
        requires_multiple_structures=False,
    )

    def run(self, atoms_list: list[Atoms], fmax: float = 0.05, steps: int = 100,
            step_size: float = 0.1, direction: str = "both", **kwargs) -> dict[str, Any]:
        """Run IRC path calculation.

        Parameters
        ----------
        atoms_list : list[Atoms]
            List of structures (typically single transition state)
        fmax : float, default=0.05
            Force convergence threshold for endpoint optimization
        steps : int, default=100
            Maximum number of IRC steps in each direction
        step_size : float, default=0.1
            Step size for IRC path following (in amu^1/2 * Angstrom)
        direction : str, default="both"
            Direction to follow: "forward", "backward", or "both"
        **kwargs
            Additional keyword arguments

        Returns
        -------
        dict[str, Any]
            Standardized result dictionary
        """
        self.validate_inputs(atoms_list)

        local_optimizer_name = kwargs.get("local_optimizer_name", "bfgs")
        verbose = kwargs.get("verbose", 1)

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
        self.explorer._create_and_attach_calculator(ts_atoms)
        self.explorer._apply_constraints(ts_atoms)

        # Log IRC calculation start
        if verbose >= 1:
            logger.info("Starting IRC calculation from transition state")
            logger.info(f"Direction: {direction}, Max steps: {steps}, Step size: {step_size}")
            logger.info(f"Force threshold: {fmax} eV/Å")

        # Initialize paths (empty initially, TS will be added when constructing final trajectory)
        forward_path = []
        backward_path = []

        def follow_irc_direction(
            initial_atoms: Atoms, direction_sign: float, max_steps: int
        ) -> list[Atoms]:
            """Follow IRC in one direction."""
            path = []
            current = initial_atoms.copy()
            # Make sure calculator is attached
            if current.calc is None:
                self.explorer._create_and_attach_calculator(current)
                self.explorer._apply_constraints(current)

            for _step in range(max_steps):
                # Get current forces
                current_forces = current.get_forces()
                current_masses = current.get_masses()

                # Check if we've converged (near a minimum)
                max_force = np.max(np.abs(current_forces))
                if max_force < fmax:
                    # Optimize to local minimum
                    if verbose >= 2:
                        logger.debug(f"IRC converged to minimum (max force: {max_force:.6f} eV/Å), optimizing endpoint")
                    opt_class = _get_local_optimizer_class(local_optimizer_name)
                    opt_copy = current.copy()
                    self.explorer._create_and_attach_calculator(opt_copy)
                    self.explorer._apply_constraints(opt_copy)
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
                self.explorer._create_and_attach_calculator(next_atoms)
                self.explorer._apply_constraints(next_atoms)

                path.append(next_atoms.copy())
                current = next_atoms

            return path

        # Follow IRC in requested direction(s)
        if direction.lower() in ("forward", "both"):
            if verbose >= 2:
                logger.debug("Following IRC in forward direction")
            forward_path = follow_irc_direction(ts_atoms, 1.0, steps)

        if direction.lower() in ("backward", "both"):
            if verbose >= 2:
                logger.debug("Following IRC in backward direction")
            backward_path = follow_irc_direction(ts_atoms, -1.0, steps)

        # Combine paths: backward (reversed) + ts + forward
        if direction.lower() == "both":
            trajectory = list(reversed(backward_path)) + [ts_atoms.copy()] + forward_path
        elif direction.lower() == "forward":
            trajectory = [ts_atoms.copy()] + forward_path
        elif direction.lower() == "backward":
            trajectory = list(reversed(backward_path)) + [ts_atoms.copy()]
        else:
            raise ValueError(
                f"Invalid direction: {direction}. Must be 'forward', 'backward', or 'both'"
            )

        # Log completion
        if verbose >= 1:
            logger.info(f"IRC calculation completed: {len(trajectory)} images generated")
            logger.info(f"Forward path: {len(forward_path)} images, Backward path: {len(backward_path)} images")

        return self.prepare_result(
            trajectory,
            converged=True,
            trajectory=trajectory,  # Add trajectory key for compatibility
            forward_path=forward_path,
            backward_path=backward_path,
        )


# Register the strategy
REGISTRY.register(LocalIRCStrategy)


def local_minima_runner(
    atoms_list: list[Atoms],
    fmax: float = 0.05,
    steps: int = 1000,
    explorer: Any | None = None,
    local_optimizer_name: str = "sella",
    verbose: int = 1,
    **kwargs: Any,
) -> dict[str, Any]:
    """Default minima runner delegating to LocalMinimaStrategy.

    Public API preserved; implementation delegates to the strategy class to
    avoid duplication of optimization logic.
    """
    if explorer is None:
        raise ValueError("explorer must be provided to default_minima_runner")

    strategy = LocalMinimaStrategy(explorer=explorer)
    # Delegate, excluding explorer from forwarded kwargs
    forwarded = {
        k: v
        for k, v in {
            "fmax": fmax,
            "steps": steps,
            "local_optimizer_name": local_optimizer_name,
            "verbose": verbose,
            **kwargs,
        }.items()
        if k != "explorer"
    }
    return strategy.run(atoms_list, **forwarded)


def local_ts_runner(
    atoms_list: list[Atoms],
    fmax: float = 0.05,
    steps: int = 1000,
    explorer: Any | None = None,
    local_optimizer_name: str = "sella",
    verbose: int = 1,
    **kwargs: Any,
) -> dict[str, Any]:
    """Default TS runner delegating to LocalTSStrategy."""
    if explorer is None:
        raise ValueError("explorer must be provided to default_ts_runner")

    strategy = LocalTSStrategy(explorer=explorer)
    forwarded = {
        k: v
        for k, v in {
            "fmax": fmax,
            "steps": steps,
            "local_optimizer_name": local_optimizer_name,
            "verbose": verbose,
            **kwargs,
        }.items()
        if k != "explorer"
    }
    return strategy.run(atoms_list, **forwarded)


def local_irc_runner(
    atoms_list: list[Atoms] | Atoms,
    fmax: float = 0.05,
    steps: int = 100,
    step_size: float = 0.1,
    explorer: Any | None = None,
    local_optimizer_name: str = "bfgs",
    direction: str = "both",
    verbose: int = 1,
    **kwargs: Any,
) -> dict[str, Any]:
    """IRC path calculation delegating to LocalIRCStrategy."""
    if explorer is None:
        raise ValueError("explorer must be provided to local_irc_runner")

    strategy = LocalIRCStrategy(explorer=explorer)
    forwarded = {
        k: v
        for k, v in {
            "fmax": fmax,
            "steps": steps,
            "step_size": step_size,
            "local_optimizer_name": local_optimizer_name,
            "direction": direction,
            "verbose": verbose,
            **kwargs,
        }.items()
        if k != "explorer"
    }
    return strategy.run(atoms_list, **forwarded)
