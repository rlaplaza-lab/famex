"""IRC (Intrinsic Reaction Coordinate) path calculation strategy."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any

import numpy as np
from ase import Atoms

from qme.core.base_strategy import BaseStrategy, StrategyMetadata
from qme.core.registry import REGISTRY
from qme.strategies.helpers import _get_local_optimizer_class, validate_ts_structure
from qme.utils.logging import get_qme_logger

logger = get_qme_logger(__name__)


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

    def run(
        self,
        atoms_list: Sequence[Atoms],
        fmax: float = 0.05,
        steps: int = 100,
        step_size: float = 0.1,
        direction: str = "both",
        validate_ts: bool = True,
        **kwargs: Any,
    ) -> dict[str, Atoms | list[Atoms] | bool | int | float | str]:
        """Run IRC path calculation.

        Parameters
        ----------
        atoms_list : Sequence[Atoms]
            List of structures (typically single transition state)
        fmax : float, default=0.05
            Force convergence threshold for endpoint optimization
        steps : int, default=100
            Maximum number of IRC steps in each direction
        step_size : float, default=0.1
            Step size for IRC path following (in amu^1/2 * Angstrom)
        direction : str, default="both"
            Direction to follow: "forward", "backward", or "both"
        validate_ts : bool, default=True
            Whether to validate that the starting structure is a transition state
        **kwargs : Any
            Additional keyword arguments

        Returns
        -------
        dict[str, Atoms | list[Atoms] | bool | int | float | str]
            Standardized result dictionary containing:
            - optimized_atoms: IRC path structures (list[Atoms])
            - strategy: Strategy name (str)
            - converged: Whether IRC calculation converged (bool)
            - steps_taken: Number of IRC steps taken (int)
            - direction: Direction followed (str)
            - ts_validation: Transition state validation results (dict, optional)

        """
        # Convert Sequence to list for validation
        atoms_list = list(atoms_list)
        self.validate_inputs(atoms_list)

        local_optimizer_name = kwargs.get("local_optimizer_name", "bfgs")
        verbose = kwargs.get("verbose", 1)

        # Start profiling if available
        if self.profiler is not None:
            self.profiler.snapshot_memory()

        # Handle single Atoms or list input
        if len(atoms_list) != 1:
            msg = (
                "IRC runner expects a single structure (transition state), "
                f"got {len(atoms_list)} structures"
            )
            raise ValueError(
                msg,
            )
        atoms_input = atoms_list[0]

        # Create a copy to avoid modifying the original
        ts_atoms = atoms_input.copy()

        # Attach calculator and constraints
        with self.profiler.profile_section("calculator_setup") if self.profiler else nullcontext():
            self.explorer._create_and_attach_calculator(ts_atoms)
            self.explorer._apply_constraints(ts_atoms)

        # Validate that the starting structure is a transition state (optional)
        hessian = None
        if validate_ts:
            with self.profiler.profile_section("ts_validation") if self.profiler else nullcontext():
                _validation_result, hessian = validate_ts_structure(
                    ts_atoms,
                    self.explorer,
                    threshold=50.0,
                    return_hessian=True,
                    verbose=verbose,
                )

        # Log IRC calculation start
        if verbose >= 1:
            logger.info("Starting IRC calculation from transition state")
            logger.info(f"Direction: {direction}, Max steps: {steps}, Step size: {step_size}")
            logger.info(f"Force threshold: {fmax} eV/Å")

        # Initialize paths (empty initially, TS will be added when constructing final trajectory)
        forward_path = []
        backward_path = []

        def follow_irc_direction(
            initial_atoms: Atoms,
            direction_sign: float,
            max_steps: int,
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
                        logger.debug(
                            f"IRC converged to minimum (max force: {max_force:.6f} eV/Å), optimizing endpoint",
                        )
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
                    direction_sign
                    * step_size
                    * step_direction
                    * np.sqrt(current_masses[:, np.newaxis])
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
        with self.profiler.profile_section("irc_calculation") if self.profiler else nullcontext():
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
            trajectory = [
                *list(reversed(backward_path)),
                ts_atoms.copy(),
                *forward_path,
            ]
        elif direction.lower() == "forward":
            trajectory = [ts_atoms.copy(), *forward_path]
        elif direction.lower() == "backward":
            trajectory = [*list(reversed(backward_path)), ts_atoms.copy()]
        else:
            msg = f"Invalid direction: {direction}. Must be 'forward', 'backward', or 'both'"
            raise ValueError(
                msg,
            )

        # Log completion
        if verbose >= 1:
            logger.info(f"IRC calculation completed: {len(trajectory)} images generated")
            logger.info(
                f"Forward path: {len(forward_path)} images, Backward path: {len(backward_path)} images",
            )

        result = self.prepare_result(
            trajectory,
            converged=True,
            trajectory=trajectory,
            forward_path=forward_path,
            backward_path=backward_path,
        )

        # Add information about Hessian computation
        if validate_ts and hessian is not None:
            result["hessian_computed"] = True
            result["hessian"] = hessian
        else:
            result["hessian_computed"] = False

        return self._merge_profiler_results(result)


# Register the strategy
REGISTRY.register(LocalIRCStrategy)
