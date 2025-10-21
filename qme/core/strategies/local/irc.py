"""IRC (Intrinsic Reaction Coordinate) path calculation strategy."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
from ase import Atoms

from qme.core.strategies.local.helpers import _get_local_optimizer_class
from qme.core.strategy import REGISTRY, BaseStrategy, StrategyMetadata
from qme.logging_utils import get_qme_logger

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
        atoms_list: list[Atoms],
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
        validate_ts : bool, default=True
            Whether to validate that the starting structure is a transition state
        **kwargs : Any
            Additional keyword arguments

        Returns
        -------
        dict[str, Union[Atoms, list[Atoms], bool, int, float, str]]
            Standardized result dictionary containing:
            - optimized_atoms: IRC path structures (list[Atoms])
            - strategy: Strategy name (str)
            - converged: Whether IRC calculation converged (bool)
            - steps_taken: Number of IRC steps taken (int)
            - direction: Direction followed (str)
            - ts_validation: Transition state validation results (dict, optional)
        """
        self.validate_inputs(atoms_list)

        local_optimizer_name = kwargs.get("local_optimizer_name", "bfgs")
        verbose = kwargs.get("verbose", 1)

        # Start profiling if available
        if self.profiler is not None:
            self.profiler.snapshot_memory()

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
        with self.profiler.profile_section("calculator_setup") if self.profiler else nullcontext():
            self.explorer._create_and_attach_calculator(ts_atoms)
            self.explorer._apply_constraints(ts_atoms)

        # Validate that the starting structure is a transition state (optional)
        hessian = None
        if validate_ts:
            with self.profiler.profile_section("ts_validation") if self.profiler else nullcontext():
                hessian = self._validate_transition_state(ts_atoms, verbose)

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
                        logger.debug(
                            f"IRC converged to minimum (max force: {max_force:.6f} eV/Å), optimizing endpoint"
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
            logger.info(
                f"Forward path: {len(forward_path)} images, Backward path: {len(backward_path)} images"
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

    def _validate_transition_state(self, atoms: Atoms, verbose: int) -> np.ndarray | None:
        """Validate that the starting structure is a transition state using frequency analysis.

        Parameters
        ----------
        atoms : Atoms
            Structure to validate as transition state
        verbose : int
            Verbosity level for logging

        Returns
        -------
        np.ndarray | None
            Computed Hessian matrix if validation succeeds, None if validation fails
        """
        try:
            # Import frequency analysis
            from qme.analysis.frequency import FrequencyAnalysis

            # Perform frequency analysis to check if structure is a transition state
            freq_analysis = FrequencyAnalysis(atoms=atoms, calculator=atoms.calc, verbose=0)
            freq_analysis.calculate_hessian(method="auto")
            freq_analysis.diagonalize_hessian()

            # Check if it's a transition state
            ts_analysis = freq_analysis.is_transition_state(threshold=50.0)

            if not ts_analysis["is_transition_state"]:
                # Log warning about non-transition state
                n_imaginary = ts_analysis["n_imaginary_frequencies"]
                assessment = ts_analysis["assessment"]

                if verbose >= 1:
                    logger.warning(
                        f"WARNING: Starting structure does not appear to be a transition state. "
                        f"Found {n_imaginary} imaginary frequency(ies). Assessment: {assessment}. "
                        f"IRC calculation will proceed but results may be unreliable."
                    )

                if verbose >= 2:
                    imaginary_freqs = ts_analysis["imaginary_frequencies"]
                    if imaginary_freqs:
                        logger.warning(f"Imaginary frequencies (cm^-1): {[f'{f:.1f}' for f in imaginary_freqs]}")

            elif verbose >= 2:
                # Log confirmation for valid transition state
                logger.info("✓ Starting structure validated as transition state (1 imaginary frequency)")

            # Return the computed Hessian for potential reuse
            return freq_analysis._hessian

        except Exception as e:
            # If frequency analysis fails, log warning but continue
            if verbose >= 1:
                logger.warning(
                    f"Could not validate transition state: {e}. "
                    f"IRC calculation will proceed without validation."
                )
            return None


# Register the strategy
REGISTRY.register(LocalIRCStrategy)
