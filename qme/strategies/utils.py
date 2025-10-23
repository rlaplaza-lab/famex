"""Shared utilities for strategy implementations.

This module provides common helper functions used across different strategies,
consolidating duplicate code and providing a consistent interface.
"""

from typing import Any

import numpy as np
from ase import Atoms

from qme.utils.logging import get_qme_logger

logger = get_qme_logger(__name__)


class StrategyUtils:
    """Shared utilities for strategy implementations."""

    @staticmethod
    def ensure_charge_spin_info(atoms: Atoms, charge: int = 0, spin: int = 1) -> None:
        """Ensure atoms.info has charge and spin information.

        This prevents warnings from backends that expect this information.

        Parameters
        ----------
        atoms : Atoms
            Structure to update
        charge : int, default=0
            Default charge if not present
        spin : int, default=1
            Default spin if not present

        """
        if hasattr(atoms, "info") and atoms.info is not None:
            atoms.info.setdefault("charge", charge)
            atoms.info.setdefault("spin", spin)

    @staticmethod
    def get_step_count(optimizer: Any) -> int | None:
        """Extract step count from various optimizer types.

        Parameters
        ----------
        optimizer : Any
            The optimizer instance

        Returns:
        -------
        int or None
            Number of steps taken, or None if not available

        """
        # For optimizers, prioritize step_count attribute over get_number_of_steps()
        # because get_number_of_steps() returns 0 by default from ASE Optimizer base class
        if hasattr(optimizer, "step_count") and optimizer.step_count is not None:
            return optimizer.step_count

        # Fallback to ASE's get_number_of_steps() for other optimizers
        if hasattr(optimizer, "get_number_of_steps"):
            return optimizer.get_number_of_steps()

        return None

    @staticmethod
    def get_convergence_status(optimizer: Any, atoms: Atoms) -> bool:
        """Extract convergence status from various optimizer types.

        Parameters
        ----------
        optimizer : Any
            The optimizer instance
        atoms : Atoms
            ASE Atoms object

        Returns:
        -------
        bool
            True if converged, False otherwise

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

    @staticmethod
    def check_batch_support(calculator: Any) -> bool:
        """Check if calculator supports batch evaluation.

        Parameters
        ----------
        calculator : Any
            Calculator to check

        Returns:
        -------
        bool
            True if calculator supports batch evaluation

        """
        return (
            calculator is not None
            and hasattr(calculator, "calculate_batch")
            and hasattr(calculator, "supports_batch_evaluation")
            and calculator.supports_batch_evaluation
        )

    @staticmethod
    def calculate_batch_energies_forces(
        path: list[Atoms],
        calculator: Any,
        supports_batch: bool,
    ) -> tuple[list[float], list[np.ndarray]]:
        """Calculate energies and forces for all images in the path.

        Parameters
        ----------
        path : list[Atoms]
            List of structures
        calculator : Any
            Calculator for energy/force calculations
        supports_batch : bool
            Whether calculator supports batch evaluation

        Returns:
        -------
        tuple[list[float], list[np.ndarray]]
            Energies and forces for all structures

        """
        energies = []
        forces_list = []

        if supports_batch:
            # Use batch evaluation for better performance
            try:
                batch_results = calculator.calculate_batch(path, properties=["energy", "forces"])

                for result in batch_results:
                    energies.append(result.get("energy", float("inf")))
                    forces_list.append(result.get("forces", np.zeros((len(path[0]), 3))))

            except (RuntimeError, AttributeError, TypeError) as e:
                logger.warning(
                    f"Batch evaluation failed, falling back to individual calculations: {e}",
                )
                supports_batch = False  # Disable batch for future iterations

        if not supports_batch:
            # Fallback to individual calculations
            for atoms in path:
                try:
                    energy = atoms.get_potential_energy()
                    forces = atoms.get_forces()
                    energies.append(energy)
                    forces_list.append(forces)
                except (RuntimeError, ValueError, AttributeError) as e:
                    logger.warning(f"Failed to calculate energy/forces: {e}")
                    energies.append(float("inf"))
                    forces_list.append(np.zeros((len(atoms), 3)))

        return energies, forces_list

    @staticmethod
    def check_convergence(forces_list: list[np.ndarray], fmax: float, step: int) -> bool:
        """Check if optimization has converged.

        Parameters
        ----------
        forces_list : list[np.ndarray]
            Forces for all structures
        fmax : float
            Force convergence threshold
        step : int
            Current step number

        Returns:
        -------
        bool
            True if converged

        """
        max_force = max(np.linalg.norm(forces, axis=1).max() for forces in forces_list)
        if max_force < fmax:
            logger.info(
                "Optimization converged after %d steps (max force: %.6f)",
                step + 1,
                max_force,
            )
            return True
        return False

    @staticmethod
    def grow_string_node(
        previous_node: Atoms,
        direction: str,
        step_size: float,
        fmax: float,
        explorer: Any | None = None,
    ) -> Atoms | None:
        """Grow string by adding a new node along the steepest descent direction.

        Parameters
        ----------
        previous_node : Atoms
            The last node in the string
        direction : str
            "forward" or "backward" to indicate growth direction
        step_size : float
            Step size for new node placement (Angstroms)
        fmax : float
            Force threshold for perpendicular optimization
        explorer : Any, optional
            Explorer instance for calculator management

        Returns:
        -------
        Atoms or None
            New node, or None if growth failed

        """
        try:
            # Get forces on previous node
            forces = previous_node.get_forces()

            # Create new node by copying and adjusting positions
            new_node = previous_node.copy()

            # Manually copy calculator reference (ASE copy() doesn't copy calculator)
            if hasattr(previous_node, "calc") and previous_node.calc is not None:
                new_node.calc = previous_node.calc

            # For forward growth, move along negative gradient (downhill)
            # For backward growth, also move along negative gradient
            # The direction is handled by which end we're growing from
            force_magnitude = np.linalg.norm(forces)
            if force_magnitude < 1e-6:
                logger.warning(
                    f"Growing String: Very small forces ({force_magnitude:.2e}), skipping node",
                )
                return None

            # Normalize and scale forces to get step direction
            step_direction = -forces / force_magnitude  # Negative for downhill
            displacement = step_direction * step_size

            new_node.positions = previous_node.positions + displacement

            # Re-attach calculator if using explorer (to ensure proper setup)
            if explorer is not None:
                explorer._create_and_attach_calculator(new_node)
                explorer._apply_constraints(new_node)

            # Optimize perpendicular to the path
            # This is a simplified version - a full implementation would:
            # 1. Calculate tangent to the path
            # 2. Project forces perpendicular to tangent
            # 3. Optimize only in perpendicular directions
            # For now, we do a quick optimization with few steps
            try:
                from qme.strategies.helpers import _get_local_optimizer_class

                OptClass = _get_local_optimizer_class("lbfgs")
                opt = OptClass(new_node)
                opt.run(fmax=fmax, steps=5)  # Limited optimization
            except Exception as e:
                logger.warning(f"Growing String: Perpendicular optimization failed: {e}")
                # Continue anyway with unoptimized node

            return new_node

        except Exception as e:
            logger.warning(f"Growing String: Failed to grow node: {e}")
            return None
