"""Shared utilities for strategy implementations.

This module provides common helper functions used across different strategies,
consolidating duplicate code and providing a consistent interface.
"""

from typing import Any

import numpy as np
from ase import Atoms

from qme.logging_utils import get_qme_logger

logger = get_qme_logger(__name__)


class StrategyUtils:
    """Shared utilities for strategy implementations."""

    @staticmethod
    def supports_batch_evaluation(calculator: Any) -> bool:
        """Check if calculator supports batch evaluation.

        Parameters
        ----------
        calculator : Any
            Calculator to check

        Returns
        -------
        bool
            True if calculator supports batch evaluation
        """
        return (
            calculator is not None
            and hasattr(calculator, "supports_batch_evaluation")
            and calculator.supports_batch_evaluation
        )

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

        Returns
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

        Returns
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
    def calculate_spring_force(
        path: list[Atoms], index: int, spring_constant: float, energies: list[float]
    ) -> np.ndarray:
        """Calculate spring forces for NEB.

        Parameters
        ----------
        path : list[Atoms]
            List of structures in the path
        index : int
            Index of the structure to calculate forces for
        spring_constant : float
            Spring constant for NEB
        energies : list[float]
            Energies of all structures

        Returns
        -------
        np.ndarray
            Spring forces for the structure
        """
        if index in (0, len(path) - 1):
            return np.zeros((len(path[index]), 3))

        # Distance between adjacent images
        prev_pos = path[index - 1].get_positions()
        curr_pos = path[index].get_positions()
        next_pos = path[index + 1].get_positions()

        # Spring force towards previous image
        spring_prev = spring_constant * (prev_pos - curr_pos)
        # Spring force towards next image
        spring_next = spring_constant * (next_pos - curr_pos)

        return spring_prev + spring_next

    @staticmethod
    def calculate_tangent(path: list[Atoms], index: int, energies: list[float]) -> np.ndarray | None:
        """Calculate tangent vector for NEB force projection.

        Parameters
        ----------
        path : list[Atoms]
            List of structures in the path
        index : int
            Index of the structure to calculate tangent for
        energies : list[float]
            Energies of all structures

        Returns
        -------
        np.ndarray or None
            Normalized tangent vector, or None for endpoints
        """
        if index in (0, len(path) - 1):
            return None

        # Use energy-weighted tangent calculation
        prev_pos = path[index - 1].get_positions()
        curr_pos = path[index].get_positions()
        next_pos = path[index + 1].get_positions()

        # Forward difference
        forward = next_pos - curr_pos
        # Backward difference
        backward = curr_pos - prev_pos

        # Energy-weighted tangent
        if energies[index + 1] > energies[index - 1]:
            tangent = forward
        else:
            tangent = backward

        # Normalize
        norm = np.linalg.norm(tangent)
        if norm > 1e-10:
            return tangent.flatten() / norm
        else:
            return None

    @staticmethod
    def check_batch_support(calculator: Any) -> bool:
        """Check if calculator supports batch evaluation.

        Parameters
        ----------
        calculator : Any
            Calculator to check

        Returns
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
        path: list[Atoms], calculator: Any, supports_batch: bool
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

        Returns
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
                    f"Batch evaluation failed, falling back to individual calculations: {e}"
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

        Returns
        -------
        bool
            True if converged
        """
        max_force = max(np.linalg.norm(forces, axis=1).max() for forces in forces_list)
        if max_force < fmax:
            logger.info("Optimization converged after %d steps (max force: %.6f)", step + 1, max_force)
            return True
        return False
