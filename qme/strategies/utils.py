"""Shared utilities for strategy implementations.

This module provides common helper functions used across different strategies,
consolidating duplicate code and providing a consistent interface.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms

from qme.utils.logging import get_qme_logger

logger = get_qme_logger(__name__)


class StrategyUtils:
    """Shared utilities for strategy implementations."""

    @staticmethod
    def ensure_charge_spin_info(atoms: Atoms, charge: int = 0, spin: int = 1) -> None:
        """Ensure atoms.info has charge and spin."""
        if hasattr(atoms, "info") and atoms.info is not None:
            atoms.info.setdefault("charge", charge)
            atoms.info.setdefault("spin", spin)

    @staticmethod
    def get_step_count(optimizer: Any) -> int | None:
        """Get step count from optimizer."""
        if hasattr(optimizer, "step_count") and optimizer.step_count is not None:
            step_count = optimizer.step_count
            return int(step_count) if isinstance(step_count, int | float) else None

        if hasattr(optimizer, "get_number_of_steps"):
            steps = optimizer.get_number_of_steps()
            return int(steps) if isinstance(steps, int | float) else None

        return None

    @staticmethod
    def get_convergence_status(optimizer: Any, atoms: Atoms) -> bool:
        """Get convergence status from optimizer."""
        converged_attr = getattr(optimizer, "converged", None)

        if callable(converged_attr):
            try:
                result = converged_attr()
                return bool(result)
            except TypeError:
                forces = atoms.get_forces()
                result = converged_attr(forces.flatten())
                return bool(result)

        return bool(converged_attr)

    @staticmethod
    def check_batch_support(calculator: Any) -> bool:
        """Check if calculator supports batch evaluation."""
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
        """Calculate energies and forces for path images."""
        energies = []
        forces_list = []

        if supports_batch:
            try:
                batch_results = calculator.calculate_batch(path, properties=["energy", "forces"])

                for result in batch_results:
                    energies.append(result.get("energy", float("inf")))
                    forces_list.append(result.get("forces", np.zeros((len(path[0]), 3))))

            except (RuntimeError, AttributeError, TypeError) as e:
                logger.warning(
                    f"Batch evaluation failed, falling back to individual calculations: {e}",
                )
                supports_batch = False

        if not supports_batch:
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
        """Check if optimization has converged."""
        max_force = max(np.linalg.norm(forces, axis=1).max() for forces in forces_list)
        if max_force < fmax:
            logger.info(
                "Optimization converged after %d steps (max force: %.6f)",
                step + 1,
                max_force,
            )
            return True
        return False
