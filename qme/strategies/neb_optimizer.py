"""Unified NEB optimizer for both regular NEB and CI-NEB."""

from typing import Any

import numpy as np
from ase import Atoms

from qme.strategies.utils import StrategyUtils
from qme.utils.logging import get_qme_logger

logger = get_qme_logger(__name__)


class NEBOptimizer:
    """Unified NEB optimizer that handles both regular NEB and CI-NEB.

    This optimizer automatically detects batch evaluation support and uses it when available.
    It consolidates the logic from both the original NEBOptimizer and BatchNEBOptimizer classes.
    """

    def __init__(
        self,
        images: list[Atoms],
        spring_constant: float = 5.0,
        climb: bool = False,
        fmax: float = 0.05,
        steps: int = 1000,
        **kwargs: Any,
    ) -> None:
        """Initialize unified NEB optimizer.

        Parameters
        ----------
        images : list[Atoms]
            List of Atoms objects representing the NEB path
        spring_constant : float
            Spring constant for NEB spring forces
        climb : bool
            Whether to use climbing image NEB (CI-NEB)
        fmax : float
            Force convergence criterion
        steps : int
            Maximum optimization steps
        **kwargs
            Additional arguments

        """
        # Copy images but preserve attached calculators (ASE copy() drops calc)
        self.images = []
        for atoms in images:
            copied = atoms.copy()
            if getattr(atoms, "calc", None) is not None:
                copied.calc = atoms.calc
            self.images.append(copied)
        self.spring_constant = spring_constant
        self.climb = climb
        self.fmax = fmax
        self.steps = steps
        self.kwargs = kwargs
        self.climbing_image = None

        # Ensure charge and spin info are set and attach calculator to all images
        for atoms in self.images:
            StrategyUtils.ensure_charge_spin_info(atoms)

        # Check if we can use batch evaluation
        calculator = self.images[0].calc if self.images[0].calc is not None else None
        self.supports_batch = StrategyUtils.check_batch_support(calculator)

        method_name = "CI-NEB" if self.climb else "NEB"
        if self.supports_batch:
            logger.info(
                f"Using batch evaluation for {method_name} optimization with {len(self.images)} images",
            )
        else:
            logger.info(f"Starting {method_name} optimization with {len(self.images)} images")

    def optimize(self) -> list[Atoms]:
        """Optimize NEB path using batch evaluation when available."""
        method_name = "CI-NEB" if self.climb else "NEB"

        for step in range(self.steps):
            # Calculate forces for all images
            energies, forces_list = self._calculate_energies_forces()

            # Determine climbing image for CI-NEB
            if self.climb and len(energies) > 2:
                self.climbing_image = self._determine_climbing_image(energies)

            # Apply NEB forces (spring + nudging + climbing if enabled)
            neb_forces = self._apply_neb_forces(forces_list, energies)

            # Update positions
            self._update_positions(neb_forces)

            # Check convergence
            max_force = max(np.max(np.abs(force)) for force in neb_forces)
            if max_force < self.fmax:
                logger.info(
                    f"{method_name} converged after {step + 1} steps (max force: {max_force:.6f})",
                )
                if self.climb and self.climbing_image is not None:
                    logger.info("Climbing image was image %d", self.climbing_image)
                break

        return self.images

    def _calculate_energies_forces(self) -> tuple[list[float], list[np.ndarray]]:
        """Calculate energies and forces for all images."""
        calculator = self.images[0].calc if self.images[0].calc is not None else None
        return StrategyUtils.calculate_batch_energies_forces(
            self.images,
            calculator,
            self.supports_batch,
        )

    def _determine_climbing_image(self, energies: list[float]) -> int | None:
        """Determine which image should be the climbing image."""
        if not self.climb or len(energies) <= 2:
            return None

        # Find highest energy image (excluding endpoints)
        valid_energies = [(i, e) for i, e in enumerate(energies[1:-1], 1) if not np.isnan(e)]
        if not valid_energies:
            return None

        return max(valid_energies, key=lambda x: x[1])[0]

    def _apply_neb_forces(
        self,
        forces_list: list[np.ndarray],
        energies: list[float],
    ) -> list[np.ndarray]:
        """Apply NEB forces (spring + nudging + climbing if enabled)."""
        neb_forces = []

        for i in range(len(self.images)):
            if i in (0, len(self.images) - 1):
                # Endpoints: use only spring forces
                neb_forces.append(self._spring_forces(i))
            else:
                # Middle images: spring forces + nudging
                spring_f = self._spring_forces(i)
                nudged_f = self._nudge_forces(forces_list[i], spring_f)

                # Apply climbing image behavior if enabled and this is the climbing image
                if self.climb and self.climbing_image == i:
                    # Invert the parallel component of the force for climbing
                    climbing_f = self._apply_climbing_forces(forces_list[i], spring_f, i)
                    neb_forces.append(climbing_f)
                else:
                    neb_forces.append(nudged_f)

        return neb_forces

    def _spring_forces(self, i: int) -> np.ndarray:
        """Calculate spring forces for image i."""
        if i == 0:
            # First image: spring to next
            return self.spring_constant * (self.images[i + 1].positions - self.images[i].positions)
        if i == len(self.images) - 1:
            # Last image: spring to previous
            return self.spring_constant * (self.images[i - 1].positions - self.images[i].positions)
        # Middle images: spring to both neighbors
        f_prev = self.spring_constant * (self.images[i - 1].positions - self.images[i].positions)
        f_next = self.spring_constant * (self.images[i + 1].positions - self.images[i].positions)
        return f_prev + f_next

    def _nudge_forces(self, forces: np.ndarray, spring_forces: np.ndarray) -> np.ndarray:
        """Apply nudging to forces (project out parallel component)."""
        # Calculate tangent vector (simplified)
        if len(self.images) > 1:
            tangent = self.images[1].positions - self.images[0].positions
            tangent = tangent / np.linalg.norm(tangent)
        else:
            tangent = np.zeros_like(forces[0])

        # Project forces perpendicular to tangent
        parallel_component = np.sum(forces * tangent) * tangent
        perpendicular_forces = forces - parallel_component

        return perpendicular_forces + spring_forces

    def _apply_climbing_forces(
        self,
        forces: np.ndarray,
        spring_forces: np.ndarray,
        index: int,
    ) -> np.ndarray:
        """Apply climbing image forces by inverting parallel component."""
        # Calculate tangent vector for this image
        tangent = self._calculate_tangent_for_climbing(index)
        if tangent is None:
            # Fallback to regular NEB forces if tangent calculation fails
            return self._nudge_forces(forces, spring_forces)

        # Project forces onto tangent (parallel component)
        parallel_component = np.sum(forces.flatten() * tangent) * tangent
        parallel_component = parallel_component.reshape(-1, 3)

        # Invert parallel component for climbing (make it point uphill)
        return forces - 2 * parallel_component + spring_forces

    def _calculate_tangent_for_climbing(self, index: int) -> np.ndarray | None:
        """Calculate tangent vector for climbing image."""
        if index <= 0 or index >= len(self.images) - 1:
            return None

        # Use energy-weighted tangent calculation
        prev_pos = self.images[index - 1].positions
        curr_pos = self.images[index].positions
        next_pos = self.images[index + 1].positions

        # Forward difference
        forward = next_pos - curr_pos
        # Backward difference
        backward = curr_pos - prev_pos

        # Energy-weighted tangent (same as regular NEB)
        try:
            _ = self.images[index].get_potential_energy()  # curr_energy
            next_energy = self.images[index + 1].get_potential_energy()
            prev_energy = self.images[index - 1].get_potential_energy()

            tangent = forward if next_energy > prev_energy else backward
        except Exception:
            # Fallback to simple average
            tangent = (forward + backward) / 2

        # Normalize
        norm = np.linalg.norm(tangent)
        if norm > 1e-10:
            return tangent.flatten() / norm
        return None

    def _update_positions(self, forces: list[np.ndarray], step_size: float = 0.01) -> None:
        """Update positions using forces."""
        for atoms, force in zip(self.images, forces, strict=False):
            atoms.positions += step_size * force
