"""Growing string method strategy for TS search.

This implementation follows the pysisyphus GrowingString algorithm,
which properly handles perpendicular force convergence, reparametrization,
and node growth based on parametrization density.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from ase import Atoms

from qme.core.base_strategy import BaseStrategy, StrategyMetadata
from qme.core.registry import REGISTRY
from qme.io.path_manager import PathManager
from qme.strategies.helpers import validate_ts_structure
from qme.strategies.minima import LocalMinimaStrategy
from qme.strategies.ts import LocalTSStrategy
from qme.strategies.utils import StrategyUtils
from qme.utils.logging import get_qme_logger

logger = get_qme_logger(__name__)


def ensure_atoms(obj: Any, context: str) -> Atoms:
    """Ensure the provided object is an ASE Atoms instance."""
    if isinstance(obj, Atoms):
        return obj
    if isinstance(obj, list) and obj and all(isinstance(item, Atoms) for item in obj):
        first_item = cast(Atoms, obj[0])
        return cast(Atoms, first_item.copy())
    msg = f"{context} expected ase.Atoms, received {type(obj).__name__}"
    raise TypeError(msg)


class MultiStructureGrowingStringStrategy(BaseStrategy):
    """Growing string method strategy for TS search.

    Implements the proper GSM algorithm with:
    - Perpendicular force convergence checking
    - Arc length-based reparametrization
    - Node growth based on parametrization density
    - Energy-weighted parametrization option
    """

    metadata = StrategyMetadata(
        name="ts:growing_string",
        target="ts",
        strategy="growing_string",
        description="Growing string method for TS search (DE-GSM style)",
        aliases=["growing_string", "gsm"],
        requires_multiple_structures=True,
    )

    def __init__(self, explorer: Any, profiler: Any | None = None) -> None:
        """Initialize growing string strategy."""
        super().__init__(explorer, profiler)
        # These will be set in run()
        self.left_string: list[Atoms] = []
        self.right_string: list[Atoms] = []
        self.images: list[Atoms] = []
        self.all_energies: list[list[float]] = []
        self.all_forces: list[list[np.ndarray]] = []
        self.perp_forces_list: list[np.ndarray] = []
        self.reparam_in: int = 0
        self.max_nodes: int = 15
        self.sk: float = 0.0  # Desired spacing on normalized arclength

    def get_cur_param_density(self, kind: str | None = None) -> np.ndarray:
        """Calculate current parametrization density (arc length or energy-weighted).

        Parameters
        ----------
        kind : str, optional
            "energy" for energy-weighted, None for equal spacing

        Returns:
        -------
        np.ndarray
            Normalized parametrization density (0 to 1)
        """
        if len(self.images) < 2:
            return np.array([0.0, 1.0])

        # Calculate distances between consecutive images
        diffs = []
        for i in range(len(self.images)):
            if i == 0:
                diffs.append(np.zeros(self.images[0].positions.shape))
            else:
                diff = self.images[i].positions - self.images[i - 1].positions
                diffs.append(diff)

        norms = np.array([np.linalg.norm(diff) for diff in diffs])
        param_density = np.cumsum(norms)
        total_length = param_density[-1]

        if total_length > 0:
            logger.debug(f"Current string length={total_length:.6f} Å")
        else:
            logger.warning("String length is zero, using equal spacing")
            return np.linspace(0.0, 1.0, len(self.images))

        # Energy weighted parametrization density
        if kind == "energy" and len(self.all_energies) > 0:
            prev_energies = np.array(self.all_energies[-1])
            if len(prev_energies) == len(self.images):
                mean_energies = (prev_energies[1:] + prev_energies[:-1]) / 2
                weights = mean_energies - prev_energies.min()
                # Damp everything a bit
                weights = np.sqrt(np.maximum(weights, 1e-10))  # Avoid negative weights
                param_density = np.array([0.0])
                for weight, norm in zip(weights, norms[1:], strict=False):
                    param_density = np.append(param_density, param_density[-1] + weight * norm)
                total_length = param_density[-1]
                if total_length > 0:
                    param_density /= total_length
                    return param_density

        # Normalize to [0, 1]
        if total_length > 0:
            param_density /= total_length
        else:
            param_density = np.linspace(0.0, 1.0, len(self.images))

        return param_density

    def get_tangent(self, i: int) -> np.ndarray:
        """Calculate tangent vector for image i.

        For frontier nodes, uses simple normalized coordinate difference.
        For internal nodes, uses energy-weighted tangent.

        Parameters
        ----------
        i : int
            Image index

        Returns:
        -------
        np.ndarray
            Normalized tangent vector
        """
        nimages = len(self.images)
        if nimages < 2:
            result: np.ndarray = np.zeros(self.images[0].positions.shape).flatten()
            return result

        # Check if this is a frontier node
        lf_ind = len(self.left_string) - 1
        rf_ind = len(self.left_string)  # First node of right string

        # For frontier nodes, use simple normalized coordinate difference
        if i == lf_ind and i < nimages - 1:
            tangent = self.images[i + 1].positions - self.images[i].positions
            tangent_flat: np.ndarray = tangent.flatten()
            norm = np.linalg.norm(tangent_flat)
            if norm > 1e-10:
                normalized = cast(np.ndarray, tangent_flat / norm)
                return normalized
            return tangent_flat

        if i == rf_ind and i > 0:
            tangent = self.images[i].positions - self.images[i - 1].positions
            tangent_flat = tangent.flatten()
            norm = np.linalg.norm(tangent_flat)
            if norm > 1e-10:
                normalized = cast(np.ndarray, tangent_flat / norm)
                return normalized
            return tangent_flat

        # For internal nodes, use energy-weighted tangent
        if len(self.all_energies) > 0 and len(self.all_energies[-1]) == nimages:
            energies = np.array(self.all_energies[-1])

            if i == 0:
                tangent = self.images[1].positions - self.images[0].positions
            elif i == nimages - 1:
                tangent = self.images[nimages - 1].positions - self.images[nimages - 2].positions
            else:
                # Energy-weighted tangent
                v_prev = self.images[i].positions - self.images[i - 1].positions
                v_next = self.images[i + 1].positions - self.images[i].positions

                dE_prev = abs(energies[i] - energies[i - 1])
                dE_next = abs(energies[i + 1] - energies[i])

                if dE_prev > dE_next:
                    tangent = v_prev
                elif dE_next > dE_prev:
                    tangent = v_next
                else:
                    tangent = v_next - v_prev

            tangent_flat = tangent.flatten()
            norm = np.linalg.norm(tangent_flat)
            if norm > 1e-10:
                normalized = cast(np.ndarray, tangent_flat / norm)
                return normalized
            return tangent_flat

        # Fallback: simple difference
        if i < nimages - 1:
            tangent = self.images[i + 1].positions - self.images[i].positions
        elif i > 0:
            tangent = self.images[i].positions - self.images[i - 1].positions
        else:
            zero_tangent = np.zeros(self.images[0].positions.shape).flatten()
            return zero_tangent

        tangent_flat = tangent.flatten()
        norm = np.linalg.norm(tangent_flat)
        if norm > 1e-10:
            normalized = cast(np.ndarray, tangent_flat / norm)
            return normalized
        return tangent_flat

    def get_perpendicular_forces(self) -> list[np.ndarray]:
        """Calculate forces perpendicular to the path tangent.

        Returns:
        -------
        list[np.ndarray]
            Perpendicular forces for each image
        """
        if len(self.all_forces) == 0:
            return []

        forces_list = self.all_forces[-1]
        perp_forces = []

        for i, forces in enumerate(forces_list):
            tangent = self.get_tangent(i)
            forces_flat = forces.flatten()

            # Project forces perpendicular to tangent
            parallel_component = np.dot(forces_flat, tangent) * tangent
            perp_force = forces_flat - parallel_component
            perp_forces.append(perp_force.reshape(forces.shape))

        return perp_forces

    def get_new_image(self, ref_index: int) -> Atoms:
        """Get new image by taking a step from ref_index towards the center.

        Parameters
        ----------
        ref_index : int
            Reference image index

        Returns:
        -------
        Atoms
            New image
        """
        new_img = cast(Atoms, self.images[ref_index].copy())

        # Determine tangent direction
        if ref_index <= len(self.left_string) - 1:
            tangent_ind = ref_index + 1
        else:
            tangent_ind = ref_index - 1

        if tangent_ind < 0 or tangent_ind >= len(self.images):
            # Fallback: just copy the reference
            return new_img

        tangent_img = self.images[tangent_ind]

        # Calculate distance vector (negative because we step from new_img towards tangent_img)
        distance = -(new_img.positions - tangent_img.positions)

        # Calculate step based on desired spacing
        cpd = self.get_cur_param_density()
        if len(cpd) > max(ref_index, tangent_ind):
            param_dens_diff = abs(cpd[ref_index] - cpd[tangent_ind])
            if param_dens_diff > 1e-10:
                step_length = self.sk / param_dens_diff
            else:
                step_length = self.sk * 10.0  # Fallback
        else:
            step_length = self.sk * 10.0  # Fallback

        step = step_length * distance
        new_img.positions = new_img.positions + step

        # Attach calculator using PathManager
        if self.explorer is not None:
            PathManager.attach_calculators(self.explorer, [new_img])
            StrategyUtils.ensure_charge_spin_info(new_img)

        return new_img

    def reparam_cart(self, desired_param_density: np.ndarray, image_energies: list[float]) -> None:
        """Reparametrize images in cartesian coordinates.

        Parameters
        ----------
        desired_param_density : np.ndarray
            Desired parametrization density (0 to 1)
        image_energies : list[float]
            Current energies for all images
        """
        max_micro_cycles = 5
        for i in range(1, len(self.images) - 1):  # Skip endpoints
            desired = desired_param_density[i]
            reparam_image = self.images[i]

            for _ in range(max_micro_cycles):
                cur = self.get_cur_param_density()
                diff = cur[i] - desired

                if abs(diff) < 1e-4:  # Convergence threshold
                    break

                # Determine direction to shift
                sign = -1 if diff > 0 else 1
                tangent_ind = i + sign

                if tangent_ind < 0 or tangent_ind >= len(self.images):
                    break

                tangent_image = self.images[tangent_ind]
                distance = -(reparam_image.positions - tangent_image.positions)

                param_dens_diff = abs(cur[tangent_ind] - cur[i])
                if param_dens_diff > 1e-10:
                    step_length = abs(diff) / param_dens_diff
                else:
                    step_length = abs(diff) * 10.0

                step = step_length * distance
                reparam_image.positions = reparam_image.positions + step

                # Recalculate after shift

        logger.debug(f"Reparametrization complete. Param density: {self.get_cur_param_density()}")

    def reparametrize(
        self,
        image_energies: list[float],
        forces: list[np.ndarray],
        perp_thresh: float = 0.05,
        reparam_every: int = 2,
        reparam_every_full: int = 3,
    ) -> bool:
        """Main reparametrization routine.

        Checks perpendicular forces on frontier nodes and adds new nodes if converged.
        Reparametrizes the string periodically.

        Parameters
        ----------
        image_energies : list[float]
            Energies for all images
        forces : list[np.ndarray]
            Forces for all images
        perp_thresh : float
            Perpendicular force threshold for frontier convergence
        reparam_every : int
            Reparametrization frequency for growing string
        reparam_every_full : int
            Reparametrization frequency for fully grown string

        Returns:
        -------
        bool
            True if reparametrization occurred
        """
        reparametrized = False
        self.reparam_in -= 1

        # Check if new images can be added
        fully_grown = len(self.images) >= self.max_nodes
        lf_ind = len(self.left_string) - 1
        rf_ind = len(self.left_string)

        if not fully_grown:
            # Calculate perpendicular forces
            perp_forces = self.get_perpendicular_forces()
            if len(perp_forces) > 0:
                self.perp_forces_list.append(np.array([pf.flatten() for pf in perp_forces]))

                # Check frontier node convergence
                reparam_check = "rms"  # Use RMS for checking

                if len(perp_forces) > lf_ind:
                    perp_force_lf = perp_forces[lf_ind]
                    if reparam_check == "rms":
                        check_val_lf = np.sqrt(np.mean(perp_force_lf**2))
                    else:
                        check_val_lf = np.linalg.norm(perp_force_lf)

                    logger.debug(
                        f"Left frontier node {lf_ind}: {reparam_check}(perp_forces)={check_val_lf:.6f}"
                    )

                    if check_val_lf <= perp_thresh:
                        # Add new left frontier node
                        new_left_frontier = self.get_new_image(lf_ind)
                        self.left_string.append(new_left_frontier)
                        self.images = self.left_string + self.right_string
                        logger.info(
                            f"Added new left frontier node. String size: {len(self.images)}"
                        )
                        self.reparam_in = 0  # Force reparametrization

                # Re-evaluate fully_grown after potential left growth
                fully_grown = len(self.images) >= self.max_nodes

                if not fully_grown and len(perp_forces) > rf_ind:
                    perp_force_rf = perp_forces[rf_ind]
                    if reparam_check == "rms":
                        check_val_rf = np.sqrt(np.mean(perp_force_rf**2))
                    else:
                        check_val_rf = np.linalg.norm(perp_force_rf)

                    logger.debug(
                        f"Right frontier node {rf_ind}: {reparam_check}(perp_forces)={check_val_rf:.6f}"
                    )

                    if check_val_rf <= perp_thresh:
                        # Add new right frontier node
                        new_right_frontier = self.get_new_image(rf_ind)
                        self.right_string.insert(0, new_right_frontier)
                        self.images = self.left_string + self.right_string
                        logger.info(
                            f"Added new right frontier node. String size: {len(self.images)}"
                        )
                        self.reparam_in = 0  # Force reparametrization

        if self.reparam_in > 0:
            logger.debug(f"Skipping reparametrization. Next in {self.reparam_in} cycles.")
        else:
            # Reparametrize images
            desired_param_density = self.sk * np.arange(len(self.images))
            self.reparam_cart(desired_param_density, image_energies)

            self.reparam_in = (
                reparam_every_full if len(self.images) >= self.max_nodes else reparam_every
            )
            reparametrized = True

        return reparametrized

    def optimize_perpendicular(self, fmax: float, max_steps: int = 20) -> None:
        """Optimize images perpendicular to the path using simple steepest descent.

        Parameters
        ----------
        fmax : float
            Force convergence threshold
        max_steps : int
            Maximum optimization steps per image
        """
        perp_forces = self.get_perpendicular_forces()

        if len(perp_forces) == 0:
            return

        for i, atoms in enumerate(self.images):
            if i == 0 or i == len(self.images) - 1:
                continue  # Skip endpoints

            if i >= len(perp_forces):
                continue

            perp_force = perp_forces[i]
            max_perp_force = np.max(np.linalg.norm(perp_force, axis=1))

            if max_perp_force < fmax:
                continue  # Already converged

            # Simple steepest descent along perpendicular forces
            step_size = 0.01  # Small step size
            for _ in range(max_steps):
                # Get current perpendicular forces
                forces = atoms.get_forces()
                tangent = self.get_tangent(i)
                forces_flat = forces.flatten()
                parallel = np.dot(forces_flat, tangent) * tangent
                perp_force = forces_flat - parallel
                perp_force_reshaped = perp_force.reshape(forces.shape)

                max_perp_force = np.max(np.linalg.norm(perp_force_reshaped, axis=1))

                if max_perp_force < fmax:
                    break

                # Take a step along perpendicular forces
                atoms.positions += step_size * perp_force_reshaped

                # Recalculate energy/forces
                try:
                    atoms.get_potential_energy()
                except Exception:
                    break

    def run(
        self,
        atoms_list: list[Atoms],
        validate_ts: bool = False,
        calculate_frequencies: bool = False,
        require_ts: bool | None = None,
        **kwargs: Any,
    ) -> dict[str, Atoms | list[Atoms] | bool | int | float | str]:
        """Run growing string method.

        Parameters
        ----------
        atoms_list : list[Atoms]
            List of exactly 2 structures (reactant, product)
        validate_ts : bool, default=False
            Whether to validate TS structure
        calculate_frequencies : bool, default=False
            Whether to calculate frequencies
        require_ts : bool, optional
            If True, raise an error unless the method returns a validated first-order
            saddle (strings meet, refinement converges, and one imaginary mode).
        **kwargs : Any
            Additional parameters:
            - npoints: Maximum number of images (default: 15)
            - fmax: Force convergence threshold (default: 0.05)
            - steps: Maximum iterations (default: 200)
            - perp_thresh: Perpendicular force threshold (default: 0.05)
            - reparam_every: Reparametrization frequency (default: 2)
            - reparam_every_full: Reparametrization frequency when fully grown (default: 3)
            - optimize_endpoints: Optimize endpoints first (default: True)
            - refine_ts: Refine TS after finding (default: True)

        Returns:
        -------
        dict
            Result dictionary with optimized_atoms, trajectory, etc.
        """
        self.validate_inputs(atoms_list)

        # Parse parameters
        if require_ts is None:
            explorer_ts_kwargs = getattr(self.explorer, "ts_kwargs", {}) or {}
            require_ts = bool(explorer_ts_kwargs.get("require_ts", False))
        else:
            require_ts = bool(require_ts)

        kwargs.pop("require_ts", None)

        self.max_nodes = kwargs.get("npoints", 15)
        fmax = kwargs.get("fmax", 0.05)
        max_steps = kwargs.get("steps", 200)
        perp_thresh = kwargs.get("perp_thresh", 0.05)
        reparam_every = kwargs.get("reparam_every", 2)
        reparam_every_full = kwargs.get("reparam_every_full", 3)
        optimize_endpoints = kwargs.get("optimize_endpoints", True)
        refine_ts = kwargs.get("refine_ts", True)
        local_optimizer_name = kwargs.get("local_optimizer_name", "sella")

        # Desired spacing on normalized arclength
        self.sk = 1.0 / (self.max_nodes + 1)

        # Enforce exactly two structures
        if len(atoms_list) != 2:
            msg = f"Growing string method requires exactly 2 Atoms objects, got {len(atoms_list)}"
            raise ValueError(msg)

        reactant, product = atoms_list[0].copy(), atoms_list[1].copy()

        # Attach calculators using PathManager for consistency
        if self.explorer is not None:
            PathManager.attach_calculators(self.explorer, [reactant, product])
            # Ensure charge/spin info is set
            StrategyUtils.ensure_charge_spin_info(reactant)
            StrategyUtils.ensure_charge_spin_info(product)

        # Optimize endpoints if requested
        if optimize_endpoints:
            logger.info("Growing String: Optimizing reactant to local minimum...")
            minima_strategy = LocalMinimaStrategy(explorer=self.explorer)
            r_result = minima_strategy.run(
                reactant, fmax=fmax, steps=200, local_optimizer_name="lbfgs"
            )
            reactant = ensure_atoms(r_result["optimized_atoms"], "Local minima optimization")

            logger.info("Growing String: Optimizing product to local minimum...")
            p_result = minima_strategy.run(
                product, fmax=fmax, steps=200, local_optimizer_name="lbfgs"
            )
            product = ensure_atoms(p_result["optimized_atoms"], "Local minima optimization")

            # Re-attach calculators using PathManager
            if self.explorer is not None:
                PathManager.attach_calculators(self.explorer, [reactant, product])
                StrategyUtils.ensure_charge_spin_info(reactant)
                StrategyUtils.ensure_charge_spin_info(product)

        # Initialize strings
        self.left_string = [reactant.copy()]
        self.right_string = [product.copy()]
        self.images = self.left_string + self.right_string

        # Ensure calculators are attached using PathManager
        if self.explorer is not None:
            PathManager.attach_calculators(self.explorer, self.images)
            for atoms in self.images:
                StrategyUtils.ensure_charge_spin_info(atoms)

        logger.info(
            f"Growing String: Starting with {len(self.images)} images, max {self.max_nodes}"
        )

        # Initialize reparametrization counter
        self.reparam_in = reparam_every

        # Main GSM loop
        strings_met = False
        for iteration in range(max_steps):
            # Check if fully grown
            if len(self.images) >= self.max_nodes:
                logger.info(f"Growing String: Reached maximum images ({self.max_nodes})")
                break

            # Calculate energies and forces
            energies = []
            forces_list = []
            for atoms in self.images:
                try:
                    energy = atoms.get_potential_energy()
                    forces = atoms.get_forces()
                    energies.append(energy)
                    forces_list.append(forces)
                except Exception as e:
                    logger.warning(f"Failed to calculate energy/forces: {e}")
                    energies.append(float("inf"))
                    forces_list.append(np.zeros((len(atoms), 3)))

            self.all_energies.append(energies)
            self.all_forces.append(forces_list)

            # Check if strings have met (distance between tips)
            if len(self.left_string) > 0 and len(self.right_string) > 0:
                forward_tip = self.left_string[-1].positions
                backward_tip = self.right_string[0].positions
                distance = np.linalg.norm(forward_tip - backward_tip)

                distance_threshold = kwargs.get("distance_threshold", 0.05)
                if distance < distance_threshold:
                    strings_met = True

            # Optimize perpendicular to path first
            self.optimize_perpendicular(fmax, max_steps=20)

            # Reparametrize and potentially add new nodes
            self.reparametrize(
                energies, forces_list, perp_thresh, reparam_every, reparam_every_full
            )

            # Log progress
            if (iteration + 1) % 10 == 0:
                logger.info(
                    f"Growing String: Iteration {iteration + 1}, "
                    f"images={len(self.images)}, "
                    f"left={len(self.left_string)}, right={len(self.right_string)}"
                )

        # Combine strings into full path
        full_path = self.left_string + self.right_string[::-1]

        logger.info(
            f"Growing String: Complete with {len(full_path)} images "
            f"(left: {len(self.left_string)}, right: {len(self.right_string)}, "
            f"strings_met={strings_met})"
        )

        # Filter redundant structures for consistency with other path strategies
        # For GSM we retain all images to avoid collapsing back onto endpoints

        # Find TS as highest energy image
        energies = []
        for atoms in full_path:
            try:
                energy = atoms.get_potential_energy()
                energies.append(energy)
            except Exception:
                energies.append(float("-inf"))

        if not energies or all(e == float("-inf") for e in energies):
            ts_index = len(full_path) // 2
            logger.warning("Could not calculate energies, using middle image as TS guess")
        else:
            # Find highest energy (avoid endpoints)
            search_start = 1 if len(full_path) > 2 else 0
            search_end = len(full_path) - 1 if len(full_path) > 2 else len(full_path)
            search_range = range(search_start, search_end)

            valid_energies = [
                (i, energies[i]) for i in search_range if energies[i] != float("-inf")
            ]
            if not valid_energies:
                ts_index = len(full_path) // 2
            else:
                valid_energies.sort(key=lambda x: x[1], reverse=True)
                ts_index = valid_energies[0][0]
                logger.info(
                    f"Growing String: Highest energy at image {ts_index} "
                    f"(E = {energies[ts_index]:.6f} eV)"
                )

        ts_guess = full_path[ts_index]

        # Refine TS if requested
        ts_result = None
        if refine_ts:
            logger.info("Growing String: Refining TS with local optimization...")
            ts_strategy = LocalTSStrategy(explorer=self.explorer)
            ts_refinement_fmax = 0.005
            ts_refinement_steps = 2000
            ts_refinement_optimizer = local_optimizer_name
            if ts_refinement_optimizer.lower() == "sella":
                ts_refinement_optimizer = "rfo"

            ts_call_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in {"fmax", "steps", "explorer", "local_optimizer_name"}
            }

            ts_call_kwargs.pop("require_ts", None)
            ts_result = ts_strategy.run(
                [ts_guess],  # LocalTSStrategy expects Sequence[Atoms]
                fmax=ts_refinement_fmax,
                steps=ts_refinement_steps,
                local_optimizer_name=ts_refinement_optimizer,
                calculate_frequencies=True,
                **ts_call_kwargs,
            )
            ts_structure = ensure_atoms(ts_result["optimized_atoms"], "TS refinement")
            ts_converged = ts_result["converged"]
        else:
            ts_structure = ts_guess
            ts_converged = strings_met

        ts_structure = ensure_atoms(ts_structure, "Growing string TS candidate")

        # Validate TS if requested
        validation_result = None
        if validate_ts:
            validation_result = validate_ts_structure(ts_structure, self.explorer)

        # Prepare result
        result = self.prepare_result(
            ts_structure,
            converged=ts_converged,
            trajectory=full_path,
            forward_string=self.left_string,
            backward_string=self.right_string,
            strings_met=strings_met,
        )

        freq_analysis = None
        if ts_result is not None and isinstance(ts_result, dict):
            freq_analysis = ts_result.get("frequency_analysis")

        if (
            (require_ts or calculate_frequencies)
            and freq_analysis is None
            and isinstance(ts_structure, Atoms)
        ):
            if self.explorer is not None:
                freq_analysis = self.explorer.calculate_frequencies(
                    atoms=ts_structure,
                    temperature=kwargs.get("temperature", 298.15),
                    save_hessian=False,
                )

        validation_error_messages: list[str] = []
        if require_ts:
            if not strings_met:
                validation_error_messages.append(
                    "Growing string did not converge: forward and backward strings never met."
                )
            if not ts_converged:
                validation_error_messages.append("Local TS refinement failed to converge.")
            if freq_analysis is None:
                validation_error_messages.append(
                    "Frequency analysis unavailable for TS validation."
                )
            else:
                ts_info = freq_analysis.get("ts_analysis") or {}
                imag_modes = (
                    freq_analysis.get("num_imaginary_modes")
                    or freq_analysis.get("n_imaginary_modes")
                    or freq_analysis.get("n_imaginary_frequencies")
                    or (
                        ts_info.get("n_imaginary_frequencies")
                        if isinstance(ts_info, dict)
                        else None
                    )
                )
                is_ts = freq_analysis.get("is_ts")
                if is_ts is None and isinstance(ts_info, dict):
                    is_ts = ts_info.get("is_transition_state")
                if not is_ts or (imag_modes is not None and imag_modes != 1):
                    validation_error_messages.append(
                        "Refined structure is not a first-order saddle (frequency analysis failed)."
                    )

        if validation_error_messages:
            raise RuntimeError(
                "TS validation failed with require_ts=True:\n- "
                + "\n- ".join(validation_error_messages)
            )

        if validation_result is not None:
            result["ts_validation"] = validation_result

        if freq_analysis is not None:
            result["frequency_analysis"] = freq_analysis
            result["is_ts"] = freq_analysis.get("is_ts")
            if ts_result is not None and isinstance(ts_result, dict):
                if "free_energy_correction" in ts_result:
                    result["free_energy_correction"] = ts_result["free_energy_correction"]

        return result


# Register the strategy
REGISTRY.register(MultiStructureGrowingStringStrategy)
