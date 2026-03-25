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
        self.new_image_inds: list[int] = []  # Track indices of newly added images
        self.reparam_check: str = "rms"  # Method for checking convergence: "rms" or "norm"

    @property
    def left_size(self) -> int:
        """Number of images in the left string."""
        return len(self.left_string)

    @property
    def right_size(self) -> int:
        """Number of images in the right string."""
        return len(self.right_string)

    @property
    def string_size(self) -> int:
        """Total number of images in the string."""
        return len(self.images)

    @property
    def nodes_missing(self) -> int:
        """Number of nodes still to be grown."""
        return max(0, self.max_nodes - self.string_size)

    @property
    def fully_grown(self) -> bool:
        """Whether the string has reached maximum size."""
        return self.string_size >= self.max_nodes

    @property
    def lf_ind(self) -> int:
        """Index of the left frontier node."""
        return self.left_size - 1

    @property
    def rf_ind(self) -> int:
        """Index of the right frontier node (first node of right string)."""
        return self.left_size

    @property
    def full_string_image_inds(self) -> np.ndarray:
        """Array of image indices for the full string (0 to string_size-1)."""
        return np.arange(self.string_size)

    def get_cur_param_density(self, kind: str | None = None) -> np.ndarray:
        """Calculate current parametrization density (arc length or energy-weighted).

        Parameters
        ----------
        kind : str, optional
            "energy" for energy-weighted, None for equal spacing

        Returns
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

        Returns
        -------
        np.ndarray
            Normalized tangent vector
        """
        nimages = len(self.images)
        if nimages < 2:
            result: np.ndarray = np.zeros(self.images[0].positions.shape).flatten()
            return result

        # Check if this is a frontier node
        lf_ind = self.lf_ind
        rf_ind = self.rf_ind

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

        Returns
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

        Returns
        -------
        Atoms
            New image
        """
        new_img = cast(Atoms, self.images[ref_index].copy())

        # Determine tangent direction and insertion index
        if ref_index <= self.lf_ind:
            tangent_ind = ref_index + 1
        else:
            tangent_ind = ref_index - 1

        if tangent_ind < 0 or tangent_ind >= len(self.images):
            # Fallback: just copy the reference
            logger.warning(
                f"get_new_image: Invalid tangent_ind {tangent_ind} for ref_index {ref_index}, "
                f"string_size={self.string_size}"
            )
            return new_img

        tangent_img = self.images[tangent_ind]

        # Calculate distance vector (negative because we step from new_img towards tangent_img)
        # This ensures the distance is computed in the coordinate space of new_img
        try:
            distance = -(new_img.positions - tangent_img.positions)
        except Exception as e:
            logger.warning(f"get_new_image: Error calculating distance: {e}")
            return new_img

        # Calculate step based on desired spacing
        # The desired step length is determined from the parametrization density
        # Δparam_density / distance = self.sk / step
        # step = self.sk / Δparam_density * distance
        cpd = self.get_cur_param_density()
        if len(cpd) > max(ref_index, tangent_ind):
            param_dens_diff = abs(cpd[ref_index] - cpd[tangent_ind])
            if param_dens_diff > 1e-10:
                step_length = self.sk / param_dens_diff
            else:
                # Fallback: use a reasonable step size
                step_length = self.sk * 10.0
                logger.debug(
                    f"get_new_image: Small param_dens_diff ({param_dens_diff:.2e}), "
                    f"using fallback step_length"
                )
        else:
            step_length = self.sk * 10.0
            logger.debug(
                f"get_new_image: Invalid cpd length ({len(cpd)}), using fallback step_length"
            )

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
        """Perform main reparametrization routine.

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

        Returns
        -------
        bool
            True if reparametrization occurred
        """
        reparametrized = False
        # If this counter reaches 0 reparametrization will occur
        self.reparam_in -= 1

        self.new_image_inds = []

        # Check if new images can be added for incomplete strings
        if not self.fully_grown:
            perp_forces = self.get_perpendicular_forces()
            if len(perp_forces) > 0:
                perp_forces_reshaped = np.array([pf.flatten() for pf in perp_forces])
                self.perp_forces_list.append(perp_forces_reshaped)

                # Calculate norm and rms of the perpendicular force for every node/image
                to_check = {
                    "norm": np.linalg.norm(perp_forces_reshaped, axis=1),
                    "rms": np.sqrt(np.mean(perp_forces_reshaped**2, axis=1)),
                }

                logger.debug(f"Checking frontier node convergence, threshold={perp_thresh:.6f}")

                # We can add a new node if the norm/rms of the perpendicular force is below
                # the threshold. Also allow growth if string is small and forces are improving.
                def converged(i: int) -> bool:
                    if i >= len(to_check[self.reparam_check]):
                        return False
                    cur_val: float = float(to_check[self.reparam_check][i])

                    # Strict convergence check
                    is_converged_strict: bool = cur_val <= perp_thresh

                    # Relaxed convergence: allow growth if string is small (< 6 images) and
                    # forces are below relaxed threshold, or if forces are decreasing
                    is_converged_relaxed = False
                    if self.string_size < 6:
                        is_converged_relaxed = cur_val <= self.perp_thresh_relaxed
                    else:
                        # Check if forces are decreasing (comparing to previous iteration)
                        if len(self.perp_forces_list) > 1:
                            prev_forces = self.perp_forces_list[-2]
                            if i < len(prev_forces):
                                prev_val = float(
                                    np.linalg.norm(prev_forces[i])
                                    if self.reparam_check == "norm"
                                    else np.sqrt(np.mean(prev_forces[i] ** 2))
                                )
                                # Allow growth if forces decreased by at least 10%
                                if cur_val < prev_val * 0.9:
                                    is_converged_relaxed = True

                    is_converged = is_converged_strict or is_converged_relaxed
                    conv_str = ", converged" if is_converged else ""
                    logger.debug(
                        f"\tnode {i:02d}: {self.reparam_check}(perp_forces)={cur_val:.6f}{conv_str}"
                    )
                    return bool(is_converged)

                # New images are added with the same coordinates as the frontier image.
                # We force reparametrization by setting self.reparam_in to 0 to get sane
                # coordinates for the new image(s).
                if converged(self.lf_ind):
                    # Insert at the end of the left string, just before the right frontier node
                    new_left_frontier = self.get_new_image(self.lf_ind)
                    self.new_image_inds.append(self.left_size)
                    self.left_string.append(new_left_frontier)
                    self.images = self.left_string + self.right_string
                    logger.info("Added new left frontier node.")
                    self.reparam_in = 0

                # If an image was just grown in the left substring the string may now
                # be fully grown, so we re-evaluate 'self.fully_grown' here
                if (not self.fully_grown) and converged(self.rf_ind):
                    # Insert at the end of the right string, just before the current right frontier node
                    # Match pysisyphus: append to right_string (grows forward from product)
                    new_right_frontier = self.get_new_image(self.rf_ind)
                    self.new_image_inds.append(self.left_size)
                    self.right_string.append(new_right_frontier)
                    self.images = self.left_string + self.right_string
                    logger.info("Added new right frontier node.")
                    self.reparam_in = 0

                logger.debug(f"New image indices: {self.new_image_inds}")

        logger.debug(
            f"Current string size is {self.left_size}+{self.right_size}="
            f"{self.string_size}. There are still {self.nodes_missing} "
            "nodes to be grown."
            if not self.fully_grown
            else "String is fully grown."
        )

        if self.reparam_in > 0:
            logger.debug(
                f"Skipping reparametrization. Next reparametrization in {self.reparam_in} cycles."
            )
        else:
            # Prepare image reparametrization
            desired_param_density = self.sk * self.full_string_image_inds
            pd_str = np.array2string(desired_param_density, precision=4)
            logger.debug(f"Desired param density: {pd_str}")

            # Reparametrize images
            self.reparam_cart(desired_param_density, image_energies)

            self.reparam_in = reparam_every_full if self.fully_grown else reparam_every
            reparametrized = True

        return reparametrized

    def optimize_perpendicular(self, fmax: float, max_steps: int = 20) -> None:
        """Optimize images perpendicular to the path using adaptive steepest descent.

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

            # Adaptive step size based on force magnitude
            # Use larger steps for larger forces, but cap at reasonable maximum
            initial_step_size = min(0.1, max(0.01, max_perp_force * 0.001))
            step_size = initial_step_size
            min_step_size = 1e-4
            max_step_size = 0.1

            previous_energy = atoms.get_potential_energy()
            for _step_iter in range(max_steps):
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

                # Normalize perpendicular force for step direction
                perp_norm = np.linalg.norm(perp_force_reshaped)
                if perp_norm < 1e-10:
                    break

                # Take adaptive step along perpendicular forces
                step = step_size * (perp_force_reshaped / perp_norm)
                old_positions = atoms.positions.copy()
                atoms.positions += step

                # Recalculate energy/forces
                try:
                    new_energy = atoms.get_potential_energy()
                    energy_change = new_energy - previous_energy

                    # Adaptive step size: increase if energy decreases, decrease if it increases
                    if energy_change < 0:
                        step_size = min(max_step_size, step_size * 1.2)
                        previous_energy = new_energy
                    else:
                        # Energy increased, reject step and reduce step size
                        atoms.positions = old_positions
                        step_size = max(min_step_size, step_size * 0.5)
                        if step_size < min_step_size:
                            break
                except Exception:
                    # If calculation fails, revert step
                    atoms.positions = old_positions
                    step_size = max(min_step_size, step_size * 0.5)
                    if step_size < min_step_size:
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

        Returns
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
        # Use more lenient threshold for growth - allow growth if forces are improving
        perp_thresh = kwargs.get("perp_thresh", 0.05)
        # Fallback: allow growth if string is small and forces aren't too high
        perp_thresh_relaxed = kwargs.get("perp_thresh_relaxed", 1.0)
        reparam_every = kwargs.get("reparam_every", 2)
        reparam_every_full = kwargs.get("reparam_every_full", 3)
        optimize_endpoints = kwargs.get("optimize_endpoints", True)
        refine_ts = kwargs.get("refine_ts", True)
        local_optimizer_name = kwargs.get("local_optimizer_name", "sella")
        self.reparam_check = kwargs.get("reparam_check", "rms")  # "rms" or "norm"
        self.perp_thresh_relaxed = perp_thresh_relaxed

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
                [reactant], fmax=fmax, steps=200, local_optimizer_name="lbfgs"
            )
            reactant = ensure_atoms(r_result["optimized_atoms"], "Local minima optimization")

            logger.info("Growing String: Optimizing product to local minimum...")
            p_result = minima_strategy.run(
                [product], fmax=fmax, steps=200, local_optimizer_name="lbfgs"
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

        # If starting with 2 images, immediately grow one node on each side
        # This matches pysisyphus behavior
        if len(self.images) == 2:
            logger.info("Growing String: Starting with 2 images, growing initial nodes...")
            # Grow left frontier first (from reactant toward center)
            left_frontier = self.get_new_image(self.lf_ind)
            self.left_string.append(left_frontier)
            # Update images after left growth
            self.images = self.left_string + self.right_string
            # Now grow right frontier (from product toward center)
            # rf_ind is now valid after updating images
            right_frontier = self.get_new_image(self.rf_ind)
            self.right_string.append(right_frontier)
            self.images = self.left_string + self.right_string

            # Ensure calculators are attached to new images
            if self.explorer is not None:
                PathManager.attach_calculators(self.explorer, self.images)
                for atoms in self.images:
                    StrategyUtils.ensure_charge_spin_info(atoms)

            logger.info(
                f"Growing String: Initial growth complete. Now have {len(self.images)} images"
            )

        # Main GSM loop
        strings_met = False
        for iteration in range(max_steps):
            # Check if fully grown
            if self.fully_grown:
                logger.info(
                    f"Growing String: Reached maximum images ({self.max_nodes}). "
                    f"String size: {self.left_size}+{self.right_size}={self.string_size}"
                )
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
            # Only consider strings met if we have enough images to form a reasonable path
            if len(self.left_string) > 0 and len(self.right_string) > 0:
                forward_tip = self.left_string[-1].positions
                backward_tip = self.right_string[0].positions
                distance = np.linalg.norm(forward_tip - backward_tip)

                distance_threshold = kwargs.get("distance_threshold", 0.05)
                # Only mark as met if we have at least 6 images (3 per string) or are fully grown
                # This prevents premature termination when strings are still far from TS
                min_images_for_convergence = min(6, self.max_nodes // 2)
                if distance < distance_threshold and (
                    self.string_size >= min_images_for_convergence or self.fully_grown
                ):
                    strings_met = True
                    logger.info(
                        f"Strings met: distance={distance:.6f} Å < threshold={distance_threshold:.6f} Å, "
                        f"string_size={self.string_size}"
                    )

            # Optimize perpendicular to path first
            self.optimize_perpendicular(fmax, max_steps=20)

            # Reparametrize and potentially add new nodes
            self.reparametrize(
                energies, forces_list, perp_thresh, reparam_every, reparam_every_full
            )

            # Log progress
            if (iteration + 1) % 10 == 0:
                # Note: self.fully_grown check is redundant here since we break above if fully_grown
                size_str = f"{self.left_size}+{self.right_size}"
                size_info = f"String={size_str: >5s}"
                barrier_info = ""
                hei_info = ""
                hei_str = ""
                if len(energies) > 0:
                    energies_arr = np.array(energies)
                    barrier = energies_arr.max() - energies_arr[0]  # Energy difference in eV
                    barrier_info = f"(E_hei-E_0)={barrier:6.1f} eV"
                    hei_ind = energies_arr.argmax()
                    if len(forces_list) > hei_ind:
                        hei_norm = np.linalg.norm(forces_list[hei_ind])
                        hei_info = f"norm(forces_true,hei)={hei_norm:.6f} eV/Å"
                    hei_str = f"HEI={hei_ind + 1:02d}/{energies_arr.size:02d}"

                strs = [size_info]
                if hei_str:
                    strs.append(hei_str)
                if barrier_info:
                    strs.append(barrier_info)
                if hei_info:
                    strs.append(hei_info)

                logger.info(f"Growing String: Iteration {iteration + 1}, " + "\t".join(strs))

        # Combine strings into full path
        # right_string grows forward from product [P, R1, R2, ...] where R2 is closest to center
        # In the final path, right_string nodes appear in reverse: ... -> R2 -> R1 -> P
        full_path = self.left_string + self.right_string[::-1]

        logger.info(
            f"Growing String: Complete with {len(full_path)} images "
            f"(left: {len(self.left_string)}, right: {len(self.right_string)}, "
            f"strings_met={strings_met})"
        )

        # Filter redundant structures for consistency with other path strategies
        # For GSM we retain all images to avoid collapsing back onto endpoints

        # Find TS as highest energy image
        # Try to find a structure with imaginary frequencies if possible
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
                # Select highest energy image as TS guess
                # (Checking frequencies for all candidates is too expensive)
                valid_energies.sort(key=lambda x: x[1], reverse=True)
                ts_index = valid_energies[0][0]
                logger.info(
                    f"Growing String: Selected TS guess at image {ts_index} "
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

        freq_analysis: dict[str, Any] | None = None
        if ts_result is not None and isinstance(ts_result, dict):
            # ts_result is dict[str, Atoms | list[Atoms] | bool | int | float | str]
            # but frequency_analysis is actually dict[str, Any], so we need to handle this
            freq_analysis_val: Any = ts_result.get("frequency_analysis")
            # Type narrowing: check if it's a dict before assigning
            if freq_analysis_val is not None and isinstance(freq_analysis_val, dict):
                freq_analysis = freq_analysis_val

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
