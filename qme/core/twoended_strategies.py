"""Two-ended strategy runners for Explorer.

This module provides runners that operate on two or more ASE Atoms objects.
Input can be either exactly two Atoms (reactant, product) or a sequence of Atoms
interpreted as [reactant, intermediate_guess..., product]. When the sequence contains
more than two states the module will interpolate each consecutive pair and stitch
the segments into a single path.

Two-ended strategies work with multiple structures and perform interpolation/optimization:
- ts:interpolate - TS guess from interpolated path with local TS refinement
- minima:interpolate - Minima optimization on interpolated path frames
- path:neb - NEB path optimization with geodesic interpolation
- path:cineb - CI-NEB path optimization with geodesic interpolation
- path:interpolate - Generate interpolated path only (no optimization)

These strategies are registered in Explorer with the new target:strategy naming scheme.
"""

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from ase import Atoms
from ase.optimize import BFGS
from ase.optimize.optimize import Optimizer

from qme.core.local_strategies import _get_local_optimizer_class
from qme.core.path_manager import PathManager
from qme.core.strategy import REGISTRY, BaseStrategy, StrategyMetadata
from qme.core.strategy_utils import StrategyUtils
from qme.logging_utils import get_qme_logger

logger = get_qme_logger(__name__)


def _supports_batch_evaluation(calculator: Any) -> bool:
    """Check if calculator supports batch evaluation (delegates to StrategyUtils)."""
    return StrategyUtils.supports_batch_evaluation(calculator)


def _ensure_charge_spin_info(atoms: Atoms, default_charge: int = 0, default_spin: int = 1) -> None:
    """Ensure atoms.info has charge and spin info (delegates to StrategyUtils)."""
    StrategyUtils.ensure_charge_spin_info(atoms, default_charge, default_spin)


# =============================================================================
# NEB Optimizer Class
# =============================================================================

class NEBOptimizer(Optimizer):
    """
    ASE-compatible NEB optimizer that manages NEB force computation internally.

    This optimizer wraps a list of Atoms images and computes NEB forces (spring + projected)
    in each step, allowing standard ASE optimizers to operate on the NEB system.
    The 'steps' parameter now controls total optimizer iterations, not outer NEB loops.
    """

    def __init__(
        self,
        images: list[Atoms],
        spring_constant: float = 5.0,
        climb: bool = False,
        optimizer_class: type[Optimizer] = BFGS,
        fmax: float = 0.05,
        **kwargs: Any,
    ):
        """Initialize NEB optimizer.

        Parameters
        ----------
        images : list[Atoms]
            List of Atoms objects representing the NEB path
        spring_constant : float
            Spring constant for NEB spring forces
        climb : bool
            Whether to use climbing image NEB (CI-NEB)
        optimizer_class : type[Optimizer]
            ASE optimizer class to use for internal image optimization
        fmax : float
            Force convergence criterion
        **kwargs
            Additional arguments passed to the wrapped optimizer
        """
        # Store NEB-specific parameters
        self.images = images
        self.spring_constant = spring_constant
        self.climb = climb
        self.optimizer_class = optimizer_class
        self.fmax = fmax

        # Internal images (excluding endpoints)
        self.internal_images = images[1:-1] if len(images) > 2 else []
        self.n_internal = len(self.internal_images)

        # Check if we can use batch evaluation
        calculator = images[0].calc if images[0].calc is not None else None
        self.supports_batch = _check_batch_support(calculator)

        # Climbing image tracking
        self.climbing_image = None

        # Initialize base optimizer with the first internal image
        # (we'll override get_forces() to return NEB forces)
        if self.internal_images:
            super().__init__(self.internal_images[0], **kwargs)
        else:
            # Fallback for edge case with only 2 images
            super().__init__(images[0], **kwargs)

        if self.supports_batch:
            logger.info("Using batch evaluation for NEB optimization with %d images", len(images))

        if climb:
            logger.info("Starting CI-NEB optimization with %d images", len(images))
        else:
            logger.info("Starting NEB optimization with %d images", len(images))

    def get_forces(self) -> np.ndarray:
        """Return concatenated NEB forces for all internal images."""
        if not self.internal_images:
            return np.array([])

        # Calculate energies and forces for all images
        energies, forces_list = self._calculate_energies_forces()

        # Compute NEB forces for internal images
        neb_forces = []
        for i in range(1, len(self.images) - 1):  # Internal images only
            atoms = self.images[i]
            forces = forces_list[i].copy()

            # Spring forces
            spring_force = self._compute_spring_force(i, energies)

            # Project forces perpendicular to path (nudging)
            tangent = self._compute_tangent(i, energies)
            if tangent is not None:
                # Remove component parallel to tangent
                parallel_component = np.dot(forces.flatten(), tangent) * tangent
                forces = forces.flatten() - parallel_component
                forces = forces.reshape(-1, 3)

            # Apply climbing image behavior if this is the climbing image
            if self.climb and self.climbing_image == i:
                forces = self._apply_climbing_image(forces, forces_list[i], tangent)

            # Add spring forces
            forces += spring_force

            # Store NEB forces in calculator results
            self._store_forces(atoms, forces)

            # Collect forces for concatenated return
            neb_forces.append(forces.flatten())

        return np.concatenate(neb_forces) if neb_forces else np.array([])

    def step(self) -> None:
        """Perform one NEB optimization step."""
        if not self.internal_images:
            return

        # Calculate energies and forces for all images
        energies, forces_list = self._calculate_energies_forces()

        # Determine climbing image for CI-NEB
        if self.climb:
            self.climbing_image = self._determine_climbing_image(energies)

        # Compute and apply NEB forces to internal images
        for i in range(1, len(self.images) - 1):  # Internal images only
            atoms = self.images[i]
            forces = forces_list[i].copy()

            # Spring forces
            spring_force = self._compute_spring_force(i, energies)

            # Project forces perpendicular to path (nudging)
            tangent = self._compute_tangent(i, energies)
            if tangent is not None:
                # Remove component parallel to tangent
                parallel_component = np.dot(forces.flatten(), tangent) * tangent
                forces = forces.flatten() - parallel_component
                forces = forces.reshape(-1, 3)

            # Apply climbing image behavior if this is the climbing image
            if self.climb and self.climbing_image == i:
                forces = self._apply_climbing_image(forces, forces_list[i], tangent)

            # Add spring forces
            forces += spring_force

            # Store NEB forces in calculator results
            self._store_forces(atoms, forces)

            # Update positions using simple steepest descent
            # This is a simplified approach - in practice, you might want to use
            # a more sophisticated optimizer like FIRE or LBFGS
            step_size = 0.01  # Small step size for stability
            atoms.positions -= step_size * forces

    def converged(self, forces: np.ndarray | None = None) -> bool:
        """Check if NEB optimization has converged."""
        if not self.internal_images:
            return True

        # Get current forces from internal images
        if forces is None:
            forces = self.get_forces()

        # Reshape forces to per-image format
        n_atoms = len(self.internal_images[0])
        forces_reshaped = forces.reshape(self.n_internal, n_atoms, 3)

        # Check convergence for each internal image
        max_force = 0.0
        for img_forces in forces_reshaped:
            max_force = max(max_force, np.max(np.abs(img_forces)))

        return max_force < self.fmax

    def _calculate_energies_forces(self) -> tuple[list[float], list[np.ndarray]]:
        """Calculate energies and forces for all images."""
        if self.supports_batch:
            return self._calculate_batch_energies_forces()
        else:
            return self._calculate_individual_energies_forces()

    def _calculate_batch_energies_forces(self) -> tuple[list[float], list[np.ndarray]]:
        """Calculate energies and forces using batch evaluation."""
        calculator = self.images[0].calc
        energies = []
        forces_list = []

        try:
            batch_results = calculator.calculate_batch(self.images, properties=["energy", "forces"])

            for result in batch_results:
                energies.append(result.get("energy", float("inf")))
                forces_list.append(result.get("forces", np.zeros((len(self.images[0]), 3))))

        except (RuntimeError, AttributeError, TypeError) as e:
            logger.warning(f"Batch evaluation failed, falling back to individual calculations: {e}")
            self.supports_batch = False
            return self._calculate_individual_energies_forces()

        return energies, forces_list

    def _calculate_individual_energies_forces(self) -> tuple[list[float], list[np.ndarray]]:
        """Calculate energies and forces individually for each image."""
        energies = []
        forces_list = []

        for atoms in self.images:
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

    def _compute_spring_force(self, index: int, energies: list[float]) -> np.ndarray:
        """Compute spring force for image at given index."""
        return StrategyUtils.calculate_spring_force(self.images, index, self.spring_constant, energies)

    def _compute_tangent(self, index: int, energies: list[float]) -> np.ndarray | None:
        """Compute tangent vector for image at given index."""
        return StrategyUtils.calculate_tangent(self.images, index, energies)

    def _determine_climbing_image(self, energies: list[float]) -> int | None:
        """Determine which image should be the climbing image."""
        if not self.climb or len(energies) <= 2:
            return None

        # Find highest energy image (excluding endpoints)
        valid_energies = [(i, e) for i, e in enumerate(energies[1:-1], 1) if not np.isnan(e)]
        if not valid_energies:
            return None

        return max(valid_energies, key=lambda x: x[1])[0]

    def _apply_climbing_image(self, forces: np.ndarray, original_forces: np.ndarray, tangent: np.ndarray | None) -> np.ndarray:
        """Apply climbing image behavior by inverting parallel component."""
        if tangent is not None:
            parallel_component = np.dot(original_forces.flatten(), tangent) * tangent
            parallel_component = parallel_component.reshape(-1, 3)
            # Invert parallel component (make it point uphill)
            forces = forces - 2 * parallel_component
        return forces

    def _store_forces(self, atoms: Atoms, forces: np.ndarray) -> None:
        """Store forces in calculator results (ASE-compatible approach)."""
        if atoms.calc is not None:
            if not hasattr(atoms.calc, "results"):
                atoms.calc.results = {}
            atoms.calc.results["forces"] = forces
        else:
            # Fallback: store in atoms info for compatibility
            atoms.info["forces"] = forces


# =============================================================================
# Strategy Classes
# =============================================================================

class TwoEndedTSGuessStrategy(BaseStrategy):
    """Two-ended TS guess strategy via interpolation with local TS refinement."""

    metadata = StrategyMetadata(
        name="ts:interpolate",
        target="ts",
        strategy="interpolate",
        description="TS guess via interpolation with local TS refinement",
        aliases=["twoended:ts", "twoended-ts"],
        requires_multiple_structures=True,
    )

    def run(self, atoms_list: list[Atoms], npoints: int = 11, method: str = "geodesic",
            fmax: float = 0.05, steps: int = 1000, validate_ts: bool = False, **kwargs) -> dict[str, Any]:
        """Run two-ended TS guess via interpolation.

        Parameters
        ----------
        atoms_list : list[Atoms]
            List of structures defining the path endpoints
        npoints : int, default=11
            Number of images in the interpolated path
        method : str, default="geodesic"
            Interpolation method for initial path generation
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

        # Generate interpolated path using PathManager
        path_mgr = PathManager(atoms_list)
        # Filter out mode parameter which is not accepted by PathManager.interpolate
        interpolate_kwargs = {k: v for k, v in kwargs.items() if k != "mode"}
        path = path_mgr.interpolate(
            npoints=npoints,
            method=method,
            optimize_path=False,  # Raw interpolation
            explorer=self.explorer,
            **interpolate_kwargs,
        )

        # Attach calculators to all images
        if self.explorer is not None:
            PathManager.attach_calculators(self.explorer, path)

        # Find highest energy structure as TS guess using PathManager
        ts_guess, ts_index = PathManager.find_ts_guess(path)

        # Refine with local TS optimization
        from qme.core.local_strategies import LocalTSStrategy

        ts_strategy = LocalTSStrategy(self.explorer)
        ts_result = ts_strategy.run(
            ts_guess,
            fmax=fmax,
            steps=steps,
            local_optimizer_name=local_optimizer_name,
            **kwargs,
        )

        # Validate TS structure if requested
        validation_result = None
        if validate_ts:
            from qme.core.local_strategies import validate_ts_structure
            validation_result = validate_ts_structure(ts_result["optimized_atoms"], self.explorer)

        # Return single TS structure
        result = self.prepare_result(
            ts_result["optimized_atoms"],
            steps_taken=ts_result["steps_taken"],
            converged=ts_result["converged"],
        )
        if validation_result is not None:
            result["ts_validation"] = validation_result
        return result


# Register the strategy
REGISTRY.register(TwoEndedTSGuessStrategy)


class PathInterpolateStrategy(BaseStrategy):
    """Generate interpolated path only (no optimization)."""

    metadata = StrategyMetadata(
        name="path:interpolate",
        target="path",
        strategy="interpolate",
        description="Generate interpolated path only (no optimization)",
        aliases=["interpolate", "twoended:interpolate", "twoended-interpolate"],
        requires_multiple_structures=True,
    )

    def run(self, atoms_list: list[Atoms], npoints: int = 11, method: str = "geodesic", **kwargs) -> dict[str, Any]:
        """Generate interpolated path using PathManager without optimization."""
        self.validate_inputs(atoms_list)

        # Use PathManager to interpolate; do not optimize path
        path_mgr = PathManager(atoms_list)
        # Filter out 'mode' if present
        # Only forward kwargs accepted by PathManager.interpolate; drop unrelated CLI args
        allowed_keys = {"rmsd_threshold", "energy_threshold", "calculator"}
        interpolate_kwargs = {
            k: v for k, v in kwargs.items() if False  # drop mode entirely
        }
        # Keep only allowed keys from kwargs
        for k in list(kwargs.keys()):
            if k in allowed_keys:
                interpolate_kwargs[k] = kwargs[k]
        path = path_mgr.interpolate(
            npoints=npoints,
            method=method,
            optimize_path=False,
            explorer=self.explorer,
            **interpolate_kwargs,
        )

        # Attach calculators to images if explorer is provided (for downstream usage)
        if self.explorer is not None:
            try:
                PathManager.attach_calculators(self.explorer, path)
            except Exception:
                pass

        return self.prepare_result(
            optimized_atoms=path,
            converged=True,
            trajectory=path,
        )


# Register the strategy
REGISTRY.register(PathInterpolateStrategy)

class TwoEndedNEBStrategy(BaseStrategy):
    """NEB (Nudged Elastic Band) path optimization strategy."""

    metadata = StrategyMetadata(
        name="path:neb",
        target="path",
        strategy="neb",
        description="NEB path optimization with geodesic interpolation",
        aliases=["twoended:neb", "twoended-neb", "neb"],
        requires_multiple_structures=True,
    )

    def run(self, atoms_list: list[Atoms], npoints: int = 11, method: str = "geodesic",
            fmax: float = 0.05, steps: int = 1000, spring_constant: float = 5.0, **kwargs) -> dict[str, Any]:
        """Run NEB path optimization.

        Parameters
        ----------
        atoms_list : list[Atoms]
            List of structures defining the path endpoints
        npoints : int, default=11
            Number of images in the NEB path
        method : str, default="geodesic"
            Interpolation method for initial path generation
        fmax : float, default=0.05
            Force convergence threshold
        steps : int, default=1000
            Maximum optimization steps
        spring_constant : float, default=5.0
            Spring constant for NEB spring forces
        **kwargs
            Additional keyword arguments

        Returns
        -------
        dict[str, Any]
            Standardized result dictionary
        """
        self.validate_inputs(atoms_list)

        local_optimizer_name = kwargs.get("local_optimizer_name", "sella")

        # Check if we should use batch evaluation
        calculator = None
        if self.explorer is not None:
            # Create calculator using explorer's parameters
            from qme.core.calculator_setup import create_calculator

            calculator = create_calculator(
                backend=self.explorer.backend,
                model_name=self.explorer.model_name,
                model_path=self.explorer.model_path,
                device=self.explorer.device,
                default_charge=self.explorer.default_charge,
                default_spin=self.explorer.default_spin,
                verbose=self.explorer.verbose,
            )

        if calculator is not None and StrategyUtils.supports_batch_evaluation(calculator):
            # Use batch NEB optimizer
            return self._run_batch_neb(atoms_list, calculator, npoints, method, fmax, steps, spring_constant, **kwargs)

        # Generate initial path using PathManager
        path_mgr = PathManager(atoms_list)
        # Filter out mode parameter which is not accepted by PathManager.interpolate
        interpolate_kwargs = {k: v for k, v in kwargs.items() if k != "mode"}
        path = path_mgr.interpolate(
            npoints=npoints,
            method=method,
            optimize_path=False,  # Don't optimize initially, we'll do NEB
            explorer=self.explorer,
            **interpolate_kwargs,
        )

        # Flatten nested segments if needed
        if path and isinstance(path[0], list):
            flat = []
            for seg in path:
                flat.extend(seg)
            path = flat

        if len(path) < 3:
            raise ValueError("NEB requires at least 3 images (npoints >= 3)")

        # Attach calculators to all images
        if self.explorer is not None:
            for atoms in path:
                self.explorer._create_and_attach_calculator(atoms)
                self.explorer._apply_constraints(atoms)

        # NEB implementation (shared helper)
        _neb_optimize(
            path,
            spring_constant=spring_constant,
            fmax=fmax,
            steps=steps,
            local_optimizer_name=local_optimizer_name,
            climb=False,
            **kwargs,
        )

        # Filter redundant structures and issue warnings
        if path:
            # Convert atoms_list to list for comparison
            input_atoms = list(atoms_list) if not isinstance(atoms_list, Atoms) else [atoms_list]

            filtered_path, removed_indices, warnings_list = PathManager.filter_redundant_structures(
                path,
                input_structures=input_atoms,
                rmsd_threshold=kwargs.get("rmsd_threshold", 0.1),
                energy_threshold=kwargs.get("energy_threshold", 0.001),
                strategy_name="twoended:neb",
            )

            # Issue warnings
            for warning_msg in warnings_list:
                warnings.warn(warning_msg, stacklevel=2)

            path = filtered_path

        return self.prepare_result(
            path,
            converged=True,
            trajectory=path,
        )

    def _run_batch_neb(self, atoms_list, calculator, npoints, method, fmax, steps, spring_constant, **kwargs):
        """Run NEB using batch evaluation."""
        # Generate initial path using PathManager
        path_mgr = PathManager(atoms_list, calculator=calculator)
        # Filter out mode parameter which is not accepted by PathManager.interpolate
        interpolate_kwargs = {k: v for k, v in kwargs.items() if k != "mode"}
        path = path_mgr.interpolate(
            npoints=npoints,
            method=method,
            optimize_path=False,  # Don't optimize initially, we'll do NEB
            **interpolate_kwargs,
        )

        # Create batch NEB optimizer
        batch_neb = BatchNEBOptimizer(
            path,
            calculator=calculator,
            fmax=fmax,
            steps=steps,
            spring_constant=spring_constant,
            climb=False,
            **kwargs,
        )

        # Optimize using batch evaluation
        optimized_path = batch_neb.optimize()

        # Filter redundant structures and issue warnings
        if optimized_path:
            # Convert atoms_list to list for comparison
            input_atoms = list(atoms_list) if not isinstance(atoms_list, Atoms) else [atoms_list]

            filtered_path, removed_indices, warnings_list = PathManager.filter_redundant_structures(
                optimized_path,
                input_structures=input_atoms,
                rmsd_threshold=kwargs.get("rmsd_threshold", 0.1),
                energy_threshold=kwargs.get("energy_threshold", 0.001),
                strategy_name="twoended:neb",
            )

            # Issue warnings
            for warning_msg in warnings_list:
                warnings.warn(warning_msg, stacklevel=2)

            optimized_path = filtered_path

        return self.prepare_result(
            optimized_path,
            converged=True,
            trajectory=optimized_path,
        )

    def _run_simple_neb(self, path, spring_constant=5.0, fmax=0.05, steps=1000, local_optimizer_name="sella", **kwargs):
        """Run a simple NEB optimization on the given path."""
        # Setup optimizer
        try:
            OptClass = _get_local_optimizer_class(local_optimizer_name)
        except (ImportError, ValueError, AttributeError) as e:
            warnings.warn(f"Could not select optimizer '{local_optimizer_name}': {e}", stacklevel=2)
            OptClass = BFGS  # Fallback to ASE BFGS

        # Check if we can use batch evaluation
        calculator = path[0].calc if path[0].calc is not None else None
        supports_batch = StrategyUtils.check_batch_support(calculator)

        if supports_batch:
            logger.info("Using batch evaluation for NEB optimization with %d images", len(path))

        # NEB optimization loop
        for step in range(steps):
            # Calculate energies and forces for all images
            energies, forces_list = StrategyUtils.calculate_batch_energies_forces(path, calculator, supports_batch)

            # Check convergence
            if StrategyUtils.check_convergence(forces_list, fmax, step):
                break

            # Apply NEB forces
            self._apply_neb_forces(path, forces_list, energies, spring_constant)

            # Optimize each image (except endpoints)
            for i in range(1, len(path) - 1):
                atoms = path[i]
                try:
                    # Filter out incompatible kwargs for optimizer constructor
                    opt_kwargs = {
                        k: v
                        for k, v in kwargs.items()
                        if k
                        not in [
                            "verbose",
                            "explorer",
                            "local_optimizer_name",
                            "climb",
                            "product",
                            "npoints",
                        ]
                    }
                    opt = OptClass(atoms, **opt_kwargs)
                    opt.run(fmax=fmax, steps=1)  # Single step per iteration
                except (RuntimeError, ValueError, AttributeError) as e:
                    warnings.warn(f"Optimization failed for image {i}: {e}", stacklevel=2)

        return path

    def _apply_neb_forces(self, path, forces_list, energies, spring_constant):
        """Apply NEB forces to path images."""
        for i in range(1, len(path) - 1):  # Skip endpoints
            atoms = path[i]
            forces = forces_list[i].copy()

            # Spring forces
            spring_force = StrategyUtils.calculate_spring_force(path, i, spring_constant, energies)

            # Project forces perpendicular to path (nudging)
            tangent = StrategyUtils.calculate_tangent(path, i, energies)
            if tangent is not None:
                # Remove component parallel to tangent
                parallel_component = np.dot(forces.flatten(), tangent) * tangent
                forces = forces.flatten() - parallel_component
                forces = forces.reshape(-1, 3)

            # Add spring forces
            forces += spring_force

            # Store forces in calculator results (ASE-compatible approach)
            if atoms.calc is not None:
                if not hasattr(atoms.calc, "results"):
                    atoms.calc.results = {}
                atoms.calc.results["forces"] = forces
            else:
                # Fallback: store in atoms info for compatibility
                atoms.info["forces"] = forces


# Register the strategy
REGISTRY.register(TwoEndedNEBStrategy)


class TwoEndedCINEBStrategy(BaseStrategy):
    """CI-NEB (Climbing Image Nudged Elastic Band) path optimization strategy."""

    metadata = StrategyMetadata(
        name="path:cineb",
        target="path",
        strategy="cineb",
        description="Climbing Image NEB (CI-NEB) optimization with geodesic interpolation",
        aliases=["twoended:cineb", "twoended-cineb", "cineb"],
        requires_multiple_structures=True,
    )

    def run(self, atoms_list: list[Atoms], npoints: int = 11, method: str = "geodesic",
            fmax: float = 0.05, steps: int = 1000, spring_constant: float = 5.0,
            climb: bool = True, **kwargs) -> dict[str, Any]:
        """Run CI-NEB path optimization.

        Parameters
        ----------
        atoms_list : list[Atoms]
            List of structures defining the path endpoints
        npoints : int, default=11
            Number of images in the CI-NEB path
        method : str, default="geodesic"
            Interpolation method for initial path generation
        fmax : float, default=0.05
            Force convergence threshold
        steps : int, default=1000
            Maximum optimization steps
        spring_constant : float, default=5.0
            Spring constant for NEB spring forces
        climb : bool, default=True
            Whether to enable climbing image behavior
        **kwargs
            Additional keyword arguments

        Returns
        -------
        dict[str, Any]
            Standardized result dictionary
        """
        self.validate_inputs(atoms_list)

        local_optimizer_name = kwargs.get("local_optimizer_name", "sella")

        # Check if we should use batch evaluation
        calculator = None
        if self.explorer is not None:
            # Create calculator using explorer's parameters
            from qme.core.calculator_setup import create_calculator

            calculator = create_calculator(
                backend=self.explorer.backend,
                model_name=self.explorer.model_name,
                model_path=self.explorer.model_path,
                device=self.explorer.device,
                default_charge=self.explorer.default_charge,
                default_spin=self.explorer.default_spin,
                verbose=self.explorer.verbose,
            )

        if calculator is not None and StrategyUtils.supports_batch_evaluation(calculator):
            # Use batch NEB optimizer with climbing enabled
            return self._run_batch_cineb(atoms_list, calculator, npoints, method, fmax, steps, spring_constant, climb, **kwargs)

        # Generate initial path using PathManager
        path_mgr = PathManager(atoms_list)
        # Filter out mode parameter which is not accepted by PathManager.interpolate
        interpolate_kwargs = {k: v for k, v in kwargs.items() if k != "mode"}
        path = path_mgr.interpolate(
            npoints=npoints,
            method=method,
            optimize_path=False,  # Don't optimize initially, we'll do CI-NEB
            explorer=self.explorer,
            **interpolate_kwargs,
        )

        # Flatten nested segments if needed
        if path and isinstance(path[0], list):
            flat = []
            for seg in path:
                flat.extend(seg)
            path = flat

        if len(path) < 3:
            raise ValueError("CI-NEB requires at least 3 images (npoints >= 3)")

        # Attach calculators to all images
        if self.explorer is not None:
            for i, atoms in enumerate(path):
                try:
                    # Only create and attach calculator if atoms doesn't already have one
                    if getattr(atoms, "calc", None) is None:
                        self.explorer._create_and_attach_calculator(atoms)
                    self.explorer._apply_constraints(atoms)
                except Exception as e:
                    warnings.warn(f"Failed to attach calculator to image {i}: {e}", stacklevel=2)

        # Run CI-NEB optimization (shared helper)
        _neb_optimize(
            path,
            spring_constant=spring_constant,
            fmax=fmax,
            steps=steps,
            local_optimizer_name=local_optimizer_name,
            climb=climb,
            **kwargs,
        )

        # Filter redundant structures and issue warnings
        if path:
            # Convert atoms_list to list for comparison
            input_atoms = list(atoms_list) if not isinstance(atoms_list, Atoms) else [atoms_list]

            filtered_path, removed_indices, warnings_list = PathManager.filter_redundant_structures(
                path,
                input_structures=input_atoms,
                rmsd_threshold=kwargs.get("rmsd_threshold", 0.1),
                energy_threshold=kwargs.get("energy_threshold", 0.001),
                strategy_name="twoended:cineb",
            )

            # Issue warnings
            for warning_msg in warnings_list:
                warnings.warn(warning_msg, stacklevel=2)

            path = filtered_path

        return self.prepare_result(
            path,
            converged=True,
            trajectory=path,
        )

    def _run_batch_cineb(self, atoms_list, calculator, npoints, method, fmax, steps, spring_constant, climb, **kwargs):
        """Run CI-NEB using batch evaluation."""
        # Generate initial path using PathManager
        path_mgr = PathManager(atoms_list, calculator=calculator)
        # Filter out mode parameter which is not accepted by PathManager.interpolate
        interpolate_kwargs = {k: v for k, v in kwargs.items() if k != "mode"}
        path = path_mgr.interpolate(
            npoints=npoints,
            method=method,
            optimize_path=False,  # Don't optimize initially, we'll do CI-NEB
            **interpolate_kwargs,
        )

        # Create batch NEB optimizer with climbing enabled
        batch_neb = BatchNEBOptimizer(
            path,
            calculator=calculator,
            fmax=fmax,
            steps=steps,
            spring_constant=spring_constant,
            climb=climb,
            **kwargs,
        )

        # Optimize using batch evaluation
        optimized_path = batch_neb.optimize()

        # Filter redundant structures and issue warnings
        if optimized_path:
            # Convert atoms_list to list for comparison
            input_atoms = list(atoms_list) if not isinstance(atoms_list, Atoms) else [atoms_list]

            filtered_path, removed_indices, warnings_list = PathManager.filter_redundant_structures(
                optimized_path,
                input_structures=input_atoms,
                rmsd_threshold=kwargs.get("rmsd_threshold", 0.1),
                energy_threshold=kwargs.get("energy_threshold", 0.001),
                strategy_name="twoended:cineb",
            )

            # Issue warnings
            for warning_msg in warnings_list:
                warnings.warn(warning_msg, stacklevel=2)

            optimized_path = filtered_path

        return self.prepare_result(
            optimized_path,
            converged=True,
            trajectory=optimized_path,
        )

    def _run_cineb(self, path, spring_constant=5.0, fmax=0.05, steps=1000, local_optimizer_name="sella", climb=True, **kwargs):
        """Run a CI-NEB optimization on the given path."""
        # Setup optimizer
        try:
            OptClass = _get_local_optimizer_class(local_optimizer_name)
        except (ImportError, ValueError, AttributeError) as e:
            warnings.warn(f"Could not select optimizer '{local_optimizer_name}': {e}", stacklevel=2)
            OptClass = BFGS  # Fallback to ASE BFGS

        # Check if we can use batch evaluation
        calculator = path[0].calc if path[0].calc is not None else None
        supports_batch = StrategyUtils.check_batch_support(calculator)

        if supports_batch:
            logger.info("Using batch evaluation for CI-NEB optimization with %d images", len(path))

        climbing_image = None
        logger.info(
            "Starting CI-NEB optimization with %d images (climb=%s)",
            len(path),
            "enabled" if climb else "disabled",
        )

        # CI-NEB optimization loop
        for step in range(steps):
            # Calculate energies and forces for all images
            energies, forces_list = StrategyUtils.calculate_batch_energies_forces(path, calculator, supports_batch)

            # Determine climbing image
            climbing_image = self._determine_climbing_image(energies, climb)

            # Check convergence
            if StrategyUtils.check_convergence(forces_list, fmax, step):
                if climbing_image is not None:
                    logger.info("Climbing image was image %d", climbing_image)
                break

            # Apply CI-NEB forces
            self._apply_cineb_forces(path, forces_list, energies, spring_constant, climbing_image)

            # Optimize each image (except endpoints)
            for i in range(1, len(path) - 1):
                atoms = path[i]
                try:
                    # Filter out incompatible kwargs for optimizer constructor
                    opt_kwargs = {
                        k: v
                        for k, v in kwargs.items()
                        if k
                        not in [
                            "verbose",
                            "explorer",
                            "local_optimizer_name",
                            "climb",
                            "product",
                            "npoints",
                        ]
                    }
                    opt = OptClass(atoms, **opt_kwargs)
                    opt.run(fmax=fmax, steps=1)  # Single step per iteration
                except (RuntimeError, ValueError, AttributeError) as e:
                    warnings.warn(f"Optimization failed for image {i}: {e}", stacklevel=2)

        return path

    def _determine_climbing_image(self, energies, climb):
        """Determine which image should be the climbing image."""
        if not climb or len(energies) <= 2:
            return None

        # Find highest energy image (excluding endpoints)
        valid_energies = [(i, e) for i, e in enumerate(energies[1:-1], 1) if not np.isnan(e)]
        if not valid_energies:
            return None

        climbing_image = max(valid_energies, key=lambda x: x[1])[0]
        return climbing_image

    def _apply_cineb_forces(self, path, forces_list, energies, spring_constant, climbing_image):
        """Apply CI-NEB forces to path images."""
        for i in range(1, len(path) - 1):  # Skip endpoints
            atoms = path[i]
            forces = forces_list[i].copy()

            # Spring forces
            spring_force = StrategyUtils.calculate_spring_force(path, i, spring_constant, energies)

            # Project forces perpendicular to path (nudging)
            tangent = StrategyUtils.calculate_tangent(path, i, energies)
            if tangent is not None:
                # Remove component parallel to tangent
                parallel_component = np.dot(forces.flatten(), tangent) * tangent
                forces = forces.flatten() - parallel_component
                forces = forces.reshape(-1, 3)

            # Apply climbing image behavior if this is the climbing image
            if climbing_image == i:
                # Invert the parallel component for climbing
                if tangent is not None:
                    parallel_component = np.dot(forces_list[i].flatten(), tangent) * tangent
                    parallel_component = parallel_component.reshape(-1, 3)
                    # Invert parallel component (make it point uphill)
                    forces = forces - 2 * parallel_component

            # Add spring forces
            forces += spring_force

            # Store forces in calculator results (ASE-compatible approach)
            if atoms.calc is not None:
                if not hasattr(atoms.calc, "results"):
                    atoms.calc.results = {}
                atoms.calc.results["forces"] = forces
            else:
                # Fallback: store in atoms info for compatibility
                atoms.info["forces"] = forces


# Register the strategy
REGISTRY.register(TwoEndedCINEBStrategy)


class TwoEndedGrowingStringStrategy(BaseStrategy):
    """Growing string method strategy for TS search."""

    metadata = StrategyMetadata(
        name="ts:growing_string",
        target="ts",
        strategy="growing_string",
        description="Growing string method for TS search (DE-GSM style)",
        aliases=["growing_string", "gsm", "twoended:growing_string"],
        requires_multiple_structures=True,
    )

    def run(self, atoms_list: list[Atoms], validate_ts: bool = False, **kwargs) -> dict[str, Any]:
        """Run growing string method using in-module helpers (no runner delegation)."""
        self.validate_inputs(atoms_list)

        # Parse params with defaults matching the function runner
        npoints: int = kwargs.get("npoints", 15)
        fmax: float = kwargs.get("fmax", 0.05)
        steps: int = kwargs.get("steps", 100)
        step_size: float = kwargs.get("step_size", 0.1)
        distance_threshold: float = kwargs.get("distance_threshold", 0.5)
        local_optimizer_name: str = kwargs.get("local_optimizer_name", "sella")
        optimize_endpoints: bool = kwargs.get("optimize_endpoints", True)
        refine_ts: bool = kwargs.get("refine_ts", True)

        # Enforce exactly two structures (reactant, product)
        if isinstance(atoms_list, Atoms):
            raise ValueError("Growing string method requires two Atoms objects (reactant and product)")
        seq = list(atoms_list)
        if len(seq) != 2:
            raise ValueError(f"Growing string method requires exactly 2 Atoms objects, got {len(seq)}")

        reactant, product = seq[0].copy(), seq[1].copy()

        # Attach calculators
        if self.explorer is not None:
            self.explorer._create_and_attach_calculator(reactant)
            self.explorer._apply_constraints(reactant)
            self.explorer._create_and_attach_calculator(product)
            self.explorer._apply_constraints(product)

        # Step 1: Optimize endpoints to local minima if requested
        if optimize_endpoints:
            from qme.core.local_strategies import local_minima_runner

            logger.info("Growing String: Optimizing reactant to local minimum...")
            r_result = local_minima_runner(
                reactant, fmax=fmax, steps=200, explorer=self.explorer, local_optimizer_name="lbfgs"
            )
            reactant = r_result["optimized_atoms"]

            logger.info("Growing String: Optimizing product to local minimum...")
            p_result = local_minima_runner(
                product, fmax=fmax, steps=200, explorer=self.explorer, local_optimizer_name="lbfgs"
            )
            product = p_result["optimized_atoms"]

            # Re-attach calculators after optimization (may have been lost)
            if self.explorer is not None:
                self.explorer._create_and_attach_calculator(reactant)
                self.explorer._apply_constraints(reactant)
                self.explorer._create_and_attach_calculator(product)
                self.explorer._apply_constraints(product)

        # Initialize strings
        forward_string = [reactant.copy()]  # Growing from reactant
        backward_string = [product.copy()]  # Growing from product

        # Ensure calculators are attached to initial string nodes
        if self.explorer is not None:
            for atoms in forward_string:
                self.explorer._create_and_attach_calculator(atoms)
                self.explorer._apply_constraints(atoms)
            for atoms in backward_string:
                self.explorer._create_and_attach_calculator(atoms)
                self.explorer._apply_constraints(atoms)

        logger.info(f"Growing String: Starting with reactant and product, max {npoints} images")

        # Step 2: Grow strings iteratively
        converged = False
        for iteration in range(steps):
            # Check if we've reached max images
            total_images = len(forward_string) + len(backward_string)
            if total_images >= npoints:
                logger.info(f"Growing String: Reached maximum images ({npoints})")
                break

            # Check if strings have met
            forward_tip = forward_string[-1].positions
            backward_tip = backward_string[-1].positions
            distance = np.linalg.norm(forward_tip - backward_tip)

            if distance < distance_threshold:
                logger.info(
                    f"Growing String: Strings met after {iteration} iterations "
                    f"(distance: {distance:.4f} Å)"
                )
                converged = True
                break

            # Grow forward string (from reactant)
            if total_images < npoints:
                new_forward = _grow_string_node(
                    forward_string[-1],
                    direction="forward",
                    step_size=step_size,
                    fmax=fmax,
            explorer=self.explorer,
                )
                if new_forward is not None:
                    forward_string.append(new_forward)

            # Grow backward string (from product)
            total_images = len(forward_string) + len(backward_string)
            if total_images < npoints:
                new_backward = _grow_string_node(
                    backward_string[-1],
                    direction="backward",
                    step_size=step_size,
                    fmax=fmax,
                    explorer=self.explorer,
                )
                if new_backward is not None:
                    backward_string.append(new_backward)

            # Log progress every 10 iterations
            if (iteration + 1) % 10 == 0:
                logger.info(
                    f"Growing String: Iteration {iteration + 1}, "
                    f"forward={len(forward_string)}, backward={len(backward_string)}, "
                    f"distance={distance:.4f} Å"
                )

        # Step 3: Combine strings into full path
        # Reverse backward string so it goes from TS to product
        full_path = forward_string + backward_string[::-1]

        logger.info(
            f"Growing String: Complete with {len(full_path)} images "
            f"(forward: {len(forward_string)}, backward: {len(backward_string)})"
        )

        # Step 4: Find transition state as highest energy image
        energies = []
        for atoms in full_path:
            try:
                energy = atoms.get_potential_energy()
                energies.append(energy)
            except Exception:
                energies.append(float("-inf"))

        if not energies or all(e == float("-inf") for e in energies):
            # Fallback to middle structure
            ts_index = len(full_path) // 2
            logger.warning(
                "Growing String: Could not calculate energies, using middle image as TS guess"
            )
        else:
            ts_index = energies.index(max(energies))
            logger.info(
                f"Growing String: Highest energy at image {ts_index} "
                f"(E = {energies[ts_index]:.6f} eV)"
            )

        ts_guess = full_path[ts_index]

        # Step 5: Optionally refine TS with local optimization
        if refine_ts:
            logger.info("Growing String: Refining TS with local optimization...")
            from qme.core.local_strategies import local_ts_runner

            # Avoid passing duplicate keys via kwargs when calling the runner
            ts_call_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in {"fmax", "steps", "explorer", "local_optimizer_name"}
            }
            ts_result = local_ts_runner(
                ts_guess,
                fmax=fmax,
                steps=1000,
                explorer=self.explorer,
                local_optimizer_name=local_optimizer_name,
                **ts_call_kwargs,
            )
            ts_structure = ts_result["optimized_atoms"]
            ts_converged = ts_result["converged"]
        else:
            ts_structure = ts_guess
            ts_converged = converged

        # Validate TS structure if requested
        validation_result = None
        if validate_ts:
            from qme.core.local_strategies import validate_ts_structure
            validation_result = validate_ts_structure(ts_structure, self.explorer)

        result = self.prepare_result(
            ts_structure,
            converged=ts_converged,
            trajectory=full_path,
            forward_string=forward_string,
            backward_string=backward_string,
            strings_met=converged,
        )
        if validation_result is not None:
            result["ts_validation"] = validation_result
        return result


# Register the strategy
REGISTRY.register(TwoEndedGrowingStringStrategy)


def twoended_ts_guess_runner(
    atoms_list: Sequence[Atoms] | Atoms,
    npoints: int = 11,
    method: str = "geodesic",
    explorer: Any | None = None,
    fmax: float = 0.05,
    steps: int = 1000,
    local_optimizer_name: str = "sella",
    verbose: int = 1,
    **kwargs: Any,
) -> dict[str, Any]:
    """Delegate to TwoEndedTSGuessStrategy to avoid duplication."""
    strategy = TwoEndedTSGuessStrategy(explorer=explorer)
    forwarded = {
        k: v
        for k, v in {
            "npoints": npoints,
            "method": method,
            "fmax": fmax,
            "steps": steps,
            "local_optimizer_name": local_optimizer_name,
            "verbose": verbose,
        **kwargs,
        }.items()
        if k != "explorer"
    }
    return strategy.run(atoms_list, **forwarded)


__all__ = [
    "twoended_ts_guess_runner",
]


def _generate_interpolated_path(
    atoms_list: Sequence[Atoms] | Atoms,
    npoints: int,
    method: str,
    explorer: Any | None,
    **kwargs,
) -> list[Atoms]:
    """Generate interpolated path from atoms list using PathManager."""
    # Create PathManager and generate interpolated path
    path_mgr = PathManager(atoms_list)

    # Attach calculators if explorer provided
    if explorer is not None:
        PathManager.attach_calculators(explorer, path_mgr.structures)

    # Filter out mode parameter which is not accepted by PathManager.interpolate
    interpolate_kwargs = {k: v for k, v in kwargs.items() if k != "mode"}

    # Create calculator for path optimization if explorer is provided
    calculator = None
    if explorer is not None:
        from qme.core.calculator_setup import create_calculator
        calculator = create_calculator(
            backend=explorer.backend,
            model_name=explorer.model_name,
            model_path=explorer.model_path,
            device=explorer.device,
            default_charge=explorer.default_charge,
            default_spin=explorer.default_spin,
        )

    path = path_mgr.interpolate(
        npoints=npoints,
        method=method,
        optimize_path=True,
        calculator=calculator,
        explorer=explorer,
        **interpolate_kwargs,
    )

    if len(path) < 1:
        raise ValueError("Interpolated path is empty; cannot locate minima")

    return path


def _optimize_minima_frames(
    path: list[Atoms],
    minima_idxs: list[int],
    explorer: Any | None,
    fmax: float,
    steps: int,
    local_optimizer_name: str,
) -> list[Atoms]:
    """Optimize minima frames using local optimizer."""
    results = []

    # Select optimizer class
    opt_class = _get_local_optimizer_class(local_optimizer_name)

    for idx in minima_idxs:
        geom = path[idx]
        # CRITICAL FIX: Make a copy of atoms before optimization to prevent in-place modifications
        # that corrupt the coordinate system for subsequent Hessian calculations
        geom_copy = geom.copy()

        # Attach calculator and apply constraints
        if explorer is not None:
            explorer._create_and_attach_calculator(geom_copy)
            explorer._apply_constraints(geom_copy)

        opt_kwargs = getattr(explorer, "optimizer_kwargs", {}) or {}
        if local_optimizer_name.lower() == "sella":
            opt_kwargs = dict(opt_kwargs)
            opt_kwargs.setdefault("internal", True)
            opt_kwargs.setdefault("order", 0)

        opt = opt_class(geom_copy, **opt_kwargs)
        opt.run(fmax=fmax, steps=steps)
        results.append(geom_copy)

    return results


def twoended_minima_runner(
    atoms_list: Sequence[Atoms] | Atoms,
    npoints: int = 11,
    method: str = "geodesic",
    explorer: Any | None = None,
    fmax: float = 0.05,
    steps: int = 1000,
    local_optimizer_name: str = "sella",
    verbose: int = 1,
    **kwargs: Any,
) -> dict[str, Any]:
    """Interpolate path and attempt local minima optimizations on low-energy frames.

    This runner generates an interpolated path, computes approximate
    energies along it, finds local minima (or global min if none), and
    runs a local minima optimization on those frames. Returns optimized
    geometries (single geometry if input was two endpoints, otherwise a list).

    Parameters
    ----------
    atoms_list : Union[Sequence[Atoms], Atoms]
        Two or more Atoms objects defining the path endpoints
    npoints : int, default=11
        Number of interpolation points
    method : str, default="geodesic"
        Interpolation method for initial path generation
    explorer : Any, optional
        Explorer instance for calculator and constraint management
    fmax : float, default=0.05
        Force convergence threshold
    steps : int, default=1000
        Maximum optimization steps
    local_optimizer_name : str, default="sella"
        Local optimizer to use for minima optimization
    rmsd_threshold : float, default=0.1
        RMSD threshold below which structures are considered redundant (Å)
    energy_threshold : float, default=0.001
        Energy threshold below which structures are considered redundant (eV)
    **kwargs
        Additional arguments passed to optimizer

    Returns
    -------
    Union[Atoms, List[Atoms]]
        Single optimized geometry for 2 atoms input, list for 3+ atoms input
    """
    # Generate interpolated path
    path = _generate_interpolated_path(atoms_list, npoints, method, explorer, **kwargs)

    # Find local minima frames using PathManager
    minima_idxs = PathManager.find_local_minima(path)

    # Optimize minima frames
    results = _optimize_minima_frames(
        path, minima_idxs, explorer, fmax, steps, local_optimizer_name
    )

    # Filter redundant structures and issue warnings using PathManager
    if results:
        # Convert atoms_list to list for comparison
        input_atoms = list(atoms_list) if not isinstance(atoms_list, Atoms) else [atoms_list]

        filtered_results, removed_indices, warnings_list = PathManager.filter_redundant_structures(
            results,
            input_structures=input_atoms,
            rmsd_threshold=kwargs.get("rmsd_threshold", 0.1),
            energy_threshold=kwargs.get("energy_threshold", 0.001),
            strategy_name="twoended:minima",
        )

        # Issue warnings
        for warning_msg in warnings_list:
            warnings.warn(warning_msg, stacklevel=2)

        results = filtered_results

    # Return single geometry if original input was pair/endpoints
    if isinstance(atoms_list, Atoms) or (
        isinstance(atoms_list, (list, tuple)) and len(atoms_list) == 2
    ):
        return {
            "optimized_atoms": results[0] if results else None,
            "steps_taken": None,
            "converged": True,
            "strategy": "twoended_minima_runner",
        }
    return {
        "optimized_atoms": results,
        "steps_taken": None,
        "converged": True,
        "strategy": "twoended_minima_runner",
    }


class BatchNEBOptimizer:
    """
    NEB optimizer with batch evaluation support.

    This class implements a NEB optimization that leverages batch evaluation
    capabilities of calculators like TorchSim to calculate energies and forces
    for all NEB images simultaneously. Supports both regular NEB and CI-NEB.
    """

    def __init__(
        self, path, calculator, fmax=0.05, steps=1000, spring_constant=5.0, climb=False, **kwargs
    ):
        """Initialize batch NEB optimizer.

        Parameters
        ----------
        path : list of Atoms
            Initial path for NEB optimization
        calculator : Calculator
            Calculator for energy/force calculations
        fmax : float
            Force convergence threshold
        steps : int
            Maximum optimization steps
        spring_constant : float
            Spring constant for NEB forces
        climb : bool
            Whether to enable CI-NEB climbing behavior
        **kwargs
            Additional optimizer parameters
        """
        self.path = [atoms.copy() for atoms in path]
        self.calculator = calculator
        self.fmax = fmax
        self.steps = steps
        self.spring_constant = spring_constant
        self.climb = climb
        self.kwargs = kwargs
        self.climbing_image = None  # Will be determined dynamically

        # Ensure charge and spin info are set and attach calculator to all images
        for atoms in self.path:
            # Set charge and spin info to prevent UMA backend warnings
            _ensure_charge_spin_info(atoms)
            atoms.calc = calculator

    def optimize(self):
        """Optimize NEB path using batch evaluation."""
        method_name = "CI-NEB" if self.climb else "NEB"
        logger.info(
            f"Optimizing {method_name} path with {len(self.path)} images using batch evaluation..."
        )

        for step in range(self.steps):
            # Calculate forces for all images in one batch
            batch_results = self.calculator.calculate_batch(
                self.path, properties=["energy", "forces"]
            )

            # Extract energies and forces from batch results
            energies = [result.get("energy", float("inf")) for result in batch_results]
            forces = [result["forces"] for result in batch_results]

            # Determine climbing image for CI-NEB
            if self.climb and len(energies) > 2:
                # Find the highest energy image (excluding endpoints)
                valid_energies = [
                    (i, e) for i, e in enumerate(energies[1:-1], 1) if not np.isnan(e)
                ]
                if valid_energies:
                    self.climbing_image = max(valid_energies, key=lambda x: x[1])[0]
                else:
                    self.climbing_image = None

            # Apply NEB forces (spring + nudging + climbing if enabled)
            neb_forces = self._apply_neb_forces(forces, energies)

            # Update positions
            self._update_positions(neb_forces)

            # Check convergence
            max_force = max(np.max(np.abs(force)) for force in neb_forces)
            if max_force < self.fmax:
                logger.info(
                    f"{method_name} converged after {step + 1} steps (max force: {max_force:.6f})"
                )
                if self.climb and self.climbing_image is not None:
                    logger.info("Climbing image was image %d", self.climbing_image)
                break

        return self.path

    def _apply_neb_forces(self, forces, energies=None):
        """Apply NEB forces (spring + nudging + climbing if enabled)."""
        neb_forces = []

        for i in range(len(self.path)):
            if i in (0, len(self.path) - 1):
                # Endpoints: use only spring forces
                neb_forces.append(self._spring_forces(i))
            else:
                # Middle images: spring forces + nudging
                spring_f = self._spring_forces(i)
                nudged_f = self._nudge_forces(forces[i], spring_f)

                # Apply climbing image behavior if enabled and this is the climbing image
                if self.climb and self.climbing_image == i:
                    # Invert the parallel component of the force for climbing
                    climbing_f = self._apply_climbing_forces(forces[i], spring_f, i)
                    neb_forces.append(climbing_f)
                else:
                    neb_forces.append(nudged_f)

        return neb_forces

    def _spring_forces(self, i):
        """Calculate spring forces for image i."""
        if i == 0:
            # First image: spring to next
            return self.spring_constant * (self.path[i + 1].positions - self.path[i].positions)
        elif i == len(self.path) - 1:
            # Last image: spring to previous
            return self.spring_constant * (self.path[i - 1].positions - self.path[i].positions)
        else:
            # Middle images: spring to both neighbors
            f_prev = self.spring_constant * (self.path[i - 1].positions - self.path[i].positions)
            f_next = self.spring_constant * (self.path[i + 1].positions - self.path[i].positions)
            return f_prev + f_next

    def _nudge_forces(self, forces, spring_forces):
        """Apply nudging to forces (project out parallel component)."""
        # Calculate tangent vector (simplified)
        if len(self.path) > 1:
            tangent = self.path[1].positions - self.path[0].positions
            tangent = tangent / np.linalg.norm(tangent)
        else:
            tangent = np.zeros_like(forces[0])

        # Project forces perpendicular to tangent
        parallel_component = np.sum(forces * tangent) * tangent
        perpendicular_forces = forces - parallel_component

        return perpendicular_forces + spring_forces

    def _apply_climbing_forces(self, forces, spring_forces, index):
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
        climbing_forces = forces - 2 * parallel_component + spring_forces

        return climbing_forces

    def _calculate_tangent_for_climbing(self, index):
        """Calculate tangent vector for climbing image."""
        if index <= 0 or index >= len(self.path) - 1:
            return None

        # Use energy-weighted tangent calculation
        prev_pos = self.path[index - 1].positions
        curr_pos = self.path[index].positions
        next_pos = self.path[index + 1].positions

        # Forward difference
        forward = next_pos - curr_pos
        # Backward difference
        backward = curr_pos - prev_pos

        # Energy-weighted tangent (same as regular NEB)
        try:
            _ = self.path[index].get_potential_energy()  # curr_energy
            next_energy = self.path[index + 1].get_potential_energy()
            prev_energy = self.path[index - 1].get_potential_energy()

            if next_energy > prev_energy:
                tangent = forward
            else:
                tangent = backward
        except Exception:
            # Fallback to simple average
            tangent = (forward + backward) / 2

        # Normalize
        norm = np.linalg.norm(tangent)
        if norm > 1e-10:
            return tangent.flatten() / norm
        else:
            return None

    def _update_positions(self, forces, step_size=0.01):
        """Update positions using forces."""
        for _i, (atoms, force) in enumerate(zip(self.path, forces, strict=False)):
            atoms.positions += step_size * force


def _batch_neb_runner(
    atoms_list: Sequence[Atoms] | Atoms,
    calculator,
    npoints: int = 11,
    method: str = "geodesic",
    fmax: float = 0.05,
    steps: int = 1000,
    spring_constant: float = 5.0,
    climb: bool = False,
    **kwargs,
):
    """Batch NEB runner for calculators that support batch evaluation.

    Supports both regular NEB and CI-NEB depending on the climb parameter.
    """
    # Generate initial path using PathManager
    path_mgr = PathManager(atoms_list, calculator=calculator)
    # Filter out mode parameter which is not accepted by PathManager.interpolate
    interpolate_kwargs = {k: v for k, v in kwargs.items() if k != "mode"}
    path = path_mgr.interpolate(
        npoints=npoints,
        method=method,
        optimize_path=False,  # Don't optimize initially, we'll do NEB
        **interpolate_kwargs,
    )

    # Create batch NEB optimizer
    batch_neb = BatchNEBOptimizer(
        path,
        calculator=calculator,
        fmax=fmax,
        steps=steps,
        spring_constant=spring_constant,
        climb=climb,
        **kwargs,
    )

    # Optimize using batch evaluation
    optimized_path = batch_neb.optimize()

    # Filter redundant structures and issue warnings
    if optimized_path:
        # Convert atoms_list to list for comparison
        input_atoms = list(atoms_list) if not isinstance(atoms_list, Atoms) else [atoms_list]

        strategy_name = "twoended:cineb" if climb else "twoended:neb"
        filtered_path, removed_indices, warnings_list = PathManager.filter_redundant_structures(
            optimized_path,
            input_structures=input_atoms,
            rmsd_threshold=kwargs.get("rmsd_threshold", 0.1),
            energy_threshold=kwargs.get("energy_threshold", 0.001),
            strategy_name=strategy_name,
        )

        # Issue warnings
        for warning_msg in warnings_list:
            warnings.warn(warning_msg, stacklevel=2)

        optimized_path = filtered_path

    return optimized_path


def twoended_neb_runner(
    atoms_list: Sequence[Atoms] | Atoms,
    npoints: int = 11,
    method: str = "geodesic",
    explorer: Any | None = None,
    fmax: float = 0.05,
    steps: int = 1000,
    local_optimizer_name: str = "sella",
    spring_constant: float = 5.0,
    verbose: int = 1,
    **kwargs: Any,
) -> list[Atoms]:
    """Delegate to TwoEndedNEBStrategy to avoid duplication."""
    strategy = TwoEndedNEBStrategy(explorer=explorer)
    forwarded = {
        k: v
        for k, v in {
            "npoints": npoints,
            "method": method,
            "fmax": fmax,
            "steps": steps,
            "spring_constant": spring_constant,
            "local_optimizer_name": local_optimizer_name,
            "verbose": verbose,
            **kwargs,
        }.items()
        if k != "explorer"
    }
    return strategy.run(atoms_list, **forwarded)


def _setup_neb_optimizer(local_optimizer_name: str):
    """Setup optimizer class for NEB calculations."""
    try:
        OptClass = _get_local_optimizer_class(local_optimizer_name)
    except (ImportError, ValueError, AttributeError) as e:
        warnings.warn(f"Could not select optimizer '{local_optimizer_name}': {e}", stacklevel=2)
        OptClass = BFGS  # Fallback to ASE BFGS
    return OptClass


def _check_batch_support(calculator: Any) -> bool:
    """Check if calculator supports batch evaluation (delegates to StrategyUtils)."""
    return StrategyUtils.check_batch_support(calculator)


def _calculate_neb_energies_forces(
    path: list[Atoms], calculator: Any, supports_batch: bool
) -> tuple[list[float], list[np.ndarray]]:
    """Calculate energies and forces for all images (delegates to StrategyUtils)."""
    return StrategyUtils.calculate_batch_energies_forces(path, calculator, supports_batch)


def _check_neb_convergence(forces_list: list[np.ndarray], fmax: float, step: int) -> bool:
    """Check convergence (delegates to StrategyUtils)."""
    return StrategyUtils.check_convergence(forces_list, fmax, step)


def _apply_neb_forces(
    path: list[Atoms], forces_list: list[np.ndarray], energies: list[float], spring_constant: float
):
    """Apply NEB forces to path images."""
    for i in range(1, len(path) - 1):  # Skip endpoints
        atoms = path[i]
        forces = forces_list[i].copy()

        # Spring forces
        spring_force = _calculate_spring_force(path, i, spring_constant, energies)

        # Project forces perpendicular to path (nudging)
        tangent = _calculate_tangent(path, i, energies)
        if tangent is not None:
            # Remove component parallel to tangent
            parallel_component = np.dot(forces.flatten(), tangent) * tangent
            forces = forces.flatten() - parallel_component
            forces = forces.reshape(-1, 3)

        # Add spring forces
        forces += spring_force

        # Store forces in calculator results (ASE-compatible approach)
        if atoms.calc is not None:
            if not hasattr(atoms.calc, "results"):
                atoms.calc.results = {}
            atoms.calc.results["forces"] = forces
        else:
            # Fallback: store in atoms info for compatibility
            atoms.info["forces"] = forces


def _run_simple_neb(
    path: list[Atoms],
    spring_constant: float = 5.0,
    fmax: float = 0.05,
    steps: int = 1000,
    local_optimizer_name: str = "sella",
    explorer: Any | None = None,
    **kwargs,
):
    """Run a simple NEB optimization on the given path using NEBOptimizer.

    This uses ASE-like semantics where 'steps' controls total optimizer iterations,
    not outer NEB loops. Much cleaner output and more intuitive behavior.
    """
    # Setup optimizer
    OptClass = _setup_neb_optimizer(local_optimizer_name)

    # Filter out incompatible kwargs for NEBOptimizer
    filtered_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k
        not in [
            "verbose",
            "explorer",
            "local_optimizer_name",
            "climb",
            "product",
            "npoints",
            "validate_ts",
        ]
    }

    # Create NEBOptimizer instance
    neb_opt = NEBOptimizer(
        images=path,
        spring_constant=spring_constant,
        climb=False,
        optimizer_class=OptClass,
        fmax=fmax,
        **filtered_kwargs,
    )

    # Run optimizer with steps parameter controlling total iterations
    neb_opt.run(fmax=fmax, steps=steps)

    return path


def _neb_optimize(
    path: list[Atoms],
    spring_constant: float,
    fmax: float,
    steps: int,
    local_optimizer_name: str,
    climb: bool,
    **kwargs,
):
    """Shared NEB optimization helper for NEB and CI-NEB.

    Uses existing simple NEB/CI-NEB implementations under the hood to avoid
    duplicating the stepping logic in strategies.
    """
    if climb:
        _run_cineb(
            path,
            spring_constant=spring_constant,
            fmax=fmax,
            steps=steps,
            local_optimizer_name=local_optimizer_name,
            climb=True,
            **kwargs,
        )
    else:
        _run_simple_neb(
            path,
            spring_constant=spring_constant,
            fmax=fmax,
            steps=steps,
            local_optimizer_name=local_optimizer_name,
            **kwargs,
        )


def _calculate_spring_force(
    path: list[Atoms], index: int, spring_constant: float, energies: list[float]
) -> np.ndarray:
    """Calculate spring forces for NEB (delegates to StrategyUtils)."""
    return StrategyUtils.calculate_spring_force(path, index, spring_constant, energies)


def _calculate_tangent(path: list[Atoms], index: int, energies: list[float]) -> np.ndarray | None:
    """Calculate tangent vector for NEB force projection (delegates to StrategyUtils)."""
    return StrategyUtils.calculate_tangent(path, index, energies)


def twoended_cineb_runner(
    atoms_list: Sequence[Atoms] | Atoms,
    npoints: int = 11,
    method: str = "geodesic",
    explorer: Any | None = None,
    fmax: float = 0.05,
    steps: int = 1000,
    local_optimizer_name: str = "sella",
    spring_constant: float = 5.0,
    climb: bool = True,
    verbose: int = 1,
    **kwargs: Any,
) -> list[Atoms]:
    """Delegate to TwoEndedCINEBStrategy to avoid duplication."""
    strategy = TwoEndedCINEBStrategy(explorer=explorer)
    forwarded = {
        k: v
        for k, v in {
            "npoints": npoints,
            "method": method,
            "fmax": fmax,
            "steps": steps,
            "spring_constant": spring_constant,
            "climb": climb,
            "local_optimizer_name": local_optimizer_name,
            "verbose": verbose,
            **kwargs,
        }.items()
        if k != "explorer"
    }
    return strategy.run(atoms_list, **forwarded)


def _setup_cineb_optimizer(local_optimizer_name: str):
    """Setup optimizer class for CI-NEB calculations."""
    try:
        OptClass = _get_local_optimizer_class(local_optimizer_name)
    except (ImportError, ValueError, AttributeError) as e:
        warnings.warn(f"Could not select optimizer '{local_optimizer_name}': {e}", stacklevel=2)
        OptClass = BFGS  # Fallback to ASE BFGS
    return OptClass


def _calculate_cineb_energies_forces(
    path: list[Atoms], calculator: Any, supports_batch: bool
) -> tuple[list[float], list[np.ndarray]]:
    """Calculate energies and forces for all images in the CI-NEB path."""
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
            warnings.warn(
                f"Batch evaluation failed, falling back to individual " f"calculations: {e}",
                stacklevel=2,
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
                warnings.warn(f"Failed to calculate energy/forces: {e}", stacklevel=2)
                energies.append(float("inf"))
                forces_list.append(np.zeros((len(atoms), 3)))

    return energies, forces_list


def _determine_climbing_image(energies: list[float], climb: bool) -> int | None:
    """Determine which image should be the climbing image."""
    if not climb or len(energies) <= 2:
        return None

    # Find highest energy image (excluding endpoints)
    valid_energies = [(i, e) for i, e in enumerate(energies[1:-1], 1) if not np.isnan(e)]
    if not valid_energies:
        return None

    climbing_image = max(valid_energies, key=lambda x: x[1])[0]
    return climbing_image


def _apply_cineb_forces(
    path: list[Atoms],
    forces_list: list[np.ndarray],
    energies: list[float],
    spring_constant: float,
    climbing_image: int | None,
):
    """Apply CI-NEB forces to path images."""
    for i in range(1, len(path) - 1):  # Skip endpoints
        atoms = path[i]
        forces = forces_list[i].copy()

        # Spring forces
        spring_force = _calculate_spring_force(path, i, spring_constant, energies)

        # Project forces perpendicular to path (nudging)
        tangent = _calculate_tangent(path, i, energies)
        if tangent is not None:
            # Remove component parallel to tangent
            parallel_component = np.dot(forces.flatten(), tangent) * tangent
            forces = forces.flatten() - parallel_component
            forces = forces.reshape(-1, 3)

        # Apply climbing image behavior if this is the climbing image
        if climbing_image == i:
            # Invert the parallel component for climbing
            if tangent is not None:
                parallel_component = np.dot(forces_list[i].flatten(), tangent) * tangent
                parallel_component = parallel_component.reshape(-1, 3)
                # Invert parallel component (make it point uphill)
                forces = forces - 2 * parallel_component

        # Add spring forces
        forces += spring_force

        # Store forces in calculator results (ASE-compatible approach)
        if atoms.calc is not None:
            if not hasattr(atoms.calc, "results"):
                atoms.calc.results = {}
            atoms.calc.results["forces"] = forces
        else:
            # Fallback: store in atoms info for compatibility
            atoms.info["forces"] = forces


def _run_cineb(
    path: list[Atoms],
    spring_constant: float = 5.0,
    fmax: float = 0.05,
    steps: int = 1000,
    local_optimizer_name: str = "sella",
    climb: bool = True,
    explorer: Any | None = None,
    **kwargs,
):
    """Run a CI-NEB optimization on the given path using NEBOptimizer.

    This uses ASE-like semantics where 'steps' controls total optimizer iterations,
    not outer NEB loops. Much cleaner output and more intuitive behavior.
    """
    # Setup optimizer
    OptClass = _setup_cineb_optimizer(local_optimizer_name)

    # Filter out incompatible kwargs for NEBOptimizer
    filtered_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k
        not in [
            "verbose",
            "explorer",
            "local_optimizer_name",
            "climb",
            "product",
            "npoints",
            "validate_ts",
        ]
    }

    # Create NEBOptimizer with climbing enabled
    neb_opt = NEBOptimizer(
        images=path,
        spring_constant=spring_constant,
        climb=True,
        optimizer_class=OptClass,
        fmax=fmax,
        **filtered_kwargs,
    )

    # Run optimizer with steps parameter controlling total iterations
    neb_opt.run(fmax=fmax, steps=steps)

    return path


def twoended_growing_string_runner(
    atoms_list: Sequence[Atoms] | Atoms,
    npoints: int = 15,
    explorer: Any | None = None,
    fmax: float = 0.05,
    steps: int = 100,
    step_size: float = 0.1,
    distance_threshold: float = 0.5,
    local_optimizer_name: str = "sella",
    optimize_endpoints: bool = True,
    refine_ts: bool = True,
    verbose: int = 1,
    **kwargs: Any,
) -> dict[str, Any]:
    """Delegate to TwoEndedGrowingStringStrategy to avoid duplication."""
    strategy = TwoEndedGrowingStringStrategy(explorer=explorer)
    forwarded = {
        k: v
        for k, v in {
            "npoints": npoints,
            "fmax": fmax,
            "steps": steps,
            "step_size": step_size,
            "distance_threshold": distance_threshold,
            "local_optimizer_name": local_optimizer_name,
            "optimize_endpoints": optimize_endpoints,
            "refine_ts": refine_ts,
            "verbose": verbose,
            **kwargs,
        }.items()
        if k != "explorer"
    }
    _res = strategy.run(atoms_list, **forwarded)
    # Preserve legacy strategy label expected by tests/CLI for the function facade
    return {
        **_res,
        "strategy": "twoended_growing_string_runner",
    }


def _grow_string_node(
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

    Returns
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
                f"Growing String: Very small forces ({force_magnitude:.2e}), " "skipping node"
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


__all__.append("twoended_minima_runner")
__all__.append("twoended_neb_runner")
__all__.append("twoended_cineb_runner")
__all__.append("twoended_growing_string_runner")
