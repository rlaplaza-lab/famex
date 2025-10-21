"""CI-NEB (Climbing Image Nudged Elastic Band) path optimization strategy."""

from __future__ import annotations

import warnings
from typing import Any

from ase import Atoms

from qme.core.path_manager import PathManager
from qme.core.strategies.multistructure.neb_optimizer import NEBOptimizer
from qme.core.strategy import REGISTRY, BaseStrategy, StrategyMetadata


class MultiStructureCINEBStrategy(BaseStrategy):
    """CI-NEB (Climbing Image Nudged Elastic Band) path optimization strategy."""

    metadata = StrategyMetadata(
        name="path:cineb",
        target="path",
        strategy="cineb",
        description="Climbing Image NEB (CI-NEB) optimization with geodesic interpolation",
        aliases=["cineb"],
        requires_multiple_structures=True,
    )

    def run(
        self,
        atoms_list: list[Atoms],
        npoints: int = 11,
        method: str = "geodesic",
        fmax: float = 0.05,
        steps: int = 1000,
        spring_constant: float = 5.0,
        climb: bool = True,
        **kwargs: Any,
    ) -> dict[str, Atoms | list[Atoms] | bool | int | float | str]:
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
        **kwargs : Any
            Additional keyword arguments

        Returns
        -------
        dict[str, Union[Atoms, list[Atoms], bool, int, float, str]]
            Standardized result dictionary containing:
            - optimized_atoms: CI-NEB path structures (list[Atoms])
            - strategy: Strategy name (str)
            - converged: Whether CI-NEB optimization converged (bool)
            - steps_taken: Number of optimization steps (int)
            - npoints: Number of images in the path (int)
            - method: Interpolation method used (str)
            - climb: Whether climbing image was used (bool)
        """
        self.validate_inputs(atoms_list)

        # Generate initial path using PathManager
        path_mgr = PathManager(atoms_list)
        path = path_mgr.interpolate(
            npoints=npoints,
            method=method,
            optimize_path=False,  # Don't optimize initially, we'll do CI-NEB
            explorer=self.explorer,
            **kwargs,
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

        # Use unified NEB optimizer with climbing enabled
        neb_opt = NEBOptimizer(
            images=path,
            spring_constant=spring_constant,
            climb=climb,
            fmax=fmax,
            steps=steps,
            **kwargs,
        )

        optimized_path = neb_opt.optimize()

        # Filter redundant structures and issue warnings
        if optimized_path:
            # Convert atoms_list to list for comparison
            input_atoms = list(atoms_list) if not isinstance(atoms_list, Atoms) else [atoms_list]

            filtered_path, removed_indices, warnings_list = PathManager.filter_redundant_structures(
                optimized_path,
                input_structures=input_atoms,
                rmsd_threshold=kwargs.get("rmsd_threshold", 0.1),
                energy_threshold=kwargs.get("energy_threshold", 0.001),
                strategy_name="path:cineb",
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


# Register the strategy
REGISTRY.register(MultiStructureCINEBStrategy)
