"""NEB (Nudged Elastic Band) path optimization strategy."""

from __future__ import annotations

import warnings
from typing import Any, Union

from ase import Atoms

from qme.core.path_manager import PathManager
from qme.core.strategies.multistructure.neb_optimizer import NEBOptimizer
from qme.core.strategy import REGISTRY, BaseStrategy, StrategyMetadata


class MultiStructureNEBStrategy(BaseStrategy):
    """NEB (Nudged Elastic Band) path optimization strategy."""

    metadata = StrategyMetadata(
        name="path:neb",
        target="path",
        strategy="neb",
        description="NEB path optimization with geodesic interpolation",
        aliases=["neb"],
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
        **kwargs: Any,
    ) -> dict[str, Union[Atoms, list[Atoms], bool, int, float, str]]:
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
        **kwargs : Any
            Additional keyword arguments

        Returns
        -------
        dict[str, Union[Atoms, list[Atoms], bool, int, float, str]]
            Standardized result dictionary containing:
            - optimized_atoms: NEB path structures (list[Atoms])
            - strategy: Strategy name (str)
            - converged: Whether NEB optimization converged (bool)
            - steps_taken: Number of optimization steps (int)
            - npoints: Number of images in the path (int)
            - method: Interpolation method used (str)
        """
        self.validate_inputs(atoms_list)

        # Generate initial path using PathManager
        path_mgr = PathManager(atoms_list)
        path = path_mgr.interpolate(
            npoints=npoints,
            method=method,
            optimize_path=False,  # Don't optimize initially, we'll do NEB
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
            raise ValueError("NEB requires at least 3 images (npoints >= 3)")

        # Attach calculators to all images
        if self.explorer is not None:
            for atoms in path:
                self.explorer._create_and_attach_calculator(atoms)
                self.explorer._apply_constraints(atoms)

        # Use unified NEB optimizer
        neb_opt = NEBOptimizer(
            images=path,
            spring_constant=spring_constant,
            climb=False,
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
                strategy_name="path:neb",
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
REGISTRY.register(MultiStructureNEBStrategy)
