"""Multi-structure minima optimization strategy via interpolation."""

from typing import Any

from ase import Atoms

from qme.core.path_manager import PathManager
from qme.core.strategies.local.minima import LocalMinimaStrategy
from qme.core.strategy import REGISTRY, BaseStrategy, StrategyMetadata


class MultiStructureMinimaInterpolateStrategy(BaseStrategy):
    """Multi-structure minima optimization strategy via interpolation."""

    metadata = StrategyMetadata(
        name="minima:interpolate",
        target="minima",
        strategy="interpolate",
        description="Minima optimization via interpolated path",
        aliases=[],
        requires_multiple_structures=True,
    )

    def run(
        self,
        atoms_list: list[Atoms],
        npoints: int = 11,
        method: str = "geodesic",
        fmax: float = 0.05,
        steps: int = 1000,
        **kwargs,
    ) -> dict[str, Any]:
        """Run multi-structure minima optimization via interpolation.

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
            Results dictionary containing optimized structures and metadata
        """
        self.validate_inputs(atoms_list)

        # Use PathManager to interpolate initial path
        path_mgr = PathManager(atoms_list)
        # Only forward kwargs accepted by PathManager.interpolate
        allowed_keys = {"rmsd_threshold", "energy_threshold", "calculator"}
        interpolate_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
        initial_path = path_mgr.interpolate(
            npoints=npoints,
            method=method,
            optimize_path=False,
            explorer=self.explorer,
            **interpolate_kwargs,
        )

        # Now optimize each structure in the path to minima
        local_minima_strategy = LocalMinimaStrategy(explorer=self.explorer, profiler=self.profiler)

        optimized_structures = []
        converged_flags = []
        steps_taken = []

        for atoms in initial_path:
            try:
                result = local_minima_strategy.run(
                    atoms_list=[atoms],
                    fmax=fmax,
                    steps=steps,
                    **kwargs
                )
                optimized_structures.append(result["optimized_atoms"])
                converged_flags.append(result.get("converged", True))
                steps_taken.append(result.get("steps_taken", 0))
            except Exception:
                # If optimization fails for one structure, use the original
                optimized_structures.append(atoms)
                converged_flags.append(False)
                steps_taken.append(0)

        return self.prepare_result(
            optimized_atoms=optimized_structures,
            trajectory=optimized_structures,
            converged=converged_flags,
            steps_taken=steps_taken,
            strategy="minima:interpolate",
        )


# Register the strategy
REGISTRY.register(MultiStructureMinimaInterpolateStrategy)
