"""Multi-structure TS guess strategy via interpolation with local TS refinement."""

from typing import Any

from ase import Atoms

from qme.core.path_manager import PathManager
from qme.core.strategies.local.helpers import validate_ts_structure
from qme.core.strategies.local.ts import LocalTSStrategy
from qme.core.strategy import REGISTRY, BaseStrategy, StrategyMetadata


class MultiStructureTSGuessStrategy(BaseStrategy):
    """Multi-structure TS guess strategy via interpolation with local TS refinement."""

    metadata = StrategyMetadata(
        name="ts:interpolate",
        target="ts",
        strategy="interpolate",
        description="TS guess via interpolation with local TS refinement",
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
        validate_ts: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Run multi-structure TS guess via interpolation.

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
        path = path_mgr.interpolate(
            npoints=npoints,
            method=method,
            optimize_path=False,  # Raw interpolation
            explorer=self.explorer,
            **kwargs,
        )

        # Attach calculators to all images
        if self.explorer is not None:
            PathManager.attach_calculators(self.explorer, path)

        # Find highest energy structure as TS guess using PathManager
        ts_guess, ts_index = PathManager.find_ts_guess(path)

        # Refine with local TS optimization
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
REGISTRY.register(MultiStructureTSGuessStrategy)
