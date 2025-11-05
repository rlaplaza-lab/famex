"""Multi-structure TS guess strategy via interpolation with local TS refinement."""

from typing import Any

from ase import Atoms

from qme.core.base_strategy import BaseStrategy, StrategyMetadata
from qme.core.registry import REGISTRY
from qme.io.path_manager import PathManager
from qme.strategies.helpers import validate_ts_structure
from qme.strategies.ts import LocalTSStrategy
from qme.utils.logging import get_qme_logger

logger = get_qme_logger(__name__)


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
        calculate_frequencies: bool = False,
        **kwargs: Any,
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
        validate_ts : bool, default=False
            Whether to validate TS structure via basic frequency analysis
        calculate_frequencies : bool, default=False
            Whether to perform comprehensive frequency analysis after optimization
        **kwargs
            Additional keyword arguments

        Returns:
        -------
        dict[str, Any]
            Standardized result dictionary

        """
        self.validate_inputs(atoms_list)

        local_optimizer_name = kwargs.get("local_optimizer_name", "sella")

        # Generate interpolated path using PathManager
        path_mgr = PathManager(atoms_list)
        # Filter kwargs to only include parameters accepted by PathManager.interpolate
        interpolate_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ["calculator"]  # Only pass calculator if present
        }
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
        ts_guess, _ts_index = PathManager.find_ts_guess(path)

        # Refine with local TS optimization
        ts_strategy = LocalTSStrategy(self.explorer)
        # Remove local_optimizer_name from kwargs to avoid duplicate argument
        ts_kwargs = {k: v for k, v in kwargs.items() if k != "local_optimizer_name"}
        ts_result = ts_strategy.run(
            [ts_guess],  # LocalTSStrategy expects list[Atoms]
            fmax=fmax,
            steps=steps,
            local_optimizer_name=local_optimizer_name,
            calculate_frequencies=calculate_frequencies,
            **ts_kwargs,
        )

        # Validate TS structure if requested
        validation_result = None
        optimized_atoms = ts_result["optimized_atoms"]
        if validate_ts:
            # Type narrowing: validate_ts_structure expects Atoms
            if isinstance(optimized_atoms, Atoms):
                validation_result = validate_ts_structure(optimized_atoms, self.explorer)

        # Return single TS structure
        # Type narrowing: prepare_result expects Atoms | Sequence[Atoms]
        if isinstance(optimized_atoms, Atoms) or (
            isinstance(optimized_atoms, list) and all(isinstance(a, Atoms) for a in optimized_atoms)
        ):
            result = self.prepare_result(
                optimized_atoms,
                steps_taken=ts_result["steps_taken"],
                converged=ts_result["converged"],
            )
        else:
            # Fallback: create result with original structure
            from qme.core.base_strategy import BaseStrategy

            result = BaseStrategy.prepare_result(
                self,
                optimized_atoms
                if isinstance(optimized_atoms, (Atoms, list))
                else self.explorer.atoms_list[0],
                steps_taken=ts_result["steps_taken"],
                converged=ts_result["converged"],
            )
        if validation_result is not None:
            result["ts_validation"] = validation_result
        # Pass through frequency analysis results if available
        if "frequency_analysis" in ts_result:
            result["frequency_analysis"] = ts_result["frequency_analysis"]
            result["is_ts"] = ts_result.get("is_ts")
            result["free_energy_correction"] = ts_result.get("free_energy_correction")
        return result


# Register the strategy
REGISTRY.register(MultiStructureTSGuessStrategy)
