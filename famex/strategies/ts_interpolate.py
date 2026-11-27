"""Multi-structure TS guess strategy via interpolation with local TS refinement."""

from __future__ import annotations

from typing import Any

from ase import Atoms

from famex.core.base_strategy import BaseStrategy, StrategyMetadata
from famex.core.registry import REGISTRY
from famex.io.path_manager import PathManager
from famex.strategies.helpers import preflight_band_path, refine_ts_guess_workflow


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
        method: str = "idpp",
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
        method : str, default="idpp"
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

        Returns
        -------
        dict[str, Any]
            Standardized result dictionary

        """
        self.validate_inputs(atoms_list)

        local_optimizer_name = kwargs.get("local_optimizer_name", "rfo")
        record_history = bool(kwargs.get("record_history", False))

        path_mgr = PathManager(atoms_list)
        path = path_mgr.interpolate(npoints=npoints, method=method)
        preflight_band_path(path, self.explorer, method_name="TS interpolation")

        ts_guess, _ts_index = PathManager.find_ts_guess(path)
        ts_run_kwargs = {k: v for k, v in kwargs.items() if k != "local_optimizer_name"}
        return refine_ts_guess_workflow(
            self,
            ts_guess,
            fmax=fmax,
            steps=steps,
            local_optimizer_name=local_optimizer_name,
            calculate_frequencies=calculate_frequencies,
            validate_ts=validate_ts,
            ts_run_kwargs=ts_run_kwargs,
            record_history=record_history,
            search_path=path if record_history else None,
        )


# Register the strategy
REGISTRY.register(MultiStructureTSGuessStrategy)
