"""Multi-structure TS guess strategy via CI-NEB with local TS refinement."""

from __future__ import annotations

from typing import Any

from ase import Atoms

from famex.core.base_strategy import BaseStrategy, StrategyMetadata
from famex.core.registry import REGISTRY
from famex.io.path_manager import PathManager
from famex.strategies.helpers import filter_interpolation_kwargs, validate_ts_structure
from famex.strategies.neb_optimizer import NEBOptimizer
from famex.strategies.ts import LocalTSStrategy
from famex.utils.logging import get_famex_logger

logger = get_famex_logger(__name__)


class MultiStructureTSCINEBStrategy(BaseStrategy):
    """Multi-structure TS guess strategy via CI-NEB with local TS refinement."""

    metadata = StrategyMetadata(
        name="ts:cineb",
        target="ts",
        strategy="cineb",
        description="TS guess via CI-NEB with local TS refinement",
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
        spring_constant: float = 5.0,
        validate_ts: bool = False,
        calculate_frequencies: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run multi-structure TS search via CI-NEB."""
        self.validate_inputs(atoms_list)

        local_optimizer_name = kwargs.get("local_optimizer_name", "rfo")

        path_mgr = PathManager(atoms_list)
        interpolate_kwargs = filter_interpolation_kwargs(kwargs, allowed_keys={"calculator"})
        path = path_mgr.interpolate(
            npoints=npoints,
            method=method,
            optimize_path=False,
            explorer=self.explorer,
            **interpolate_kwargs,
        )

        if len(path) < 3:
            msg = "CI-NEB requires at least 3 images (npoints >= 3)"
            raise ValueError(msg)

        if self.explorer is not None:
            PathManager.attach_calculators(self.explorer, path)
            if any(getattr(img, "calc", None) is None for img in path):
                raise RuntimeError(
                    "Failed to attach calculators to CI-NEB images. "
                    "Check backend/model availability.",
                )

        neb_opt = NEBOptimizer(
            images=path,
            spring_constant=spring_constant,
            climb=True,
            fmax=fmax,
            steps=steps,
            **kwargs,
        )
        optimized_path = neb_opt.optimize()

        if neb_opt.climbing_image is not None:
            ts_guess = optimized_path[neb_opt.climbing_image]
            ts_index = neb_opt.climbing_image
            logger.info("Using climbing image %d as TS guess", ts_index)
        else:
            ts_guess, ts_index = PathManager.find_ts_guess(optimized_path)
            logger.info("Using highest-energy image %d as TS guess", ts_index)

        ts_strategy = LocalTSStrategy(self.explorer)
        ts_kwargs = {k: v for k, v in kwargs.items() if k != "local_optimizer_name"}
        ts_result = ts_strategy.run(
            [ts_guess],
            fmax=fmax,
            steps=steps,
            local_optimizer_name=local_optimizer_name,
            calculate_frequencies=calculate_frequencies,
            **ts_kwargs,
        )

        validation_result = None
        optimized_atoms = ts_result["optimized_atoms"]
        if validate_ts and isinstance(optimized_atoms, Atoms):
            validation_result = validate_ts_structure(optimized_atoms, self.explorer)

        if isinstance(optimized_atoms, Atoms) or (
            isinstance(optimized_atoms, list) and all(isinstance(a, Atoms) for a in optimized_atoms)
        ):
            result = self.prepare_result(
                optimized_atoms,
                steps_taken=ts_result["steps_taken"],
                converged=ts_result["converged"],
                climb=True,
                ts_guess_index=ts_index,
            )
        else:
            result = BaseStrategy.prepare_result(
                self,
                (
                    optimized_atoms
                    if isinstance(optimized_atoms, Atoms | list)
                    else self.explorer.atoms_list[0]
                ),
                steps_taken=ts_result["steps_taken"],
                converged=ts_result["converged"],
                climb=True,
                ts_guess_index=ts_index,
            )

        if validation_result is not None:
            result["ts_validation"] = validation_result
        if "frequency_analysis" in ts_result:
            result["frequency_analysis"] = ts_result["frequency_analysis"]
            result["is_ts"] = ts_result.get("is_ts")
            result["free_energy_correction"] = ts_result.get("free_energy_correction")
        return result


REGISTRY.register(MultiStructureTSCINEBStrategy)
