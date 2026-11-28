"""Multi-structure TS guess strategy via CI-NEB with local TS refinement."""

from __future__ import annotations

from typing import Any

from ase import Atoms

from famex.core.base_strategy import BaseStrategy, StrategyMetadata
from famex.core.registry import REGISTRY
from famex.io.path_manager import PathManager
from famex.strategies.helpers import neb_method_name, preflight_band_path, refine_ts_guess_workflow
from famex.strategies.kwargs_policy import (
    NEB_OPTIMIZER_KEYS,
    TS_GUESS_RUN_KEYS,
    pick_pass_through_kwargs,
)
from famex.strategies.neb_optimizer import NEBOptimizer
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
        method: str = "idpp",
        fmax: float = 0.05,
        steps: int = 1000,
        spring_constant: float = 5.0,
        validate_ts: bool = False,
        calculate_frequencies: bool = False,
        path_fmax: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run multi-structure TS search via CI-NEB."""
        self.validate_inputs(atoms_list)

        run_extra = pick_pass_through_kwargs(kwargs, TS_GUESS_RUN_KEYS)
        local_optimizer_name = run_extra.get("local_optimizer_name", "rfo")
        record_history = bool(run_extra.get("record_history", False))
        effective_path_fmax = path_fmax if path_fmax is not None else run_extra.get("path_fmax")
        band_fmax = 0.10 if effective_path_fmax is None else float(effective_path_fmax)

        path_mgr = PathManager(atoms_list)
        path = path_mgr.interpolate(npoints=npoints, method=method)
        preflight_band_path(path, self.explorer, method_name=neb_method_name(climb=True))

        neb_kwargs = pick_pass_through_kwargs(kwargs, NEB_OPTIMIZER_KEYS)
        neb_opt = NEBOptimizer(
            images=path,
            spring_constant=spring_constant,
            climb=True,
            fmax=band_fmax,
            steps=steps,
            **neb_kwargs,
        )
        optimized_path, _band_converged = neb_opt.optimize()

        if neb_opt.climbing_image is not None:
            ts_guess = optimized_path[neb_opt.climbing_image]
            ts_index = neb_opt.climbing_image
            logger.info("Using climbing image %d as TS guess", ts_index)
        else:
            ts_guess, ts_index = PathManager.find_ts_guess(optimized_path)
            logger.info("Using highest-energy image %d as TS guess", ts_index)

        ts_run_kwargs = {
            k: v
            for k, v in run_extra.items()
            if k not in {"local_optimizer_name", "path_fmax", "record_history"}
        }
        search_path = optimized_path if record_history and optimized_path else None
        return refine_ts_guess_workflow(
            self,
            ts_guess,
            fmax=fmax,
            steps=steps,
            local_optimizer_name=local_optimizer_name,
            calculate_frequencies=calculate_frequencies,
            validate_ts=validate_ts,
            ts_run_kwargs=ts_run_kwargs,
            prepare_result_kwargs={"climb": True, "ts_guess_index": ts_index},
            record_history=record_history,
            search_path=search_path,
        )


REGISTRY.register(MultiStructureTSCINEBStrategy)
