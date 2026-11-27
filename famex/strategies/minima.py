"""Local minima optimization strategy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ase import Atoms

from famex.core.base_strategy import BaseStrategy, StrategyMetadata
from famex.core.registry import REGISTRY
from famex.strategies.helpers import _run_local_optimization_common, build_optimizer_kwargs
from famex.strategies.kwargs_policy import LOCAL_OPTIMIZATION_RUN_KEYS, pick_pass_through_kwargs


class LocalMinimaStrategy(BaseStrategy):
    """Local minima optimization strategy."""

    metadata = StrategyMetadata(
        name="minima:local",
        target="minima",
        strategy="local",
        description="Local minima optimization (ASE/LBFGS or SELLA)",
        aliases=["minima", "local:minima", "local-minima"],
        requires_multiple_structures=False,
    )

    def run(
        self,
        atoms_list: Sequence[Atoms],
        fmax: float = 0.05,
        steps: int = 1000,
        calculate_frequencies: bool = False,
        cleanup_frequencies: bool = False,
        **kwargs: Any,
    ) -> dict[str, Atoms | list[Atoms] | bool | int | float | str]:
        run_extra = pick_pass_through_kwargs(kwargs, LOCAL_OPTIMIZATION_RUN_KEYS)
        local_optimizer_name = run_extra.get("local_optimizer_name", "lbfgs")
        verbose = run_extra.get("verbose", 1)
        temperature = run_extra.get("temperature", 298.15)
        if cleanup_frequencies is False:
            cleanup_frequencies = bool(getattr(self.explorer, "cleanup_frequencies", False))

        force_fd = bool(getattr(self.explorer, "force_finite_diff_hessian", False))

        def prepare_frequency_kwargs(_atoms: Atoms) -> dict[str, Any]:
            return {"method": "finite_differences"} if force_fd else {}

        return _run_local_optimization_common(
            strategy=self,
            atoms_list=atoms_list,
            fmax=fmax,
            steps=steps,
            local_optimizer_name=local_optimizer_name,
            verbose=verbose,
            calculate_frequencies=calculate_frequencies,
            temperature=temperature,
            prepare_optimizer_kwargs=lambda name, explorer: build_optimizer_kwargs(
                name, explorer, target="minima_local"
            ),
            post_optimization_hook=None,
            validation_hook=None,
            prepare_frequency_kwargs=prepare_frequency_kwargs,
            result_key_name="is_minimum",
            log_prefix="Minima",
            cleanup_frequencies=cleanup_frequencies,
            stationary_point_target="minima",
            prepare_cleanup_optimizer_kwargs=lambda name, explorer: build_optimizer_kwargs(
                name, explorer, target="minima_hessian"
            ),
            record_history=bool(run_extra.get("record_history", False)),
        )


REGISTRY.register(LocalMinimaStrategy)
