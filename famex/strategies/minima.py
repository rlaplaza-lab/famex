"""Local minima optimization strategy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ase import Atoms

from famex.core.base_strategy import BaseStrategy, StrategyMetadata
from famex.core.registry import REGISTRY
from famex.optimizers.sella_utils import is_sella_optimizer
from famex.strategies.frequency_cleanup import prepare_minima_hessian_optimizer_kwargs
from famex.strategies.helpers import _run_local_optimization_common
from famex.utils.logging import get_famex_logger

logger = get_famex_logger(__name__)


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
        local_optimizer_name = kwargs.get("local_optimizer_name", "lbfgs")
        verbose = kwargs.get("verbose", 1)
        temperature = kwargs.get("temperature", 298.15)
        if cleanup_frequencies is False:
            cleanup_frequencies = bool(getattr(self.explorer, "cleanup_frequencies", False))

        def prepare_minima_optimizer_kwargs(optimizer_name: str, explorer: Any) -> dict[str, Any]:
            opt_kwargs = getattr(explorer, "optimizer_kwargs", {}) or {}
            opt_kwargs = dict(opt_kwargs)

            if is_sella_optimizer(optimizer_name.lower()):
                opt_kwargs.setdefault("internal", True)
                opt_kwargs.setdefault("order", 0)

            return opt_kwargs

        def prepare_minima_frequency_kwargs(atoms: Atoms) -> dict[str, Any]:
            freq_kwargs: dict[str, Any] = {}
            if getattr(self.explorer, "force_finite_diff_hessian", False):
                freq_kwargs["method"] = "finite_differences"
            return freq_kwargs

        return _run_local_optimization_common(
            strategy=self,
            atoms_list=atoms_list,
            fmax=fmax,
            steps=steps,
            local_optimizer_name=local_optimizer_name,
            verbose=verbose,
            calculate_frequencies=calculate_frequencies,
            temperature=temperature,
            prepare_optimizer_kwargs=prepare_minima_optimizer_kwargs,
            post_optimization_hook=None,
            validation_hook=None,
            prepare_frequency_kwargs=prepare_minima_frequency_kwargs,
            result_key_name="is_minimum",
            log_prefix="Minima",
            cleanup_frequencies=cleanup_frequencies,
            stationary_point_target="minima",
            prepare_cleanup_optimizer_kwargs=prepare_minima_hessian_optimizer_kwargs,
        )


REGISTRY.register(LocalMinimaStrategy)
