"""Local minima optimization strategy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ase import Atoms

from famex.core.base_strategy import BaseStrategy, StrategyMetadata
from famex.core.registry import REGISTRY
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
        **kwargs: Any,
    ) -> dict[str, Atoms | list[Atoms] | bool | int | float | str]:
        local_optimizer_name = kwargs.get("local_optimizer_name", "sella")
        verbose = kwargs.get("verbose", 1)
        temperature = kwargs.get("temperature", 298.15)

        def prepare_minima_optimizer_kwargs(optimizer_name: str, explorer: Any) -> dict[str, Any]:
            opt_kwargs = getattr(explorer, "optimizer_kwargs", {}) or {}
            opt_kwargs = dict(opt_kwargs)

            if optimizer_name.lower() == "sella":
                opt_kwargs.setdefault("internal", True)
                opt_kwargs.setdefault("order", 0)

            return opt_kwargs

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
            result_key_name="is_minimum",
            log_prefix="Minima",
        )


REGISTRY.register(LocalMinimaStrategy)
