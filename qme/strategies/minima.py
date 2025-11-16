"""Local minima optimization strategy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ase import Atoms

from qme.core.base_strategy import BaseStrategy, StrategyMetadata
from qme.core.registry import REGISTRY
from qme.strategies.helpers import _run_local_optimization_common
from qme.utils.logging import get_qme_logger

logger = get_qme_logger(__name__)


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
        """Run local minima optimization.

        Parameters
        ----------
        atoms_list : Sequence[Atoms]
            List of structures to optimize
        fmax : float, default=0.05
            Force convergence threshold
        steps : int, default=1000
            Maximum optimization steps
        calculate_frequencies : bool, default=False
            Whether to perform frequency analysis after optimization
        **kwargs : Any
            Additional keyword arguments

        Returns
        -------
        dict[str, Atoms | list[Atoms] | bool | int | float | str]
            Standardized result dictionary containing:
            - optimized_atoms: Optimized structure(s) (Atoms or list[Atoms])
            - strategy: Strategy name (str)
            - converged: Whether optimization converged (bool)
            - steps_taken: Number of optimization steps (int)
            - frequency_analysis: Frequency analysis results (dict, optional)
            - is_minimum: Whether structure is a minimum (bool, optional)
            - free_energy_correction: Free energy correction in eV (float, optional)

        """
        local_optimizer_name = kwargs.get("local_optimizer_name", "sella")
        verbose = kwargs.get("verbose", 1)
        temperature = kwargs.get("temperature", 298.15)

        def prepare_minima_optimizer_kwargs(optimizer_name: str, explorer: Any) -> dict[str, Any]:
            """Prepare optimizer kwargs for minima optimization."""
            opt_kwargs = getattr(explorer, "optimizer_kwargs", {}) or {}
            opt_kwargs = dict(opt_kwargs)

            if optimizer_name.lower() == "sella":
                # Sella-specific kwargs for minima search
                opt_kwargs.setdefault("internal", True)
                opt_kwargs.setdefault("order", 0)
                # Note: SELLA computes its own Hessian internally and doesn't accept
                # an initial Hessian as a keyword argument. The initial_hessian from
                # explorer is not used for SELLA.

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


# Register the strategy
REGISTRY.register(LocalMinimaStrategy)
