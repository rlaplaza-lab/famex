"""Local transition state optimization strategy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ase import Atoms

from qme.core.base_strategy import BaseStrategy, StrategyMetadata
from qme.core.registry import REGISTRY
from qme.strategies.helpers import (
    _run_local_optimization_common,
    _validate_ts_optimization_setup,
    validate_ts_structure,
)
from qme.strategies.utils import StrategyUtils
from qme.utils.logging import get_qme_logger

logger = get_qme_logger(__name__)


class LocalTSStrategy(BaseStrategy):
    """Local transition state optimization strategy."""

    metadata = StrategyMetadata(
        name="ts:local",
        target="ts",
        strategy="local",
        description="Local transition-state optimization (SELLA preferred)",
        aliases=["ts", "local:ts", "local-ts"],
        requires_multiple_structures=False,
    )

    def run(
        self,
        atoms_list: Sequence[Atoms],
        fmax: float = 0.05,
        steps: int = 1000,
        validate_ts: bool = False,
        calculate_frequencies: bool = False,
        **kwargs: Any,
    ) -> dict[str, Atoms | list[Atoms] | bool | int | float | str]:
        """Run local transition state optimization.

        Parameters
        ----------
        atoms_list : Sequence[Atoms]
            List of structures to optimize
        fmax : float, default=0.05
            Force convergence threshold
        steps : int, default=1000
            Maximum optimization steps
        validate_ts : bool, default=False
            Whether to validate TS structure via basic frequency analysis
        calculate_frequencies : bool, default=False
            Whether to perform comprehensive frequency analysis after optimization
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
            - ts_validation: Transition state validation results (dict, optional)
            - frequency_analysis: Frequency analysis results (dict, optional)
            - is_ts: Whether structure is a transition state (bool, optional)
            - free_energy_correction: Free energy correction in eV (float, optional)

        """
        local_optimizer_name = kwargs.get("local_optimizer_name", "sella")
        verbose = kwargs.get("verbose", 1)
        temperature = kwargs.get("temperature", 298.15)

        _validate_ts_optimization_setup(self.explorer.backend, local_optimizer_name)

        def prepare_ts_optimizer_kwargs(optimizer_name: str, explorer: Any) -> dict[str, Any]:
            """Prepare optimizer kwargs for TS optimization."""
            opt_kwargs = getattr(explorer, "ts_kwargs", {}) or {}
            opt_kwargs = dict(opt_kwargs)

            normalized_name = optimizer_name.lower()
            if normalized_name == "sella":
                opt_kwargs.setdefault("internal", True)
                opt_kwargs.setdefault("order", 1)
                opt_kwargs.pop("hessian_method", None)
            elif normalized_name in (
                "rfo",
                "rfo-ts",
                "rational-function",
                "rational_function",
            ):
                opt_kwargs.setdefault("hessian_update_freq", 10)
                opt_kwargs.setdefault("hessian_method", "auto")
                opt_kwargs.setdefault("trust_radius", 0.02)
                opt_kwargs.setdefault("max_trust_radius", 0.06)

            if getattr(explorer, "force_finite_diff_hessian", False):
                if normalized_name in {
                    "rfo",
                    "rfo-ts",
                    "rational-function",
                    "rational_function",
                }:
                    opt_kwargs["hessian_method"] = "finite_differences"

            return opt_kwargs

        def post_ts_optimization_hook(opt: Any, atoms: Atoms, opt_kwargs: dict[str, Any]) -> None:
            """Post-optimization hook for TS diagnostics."""
            if hasattr(opt, "hessian_calls"):
                if self.explorer.verbose >= 1:
                    logger.info(
                        f"Hessian computed {opt.hessian_calls} time(s) "
                        f"over {StrategyUtils.get_step_count(opt)} steps "
                        f"(update_freq={opt_kwargs.get('hessian_update_freq', 'default')})"
                    )

        def ts_validation_hook(results: list[Atoms]) -> list[dict[str, Any] | None]:
            """Validate TS structures."""
            validation_results: list[dict[str, Any] | None] = []
            for atoms_copy in results:
                validation_result = validate_ts_structure(atoms_copy, self.explorer)
                if isinstance(validation_result, tuple):
                    validation_results.append(validation_result[0])
                else:
                    validation_results.append(validation_result)
            return validation_results

        def prepare_ts_frequency_kwargs(atoms: Atoms) -> dict[str, Any]:
            """Prepare frequency kwargs for TS."""
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
            prepare_optimizer_kwargs=prepare_ts_optimizer_kwargs,
            post_optimization_hook=post_ts_optimization_hook if verbose >= 1 else None,
            validation_hook=ts_validation_hook if validate_ts else None,
            prepare_frequency_kwargs=prepare_ts_frequency_kwargs,
            result_key_name="is_ts",
            log_prefix="Transition state",
        )


REGISTRY.register(LocalTSStrategy)
