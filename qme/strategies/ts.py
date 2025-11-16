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

        # Validate TS optimization setup - hardcoded restrictions
        _validate_ts_optimization_setup(self.explorer.backend, local_optimizer_name)

        def prepare_ts_optimizer_kwargs(optimizer_name: str, explorer: Any) -> dict[str, Any]:
            """Prepare optimizer kwargs for TS optimization."""
            opt_kwargs = getattr(explorer, "ts_kwargs", {}) or {}
            opt_kwargs = dict(opt_kwargs)

            normalized_name = optimizer_name.lower()
            if normalized_name == "sella":
                # Sella-specific kwargs for TS search
                opt_kwargs.setdefault("internal", True)
                opt_kwargs.setdefault("order", 1)
                # Remove hessian_method if present (Sella doesn't accept it)
                opt_kwargs.pop("hessian_method", None)
                # Note: SELLA computes its own Hessian internally and doesn't accept
                # an initial Hessian as a keyword argument. The initial_hessian from
                # explorer is not used for SELLA.
            elif normalized_name in (
                "rfo",
                "rfo-ts",
                "rational-function",
                "rational_function",
            ):
                # RFO optimizer: recompute Hessian every 10 steps for better convergence
                # TS optimization needs accurate Hessian information
                opt_kwargs.setdefault("hessian_update_freq", 10)
                opt_kwargs.setdefault("hessian_method", "auto")
                # Use slightly larger trust radius initially for better convergence
                opt_kwargs.setdefault("trust_radius", 0.02)  # Double the default
                opt_kwargs.setdefault("max_trust_radius", 0.06)  # Double the default

            # Force finite difference hessian if flag is set
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
            """Post-optimization hook for TS-specific diagnostic logging."""
            # Diagnostic: log Hessian call count for rfo
            if hasattr(opt, "hessian_calls"):
                if self.explorer.verbose >= 1:
                    logger.info(
                        f"Hessian computed {opt.hessian_calls} time(s) "
                        f"over {StrategyUtils.get_step_count(opt)} steps "
                        f"(update_freq={opt_kwargs.get('hessian_update_freq', 'default')})"
                    )

        def ts_validation_hook(results: list[Atoms]) -> list[dict[str, Any] | None]:
            """TS validation hook."""
            validation_results: list[dict[str, Any] | None] = []
            for atoms_copy in results:
                validation_result = validate_ts_structure(atoms_copy, self.explorer)
                # Handle tuple return (validation_result, hessian) or dict return
                # validate_ts_structure returns dict[str, Any] | tuple[dict[str, Any], Any]
                if isinstance(validation_result, tuple):
                    validation_results.append(validation_result[0])
                else:
                    # Must be dict[str, Any] based on return type
                    validation_results.append(validation_result)
            return validation_results

        def prepare_ts_frequency_kwargs(atoms: Atoms) -> dict[str, Any]:
            """Prepare frequency analysis kwargs for TS (supports finite difference override)."""
            freq_kwargs: dict[str, Any] = {}
            # Pass method="finite_differences" if force_finite_diff_hessian is True
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


# Register the strategy
REGISTRY.register(LocalTSStrategy)
