"""Local transition state optimization strategy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ase import Atoms

from famex.core.base_strategy import BaseStrategy, StrategyMetadata
from famex.core.registry import REGISTRY
from famex.strategies.helpers import (
    _run_local_optimization_common,
    _validate_ts_optimization_setup,
    build_optimizer_kwargs,
    enforce_require_ts,
    resolve_require_ts,
    validate_ts_structure,
)
from famex.strategies.kwargs_policy import LOCAL_OPTIMIZATION_RUN_KEYS, pick_pass_through_kwargs
from famex.strategies.utils import StrategyUtils
from famex.utils.logging import get_famex_logger

logger = get_famex_logger(__name__)


class LocalTSStrategy(BaseStrategy):
    """Local transition state optimization strategy."""

    metadata = StrategyMetadata(
        name="ts:local",
        target="ts",
        strategy="local",
        description="Local transition-state optimization (RFO preferred)",
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
        cleanup_frequencies: bool = False,
        require_ts: bool | None = None,
        **kwargs: Any,
    ) -> dict[str, Atoms | list[Atoms] | bool | int | float | str]:
        run_extra = pick_pass_through_kwargs(kwargs, LOCAL_OPTIMIZATION_RUN_KEYS)
        local_optimizer_name = run_extra.get("local_optimizer_name", "rfo")
        verbose = int(run_extra.get("verbose", getattr(self.explorer, "verbose", 1)))
        temperature = run_extra.get("temperature", 298.15)
        if cleanup_frequencies is False:
            cleanup_frequencies = bool(getattr(self.explorer, "cleanup_frequencies", False))

        require_ts = resolve_require_ts(require_ts, self.explorer)
        kwargs.pop("require_ts", None)

        _validate_ts_optimization_setup(self.explorer.backend, local_optimizer_name)

        def post_ts_optimization_hook(opt: Any, atoms: Atoms, opt_kwargs: dict[str, Any]) -> None:
            if hasattr(opt, "hessian_calls"):
                if verbose >= 1:
                    logger.info(
                        "Hessian computed %s time(s) over %s steps (update_freq=%s)",
                        opt.hessian_calls,
                        StrategyUtils.get_step_count(opt),
                        opt_kwargs.get("hessian_update_freq", "default"),
                    )

        def ts_validation_hook(results: list[Atoms]) -> list[dict[str, Any] | None]:
            validation_results: list[dict[str, Any] | None] = []
            for atoms_copy in results:
                validation_result = validate_ts_structure(atoms_copy, self.explorer)
                if isinstance(validation_result, tuple):
                    validation_results.append(validation_result[0])
                else:
                    validation_results.append(validation_result)
            return validation_results

        force_fd = bool(getattr(self.explorer, "force_finite_diff_hessian", False))

        def prepare_frequency_kwargs(_atoms: Atoms) -> dict[str, Any]:
            return {"method": "finite_differences"} if force_fd else {}

        result = _run_local_optimization_common(
            strategy=self,
            atoms_list=atoms_list,
            fmax=fmax,
            steps=steps,
            local_optimizer_name=local_optimizer_name,
            verbose=verbose,
            calculate_frequencies=calculate_frequencies,
            temperature=temperature,
            prepare_optimizer_kwargs=lambda name, explorer: build_optimizer_kwargs(
                name, explorer, target="ts"
            ),
            post_optimization_hook=post_ts_optimization_hook if verbose >= 1 else None,
            validation_hook=ts_validation_hook if validate_ts else None,
            prepare_frequency_kwargs=prepare_frequency_kwargs,
            result_key_name="is_ts",
            log_prefix="Transition state",
            cleanup_frequencies=cleanup_frequencies,
            stationary_point_target="ts",
            prepare_cleanup_optimizer_kwargs=lambda name, explorer: build_optimizer_kwargs(
                name, explorer, target="ts"
            ),
            record_history=bool(run_extra.get("record_history", False)),
        )

        optimized_atoms = result.get("optimized_atoms")
        freq_val = result.get("frequency_analysis")
        freq_analysis = freq_val if isinstance(freq_val, dict) else None
        freq_analysis = enforce_require_ts(
            require_ts=require_ts,
            atoms=optimized_atoms if isinstance(optimized_atoms, Atoms) else None,
            explorer=self.explorer,
            freq_analysis=freq_analysis,
            check_converged=(False if not result.get("converged") else None),
            temperature=temperature,
        )
        if require_ts and freq_analysis is not None:
            result["frequency_analysis"] = freq_analysis
            result["is_ts"] = freq_analysis.get("is_ts")
        return result


REGISTRY.register(LocalTSStrategy)
