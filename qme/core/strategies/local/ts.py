"""Local transition state optimization strategy."""

from __future__ import annotations

from typing import Any, Union

from ase import Atoms

from qme.core.strategies.local.helpers import (
    _get_local_optimizer_class,
    _validate_ts_optimization_setup,
    validate_ts_structure,
)
from qme.core.strategy import REGISTRY, BaseStrategy, StrategyMetadata
from qme.core.strategy_utils import StrategyUtils
from qme.logging_utils import get_qme_logger

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
        atoms_list: list[Atoms],
        fmax: float = 0.05,
        steps: int = 1000,
        validate_ts: bool = False,
        **kwargs: Any,
    ) -> dict[str, Union[Atoms, list[Atoms], bool, int, float, str]]:
        """Run local transition state optimization.

        Parameters
        ----------
        atoms_list : list[Atoms]
            List of structures to optimize
        fmax : float, default=0.05
            Force convergence threshold
        steps : int, default=1000
            Maximum optimization steps
        **kwargs : Any
            Additional keyword arguments

        Returns
        -------
        dict[str, Union[Atoms, list[Atoms], bool, int, float, str]]
            Standardized result dictionary containing:
            - optimized_atoms: Optimized structure(s) (Atoms or list[Atoms])
            - strategy: Strategy name (str)
            - converged: Whether optimization converged (bool)
            - steps_taken: Number of optimization steps (int)
            - ts_validation: Transition state validation results (dict, optional)
        """
        self.validate_inputs(atoms_list)

        local_optimizer_name = kwargs.get("local_optimizer_name", "sella")
        verbose = kwargs.get("verbose", 1)

        # Log TS optimization start
        if verbose >= 1:
            logger.info(f"Starting local transition state optimization with {local_optimizer_name}")
            logger.info(f"Force threshold: {fmax} eV/Å, Max steps: {steps}")

        # Validate TS optimization setup - hardcoded restrictions
        _validate_ts_optimization_setup(self.explorer.backend, local_optimizer_name)

        opt_class = _get_local_optimizer_class(local_optimizer_name)
        # Accept either a single Atoms instance or a list of them
        single_input = False
        if not isinstance(atoms_list, (list, tuple)):
            single_input = True
            atoms_iter = [atoms_list]
        else:
            atoms_iter = atoms_list
            # If it's a single-element list, treat as single input
            if len(atoms_list) == 1:
                single_input = True

        results = []
        step_counts = []
        converged_flags = []

        for atoms in atoms_iter:
            # CRITICAL FIX: Make a copy of atoms before optimization to prevent in-place modifications
            # that corrupt the coordinate system for subsequent Hessian calculations
            atoms_copy = atoms.copy()

            self.explorer._create_and_attach_calculator(atoms_copy)
            self.explorer._apply_constraints(atoms_copy)
            opt_kwargs = getattr(self.explorer, "ts_kwargs", {}) or {}
            # Add verbosity control
            opt_kwargs = dict(opt_kwargs)
            opt_kwargs.setdefault("verbose", getattr(self.explorer, "verbose", 1))

            normalized_name = local_optimizer_name.lower()
            if normalized_name == "sella":
                # Sella-specific kwargs for TS search
                opt_kwargs.setdefault("internal", True)
                opt_kwargs.setdefault("order", 1)
                # Check if Hessian is provided in explorer
                if (
                    hasattr(self.explorer, "initial_hessian")
                    and self.explorer.initial_hessian is not None
                ):
                    opt_kwargs["hessian"] = self.explorer.initial_hessian
            elif normalized_name in {
                "trust-krylov-ts",
                "trustkrylovts",
                "trust_krylov_ts",
                "trust-krylov-transition",
            }:
                opt_kwargs.setdefault("hessian_update_freq", 1)
                opt_kwargs.setdefault("mode_recompute_interval", 1)
                opt_kwargs.setdefault("index_tolerance", 5e-4)
                opt_kwargs.setdefault("min_positive_eigenvalue", 4e-3)
                opt_kwargs.setdefault("negative_mode_boost", 8e-3)

            opt = opt_class(atoms_copy, **opt_kwargs)
            opt.run(fmax=fmax, steps=steps)

            # Get step count and convergence status using helpers
            steps_taken = StrategyUtils.get_step_count(opt)
            converged = StrategyUtils.get_convergence_status(opt, atoms_copy)

            results.append(atoms_copy)
            step_counts.append(steps_taken)
            converged_flags.append(converged)

        # Validate TS structures if requested
        validation_results = []
        if validate_ts:
            for atoms_copy in results:
                validation_result = validate_ts_structure(atoms_copy, self.explorer)
                validation_results.append(validation_result)
        else:
            validation_results = [None] * len(results)

        # Log completion
        if verbose >= 1:
            if single_input and results:
                converged = bool(converged_flags[0])
                steps_taken = step_counts[0]
                logger.info(
                    f"Transition state optimization completed: converged={converged}, steps={steps_taken}"
                )
            else:
                total_converged = sum(converged_flags)
                total_structures = len(converged_flags)
                logger.info(
                    f"Transition state optimization completed: {total_converged}/{total_structures} structures converged"
                )

        if single_input and results:
            result = self.prepare_result(
                results[0],
                steps_taken=step_counts[0],
                converged=bool(converged_flags[0]),
            )
            if validation_results[0] is not None:
                result["ts_validation"] = validation_results[0]
            return result
        else:
            result = self.prepare_result(
                results,
                steps_taken=step_counts,
                converged=[bool(c) for c in converged_flags],
            )
            if any(v is not None for v in validation_results):
                result["ts_validation"] = validation_results
            return result


# Register the strategy
REGISTRY.register(LocalTSStrategy)
