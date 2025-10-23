"""Local transition state optimization strategy."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

from qme.core.base_strategy import BaseStrategy, StrategyMetadata
from qme.core.registry import REGISTRY
from qme.strategies.helpers import (
    _get_local_optimizer_class,
    _validate_ts_optimization_setup,
    validate_ts_structure,
)
from qme.strategies.utils import StrategyUtils
from qme.utils.logging import get_qme_logger

if TYPE_CHECKING:
    from ase import Atoms

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
        calculate_frequencies: bool = False,
        **kwargs: Any,
    ) -> dict[str, Atoms | list[Atoms] | bool | int | float | str]:
        """Run local transition state optimization.

        Parameters
        ----------
        atoms_list : list[Atoms]
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

        Returns:
        -------
        dict[str, Union[Atoms, list[Atoms], bool, int, float, str]]
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
        self.validate_inputs(atoms_list)

        local_optimizer_name = kwargs.get("local_optimizer_name", "sella")
        verbose = kwargs.get("verbose", 1)

        # Start profiling if available
        if self.profiler is not None:
            self.profiler.snapshot_memory()

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

            # Profile calculator setup
            with (
                self.profiler.profile_section("calculator_setup")
                if self.profiler
                else nullcontext()
            ):
                self.explorer._create_and_attach_calculator(atoms_copy)
                self.explorer._apply_constraints(atoms_copy)

            opt_kwargs = getattr(self.explorer, "ts_kwargs", {}) or {}
            # Add verbosity control and profiler
            opt_kwargs = dict(opt_kwargs)
            opt_kwargs.setdefault("verbose", getattr(self.explorer, "verbose", 1))
            if self.profiler is not None:
                opt_kwargs["profiler"] = self.profiler

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

            # Profile optimization execution
            with self.profiler.profile_section("optimization") if self.profiler else nullcontext():
                opt.run(fmax=fmax, steps=steps)

            # Get step count and convergence status using helpers
            steps_taken = StrategyUtils.get_step_count(opt)
            converged = StrategyUtils.get_convergence_status(opt, atoms_copy)

            # Increment step counter in profiler
            if self.profiler is not None:
                self.profiler.increment_call("optimizer_steps", steps_taken)

            results.append(atoms_copy)
            step_counts.append(steps_taken)
            converged_flags.append(converged)

        # Validate TS structures if requested
        validation_results = []
        if validate_ts:
            with self.profiler.profile_section("ts_validation") if self.profiler else nullcontext():
                for atoms_copy in results:
                    validation_result = validate_ts_structure(atoms_copy, self.explorer)
                    validation_results.append(validation_result)
        else:
            validation_results = [None] * len(results)

        # Perform frequency analysis if requested
        frequency_results = []
        if calculate_frequencies:
            with (
                self.profiler.profile_section("frequency_analysis")
                if self.profiler
                else nullcontext()
            ):
                temperature = kwargs.get("temperature", 298.15)
                for atoms_copy in results:
                    freq_result = self.explorer.calculate_frequencies(
                        atoms=atoms_copy,
                        temperature=temperature,
                        save_hessian=False,  # Don't save large Hessian matrix by default
                    )
                    frequency_results.append(freq_result)
        else:
            frequency_results = [None] * len(results)

        # Log completion
        if verbose >= 1:
            if single_input and results:
                converged = bool(converged_flags[0])
                steps_taken = step_counts[0]
                logger.info(
                    f"Transition state optimization completed: converged={converged}, steps={steps_taken}",
                )
            else:
                total_converged = sum(converged_flags)
                total_structures = len(converged_flags)
                logger.info(
                    f"Transition state optimization completed: {total_converged}/{total_structures} structures converged",
                )

        # Prepare result and merge profiler data
        if single_input and results:
            result = self.prepare_result(
                results[0],
                steps_taken=step_counts[0],
                converged=bool(converged_flags[0]),
            )
            if validation_results[0] is not None:
                result["ts_validation"] = validation_results[0]
            if frequency_results[0] is not None:
                result["frequency_analysis"] = frequency_results[0]
                result["is_ts"] = frequency_results[0]["is_ts"]
                # Calculate free energy correction (G - E) = H - TS - E = (E + PV) - TS - E = -TS
                # For ideal gas: PV = 0, so G - E = -TS
                thermo = frequency_results[0]["thermodynamic_properties"]
                result["free_energy_correction"] = (
                    -thermo["entropy"] * temperature / 1000.0
                )  # Convert K to eV/K
        else:
            result = self.prepare_result(
                results,
                steps_taken=step_counts,
                converged=[bool(c) for c in converged_flags],
            )
            if any(v is not None for v in validation_results):
                result["ts_validation"] = validation_results
            if any(f is not None for f in frequency_results):
                result["frequency_analysis"] = frequency_results
                result["is_ts"] = [f["is_ts"] if f is not None else None for f in frequency_results]
                # Calculate free energy corrections for all structures
                free_energy_corrections = []
                for f in frequency_results:
                    if f is not None:
                        thermo = f["thermodynamic_properties"]
                        free_energy_corrections.append(-thermo["entropy"] * temperature / 1000.0)
                    else:
                        free_energy_corrections.append(None)
                result["free_energy_correction"] = free_energy_corrections

        return self._merge_profiler_results(result)


# Register the strategy
REGISTRY.register(LocalTSStrategy)
