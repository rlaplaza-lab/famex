"""Local minima optimization strategy."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any

from ase import Atoms

from qme.core.base_strategy import BaseStrategy, StrategyMetadata
from qme.core.registry import REGISTRY
from qme.strategies.helpers import _get_local_optimizer_class
from qme.strategies.utils import StrategyUtils
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

        Returns:
        -------
        dict[str, Union[Atoms, list[Atoms], bool, int, float, str]]
            Standardized result dictionary containing:
            - optimized_atoms: Optimized structure(s) (Atoms or list[Atoms])
            - strategy: Strategy name (str)
            - converged: Whether optimization converged (bool)
            - steps_taken: Number of optimization steps (int)
            - frequency_analysis: Frequency analysis results (dict, optional)
            - is_minimum: Whether structure is a minimum (bool, optional)
            - free_energy_correction: Free energy correction in eV (float, optional)

        """
        # Handle single Atoms object (runtime check for API misuse)
        # This is defensive programming - type signature says Sequence[Atoms] but
        # we handle single Atoms at runtime for better API ergonomics
        if isinstance(atoms_list, Atoms):  # type: ignore[unreachable]
            atoms_list = [atoms_list]  # type: ignore[unreachable]

        # Convert Sequence to list for validation and iteration
        atoms_list = list(atoms_list)
        self.validate_inputs(atoms_list)

        local_optimizer_name = kwargs.get("local_optimizer_name", "sella")
        verbose = kwargs.get("verbose", 1)

        # Start profiling if available
        if self.profiler is not None:
            self.profiler.snapshot_memory()

        # Log minima optimization start
        if verbose >= 1:
            logger.info(f"Starting local minima optimization with {local_optimizer_name}")
            logger.info(f"Force threshold: {fmax} eV/Å, Max steps: {steps}")

        opt_class = _get_local_optimizer_class(local_optimizer_name)
        # Check if single-element list
        single_input = len(atoms_list) == 1
        atoms_iter = atoms_list

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

            opt_kwargs = getattr(self.explorer, "optimizer_kwargs", {}) or {}
            # Add verbosity control and profiler
            opt_kwargs = dict(opt_kwargs)
            opt_kwargs.setdefault("verbose", getattr(self.explorer, "verbose", 1))
            if self.profiler is not None:
                opt_kwargs["profiler"] = self.profiler

            if local_optimizer_name.lower() == "sella":
                # Sella-specific kwargs for minima search
                opt_kwargs.setdefault("internal", True)
                opt_kwargs.setdefault("order", 0)
                # Note: SELLA computes its own Hessian internally and doesn't accept
                # an initial Hessian as a keyword argument. The initial_hessian from
                # explorer is not used for SELLA.

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

        # Log completion
        if verbose >= 1:
            if single_input and results:
                converged = bool(converged_flags[0])
                steps_taken = step_counts[0]
                logger.info(
                    f"Minima optimization completed: converged={converged}, steps={steps_taken}",
                )
            else:
                total_converged = sum(converged_flags)
                total_structures = len(converged_flags)
                logger.info(
                    f"Minima optimization completed: {total_converged}/{total_structures} structures converged",
                )

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

        # Prepare result and merge profiler data
        if single_input and results:
            result = self.prepare_result(
                results[0],
                steps_taken=step_counts[0],
                converged=bool(converged_flags[0]),
            )
            if frequency_results[0] is not None:
                result["frequency_analysis"] = frequency_results[0]
                result["is_minimum"] = frequency_results[0]["is_minimum"]
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
            if any(f is not None for f in frequency_results):
                result["frequency_analysis"] = frequency_results
                result["is_minimum"] = [
                    f["is_minimum"] if f is not None else None for f in frequency_results
                ]
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
REGISTRY.register(LocalMinimaStrategy)
