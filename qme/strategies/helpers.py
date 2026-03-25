"""Helper functions for local strategies."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import nullcontext
from typing import Any, cast

from ase import Atoms

from qme.strategies.utils import StrategyUtils
from qme.utils.logging import get_qme_logger

logger = get_qme_logger(__name__)


def validate_ts_structure(
    atoms: Atoms,
    explorer: Any,
    threshold: float = 50.0,
    return_hessian: bool = False,
    verbose: int = 1,
) -> dict[str, Any] | tuple[dict[str, Any], Any]:
    from qme.analysis.frequency import FrequencyAnalysis

    if getattr(atoms, "calc", None) is None:
        explorer._create_and_attach_calculator(atoms)

    try:
        freq_analysis = FrequencyAnalysis(atoms=atoms, calculator=atoms.calc, verbose=0)
        freq_analysis.calculate_hessian(method="auto")
        freq_analysis.diagonalize_hessian()

        validation_result = freq_analysis.is_transition_state(threshold=threshold)
        if not validation_result["is_transition_state"]:
            n_imaginary = validation_result["n_imaginary_frequencies"]
            assessment = validation_result["assessment"]

            if verbose >= 1:
                logger.warning(
                    "TS validation failed: %s. Expected exactly 1 imaginary frequency, "
                    "found %d imaginary frequencies.",
                    assessment,
                    n_imaginary,
                )

            if verbose >= 2:
                imaginary_freqs = validation_result.get("imaginary_frequencies")
                if imaginary_freqs and isinstance(imaginary_freqs, list | tuple):
                    logger.warning(
                        f"Imaginary frequencies (cm^-1): {[f'{f:.1f}' for f in imaginary_freqs]}",
                    )
        else:
            if verbose >= 1:
                logger.info("TS validation passed: structure has exactly 1 imaginary frequency")
            elif verbose >= 2:
                logger.info(
                    "✓ Starting structure validated as transition state (1 imaginary frequency)",
                )

        hessian = getattr(freq_analysis, "_hessian", None) if return_hessian else None

        if return_hessian:
            return validation_result, hessian
        return validation_result

    except Exception as e:
        if verbose >= 1:
            logger.warning(
                f"Could not validate transition state: {e}. "
                "TS validation will proceed without validation.",
            )
        if return_hessian:
            return {
                "is_transition_state": False,
                "assessment": f"Validation failed: {e}",
            }, None
        return {"is_transition_state": False, "assessment": f"Validation failed: {e}"}


def _validate_ts_optimization_setup(backend: str, optimizer_name: str) -> None:
    FORBIDDEN_BACKENDS_FOR_TS = {"mock"}
    FORBIDDEN_OPTIMIZERS_FOR_TS = {"lbfgs", "l-bfgs", "l_bfgs", "bfgs", "fire"}

    if backend.lower() in FORBIDDEN_BACKENDS_FOR_TS:
        msg = (
            f"Backend '{backend}' is not suitable for transition state optimization. "
            f"Use a real ML potential backend (uma, aimnet2, mace, so3lr) instead."
        )
        raise ValueError(
            msg,
        )

    normalized_name = optimizer_name.lower()
    if normalized_name in FORBIDDEN_OPTIMIZERS_FOR_TS:
        msg = (
            f"Optimizer '{optimizer_name}' is not suitable for transition state "
            "optimization. Use 'sella' or 'rfo' for TS searches."
        )
        raise ValueError(
            msg,
        )


def _get_local_optimizer_class(name: str) -> type[Any]:
    name = (name or "").lower()

    if name == "sella":
        from qme.optimizers.ase_wrappers import VerboseSella

        return VerboseSella

    if name in ("trust-krylov", "trustkrylov", "trust_krylov"):
        from qme.optimizers.scipy_optimizers import TrustKrylov

        return TrustKrylov
    if name in ("rfo", "rfo-ts", "rational-function", "rational_function"):
        from qme.optimizers.rfo_optimizer import RFOTransitionState

        return RFOTransitionState
    if name in ("trust-ncg", "trustncg", "trust_ncg"):
        from qme.optimizers.scipy_optimizers import TrustNCG

        return TrustNCG
    if name in ("trust-exact", "trustexact", "trust_exact"):
        from qme.optimizers.scipy_optimizers import TrustExact

        return TrustExact
    if name in ("newton-cg", "newtoncg", "newton_cg"):
        from qme.optimizers.scipy_optimizers import NewtonCG

        return NewtonCG

    try:
        if name in ("lbfgs", "l-bfgs", "l_bfgs"):
            from qme.optimizers.ase_wrappers import VerboseLBFGS

            return VerboseLBFGS
        if name in ("bfgs",):
            from qme.optimizers.ase_wrappers import VerboseBFGS

            return VerboseBFGS
        if name in ("fire",):
            from qme.optimizers.ase_wrappers import VerboseFIRE

            return VerboseFIRE
    except Exception as e:  # pragma: no cover - ASE optional in some envs
        msg = f"Requested optimizer '{name}' is not available: {e}"
        raise ImportError(msg)

    msg = f"Unknown optimizer name: {name}"
    raise ValueError(msg)


def _calculate_free_energy_correction(
    frequency_result: dict[str, Any] | None, temperature: float
) -> float | None:
    if frequency_result is None:
        return None
    thermo = frequency_result.get("thermodynamic_properties", {})
    if not isinstance(thermo, dict):
        return None
    entropy = thermo.get("entropy", 0.0)
    if not isinstance(entropy, int | float):
        return None
    return -entropy * temperature / 1000.0


def _run_local_optimization_common(
    strategy: Any,
    atoms_list: Sequence[Atoms],
    fmax: float,
    steps: int,
    local_optimizer_name: str,
    verbose: int,
    calculate_frequencies: bool,
    temperature: float,
    prepare_optimizer_kwargs: Callable[[str, Any], dict[str, Any]],
    post_optimization_hook: Callable[[Any, Atoms, dict[str, Any]], None] | None = None,
    validation_hook: Callable[[list[Atoms]], list[dict[str, Any] | None]] | None = None,
    prepare_frequency_kwargs: Callable[[Atoms], dict[str, Any]] | None = None,
    result_key_name: str = "is_minimum",
    log_prefix: str = "optimization",
) -> dict[str, Any]:
    """Execute common local optimization pattern for minima and TS."""
    atoms_list = list(atoms_list)
    strategy.validate_inputs(atoms_list)

    if strategy.profiler is not None:
        strategy.profiler.snapshot_memory()

    if verbose >= 1:
        logger.info(f"Starting local {log_prefix.lower()} optimization with {local_optimizer_name}")
        logger.info(f"Force threshold: {fmax} eV/Å, Max steps: {steps}")

    opt_class = _get_local_optimizer_class(local_optimizer_name)
    single_input = len(atoms_list) == 1

    results = []
    step_counts = []
    converged_flags = []

    for atoms in atoms_list:
        atoms_copy = atoms.copy()

        with (
            strategy.profiler.profile_section("calculator_setup")
            if strategy.profiler
            else nullcontext()
        ):
            strategy.explorer._create_and_attach_calculator(atoms_copy)
            strategy.explorer._apply_constraints(atoms_copy)

        opt_kwargs = prepare_optimizer_kwargs(local_optimizer_name, strategy.explorer)
        opt_kwargs.setdefault("verbose", getattr(strategy.explorer, "verbose", 1))
        if strategy.profiler is not None:
            opt_kwargs["profiler"] = strategy.profiler

        opt = opt_class(atoms_copy, **opt_kwargs)

        with (
            strategy.profiler.profile_section("optimization")
            if strategy.profiler
            else nullcontext()
        ):
            opt.run(fmax=fmax, steps=steps)

        steps_taken = StrategyUtils.get_step_count(opt)
        converged = StrategyUtils.get_convergence_status(opt, atoms_copy)

        if post_optimization_hook is not None:
            post_optimization_hook(opt, atoms_copy, opt_kwargs)

        if strategy.profiler is not None:
            strategy.profiler.increment_call("optimizer_steps", steps_taken)

        results.append(atoms_copy)
        step_counts.append(steps_taken)
        converged_flags.append(converged)

    if verbose >= 1:
        if single_input and results:
            converged = bool(converged_flags[0])
            steps_taken = step_counts[0]
            logger.info(
                f"{log_prefix} optimization completed: converged={converged}, steps={steps_taken}",
            )
        else:
            total_converged = sum(converged_flags)
            total_structures = len(converged_flags)
            logger.info(
                f"{log_prefix} optimization completed: "
                f"{total_converged}/{total_structures} structures converged",
            )

    validation_results: list[dict[str, Any] | None] = []
    if validation_hook is not None:
        with (
            strategy.profiler.profile_section("validation") if strategy.profiler else nullcontext()
        ):
            validation_results = validation_hook(results)
    else:
        validation_results = [None] * len(results)

    frequency_results = []
    if calculate_frequencies:
        with (
            strategy.profiler.profile_section("frequency_analysis")
            if strategy.profiler
            else nullcontext()
        ):
            for atoms_copy in results:
                freq_kwargs = {
                    "atoms": atoms_copy,
                    "temperature": temperature,
                    "save_hessian": False,
                }
                if prepare_frequency_kwargs is not None:
                    custom_kwargs = prepare_frequency_kwargs(atoms_copy)
                    freq_kwargs.update(custom_kwargs)
                freq_result = strategy.explorer.calculate_frequencies(**freq_kwargs)
                frequency_results.append(freq_result)
    else:
        frequency_results = [None] * len(results)

    if single_input and results:
        result = strategy.prepare_result(
            results[0],
            steps_taken=step_counts[0],
            converged=bool(converged_flags[0]),
        )
        if validation_results[0] is not None:
            result["ts_validation"] = validation_results[0]
        if frequency_results[0] is not None:
            result["frequency_analysis"] = frequency_results[0]
            result[result_key_name] = frequency_results[0].get(result_key_name)
            result["free_energy_correction"] = _calculate_free_energy_correction(
                frequency_results[0], temperature
            )
    else:
        result = strategy.prepare_result(
            results,
            steps_taken=step_counts,
            converged=[bool(c) for c in converged_flags],
        )
        if any(v is not None for v in validation_results):
            result["ts_validation"] = validation_results
        if any(f is not None for f in frequency_results):
            result["frequency_analysis"] = frequency_results
            result[result_key_name] = [
                f.get(result_key_name) if f is not None else None for f in frequency_results
            ]
            free_energy_corrections = [
                _calculate_free_energy_correction(f, temperature) for f in frequency_results
            ]
            result["free_energy_correction"] = free_energy_corrections

    merged_result = strategy._merge_profiler_results(result)
    return cast(dict[str, Any], merged_result)


def filter_interpolation_kwargs(
    kwargs: dict[str, Any], allowed_keys: set[str] | None = None
) -> dict[str, Any]:
    if allowed_keys is None:
        allowed_keys = {"rmsd_threshold", "energy_threshold", "calculator"}
    return {k: v for k, v in kwargs.items() if k in allowed_keys}
