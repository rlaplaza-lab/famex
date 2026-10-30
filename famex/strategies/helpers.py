"""Helper functions for local strategies."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import nullcontext
from typing import Any, Literal, cast

from ase import Atoms

from famex.analysis.thermodynamics import read_free_energy_correction
from famex.strategies.utils import StrategyUtils
from famex.utils.logging import get_famex_logger

logger = get_famex_logger(__name__)

# Not an optimizer keyword; stored on explorer.ts_kwargs by the CLI for strategy use.
_REQUIRE_TS_KWARG = "require_ts"


def resolve_require_ts(require_ts: bool | None, explorer: Any) -> bool:
    """Resolve whether a first-order saddle is required.

    An explicit ``require_ts`` argument wins. Otherwise read
    ``explorer.ts_kwargs["require_ts"]`` (set by ``--require-ts``).
    """
    if require_ts is not None:
        return bool(require_ts)
    explorer_ts_kwargs = getattr(explorer, "ts_kwargs", None) or {}
    return bool(explorer_ts_kwargs.get(_REQUIRE_TS_KWARG, False))


def strip_require_ts_from_optimizer_kwargs(opt_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Remove ``require_ts`` so it is not forwarded to TS optimizers."""
    opt_kwargs.pop(_REQUIRE_TS_KWARG, None)
    return opt_kwargs


def ts_has_one_imaginary_mode(freq_analysis: dict[str, Any] | None) -> bool:
    """Return True when frequency analysis confirms a first-order saddle."""
    if not freq_analysis:
        return False
    raw_ts = freq_analysis.get("ts_analysis")
    ts_info: dict[str, Any] = raw_ts if isinstance(raw_ts, dict) else {}
    n_imaginary: int | None = None
    for key in ("n_imaginary_frequencies", "num_imaginary_modes", "n_imaginary_modes"):
        if freq_analysis.get(key) is not None:
            n_imaginary = int(freq_analysis[key])
            break
    if n_imaginary is None and ts_info.get("n_imaginary_frequencies") is not None:
        n_imaginary = int(ts_info["n_imaginary_frequencies"])
    # Explicit imag-count wins over boolean is_ts flags (0 must not be treated as missing).
    if n_imaginary is not None:
        return n_imaginary == 1
    if freq_analysis.get("is_ts") is True:
        return True
    return ts_info.get("is_transition_state") is True


def enforce_require_ts(
    *,
    require_ts: bool,
    atoms: Atoms | None,
    explorer: Any,
    freq_analysis: dict[str, Any] | None = None,
    check_converged: bool | None = None,
    temperature: float = 298.15,
) -> dict[str, Any] | None:
    """Raise if ``require_ts`` and the structure is not a first-order saddle.

    When ``check_converged`` is ``False``, treat failed local refinement as an
    error. ``None``/``True`` skips that check. Returns ``freq_analysis``,
    computing it via ``explorer.calculate_frequencies`` when missing.
    """
    if not require_ts:
        return freq_analysis

    errors: list[str] = []
    if check_converged is False:
        errors.append("Local TS refinement failed to converge.")

    if freq_analysis is None and atoms is not None and explorer is not None:
        freq_analysis = explorer.calculate_frequencies(
            atoms=atoms,
            temperature=temperature,
            save_hessian=False,
        )

    if freq_analysis is None:
        errors.append("Frequency analysis unavailable for TS validation.")
    elif not ts_has_one_imaginary_mode(freq_analysis):
        errors.append(
            "Refined structure is not a first-order saddle "
            "(vibrational characterization failed: need exactly one imaginary mode)."
        )

    if errors:
        raise RuntimeError("TS validation failed with require_ts=True:\n- " + "\n- ".join(errors))
    return freq_analysis


def validate_ts_structure(
    atoms: Atoms,
    explorer: Any,
    threshold: float = 50.0,
    return_hessian: bool = False,
    verbose: int = 1,
) -> dict[str, Any] | tuple[dict[str, Any], Any]:
    from famex.analysis.frequency import FrequencyAnalysis

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
                        "Imaginary frequencies (cm^-1): %s",
                        [f"{f:.1f}" for f in imaginary_freqs],
                    )
        else:
            if verbose >= 1:
                logger.info("TS validation passed: structure has exactly 1 imaginary frequency")

        hessian = getattr(freq_analysis, "_hessian", None) if return_hessian else None

        if return_hessian:
            return validation_result, hessian
        return validation_result

    except Exception as e:
        if verbose >= 1:
            logger.warning("Could not validate transition state: %s", e)
        raise


def _validate_ts_optimization_setup(backend: str, optimizer_name: str) -> None:
    FORBIDDEN_BACKENDS_FOR_TS = {"mock"}
    FORBIDDEN_OPTIMIZERS_FOR_TS = {
        "lbfgs",
        "l-bfgs",
        "l_bfgs",
        "bfgs",
        "fire",
        "newton-cg",
        "newtoncg",
        "newton_cg",
    }

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
            "optimization. Use 'sella', 'sella-analytical', 'rfo', or a SciPy trust-region "
            "TS optimizer "
            "('trust-krylov', 'trust-ncg', 'trust-exact') for TS searches."
        )
        raise ValueError(
            msg,
        )


def _get_local_optimizer_class(name: str) -> type[Any]:
    name = (name or "").lower()

    if name == "sella":
        from famex.optimizers.ase_wrappers import VerboseSella

        return VerboseSella

    if name in ("sella-analytical", "sella_analytical", "sellaanalytical"):
        from famex.optimizers.ase_wrappers import VerboseSellaAnalytical

        return VerboseSellaAnalytical

    if name in ("trust-krylov", "trustkrylov", "trust_krylov"):
        from famex.optimizers.scipy_optimizers import TrustKrylov

        return TrustKrylov
    if name in ("rfo", "rfo-ts", "rational-function", "rational_function"):
        from famex.optimizers.rfo_optimizer import RFOTransitionState

        return RFOTransitionState
    if name in ("trust-ncg", "trustncg", "trust_ncg"):
        from famex.optimizers.scipy_optimizers import TrustNCG

        return TrustNCG
    if name in ("trust-exact", "trustexact", "trust_exact"):
        from famex.optimizers.scipy_optimizers import TrustExact

        return TrustExact
    if name in ("newton-cg", "newtoncg", "newton_cg"):
        from famex.optimizers.scipy_optimizers import NewtonCG

        return NewtonCG

    try:
        if name in ("lbfgs", "l-bfgs", "l_bfgs"):
            from famex.optimizers.ase_wrappers import VerboseLBFGS

            return VerboseLBFGS
        if name in ("bfgs",):
            from famex.optimizers.ase_wrappers import VerboseBFGS

            return VerboseBFGS
        if name in ("fire",):
            from famex.optimizers.ase_wrappers import VerboseFIRE

            return VerboseFIRE
    except Exception as e:  # pragma: no cover - ASE optional in some envs
        msg = f"Requested optimizer '{name}' is not available: {e}"
        raise ImportError(msg)

    msg = f"Unknown optimizer name: {name}"
    raise ValueError(msg)


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
    cleanup_frequencies: bool = False,
    stationary_point_target: Literal["minima", "ts"] = "minima",
    prepare_cleanup_optimizer_kwargs: Callable[[str, Any], dict[str, Any]] | None = None,
    record_history: bool = False,
) -> dict[str, Any]:
    """Execute common local optimization pattern for minima and TS."""
    atoms_list = list(atoms_list)
    strategy.validate_inputs(atoms_list)

    if strategy.profiler is not None:
        strategy.profiler.snapshot_memory()

    if verbose >= 1:
        logger.info(
            "Starting local %s optimization with %s", log_prefix.lower(), local_optimizer_name
        )
        logger.info("Force threshold: %s eV/Å, Max steps: %s", fmax, steps)

    opt_class = _get_local_optimizer_class(local_optimizer_name)
    single_input = len(atoms_list) == 1

    results = []
    step_counts = []
    converged_flags = []
    optimization_trajectory: list[Atoms] = []

    for image_index, atoms in enumerate(atoms_list):
        atoms_copy = atoms.copy()

        with (
            strategy.profiler.profile_section("calculator_setup")
            if strategy.profiler
            else nullcontext()
        ):
            needs_hessian = strategy.explorer._needs_hessian(
                calculate_frequencies=calculate_frequencies,
                cleanup_frequencies=cleanup_frequencies,
            )
            strategy.explorer._create_and_attach_calculator(
                atoms_copy,
                needs_hessian=needs_hessian,
            )
            strategy.explorer._apply_constraints(atoms_copy)

        opt_kwargs = prepare_optimizer_kwargs(local_optimizer_name, strategy.explorer)
        opt_kwargs.setdefault("verbose", getattr(strategy.explorer, "verbose", 1))
        if strategy.profiler is not None:
            opt_kwargs["profiler"] = strategy.profiler

        opt = opt_class(atoms_copy, **opt_kwargs)
        if record_history:
            image_tag = image_index if len(atoms_list) > 1 else None
            attach_step_recorder(opt, atoms_copy, optimization_trajectory, image=image_tag)

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
                "%s optimization completed: converged=%s, steps=%s",
                log_prefix,
                converged,
                steps_taken,
            )
        else:
            total_converged = sum(converged_flags)
            total_structures = len(converged_flags)
            logger.info(
                "%s optimization completed: %s/%s structures converged",
                log_prefix,
                total_converged,
                total_structures,
            )

    validation_results: list[dict[str, Any] | None] = []
    if validation_hook is not None:
        with (
            strategy.profiler.profile_section("validation") if strategy.profiler else nullcontext()
        ):
            validation_results = validation_hook(results)
    else:
        validation_results = [None] * len(results)

    frequency_results: list[dict[str, Any] | None] = []
    cleanup_summaries: list[dict[str, Any] | None] = []

    if cleanup_frequencies:
        if prepare_cleanup_optimizer_kwargs is None:
            msg = "prepare_cleanup_optimizer_kwargs is required when cleanup_frequencies=True"
            raise ValueError(msg)

        from famex.strategies.frequency_cleanup import run_frequency_cleanup

        for idx, atoms_copy in enumerate(results):
            with (
                strategy.profiler.profile_section("frequency_cleanup")
                if strategy.profiler
                else nullcontext()
            ):
                atoms_copy, freq_result, cleanup_steps, cleanup_summary = run_frequency_cleanup(
                    atoms_copy,
                    strategy.explorer,
                    target=stationary_point_target,
                    fmax=fmax,
                    temperature=temperature,
                    verbose=verbose,
                    prepare_optimizer_kwargs=prepare_cleanup_optimizer_kwargs,
                    prepare_frequency_kwargs=prepare_frequency_kwargs,
                    profiler=strategy.profiler,
                )
            results[idx] = atoms_copy
            step_counts[idx] = (step_counts[idx] or 0) + cleanup_steps
            frequency_results.append(freq_result)
            cleanup_summaries.append(cleanup_summary)
    elif calculate_frequencies:
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
        cleanup_summaries = [None] * len(results)
    else:
        frequency_results = [None] * len(results)
        cleanup_summaries = [None] * len(results)

    recorded: dict[str, list[Atoms]] = {}
    if optimization_trajectory:
        recorded["optimization_trajectory"] = optimization_trajectory

    if single_input and results:
        result = strategy.prepare_result(
            results[0],
            steps_taken=step_counts[0],
            converged=bool(converged_flags[0]),
            **recorded,
        )
        if validation_results[0] is not None:
            result["ts_validation"] = validation_results[0]
        if frequency_results[0] is not None:
            result["frequency_analysis"] = frequency_results[0]
            result[result_key_name] = frequency_results[0].get(result_key_name)
            result["free_energy_correction"] = read_free_energy_correction(frequency_results[0])
        if cleanup_summaries[0] is not None:
            result["frequency_cleanup"] = cleanup_summaries[0]
    else:
        result = strategy.prepare_result(
            results,
            steps_taken=step_counts,
            converged=[bool(c) for c in converged_flags],
            **recorded,
        )
        if any(v is not None for v in validation_results):
            result["ts_validation"] = validation_results
        if any(f is not None for f in frequency_results):
            result["frequency_analysis"] = frequency_results
            result[result_key_name] = [
                f.get(result_key_name) if f is not None else None for f in frequency_results
            ]
            result["free_energy_correction"] = [
                read_free_energy_correction(f) for f in frequency_results
            ]
        if any(summary is not None for summary in cleanup_summaries):
            result["frequency_cleanup"] = cleanup_summaries

    merged_result = strategy._merge_profiler_results(result)
    return cast(dict[str, Any], merged_result)


def copy_labeled_frame(atoms: Atoms, **tags: int) -> Atoms:
    """Return a geometry copy annotated with integer XYZ-comment tags."""
    frame = cast(Atoms, atoms.copy())
    info = dict(frame.info or {})
    info.update(tags)
    frame.info = info
    return frame


def attach_step_recorder(
    optimizer: Any,
    atoms: Atoms,
    frames: list[Atoms],
    image: int | None = None,
) -> None:
    """Record a copy of ``atoms`` each time ``optimizer`` notifies observers.

    ASE wrappers notify on the wrapped optimizer, so the observer is attached
    there when that object exists.
    """
    target = getattr(optimizer, "wrapped_optimizer", optimizer)
    attach = getattr(target, "attach", None)
    if not callable(attach):
        logger.warning(
            "Optimizer %s cannot record geometry history",
            type(target).__name__,
        )
        return

    counter = {"step": 0}

    def _record() -> None:
        tags = {"step": counter["step"]}
        if image is not None:
            tags["image"] = image
        frames.append(copy_labeled_frame(atoms, **tags))
        counter["step"] += 1

    attach(_record)


def take_optimization_trajectory(result: dict[str, Any], source: dict[str, Any]) -> None:
    """Copy a recorded local-optimizer trajectory onto ``result`` when present."""
    frames = source.get("optimization_trajectory")
    if isinstance(frames, list) and frames:
        result["optimization_trajectory"] = frames


def filter_interpolation_kwargs(
    kwargs: dict[str, Any], allowed_keys: set[str] | None = None
) -> dict[str, Any]:
    if allowed_keys is None:
        allowed_keys = {"rmsd_threshold", "energy_threshold", "calculator"}
    return {k: v for k, v in kwargs.items() if k in allowed_keys}
