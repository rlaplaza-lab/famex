"""FAMEX Command Line Interface.

This module provides the main CLI interface for FAMEX (Fast Mechanistic Explorer),
allowing users to perform molecular optimization tasks from the command line.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import click
from ase import Atoms

from famex.backends.constants import UMA_TASKS
from famex.cli.cache_commands import cache
from famex.cli.cli_helpers import (
    input_stem,
    load_path_structures,
    load_structure_or_smiles,
    parse_kv_pairs,
    print_frequency_summary,
    require_same_atom_count,
    save_results_json,
)
from famex.core.explorer import Explorer
from famex.io.modes import modes_output_dir, write_vibrational_modes
from famex.io.run_output import select_default_frames, write_run_outputs
from famex.utils.ml_warnings import quiet_backend_loading

os.environ.setdefault("MPLBACKEND", "Agg")


def _validate_temperature(ctx: Any, param: Any, value: float) -> float:
    if value is not None and value <= 0:
        msg = f"Temperature must be positive, got {value} K"
        raise click.BadParameter(msg)
    return value


def _handle_frequency_results(
    results: dict[str, Any],
    output_path: str,
    target: str,
    calculate_frequencies: bool,
    *,
    write_modes: bool = False,
    modes_which: str = "all",
    mode_indices: list[int] | None = None,
    mode_frames: int = 20,
    mode_amplitude: float = 0.3,
) -> None:
    if not calculate_frequencies:
        return

    if "frequency_analysis" not in results:
        click.echo(
            "Warning: Frequency analysis was requested (--freq) but not found in results. "
            "This may indicate:\n"
            "  - The optimization failed before frequency calculation\n"
            "  - Frequency calculation encountered an error\n"
            "  - The strategy does not support frequency analysis",
            err=True,
        )
        return

    freq_analysis = results["frequency_analysis"]
    if isinstance(freq_analysis, dict):
        try:
            print_frequency_summary(freq_analysis, target=target)
            save_results_json(results, output_path)
        except KeyError as e:
            click.echo(
                f"Error: Frequency analysis result is missing required key: {e}. "
                "The frequency calculation may have been incomplete.",
                err=True,
            )
        except Exception as e:
            click.echo(
                f"Error: Failed to process frequency analysis results: {e}\n"
                "The optimization completed successfully, but frequency analysis output "
                "could not be processed. Check the error message above for details.",
                err=True,
            )
    elif not (write_modes and isinstance(freq_analysis, list)):
        click.echo(
            f"Warning: Frequency analysis result has unexpected type {type(freq_analysis)}. "
            "Expected a dictionary.",
            err=True,
        )
        return

    if write_modes:
        _write_vibrational_mode_files(
            results,
            output_path,
            which=modes_which,
            mode_indices=mode_indices,
            nframes=mode_frames,
            amplitude=mode_amplitude,
        )


def _prepare_mode_output(
    write_modes: bool,
    calculate_frequencies: bool,
    mode_frames: int,
    mode_amplitude: float,
    mode_indices: tuple[int, ...],
) -> tuple[bool, list[int] | None]:
    """Turn frequency analysis on when mode animations were requested."""
    if not write_modes:
        return calculate_frequencies, None
    if mode_frames < 2:
        raise click.BadParameter("--mode-frames must be at least 2")
    if mode_amplitude <= 0:
        raise click.BadParameter("--mode-amplitude must be positive")
    return True, list(mode_indices) or None


def _write_vibrational_mode_files(
    results: dict[str, Any],
    output_path: str,
    *,
    which: str,
    mode_indices: list[int] | None,
    nframes: int,
    amplitude: float,
) -> None:
    optimized = results["optimized_atoms"]
    structures = [optimized] if isinstance(optimized, Atoms) else list(optimized)
    freq_analysis = results["frequency_analysis"]
    output_dir = modes_output_dir(output_path)

    if isinstance(freq_analysis, dict):
        if len(structures) != 1:
            raise click.ClickException(
                "Cannot write vibrational modes: expected one optimized structure."
            )
        jobs = [(structures[0], freq_analysis, "")]
    else:
        if len(structures) != len(freq_analysis):
            raise click.ClickException(
                "Cannot write vibrational modes: frequency analyses do not match "
                "the optimized structures."
            )
        jobs = []
        for image_index, (atoms, item) in enumerate(zip(structures, freq_analysis, strict=True)):
            if not isinstance(item, dict):
                click.echo(
                    f"Warning: Skipping vibrational modes for image {image_index}; "
                    "frequency analysis is missing.",
                    err=True,
                )
                continue
            jobs.append((atoms, item, f"image{image_index}_"))

    if not jobs:
        raise click.ClickException(
            "Cannot write vibrational modes: no frequency analysis was available."
        )

    for atoms, freq_dict, prefix in jobs:
        try:
            written = write_vibrational_modes(
                atoms,
                freq_dict,
                output_dir,
                which=which,
                mode_indices=mode_indices,
                amplitude=amplitude,
                nframes=nframes,
                filename_prefix=prefix,
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        for path in written:
            click.echo(f"Saved vibrational mode: {path}")


def _validate_strategy_requirements(
    strategy: str,
    target: str,
    product: str | None,
    structures_count: int = 1,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if target != "path" and strategy in ["interpolate", "growing_string", "cineb", "dhs"]:
        if product is None:
            errors.append(f"--product is required for {strategy} strategy")

    # Strategies that don't use product file
    if strategy == "local" and product is not None:
        warnings.append("--product ignored for local strategy")

    # Path-specific validations
    if target == "path":
        if strategy in ["interpolate", "neb", "cineb"]:
            if structures_count < 2:
                errors.append(
                    f"{strategy} strategy requires at least 2 structures, got {structures_count}"
                )
        elif strategy == "irc":
            if structures_count > 1:
                warnings.append(
                    f"IRC strategy uses only the first structure, "
                    f"ignoring {structures_count - 1} additional structure(s)"
                )

    return errors, warnings


def _create_explorer(
    atoms: Any,
    target: str,
    strategy: str,
    backend: str,
    model_name: str | None,
    model_path: str | None,
    device: str | None,
    default_charge: int,
    default_spin: int,
    local_optimizer: str,
    optimizer_kwargs: dict[str, Any],
    ts_kwargs: dict[str, Any],
    constraints: str | None,
    verbose: int,
    force_finite_diff_hessian: bool,
    cleanup_frequencies: bool,
    task_name: str | None = None,
    dftd3_functional: str | None = None,
) -> tuple[Explorer, AbstractContextManager[list[str]]]:
    ctx: AbstractContextManager[list[str]]
    if verbose == 0:
        ctx = quiet_backend_loading(backend, model_name, model_path, device, show_model_info=True)
    else:
        ctx = nullcontext([])

    atoms_for_explorer = atoms[0] if isinstance(atoms, list) and len(atoms) == 1 else atoms

    exp = Explorer(
        atoms=atoms_for_explorer,
        backend=backend,
        model_name=model_name,
        model_path=model_path,
        device=device,
        default_charge=default_charge,
        default_spin=default_spin,
        local_optimizer=local_optimizer,
        optimizer_kwargs=optimizer_kwargs,
        target=target,
        strategy=strategy,
        ts_kwargs=ts_kwargs,
        constraints=constraints,
        verbose=verbose,
        force_finite_diff_hessian=force_finite_diff_hessian,
        cleanup_frequencies=cleanup_frequencies,
        task_name=task_name,
        dftd3_functional=dftd3_functional,
    )

    return exp, ctx


def _handle_dry_run(exp: Explorer) -> None:
    """Handle dry-run mode by printing analysis and exiting.

    Parameters
    ----------
    exp : Explorer
        Explorer instance
    """
    explanation = exp.explain_run()
    click.echo("🔍 Dry-run analysis:")
    click.echo(f"   Target: {explanation['target']}")
    click.echo(f"   Strategy: {explanation['strategy']}")
    click.echo(f"   Runner: {explanation['runner']}")
    click.echo(f"   Valid: {explanation['valid']}")
    click.echo(f"   Notes: {explanation['notes']}")


def _save_run_output(
    results: dict[str, Any],
    input_path: str,
    strategy: str,
    target: str,
    output: str | None,
    history: bool,
    completed_message: str,
) -> str:
    """Write the target's default file and, when asked, geometry history."""
    stem = input_stem(input_path)
    written = write_run_outputs(
        results,
        stem,
        target=target,
        strategy=strategy,
        output=output,
        history=history,
    )
    default_path = written[0]
    frames = select_default_frames(results, target)
    if isinstance(frames, list):
        click.echo(f"{completed_message} Saved {len(frames)} images to: {default_path}")
    else:
        click.echo(f"{completed_message} Saved: {default_path}")
    for extra in written[1:]:
        click.echo(f"Saved history: {extra}")
    return default_path


def _run_optimization(
    exp: Explorer,
    strategy: str,
    target: str,
    fmax: float,
    steps: int | None,
    calculate_frequencies: bool,
    temperature: float,
    npoints: int | None = None,
    interp: str | None = None,
    max_images: int | None = None,
    distance_threshold: float | None = None,
    step_size: float | None = None,
    spring_constant: float | None = None,
    direction: str | None = None,
    climb: bool = False,
    cleanup_frequencies: bool = False,
    dhs_small_step: float | None = None,
    dhs_switch_thresh: float | None = None,
    record_history: bool = False,
) -> dict[str, Any]:
    run_kwargs: dict[str, Any] = {
        "calculate_frequencies": calculate_frequencies,
        "cleanup_frequencies": cleanup_frequencies,
        "temperature": temperature,
        "record_history": record_history,
    }

    if target == "path":
        if strategy == "interpolate":
            run_kwargs.update(
                {"npoints": npoints, "method": interp.lower() if interp else "geodesic"}
            )
        elif strategy == "neb":
            run_kwargs.update(
                {
                    "npoints": npoints,
                    "method": interp.lower() if interp else "geodesic",
                    "fmax": fmax,
                    "steps": steps,
                    "spring_constant": spring_constant,
                }
            )
        elif strategy == "cineb":
            run_kwargs.update(
                {
                    "npoints": npoints,
                    "method": interp.lower() if interp else "geodesic",
                    "fmax": fmax,
                    "steps": steps,
                    "spring_constant": spring_constant,
                    "climb": climb,
                }
            )
        elif strategy == "irc":
            run_kwargs.update(
                {
                    "fmax": fmax,
                    "steps": steps,
                    "step_size": step_size,
                    "direction": direction.lower() if direction else "both",
                }
            )
    elif target == "ts":
        if strategy == "local":
            run_kwargs.update({"fmax": fmax, "steps": steps})
        elif strategy == "interpolate":
            run_kwargs.update(
                {
                    "npoints": npoints,
                    "method": interp.lower() if interp else "idpp",
                    "fmax": fmax,
                    "steps": steps,
                }
            )
        elif strategy == "growing_string":
            run_kwargs.update(
                {
                    "npoints": npoints,
                    "max_images": max_images,
                    "distance_threshold": distance_threshold,
                    "step_size": step_size,
                    "fmax": fmax,
                    "steps": steps,
                }
            )
        elif strategy == "cineb":
            run_kwargs.update(
                {
                    "npoints": npoints,
                    "method": interp.lower() if interp else "idpp",
                    "fmax": fmax,
                    "steps": steps,
                    "spring_constant": spring_constant,
                }
            )
        elif strategy == "dhs":
            run_kwargs.update(
                {
                    "fmax": fmax,
                    "steps": steps,
                    "dist_tol": distance_threshold if distance_threshold is not None else 0.5,
                    "large_step": step_size if step_size is not None else 0.2,
                    "small_step": dhs_small_step,
                    "switch_thresh": dhs_switch_thresh,
                }
            )
    else:  # minima
        if strategy == "local":
            run_kwargs.update({"fmax": fmax, "steps": steps})
        else:  # interpolate
            run_kwargs.update(
                {
                    "npoints": npoints,
                    "method": interp.lower() if interp else "geodesic",
                    "fmax": fmax,
                    "steps": steps,
                }
            )

    # Omit unset options so each strategy keeps its own default.
    return exp.run(**{key: value for key, value in run_kwargs.items() if value is not None})


def _common_explorer_options(*, include_freq: bool = True) -> Callable[[Any], Any]:
    def decorator(f: Any) -> Any:
        opts: list[Any] = [
            click.option(
                "--backend",
                default="uma",
                show_default=True,
                help="Backend: uma|aimnet2|mace|orb|so3lr|tblite|pet|pyscf|mock",
            ),
            click.option("--model-name", default=None, help="Model name for backend"),
            click.option("--model-path", default=None, help="Path to model file (if applicable)"),
            click.option("--device", default=None, help="Device: cpu|cuda"),
            click.option(
                "--task-name",
                default=None,
                help=(
                    f"UMA task head: {'|'.join(UMA_TASKS)} "
                    "(default: omol; ignored for non-UMA backends)"
                ),
            ),
            click.option(
                "--dftd3-functional",
                default=None,
                type=click.Choice(["r2scan", "pbe"], case_sensitive=False),
                help=(
                    "Optional FairChem DFT-D3(BJ) wrap for UMA "
                    "(e.g. r2scan with uma-s-1p2p1-ft-r2scan)"
                ),
            ),
            click.option(
                "--default-charge",
                type=int,
                default=0,
                show_default=True,
                help="Default molecular charge",
            ),
            click.option(
                "--default-spin",
                type=int,
                default=1,
                show_default=True,
                help="Default spin multiplicity (use 0 for non-omol UMA tasks)",
            ),
            click.option(
                "--local-optimizer",
                "local_optimizer",
                default="default",
                show_default=True,
                help=(
                    "Local optimizer: default|lbfgs|bfgs|fire|sella|sella-analytical|"
                    "trust-krylov|"
                    "trust-ncg|trust-exact|rfo|newton-cg (newton-cg is minima-only; "
                    "default=auto-select based on target)"
                ),
            ),
            click.option(
                "--optimizer-kw",
                multiple=True,
                help="Optimizer kwargs as key=value, repeatable",
            ),
            click.option(
                "--ts-kw",
                multiple=True,
                help="TS optimizer kwargs as key=value, repeatable",
            ),
            click.option(
                "--constraints",
                default=None,
                help=(
                    "Constraints spec string; e.g., "
                    "'fix 0,1; harmonic_bond 2,3 k=5.0; fixinternals_bond 4,5 value=1.25'"
                ),
            ),
            click.option(
                "--verbose",
                "-v",
                count=True,
                default=0,
                help="Verbosity: default quiet, -v normal, -vv or more debug",
            ),
            click.option(
                "--dry-run",
                is_flag=True,
                default=False,
                help="Validate inputs and show strategy selection without running",
            ),
        ]
        if include_freq:
            opts.extend(
                [
                    click.option(
                        "--freq",
                        "--frequencies",
                        "calculate_frequencies",
                        is_flag=True,
                        default=False,
                        help=(
                            "Perform frequency analysis after optimization "
                            "(includes thermodynamic properties)"
                        ),
                    ),
                    click.option(
                        "--temperature",
                        type=float,
                        default=298.15,
                        show_default=True,
                        callback=lambda ctx, param, value: _validate_temperature(ctx, param, value),
                        help="Temperature in Kelvin for thermodynamic calculations (must be > 0)",
                    ),
                ]
            )
        opts.append(
            click.option(
                "--force-finite-diff-hessian",
                "force_finite_diff_hessian",
                is_flag=True,
                default=False,
                help=(
                    "Force use of finite difference hessians for TS optimizers "
                    "and frequency calculations"
                ),
            )
        )
        # Cleanup is only meaningful for minima/ts (include_freq=True), not path.
        if include_freq:
            opts.extend(
                [
                    click.option(
                        "--cleanup-frequencies",
                        "cleanup_frequencies",
                        is_flag=True,
                        default=False,
                        help=(
                            "After optimization, run Hessian-based cleanup from the final "
                            "geometry until there are 0 (minima) or 1 (TS) relevant "
                            "imaginary frequencies"
                        ),
                    ),
                    click.option(
                        "--write-modes",
                        is_flag=True,
                        default=False,
                        help=(
                            "Write each vibrational mode as a multi-frame XYZ animation. "
                            "Implies --freq."
                        ),
                    ),
                    click.option(
                        "--modes",
                        "modes_which",
                        type=click.Choice(["all", "imaginary"]),
                        default="all",
                        show_default=True,
                        help=(
                            "Which modes to animate when --write-modes is set. "
                            "Ignored when --mode-index is given."
                        ),
                    ),
                    click.option(
                        "--mode-index",
                        "mode_indices",
                        multiple=True,
                        type=int,
                        help=("Vibrational mode index to write, repeatable. Overrides --modes."),
                    ),
                    click.option(
                        "--mode-frames",
                        type=int,
                        default=20,
                        show_default=True,
                        help="Frames per vibrational-mode animation.",
                    ),
                    click.option(
                        "--mode-amplitude",
                        type=float,
                        default=0.3,
                        show_default=True,
                        help="Largest atomic displacement in each mode animation, in angstroms.",
                    ),
                ]
            )
        for opt in reversed(opts):
            f = opt(f)
        return f

    return decorator


def _embedder_option(func: Any) -> Any:
    """Add ``--embedder`` for commands that accept SMILES."""
    return click.option(
        "--embedder",
        type=click.Choice(["pysmiles", "rdkit"]),
        default="pysmiles",
        show_default=True,
        help="SMILES 3D method: pysmiles (UFF distance geometry) or rdkit (ETKDGv3)",
    )(func)


@click.group(
    help='FAMEX CLI: Fast mechanistic explorer with ML potentials.\n\n\b\nCommands:\n  famex minima : Minima optimization (outputs single structure)\n  famex ts     : Transition state optimization (outputs single TS)\n  famex path   : Reaction path optimization (outputs trajectories)\n  famex cache  : Manage model cache\n\n\b\nExamples:\n  # Structure inputs: XYZ file, SMILES, or pubchem:NAME / cid: / numeric CID\n  famex minima --strategy local "CCO" --backend aimnet2\n  famex minima --strategy local pubchem:aspirin --backend aimnet2\n\n\b\n  # Minima optimization (outputs single structure)\n  famex minima --strategy local reactant.xyz --backend aimnet2 --fmax 0.03  # Local optimization\n  famex minima --strategy interpolate r.xyz --product p.xyz --interp geodesic --npoints 21  # Via interpolation\n\n\b\n  # Transition state optimization (outputs single TS)\n  famex ts --strategy local ts_guess.xyz --ts-kw order=1  # Local TS optimization\n  famex ts --strategy interpolate r.xyz --product p.xyz --npoints 15  # TS via interpolation\n  famex ts --strategy growing_string r.xyz --product p.xyz --npoints 20 --step-size 0.1  # Growing string method\n  famex ts --strategy local ts_guess.xyz --local-optimizer rfo --fmax 0.02  # RFO TS optimizer\n\n\b\n  # Reaction path optimization (outputs trajectories)\n  famex path --strategy interpolate r.xyz p.xyz --npoints 15  # Raw interpolation\n  famex path --strategy neb r.xyz p.xyz --npoints 11 --spring-constant 5.0  # NEB path\n  famex path --strategy cineb r.xyz p.xyz --npoints 11 --spring-constant 5.0  # CI-NEB path\n  famex path --strategy neb r.xyz intermediate.xyz p.xyz --npoints 11  # Multiple structures\n  famex path --strategy irc ts.xyz --direction both --steps 100  # IRC from transition state\n\n\b\n  # Advanced backends\n  famex minima --strategy local molecule.xyz --backend mace --model-name mace-omol-0 --device cuda\n\n\b\n  # Cache management\n  famex cache info  # Show cache information\n  famex cache clear # Clear model cache',
)
@click.version_option()
def main() -> None:
    """Provide main CLI entry point."""


@main.command()
@click.option(
    "--strategy",
    type=click.Choice(["local", "interpolate"]),
    default="local",
    show_default=True,
    help="Optimization strategy",
)
@click.argument("input", metavar="INPUT")
@click.option(
    "--product",
    default=None,
    help="Product structure or SMILES (required for interpolate strategy)",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False),
    default=None,
    help="Output optimized XYZ path",
)
@click.option(
    "--history",
    is_flag=True,
    default=False,
    help="Also write the optimizer step history",
)
@click.option("--fmax", type=float, default=0.05, show_default=True, help="Convergence threshold")
@click.option("--steps", type=int, default=1000, show_default=True, help="Max optimization steps")
@click.option(
    "--npoints",
    type=int,
    default=11,
    show_default=True,
    help="Number of interpolation points (interpolate strategy only)",
)
@click.option(
    "--interp",
    type=click.Choice(["linear", "geodesic", "idpp"], case_sensitive=False),
    default="geodesic",
    show_default=True,
    help="Interpolation method (interpolate strategy only)",
)
@_embedder_option
@_common_explorer_options()
def minima(
    strategy: str,
    input: str,
    product: str | None,
    embedder: str,
    output: str | None,
    history: bool,
    fmax: float,
    steps: int,
    npoints: int,
    interp: str,
    backend: str,
    model_name: str | None,
    model_path: str | None,
    device: str | None,
    task_name: str | None,
    dftd3_functional: str | None,
    default_charge: int,
    default_spin: int,
    local_optimizer: str,
    optimizer_kw: list[str],
    ts_kw: list[str],
    constraints: str | None,
    verbose: int,
    dry_run: bool,
    calculate_frequencies: bool,
    temperature: float,
    force_finite_diff_hessian: bool,
    cleanup_frequencies: bool,
    write_modes: bool,
    modes_which: str,
    mode_indices: tuple[int, ...],
    mode_frames: int,
    mode_amplitude: float,
) -> None:
    """Minima optimization using various strategies."""
    calculate_frequencies, selected_modes = _prepare_mode_output(
        write_modes,
        calculate_frequencies,
        mode_frames,
        mode_amplitude,
        mode_indices,
    )
    # Validate strategy-specific requirements
    errors, warnings = _validate_strategy_requirements(
        strategy=strategy, target="minima", product=product, structures_count=1
    )
    for error in errors:
        raise click.BadParameter(error)
    for warning in warnings:
        click.echo(f"Warning: {warning}")

    # Load structures (existing files, or SMILES when the path does not exist)
    atoms = load_structure_or_smiles(input, embedder=embedder)
    atoms_list = [atoms]

    if strategy == "interpolate":
        if product is None:
            raise ValueError("Product file is required for interpolate strategy")
        atoms_product = load_structure_or_smiles(product, embedder=embedder)
        atoms_list = [atoms, atoms_product]
        try:
            require_same_atom_count(atoms_list, ["reactant", "product"])
        except ValueError as exc:
            raise click.BadParameter(str(exc)) from exc

    # Parse kwargs
    optimizer_kwargs = parse_kv_pairs(list(optimizer_kw))
    ts_kwargs = parse_kv_pairs(list(ts_kw))

    # Create explorer
    exp, ctx = _create_explorer(
        atoms=atoms_list,
        target="minima",
        strategy=strategy,
        backend=backend,
        model_name=model_name,
        model_path=model_path,
        device=device,
        default_charge=default_charge,
        default_spin=default_spin,
        local_optimizer=local_optimizer,
        optimizer_kwargs=optimizer_kwargs,
        ts_kwargs=ts_kwargs,
        constraints=constraints,
        verbose=verbose,
        force_finite_diff_hessian=force_finite_diff_hessian,
        cleanup_frequencies=cleanup_frequencies,
        task_name=task_name,
        dftd3_functional=dftd3_functional,
    )

    with ctx:
        if dry_run:
            _handle_dry_run(exp)
            return

        # Run optimization
        results = _run_optimization(
            exp=exp,
            strategy=strategy,
            target="minima",
            fmax=fmax,
            steps=steps,
            calculate_frequencies=calculate_frequencies,
            cleanup_frequencies=cleanup_frequencies,
            temperature=temperature,
            npoints=npoints if strategy == "interpolate" else None,
            interp=interp if strategy == "interpolate" else None,
            record_history=history,
        )

    out = _save_run_output(
        results,
        input,
        strategy,
        "minima",
        output,
        history,
        "Minima optimization completed.",
    )

    # Handle frequency analysis results (chained after optimization)
    _handle_frequency_results(
        results,
        out,
        target="minima",
        calculate_frequencies=calculate_frequencies,
        write_modes=write_modes,
        modes_which=modes_which,
        mode_indices=selected_modes,
        mode_frames=mode_frames,
        mode_amplitude=mode_amplitude,
    )


@main.command()
@click.option(
    "--strategy",
    type=click.Choice(["local", "interpolate", "cineb", "growing_string", "dhs"]),
    default="local",
    show_default=True,
    help="Optimization strategy",
)
@click.argument("input", metavar="INPUT")
@click.option(
    "--product",
    default=None,
    help="Product structure or SMILES (required for interpolate/cineb/growing_string/dhs)",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False),
    default=None,
    help="Output TS XYZ path",
)
@click.option(
    "--history",
    is_flag=True,
    default=False,
    help="Also write the search path and the TS optimizer history",
)
@click.option("--fmax", type=float, default=0.05, show_default=True, help="Convergence threshold")
@click.option(
    "--steps",
    type=int,
    default=None,
    show_default="1000 (local, interpolate, cineb); 200 (growing_string, dhs)",
    help="Max optimization steps. Omit to use the strategy default.",
)
@click.option(
    "--npoints",
    type=int,
    default=None,
    show_default="11 (interpolate, cineb); 15 (growing_string)",
    help="Interpolation images, or growing-string image count when --max-images is omitted.",
)
@click.option(
    "--interp",
    type=click.Choice(["linear", "geodesic", "idpp"], case_sensitive=False),
    default="idpp",
    show_default=True,
    help="Interpolation method (interpolate/cineb strategies only)",
)
@click.option(
    "--spring-constant",
    type=float,
    default=5.0,
    show_default=True,
    help="Spring constant for CI-NEB (cineb strategy only)",
)
@click.option(
    "--max-images",
    type=int,
    default=None,
    show_default="npoints, or 15 when both are omitted",
    help="Growing-string image count. Overrides --npoints when set.",
)
@click.option(
    "--distance-threshold",
    type=float,
    default=None,
    show_default=False,
    help="Distance threshold: growing_string default 0.1; DHS dist_tol default 0.5",
)
@click.option(
    "--step-size",
    type=float,
    default=None,
    show_default=False,
    help="Step size: growing_string default 0.1; DHS large_step default 0.2",
)
@click.option(
    "--dhs-small-step",
    type=float,
    default=0.05,
    show_default=True,
    help="DHS small step size when images are close (dhs strategy only)",
)
@click.option(
    "--dhs-switch-thresh",
    type=float,
    default=1.5,
    show_default=True,
    help="DHS distance below which small_step is used (dhs strategy only)",
)
@click.option(
    "--require-ts/--allow-ts",
    "require_ts",
    default=False,
    show_default=True,
    help="If set, fail when the TS strategy does not yield a validated first-order saddle.",
)
@_embedder_option
@_common_explorer_options()
def ts(
    strategy: str,
    input: str,
    product: str | None,
    embedder: str,
    output: str | None,
    history: bool,
    fmax: float,
    steps: int | None,
    npoints: int | None,
    interp: str,
    spring_constant: float,
    max_images: int | None,
    distance_threshold: float | None,
    step_size: float | None,
    dhs_small_step: float,
    dhs_switch_thresh: float,
    require_ts: bool,
    backend: str,
    model_name: str | None,
    model_path: str | None,
    device: str | None,
    task_name: str | None,
    dftd3_functional: str | None,
    default_charge: int,
    default_spin: int,
    local_optimizer: str,
    optimizer_kw: list[str],
    ts_kw: list[str],
    constraints: str | None,
    verbose: int,
    dry_run: bool,
    calculate_frequencies: bool,
    temperature: float,
    force_finite_diff_hessian: bool,
    cleanup_frequencies: bool,
    write_modes: bool,
    modes_which: str,
    mode_indices: tuple[int, ...],
    mode_frames: int,
    mode_amplitude: float,
) -> None:
    """Transition state optimization using various strategies."""
    calculate_frequencies, selected_modes = _prepare_mode_output(
        write_modes,
        calculate_frequencies,
        mode_frames,
        mode_amplitude,
        mode_indices,
    )
    # Validate strategy-specific requirements
    errors, warnings = _validate_strategy_requirements(
        strategy=strategy, target="ts", product=product, structures_count=1
    )
    for error in errors:
        raise click.BadParameter(error)
    for warning in warnings:
        click.echo(f"Warning: {warning}")

    # Load structures (existing files, or SMILES when the path does not exist)
    atoms = load_structure_or_smiles(input, embedder=embedder)
    atoms_list = [atoms]

    if strategy in ["interpolate", "growing_string", "cineb", "dhs"]:
        if product is None:
            raise ValueError(
                f"Product file is required for {strategy} strategy",
            )
        atoms_product = load_structure_or_smiles(product, embedder=embedder)
        atoms_list = [atoms, atoms_product]
        try:
            require_same_atom_count(atoms_list, ["reactant", "product"])
        except ValueError as exc:
            raise click.BadParameter(str(exc)) from exc

    # Parse kwargs
    optimizer_kwargs = parse_kv_pairs(list(optimizer_kw))
    ts_kwargs = parse_kv_pairs(list(ts_kw))
    if require_ts:
        ts_kwargs["require_ts"] = True

    # Create explorer
    exp, ctx = _create_explorer(
        atoms=atoms_list,
        target="ts",
        strategy=strategy,
        backend=backend,
        model_name=model_name,
        model_path=model_path,
        device=device,
        default_charge=default_charge,
        default_spin=default_spin,
        local_optimizer=local_optimizer,
        optimizer_kwargs=optimizer_kwargs,
        ts_kwargs=ts_kwargs,
        constraints=constraints,
        verbose=verbose,
        force_finite_diff_hessian=force_finite_diff_hessian,
        cleanup_frequencies=cleanup_frequencies,
        task_name=task_name,
        dftd3_functional=dftd3_functional,
    )

    with ctx:
        if dry_run:
            _handle_dry_run(exp)
            return

        dist_default = 0.5 if strategy == "dhs" else 0.1
        step_default = 0.2 if strategy == "dhs" else 0.1
        results = _run_optimization(
            exp=exp,
            strategy=strategy,
            target="ts",
            fmax=fmax,
            steps=steps,
            calculate_frequencies=calculate_frequencies,
            cleanup_frequencies=cleanup_frequencies,
            temperature=temperature,
            npoints=npoints if strategy in ["interpolate", "growing_string", "cineb"] else None,
            interp=interp if strategy in ["interpolate", "cineb"] else None,
            spring_constant=spring_constant if strategy == "cineb" else None,
            max_images=max_images if strategy == "growing_string" else None,
            distance_threshold=(
                (distance_threshold if distance_threshold is not None else dist_default)
                if strategy in ("growing_string", "dhs")
                else None
            ),
            step_size=(
                (step_size if step_size is not None else step_default)
                if strategy in ("growing_string", "dhs")
                else None
            ),
            dhs_small_step=dhs_small_step if strategy == "dhs" else None,
            dhs_switch_thresh=dhs_switch_thresh if strategy == "dhs" else None,
            record_history=history,
        )

    out = _save_run_output(
        results,
        input,
        strategy,
        "ts",
        output,
        history,
        "Transition state optimization completed.",
    )

    # Handle frequency analysis results (chained after optimization)
    _handle_frequency_results(
        results,
        out,
        target="ts",
        calculate_frequencies=calculate_frequencies,
        write_modes=write_modes,
        modes_which=modes_which,
        mode_indices=selected_modes,
        mode_frames=mode_frames,
        mode_amplitude=mode_amplitude,
    )


main.add_command(cache)


@main.command()
@click.option(
    "--strategy",
    type=click.Choice(["interpolate", "neb", "cineb", "irc"]),
    default="neb",
    show_default=True,
    help="Path optimization strategy",
)
@click.argument("structures", nargs=-1, required=True)
@_embedder_option
@click.option(
    "--output",
    type=click.Path(dir_okay=False),
    default=None,
    help="Output trajectory XYZ path",
)
@click.option(
    "--history",
    is_flag=True,
    default=False,
    help="Also write the NEB iteration history or IRC forward and reverse branches",
)
@click.option("--fmax", type=float, default=0.05, show_default=True, help="Convergence threshold")
@click.option(
    "--steps",
    type=int,
    default=None,
    show_default="1000 (neb, cineb); 100 (irc)",
    help="Max optimization steps. Omit to use the strategy default.",
)
@click.option(
    "--npoints",
    type=int,
    default=11,
    show_default=True,
    help="Number of images in path",
)
@click.option(
    "--interp",
    type=click.Choice(["linear", "geodesic", "idpp"], case_sensitive=False),
    default="geodesic",
    show_default=True,
    help="Initial interpolation method",
)
@click.option(
    "--spring-constant",
    type=float,
    default=5.0,
    show_default=True,
    help="Spring constant for NEB/CI-NEB",
)
@click.option(
    "--step-size",
    type=float,
    default=0.1,
    show_default=True,
    help="IRC step size (amu^1/2 * Angstrom) (IRC strategy only)",
)
@click.option(
    "--direction",
    type=click.Choice(["forward", "backward", "both"], case_sensitive=False),
    default="both",
    show_default=True,
    help="Direction to follow from TS (IRC strategy only)",
)
@_common_explorer_options(include_freq=False)
def path(
    strategy: str,
    structures: tuple[str, ...],
    output: str | None,
    history: bool,
    fmax: float,
    steps: int | None,
    npoints: int,
    interp: str,
    spring_constant: float,
    step_size: float,
    direction: str,
    embedder: str,
    backend: str,
    model_name: str | None,
    model_path: str | None,
    device: str | None,
    task_name: str | None,
    dftd3_functional: str | None,
    default_charge: int,
    default_spin: int,
    local_optimizer: str,
    optimizer_kw: list[str],
    ts_kw: list[str],
    constraints: str | None,
    verbose: int,
    dry_run: bool,
    force_finite_diff_hessian: bool,
) -> None:
    """Reaction path optimization using various strategies.

    Strategies:
    - interpolate: Raw geodesic interpolation (no optimization)
    - neb: Nudged Elastic Band path optimization
    - cineb: Climbing Image NEB path optimization
    - irc: IRC path from transition state

    Input structures can be provided as:
    - Files, SMILES, ``cid:``, or ``pubchem:`` names, including a mix
    - Multiple endpoints: famex path reactant.xyz "CCO" cid:702
    - Single multi-frame XYZ: famex path path_guess.xyz (all frames used)
    - Single single-frame XYZ: famex path ts.xyz (for IRC strategy)

    Endpoints must have the same number of atoms.

    For interpolate, neb, and cineb: provide at least 2 structures.
    For irc: provide 1 structure (transition state).
    """
    # Load structures from variadic inputs (files, SMILES, or PubChem)
    try:
        atoms_list = load_path_structures(structures, embedder=embedder)
    except click.ClickException:
        raise
    except (ValueError, RuntimeError) as exc:
        raise click.BadParameter(str(exc)) from exc

    # Validate strategy-specific requirements
    errors, warnings = _validate_strategy_requirements(
        strategy=strategy, target="path", product=None, structures_count=len(atoms_list)
    )
    for error in errors:
        raise click.BadParameter(error)
    for warning in warnings:
        click.echo(f"Warning: {warning}")
        # Handle IRC warning by using only first structure
        if strategy == "irc" and len(atoms_list) > 1:
            atoms_list = [atoms_list[0]]

    if len(structures) == len(atoms_list):
        labels = list(structures)
    else:
        labels = [f"{structures[0]} frame {index}" for index in range(len(atoms_list))]
    try:
        require_same_atom_count(atoms_list, labels)
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc

    # Parse kwargs
    optimizer_kwargs = parse_kv_pairs(list(optimizer_kw))
    ts_kwargs = parse_kv_pairs(list(ts_kw))

    # Create explorer
    exp, ctx = _create_explorer(
        atoms=atoms_list,
        target="path",
        strategy=strategy,
        backend=backend,
        model_name=model_name,
        model_path=model_path,
        device=device,
        default_charge=default_charge,
        default_spin=default_spin,
        local_optimizer=local_optimizer,
        optimizer_kwargs=optimizer_kwargs,
        ts_kwargs=ts_kwargs,
        constraints=constraints,
        verbose=verbose,
        force_finite_diff_hessian=force_finite_diff_hessian,
        cleanup_frequencies=False,
        task_name=task_name,
        dftd3_functional=dftd3_functional,
    )
    first_file = structures[0]

    with ctx:
        if dry_run:
            _handle_dry_run(exp)
            return

        result = _run_optimization(
            exp=exp,
            strategy=strategy,
            target="path",
            fmax=fmax,
            steps=steps,
            calculate_frequencies=False,
            temperature=298.15,
            npoints=npoints if strategy in ["interpolate", "neb", "cineb"] else None,
            interp=interp if strategy in ["interpolate", "neb", "cineb"] else None,
            spring_constant=spring_constant if strategy in ["neb", "cineb"] else None,
            step_size=step_size if strategy == "irc" else None,
            direction=direction if strategy == "irc" else None,
            climb=(strategy == "cineb"),
            record_history=history,
        )

    _save_run_output(
        result,
        first_file,
        strategy,
        "path",
        output,
        history,
        "Path optimization completed.",
    )


__all__ = ["main", "minima", "path", "ts"]


if __name__ == "__main__":
    main()
