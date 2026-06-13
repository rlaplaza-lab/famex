"""FAMEX Command Line Interface.

This module provides the main CLI interface for FAMEX (Fast Mechanistic Explorer),
allowing users to perform molecular optimization tasks from the command line.
"""

from __future__ import annotations

import os
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import click

from famex.cli.cache_commands import cache
from famex.cli.cli_helpers import (
    load_atoms_from_xyz,
    load_path_structures,
    parse_kv_pairs,
    print_frequency_summary,
    save_results_json,
    write_atoms,
)
from famex.core.explorer import Explorer
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
    if not isinstance(freq_analysis, dict):
        click.echo(
            f"Warning: Frequency analysis result has unexpected type {type(freq_analysis)}. "
            "Expected a dictionary.",
            err=True,
        )
        return

    # Type narrowing: freq_analysis is now dict[str, Any]
    freq_dict: dict[str, Any] = freq_analysis

    try:
        print_frequency_summary(freq_dict, target=target)
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
            "The optimization completed successfully, but frequency analysis output could not be processed. "
            "Check the error message above for details.",
            err=True,
        )


def _validate_strategy_requirements(
    strategy: str,
    target: str,
    product: str | None,
    structures_count: int = 1,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if target != "path" and strategy in ["interpolate", "growing_string"]:
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
    dry_run: bool,
) -> tuple[Explorer, AbstractContextManager[list[str]]]:
    verbosity = max(0, verbose - 1)

    ctx: AbstractContextManager[list[str]]
    if verbosity == 0:
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
        verbose=verbosity,
        force_finite_diff_hessian=force_finite_diff_hessian,
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


def _generate_output_path(input_path: str, strategy: str, target: str) -> str:
    """Generate default output path based on strategy and target.

    Parameters
    ----------
    input_path : str
        Input file path
    strategy : str
        Strategy name
    target : str
        Target type ("minima", "ts", or "path")

    Returns
    -------
    str
        Default output path
    """
    base = os.path.splitext(input_path)[0]
    if target == "minima":
        suffix = ".opt.local.xyz" if strategy == "local" else ".opt.interpolate.xyz"
    elif target == "ts":
        if strategy == "local":
            suffix = ".ts.local.xyz"
        elif strategy == "interpolate":
            suffix = ".ts.interpolate.xyz"
        else:  # growing_string
            suffix = ".ts.gsm.xyz"
    else:  # path
        if strategy == "interpolate":
            suffix = ".path.interpolate.xyz"
        elif strategy == "neb":
            suffix = ".path.neb.xyz"
        elif strategy == "cineb":
            suffix = ".path.cineb.xyz"
        else:  # irc
            suffix = ".path.irc.xyz"
    return base + suffix


def _extract_result_atoms(results: dict[str, Any], target: str) -> Any:
    if target == "path":
        if "trajectory" in results:
            return results["trajectory"]
        return results.get("optimized_atoms", [])
    else:
        return results["optimized_atoms"]


def _run_optimization(
    exp: Explorer,
    strategy: str,
    target: str,
    fmax: float,
    steps: int,
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
) -> dict[str, Any]:
    run_kwargs: dict[str, Any] = {
        "calculate_frequencies": calculate_frequencies,
        "temperature": temperature,
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
                    "method": interp.lower() if interp else "geodesic",
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

    return exp.run(**run_kwargs)


def _common_explorer_options(f: Any) -> Any:
    opts = [
        click.option(
            "--backend",
            default="uma",
            show_default=True,
            help="Backend: uma|aimnet2|mace|orb|so3lr|tblite|pet|mock",
        ),
        click.option("--model-name", default=None, help="Model name for backend"),
        click.option("--model-path", default=None, help="Path to model file (if applicable)"),
        click.option("--device", default=None, help="Device: cpu|cuda"),
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
            help="Default spin multiplicity",
        ),
        click.option(
            "--local-optimizer",
            "local_optimizer",
            default="default",
            show_default=True,
            help=(
                "Local optimizer: default|lbfgs|bfgs|fire|sella|trust-krylov|"
                "trust-ncg|trust-exact|newton-cg|rfo (default=auto-select based on target)"
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
            default=1,
            help="Verbosity level: -v=quiet, -vv=normal, -vvv=debug",
        ),
        click.option(
            "--dry-run",
            is_flag=True,
            default=False,
            help="Validate inputs and show strategy selection without running",
        ),
        click.option(
            "--freq",
            "--frequencies",
            "calculate_frequencies",
            is_flag=True,
            default=False,
            help="Perform frequency analysis after optimization (includes thermodynamic properties)",
        ),
        click.option(
            "--temperature",
            type=float,
            default=298.15,
            show_default=True,
            callback=lambda ctx, param, value: _validate_temperature(ctx, param, value),
            help="Temperature in Kelvin for thermodynamic calculations (must be > 0)",
        ),
        click.option(
            "--force-finite-diff-hessian",
            "force_finite_diff_hessian",
            is_flag=True,
            default=False,
            help="Force use of finite difference hessians for TS optimizers and frequency calculations",
        ),
    ]
    for opt in reversed(opts):
        f = opt(f)
    return f


@click.group(
    help="FAMEX CLI: Fast mechanistic explorer with ML potentials.\n\n\b\nCommands:\n  famex minima : Minima optimization (outputs single structure)\n  famex ts     : Transition state optimization (outputs single TS)\n  famex path   : Reaction path optimization (outputs trajectories)\n  famex cache  : Manage model cache\n\n\b\nExamples:\n  # Minima optimization (outputs single structure)\n  famex minima --strategy local reactant.xyz --backend aimnet2 --fmax 0.03  # Local optimization\n  famex minima --strategy interpolate r.xyz --product p.xyz --interp geodesic --npoints 21  # Via interpolation\n\n\b\n  # Transition state optimization (outputs single TS)\n  famex ts --strategy local ts_guess.xyz --ts-kw order=1  # Local TS optimization\n  famex ts --strategy interpolate r.xyz --product p.xyz --npoints 15  # TS via interpolation\n  famex ts --strategy growing_string r.xyz --product p.xyz --npoints 20 --step-size 0.1  # Growing string method\n  famex ts --strategy local ts_guess.xyz --local-optimizer rfo --fmax 0.02  # RFO TS optimizer\n\n\b\n  # Reaction path optimization (outputs trajectories)\n  famex path --strategy interpolate r.xyz p.xyz --npoints 15  # Raw interpolation\n  famex path --strategy neb r.xyz p.xyz --npoints 11 --spring-constant 5.0  # NEB path\n  famex path --strategy cineb r.xyz p.xyz --npoints 11 --spring-constant 5.0  # CI-NEB path\n  famex path --strategy neb r.xyz intermediate.xyz p.xyz --npoints 11  # Multiple structures\n  famex path --strategy irc ts.xyz --direction both --steps 100  # IRC from transition state\n\n\b\n  # Advanced backends\n  famex minima --strategy local molecule.xyz --backend torchsim_mace --model-name mace-omol-0 --device cuda\n\n\b\n  # Cache management\n  famex cache info  # Show cache information\n  famex cache clear # Clear model cache",
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
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--product",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Product structure (required for interpolate strategy)",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False),
    default=None,
    help="Output optimized XYZ path",
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
    type=click.Choice(["linear", "geodesic", "idpp", "quadratic", "spline"], case_sensitive=False),
    default="geodesic",
    show_default=True,
    help="Interpolation method (interpolate strategy only)",
)
@_common_explorer_options
def minima(
    strategy: str,
    input: str,
    product: str | None,
    output: str | None,
    fmax: float,
    steps: int,
    npoints: int,
    interp: str,
    backend: str,
    model_name: str | None,
    model_path: str | None,
    device: str | None,
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
) -> None:
    """Minima optimization using various strategies."""
    # Validate strategy-specific requirements
    errors, warnings = _validate_strategy_requirements(
        strategy=strategy, target="minima", product=product, structures_count=1
    )
    for error in errors:
        raise click.BadParameter(error)
    for warning in warnings:
        click.echo(f"Warning: {warning}")

    # Load structures
    atoms = load_atoms_from_xyz(input)
    atoms_list = [atoms]

    if strategy == "interpolate":
        if product is None:
            raise ValueError("Product file is required for interpolate strategy")
        atoms_product = load_atoms_from_xyz(product)
        atoms_list = [atoms, atoms_product]

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
        dry_run=dry_run,
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
            temperature=temperature,
            npoints=npoints if strategy == "interpolate" else None,
            interp=interp if strategy == "interpolate" else None,
        )

        # Extract results and generate output path
        result_atoms = _extract_result_atoms(results, target="minima")
        out_default = _generate_output_path(input, strategy, "minima")

    # Save results
    out = output or out_default
    write_atoms(result_atoms, out)  # type: ignore[arg-type]
    click.echo(f"Minima optimization completed. Saved: {out}")

    # Handle frequency analysis results (chained after optimization)
    _handle_frequency_results(
        results, out, target="minima", calculate_frequencies=calculate_frequencies
    )


@main.command()
@click.option(
    "--strategy",
    type=click.Choice(["local", "interpolate", "growing_string"]),
    default="local",
    show_default=True,
    help="Optimization strategy",
)
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--product",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Product structure (required for interpolate/growing_string strategies)",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False),
    default=None,
    help="Output TS XYZ path",
)
@click.option("--fmax", type=float, default=0.05, show_default=True, help="Convergence threshold")
@click.option("--steps", type=int, default=1000, show_default=True, help="Max optimization steps")
@click.option(
    "--npoints",
    type=int,
    default=11,
    show_default=True,
    help="Number of interpolation points (interpolate/growing_string strategies only)",
)
@click.option(
    "--interp",
    type=click.Choice(["linear", "geodesic", "idpp", "quadratic", "spline"], case_sensitive=False),
    default="geodesic",
    show_default=True,
    help="Interpolation method (interpolate strategy only)",
)
@click.option(
    "--max-images",
    type=int,
    default=100,
    show_default=True,
    help="Maximum number of images (growing_string strategy only)",
)
@click.option(
    "--distance-threshold",
    type=float,
    default=0.1,
    show_default=True,
    help="Distance threshold for convergence (growing_string strategy only)",
)
@click.option(
    "--step-size",
    type=float,
    default=0.1,
    show_default=True,
    help="Step size for growing string method (growing_string strategy only)",
)
@click.option(
    "--require-ts/--allow-ts",
    "require_ts",
    default=False,
    show_default=True,
    help="If set, fail when the TS strategy does not yield a validated first-order saddle.",
)
@_common_explorer_options
def ts(
    strategy: str,
    input: str,
    product: str | None,
    output: str | None,
    fmax: float,
    steps: int,
    npoints: int,
    interp: str,
    max_images: int,
    distance_threshold: float,
    step_size: float,
    require_ts: bool,
    backend: str,
    model_name: str | None,
    model_path: str | None,
    device: str | None,
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
) -> None:
    """Transition state optimization using various strategies."""
    # Validate strategy-specific requirements
    errors, warnings = _validate_strategy_requirements(
        strategy=strategy, target="ts", product=product, structures_count=1
    )
    for error in errors:
        raise click.BadParameter(error)
    for warning in warnings:
        click.echo(f"Warning: {warning}")

    # Load structures
    atoms = load_atoms_from_xyz(input)
    atoms_list = [atoms]

    if strategy in ["interpolate", "growing_string"]:
        if product is None:
            raise ValueError("Product file is required for interpolate/growing_string strategy")
        atoms_product = load_atoms_from_xyz(product)
        atoms_list = [atoms, atoms_product]

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
        dry_run=dry_run,
    )

    with ctx:
        if dry_run:
            _handle_dry_run(exp)
            return

        # Run optimization
        results = _run_optimization(
            exp=exp,
            strategy=strategy,
            target="ts",
            fmax=fmax,
            steps=steps,
            calculate_frequencies=calculate_frequencies,
            temperature=temperature,
            npoints=npoints if strategy in ["interpolate", "growing_string"] else None,
            interp=interp if strategy == "interpolate" else None,
            max_images=max_images if strategy == "growing_string" else None,
            distance_threshold=distance_threshold if strategy == "growing_string" else None,
            step_size=step_size if strategy == "growing_string" else None,
        )

        # Extract results and generate output path
        result_atoms = _extract_result_atoms(results, target="ts")
        out_default = _generate_output_path(input, strategy, "ts")

    # Save results
    out = output or out_default
    write_atoms(result_atoms, out)  # type: ignore[arg-type]
    click.echo(f"Transition state optimization completed. Saved: {out}")

    # Handle frequency analysis results (chained after optimization)
    _handle_frequency_results(
        results, out, target="ts", calculate_frequencies=calculate_frequencies
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
@click.argument("structures", nargs=-1, type=click.Path(exists=True, dir_okay=False), required=True)
@click.option(
    "--output",
    type=click.Path(dir_okay=False),
    default=None,
    help="Output trajectory XYZ path",
)
@click.option("--fmax", type=float, default=0.05, show_default=True, help="Convergence threshold")
@click.option("--steps", type=int, default=1000, show_default=True, help="Max optimization steps")
@click.option(
    "--npoints",
    type=int,
    default=11,
    show_default=True,
    help="Number of images in path",
)
@click.option(
    "--interp",
    type=click.Choice(["linear", "geodesic", "idpp", "quadratic", "spline"], case_sensitive=False),
    default="geodesic",
    show_default=True,
    help="Initial interpolation method",
)
@click.option(
    "--spring-constant",
    type=float,
    default=0.5,
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
@_common_explorer_options
def path(
    strategy: str,
    structures: tuple[str, ...],
    output: str | None,
    fmax: float,
    steps: int,
    npoints: int,
    interp: str,
    spring_constant: float,
    step_size: float,
    direction: str,
    backend: str,
    model_name: str | None,
    model_path: str | None,
    device: str | None,
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
) -> None:
    """Reaction path optimization using various strategies.

    Strategies:
    - interpolate: Raw geodesic interpolation (no optimization)
    - neb: Nudged Elastic Band path optimization
    - cineb: Climbing Image NEB path optimization
    - irc: IRC path from transition state

    Input structures can be provided as:
    - Multiple files: famex path reactant.xyz product.xyz [intermediate.xyz ...]
    - Single multi-frame XYZ: famex path path_guess.xyz (all frames used)
    - Single single-frame XYZ: famex path ts.xyz (for IRC strategy)

    For interpolate, neb, and cineb: provide at least 2 structures.
    For irc: provide 1 structure (transition state).
    """
    # Load structures from variadic inputs
    atoms_list = load_path_structures(structures)

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
        dry_run=dry_run,
    )

    # Use first structure file for default output naming
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
            calculate_frequencies=calculate_frequencies,
            temperature=temperature,
            npoints=npoints if strategy in ["interpolate", "neb", "cineb"] else None,
            interp=interp if strategy in ["interpolate", "neb", "cineb"] else None,
            spring_constant=spring_constant if strategy in ["neb", "cineb"] else None,
            step_size=step_size if strategy == "irc" else None,
            direction=direction if strategy == "irc" else None,
            climb=(strategy == "cineb"),
        )

        trajectory = _extract_result_atoms(result, target="path")
        out_default = _generate_output_path(first_file, strategy, "path")

    out = output or out_default
    write_atoms(trajectory, out)  # type: ignore[arg-type]
    click.echo(f"Path optimization completed. Saved {len(trajectory)} images to: {out}")  # type: ignore[arg-type]

    _handle_frequency_results(
        result, out, target="path", calculate_frequencies=calculate_frequencies
    )


__all__ = ["main", "minima", "path", "ts"]


if __name__ == "__main__":
    main()
