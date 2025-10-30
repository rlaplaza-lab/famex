"""QME Command Line Interface.

This module provides the main CLI interface for QME (Quick Mechanistic Exploration),
allowing users to perform molecular optimization tasks from the command line.
"""

import os
from contextlib import nullcontext
from typing import Any

import click

from qme.cli.cache_commands import cache
from qme.cli.cli_helpers import (
    load_atoms_from_xyz,
    parse_kv_pairs,
    print_frequency_summary,
    save_results_json,
    write_atoms,
)
from qme.core.explorer import Explorer
from qme.utils.ml_warnings import quiet_backend_loading

# Disable matplotlib and ASE GUI before importing ASE
# This prevents unwanted popup windows during CLI operations
os.environ.setdefault("MPLBACKEND", "Agg")


def _common_explorer_options(f: Any) -> Any:
    opts = [
        click.option(
            "--backend",
            default="uma",
            show_default=True,
            help="Backend: uma|aimnet2|mace|orb|so3lr|tblite|torchsim_mace|torchsim_uma|mock",
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
                "Local optimizer: default|lbfgs|bfgs|fire|sella|trust-krylov|trust-krylov-ts|"
                "trust-ncg|trust-exact|newton-cg (default=auto-select based on target)"
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
            help="Constraints spec string; e.g., 'fix 0,1; harmonic_bond 2,3 k=5.0'",
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
            help="Temperature in Kelvin for thermodynamic calculations",
        ),
    ]
    for opt in reversed(opts):
        f = opt(f)
    return f


@click.group(
    help="QME CLI: Quick mechanistic exploration with ML potentials.\n\n\b\nCommands:\n  qme minima : Minima optimization (outputs single structure)\n  qme ts     : Transition state optimization (outputs single TS)\n  qme path   : Reaction path optimization (outputs trajectories)\n  qme cache  : Manage model cache\n\n\b\nExamples:\n  # Minima optimization (outputs single structure)\n  qme minima --strategy local reactant.xyz --backend aimnet2 --fmax 0.03  # Local optimization\n  qme minima --strategy interpolate r.xyz --product p.xyz --interp geodesic --npoints 21  # Via interpolation\n\n\b\n  # Transition state optimization (outputs single TS)\n  qme ts --strategy local ts_guess.xyz --ts-kw order=1  # Local TS optimization\n  qme ts --strategy interpolate r.xyz --product p.xyz --npoints 15  # TS via interpolation\n  qme ts --strategy growing_string r.xyz --product p.xyz --npoints 20 --step-size 0.1  # Growing string method\n  qme ts --strategy local ts_guess.xyz --local-optimizer trust-krylov-ts --fmax 0.02\n\n\b\n  # Reaction path optimization (outputs trajectories)\n  qme path --strategy interpolate r.xyz p.xyz --npoints 15  # Raw interpolation\n  qme path --strategy neb r.xyz p.xyz --npoints 11 --spring-constant 5.0  # NEB path\n  qme path --strategy cineb r.xyz p.xyz --npoints 11 --spring-constant 5.0  # CI-NEB path\n  qme path --strategy irc ts.xyz --direction both --steps 100  # IRC from transition state\n\n\b\n  # Advanced backends\n  qme minima --strategy local molecule.xyz --backend torchsim_mace --model-name mace-omol-0 --device cuda\n\n\b\n  # Cache management\n  qme cache info  # Show cache information\n  qme cache clear # Clear model cache",
)
@click.version_option()
def main() -> None:
    """Main CLI entry point."""


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
) -> None:
    """Minima optimization using various strategies."""
    # Validate strategy-specific requirements
    if strategy == "interpolate" and product is None:
        msg = "--product is required for interpolate strategy"
        raise click.BadParameter(msg)

    if strategy == "local" and product is not None:
        click.echo("Warning: --product ignored for local strategy")

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

    # Set up explorer
    verbosity = max(0, verbose - 1)

    if verbosity == 0:
        ctx = quiet_backend_loading(backend, model_name, model_path, device, show_model_info=True)
    else:
        ctx = nullcontext()  # type: ignore[assignment]

    with ctx:
        exp = Explorer(
            atoms=atoms_list[0] if len(atoms_list) == 1 else atoms_list,
            backend=backend,
            model_name=model_name,
            model_path=model_path,
            device=device,
            default_charge=default_charge,
            default_spin=default_spin,
            local_optimizer=local_optimizer,
            optimizer_kwargs=optimizer_kwargs,
            target="minima",
            strategy=strategy,
            ts_kwargs=ts_kwargs,
            constraints=constraints,
            verbose=verbosity,
        )

        if dry_run:
            explanation = exp.explain_run()
            click.echo("🔍 Dry-run analysis:")
            click.echo(f"   Target: {explanation['target']}")
            click.echo(f"   Strategy: {explanation['strategy']}")
            click.echo(f"   Runner: {explanation['runner']}")
            click.echo(f"   Valid: {explanation['valid']}")
            click.echo(f"   Notes: {explanation['notes']}")
            return

        # Run optimization
        if strategy == "local":
            results = exp.run(
                fmax=fmax,
                steps=steps,
                calculate_frequencies=calculate_frequencies,
                temperature=temperature,
            )
            result_atoms = results["optimized_atoms"]
            out_default = os.path.splitext(input)[0] + ".opt.local.xyz"
        else:  # interpolate
            results = exp.run(
                npoints=npoints,
                method=interp.lower(),
                fmax=fmax,
                steps=steps,
                calculate_frequencies=calculate_frequencies,
                temperature=temperature,
            )
            result_atoms = results["optimized_atoms"]
            out_default = os.path.splitext(input)[0] + ".opt.interpolate.xyz"

    # Save results
    out = output or out_default
    write_atoms(result_atoms, out)  # type: ignore[arg-type]
    click.echo(f"Minima optimization completed. Saved: {out}")

    # Print frequency analysis summary and save JSON if requested
    if calculate_frequencies and "frequency_analysis" in results:
        print_frequency_summary(results["frequency_analysis"], target="minima")  # type: ignore[arg-type]
        save_results_json(results, out)


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
) -> None:
    """Transition state optimization using various strategies."""
    # Validate strategy-specific requirements
    if strategy in ["interpolate", "growing_string"] and product is None:
        msg = f"--product is required for {strategy} strategy"
        raise click.BadParameter(msg)

    if strategy == "local" and product is not None:
        click.echo("Warning: --product ignored for local strategy")

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

    # Set up explorer
    verbosity = max(0, verbose - 1)

    if verbosity == 0:
        ctx = quiet_backend_loading(backend, model_name, model_path, device, show_model_info=True)
    else:
        ctx = nullcontext()  # type: ignore[assignment]

    with ctx:
        exp = Explorer(
            atoms=atoms_list[0] if len(atoms_list) == 1 else atoms_list,
            backend=backend,
            model_name=model_name,
            model_path=model_path,
            device=device,
            default_charge=default_charge,
            default_spin=default_spin,
            local_optimizer=local_optimizer,
            optimizer_kwargs=optimizer_kwargs,
            target="ts",
            strategy=strategy,
            ts_kwargs=ts_kwargs,
            constraints=constraints,
            verbose=verbosity,
        )

        if dry_run:
            explanation = exp.explain_run()
            click.echo("🔍 Dry-run analysis:")
            click.echo(f"   Target: {explanation['target']}")
            click.echo(f"   Strategy: {explanation['strategy']}")
            click.echo(f"   Runner: {explanation['runner']}")
            click.echo(f"   Valid: {explanation['valid']}")
            click.echo(f"   Notes: {explanation['notes']}")
            return

        # Run optimization
        if strategy == "local":
            results = exp.run(
                fmax=fmax,
                steps=steps,
                calculate_frequencies=calculate_frequencies,
                temperature=temperature,
            )
            result_atoms = results["optimized_atoms"]
            out_default = os.path.splitext(input)[0] + ".ts.local.xyz"
        elif strategy == "interpolate":
            results = exp.run(
                npoints=npoints,
                method=interp.lower(),
                fmax=fmax,
                steps=steps,
                calculate_frequencies=calculate_frequencies,
                temperature=temperature,
            )
            result_atoms = results["optimized_atoms"]
            out_default = os.path.splitext(input)[0] + ".ts.interpolate.xyz"
        else:  # growing_string
            results = exp.run(
                npoints=npoints,
                max_images=max_images,
                distance_threshold=distance_threshold,
                step_size=step_size,
                fmax=fmax,
                steps=steps,
                calculate_frequencies=calculate_frequencies,
                temperature=temperature,
            )
            result_atoms = results["optimized_atoms"]
            out_default = os.path.splitext(input)[0] + ".ts.gsm.xyz"

    # Save results
    out = output or out_default
    write_atoms(result_atoms, out)  # type: ignore[arg-type]
    click.echo(f"Transition state optimization completed. Saved: {out}")

    # Print frequency analysis summary and save JSON if requested
    if calculate_frequencies and "frequency_analysis" in results:
        print_frequency_summary(results["frequency_analysis"], target="ts")  # type: ignore[arg-type]
        save_results_json(results, out)


# Add cache commands
main.add_command(cache)


@main.command()
@click.option(
    "--strategy",
    type=click.Choice(["interpolate", "neb", "cineb", "irc"]),
    default="neb",
    show_default=True,
    help="Path optimization strategy",
)
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--product",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Product structure (required for interpolate, neb, cineb strategies)",
)
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
    input: str,
    product: str | None,
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
) -> None:
    """Reaction path optimization using various strategies.

    Strategies:
    - interpolate: Raw geodesic interpolation (no optimization)
    - neb: Nudged Elastic Band path optimization
    - cineb: Climbing Image NEB path optimization
    - irc: IRC path from transition state

    For interpolate, neb, and cineb: provide reactant and product structures.
    For irc: provide only the transition state structure.
    """
    # Validate strategy-specific requirements
    if strategy in ["interpolate", "neb", "cineb"] and product is None:
        msg = f"--product is required for {strategy} strategy"
        raise click.BadParameter(msg)

    if strategy == "irc" and product is not None:
        click.echo("Warning: --product ignored for irc strategy")

    # Load structures
    atoms = load_atoms_from_xyz(input)
    atoms_list = [atoms]

    if strategy in ["interpolate", "neb", "cineb"]:
        if product is None:
            raise ValueError("Product file is required for interpolate/neb/cineb strategy")
        atoms_product = load_atoms_from_xyz(product)
        atoms_list = [atoms, atoms_product]

    # Parse kwargs
    optimizer_kwargs = parse_kv_pairs(list(optimizer_kw))
    ts_kwargs = parse_kv_pairs(list(ts_kw))

    # Set up explorer
    verbosity = max(0, verbose - 1)

    if verbosity == 0:
        ctx = quiet_backend_loading(backend, model_name, model_path, device, show_model_info=True)
    else:
        ctx = nullcontext()  # type: ignore[assignment]

    with ctx:
        exp = Explorer(
            atoms=atoms_list[0] if len(atoms_list) == 1 else atoms_list,
            backend=backend,
            model_name=model_name,
            model_path=model_path,
            device=device,
            default_charge=default_charge,
            default_spin=default_spin,
            local_optimizer=local_optimizer,
            optimizer_kwargs=optimizer_kwargs,
            target="path",
            strategy=strategy,
            ts_kwargs=ts_kwargs,
            constraints=constraints,
            verbose=verbosity,
        )

        if dry_run:
            explanation = exp.explain_run()
            click.echo("🔍 Dry-run analysis:")
            click.echo(f"   Target: {explanation['target']}")
            click.echo(f"   Strategy: {explanation['strategy']}")
            click.echo(f"   Runner: {explanation['runner']}")
            click.echo(f"   Valid: {explanation['valid']}")
            click.echo(f"   Notes: {explanation['notes']}")
            return

        # Run path optimization
        if strategy == "interpolate":
            result = exp.run(
                npoints=npoints,
                method=interp.lower(),
                calculate_frequencies=calculate_frequencies,
                temperature=temperature,
            )
            out_default = os.path.splitext(input)[0] + ".path.interpolate.xyz"
        elif strategy == "neb":
            result = exp.run(
                npoints=npoints,
                method=interp.lower(),
                fmax=fmax,
                steps=steps,
                spring_constant=spring_constant,
                calculate_frequencies=calculate_frequencies,
                temperature=temperature,
            )
            out_default = os.path.splitext(input)[0] + ".path.neb.xyz"
        elif strategy == "cineb":
            result = exp.run(
                npoints=npoints,
                method=interp.lower(),
                fmax=fmax,
                steps=steps,
                spring_constant=spring_constant,
                climb=True,
                calculate_frequencies=calculate_frequencies,
                temperature=temperature,
            )
            out_default = os.path.splitext(input)[0] + ".path.cineb.xyz"
        elif strategy == "irc":
            result = exp.run(
                fmax=fmax,
                steps=steps,
                step_size=step_size,
                direction=direction.lower(),
                calculate_frequencies=calculate_frequencies,
                temperature=temperature,
            )
            out_default = os.path.splitext(input)[0] + ".path.irc.xyz"

        # Extract trajectory from result
        if isinstance(result, dict) and "trajectory" in result:
            trajectory = result["trajectory"]
        else:
            # For path strategies, the result should contain trajectory
            trajectory = result.get("optimized_atoms", [])

    # Save results
    out = output or out_default
    write_atoms(trajectory, out)  # type: ignore[arg-type]
    click.echo(f"Path optimization completed. Saved {len(trajectory)} images to: {out}")  # type: ignore[arg-type]

    # Print frequency analysis summary and save JSON if requested
    if calculate_frequencies and "frequency_analysis" in result:
        # For path strategies, we might have frequency analysis on the final structure
        print_frequency_summary(result["frequency_analysis"], target="path")  # type: ignore[arg-type]
        save_results_json(result, out)


__all__ = ["main", "minima", "path", "ts"]


if __name__ == "__main__":
    main()
