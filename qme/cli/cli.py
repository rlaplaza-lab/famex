"""QME Command Line Interface.

This module provides the main CLI interface for QME (Quick Mechanistic Exploration),
allowing users to perform molecular optimization tasks from the command line.
"""

import os
from contextlib import nullcontext
from typing import Any

import click
from ase import Atoms

from qme.cli.cache_commands import cache
from qme.cli.cli_helpers import load_atoms_from_xyz, parse_kv_pairs, write_atoms
from qme.core.explorer import Explorer
from qme.logging_utils import quiet_backend_loading

# Disable matplotlib and ASE GUI before importing ASE
# This prevents unwanted popup windows during CLI operations
os.environ.setdefault("MPLBACKEND", "Agg")


@click.group(
    help=(
        "QME CLI: Quick mechanistic exploration with ML potentials.\n\n"
        "Commands:\n"
        "  qme opt   : Minima optimization (outputs single structure, defaults to BFGS)\n"
        "  qme tsopt : Transition state optimization (outputs single TS, defaults to Sella)\n"
        "  qme path  : Reaction path optimization (outputs trajectories)\n"
        "  qme cache : Manage model cache\n\n"
        "Examples:\n"
        "  # Minima optimization (outputs single structure)\n"
        "  qme opt reactant.xyz --backend aimnet2 --fmax 0.03  # Uses BFGS by default\n"
        "  qme opt reactant.xyz --product product.xyz --interp geodesic --npoints 21\n"
        "  \n"
        "  # Transition state optimization (outputs single TS)\n"
        "  qme tsopt ts_guess.xyz --ts-kw order=1  # Uses Sella by default\n"
        "  qme tsopt r.xyz --product p.xyz --npoints 15  # Two-ended TS guess\n"
        "  \n"
        "  # Reaction path optimization (outputs trajectories)\n"
        "  qme path interpolate r.xyz p.xyz --npoints 15  # Raw interpolation\n"
        "  qme path neb r.xyz p.xyz --npoints 11 --spring-constant 5.0  # NEB path\n"
        "  qme path cineb r.xyz p.xyz --npoints 11 --spring-constant 5.0  # CI-NEB path\n"
        "  \n"
        "  # Advanced backends\n"
        "  qme opt molecule.xyz --backend torchsim_mace --model-name mace-omol-0 --device cuda\n"
        "  qme opt molecule.xyz --backend torchsim_fairchem \\\n"
        "      --model-name equiformer_v2_31M_s2ef_all_md\n"
        "  \n"
        "  # Cache management\n"
        "  qme cache info  # Show cache information\n"
        "  qme cache clear # Clear model cache\n"
    )
)
@click.version_option()
def main() -> None:
    """Main CLI entry point."""


# Add cache commands
main.add_command(cache)


def _common_explorer_options(f: Any) -> Any:
    opts = [
        click.option(
            "--backend",
            default="uma",
            show_default=True,
            help="Backend: uma|so3lr|aimnet2|mace|orb|torchsim|torchsim_mace|"
            "torchsim_fairchem|mock",
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
            "--optimizer",
            "local_optimizer",
            default="bfgs",
            show_default=True,
            help="Local optimizer: sella|tric|lbfgs|bfgs|fire",
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
    ]
    for opt in reversed(opts):
        f = opt(f)
    return f


@main.command(
    help=(
        "Local or two-ended minima optimization (outputs single optimized structure).\n\n"
        "If only INPUT is given, runs a local optimization.\n"
        "If --product is provided, interpolates between INPUT and PRODUCT and\n"
        "optimizes the lowest energy frame (two-ended minima)."
    )
)
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--product",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Product XYZ for two-ended minima path",
)
@click.option(
    "--output",
    "output_path",
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
    help="Number of interpolation points for two-ended minima search",
)
@click.option(
    "--interp",
    type=click.Choice(["linear", "geodesic"], case_sensitive=False),
    default="geodesic",
    show_default=True,
    help="Interpolation method for two-ended minima search",
)
@_common_explorer_options
def opt(
    input: str,
    product: str | None,
    output_path: str | None,
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
) -> None:
    atoms = load_atoms_from_xyz(input)
    p_atoms: Atoms | None = load_atoms_from_xyz(product) if product else None

    optimizer_kwargs = parse_kv_pairs(list(optimizer_kw))
    ts_kwargs = parse_kv_pairs(list(ts_kw))

    # Convert verbose count to verbosity level (0=quiet, 1=normal, 2+=debug)
    verbosity = max(0, verbose - 1)

    if verbosity == 0:
        ctx = quiet_backend_loading(backend, model_name, model_path, device, show_model_info=True)
    else:

        ctx = nullcontext()  # type: ignore

    with ctx:
        if p_atoms is None:
            # Local minima optimization
            exp = Explorer(
                atoms=atoms,
                backend=backend,
                model_name=model_name,
                model_path=model_path,
                device=device,
                default_charge=default_charge,
                default_spin=default_spin,
                local_optimizer=local_optimizer,
                optimizer_kwargs=optimizer_kwargs,
                target="minima",
                strategy="local",
                ts_kwargs=ts_kwargs,
                constraints=constraints,
                verbose=verbosity,
            )

            if dry_run:
                explanation = exp.explain_run(mode="minima")
                click.echo("🔍 Dry-run analysis:")
                click.echo(f"   Target: {explanation['target']}")
                click.echo(f"   Strategy: {explanation['strategy']}")
                click.echo(f"   Runner: {explanation['runner']}")
                click.echo(f"   Valid: {explanation['valid']}")
                click.echo(f"   Notes: {explanation['notes']}")
                return

            results = exp.run(mode="minima", fmax=fmax, steps=steps)
            result_atoms = results["optimized_atoms"]
        else:
            # Two-ended minima optimization (interpolate strategy)
            exp = Explorer(
                atoms=[atoms, p_atoms],
                backend=backend,
                model_name=model_name,
                model_path=model_path,
                device=device,
                default_charge=default_charge,
                default_spin=default_spin,
                local_optimizer=local_optimizer,
                optimizer_kwargs=optimizer_kwargs,
                target="minima",
                strategy="interpolate",
                ts_kwargs=ts_kwargs,
                constraints=constraints,
                verbose=verbosity,
            )

            if dry_run:
                explanation = exp.explain_run(mode="interpolate")
                click.echo("🔍 Dry-run analysis:")
                click.echo(f"   Strategy: {explanation['strategy']}")
                click.echo(f"   Runner: {explanation['runner']}")
                click.echo(f"   Valid: {explanation['valid']}")
                click.echo(f"   Notes: {explanation['notes']}")
                return

            result_atoms = exp.run(
                mode="interpolate",
                npoints=npoints,
                method=interp.lower(),
                fmax=fmax,
                steps=steps,
            )

    out_default = os.path.splitext(input)[0] + (
        ".opt.twoended.xyz" if p_atoms is not None else ".opt.xyz"
    )
    out = output_path or out_default
    write_atoms(result_atoms, out)
    click.echo(f"Optimization completed. Saved: {out}")


@main.command(
    help=(
        "Transition state optimization from one or two XYZ files.\n\n"
        "If only REACTANT is given, runs local TS optimization.\n"
        "If --product is provided, generates TS guess via interpolation and refines it.\n\n"
        "For two-ended TS guess, --product is required."
    )
)
@click.argument("reactant", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--product",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Product XYZ for two-ended TS guess",
)
@click.option(
    "--output",
    "output_path",
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
    help="Number of interpolation points for two-ended TS guess",
)
@click.option(
    "--interp",
    type=click.Choice(["linear", "geodesic"], case_sensitive=False),
    default="geodesic",
    show_default=True,
    help="Interpolation method for two-ended TS guess",
)
@_common_explorer_options
def tsopt(
    reactant: str,
    product: str | None,
    output_path: str | None,
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
) -> None:
    r_atoms = load_atoms_from_xyz(reactant)
    p_atoms: Atoms | None = load_atoms_from_xyz(product) if product else None

    optimizer_kwargs = parse_kv_pairs(list(optimizer_kw))
    ts_kwargs = parse_kv_pairs(list(ts_kw))

    # Convert verbose count to verbosity level (0=quiet, 1=normal, 2+=debug)
    verbosity = max(0, verbose - 1)

    if verbosity == 0:
        ctx = quiet_backend_loading(backend, model_name, model_path, device, show_model_info=True)
    else:

        ctx = nullcontext()  # type: ignore

    with ctx:
        if p_atoms is None:
            # Local TS optimization
            exp = Explorer(
                atoms=r_atoms,
                backend=backend,
                model_name=model_name,
                model_path=model_path,
                device=device,
                default_charge=default_charge,
                default_spin=default_spin,
                local_optimizer=local_optimizer,
                optimizer_kwargs=optimizer_kwargs,
                target="ts",
                strategy="local",
                ts_kwargs=ts_kwargs,
                constraints=constraints,
                verbose=verbosity,
            )
            results = exp.run(mode="ts", fmax=fmax, steps=steps)
            ts_atoms = results["optimized_atoms"]
        else:
            # TS from interpolated guess (interpolate strategy)
            exp = Explorer(
                atoms=[r_atoms, p_atoms],
                backend=backend,
                model_name=model_name,
                model_path=model_path,
                device=device,
                default_charge=default_charge,
                default_spin=default_spin,
                local_optimizer=local_optimizer,
                optimizer_kwargs=optimizer_kwargs,
                target="ts",
                strategy="interpolate",
                ts_kwargs=ts_kwargs,
                constraints=constraints,
                verbose=verbosity,
            )

            result = exp.run(
                mode="ts",
                npoints=npoints,
                method=interp.lower(),
                fmax=fmax,
                steps=steps,
            )
            ts_atoms = result["optimized_atoms"]

    # Default output next to the reactant file
    out_base = os.path.splitext(reactant)[0]
    out = output_path or (out_base + ".ts.xyz")

    write_atoms(ts_atoms, out)
    click.echo(f"TS optimization completed. Saved: {out}")


@main.group(
    help=(
        "Reaction path optimization between reactant and product structures.\n\n"
        "Subcommands:\n"
        "  interpolate : Raw geodesic interpolation (no optimization)\n"
        "  neb         : Nudged Elastic Band path optimization\n"
        "  cineb       : Climbing Image NEB path optimization\n"
        "  irc         : IRC path from transition state\n\n"
        "All subcommands save complete reaction pathways as trajectory files."
    )
)
def path() -> None:
    """Reaction path optimization between reactant and product structures."""


@path.command(
    help=(
        "Generate raw interpolation path between reactant and product.\n\n"
        "Creates a smooth geometric interpolation between structures without\n"
        "energy optimization. Useful for initial path visualization."
    )
)
@click.argument("reactant", type=click.Path(exists=True, dir_okay=False))
@click.argument("product", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False),
    default=None,
    help="Output trajectory XYZ path",
)
@click.option(
    "--npoints",
    type=int,
    default=11,
    show_default=True,
    help="Number of interpolation points",
)
@click.option(
    "--interp",
    type=click.Choice(["linear", "geodesic"], case_sensitive=False),
    default="geodesic",
    show_default=True,
    help="Interpolation method",
)
@_common_explorer_options
def interpolate(
    reactant: str,
    product: str,
    output_path: str | None,
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
) -> None:
    r_atoms = load_atoms_from_xyz(reactant)
    p_atoms = load_atoms_from_xyz(product)

    # Note: optimizer_kwargs and ts_kwargs are not used for raw interpolation
    # but are parsed for consistency with other commands
    _ = parse_kv_pairs(list(optimizer_kw))  # optimizer_kwargs
    _ = parse_kv_pairs(list(ts_kw))  # ts_kwargs

    # Convert verbose count to verbosity level (0=quiet, 1=normal, 2+=debug)
    verbosity = max(0, verbose - 1)

    if verbosity == 0:
        ctx = quiet_backend_loading(backend, model_name, model_path, device, show_model_info=True)
    else:

        ctx = nullcontext()  # type: ignore

    with ctx:
        # Use Explorer with path:interpolate strategy for raw interpolation
        exp = Explorer(
            atoms=[r_atoms, p_atoms],
            backend=backend,
            model_name=model_name,
            model_path=model_path,
            device=device,
            default_charge=default_charge,
            default_spin=default_spin,
            local_optimizer=local_optimizer,
            optimizer_kwargs={},
            target="path",
            strategy="interpolate",
            ts_kwargs={},
            constraints=constraints,
            verbose=verbosity,
        )

        if dry_run:
            explanation = exp.explain_run(mode="interpolate")
            click.echo("🔍 Dry-run analysis:")
            click.echo(f"   Target: {explanation['target']}")
            click.echo(f"   Strategy: {explanation['strategy']}")
            click.echo(f"   Runner: {explanation['runner']}")
            click.echo(f"   Valid: {explanation['valid']}")
            click.echo(f"   Notes: {explanation['notes']}")
            return

        # Run interpolation
        result = exp.run(
            mode="interpolate",
            npoints=npoints,
            method=interp.lower(),
        )

        # Extract path from result
        if isinstance(result, dict) and "trajectory" in result:
            path_result = result["trajectory"]
        else:
            path_result = result

    # Save trajectory
    out_base = os.path.splitext(reactant)[0]
    out = output_path or (out_base + ".interpolate.xyz")
    write_atoms(path_result, out)
    click.echo(f"Interpolation completed. Saved {len(path_result)} images to: {out}")


@path.command(
    help=(
        "Nudged Elastic Band (NEB) path optimization.\n\n"
        "Optimizes a reaction pathway using NEB with spring forces between\n"
        "images. Saves complete reaction pathway as trajectory file."
    )
)
@click.argument("reactant", type=click.Path(exists=True, dir_okay=False))
@click.argument("product", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output",
    "output_path",
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
    type=click.Choice(["linear", "geodesic"], case_sensitive=False),
    default="geodesic",
    show_default=True,
    help="Initial interpolation method",
)
@click.option(
    "--spring-constant",
    type=float,
    default=5.0,
    show_default=True,
    help="Spring constant for NEB",
)
@_common_explorer_options
def neb(
    reactant: str,
    product: str,
    output_path: str | None,
    fmax: float,
    steps: int,
    npoints: int,
    interp: str,
    spring_constant: float,
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
) -> None:
    r_atoms = load_atoms_from_xyz(reactant)
    p_atoms = load_atoms_from_xyz(product)

    optimizer_kwargs = parse_kv_pairs(list(optimizer_kw))
    ts_kwargs = parse_kv_pairs(list(ts_kw))

    # Convert verbose count to verbosity level (0=quiet, 1=normal, 2+=debug)
    verbosity = max(0, verbose - 1)

    if verbosity == 0:
        ctx = quiet_backend_loading(backend, model_name, model_path, device, show_model_info=True)
    else:

        ctx = nullcontext()  # type: ignore

    with ctx:
        exp = Explorer(
            atoms=[r_atoms, p_atoms],
            backend=backend,
            model_name=model_name,
            model_path=model_path,
            device=device,
            default_charge=default_charge,
            default_spin=default_spin,
            local_optimizer=local_optimizer,
            optimizer_kwargs=optimizer_kwargs,
            target="path",
            strategy="neb",
            ts_kwargs=ts_kwargs,
            constraints=constraints,
            verbose=verbosity,
        )

        if dry_run:
            explanation = exp.explain_run(mode="neb")
            click.echo("🔍 Dry-run analysis:")
            click.echo(f"   Target: {explanation['target']}")
            click.echo(f"   Strategy: {explanation['strategy']}")
            click.echo(f"   Runner: {explanation['runner']}")
            click.echo(f"   Valid: {explanation['valid']}")
            click.echo(f"   Notes: {explanation['notes']}")
            return

        # Run NEB optimization
        neb_result = exp.run(
            mode="neb",
            npoints=npoints,
            method=interp.lower(),
            fmax=fmax,
            steps=steps,
            spring_constant=spring_constant,
        )

    # Save trajectory
    out_base = os.path.splitext(reactant)[0]
    out = output_path or (out_base + ".neb.xyz")
    write_atoms(neb_result, out)
    click.echo(f"NEB optimization completed. Saved {len(neb_result)} images to: {out}")


@path.command(
    help=(
        "Climbing Image NEB (CI-NEB) path optimization.\n\n"
        "Optimizes a reaction pathway using CI-NEB with climbing image\n"
        "behavior for better transition state location. Saves complete\n"
        "reaction pathway as trajectory file."
    )
)
@click.argument("reactant", type=click.Path(exists=True, dir_okay=False))
@click.argument("product", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output",
    "output_path",
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
    type=click.Choice(["linear", "geodesic"], case_sensitive=False),
    default="geodesic",
    show_default=True,
    help="Initial interpolation method",
)
@click.option(
    "--spring-constant",
    type=float,
    default=5.0,
    show_default=True,
    help="Spring constant for CI-NEB",
)
@_common_explorer_options
def cineb(
    reactant: str,
    product: str,
    output_path: str | None,
    fmax: float,
    steps: int,
    npoints: int,
    interp: str,
    spring_constant: float,
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
) -> None:
    r_atoms = load_atoms_from_xyz(reactant)
    p_atoms = load_atoms_from_xyz(product)

    optimizer_kwargs = parse_kv_pairs(list(optimizer_kw))
    ts_kwargs = parse_kv_pairs(list(ts_kw))

    # Convert verbose count to verbosity level (0=quiet, 1=normal, 2+=debug)
    verbosity = max(0, verbose - 1)

    if verbosity == 0:
        ctx = quiet_backend_loading(backend, model_name, model_path, device, show_model_info=True)
    else:

        ctx = nullcontext()  # type: ignore

    with ctx:
        exp = Explorer(
            atoms=[r_atoms, p_atoms],
            backend=backend,
            model_name=model_name,
            model_path=model_path,
            device=device,
            default_charge=default_charge,
            default_spin=default_spin,
            local_optimizer=local_optimizer,
            optimizer_kwargs=optimizer_kwargs,
            target="path",
            strategy="cineb",
            ts_kwargs=ts_kwargs,
            constraints=constraints,
            verbose=verbosity,
        )

        if dry_run:
            explanation = exp.explain_run(mode="cineb")
            click.echo("🔍 Dry-run analysis:")
            click.echo(f"   Target: {explanation['target']}")
            click.echo(f"   Strategy: {explanation['strategy']}")
            click.echo(f"   Runner: {explanation['runner']}")
            click.echo(f"   Valid: {explanation['valid']}")
            click.echo(f"   Notes: {explanation['notes']}")
            return

        # Run CI-NEB optimization
        cineb_result = exp.run(
            mode="cineb",
            npoints=npoints,
            method=interp.lower(),
            fmax=fmax,
            steps=steps,
            spring_constant=spring_constant,
            climb=True,
        )

    # Save trajectory
    out_base = os.path.splitext(reactant)[0]
    out = output_path or (out_base + ".cineb.xyz")
    write_atoms(cineb_result, out)
    click.echo(f"CI-NEB optimization completed. Saved {len(cineb_result)} images to: {out}")


@path.command(
    help=(
        "IRC (Intrinsic Reaction Coordinate) path from transition state.\n\n"
        "Follows the gradient downhill from a transition state in both forward\n"
        "and backward directions to generate a reaction path. This produces the\n"
        "minimum energy path connecting reactants and products through the TS."
    )
)
@click.argument("ts_structure", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False),
    default=None,
    help="Output trajectory XYZ path",
)
@click.option("--fmax", type=float, default=0.05, show_default=True, help="Convergence threshold")
@click.option("--steps", type=int, default=100, show_default=True, help="Max IRC steps per direction")
@click.option(
    "--step-size",
    type=float,
    default=0.1,
    show_default=True,
    help="IRC step size (amu^1/2 * Angstrom)",
)
@click.option(
    "--direction",
    type=click.Choice(["forward", "backward", "both"], case_sensitive=False),
    default="both",
    show_default=True,
    help="Direction to follow from TS",
)
@_common_explorer_options
def irc(
    ts_structure: str,
    output_path: str | None,
    fmax: float,
    steps: int,
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
) -> None:
    ts_atoms = load_atoms_from_xyz(ts_structure)

    optimizer_kwargs = parse_kv_pairs(list(optimizer_kw))
    ts_kwargs = parse_kv_pairs(list(ts_kw))

    # Convert verbose count to verbosity level (0=quiet, 1=normal, 2+=debug)
    verbosity = max(0, verbose - 1)

    if verbosity == 0:
        ctx = quiet_backend_loading(backend, model_name, model_path, device, show_model_info=True)
    else:

        ctx = nullcontext()  # type: ignore

    with ctx:
        exp = Explorer(
            atoms=ts_atoms,
            backend=backend,
            model_name=model_name,
            model_path=model_path,
            device=device,
            default_charge=default_charge,
            default_spin=default_spin,
            local_optimizer=local_optimizer,
            optimizer_kwargs=optimizer_kwargs,
            target="path",
            strategy="irc",
            ts_kwargs=ts_kwargs,
            constraints=constraints,
            verbose=verbosity,
        )

        if dry_run:
            explanation = exp.explain_run(mode="irc")
            click.echo("🔍 Dry-run analysis:")
            click.echo(f"   Target: {explanation['target']}")
            click.echo(f"   Strategy: {explanation['strategy']}")
            click.echo(f"   Runner: {explanation['runner']}")
            click.echo(f"   Valid: {explanation['valid']}")
            click.echo(f"   Notes: {explanation['notes']}")
            return

        # Run IRC calculation
        irc_result = exp.run(
            mode="irc",
            fmax=fmax,
            steps=steps,
            step_size=step_size,
            direction=direction.lower(),
        )

        # Extract trajectory from result
        if isinstance(irc_result, dict) and "trajectory" in irc_result:
            trajectory = irc_result["trajectory"]
        else:
            trajectory = irc_result

    # Save trajectory
    out_base = os.path.splitext(ts_structure)[0]
    out = output_path or (out_base + ".irc.xyz")
    write_atoms(trajectory, out)
    click.echo(f"IRC calculation completed. Saved {len(trajectory)} images to: {out}")


__all__ = ["main", "opt", "tsopt", "path"]


if __name__ == "__main__":
    main()
