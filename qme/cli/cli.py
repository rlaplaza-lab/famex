import os
from typing import List, Optional

import click
from ase import Atoms

from qme.cli.cache_commands import cache
from qme.cli.cli_helpers import (
    load_atoms_from_xyz,
    parse_kv_pairs,
    write_atoms,
)
from qme.core.explorer import Explorer
from qme.logging_utils import quiet_backend_loading

# Disable matplotlib and ASE GUI before importing ASE
# This prevents unwanted popup windows during CLI operations
os.environ.setdefault("MPLBACKEND", "Agg")


@click.group(
    help=(
        "QME CLI: Quick mechanistic exploration with ML potentials.\n\n"
        "Commands:\n"
        "  qme opt   : Local or two-ended minima optimization from XYZ\n"
        "  qme tsopt : Local or two-ended TS optimization from XYZ\n"
        "  qme cache : Manage model cache\n\n"
        "Examples:\n"
        "  qme opt reactant.xyz --backend aimnet2 --optimizer sella --fmax 0.03\n"
        "  qme opt reactant.xyz --product product.xyz --interp geodesic --npoints 21\n"
        "  qme tsopt ts_guess.xyz --optimizer sella --ts-kw order=1\n"
        "  qme tsopt r.xyz --product p.xyz --interp geodesic --npoints 15\n"
        "  qme tsopt r.xyz --product p.xyz --mode neb --npoints 11 --spring-constant 5.0\n"
        "  qme opt molecule.xyz --backend torchsim_mace --model-name mace-omol-0 --device cuda\n"
        "  qme opt molecule.xyz --backend torchsim_fairchem --model-name equiformer_v2_31M_s2ef_all_md\n"
        "  qme cache info  # Show cache information\n"
        "  qme cache clear # Clear model cache\n"
    )
)
@click.version_option()
def main():
    pass


# Add cache commands
main.add_command(cache)


def _common_explorer_options(f):
    opts = [
        click.option(
            "--backend",
            default="uma",
            show_default=True,
            help="Backend: uma|so3lr|aimnet2|mace|torchsim|torchsim_mace|torchsim_fairchem|mock",
        ),
        click.option("--model-name", default=None, help="Model name for backend"),
        click.option(
            "--model-path", default=None, help="Path to model file (if applicable)"
        ),
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
            default="sella",
            show_default=True,
            help="Local optimizer: sella|lbfgs|bfgs|fire",
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
            "--quiet/--no-quiet",
            default=True,
            show_default=True,
            help="Suppress verbose backend logs",
        ),
    ]
    for opt in reversed(opts):
        f = opt(f)
    return f


@main.command(
    help=(
        "Local or two-ended minima optimization.\n\n"
        "If only INPUT is given, runs a local optimization.\n"
        "If --product is provided, interpolates between INPUT and PRODUCT and\n"
        "optimizes low-energy frames (two-ended minima)."
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
@click.option(
    "--fmax", type=float, default=0.05, show_default=True, help="Convergence threshold"
)
@click.option(
    "--steps", type=int, default=1000, show_default=True, help="Max optimization steps"
)
@click.option(
    "--npoints",
    type=int,
    default=11,
    show_default=True,
    help="Interpolation points for two-ended path",
)
@click.option(
    "--interp",
    type=click.Choice(["linear", "geodesic"], case_sensitive=False),
    default="geodesic",
    show_default=True,
    help="Interpolation method for two-ended",
)
@_common_explorer_options
def opt(
    input: str,
    product: Optional[str],
    output_path: Optional[str],
    fmax: float,
    steps: int,
    npoints: int,
    interp: str,
    backend: str,
    model_name: Optional[str],
    model_path: Optional[str],
    device: Optional[str],
    default_charge: int,
    default_spin: int,
    local_optimizer: str,
    optimizer_kw: List[str],
    ts_kw: List[str],
    constraints: Optional[str],
    quiet: bool,
):
    atoms = load_atoms_from_xyz(input)
    p_atoms: Optional[Atoms] = load_atoms_from_xyz(product) if product else None

    optimizer_kwargs = parse_kv_pairs(list(optimizer_kw))
    ts_kwargs = parse_kv_pairs(list(ts_kw))

    if quiet:
        ctx = quiet_backend_loading(
            backend, model_name, model_path, device, show_model_info=True
        )
    else:
        from contextlib import nullcontext

        ctx = nullcontext()

    with ctx:
        if p_atoms is None:
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
                strategy="local",
                target="minima",
                ts_kwargs=ts_kwargs,
                constraints=constraints,
            )
            results = exp.run(mode="minima", fmax=fmax, steps=steps)
            result_atoms = (
                results
                if isinstance(results, Atoms)
                else (results[0] if results else atoms)
            )
        else:
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
                strategy="two-ended",
                target="minima",
                ts_kwargs=ts_kwargs,
                constraints=constraints,
            )
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


@main.command(help="Transition-state optimization from one or two XYZ files")
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
@click.option(
    "--fmax", type=float, default=0.05, show_default=True, help="Convergence threshold"
)
@click.option(
    "--steps", type=int, default=1000, show_default=True, help="Max optimization steps"
)
@click.option(
    "--npoints",
    type=int,
    default=11,
    show_default=True,
    help="Interpolation points for two-ended path",
)
@click.option(
    "--interp",
    type=click.Choice(["linear", "geodesic"], case_sensitive=False),
    default="geodesic",
    show_default=True,
    help="Interpolation method for two-ended",
)
@click.option(
    "--mode",
    type=click.Choice(["interpolate", "neb"], case_sensitive=False),
    default="interpolate",
    show_default=True,
    help="Two-ended mode: interpolate (TS guess) or neb (NEB path optimization)",
)
@click.option(
    "--spring-constant",
    type=float,
    default=5.0,
    show_default=True,
    help="Spring constant for NEB mode (only used with --mode neb)",
)
@_common_explorer_options
def tsopt(
    reactant: str,
    product: Optional[str],
    output_path: Optional[str],
    fmax: float,
    steps: int,
    npoints: int,
    interp: str,
    mode: str,
    spring_constant: float,
    backend: str,
    model_name: Optional[str],
    model_path: Optional[str],
    device: Optional[str],
    default_charge: int,
    default_spin: int,
    local_optimizer: str,
    optimizer_kw: List[str],
    ts_kw: List[str],
    constraints: Optional[str],
    quiet: bool,
):
    r_atoms = load_atoms_from_xyz(reactant)
    p_atoms: Optional[Atoms] = load_atoms_from_xyz(product) if product else None

    optimizer_kwargs = parse_kv_pairs(list(optimizer_kw))
    ts_kwargs = parse_kv_pairs(list(ts_kw))

    if quiet:
        ctx = quiet_backend_loading(
            backend, model_name, model_path, device, show_model_info=True
        )
    else:
        from contextlib import nullcontext

        ctx = nullcontext()

    with ctx:
        if p_atoms is None:
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
                strategy="local",
                target="ts",
                ts_kwargs=ts_kwargs,
                constraints=constraints,
            )
            results = exp.run(mode="ts", fmax=fmax, steps=steps)
            ts_atoms = (
                results
                if isinstance(results, Atoms)
                else (results[0] if results else r_atoms)
            )
        else:
            # Two-ended case - determine target based on mode
            if mode == "neb":
                target = "neb"
            else:
                target = "ts"

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
                strategy="two-ended",
                target=target,
                ts_kwargs=ts_kwargs,
                constraints=constraints,
            )

            if mode == "neb":
                # NEB mode - return the full path
                ts_atoms = exp.run(
                    mode="neb",
                    npoints=npoints,
                    method=interp.lower(),
                    fmax=fmax,
                    steps=steps,
                    spring_constant=spring_constant,
                )
            else:
                # Interpolate mode - return single TS guess
                ts_atoms = exp.run(
                    mode="interpolate",
                    npoints=npoints,
                    method=interp.lower(),
                    fmax=fmax,
                    steps=steps,
                )

    # Default output next to the reactant file
    out_base = os.path.splitext(reactant)[0]
    if mode == "neb":
        out = output_path or (out_base + ".neb.xyz")
        write_atoms(ts_atoms, out)
        click.echo(
            f"NEB optimization completed. Saved {len(ts_atoms)} images to: {out}"
        )
    else:
        out = output_path or (out_base + ".ts.xyz")
        write_atoms(ts_atoms, out)
        click.echo(f"TS optimization completed. Saved: {out}")


__all__ = ["main", "opt", "tsopt"]
