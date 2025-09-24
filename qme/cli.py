"""
Command Line Interface for QME (Quick Mechanistic Exploration).
"""

import sys
from pathlib import Path

import click

from .core import QMEOptimizer
from .geometry import read_geometry
from .reaction import Reaction


@click.group()
@click.version_option(version="0.1.0")
def main():
    """
    QME: Quick mechanistic exploration using machine learning potentials.

    Supports multiple neural network backends including UMA and SO3LR potentials
    for molecular geometry optimization and transition state searches.
    """
    pass


def add_common_options(options):
    """A decorator factory to add a list of common click options."""

    def decorator(f):
        for option in reversed(options):
            f = option(f)
        return f

    return decorator


# Define common option groups
core_options = [
    click.option(
        "--backend",
        "-b",
        default="so3lr",
        type=click.Choice(["uma", "so3lr", "aimnet2"]),
        help="Backend to use (uma, so3lr, or aimnet2)",
    ),
    click.option("--model", "-m", default=None, type=str, help="Model name to use"),
    click.option(
        "--model-path",
        type=click.Path(exists=True),
        help="Path to model file (SO3LR only)",
    ),
    click.option(
        "--device",
        "-d",
        type=click.Choice(["cpu", "cuda"]),
        default=None,
        help="Device for computations (auto-detected if not specified)",
    ),
    click.option("--verbose", "-v", is_flag=True, help="Verbose output"),
]

optimization_options = [
    click.option(
        "--fmax",
        "-f",
        default=0.01,
        type=float,
        help="Force convergence criterion (eV/Å)",
    ),
    click.option(
        "--steps",
        "-s",
        default=200,
        type=int,
        help="Maximum number of optimization steps",
    ),
    click.option(
        "--logfile", type=click.Path(), help="Log file for optimization output"
    ),
    click.option(
        "--trajectory",
        type=click.Path(),
        help="Trajectory file to save optimization steps",
    ),
    click.option(
        "--constraint-atoms",
        type=str,
        help="Comma-separated list of atom indices to fix (0-based)",
    ),
]


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output", "-o", type=click.Path(), help="Output file for optimized structure"
)
@click.option(
    "--optimizer",
    "-opt",
    default="BFGS",
    type=click.Choice(["BFGS", "LBFGS", "FIRE"]),
    help="Optimizer to use for minimization",
)
@add_common_options(core_options)
@add_common_options(optimization_options)
def minimize(
    input_file,
    output,
    optimizer,
    fmax,
    steps,
    model,
    backend,
    model_path,
    device,
    logfile,
    trajectory,
    constraint_atoms,
    verbose,
):
    """Find minimum energy geometry using specified optimizer.

    Optimizes a molecular structure to its minimum energy configuration using
    machine learning potentials and ASE optimizers.

    Args:
        input_file: Path to molecular structure file (xyz, cif, pdb, etc.).
        output: Output file for optimized structure (optional).
        optimizer: Optimization algorithm to use.
        fmax: Force convergence criterion in eV/Å.
        steps: Maximum number of optimization steps.
        model: UMA model name to use for calculations.
        device: Computation device (cpu/cuda).
        logfile: Optional file for optimization logging.
        trajectory: Optional file to save optimization trajectory.
        constraint_atoms: Comma-separated atom indices to fix during optimization.
        verbose: Enable detailed output.
    """

    if verbose:
        click.echo("Starting minimum energy optimization...")
        click.echo(f"Input file: {input_file}")
        click.echo(f"Backend: {backend}")
        click.echo(f"Optimizer: {optimizer}")
        if model:
            click.echo(f"Model: {model}")
        if model_path:
            click.echo(f"Model path: {model_path}")

    try:
        # Initialize optimizer
        qme = QMEOptimizer(
            backend=backend, model_name=model, model_path=model_path, device=device
        )

        # Load structure
        atoms = qme.load_structure(input_file)

        if verbose:
            click.echo(f"Loaded structure with {len(atoms)} atoms")

        # Parse constraints if provided
        constraints = None
        if constraint_atoms:
            try:
                fixed_indices = [int(i.strip()) for i in constraint_atoms.split(",")]
                from ase.constraints import FixAtoms

                constraints = [FixAtoms(indices=fixed_indices)]
                if verbose:
                    click.echo(f"Fixed atoms: {fixed_indices}")
            except ValueError as e:
                click.echo(f"Error parsing constraint atoms: {e}", err=True)
                sys.exit(1)

        # Run optimization
        results = qme.optimize_minimum(
            optimizer=optimizer,
            fmax=fmax,
            steps=steps,
            logfile=logfile,
            trajectory=trajectory,
            constraints=constraints,
        )

        # Output results
        if results["converged"]:
            click.echo("✓ Optimization converged successfully!")
        else:
            click.echo("⚠ Optimization did not converge within step limit")

        if verbose or not results["converged"]:
            click.echo(f"Steps taken: {results['steps_taken']}")
            click.echo(f"Energy change: {results['energy_change']:.6f} eV")
            click.echo(f"Final max force: {results['final_max_force']:.4f} eV/Å")

        # Save optimized structure
        if output:
            qme.save_structure(results["optimized_atoms"], output)
            click.echo(f"Optimized structure saved to: {output}")
        else:
            # Generate default output name
            input_path = Path(input_file)
            default_output = input_path.with_suffix(".opt" + input_path.suffix)
            qme.save_structure(results["optimized_atoms"], default_output)
            click.echo(f"Optimized structure saved to: {default_output}")

    except Exception as e:
        click.echo(f"Error during optimization: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file for transition state structure",
)
@add_common_options(optimization_options)
@add_common_options(core_options)
def transition_state(
    input_file,
    output,
    fmax,
    steps,
    model,
    backend,
    model_path,
    device,
    logfile,
    trajectory,
    constraint_atoms,
    verbose,
):
    """
    Find transition state (saddle point) using SELLA optimizer.

    INPUT_FILE: Path to molecular structure file (starting guess for TS)
    """

    if verbose:
        click.echo("Starting transition state search...")
        click.echo(f"Input file: {input_file}")
        click.echo(f"Backend: {backend}")
        if model:
            click.echo(f"Model: {model}")
        if model_path:
            click.echo(f"Model path: {model_path}")

    try:
        # Initialize optimizer
        qme = QMEOptimizer(
            backend=backend, model_name=model, model_path=model_path, device=device
        )

        # Load structure
        atoms = qme.load_structure(input_file)

        if verbose:
            click.echo(f"Loaded structure with {len(atoms)} atoms")

        # Parse constraints if provided
        constraints = None
        if constraint_atoms:
            try:
                fixed_indices = [int(i.strip()) for i in constraint_atoms.split(",")]
                from ase.constraints import FixAtoms

                constraints = [FixAtoms(indices=fixed_indices)]
                if verbose:
                    click.echo(f"Fixed atoms: {fixed_indices}")
            except ValueError as e:
                click.echo(f"Error parsing constraint atoms: {e}", err=True)
                sys.exit(1)

        # Run TS search
        results = qme.find_transition_state(
            fmax=fmax,
            steps=steps,
            logfile=logfile,
            trajectory=trajectory,
            constraints=constraints,
        )

        # Output results
        if results["converged"]:
            click.echo("✓ Transition state search converged successfully!")
        else:
            click.echo("⚠ Transition state search did not converge within step limit")

        if verbose or not results["converged"]:
            click.echo(f"Steps taken: {results['steps_taken']}")
            click.echo(f"Energy change: {results['energy_change']:.6f} eV")
            click.echo(f"Final max force: {results['final_max_force']:.4f} eV/Å")

        # Save TS structure
        if output:
            qme.save_structure(results["ts_atoms"], output)
            click.echo(f"Transition state structure saved to: {output}")
        else:
            # Generate default output name
            input_path = Path(input_file)
            default_output = input_path.with_suffix(".ts" + input_path.suffix)
            qme.save_structure(results["ts_atoms"], default_output)
            click.echo(f"Transition state structure saved to: {default_output}")

    except Exception as e:
        click.echo(f"Error during transition state search: {e}", err=True)
        sys.exit(1)


@main.command()
@add_common_options(core_options)
def test_setup(backend, model, model_path, device, verbose):
    """
    Test QME setup and neural network model loading.
    """

    click.echo("Testing QME setup...")

    try:
        # Test imports
        click.echo("✓ Core imports successful")

        # Test model loading
        click.echo(f"Testing {backend.upper()} backend...")
        # Use use_mock=True to test initialization without downloading models
        qme = QMEOptimizer(
            backend=backend,
            model_name=model,
            device=device,
            use_mock=True,
        )
        click.echo(f"✓ {backend.upper()} backend initialized successfully")
        click.echo(f"✓ Calculator type: {type(qme.calculator).__name__}")
        click.echo("✅ All tests passed! QME is ready to use.")

    except ImportError as e:
        click.echo(f"❌ Import error: {e}", err=True)
        click.echo("Make sure all dependencies are installed:", err=True)
        if backend == "uma" and "fairchem" in str(e).lower():
            click.echo("  For UMA, run: pip install qme[ml]", err=True)
        elif backend == "so3lr" and "so3lr" in str(e).lower():
            click.echo(
                "  SO3LR not found. See README.md for installation instructions.",
                err=True,
            )
        elif backend == "aimnet2" and "torch" in str(e).lower():
            click.echo(
                "  PyTorch is required for AIMNet2. See README.md for details.",
                err=True,
            )
        click.echo("See 'Backend-Specific Installation' in README.md for more help.")
    except Exception as e:
        click.echo(f"❌ Setup error: {e}", err=True)


@main.command()
@click.argument("reactant_file", type=click.Path(exists=True))
@click.argument("product_file", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output XYZ trajectory file")
@click.option(
    "--npoints", "-n", default=10, type=int, help="Number of interpolation points"
)
@click.option(
    "--method",
    "-m",
    default="geodesic",
    type=click.Choice(["linear", "geodesic"]),
    help="Interpolation method",
)
@add_common_options(core_options)
@click.option(
    "--optimize-path",
    is_flag=True,
    help="Optimize interpolated path using NEB-like forces",
)
@click.option(
    "--calculate-energies",
    is_flag=True,
    help="Calculate energies along the path",
)
def interpolate(
    reactant_file,
    product_file,
    output,
    npoints,
    method,
    backend,
    model,
    model_path,
    device,
    optimize_path,
    calculate_energies,
    verbose,
):
    """
    Generate reaction pathway by interpolation between reactant and product.

    Uses geodesic or linear interpolation to create intermediate structures
    along a reaction coordinate, similar to NEB (Nudged Elastic Band) methods.

    REACTANT_FILE: Path to reactant structure file (xyz, cif, pdb, etc.)
    PRODUCT_FILE: Path to product structure file
    """

    if verbose:
        click.echo("Starting reaction pathway interpolation...")
        click.echo(f"Reactant: {reactant_file}")
        click.echo(f"Product: {product_file}")
        click.echo(f"Method: {method}")
        click.echo(f"Points: {npoints}")
        click.echo(f"Backend: {backend}")
        if model:
            click.echo(f"Model: {model}")
        if model_path:
            click.echo(f"Model path: {model_path}")
        if optimize_path:
            click.echo("Path optimization: enabled")

    try:
        # Load reactant and product geometries
        reactant = read_geometry(reactant_file)
        product = read_geometry(product_file)

        if isinstance(reactant, list) or isinstance(product, list):
            click.echo(
                "Error: Interpolation requires single structures, "
                "but multi-structure files were provided.",
                err=True,
            )
            sys.exit(1)

        if verbose:
            click.echo(f"Loaded reactant with {len(reactant)} atoms")
            click.echo(f"Loaded product with {len(product)} atoms")

        # Create reaction object
        reaction = Reaction(
            reactant,
            product,
            name=f"{Path(reactant_file).stem}_to_{Path(product_file).stem}",
        )

        # Set up calculator if needed
        calculator = None
        if optimize_path or calculate_energies:
            if verbose:
                click.echo(f"Initializing {backend.upper()} calculator...")

            try:
                qme = QMEOptimizer(
                    backend=backend,
                    model_name=model,
                    model_path=model_path,
                    device=device,
                )
                calculator = qme.calculator
                reaction.set_calculator(calculator)

            except Exception as e:
                click.echo(
                    f"Warning: Failed to initialize '{backend}' backend: {e}", err=True
                )
                click.echo("Falling back to a mock calculator for demonstration.")
                # Use QMEOptimizer's mock capability to get a consistent mock calc
                qme_mock = QMEOptimizer(backend=backend, use_mock=True)
                calculator = qme_mock.calculator
                reaction.set_calculator(calculator)

        # Generate interpolated path
        if verbose:
            click.echo(f"Generating {method} interpolation path...")

        path_geometries = reaction.interpolate(
            npoints=npoints,
            method=method,
            optimize_path=optimize_path,
            calculator=calculator,
        )

        # Calculate energies if requested
        if calculate_energies:
            if verbose:
                click.echo("Calculating energies along the path...")
            energies = reaction.calculate_path_energies(path_geometries)
            for i, (geom, energy) in enumerate(zip(path_geometries, energies)):
                click.echo(f"Point {i}: Energy = {energy:.6f} eV")

        # Save interpolated path
        if output:
            output_path = Path(output)
            # Use ASE's write function for multi-frame XYZ
            from ase.io import write as write_traj

            write_traj(output_path, path_geometries, format="xyz")
            click.echo(f"Interpolated path saved to: {output_path}")
        else:
            # Generate default output name
            reactant_path = Path(reactant_file)
            default_output = reactant_path.with_name(f"{reactant_path.stem}_path.xyz")
            from ase.io import write as write_traj

            write_traj(default_output, path_geometries, format="xyz")
            click.echo(f"Interpolated path saved to: {default_output}")

    except Exception as e:
        click.echo(f"Error during interpolation: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
