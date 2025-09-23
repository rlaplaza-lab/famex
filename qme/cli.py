"""
Command Line Interface for QME (Quick Mechanistic Exploration).
"""

import sys
from pathlib import Path

import click

from .core import QMEOptimizer
from .uma_potential import get_uma_calculator


@click.group()
@click.version_option(version="0.1.0")
def main():
    """
    QME: Quick mechanistic exploration using machine learning potentials.

    Combines ASE and SELLA optimizers with UMA potentials for molecular
    geometry optimization and transition state searches.
    """
    pass


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
@click.option(
    "--fmax", "-f", default=0.01, type=float, help="Force convergence criterion (eV/Å)"
)
@click.option(
    "--steps", "-s", default=200, type=int, help="Maximum number of optimization steps"
)
@click.option("--model", "-m", default="uma-4m", type=str, help="UMA model name to use")
@click.option(
    "--device",
    "-d",
    type=click.Choice(["cpu", "cuda"]),
    help="Device for computations (auto-detected if not specified)",
)
@click.option("--logfile", type=click.Path(), help="Log file for optimization output")
@click.option(
    "--trajectory", type=click.Path(), help="Trajectory file to save optimization steps"
)
@click.option(
    "--constraint-atoms",
    type=str,
    help="Comma-separated list of atom indices to fix (0-based)",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def minimize(
    input_file,
    output,
    optimizer,
    fmax,
    steps,
    model,
    device,
    logfile,
    trajectory,
    constraint_atoms,
    verbose,
):
    """
    Find minimum energy geometry using specified optimizer.

    INPUT_FILE: Path to molecular structure file (xyz, cif, pdb, etc.)
    """

    if verbose:
        click.echo(f"Starting minimum energy optimization...")
        click.echo(f"Input file: {input_file}")
        click.echo(f"Optimizer: {optimizer}")
        click.echo(f"UMA model: {model}")

    try:
        # Initialize optimizer
        qme = QMEOptimizer(model_name=model, device=device)

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
@click.option(
    "--fmax", "-f", default=0.01, type=float, help="Force convergence criterion (eV/Å)"
)
@click.option(
    "--steps", "-s", default=200, type=int, help="Maximum number of optimization steps"
)
@click.option("--model", "-m", default="uma-4m", type=str, help="UMA model name to use")
@click.option(
    "--device",
    "-d",
    type=click.Choice(["cpu", "cuda"]),
    help="Device for computations (auto-detected if not specified)",
)
@click.option("--logfile", type=click.Path(), help="Log file for optimization output")
@click.option(
    "--trajectory", type=click.Path(), help="Trajectory file to save optimization steps"
)
@click.option(
    "--constraint-atoms",
    type=str,
    help="Comma-separated list of atom indices to fix (0-based)",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def transition_state(
    input_file,
    output,
    fmax,
    steps,
    model,
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
        click.echo(f"Starting transition state search...")
        click.echo(f"Input file: {input_file}")
        click.echo(f"UMA model: {model}")

    try:
        # Initialize optimizer
        qme = QMEOptimizer(model_name=model, device=device)

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
@click.option(
    "--model", "-m", default="uma-4m", type=str, help="UMA model name to test"
)
@click.option(
    "--device", "-d", type=click.Choice(["cpu", "cuda"]), help="Device for computations"
)
def test_setup(model, device):
    """
    Test QME setup and UMA model loading.
    """

    click.echo("Testing QME setup...")

    try:
        # Test imports
        click.echo("✓ Core imports successful")

        # Test UMA model loading
        click.echo(f"Loading UMA model: {model}")
        calculator = get_uma_calculator(model_name=model, device=device)
        click.echo("✓ UMA model loaded successfully")

        # Test optimizer initialization
        qme = QMEOptimizer(calculator=calculator)
        click.echo("✓ QME optimizer initialized")

        click.echo("✅ All tests passed! QME is ready to use.")

    except ImportError as e:
        click.echo(f"❌ Import error: {e}", err=True)
        click.echo("Make sure all dependencies are installed:", err=True)
        click.echo("  pip install ase sella torch fairchem-core", err=True)
    except Exception as e:
        click.echo(f"❌ Setup error: {e}", err=True)


if __name__ == "__main__":
    main()
