"""
Command Line Interface for QME (Quick Mechanistic Exploration).
"""

import sys
from pathlib import Path

import click

from .core import QMEOptimizer
from .geometry import read_gaussian_input, read_geometry
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


def _setup_optimization(
    input_file,
    backend,
    model,
    model_path,
    device,
    constraint_atoms,
    verbose,
    geometry=None,
):
    """Shared setup for minimize and transition_state commands."""
    from .config import config as qme_config

    effective_backend = backend or qme_config.config.default_backend

    if verbose:
        if input_file:
            click.echo(f"Starting optimization setup for: {input_file}")
        else:
            click.echo("Starting optimization setup from provided geometry.")
        click.echo(f"Backend: {effective_backend}")
        if model:
            click.echo(f"Model: {model}")
        if model_path:
            click.echo(f"Model path: {model_path}")

    try:
        # Initialize optimizer
        qme = QMEOptimizer(
            backend=effective_backend,
            model_name=model,
            model_path=model_path,
            device=device,
        )

        # Load structure
        if geometry:
            geometry.calc = qme.calculator
            qme.atoms = geometry
            atoms = geometry
        elif input_file:
            atoms = qme.load_structure(input_file)
        else:
            raise ValueError("Either input_file or geometry must be provided.")

        if verbose:
            click.echo(f"Loaded structure with {len(atoms)} atoms")

        # Parse constraints
        constraints = _parse_constraints(constraint_atoms, verbose)

        return qme, constraints

    except Exception as e:
        click.echo(f"Error during setup: {e}", err=True)
        sys.exit(1)


def _handle_optimization_results(results, qme, input_file, output, job_type, verbose):
    """Handles the output and saving of optimization results."""
    if results["converged"]:
        click.echo(
            f"✓ {job_type.replace('_', ' ').capitalize()} converged successfully!"
        )
    else:
        click.echo(
            f"⚠ {job_type.replace('_', ' ').capitalize()} did not converge within step limit"
        )

    if verbose or not results["converged"]:
        click.echo(f"Steps taken: {results['steps_taken']}")
        click.echo(f"Energy change: {results['energy_change']:.6f} eV")
        click.echo(f"Final max force: {results['final_max_force']:.4f} eV/Å")

    suffix = ".opt" if job_type == "minimize" else ".ts"
    atoms_key = "optimized_atoms" if job_type == "minimize" else "ts_atoms"

    output_path = output or _generate_output_path(input_file, suffix)
    qme.save_structure(results[atoms_key], output_path)
    click.echo(f"Structure saved to: {output_path}")


def _parse_constraints(constraint_atoms, verbose=False):
    """Parses comma-separated atom indices for constraints."""
    if not constraint_atoms:
        return None
    try:
        fixed_indices = [int(i.strip()) for i in constraint_atoms.split(",")]
        from ase.constraints import FixAtoms

        constraints = [FixAtoms(indices=fixed_indices)]
        if verbose:
            click.echo(f"Fixed atoms: {fixed_indices}")
        return constraints
    except ValueError as e:
        click.echo(f"Error parsing constraint atoms: {e}", err=True)
        sys.exit(1)


def _generate_output_path(input_path_str, suffix):
    """Generates a default output path."""
    input_path = Path(input_path_str)
    return input_path.with_suffix(suffix + input_path.suffix)


# Define common option groups
core_options = [
    click.option(
        "--backend",
        "-b",
        default=None,  # Will use config default
        type=click.Choice(["uma", "so3lr", "aimnet2", "mock"]),
        help="Backend to use (uma, so3lr, aimnet2, or mock for testing)",
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
    """Find minimum energy geometry using specified optimizer."""

    if verbose:
        click.echo("Starting minimum energy optimization...")

    try:
        qme, constraints = _setup_optimization(
            input_file, backend, model, model_path, device, constraint_atoms, verbose
        )

        # Run optimization
        results = qme.optimize_minimum(
            optimizer=optimizer,
            fmax=fmax,
            steps=steps,
            logfile=logfile,
            trajectory=trajectory,
            constraints=constraints,
        )

        _handle_optimization_results(
            results, qme, input_file, output, "minimize", verbose
        )

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
    """Find transition state (saddle point) using SELLA optimizer."""

    if verbose:
        click.echo("Starting transition state search...")

    try:
        qme, constraints = _setup_optimization(
            input_file, backend, model, model_path, device, constraint_atoms, verbose
        )

        # Run TS search
        results = qme.find_transition_state(
            fmax=fmax,
            steps=steps,
            logfile=logfile,
            trajectory=trajectory,
            constraints=constraints,
        )

        _handle_optimization_results(
            results, qme, input_file, output, "transition_state", verbose
        )

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


@main.command()
@click.option("--show", is_flag=True, help="Show current configuration")
@click.option(
    "--backend",
    type=click.Choice(["uma", "so3lr", "aimnet2"]),
    help="Set default backend",
)
@click.option("--model", type=str, help="Set default model for backend")
@click.option("--fmax", type=float, help="Set default force convergence criterion")
@click.option("--steps", type=int, help="Set default maximum optimization steps")
@click.option("--reset", is_flag=True, help="Reset configuration to defaults")
def config(show, backend, model, fmax, steps, reset):
    """Manage QME configuration settings."""
    from .config import config as qme_config

    if reset:
        # Remove config file and recreate with defaults
        import os

        config_file = qme_config._config_file
        if config_file.exists():
            os.remove(config_file)
            click.echo("Configuration reset to defaults.")
        qme_config.__init__()  # Reinitialize with defaults
        return

    if show or not any([backend, model, fmax, steps]):
        # Show current configuration
        click.echo("Current QME Configuration:")
        click.echo(f"  Default backend: {qme_config.config.default_backend}")
        click.echo(f"  Default optimizer: {qme_config.config.default_optimizer}")
        if qme_config.config.default_models:
            click.echo("  Default models:")
            for backend_name, model_name in qme_config.config.default_models.items():
                click.echo(f"    {backend_name}: {model_name}")
        click.echo(f"  Default fmax: {qme_config.config.default_fmax}")
        click.echo(f"  Default steps: {qme_config.config.default_steps}")
        click.echo(
            f"  Preferred device: {qme_config.config.preferred_device or 'auto'}"
        )
        click.echo(f"  Warnings enabled: {qme_config.config.enable_warnings}")
        return

    # Update configuration
    if backend:
        qme_config.config.default_backend = backend
        click.echo(f"Default backend set to: {backend}")

    if model and backend:
        if qme_config.config.default_models is None:
            qme_config.config.default_models = {}
        qme_config.config.default_models[backend] = model
        click.echo(f"Default model for {backend} set to: {model}")
    elif model:
        click.echo("Error: --model requires --backend to be specified", err=True)
        sys.exit(1)

    if fmax:
        qme_config.config.default_fmax = fmax
        click.echo(f"Default fmax set to: {fmax}")

    if steps:
        qme_config.config.default_steps = steps
        click.echo(f"Default steps set to: {steps}")

    # Save configuration
    qme_config.save_config()
    click.echo("Configuration saved.")


@main.command()
def info():
    """Show system and dependency information."""
    from .config import config as qme_config
    from .dependencies import deps

    click.echo("QME System Information")
    click.echo("=" * 30)

    # Backend availability
    backends = {
        "UMA (FairChem)": deps.has("fairchem"),
        "SO3LR": deps.has("so3lr"),
        "AIMNET2": deps.has("aimnet2"),
        "SELLA (TS)": deps.has("sella"),
        "PyTorch": deps.has("torch"),
    }

    click.echo("\nBackend Availability:")
    for backend, available in backends.items():
        status = "✓" if available else "✗"
        click.echo(f"  {status} {backend}")

    # Configuration
    click.echo(f"\nDefault Backend: {qme_config.config.default_backend}")
    click.echo(f"Config Location: {qme_config._config_file}")

    # Device information
    device = qme_config.get_device_preference()
    click.echo(f"Preferred Device: {device or 'auto-detect'}")

    if deps.has("torch"):
        torch = deps.get("torch")
        if torch.cuda.is_available():
            click.echo(f"CUDA Available: ✓ ({torch.cuda.get_device_name()})")
        else:
            click.echo("CUDA Available: ✗")


@main.command()
@click.argument("gaussian_file", type=click.Path(exists=True))
@click.option(
    "--output", "-o", type=click.Path(), help="Output file for optimized structure"
)
@click.option(
    "--optimizer",
    "-opt",
    default="BFGS",
    type=click.Choice(["BFGS", "LBFGS", "FIRE"]),
    help="Optimizer to use for minimization (if applicable)",
)
@add_common_options(core_options)
@add_common_options(optimization_options)
def from_gaussian(
    gaussian_file,
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
    """
    Run a QME optimization starting from a Gaussian input file.
    The job type (minimize or transition_state) is detected automatically.
    """
    if verbose:
        click.echo(f"Reading Gaussian input file: {gaussian_file}")

    try:
        geometry, job_type = read_gaussian_input(gaussian_file)

        if verbose:
            click.echo(f"Detected job type: {job_type}")
            click.echo(f"Loaded geometry: {geometry}")

        qme, constraints = _setup_optimization(
            None,
            backend,
            model,
            model_path,
            device,
            constraint_atoms,
            verbose,
            geometry=geometry,
        )

        if job_type == "minimize":
            if verbose:
                click.echo("Starting minimum energy optimization...")
            results = qme.optimize_minimum(
                optimizer=optimizer,
                fmax=fmax,
                steps=steps,
                logfile=logfile,
                trajectory=trajectory,
                constraints=constraints,
            )
            _handle_optimization_results(
                results, qme, gaussian_file, output, "minimize", verbose
            )

        elif job_type == "transition_state":
            if verbose:
                click.echo("Starting transition state search...")
            results = qme.find_transition_state(
                fmax=fmax,
                steps=steps,
                logfile=logfile,
                trajectory=trajectory,
                constraints=constraints,
            )
            _handle_optimization_results(
                results, qme, gaussian_file, output, "transition_state", verbose
            )

    except Exception as e:
        click.echo(f"An error occurred: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
