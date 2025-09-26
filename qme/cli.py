"""
Command Line Interface for QME (Quick Mechanistic Exploration).
"""

import sys
from pathlib import Path

import click

from .cli_helpers import (
    calculate_frequencies_if_requested,
    handle_cli_error,
    handle_optimization_results,
    setup_optimization,
)
from .core import QMEOptimizer
from .geometry import read_gaussian_input, read_geometry
from .reaction import Reaction


@click.group()
@click.version_option(version="0.1.0")
def main():
    """
    QME: Quick mechanistic exploration using machine learning potentials.

    Supports multiple neural network backends including UMA, SO3LR, AIMNET2, and MACE potentials
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


def parse_constraints(
    fix_atoms=None, harmonic_constraints=None, spring_constant=10.0, verbose=False
):
    """Parse constraint options and return ASE constraints."""
    constraints = []

    # Handle fixed atoms
    if fix_atoms:
        try:
            fixed_indices = [int(i.strip()) for i in fix_atoms.split(",")]
            from ase.constraints import FixAtoms

            constraints.append(FixAtoms(indices=fixed_indices))
            if verbose:
                click.echo(f"Fixed atoms: {fixed_indices}")
        except ValueError as e:
            click.echo(f"Error parsing fixed atoms: {e}", err=True)
            sys.exit(1)

    # Handle harmonic constraints
    if harmonic_constraints:
        try:
            # For now, implement as a simple position constraint using Hookean
            # Format expected: "0,1,2" for atoms to constrain harmonically
            harmonic_indices = [int(i.strip()) for i in harmonic_constraints.split(",")]
            from ase.constraints import Hookean

            # Note: This is a simplified implementation
            # In practice, you'd want to set up position-based constraints
            # using the initial geometry as reference
            if verbose:
                click.echo(
                    f"Harmonic constraints on atoms: {harmonic_indices} (k={spring_constant} eV/Å²)"
                )
            # For now, we'll implement this as a placeholder
            # Real implementation would require access to initial geometry

        except ValueError as e:
            click.echo(f"Error parsing harmonic constraints: {e}", err=True)
            sys.exit(1)

    return constraints if constraints else None


# Define common option groups
core_options = [
    click.option(
        "--backend",
        "-b",
        default=None,  # Will use config default
        type=click.Choice(["uma", "so3lr", "aimnet2", "mace", "mock"]),
        help="Backend to use (uma, so3lr, aimnet2, mace, or mock for testing) - default: uma",
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
    click.option(
        "--charge",
        "-c",
        type=int,
        default=0,
        help="Total charge of the system (default: 0))",
    ),
    click.option(
        "--spin",
        "-s",
        type=int,
        default=1,
        help="Spin multiplicity (default: 1))",
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
        "--max-steps",
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
        "--fix-atoms",
        type=str,
        help="Comma-separated list of atom indices to fix (e.g., '0,1,2')",
    ),
    click.option(
        "--harmonic-constraints",
        type=str,
        help="Comma-separated list of atom indices for harmonic position constraints "
        "(e.g., '0,1,2')",
    ),
    click.option(
        "--spring-constant",
        "-k",
        default=0.1,
        type=float,
        help="Spring constant for harmonic constraints (eV/Å²)",
    ),
    click.option(
        "--frequencies",
        is_flag=True,
        help="Calculate frequencies after optimization",
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
    default="LBFGS",
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
    charge,
    spin,
    logfile,
    trajectory,
    fix_atoms,
    harmonic_constraints,
    spring_constant,
    frequencies,
    verbose,
):
    """Find minimum energy geometry using specified optimizer."""

    if verbose:
        click.echo("Starting minimum energy optimization...")

    try:
        # Parse constraints
        constraints = parse_constraints(
            fix_atoms, harmonic_constraints, spring_constant, verbose
        )

        qme, _ = setup_optimization(
            input_file,
            backend,
            model,
            model_path,
            device,
            None,  # constraint_atoms (deprecated)
            verbose,
            charge,
            spin,
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

        # Calculate frequencies if requested
        calculate_frequencies_if_requested(
            qme, results, frequencies, verbose, is_ts=False
        )

        handle_optimization_results(
            results, qme, input_file, output, "minimize", verbose
        )

    except Exception as e:
        handle_cli_error(e, "optimization", verbose)


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
    charge,
    spin,
    logfile,
    trajectory,
    fix_atoms,
    harmonic_constraints,
    spring_constant,
    frequencies,
    verbose,
):
    """Find transition state (saddle point) using SELLA optimizer."""

    if verbose:
        click.echo("Starting transition state search...")

    try:
        # Parse constraints
        constraints = parse_constraints(
            fix_atoms, harmonic_constraints, spring_constant, verbose
        )

        qme, _ = setup_optimization(
            input_file,
            backend,
            model,
            model_path,
            device,
            None,  # constraint_atoms (deprecated)
            verbose,
            charge,
            spin,
        )

        # Run TS search
        results = qme.find_transition_state(
            fmax=fmax,
            steps=steps,
            logfile=logfile,
            trajectory=trajectory,
            constraints=constraints,
        )

        # Calculate frequencies if requested
        calculate_frequencies_if_requested(
            qme, results, frequencies, verbose, is_ts=True
        )

        handle_optimization_results(
            results, qme, input_file, output, "transition_state", verbose
        )

    except Exception as e:
        handle_cli_error(e, "transition state search", verbose)


@main.command(name="ts-from-endpoints")
@click.argument("reactant_file", type=click.Path(exists=True))
@click.argument("product_file", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file for transition state structure",
)
@click.option(
    "--npoints", "-n", default=10, type=int, help="Number of interpolation points"
)
@click.option(
    "--interp-method",
    "-i",
    default="geodesic",
    type=click.Choice(["linear", "geodesic"]),
    help="Interpolation method",
)
@add_common_options(optimization_options)
@add_common_options(core_options)
def ts_from_endpoints(
    reactant_file,
    product_file,
    output,
    npoints,
    interp_method,
    fmax,
    steps,
    model,
    backend,
    model_path,
    device,
    charge,
    spin,
    logfile,
    trajectory,
    fix_atoms,
    harmonic_constraints,
    spring_constant,
    frequencies,
    verbose,
):
    """Find transition state by interpolating between reactant and product structures."""

    if verbose:
        click.echo("Starting transition state search from endpoints...")
        click.echo(f"Reactant: {reactant_file}")
        click.echo(f"Product: {product_file}")
        click.echo(f"Interpolation method: {interp_method}")
        click.echo(f"Number of points: {npoints}")

    try:
        # Parse constraints
        constraints = parse_constraints(
            fix_atoms, harmonic_constraints, spring_constant, verbose
        )

        qme, _ = setup_optimization(
            None,  # No single input file
            backend,
            model,
            model_path,
            device,
            None,  # constraint_atoms (deprecated)
            verbose,
            charge,
            spin,
        )

        # Load structures
        reactant_atoms = qme.load_structure(reactant_file)
        product_atoms = qme.load_structure(product_file)

        if verbose:
            click.echo(f"Loaded reactant: {len(reactant_atoms)} atoms")
            click.echo(f"Loaded product: {len(product_atoms)} atoms")

        # Generate interpolated path
        if verbose:
            click.echo(f"Generating {npoints} interpolated structures...")

        from qme.reaction import Reaction

        reaction = Reaction(reactant_atoms, product_atoms)
        path = reaction.interpolate(npoints=npoints, method=interp_method)

        if verbose:
            click.echo(f"✓ Generated interpolated path with {len(path)} structures")

        # Find the highest energy structure as TS guess
        if verbose:
            click.echo("Evaluating energies along path to find TS guess...")

        energies = []
        for i, structure in enumerate(path):
            structure.calc = qme.calculator
            energy = structure.get_potential_energy()
            energies.append(energy)
            if verbose:
                click.echo(f"  Point {i}: {energy:.6f} eV")

        # Find maximum energy structure
        max_idx = energies.index(max(energies))
        ts_guess = path[max_idx]

        if verbose:
            click.echo(
                f"✓ Using structure {max_idx} as TS guess (energy: {energies[max_idx]:.6f} eV)"
            )

        # Set up the TS guess for optimization
        qme.atoms = ts_guess

        # Run TS optimization
        if verbose:
            click.echo("Optimizing transition state guess...")

        results = qme.find_transition_state(
            fmax=fmax,
            steps=steps,
            logfile=logfile,
            trajectory=trajectory,
            constraints=constraints,
        )

        # Calculate frequencies if requested
        if frequencies and results.get("converged", False):
            if verbose:
                click.echo("Calculating frequencies...")
            try:
                freq_results = qme.calculate_frequencies(
                    atoms=results["optimized_atoms"], delta=0.01
                )
                results["frequencies"] = freq_results
                click.echo(
                    f"✓ Calculated {len(freq_results['frequencies'])} vibrational frequencies"
                )

                # For TS, also check for imaginary frequencies
                imag_freqs = [f for f in freq_results["frequencies"] if f < 0]
                if imag_freqs:
                    click.echo(
                        f"✓ Found {len(imag_freqs)} imag. freq(s): {imag_freqs[0]:.1f} cm⁻¹"
                    )
                else:
                    click.echo(
                        "⚠ No imaginary frequencies found - structure may not be a transition state"
                    )
            except Exception as e:
                click.echo(f"⚠ Frequency calculation failed: {e}")

        handle_optimization_results(
            results, qme, None, output, "ts_from_endpoints", verbose
        )

    except Exception as e:
        click.echo(f"Error during TS search from endpoints: {e}", err=True)
        sys.exit(1)


@main.command()
@add_common_options(core_options)
def test_setup(backend, model, model_path, device, charge, spin, verbose):
    """
    Test QME setup and neural network model loading.
    """

    click.echo("Testing QME setup...")

    try:
        # Test imports
        click.echo("✓ Core imports successful")

        # Test model loading
        click.echo(f"Testing {backend.upper()} backend...")
        # Initialize backend (use mock for testing if needed)
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
        elif backend == "mace" and (
            "mace" in str(e).lower() or "torch" in str(e).lower()
        ):
            click.echo(
                "  MACE and PyTorch are required. Install with: pip install mace-torch",
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
    "--interp-method",
    "-i",
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
    interp_method,
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
        click.echo(f"Method: {interp_method}")
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
                # Use mock backend for testing
                qme_mock = QMEOptimizer(backend="mock")
                calculator = qme_mock.calculator
                reaction.set_calculator(calculator)

        # Generate interpolated path
        if verbose:
            click.echo(f"Generating {interp_method} interpolation path...")

        path_geometries = reaction.interpolate(
            npoints=npoints,
            method=interp_method,
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
    type=click.Choice(["uma", "so3lr", "aimnet2", "mace"]),
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
    fix_atoms,
    harmonic_constraints,
    spring_constant,
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

        # Parse constraints
        constraints = parse_constraints(
            fix_atoms, harmonic_constraints, spring_constant, verbose
        )

        qme, _ = setup_optimization(
            None,
            backend,
            model,
            model_path,
            device,
            None,  # constraint_atoms (deprecated)
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
            handle_optimization_results(
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
            handle_optimization_results(
                results, qme, gaussian_file, output, "transition_state", verbose
            )

    except Exception as e:
        click.echo(f"An error occurred: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file for frequency analysis results",
)
@click.option(
    "--delta", default=0.01, help="Displacement for finite differences (Å)", type=float
)
@click.option(
    "--temperature",
    "-T",
    default=298.15,
    help="Temperature for thermodynamic analysis (K)",
    type=float,
)
@click.option(
    "--hessian-method",
    default="auto",
    type=click.Choice(["auto", "direct", "finite_differences"]),
    help="Hessian calculation method",
)
@click.option("--verify-ts", is_flag=True, help="Verify structure is transition state")
@click.option("--save-hessian", is_flag=True, help="Save Hessian matrix in output")
@click.option(
    "--atoms",
    type=str,
    help="Comma-separated list of atom indices to include (0-based)",
)
@add_common_options(core_options)
def frequencies(
    input_file,
    output,
    delta,
    temperature,
    hessian_method,
    verify_ts,
    save_hessian,
    atoms,
    model,
    backend,
    model_path,
    device,
    verbose,
):
    """Calculate vibrational frequencies and normal modes."""

    if verbose:
        click.echo("Starting frequency analysis...")
        click.echo(f"Method: {hessian_method}")
        click.echo(f"Temperature: {temperature} K")
        if atoms:
            click.echo(f"Analyzing atoms: {atoms}")

    try:
        qme, _ = setup_optimization(
            input_file, backend, model, model_path, device, None, verbose
        )

        # Parse atom indices if provided
        indices = None
        if atoms:
            try:
                indices = [int(i.strip()) for i in atoms.split(",")]
                if verbose:
                    click.echo(f"Including {len(indices)} atoms in analysis")
            except ValueError:
                click.echo(
                    "Error: Invalid atom indices. Use comma-separated integers.",
                    err=True,
                )
                sys.exit(1)

        # Calculate frequencies
        results = qme.calculate_frequencies(
            delta=delta,
            method=hessian_method,
            temperature=temperature,
            save_hessian=save_hessian,
            indices=indices,
        )

        # Display results
        frequencies = results["frequencies"]
        n_vib_modes = len(frequencies)
        n_imaginary = sum(1 for f in frequencies if f < 0)

        click.echo("\n✓ Frequency analysis completed!")
        click.echo(f"Found {n_vib_modes} vibrational modes")
        click.echo(f"Imaginary frequencies: {n_imaginary}")
        click.echo(f"Zero-point energy: {results['zero_point_energy']:.6f} eV")

        # Show frequency summary
        if frequencies:
            if n_imaginary > 0:
                imag_freqs = [f for f in frequencies if f < 0]
                click.echo(
                    f"Imaginary: {', '.join([f'{f:.1f}' for f in imag_freqs])} cm⁻¹"
                )

            real_freqs = [f for f in frequencies if f > 0]
            if real_freqs:
                click.echo(
                    f"Real frequencies: {real_freqs[0]:.1f} to {real_freqs[-1]:.1f} cm⁻¹"
                )

        # Transition state verification if requested
        if verify_ts:
            ts_results = results["ts_analysis"]
            click.echo("\nTransition State Verification:")
            click.echo(f"  {ts_results['assessment']}")
            if ts_results["is_transition_state"]:
                imag_freq = ts_results["imaginary_frequencies"][0]
                click.echo(
                    f"  ✓ Valid TS with imaginary frequency: {imag_freq:.1f} cm⁻¹"
                )
            elif n_imaginary == 0:
                click.echo("  ✓ Structure is a minimum (no imaginary frequencies)")
            else:
                click.echo(
                    f"  ⚠ Higher-order saddle point ({n_imaginary} imaginary frequencies)"
                )

        # Save results if output file specified
        if output:
            import json

            # Convert numpy arrays to lists for JSON serialization
            json_results = {
                "input_file": str(input_file),
                "method": results["method_used"],
                "temperature": temperature,
                "frequencies_cm-1": results["frequencies"],
                "zero_point_energy_eV": results["zero_point_energy"],
                "thermodynamic_properties": results["thermodynamic_properties"],
                "ts_analysis": results["ts_analysis"],
                "n_vibrational_modes": len(results["frequencies"]),
                "calculation_parameters": {
                    "delta": delta,
                    "method": hessian_method,
                    "backend": backend,
                    "model": model,
                    "indices": results["indices"],
                },
            }

            if save_hessian and "hessian" in results:
                json_results["hessian"] = results["hessian"]

            with open(output, "w") as f:
                json.dump(json_results, f, indent=2)

            click.echo(f"Results saved to: {output}")

    except Exception as e:
        click.echo(f"Error during frequency analysis: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
