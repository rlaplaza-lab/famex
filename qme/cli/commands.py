"""CLI command implementations moved from qme/cli.py."""

import sys
from pathlib import Path

import click

from .helpers import (
    _write_standard_xyz,
    calculate_frequencies_if_requested,
    display_backend_info,
    generate_output_path,
    handle_cli_error,
    handle_optimization_results,
    parse_enhanced_constraints,
    setup_optimization,
    validate_input_files,
)
from .main import main
from .options import add_common_options, get_core_options, get_optimization_options


@main.command(name="opt")
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output", "-o", type=click.Path(), help="Output file for optimized structure"
)
@click.option(
    "--optimizer",
    "-O",
    default="LBFGS",
    show_default=True,
    type=click.Choice(["BFGS", "LBFGS", "FIRE"]),
    help="Optimizer to use for minimization",
)
@add_common_options(get_core_options)
@add_common_options(get_optimization_options)
def opt(
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
        # Combine legacy constraint options into a single enhanced constraint spec
        constraint_spec = None
        if fix_atoms:
            constraint_spec = f"fix {fix_atoms}"
        elif harmonic_constraints:
            constraint_spec = (
                f"harmonic_position {harmonic_constraints} k={spring_constant}"
            )

        qme, constraints = setup_optimization(
            input_file,
            backend,
            model,
            model_path,
            device,
            constraint_spec,
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

        handle_optimization_results(results, qme, input_file, output, "opt", verbose)

    except Exception as e:
        handle_cli_error(e, "optimization", verbose)


# ... remaining commands (tsopt, neb, test_setup, interpolate, config, info, from_gaussian,
# frequencies) will be imported when needed; for brevity we register a subset here


@main.command(name="info")
def info():
    """Show system and dependency information."""
    from ..utils import deps
    from ..utils.settings import config as qme_config

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
        status = "\u2713" if available else "\u2717"
        click.echo(f"  {status} {backend}")

    # Configuration
    click.echo(f"\nDefault Backend: {qme_config.get_backend()}")
    if qme_config.has_config_file():
        click.echo(f"Config File: {qme_config.config_file_path()}")
    else:
        click.echo("Config File: None (using built-in defaults)")

    # Device information
    device = qme_config.get_device()
    click.echo(f"Device: {device or 'auto-detect'}")

    if deps.has("torch"):
        torch = deps.get("torch")
        if getattr(torch, "cuda", None) and torch.cuda.is_available():
            click.echo(f"CUDA Available: \u2713 ({torch.cuda.get_device_name()})")
        else:
            click.echo("CUDA Available: \u2717")


@main.command(name="tsopt")
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file for transition state structure",
)
@add_common_options(get_optimization_options)
@add_common_options(get_core_options)
def tsopt(
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
        # Combine legacy constraint options into a single enhanced constraint spec
        constraint_spec = None
        if fix_atoms:
            constraint_spec = f"fix {fix_atoms}"
        elif harmonic_constraints:
            constraint_spec = (
                f"harmonic_position {harmonic_constraints} k={spring_constant}"
            )

        qme, constraints = setup_optimization(
            input_file,
            backend,
            model,
            model_path,
            device,
            constraint_spec,
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

        handle_optimization_results(results, qme, input_file, output, "tsopt", verbose)

    except Exception as e:
        handle_cli_error(e, "transition state search", verbose)


@main.command(name="neb")
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
@add_common_options(get_optimization_options)
@add_common_options(get_core_options)
def neb(
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
    """Find transition state using NEB method (currently interpolation between reactant and
    product)."""
    if verbose:
        click.echo("Starting transition state search from endpoints...")
        click.echo(f"Reactant: {reactant_file}")
        click.echo(f"Product: {product_file}")
        click.echo(f"Interpolation method: {interp_method}")
        click.echo(f"Number of points: {npoints}")

    try:
        # Combine legacy constraint options into an enhanced constraint spec
        constraint_spec = None
        if fix_atoms:
            constraint_spec = f"fix {fix_atoms}"
        elif harmonic_constraints:
            constraint_spec = (
                f"harmonic_position {harmonic_constraints} k={spring_constant}"
            )

        qme, constraints = setup_optimization(
            None,  # No single input file
            backend,
            model,
            model_path,
            device,
            constraint_spec,
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
            click.echo(
                f"\u2713 Generated interpolated path with {len(path)} structures"
            )

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
                f"\u2713 Using structure {max_idx} as TS guess "
                f"(energy: {energies[max_idx]:.6f} eV)"
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
            calculate_frequencies_if_requested(
                qme, results, frequencies, verbose, is_ts=True
            )

        handle_optimization_results(
            results, qme, None, output, "ts_from_endpoints", verbose
        )

    except Exception as e:
        handle_cli_error(e, "TS search from endpoints", verbose)


@main.command()
@add_common_options(get_core_options)
def test_setup(backend, model, model_path, device, charge, spin, verbose):
    """
    Test QME setup and neural network model loading.
    """
    # Import only when needed
    from ..core import QMEOptimizer

    click.echo("Testing QME setup...")

    try:
        # Test imports
        click.echo("\u2713 Core imports successful")

        # Test model loading
        click.echo(f"Testing {backend.upper()} backend...")
        # Initialize backend (use mock for testing if needed)
        qme = QMEOptimizer(
            backend=backend,
            model_name=model,
            device=device,
        )
        click.echo(f"\u2713 {backend.upper()} backend initialized successfully")
        click.echo(f"\u2713 Calculator type: {type(qme.calculator).__name__}")
        click.echo("\u2705 All tests passed! QME is ready to use.")

    except ImportError as e:
        click.echo(f"\u274c Import error: {e}", err=True)
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
        click.echo(f"\u274c Setup error: {e}", err=True)


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
@add_common_options(get_core_options)
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
    """
    # Import only when needed
    from ..core import QMEOptimizer
    from ..geometry import read_geometry
    from ..reaction import Reaction

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
        handle_cli_error(e, "interpolation", verbose)


@main.command()
@click.option("--show", is_flag=True, help="Show current configuration and defaults")
@click.option(
    "--create-config",
    is_flag=True,
    help="Create a qme.json config file in current directory",
)
def config(show, create_config):
    """Show configuration defaults and optionally create a config file.

    QME uses visible defaults for all CLI options. A qme.json config file in the
    current directory can override these defaults.
    """
    from ..utils.settings import config as qme_config

    if create_config:
        config_path = qme_config.config_file_path()
        if config_path.exists():
            click.echo(f"Config file {config_path} already exists!")
            return

        # Create a sample config file with current defaults
        sample_config = {
            "backend": qme_config.defaults.backend,
            "optimizer": qme_config.defaults.optimizer,
            "fmax": qme_config.defaults.fmax,
            "steps": qme_config.defaults.steps,
            "models": qme_config.defaults.models,
        }

        import json

        with open(config_path, "w") as f:
            json.dump(sample_config, f, indent=2)

        click.echo(f"Created sample config file: {config_path}")
        click.echo("Edit this file to customize your defaults.")
        return

    # Show current configuration
    click.echo("QME Configuration:")
    click.echo("=" * 30)

    # Show whether config file is being used
    if qme_config.has_config_file():
        click.echo(f"\u2713 Using config file: {qme_config.config_file_path()}")
    else:
        click.echo("\u2713 Using built-in defaults (no config file found)")

    click.echo("\nCurrent values:")
    click.echo(f"  Backend: {qme_config.get_backend()}")
    click.echo(f"  Optimizer: {qme_config.get_optimizer()}")
    click.echo(f"  Force convergence (fmax): {qme_config.get_fmax()}")
    click.echo(f"  Max steps: {qme_config.get_steps()}")
    click.echo(f"  Device: {qme_config.get_device() or 'auto-detect'}")

    models = qme_config.get("models")
    if models:
        click.echo("  Default models:")
        for backend_name, model_name in models.items():
            click.echo(f"    {backend_name}: {model_name}")

    if not qme_config.has_config_file():
        click.echo("\nTo customize defaults, run: qme config --create-config")


@main.command()
@click.argument("gaussian_file", type=click.Path(exists=True))
@click.option(
    "--output", "-o", type=click.Path(), help="Output file for optimized structure"
)
@click.option(
    "--optimizer",
    "-O",
    default="BFGS",
    show_default=True,
    type=click.Choice(["BFGS", "LBFGS", "FIRE"]),
    help="Optimizer to use for minimization (if applicable)",
)
@add_common_options(get_core_options)
@add_common_options(get_optimization_options)
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
    # Import only when needed
    from ..geometry import read_gaussian_input

    if verbose:
        click.echo(f"Reading Gaussian input file: {gaussian_file}")

    try:
        geometry, job_type = read_gaussian_input(gaussian_file)

        if verbose:
            click.echo(f"Detected job type: {job_type}")
            click.echo(f"Loaded geometry: {geometry}")

        # Combine legacy constraint options into an enhanced constraint spec
        constraint_spec = None
        if fix_atoms:
            constraint_spec = f"fix {fix_atoms}"
        elif harmonic_constraints:
            constraint_spec = (
                f"harmonic_position {harmonic_constraints} k={spring_constant}"
            )

        qme, constraints = setup_optimization(
            None,
            backend,
            model,
            model_path,
            device,
            constraint_spec,
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
        handle_cli_error(e, "from_gaussian", verbose)


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file for frequency analysis results",
)
@click.option(
    "--delta",
    default=0.01,
    help="Displacement for finite differences (\u00c5)",
    type=float,
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
@add_common_options(get_core_options)
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
                handle_cli_error(
                    ValueError("Invalid atom indices"), "frequencies", True
                )

        # Calculate frequencies
        results = qme.calculate_frequencies(
            delta=delta,
            method=hessian_method,
            temperature=temperature,
            save_hessian=save_hessian,
            indices=indices,
        )

        # Display results
        frequencies_list = results["frequencies"]
        n_vib_modes = len(frequencies_list)
        n_imaginary = sum(1 for f in frequencies_list if f < 0)

        click.echo("\n\u2713 Frequency analysis completed!")
        click.echo(f"Found {n_vib_modes} vibrational modes")
        click.echo(f"Imaginary frequencies: {n_imaginary}")
        click.echo(f"Zero-point energy: {results['zero_point_energy']:.6f} eV")

        # Show frequency summary
        if frequencies_list:
            if n_imaginary > 0:
                imag_freqs = [f for f in frequencies_list if f < 0]
                click.echo(
                    f"Imaginary: {', '.join([f'{f:.1f}' for f in imag_freqs])} cm\u207b\u00b9"
                )

            real_freqs = [f for f in frequencies_list if f > 0]
            if real_freqs:
                click.echo(
                    f"Real frequencies: {real_freqs[0]:.1f} to "
                    f"{real_freqs[-1]:.1f} cm\u207b\u00b9"
                )

        # Transition state verification if requested
        if verify_ts:
            ts_results = results["ts_analysis"]
            click.echo("\nTransition State Verification:")
            click.echo(f"  {ts_results['assessment']}")
            if ts_results["is_transition_state"]:
                imag_freq = ts_results["imaginary_frequencies"][0]
                click.echo(
                    f"  \u2713 Valid TS with imaginary frequency: {imag_freq:.1f} cm\u207b\u00b9"
                )
            elif n_imaginary == 0:
                click.echo("  \u2713 Structure is a minimum (no imaginary frequencies)")
            else:
                click.echo(
                    f"  \u26a0 Higher-order saddle point ({n_imaginary} imaginary frequencies)"
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
        handle_cli_error(e, "frequency analysis", verbose=True)
