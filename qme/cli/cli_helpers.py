"""
Helper functions for QME CLI to reduce code duplication and improve maintainability.
"""

import sys
from pathlib import Path

import click

# Lazy import to avoid loading heavy MLP backends for simple commands


def _write_standard_xyz(atoms, filename):
    """
    Write atoms to XYZ file in standard format (just atom types and coordinates).

    Parameters:
    -----------
    atoms : ase.Atoms
        Atoms object to write
    filename : str
        Output filename
    """
    with open(filename, "w") as f:
        f.write(f"{len(atoms)}\n")
        f.write("\n")  # Comment line (empty)
        for atom in atoms:
            symbol = atom.symbol
            x, y, z = atom.position
            f.write(f"{symbol:2s} {x:12.6f} {y:12.6f} {z:12.6f}\n")


def setup_optimization(
    input_file,
    backend,
    model,
    model_path,
    device,
    constraint_atoms,
    verbose,
    charge=0,
    spin=1,
    geometry=None,
):
    """Shared setup for minimize and transition_state commands."""
    # Import only when actually setting up optimization
    from qme.core.optimizer import QMEOptimizer
    from qme.settings import config as qme_config

    effective_backend = backend or qme_config.get_backend()

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
        # Pass charge and spin for UMA backend
        calculator_kwargs = {}
        if effective_backend == "uma":
            calculator_kwargs.update({"default_charge": charge, "default_spin": spin})

        qme = QMEOptimizer(
            backend=effective_backend,
            model_name=model,
            model_path=model_path,
            device=device,
            **calculator_kwargs,
        )

        # Load structure
        atoms = None
        if geometry:
            geometry.calc = qme.calculator
            qme.atoms = geometry
            atoms = geometry
        elif input_file:
            atoms = qme.load_structure(input_file)
        # If neither is provided, we'll skip initial structure loading
        # This is useful for workflows like ts-from-endpoints where
        # structures are loaded later

        if atoms is not None and verbose:
            click.echo(f"Loaded structure with {len(atoms)} atoms")

        # Parse constraints using the enhanced constraint system
        # Only parse if we have atoms to reference
        constraints = None
        if atoms is not None:
            constraints = parse_enhanced_constraints(constraint_atoms, atoms, verbose)

        return qme, constraints

    except Exception as e:
        raise RuntimeError(f"Setup failed: {e}")


def parse_enhanced_constraints(constraint_spec, atoms, verbose=False):
    """
    Parse constraint specification using the enhanced constraint system.

    Supports both old format (comma-separated atom indices for fixing)
    and new enhanced format with harmonic constraints.

    Parameters:
    - constraint_spec: Constraint specification string or None
    - atoms: Reference atoms object for constraint parsing
    - verbose: Print constraint information

    Returns:
        List of ASE-compatible constraints or None
    """
    if not constraint_spec:
        return None

    from qme.core.optimizer import QMEOptimizer

    try:
        # Check if it's the old simple format (just comma-separated numbers)
        # vs new enhanced format (contains keywords like 'fix', 'harmonic_')
        if constraint_spec.strip() and all(
            part.strip().isdigit() or part.strip() == ","
            for part in constraint_spec.replace(",", " , ").split()
            if part.strip()
        ):
            # Old format: just comma-separated atom indices for fixing
            atom_indices = [
                int(x.strip()) for x in constraint_spec.split(",") if x.strip()
            ]
            enhanced_spec = f"fix {','.join(map(str, atom_indices))}"
            if verbose:
                click.echo(f"Converting old constraint format to: {enhanced_spec}")
        else:
            # New enhanced format
            enhanced_spec = constraint_spec

        # Use the enhanced constraint parser
        temp_optimizer = QMEOptimizer(backend="mock")  # Temporary optimizer for parsing
        constraints = temp_optimizer.parse_constraints(enhanced_spec, atoms, verbose)
        return constraints
    except ValueError as e:
        raise ValueError(f"Invalid constraint specification: {e}")
    except Exception as e:
        raise RuntimeError(f"Constraint parsing failed: {e}")


def calculate_frequencies_if_requested(qme, results, frequencies, verbose, is_ts=False):
    """Calculate frequencies if requested and add to results."""
    if frequencies and results.get("converged", False):
        if verbose:
            click.echo("Calculating frequencies...")
        try:
            freq_results = qme.calculate_frequencies(
                atoms=results["optimized_atoms"], delta=0.01
            )
            results["frequencies"] = freq_results
            click.echo(
                f"✓ Calculated {len(freq_results['frequencies'])} "
                f"vibrational frequencies"
            )

            if is_ts:
                # For TS, also check for imaginary frequencies
                imag_freqs = [f for f in freq_results["frequencies"] if f < 0]
                if imag_freqs:
                    click.echo(
                        f"✓ Found {len(imag_freqs)} imaginary frequency(ies): "
                        f"{imag_freqs[0]:.1f} cm⁻¹"
                    )
                else:
                    click.echo(
                        "⚠ No imaginary frequencies found - "
                        "structure may not be a transition state"
                    )
        except Exception as e:
            click.echo(f"⚠ Frequency calculation failed: {e}")


def handle_optimization_results(results, qme, input_file, output, job_type, verbose):
    """Handle optimization results and save output."""
    if results["converged"]:
        final_energy = results["final_energy"]
        steps_taken = results["steps_taken"]

        click.echo(f"\n✓ {job_type.title()} optimization completed!")
        click.echo(f"Final energy: {final_energy:.6f} eV")
        click.echo(f"Steps taken: {steps_taken}")

        # Display frequency information if available
        if "frequencies" in results:
            freq_data = results["frequencies"]
            click.echo(f"Frequencies calculated: {len(freq_data['frequencies'])}")

        # Save optimized structure
        if output:
            output_path = Path(output)
        elif input_file:
            input_path = Path(input_file)
            suffix = "_optimized" if job_type == "minimize" else "_ts"
            output_path = input_path.with_name(
                input_path.stem + suffix + input_path.suffix
            )
        else:
            # Generate a default name when no input file is provided
            suffix = "_optimized" if job_type == "minimize" else "_ts"
            output_path = Path(f"{job_type}{suffix}.xyz")

        # Get the optimized structure (different keys for different job types)
        optimized_atoms = results.get("optimized_atoms") or results.get("ts_atoms")
        if optimized_atoms is None:
            raise KeyError("No optimized structure found in results")

        # Save as standard XYZ format (just atom types and coordinates)
        # for better compatibility
        _write_standard_xyz(optimized_atoms, str(output_path))
        click.echo(f"Optimized structure saved to: {output_path}")

    else:
        click.echo(f"\n❌ {job_type.title()} optimization did not converge")
        max_force = results.get("max_force", "unknown")
        click.echo(f"Maximum force: {max_force}")
        sys.exit(1)


def handle_cli_error(e, operation, verbose=False):
    """Standardized error handling for CLI commands."""
    click.echo(f"Error during {operation}: {e}", err=True)
    if verbose:
        import traceback

        traceback.print_exc()
    sys.exit(1)


def generate_output_path(input_path_str, suffix):
    """Generate output path with suffix."""
    input_path = Path(input_path_str)
    return input_path.with_suffix(suffix + input_path.suffix)


def validate_input_files(*file_paths):
    """Validate that input files exist."""
    for file_path in file_paths:
        if file_path and not Path(file_path).exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")


def display_backend_info(backend, model=None, device=None):
    """Display backend information."""
    click.echo(f"Backend: {backend}")
    if model:
        click.echo(f"Model: {model}")
    if device:
        click.echo(f"Device: {device}")
