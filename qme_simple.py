#!/usr/bin/env python
"""
Simple CLI demonstration for QME.
"""

import sys
import click
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, '.')

from qme.core import QMEOptimizer


@click.group()
@click.version_option(version="0.1.0")
def main():
    """QME: Quick mechanistic exploration using machine learning potentials."""
    pass


@main.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file for optimized structure')
@click.option('--optimizer', '-opt', default='BFGS', type=click.Choice(['BFGS', 'LBFGS', 'FIRE']))
@click.option('--fmax', '-f', default=0.01, type=float, help='Force convergence criterion')
@click.option('--steps', '-s', default=200, type=int, help='Maximum optimization steps')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def minimize(input_file, output, optimizer, fmax, steps, verbose):
    """Find minimum energy geometry."""
    
    if verbose:
        click.echo("Starting minimum energy optimization...")
        click.echo(f"Input file: {input_file}")
    
    try:
        # Use mock calculator for demonstration
        qme = QMEOptimizer(use_mock=True)
        atoms = qme.load_structure(input_file)
        
        if verbose:
            click.echo(f"Loaded structure with {len(atoms)} atoms")
        
        # Run optimization
        results = qme.optimize_minimum(optimizer=optimizer, fmax=fmax, steps=steps)
        
        if results['converged']:
            click.echo("✓ Optimization converged successfully!")
        else:
            click.echo("⚠ Optimization did not converge")
        
        if verbose:
            click.echo(f"Steps: {results['steps_taken']}")
            click.echo(f"Energy change: {results['energy_change']:.6f} eV")
        
        # Save result
        if output:
            qme.save_structure(results['optimized_atoms'], output)
            click.echo(f"Optimized structure saved to: {output}")
        else:
            input_path = Path(input_file)
            default_output = input_path.with_suffix('.opt' + input_path.suffix)
            qme.save_structure(results['optimized_atoms'], default_output)
            click.echo(f"Optimized structure saved to: {default_output}")
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
def test():
    """Test QME setup."""
    click.echo("Testing QME with mock calculator...")
    
    try:
        from ase.build import molecule
        
        qme = QMEOptimizer(use_mock=True)
        atoms = molecule('H2O')
        atoms.calc = qme.calculator
        energy = atoms.get_potential_energy()
        
        click.echo("✅ Mock calculator working!")
        click.echo(f"Sample H2O energy: {energy:.4f} eV")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")


if __name__ == '__main__':
    main()