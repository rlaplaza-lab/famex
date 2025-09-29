"""Shared CLI option factories."""

import click


def add_common_options(options_func):
    """A decorator factory to add a list of common click options."""

    def decorator(f):
        options = options_func() if callable(options_func) else options_func
        if options:  # Ensure options is not None and is iterable
            for option in reversed(list(options)):
                f = option(f)
        return f

    return decorator


def get_core_options():
    """Get core options with hardcoded defaults to avoid heavy imports at CLI import time."""
    return [
        click.option(
            "--backend",
            "-b",
            default="mock",
            show_default=True,
            type=click.Choice(["uma", "so3lr", "aimnet2", "mace", "mock"]),
            help="Backend to use",
        ),
        click.option(
            "--model",
            "-m",
            default=None,
            type=str,
            help="Model name to use (auto-selected per backend if not specified)",
        ),
        click.option(
            "--model-path",
            type=click.Path(exists=True),
            help="Path to model file (SO3LR only)",
        ),
        click.option(
            "--device",
            "-d",
            type=click.Choice(["cpu", "cuda"]),
            default="cpu",
            show_default=True,
            help="Device for computations",
        ),
        click.option(
            "--charge",
            "-c",
            type=int,
            default=0,
            show_default=True,
            help="Total charge of the system",
        ),
        click.option(
            "--spin",
            "-s",
            type=int,
            default=1,
            show_default=True,
            help="Spin multiplicity",
        ),
        click.option("--verbose", "-v", is_flag=True, help="Verbose output"),
    ]


def get_optimization_options():
    """Get optimization options with hardcoded defaults to avoid heavy imports at CLI import
    time."""
    return [
        click.option(
            "--fmax",
            "-f",
            default=0.01,
            show_default=True,
            type=float,
            help="Force convergence criterion (eV/\u00c5)",
        ),
        click.option(
            "--steps",
            "--max-steps",
            default=500,
            show_default=True,
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
            show_default=True,
            type=float,
            help="Spring constant for harmonic constraints (eV/\u00c5\u00b2)",
        ),
        click.option(
            "--frequencies",
            is_flag=True,
            help="Calculate frequencies after optimization",
        ),
    ]
