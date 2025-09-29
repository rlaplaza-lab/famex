"""CLI entrypoint and group for QME."""

import click


@click.group()
@click.version_option(version="0.1.0")
def main():
    """
    QME: Quick mechanistic exploration using machine learning potentials.
    """
    pass


# Backward-compatibility import point; commands are registered via import side-effects
from . import commands  # noqa: F401,E402
