"""
Thin re-export layer for the QME CLI.

The actual commands live in qme.cli.cli. This module re-exports the
Click entrypoint and subcommands for convenience and stable imports.
"""

from qme.cli.cli import main, opt, tsopt  # noqa: F401

__all__ = ["main", "opt", "tsopt"]
