"""Thin re-export layer for the FAMEX CLI.

The actual commands live in famex.cli.cli. This module re-exports the
Click entrypoint and subcommands for convenience and stable imports.
"""

from famex.cli.cli import main, minima, path, ts

__all__ = ["main", "minima", "path", "ts"]
