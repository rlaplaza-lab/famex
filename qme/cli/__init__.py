"""qme.cli package: command-line interface subpackage for QME.

This package contains the CLI implementation and helper utilities. We keep a
lightweight __init__ that re-exports the main entrypoint to preserve
compatibility with console entry points that previously referenced
``qme.cli:main``.
"""

from .cli import main

__all__ = ["main"]
