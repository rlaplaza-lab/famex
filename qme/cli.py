"""Top-level CLI facade for QME.

Keep this file minimal so importing `qme` is lightweight. The full CLI
implementation lives in the `qme.cli` package (module: `qme.cli.main`).
"""

from .cli.main import main

__all__ = ["main"]
