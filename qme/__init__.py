"""QME: Quick Mechanistic Exploration using machine learning potentials.

This package exposes the public QME API at the package level. The previous
implementation relied on a fairly large module-level ``__getattr__`` which
proved fragile in some import contexts (tests/pytest). Provide an explicit
and conservative re-export of the common, safe symbols here so the package
always exposes the stable top-level names that tests and downstream code
expect.

Note: we intentionally avoid importing heavy ML backends (torch, fairchem,
so3lr, aimnet2, mace) at package import time. Those backends are still lazy
and will only be loaded when used via the calculator registry or the
corresponding potential modules.
"""

__version__ = "0.1.0"
__author__ = "QME Development Team"

# Minimal package-level exports. We intentionally avoid re-exporting implementation
# symbols here so consumers import from explicit subpackages (e.g. `qme.core`,
# `qme.potentials`, `qme.utils`). This makes imports explicit and avoids
# accidental heavy backend imports at package import time.

__all__ = ["__version__", "__author__"]
