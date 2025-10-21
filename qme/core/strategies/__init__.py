"""Strategy modules for QME Explorer.

This package contains all optimization strategies organized by type:
- local: Single-structure optimization strategies
- multistructure: Multi-structure optimization strategies
"""

# Import strategy modules to register them
import qme.core.strategies.local  # noqa: F401
import qme.core.strategies.multistructure  # noqa: F401
