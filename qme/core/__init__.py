"""Core API exports for QME.

This module intentionally provides a compact public surface for the
core package. New exports should be added here so users can import from
``qme.core`` directly.
"""

# Import strategy modules to register them
import qme.core.strategies.local  # noqa: F401
import qme.core.strategies.multistructure  # noqa: F401
from qme.core.explorer import Explorer
from qme.core.scipy_optimizers import TrustKrylovTS

__all__ = ["Explorer", "TrustKrylovTS"]
