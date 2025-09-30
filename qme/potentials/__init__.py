"""qme.potentials

This package exposes a small, stable API for creating ML potential
calculators. Heavy backends may require optional runtime dependencies; the
package-level module provides lightweight factory fallbacks so the rest of the
project can import ``qme.potentials`` in CI or minimal environments.

Design decisions / simplifications made here:
- Narrow import-time exception handling to ImportError where appropriate so
  unrelated runtime errors inside a backend surface during development.
- Use the centralized ``deps`` manager to check availability and emit tidy
  fallback warnings via ``deps.warn_fallback``.
"""

from importlib import import_module
from typing import Callable

from qme.dependencies import deps

# Build an exports list and set __all__ once at the end to avoid runtime
# mutations of the module-level __all__ (helps static analysis tools).
exports = []


# Base and mock are lightweight and expected to be present; if their import
# fails with ImportError, provide a clear fallback so package import still
# succeeds but constructing objects raises a helpful message.
try:
    from qme.potentials.base_potential import BasePotential
    exports.append("BasePotential")
except ImportError:
    BasePotential = None

try:
    from qme.potentials.mock_potential import MockCalculator
except ImportError:
    class _MissingMock:
        def __init__(self, *args, **kwargs):
            raise ImportError("MockCalculator implementation is missing")

    MockCalculator = _MissingMock
else:
    exports.append("MockCalculator")


def _make_fallback_factory(backend_name: str) -> Callable[..., object]:
    """Return a factory that warns and returns a configured MockCalculator.

    We purposely call ``deps.warn_fallback`` so users see a standardized
    message when a real backend is unavailable.
    """

    def _factory(**kwargs):
        deps.warn_fallback(backend_name, reason="implementation not available")
        return MockCalculator(backend=backend_name, **kwargs)

    return _factory


# Optional backends: only import their concrete modules when their external
# dependencies are available according to ``deps.has``. If the backend's local
# module import raises ImportError, fall back to the mock factory. If it raises
# any other exception we re-raise so real bugs are not hidden.
optional_backends = [
    ("uma", "uma_potential", "UMAPotential", "get_uma_calculator"),
    ("so3lr", "so3lr_potential", "SO3LRPotential", "get_so3lr_calculator"),
    ("aimnet2", "aimnet2_potential", "AIMNet2Potential", "get_aimnet2_calculator"),
    ("mace", "mace_potential", "MACEPotential", "get_mace_calculator"),
]

for backend_label, module_name, cls_name, func_name in optional_backends:
    if deps.has(backend_label):
        try:
            module = import_module(f"qme.potentials.{module_name}")
            globals()[cls_name] = getattr(module, cls_name)
            globals()[func_name] = getattr(module, func_name)
            exports.extend([cls_name, func_name])
            continue
        except ImportError:
            # Import missing pieces in the local module; fall through to
            # create a fallback factory below which will warn and return a
            # MockCalculator.
            pass
    # Backend not available or failed to import: provide fallback factory.
    globals()[func_name] = _make_fallback_factory(backend_label)
    exports.append(func_name)


# Finalize module exports
__all__ = exports

