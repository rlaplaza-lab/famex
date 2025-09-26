"""
Centralized dependency management for QME.

This module handles all optional dependencies and provides consistent
import patterns across the package.
"""

import warnings
from typing import Any


class DependencyManager:
    """Manages optional dependencies with consistent error handling."""

    def __init__(self):
        self._cache = {}
        self._check_dependencies()

    def _check_dependencies(self):
        """Check availability of all optional dependencies."""
        # PyTorch
        try:
            import torch

            self._cache["torch"] = torch
            self._cache["HAS_TORCH"] = True
        except ImportError:
            self._cache["torch"] = None
            self._cache["HAS_TORCH"] = False

        # SELLA
        try:
            from sella import Sella

            self._cache["sella"] = Sella
            self._cache["HAS_SELLA"] = True
        except ImportError:
            self._cache["sella"] = None
            self._cache["HAS_SELLA"] = False

        # AIMNET2 - now using native implementation
        # We always mark as True since we have native implementation
        self._cache["aimnet2calc"] = None  # Not used anymore
        self._cache["HAS_AIMNET2"] = True

        # FairChem (v2 API)
        try:
            from fairchem.core import FAIRChemCalculator, pretrained_mlip
            from fairchem.core.units.mlip_unit import load_predict_unit

            self._cache["fairchem_pretrained_mlip"] = pretrained_mlip
            self._cache["fairchem_calculator"] = FAIRChemCalculator
            self._cache["fairchem_load_predict_unit"] = load_predict_unit
            self._cache["HAS_FAIRCHEM"] = True
        except ImportError:
            self._cache["fairchem_pretrained_mlip"] = None
            self._cache["fairchem_calculator"] = None
            self._cache["fairchem_load_predict_unit"] = None
            self._cache["HAS_FAIRCHEM"] = False

        # SO3LR
        try:
            import so3lr

            self._cache["so3lr"] = so3lr
            self._cache["HAS_SO3LR"] = True
        except ImportError:
            self._cache["so3lr"] = None
            self._cache["HAS_SO3LR"] = False

        # MACE
        try:
            import mace.calculators

            self._cache["mace"] = mace.calculators
            self._cache["HAS_MACE"] = True
        except ImportError:
            self._cache["mace"] = None
            self._cache["HAS_MACE"] = False

    def get(self, name: str, default: Any = None) -> Any:
        """Get a dependency or capability flag."""
        return self._cache.get(name, default)

    def has(self, name: str) -> bool:
        """Check if a dependency is available."""
        return self._cache.get(f"HAS_{name.upper()}", False)

    def require(self, name: str, purpose: str = "this functionality") -> Any:
        """Require a dependency, raising ImportError if not available."""
        if not self.has(name):
            raise ImportError(
                f"{name} is required for {purpose}. "
                f"Install with: pip install {self._get_install_command(name)}"
            )
        return self.get(name.lower())

    def _get_install_command(self, name: str) -> str:
        """Get installation command for a dependency."""
        commands = {
            "torch": "torch",
            "sella": "sella",
            "aimnet2": "torch",  # Native implementation only needs torch
            "fairchem": "fairchem-core",
            "so3lr": "so3lr  # See installation instructions in README",
            "mace": "mace-torch",
        }
        return commands.get(name.lower(), name.lower())

    def warn_fallback(self, backend: str, reason: str = "dependencies not available"):
        """Issue a standardized fallback warning."""
        warnings.warn(
            f"Falling back to mock {backend.upper()} calculator: {reason}. "
            f"For production use, install the required dependencies.",
            UserWarning,
        )


# Global dependency manager instance
deps = DependencyManager()

# Export commonly used items for backward compatibility
torch = deps.get("torch")
HAS_TORCH = deps.has("torch")
HAS_SELLA = deps.has("sella")
HAS_AIMNET2 = deps.has("aimnet2")
HAS_FAIRCHEM = deps.has("fairchem")
HAS_SO3LR = deps.has("so3lr")
HAS_MACE = deps.has("mace")
HAS_MACE = deps.has("mace")
