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

        # AIMNET2
        try:
            from aimnet2calc import AIMNet2ASE

            self._cache["aimnet2calc"] = AIMNet2ASE
            self._cache["HAS_AIMNET2"] = True
        except ImportError:
            self._cache["aimnet2calc"] = None
            self._cache["HAS_AIMNET2"] = False

        # FairChem
        try:
            from fairchem.core.common.utils import build_config
            from fairchem.core.models import model_registry
            from fairchem.core.trainers import ForcesTrainer

            self._cache["fairchem_build_config"] = build_config
            self._cache["fairchem_model_registry"] = model_registry
            self._cache["fairchem_trainer"] = ForcesTrainer
            self._cache["HAS_FAIRCHEM"] = True
        except ImportError:
            self._cache["fairchem_build_config"] = None
            self._cache["fairchem_model_registry"] = None
            self._cache["fairchem_trainer"] = None
            self._cache["HAS_FAIRCHEM"] = False

        # SO3LR
        try:
            import so3lr

            self._cache["so3lr"] = so3lr
            self._cache["HAS_SO3LR"] = True
        except ImportError:
            self._cache["so3lr"] = None
            self._cache["HAS_SO3LR"] = False

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
            "aimnet2": "aimnet2calc",
            "fairchem": "fairchem-core",
            "so3lr": "so3lr  # See installation instructions in README",
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
