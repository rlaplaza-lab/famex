"""
Centralized dependency management for QME.

This module handles all optional dependencies and provides consistent
import patterns across the package with lazy loading.
"""

import importlib
import importlib.util
import warnings
from typing import Any, Dict


class _DependencyContext:
    """Context manager for dependency access."""

    def __init__(self, manager, deps_names):
        self.manager = manager
        self.deps_names = deps_names
        self.modules = None

    def __enter__(self):
        if len(self.deps_names) == 1:
            self.modules = self.manager.require_multiple(*self.deps_names)
            return self.modules
        else:
            self.modules = self.manager.require_multiple(*self.deps_names)
            return tuple(self.modules[name] for name in self.deps_names)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Nothing to clean up for now
        pass


class DependencyManager:
    """Manages optional dependencies with lazy loading and consistent error handling."""

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._availability_cache: Dict[str, bool] = {}

    def _check_availability_lazy(self, package_name: str) -> bool:
        """
        Lazily check if a package is available without importing it.

        This uses importlib.util.find_spec to check package availability
        without actually importing the module, avoiding heavy imports.
        """
        if package_name not in self._availability_cache:
            try:
                spec = importlib.util.find_spec(package_name)
                self._availability_cache[package_name] = spec is not None
            except (ImportError, ValueError, AttributeError):
                self._availability_cache[package_name] = False
        return self._availability_cache[package_name]

    def _load_dependency(self, name: str, fallback_value: Any = None) -> Any:
        """Load a dependency only when it's actually needed."""
        if name not in self._cache:
            try:
                if name == "torch":
                    import torch

                    self._cache["torch"] = torch
                elif name == "sella":
                    from sella import Sella

                    self._cache["sella"] = Sella
                elif name == "so3lr":
                    import so3lr

                    self._cache["so3lr"] = so3lr
                elif name == "mace":
                    import mace.calculators

                    self._cache["mace"] = mace.calculators
                elif name == "torch_sim":
                    import torch_sim as ts

                    self._cache["torch_sim"] = ts
                elif name == "fairchem_pretrained_mlip":
                    from fairchem.core import pretrained_mlip

                    self._cache["fairchem_pretrained_mlip"] = pretrained_mlip
                elif name == "fairchem_calculator":
                    from fairchem.core import FAIRChemCalculator

                    self._cache["fairchem_calculator"] = FAIRChemCalculator
                elif name == "fairchem_load_predict_unit":
                    from fairchem.core.units.mlip_unit import load_predict_unit

                    self._cache["fairchem_load_predict_unit"] = load_predict_unit
                else:
                    # Generic import
                    module = importlib.import_module(name)
                    self._cache[name] = module
            except ImportError:
                self._cache[name] = fallback_value
        return self._cache[name]

    def require_multiple(self, *deps_names, purpose="this functionality"):
        """
        Require multiple dependencies, raising DependencyError if any are missing.

        Parameters:
        -----------
        *deps_names : str
            Names of required dependencies
        purpose : str
            Description of what the dependencies are needed for

        Returns:
        --------
        dict or module : If one dependency, returns the module directly.
                        If multiple, returns dict mapping name -> module.
        """
        from qme.core.validation import DependencyError

        if len(deps_names) == 1:
            name = deps_names[0]
            if not self.has(name):
                install_cmd = self._get_install_command(name)
                raise DependencyError(name, purpose, install_cmd)
            return self.get(name.lower())
        else:
            results = {}
            missing = []
            for name in deps_names:
                if not self.has(name):
                    missing.append(name)
                else:
                    results[name] = self.get(name.lower())

            if missing:
                install_cmds = [self._get_install_command(name) for name in missing]
                install_command = " ".join(install_cmds)
                raise DependencyError(", ".join(missing), purpose, install_command)
            return results

    def need(self, *deps_names):
        """
        Context manager that ensures dependencies are available within a block.

        Usage:
        ------
        with deps.need("torch", "fairchem") as (torch, fairchem_modules):
            # Use torch and fairchem_modules here
            ...
        """
        return _DependencyContext(self, deps_names)

    def get(self, name: str, default: Any = None) -> Any:
        """Get a dependency, loading it lazily if needed."""
        return self._load_dependency(name, default)

    def has(self, name: str) -> bool:
        """Check if a dependency is available without importing it."""
        # Map backend names to package names for availability checking
        package_mapping = {
            "torch": "torch",
            "sella": "sella",
            "aimnet2": "torch",  # AIMNet2 only needs torch
            "fairchem": "fairchem.core",
            "so3lr": "so3lr",
            "mace": "mace.calculators",
            "torch_sim": "torch_sim",
        }

        package_name = package_mapping.get(name.lower(), name.lower())

        # Special case for aimnet2 - always True since we have native implementation
        if name.lower() == "aimnet2":
            return self._check_availability_lazy("torch")

        return self._check_availability_lazy(package_name)

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
            "torch_sim": "torch-sim-atomistic",
        }
        return commands.get(name.lower(), name.lower())



# Global dependency manager instance
deps = DependencyManager()


# Function to get lazy globals - everything is now lazy
def __getattr__(name):
    """Support for lazy loading of module-level attributes."""
    if name == "HAS_SELLA":
        return deps.has("sella")
    elif name == "HAS_TORCH":
        return deps.has("torch")
    elif name == "HAS_AIMNET2":
        return deps.has("aimnet2")
    elif name == "HAS_FAIRCHEM":
        return deps.has("fairchem")
    elif name == "HAS_SO3LR":
        return deps.has("so3lr")
    elif name == "HAS_MACE":
        return deps.has("mace")
    elif name == "HAS_TORCH_SIM":
        return deps.has("torch_sim")
    elif name == "torch":
        return deps.get("torch")
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
