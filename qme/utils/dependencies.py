"""
Centralized dependency management (moved into qme.utils).

Non-destructive copy of `qme/dependencies.py` to allow gradual refactor.
"""

import importlib
import importlib.util
import warnings
from typing import Any, Dict


class _DependencyContext:
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
        pass


class DependencyManager:
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._availability_cache: Dict[str, bool] = {}

    def _check_availability_lazy(self, package_name: str) -> bool:
        if package_name not in self._availability_cache:
            try:
                spec = importlib.util.find_spec(package_name)
                self._availability_cache[package_name] = spec is not None
            except Exception:
                # Be defensive: any unexpected exception while probing availability
                # should be treated as "not available" so callers can gracefully
                # fall back. This avoids surprising crashes for optional heavy
                # backends that can raise non-ImportError exceptions during
                # import-time probing (e.g. unpickling errors, extension import
                # issues).
                self._availability_cache[package_name] = False
        return self._availability_cache[package_name]

    def _load_dependency(self, name: str, fallback_value: Any = None) -> Any:
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
                    module = importlib.import_module(name)
                    self._cache[name] = module
            except Exception as exc:
                # Treat any exception during optional backend import as a
                # non-available dependency. Emit a concise warning so users
                # can diagnose problems, but avoid raising (tests expect
                # optional backends to be detectable as "not available").
                msg = (
                    f"Optional dependency '{name}' failed to import and will be "
                    f"treated as unavailable: {exc}"
                )
                warnings.warn(msg, UserWarning)
                self._cache[name] = fallback_value
        return self._cache[name]

    def require_multiple(self, *deps_names, purpose="this functionality"):
        if len(deps_names) == 1:
            name = deps_names[0]
            if not self.has(name):
                raise ImportError(
                    f"{name} is required for {purpose}. "
                    f"Install with: pip install {self._get_install_command(name)}"
                )
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
                raise ImportError(
                    f"Missing dependencies for {purpose}: {', '.join(missing)}. "
                    f"Install with: pip install {' '.join(install_cmds)}"
                )
            return results

    def need(self, *deps_names):
        return _DependencyContext(self, deps_names)

    def get(self, name: str, default: Any = None) -> Any:
        return self._load_dependency(name, default)

    def has(self, name: str) -> bool:
        package_mapping = {
            "torch": "torch",
            "sella": "sella",
            "aimnet2": "torch",
            "fairchem": "fairchem.core",
            "so3lr": "so3lr",
            "mace": "mace.calculators",
        }

        package_name = package_mapping.get(name.lower(), name.lower())

        if name.lower() == "aimnet2":
            return self._check_availability_lazy("torch")

        return self._check_availability_lazy(package_name)

    def require(self, name: str, purpose: str = "this functionality") -> Any:
        if not self.has(name):
            raise ImportError(
                f"{name} is required for {purpose}. "
                f"Install with: pip install {self._get_install_command(name)}"
            )
        return self.get(name.lower())

    def _get_install_command(self, name: str) -> str:
        commands = {
            "torch": "torch",
            "sella": "sella",
            "aimnet2": "torch",
            "fairchem": "fairchem-core",
            # Keep the recommendation concise to avoid long single lines
            "so3lr": "so3lr",
            "mace": "mace-torch",
        }
        return commands.get(name.lower(), name.lower())

    def warn_fallback(self, backend: str, reason: str = "dependencies not available"):
        # Wrap the warning message so line length stays within style limits
        msg = (
            f"Falling back to mock {backend.upper()} calculator: {reason}. "
            "For production use, install the required dependencies."
        )
        warnings.warn(msg, UserWarning)


# Global dependency manager instance
deps = DependencyManager()


def __getattr__(name):
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
    elif name == "torch":
        return deps.get("torch")
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
