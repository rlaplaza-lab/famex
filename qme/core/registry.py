"""
Calculator registry moved to qme.core.registry (facade copy).

Non-destructive: root-level `qme/calculator_registry.py` will continue to
re-export the registry until other modules are updated.
"""

import importlib
from typing import Callable, Dict, Optional

from ..utils import deps


class CalculatorRegistry:
    def __init__(self):
        self._registry: Dict[str, Callable] = {}
        self._lazy_registry: Dict[str, dict] = {
            "so3lr": {
                "module": "qme.so3lr_potential",
                "function": "get_so3lr_calculator",
            },
            "uma": {"module": "qme.uma_potential", "function": "get_uma_calculator"},
            "aimnet2": {
                "module": "qme.aimnet2_potential",
                "function": "get_aimnet2_calculator",
            },
            "mace": {"module": "qme.mace_potential", "function": "get_mace_calculator"},
            "mock": {
                "module": "qme.mock_calculator",
                "function": "MockCalculator",
                "is_class": True,
            },
        }

    def _load_backend(self, backend_name: str):
        if backend_name in self._registry:
            return
        if backend_name not in self._lazy_registry:
            return
        backend_info = self._lazy_registry[backend_name]
        module_name = backend_info["module"]
        function_name = backend_info["function"]
        try:
            module = importlib.import_module(module_name)
            func_or_class = getattr(module, function_name)
            if backend_info.get("is_class", False):
                self._registry[backend_name] = lambda **kwargs: func_or_class(
                    backend=kwargs.get("backend", "generic")
                )
            else:
                self._registry[backend_name] = func_or_class
        except ImportError:
            pass
        except Exception as e:
            import warnings

            warnings.warn(f"Failed to load backend {backend_name}: {e}")

    def register(self, backend_name: str, factory_func: Callable):
        self._registry[backend_name] = factory_func

    def get_available_backends(self) -> list:
        return list(self._lazy_registry.keys())

    def create_calculator(
        self,
        backend: str,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        self._load_backend(backend)
        if backend not in self._registry:
            available = self.get_available_backends()
            raise ValueError(
                f"Unknown or unavailable backend: {backend}. Available: {available}"
            )
        factory_func = self._registry[backend]
        factory_kwargs = kwargs.copy()
        if model_name is not None:
            factory_kwargs["model_name"] = model_name
        if model_path is not None:
            factory_kwargs["model_path"] = model_path
        if device is not None:
            factory_kwargs["device"] = device
        if backend == "mock":
            factory_kwargs["backend"] = kwargs.get("mock_backend", "generic")
        try:
            return factory_func(**factory_kwargs)
        except ImportError:
            # Preserve explicit ImportError from factory (propagate)
            raise
        except Exception as exc:
            # Treat any exception during calculator creation as the backend
            # being unavailable. This converts deep import/unpickle/runtime
            # errors from optional heavy backends (torch/fairchem/e3nn) into
            # ImportError so higher-level logic and tests can gracefully
            # handle missing backends.
            import warnings

            warnings.warn(
                f"Calculator factory for backend '{backend}' failed: {exc}",
                UserWarning,
            )
            raise ImportError(f"Backend '{backend}' is not available: {exc}") from exc

    def is_backend_available(self, backend: str) -> bool:
        if backend == "mock":
            return True
        elif backend == "so3lr":
            return deps.has("so3lr")
        elif backend == "uma":
            return deps.has("fairchem") and deps.has("torch")
        elif backend == "aimnet2":
            return deps.has("torch")
        elif backend == "mace":
            return deps.has("mace") and deps.has("torch")
        else:
            return backend in self._registry


# Global registry instance
calculator_registry = CalculatorRegistry()
