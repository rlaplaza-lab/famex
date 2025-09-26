"""
Configuration management for QME.

This module provides centralized configuration management with visible defaults
and optional config file overrides. No hidden files are created automatically.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class QMEDefaults:
    """Default QME configuration values - these are hardcoded visible defaults."""

    # Default backend settings
    backend: str = "uma"
    optimizer: str = "LBFGS"

    # Model default names
    models: Optional[Dict[str, str]] = None

    # Optimization defaults
    fmax: float = 0.01
    steps: int = 200

    # Device preferences
    device: Optional[str] = None  # Auto-detect

    # Mock calculator settings
    mock_force_constant: float = 1.0
    mock_bond_length: float = 1.0

    # File I/O settings
    output_format: str = "xyz"
    trajectory_format: str = "traj"

    # Advanced settings
    enable_warnings: bool = True
    verbose_fallbacks: bool = True

    def __post_init__(self):
        if self.models is None:
            self.models = {
                "uma": "uma-s-1p1",
                "so3lr": "so3lr-small",
                "aimnet2": "aimnet2",
                "mace": "mace-omol-0",
                "mock": "generic",
            }


class ConfigManager:
    """Manages QME configuration with visible defaults and optional config file overrides."""

    def __init__(self, config_file: Optional[Path] = None):
        """Initialize config manager with hard-coded defaults and optional overrides.

        Args:
            config_file: Optional path to config file. If None, looks for qme.json in current directory.
                        If no config file is found, uses only defaults.
        """
        self.defaults = QMEDefaults()
        self._overrides = {}
        self._config_file = config_file or Path("qme.json")
        self._config_loaded = False

        # Load config file if it exists (but don't create it)
        self._load_config_if_exists()

    def _load_config_if_exists(self):
        """Load configuration from file if it exists (doesn't create hidden files)."""
        if self._config_file.exists():
            try:
                with open(self._config_file, "r") as f:
                    self._overrides = json.load(f)
                self._config_loaded = True
            except (json.JSONDecodeError, IOError) as e:
                if self.defaults.enable_warnings:
                    print(
                        f"Warning: Failed to load config file {self._config_file}: {e}"
                    )

        # Always load from environment variables
        self._load_from_environment()

    def _load_from_environment(self):
        """Load settings from environment variables."""
        env_mappings = {
            "QME_BACKEND": "backend",
            "QME_OPTIMIZER": "optimizer",
            "QME_FMAX": ("fmax", float),
            "QME_STEPS": ("steps", int),
            "QME_DEVICE": "device",
        }

        for env_var, mapping in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                try:
                    if isinstance(mapping, tuple):
                        attr_name, converter = mapping
                        converted_value = converter(env_value)
                    else:
                        attr_name = mapping
                        converted_value = env_value

                    self._overrides[attr_name] = converted_value
                except (ValueError, TypeError) as e:
                    if self.defaults.enable_warnings:
                        print(f"Warning: Invalid environment variable {env_var}: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value, checking overrides first, then defaults."""
        if key in self._overrides:
            return self._overrides[key]
        elif hasattr(self.defaults, key):
            return getattr(self.defaults, key)
        return default

    def get_backend(self) -> str:
        """Get the backend to use."""
        return self.get("backend")

    def get_optimizer(self) -> str:
        """Get the optimizer to use."""
        return self.get("optimizer")

    def get_fmax(self) -> float:
        """Get the force convergence criterion."""
        return self.get("fmax")

    def get_steps(self) -> int:
        """Get the maximum optimization steps."""
        return self.get("steps")

    def get_device(self) -> Optional[str]:
        """Get preferred device, considering availability."""
        device = self.get("device")
        if device:
            return device

        # Auto-detect based on availability
        from .dependencies import deps

        if deps.has("torch"):
            torch = deps.get("torch")
            if torch.cuda.is_available():
                return "cuda"
        return "cpu"

    def get_model(self, backend: str) -> str:
        """Get default model name for a backend."""
        models = self.get("models")
        if models and backend in models:
            return models[backend]
        return f"{backend}-default"

    def has_config_file(self) -> bool:
        """Check if a config file was loaded."""
        return self._config_loaded

    def config_file_path(self) -> Path:
        """Get the path to the config file (may not exist)."""
        return self._config_file

    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues found."""
        issues = []

        # Check backend availability
        from .dependencies import deps

        backend = self.get_backend()
        if backend == "uma" and not deps.has("fairchem"):
            issues.append(f"Backend '{backend}' requires FairChem (fairchem-core)")
        elif backend == "so3lr" and not deps.has("so3lr"):
            issues.append(f"Backend '{backend}' requires SO3LR")
        elif backend == "aimnet2" and not deps.has("torch"):
            issues.append(f"Backend '{backend}' requires PyTorch")
        elif backend == "mace" and not deps.has("mace"):
            issues.append(f"Backend '{backend}' requires MACE (mace-torch)")

        return issues


# Global configuration instance
config = ConfigManager()


# Convenience functions
def get_default_backend() -> str:
    """Get the default backend name."""
    return config.get_backend()


def get_default_model(backend: str) -> str:
    """Get default model name for backend."""
    return config.get_model(backend)


def get_optimization_defaults() -> Dict[str, Any]:
    """Get default optimization parameters."""
    return {
        "optimizer": config.get_optimizer(),
        "fmax": config.get_fmax(),
        "steps": config.get_steps(),
    }
