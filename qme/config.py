"""
Configuration management for QME.

This module provides centralized configuration management with support for
default values, user preferences, and environment-based settings.
"""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class QMEConfig:
    """Main QME configuration container."""

    # Default backend settings
    default_backend: str = "so3lr"
    default_optimizer: str = "BFGS"

    # Model default names
    default_models: Optional[Dict[str, str]] = None

    # Optimization defaults
    default_fmax: float = 0.01
    default_steps: int = 200

    # Device preferences
    preferred_device: Optional[str] = None

    # Mock calculator settings
    mock_force_constant: float = 1.0
    mock_bond_length: float = 1.0

    # File I/O settings
    default_output_format: str = "xyz"
    trajectory_format: str = "traj"

    # Advanced settings
    enable_warnings: bool = True
    verbose_fallbacks: bool = True

    def __post_init__(self):
        if self.default_models is None:
            self.default_models = {
                "uma": "uma-s-1p1",
                "so3lr": "so3lr-small",
                "aimnet2": "aimnet2",
                "mock": "generic",
            }


class ConfigManager:
    """Manages QME configuration with file persistence and environment overrides."""

    def __init__(self):
        self.config = QMEConfig()
        self._config_dir = Path.home() / ".qme"
        self._config_file = self._config_dir / "config.json"
        self._load_config()

    def _load_config(self):
        """Load configuration from file and environment."""
        # Load from file if exists
        if self._config_file.exists():
            try:
                with open(self._config_file, "r") as f:
                    config_data = json.load(f)

                # Update config with loaded values
                for key, value in config_data.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
            except (json.JSONDecodeError, IOError) as e:
                if self.config.enable_warnings:
                    print(f"Warning: Failed to load config file: {e}")

        # Override with environment variables
        self._load_from_environment()

    def _load_from_environment(self):
        """Load settings from environment variables."""
        env_mappings = {
            "QME_DEFAULT_BACKEND": ("default_backend", str),
            "QME_DEFAULT_OPTIMIZER": ("default_optimizer", str),
            "QME_DEFAULT_FMAX": ("default_fmax", float),
            "QME_DEFAULT_STEPS": ("default_steps", int),
            "QME_PREFERRED_DEVICE": ("preferred_device", str),
            "QME_VERBOSE_FALLBACKS": (
                "verbose_fallbacks",
                lambda x: x.lower() == "true",
            ),
        }

        for env_var, (attr_name, converter) in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                try:
                    converted_value = converter(env_value)
                    setattr(self.config, attr_name, converted_value)
                except (ValueError, TypeError) as e:
                    if self.config.enable_warnings:
                        print(f"Warning: Invalid environment variable {env_var}: {e}")

    def save_config(self):
        """Save current configuration to file."""
        self._config_dir.mkdir(exist_ok=True)

        try:
            config_dict = asdict(self.config)
            with open(self._config_file, "w") as f:
                json.dump(config_dict, f, indent=2)
        except IOError as e:
            if self.config.enable_warnings:
                print(f"Warning: Failed to save config file: {e}")

    def get_default_model(self, backend: str) -> str:
        """Get default model name for a backend."""
        if self.config.default_models is None:
            return f"{backend}-default"
        return self.config.default_models.get(backend, f"{backend}-default")

    def set_default_model(self, backend: str, model_name: str):
        """Set default model name for a backend."""
        if self.config.default_models is None:
            self.config.default_models = {}
        self.config.default_models[backend] = model_name
        self.save_config()

    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues found."""
        issues = []

        # Check backend availability
        from .dependencies import deps

        backend = self.config.default_backend
        if backend == "uma" and not deps.has("fairchem"):
            issues.append(
                f"Default backend '{backend}' requires FairChem (fairchem-core)"
            )
        elif backend == "so3lr" and not deps.has("so3lr"):
            issues.append(f"Default backend '{backend}' requires SO3LR")
        elif backend == "aimnet2" and not deps.has("torch"):
            issues.append(f"Default backend '{backend}' requires PyTorch")

        # Check model names
        if self.config.default_models:
            for backend_name, model_name in self.config.default_models.items():
                if backend_name == "uma" and model_name not in [
                    "uma-s-1p1",
                    "uma-m-1p1",
                ]:
                    issues.append(
                        f"Unknown UMA model '{model_name}'. "
                        "Available: uma-s-1p1, uma-m-1p1"
                    )

        return issues

    def get_device_preference(self) -> Optional[str]:
        """Get preferred device, considering availability."""
        if self.config.preferred_device:
            return self.config.preferred_device

        # Auto-detect based on availability
        from .dependencies import HAS_TORCH, torch

        if HAS_TORCH and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def reset_to_defaults(self):
        """Reset configuration to defaults."""
        self.config = QMEConfig()
        self.save_config()

    def __getattr__(self, name: str) -> Any:
        """Allow direct access to config attributes."""
        if hasattr(self.config, name):
            return getattr(self.config, name)
        raise AttributeError(f"'ConfigManager' has no attribute '{name}'")

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to config attributes."""
        if hasattr(self.config, key):
            return getattr(self.config, key)
        raise KeyError(f"Configuration key '{key}' not found")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default."""
        if hasattr(self.config, key):
            return getattr(self.config, key)
        return default


# Global configuration instance
config = ConfigManager()


# Convenience functions
def get_default_backend() -> str:
    """Get the default backend name."""
    return config.default_backend


def get_default_model(backend: str) -> str:
    """Get default model name for backend."""
    return config.get_default_model(backend)


def get_optimization_defaults() -> Dict[str, Any]:
    """Get default optimization parameters."""
    return {
        "optimizer": config.default_optimizer,
        "fmax": config.default_fmax,
        "steps": config.default_steps,
    }


def set_defaults(**kwargs):
    """Set default configuration values."""
    for key, value in kwargs.items():
        if hasattr(config.config, key):
            setattr(config.config, key, value)
    config.save_config()
