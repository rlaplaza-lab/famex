"""
Logging utilities for QME to manage verbose output from ML backends.
"""

import contextlib
import logging
import sys
import warnings
from typing import Optional


class VerboseFilter(logging.Filter):
    """Filter to suppress verbose messages from ML backends."""

    SUPPRESSED_LOGGERS = [
        "numexpr.utils",
        "jax._src.xla_bridge",
        "fairchem",
        "torch",
        "transformers",
        "e3nn",
    ]

    SUPPRESSED_PATTERNS = [
        "NumExpr defaulting to",
        "Unable to initialize backend",
        "Failed to open libtpu.so",
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD",
        "cuequivariance",
        "weights_only",
        "Environment variable",
    ]

    def filter(self, record):
        """Filter out verbose messages from ML backends."""
        # Check if logger is in suppressed list
        for logger_name in self.SUPPRESSED_LOGGERS:
            if record.name.startswith(logger_name):
                return False

        # Check if message contains suppressed patterns
        message = record.getMessage()
        for pattern in self.SUPPRESSED_PATTERNS:
            if pattern in message:
                return False

        return True


@contextlib.contextmanager
def suppress_ml_warnings():
    """
    Context manager to suppress verbose warnings and info messages from ML backends.

    This captures and suppresses:
    - JAX backend initialization messages
    - PyTorch CUDA warnings
    - NumExpr threading messages
    - Transformers/E3NN loading messages
    - FairChem initialization messages
    """
    # Store original settings
    original_log_level = logging.getLogger().level
    original_warnings_filters = list(warnings.filters)

    # Set up verbose filter
    verbose_filter = VerboseFilter()

    # Get root logger and add filter
    root_logger = logging.getLogger()
    root_logger.addFilter(verbose_filter)

    # Suppress specific warning categories
    warnings.filterwarnings("ignore", category=UserWarning, module="e3nn")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
    warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
    warnings.filterwarnings("ignore", message=".*cuequivariance.*")

    # Temporarily redirect stderr to capture and filter messages
    original_stderr = sys.stderr
    captured_messages = []

    class FilteredStderr:
        def write(self, text):
            # Check if message should be suppressed
            should_suppress = any(
                pattern in text for pattern in verbose_filter.SUPPRESSED_PATTERNS
            )
            if not should_suppress and len(text.strip()) > 0:
                original_stderr.write(text)
                captured_messages.append(text)

        def flush(self):
            original_stderr.flush()

        def close(self):
            # Delegate close to original stderr if it has a close method
            if hasattr(original_stderr, "close"):
                original_stderr.close()

    try:
        sys.stderr = FilteredStderr()
        yield captured_messages
    finally:
        # Restore original settings
        sys.stderr = original_stderr
        root_logger.removeFilter(verbose_filter)
        # Reset warnings - simpler approach
        warnings.resetwarnings()


def print_model_info(
    backend: str,
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
):
    """
    Print clean model information for the user.

    Parameters:
    -----------
    backend : str
        Name of the ML backend
    model_name : str, optional
        Name of the model being used
    model_path : str, optional
        Path to model file (for local models)
    device : str, optional
        Device being used (cpu/cuda)
    """
    import click

    click.echo(f"\n🔧 Initializing {backend.upper()} Backend")
    click.echo("─" * 40)

    if model_name:
        click.echo(f"Model: {model_name}")

    if model_path:
        click.echo(f"Model Path: {model_path}")

    if device:
        click.echo(f"Device: {device}")
        if device == "cuda":
            try:
                # Only import torch if we're actually using CUDA
                from qme.dependencies import deps

                if deps.has("torch"):
                    torch = deps.get("torch")
                    if torch.cuda.is_available():
                        gpu_name = torch.cuda.get_device_name(0)
                        click.echo(f"GPU: {gpu_name}")
            except Exception as m:
                pass  # Don't let GPU info fail the whole process

    click.echo("─" * 40)


@contextlib.contextmanager
def quiet_backend_loading(
    backend: str,
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    show_model_info: bool = True,
):
    """
    Context manager for quiet backend loading with optional model info display.

    Parameters:
    -----------
    backend : str
        Name of the backend being loaded
    model_name : str, optional
        Model name to display
    model_path : str, optional
        Model path to display
    device : str, optional
        Device to display
    show_model_info : bool
        Whether to show model information (default: True)
    """
    if show_model_info:
        print_model_info(backend, model_name, model_path, device)

    with suppress_ml_warnings() as captured:
        yield captured
