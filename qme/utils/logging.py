"""Core logging configuration for QME."""

from __future__ import annotations

import logging
import sys
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Thread-local storage to track if we're already in a quiet_backend_loading context
_quiet_context_local = threading.local()

# QME logger configuration
_qme_logging_configured = False
_qme_log_level = logging.INFO


def setup_qme_logging(verbosity: int = 1, force: bool = False) -> None:
    """Configure QME logging system.

    Parameters
    ----------
    verbosity : int
        Verbosity level:
        - 0: WARNING and above (quiet)
        - 1: INFO and above (normal, default)
        - 2: DEBUG and above (verbose)
    force : bool
        Force reconfiguration even if already configured

    """
    global _qme_logging_configured, _qme_log_level

    if _qme_logging_configured and not force:
        return

    # Map verbosity to log level
    level_map = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
    }
    log_level = level_map.get(verbosity, logging.INFO)
    _qme_log_level = log_level

    # Configure root QME logger
    qme_logger = logging.getLogger("qme")
    qme_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    qme_logger.handlers.clear()

    # Create console handler with custom formatting
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    # Format: [QME] LEVEL: message
    # For INFO: just the message (clean output)
    # For DEBUG/WARNING/ERROR: include level
    class QMEFormatter(logging.Formatter):
        """Custom formatter for QME logging with clean INFO output."""

        def format(self, record: logging.LogRecord) -> str:
            """Format log record with clean output for INFO level."""
            if record.levelno == logging.INFO:
                return record.getMessage()
            if record.levelno == logging.DEBUG:
                return f"[DEBUG] {record.getMessage()}"
            if record.levelno == logging.WARNING:
                return f"⚠️  {record.getMessage()}"
            if record.levelno >= logging.ERROR:
                return f"❌ {record.getMessage()}"
            return record.getMessage()

    handler.setFormatter(QMEFormatter())
    qme_logger.addHandler(handler)

    # Allow propagation to root logger so child loggers can inherit handlers
    qme_logger.propagate = True

    _qme_logging_configured = True


def get_qme_logger(name: str) -> logging.Logger:
    """Get a QME logger for a specific module.

    Parameters
    ----------
    name : str
        Logger name, typically __name__ from the calling module

    Returns:
    -------
    logging.Logger
        Configured logger instance

    """
    # Ensure name starts with 'qme.'
    if not name.startswith("qme"):
        name = "qme" if name == "__main__" else f"qme.{name}"

    return logging.getLogger(name)


def get_qme_log_level() -> int:
    """Get current QME logging level."""
    return _qme_log_level


def is_in_quiet_context() -> bool:
    """Check if we're currently in a quiet_backend_loading context.

    Returns:
    -------
    bool
        True if we're in a quiet context, False otherwise

    """
    return getattr(_quiet_context_local, "in_quiet_context", False)


def print_model_info(
    backend: str,
    model_name: str | None = None,
    model_path: str | None = None,
    device: str | None = None,
) -> None:
    """Print clean model information for the user.

    Parameters
    ----------
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
                from qme.utils.device import get_device_info

                device_info = get_device_info(device)
                if device_info["gpu_name"]:
                    click.echo(f"GPU: {device_info['gpu_name']}")
            except Exception:
                pass  # Don't let GPU info fail the whole process

    click.echo("─" * 40)
