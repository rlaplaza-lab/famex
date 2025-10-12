"""
Logging utilities for QME to manage verbose output from ML backends.
"""

import contextlib
import logging
import sys
import threading
import warnings
from typing import Generator, List, Optional

# Thread-local storage to track if we're already in a quiet_backend_loading context
_quiet_context_local = threading.local()


class VerboseFilter(logging.Filter):
    """Filter to suppress verbose messages from ML backends.

    This filter prevents verbose output from machine learning libraries
    during QME operations, keeping the output clean and focused.
    """

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

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter out verbose messages from ML backends.

        Parameters
        ----------
        record : logging.LogRecord
            Log record to filter

        Returns
        -------
        bool
            True if message should be shown, False if suppressed
        """
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
def suppress_ml_warnings() -> Generator[List[str], None, None]:
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
        def write(self, text: str) -> None:
            # Check if message should be suppressed
            should_suppress = any(pattern in text for pattern in verbose_filter.SUPPRESSED_PATTERNS)
            if not should_suppress and len(text.strip()) > 0:
                original_stderr.write(text)
                captured_messages.append(text)

        def flush(self) -> None:
            original_stderr.flush()

        def close(self) -> None:
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
) -> None:
    """
    Print clean model information for the user.

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


def is_in_quiet_context() -> bool:
    """Check if we're currently in a quiet_backend_loading context.

    Returns
    -------
    bool
        True if we're in a quiet context, False otherwise
    """
    return getattr(_quiet_context_local, "in_quiet_context", False)


@contextlib.contextmanager
def quiet_backend_loading(
    backend: str,
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    show_model_info: bool = True,
) -> Generator[List[str], None, None]:
    """
    Context manager for quiet backend loading with optional model info display.

    Parameters
    ----------
    backend : str
        Name of the backend being loaded
    model_name : str, optional
        Model name to display
    model_path : str, optional
        Model path to display
    device : str, optional
        Device to display
    show_model_info : bool, default True
        Whether to show model information

    Yields
    ------
    List[str]
        List of captured messages during backend loading
    """
    # Check if we're already in a quiet context before setting the flag
    was_already_in_context = is_in_quiet_context()

    # Set the quiet context flag
    _quiet_context_local.in_quiet_context = True

    try:
        # Only show model info if requested AND we weren't already in a quiet context
        if show_model_info and not was_already_in_context:
            print_model_info(backend, model_name, model_path, device)

        with suppress_ml_warnings() as captured:
            yield captured
    finally:
        # Clear the quiet context flag
        _quiet_context_local.in_quiet_context = False


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
        def format(self, record):
            if record.levelno == logging.INFO:
                return record.getMessage()
            elif record.levelno == logging.DEBUG:
                return f"[DEBUG] {record.getMessage()}"
            elif record.levelno == logging.WARNING:
                return f"⚠️  {record.getMessage()}"
            elif record.levelno >= logging.ERROR:
                return f"❌ {record.getMessage()}"
            return record.getMessage()

    handler.setFormatter(QMEFormatter())
    qme_logger.addHandler(handler)

    # Prevent propagation to root logger
    qme_logger.propagate = False

    _qme_logging_configured = True


def get_qme_logger(name: str) -> logging.Logger:
    """Get a QME logger for a specific module.

    Parameters
    ----------
    name : str
        Logger name, typically __name__ from the calling module

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Ensure name starts with 'qme.'
    if not name.startswith("qme"):
        if name == "__main__":
            name = "qme"
        else:
            name = f"qme.{name}"

    return logging.getLogger(name)


def get_qme_log_level() -> int:
    """Get current QME logging level."""
    return _qme_log_level
