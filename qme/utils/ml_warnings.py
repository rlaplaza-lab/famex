"""ML warning suppression utilities for QME."""

from __future__ import annotations

import contextlib
import logging
import sys
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator


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

        Returns:
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
        return all(pattern not in message for pattern in self.SUPPRESSED_PATTERNS)


@contextlib.contextmanager
def suppress_ml_warnings() -> Generator[list[str], None, None]:
    """Context manager to suppress verbose warnings and info messages from ML backends.

    This captures and suppresses:
    - JAX backend initialization messages
    - PyTorch CUDA warnings
    - NumExpr threading messages
    - Transformers/E3NN loading messages
    - FairChem initialization messages
    """
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
        """Filtered stderr that suppresses verbose backend messages."""

        def write(self, text: str) -> None:
            """Write text to stderr if not suppressed."""
            # Check if message should be suppressed
            should_suppress = any(pattern in text for pattern in verbose_filter.SUPPRESSED_PATTERNS)
            if not should_suppress and len(text.strip()) > 0:
                original_stderr.write(text)
                captured_messages.append(text)

        def flush(self) -> None:
            """Flush the original stderr."""
            original_stderr.flush()

        def close(self) -> None:
            """Close the original stderr if it has a close method."""
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


@contextlib.contextmanager
def quiet_backend_loading(
    backend: str,
    model_name: str | None = None,
    model_path: str | None = None,
    device: str | None = None,
    show_model_info: bool = True,
) -> Generator[list[str], None, None]:
    """Context manager for quiet backend loading with optional model info display.

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

    Yields:
    ------
    List[str]
        List of captured messages during backend loading

    """
    from qme.utils.logging import is_in_quiet_context, print_model_info

    # Check if we're already in a quiet context before setting the flag
    was_already_in_context = is_in_quiet_context()

    # Set the quiet context flag
    import threading

    _quiet_context_local = threading.local()
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
