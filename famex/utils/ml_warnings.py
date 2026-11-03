"""ML warning suppression utilities for FAMEX.

Third-party backends (FairChem/UMA, PyTorch, Triton) emit noisy warnings and log
lines on every call. FAMEX suppresses them at normal verbosity (0–1) and only
allows them through at high verbosity (>=2).
"""

from __future__ import annotations

import contextlib
import logging
import sys
import warnings
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from collections.abc import Generator

# Verbosity at which third-party ML noise is shown (Explorer/CLI -vv and above).
_ML_NOISE_VERBOSITY_THRESHOLD = 2

_configured_quiet: bool | None = None
_original_showwarning = warnings.showwarning

# Substrings matched against warning messages (case-insensitive via filterwarnings).
# Keep these broad enough to catch the exact FairChem / torch / triton lines users see.
_NOISE_WARNING_SPECS: tuple[tuple[str, type[Warning]], ...] = (
    (r".*index_reduce\(\).*", UserWarning),
    (r".*torch\.jit\.script(_method)? is deprecated.*", DeprecationWarning),
    (r".*AnnAssign\.__init__ missing.*", DeprecationWarning),
    (r".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*", Warning),
    (r".*cuequivariance.*", Warning),
    (r".*extra keyword arguments on `Field`.*", DeprecationWarning),
)

# Plain substrings used by logging filters and the showwarning gate.
_NOISE_SUBSTRINGS: tuple[str, ...] = (
    "NumExpr defaulting to",
    "Unable to initialize backend",
    "Failed to open libtpu.so",
    "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD",
    "cuequivariance",
    "weights_only",
    "Environment variable",
    "Model is being compiled",
    "index_reduce()",
    "Profiler record function",
    "torch.jit.script_method",
    "torch.jit.script is deprecated",
    "AnnAssign.__init__",
)


class VerboseFilter(logging.Filter):
    """Filter to suppress verbose messages from ML backends."""

    SUPPRESSED_LOGGERS = [
        "numexpr.utils",
        "jax._src.xla_bridge",
        "fairchem",
        "torch",
        "transformers",
        "e3nn",
        "triton",
    ]

    SUPPRESSED_PATTERNS = list(_NOISE_SUBSTRINGS)

    def filter(self, record: logging.LogRecord) -> bool:
        """Return True if the record should be shown."""
        for logger_name in self.SUPPRESSED_LOGGERS:
            if record.name.startswith(logger_name):
                return False

        message = record.getMessage()
        return all(pattern not in message for pattern in self.SUPPRESSED_PATTERNS)


def _message_is_ml_noise(message: str) -> bool:
    return any(pattern in message for pattern in _NOISE_SUBSTRINGS)


def _quiet_showwarning(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: TextIO | None = None,
    line: str | None = None,
) -> None:
    """Drop known ML-backend noise while quiet; otherwise delegate to default."""
    text = str(message)
    if _message_is_ml_noise(text):
        return
    # Also drop by filename for fairchem/torch/triton deprecations we listed.
    noisy_paths = (
        "fairchem/core/models/uma/escn_moe.py",
        "torch/jit/_script.py",
        "triton/compiler/code_generator.py",
    )
    if any(path in filename.replace("\\", "/") for path in noisy_paths):
        if category in (UserWarning, DeprecationWarning, FutureWarning):
            return
    _original_showwarning(message, category, filename, lineno, file=file, line=line)


def _silence_torch_structured_logging() -> None:
    """Quiet torch._logging W… lines (Profiler record function, etc.)."""
    try:
        import torch._logging as torch_logging
    except ImportError:
        return
    try:
        torch_logging.set_logs(all=logging.ERROR)
    except Exception:
        pass


def install_runtime_ml_filters(verbosity: int = 1) -> None:
    """Install or refresh session-wide ML noise filters.

    Parameters
    ----------
    verbosity :
        FAMEX verbosity. Noise is suppressed when ``verbosity < 2`` and allowed
        when ``verbosity >= 2``.
    """
    global _configured_quiet

    quiet = verbosity < _ML_NOISE_VERBOSITY_THRESHOLD
    if _configured_quiet is quiet:
        return

    if quiet:
        for message, category in _NOISE_WARNING_SPECS:
            warnings.filterwarnings("ignore", message=message, category=category)
        # Module-scoped catches for anything that slips past message filters.
        warnings.filterwarnings("ignore", category=UserWarning, module=r"e3nn(\.|$)")
        warnings.filterwarnings("ignore", category=UserWarning, module=r"torch(\.|$)")
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=r"fairchem(\.|$)",
        )
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            module=r"torch(\.|$)",
        )
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            module=r"triton(\.|$)",
        )
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"transformers(\.|$)")

        warnings.showwarning = _quiet_showwarning

        root_logger = logging.getLogger()
        if not any(isinstance(f, VerboseFilter) for f in root_logger.filters):
            root_logger.addFilter(VerboseFilter())

        for name in ("fairchem", "torch", "triton", "transformers", "e3nn"):
            logging.getLogger(name).setLevel(logging.ERROR)

        _silence_torch_structured_logging()
    else:
        # High verbosity: restore default warning display; leave logger levels alone
        # so users can debug backends.
        if warnings.showwarning is _quiet_showwarning:
            warnings.showwarning = _original_showwarning

    _configured_quiet = quiet


@contextlib.contextmanager
def suppress_ml_warnings() -> Generator[list[str], None, None]:
    """Context manager to suppress verbose warnings and info messages from ML backends."""
    install_runtime_ml_filters(verbosity=0)

    verbose_filter = VerboseFilter()
    root_logger = logging.getLogger()
    already_filtered = any(isinstance(f, VerboseFilter) for f in root_logger.filters)
    if not already_filtered:
        root_logger.addFilter(verbose_filter)

    original_stderr = sys.stderr
    captured_messages: list[str] = []

    class FilteredStderr:
        """Filtered stderr that suppresses verbose backend messages."""

        def write(self, text: str) -> int:
            should_suppress = _message_is_ml_noise(text)
            if not should_suppress and len(text.strip()) > 0:
                original_stderr.write(text)
                captured_messages.append(text)
            return len(text)

        def flush(self) -> None:
            original_stderr.flush()

        def close(self) -> None:
            if hasattr(original_stderr, "close"):
                original_stderr.close()

    try:
        sys.stderr = FilteredStderr()
        yield captured_messages
    finally:
        sys.stderr = original_stderr
        if not already_filtered:
            root_logger.removeFilter(verbose_filter)


@contextlib.contextmanager
def quiet_backend_loading(
    backend: str,
    model_name: str | None = None,
    model_path: str | None = None,
    device: str | None = None,
    show_model_info: bool = True,
    verbose: int = 1,
) -> Generator[list[str], None, None]:
    """Context manager for quiet backend loading with optional model info display."""
    from famex.utils.logging import is_in_quiet_context, print_model_info, set_quiet_context

    was_already_in_context = is_in_quiet_context()
    set_quiet_context(True)

    try:
        if show_model_info and not was_already_in_context:
            print_model_info(backend, model_name, model_path, device, verbose=verbose)

        with suppress_ml_warnings() as captured:
            yield captured
    finally:
        set_quiet_context(False)
