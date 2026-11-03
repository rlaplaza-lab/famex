"""Core logging configuration for FAMEX."""

from __future__ import annotations

import logging
import sys
import threading

from famex.utils.ml_warnings import install_runtime_ml_filters

_quiet_context_local = threading.local()

_famex_logging_configured = False
_famex_log_level = logging.INFO


class FAMEXFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if record.levelno == logging.INFO:
            return record.getMessage()
        if record.levelno == logging.DEBUG:
            return f"[DEBUG] {record.getMessage()}"
        if record.levelno == logging.WARNING:
            return f"⚠️  {record.getMessage()}"
        if record.levelno >= logging.ERROR:
            return f"❌ {record.getMessage()}"
        return record.getMessage()


def _verbosity_to_log_level(verbosity: int) -> int:
    # 0=WARNING, 1=INFO; anything higher (incl. CLI -vvv) → DEBUG
    return {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)


def setup_famex_logging(verbosity: int = 1, force: bool = False) -> None:
    global _famex_logging_configured, _famex_log_level

    # Suppress FairChem/torch/triton noise at normal verbosity; show at >=2.
    install_runtime_ml_filters(verbosity=verbosity)

    log_level = _verbosity_to_log_level(verbosity)
    _famex_log_level = log_level

    famex_logger = logging.getLogger("famex")
    famex_logger.setLevel(log_level)

    ase_level = logging.DEBUG if verbosity >= 2 else logging.WARNING
    logging.getLogger("ase").setLevel(ase_level)

    rebuild_handlers = force or not _famex_logging_configured
    if rebuild_handlers:
        famex_logger.handlers.clear()

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        handler.setFormatter(FAMEXFormatter())
        famex_logger.addHandler(handler)

        famex_logger.propagate = False
        _famex_logging_configured = True
    else:
        for existing in famex_logger.handlers:
            existing.setLevel(log_level)


def get_famex_logger(name: str) -> logging.Logger:
    if not name.startswith("famex"):
        name = "famex" if name == "__main__" else f"famex.{name}"

    return logging.getLogger(name)


def get_famex_log_level() -> int:
    return _famex_log_level


def is_in_quiet_context() -> bool:
    return getattr(_quiet_context_local, "in_quiet_context", False)


def set_quiet_context(active: bool) -> None:
    """Mark whether the current thread is inside a quiet-backend-loading context."""
    _quiet_context_local.in_quiet_context = active


def print_model_info(
    backend: str,
    model_name: str | None = None,
    model_path: str | None = None,
    device: str | None = None,
    verbose: int = 1,
) -> None:
    if verbose == 0:
        return

    logger = get_famex_logger("famex")

    logger.info("\n🔧 Initializing %s Backend", backend.upper())
    logger.info("─" * 40)

    if model_name:
        logger.info("Model: %s", model_name)

    if model_path:
        logger.info("Model Path: %s", model_path)

    if device:
        logger.info("Device: %s", device)
        if device == "cuda":
            try:
                from famex.utils.device import get_device_info

                device_info = get_device_info(device)
                if device_info["gpu_name"]:
                    logger.info("GPU: %s", device_info["gpu_name"])
            except Exception:
                pass  # Don't let GPU info fail the whole process

    logger.info("─" * 40)
