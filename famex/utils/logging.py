"""Core logging configuration for FAMEX."""

from __future__ import annotations

import logging
import sys
import threading

from famex.utils.ml_warnings import install_runtime_ml_filters

_quiet_context_local = threading.local()

_famex_log_level = logging.WARNING

SEPARATOR_LINE = "─" * 40
MARK_OK = "✓"
MARK_FAIL = "✗"


class FAMEXFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if record.levelno == logging.INFO:
            return message
        if record.levelno == logging.DEBUG:
            return f"[DEBUG] {message}"
        if record.levelno == logging.WARNING:
            return f"⚠️  {message}"
        if record.levelno >= logging.ERROR:
            return f"❌ {message}"
        return message


def _verbosity_to_log_level(verbosity: int) -> int:
    # 0=WARNING, 1=INFO; anything higher (incl. CLI -vvv) → DEBUG
    return {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)


def setup_famex_logging(verbosity: int = 1, force: bool = False) -> None:
    global _famex_log_level

    # Suppress FairChem/torch/triton noise at normal verbosity; show at >=2.
    install_runtime_ml_filters(verbosity=verbosity)

    log_level = _verbosity_to_log_level(verbosity)
    _famex_log_level = log_level

    famex_logger = logging.getLogger("famex")
    famex_logger.setLevel(log_level)
    famex_logger.propagate = False

    ase_level = logging.DEBUG if verbosity >= 2 else logging.WARNING
    logging.getLogger("ase").setLevel(ase_level)

    if force:
        famex_logger.handlers.clear()

    if not famex_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        handler.setFormatter(FAMEXFormatter())
        famex_logger.addHandler(handler)
    else:
        for existing in famex_logger.handlers:
            existing.setLevel(log_level)


def get_famex_logger(name: str) -> logging.Logger:
    if name != "famex" and not name.startswith("famex."):
        name = "famex" if name == "__main__" else f"famex.{name}"

    return logging.getLogger(name)


def get_famex_log_level() -> int:
    return _famex_log_level


def is_in_quiet_context() -> bool:
    return getattr(_quiet_context_local, "depth", 0) > 0


def enter_quiet_context() -> None:
    depth = getattr(_quiet_context_local, "depth", 0) + 1
    _quiet_context_local.depth = depth


def exit_quiet_context() -> None:
    depth = max(0, getattr(_quiet_context_local, "depth", 0) - 1)
    _quiet_context_local.depth = depth


def set_quiet_context(active: bool) -> None:
    """Mark whether the current thread is inside a quiet-backend-loading context."""
    if active:
        enter_quiet_context()
    else:
        exit_quiet_context()


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
    logger.info("%s", SEPARATOR_LINE)

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
            except Exception as exc:
                # Don't let GPU info fail the whole process
                logger.debug("Failed to get GPU info: %s", exc)

    logger.info("%s", SEPARATOR_LINE)
