"""Core logging configuration for FAMEX."""

from __future__ import annotations

import logging
import sys
import threading

_quiet_context_local = threading.local()

_famex_logging_configured = False
_famex_log_level = logging.INFO


def setup_famex_logging(verbosity: int = 1, force: bool = False) -> None:
    global _famex_logging_configured, _famex_log_level

    if _famex_logging_configured and not force:
        return

    level_map = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
    }
    log_level = level_map.get(verbosity, logging.INFO)
    _famex_log_level = log_level

    famex_logger = logging.getLogger("famex")
    famex_logger.setLevel(log_level)

    famex_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

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

    handler.setFormatter(FAMEXFormatter())
    famex_logger.addHandler(handler)

    famex_logger.propagate = False

    _famex_logging_configured = True


def get_famex_logger(name: str) -> logging.Logger:
    if not name.startswith("famex"):
        name = "famex" if name == "__main__" else f"famex.{name}"

    return logging.getLogger(name)


def get_famex_log_level() -> int:
    return _famex_log_level


def is_in_quiet_context() -> bool:
    return getattr(_quiet_context_local, "in_quiet_context", False)


def print_model_info(
    backend: str,
    model_name: str | None = None,
    model_path: str | None = None,
    device: str | None = None,
    verbose: int = 1,
) -> None:
    if verbose == 0:
        return

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
                from famex.utils.device import get_device_info

                device_info = get_device_info(device)
                if device_info["gpu_name"]:
                    click.echo(f"GPU: {device_info['gpu_name']}")
            except Exception:
                pass  # Don't let GPU info fail the whole process

    click.echo("─" * 40)
