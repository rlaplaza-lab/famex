"""Shared helpers for backend calculator loading."""

from __future__ import annotations

from qme.utils.lazy_imports import get_module_logger

logger = get_module_logger(__name__)


def raise_backend_load_error(
    backend: str,
    model_name: str | None,
    exc: BaseException,
) -> None:
    """Re-raise *exc* with a backend-specific message."""
    label = model_name or backend
    if isinstance(exc, ImportError):
        logger.error(
            "Failed to load %s model '%s': missing required dependencies. Error: %s",
            backend,
            label,
            exc,
        )
        msg = (
            f"Failed to load {backend} model '{label}': missing required dependencies. Error: {exc}"
        )
        raise ImportError(msg) from exc
    if isinstance(exc, ValueError | TypeError | KeyError):
        logger.error(
            "Failed to load %s model '%s': invalid model configuration. Error: %s",
            backend,
            label,
            exc,
        )
        msg = f"Failed to load {backend} model '{label}': invalid model configuration. Error: {exc}"
        raise ValueError(msg) from exc
    if isinstance(exc, OSError):
        logger.error(
            "Failed to load %s model '%s': file access error. Error: %s",
            backend,
            label,
            exc,
        )
        msg = f"Failed to load {backend} model '{label}': file access error. Error: {exc}"
        raise RuntimeError(msg) from exc
    if isinstance(exc, RuntimeError):
        logger.error(
            "Failed to load %s model '%s': runtime error. Error: %s",
            backend,
            label,
            exc,
        )
        msg = f"Failed to load {backend} model '{label}': runtime error. Error: {exc}"
        raise RuntimeError(msg) from exc
    raise exc
