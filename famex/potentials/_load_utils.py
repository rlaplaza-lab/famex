"""Shared helpers for backend calculator loading."""

from __future__ import annotations

from famex.utils.logging import get_famex_logger

logger = get_famex_logger(__name__)


def raise_backend_load_error(
    backend: str,
    model_name: str | None,
    exc: BaseException,
) -> None:
    """Re-raise *exc* with a backend-specific message."""
    label = model_name or backend
    if isinstance(exc, ImportError):
        msg = (
            f"Failed to load {backend} model '{label}': missing required dependencies. Error: {exc}"
        )
        logger.error("%s", msg)
        raise ImportError(msg) from exc
    if isinstance(exc, ValueError | TypeError | KeyError):
        msg = f"Failed to load {backend} model '{label}': invalid model configuration. Error: {exc}"
        logger.error("%s", msg)
        raise ValueError(msg) from exc
    if isinstance(exc, OSError):
        msg = f"Failed to load {backend} model '{label}': file access error. Error: {exc}"
        logger.error("%s", msg)
        raise RuntimeError(msg) from exc
    if isinstance(exc, RuntimeError):
        msg = f"Failed to load {backend} model '{label}': runtime error. Error: {exc}"
        logger.error("%s", msg)
        raise RuntimeError(msg) from exc
    raise
