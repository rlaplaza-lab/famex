"""MACE / e3nn compatibility helpers."""

from __future__ import annotations

from typing import cast


def get_e3nn_version() -> str:
    try:
        import e3nn

        return cast(str, e3nn.__version__)
    except (ImportError, AttributeError):
        return "unknown"


def is_mace_e3nn_error(exc: BaseException) -> bool:
    message = str(exc)
    return any(
        phrase in message
        for phrase in ("too many values to unpack", "_compiled_main", "tensor size")
    )


def format_mace_e3nn_conflict_message(exc: BaseException) -> str:
    return (
        f"MACE compatibility issue with e3nn versions. "
        f"MACE 0.3.14 was built with e3nn==0.4.4, but e3nn "
        f"{get_e3nn_version()} is installed. "
        f"This causes serialization format incompatibilities. "
        f"\n\nWorkaround options:"
        f"\n1. Use a separate environment with e3nn==0.4.4 for MACE"
        f"\n2. Use UMA backend instead (compatible with current e3nn)"
        f"\n3. Wait for MACE update compatible with e3nn 0.5+"
        f"\n\nOriginal error: {exc}"
    )
