"""Device detection and normalization for FAMEX."""

from __future__ import annotations

import re
from functools import lru_cache

from famex.backends.constants import BACKEND_PYSCF, BACKEND_TBLITE
from famex.utils.logging import get_famex_logger

logger = get_famex_logger(__name__)

CPU_ONLY_BACKENDS: frozenset[str] = frozenset({BACKEND_TBLITE, BACKEND_PYSCF})
_CUDA_INDEX_PATTERN = re.compile(r"^cuda:\d+$")


def normalize_device(device: str) -> str:
    """Normalize ``gpu`` / ``cuda`` / ``cuda:N`` / ``cpu`` to ``cpu`` or ``cuda``."""
    normalized = device.strip().lower()
    if not normalized:
        raise ValueError("Device cannot be blank")
    if normalized == "gpu":
        return "cuda"
    if normalized in {"cpu", "cuda"}:
        return normalized
    if _CUDA_INDEX_PATTERN.match(normalized):
        return "cuda"
    raise ValueError(f"Unknown device: {device!r}")


@lru_cache(maxsize=1)
def is_cuda_available() -> bool:
    """Return whether CUDA is available via PyTorch."""
    try:
        from famex.backends.dependencies import deps

        if deps.has("torch"):
            return bool(deps.get("torch").cuda.is_available())
    except ImportError:
        pass
    return False


def get_optimal_device(device: str | None = None) -> str:
    """Return ``device`` if set, otherwise auto-select ``cuda`` or ``cpu``."""
    if device is not None and device.strip():
        return normalize_device(device)
    return "cuda" if is_cuda_available() else "cpu"


def _requests_cuda(device: str | None) -> bool:
    if device is None:
        return False
    text = device.strip().lower()
    return bool(text) and (text == "gpu" or text == "cuda" or text.startswith("cuda:"))


def resolve_backend_device(backend: str | None, device: str | None = None) -> str:
    """Resolve device for a backend; CPU-only backends reject CUDA requests."""
    backend_key = (backend or "").strip().lower()
    if backend_key in CPU_ONLY_BACKENDS:
        if _requests_cuda(device):
            raise ValueError(
                f"Backend '{backend_key}' does not support CUDA. "
                "Use device='cpu' or omit device to run on CPU."
            )
        return "cpu"
    return get_optimal_device(device)


def _torch_cuda_info() -> tuple[bool, str | None, int | None]:
    try:
        from famex.backends.dependencies import deps

        if not deps.has("torch"):
            return False, None, None
        torch = deps.get("torch")
        if not torch.cuda.is_available():
            return False, None, None
        return (
            True,
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory,
        )
    except ImportError:
        return False, None, None


def print_device_info(device: str) -> None:
    """Log which device is being used."""
    device = normalize_device(device)
    if device != "cuda":
        logger.info("Using CPU device")
        return
    available, gpu_name, _memory = _torch_cuda_info()
    if available:
        logger.info("Using CUDA device: %s", gpu_name)
    else:
        logger.warning("CUDA requested but not available, falling back to CPU")


def get_device_info(device: str) -> dict:
    """Return a dict with device / CUDA availability details."""
    device = normalize_device(device)
    info: dict = {
        "device": device,
        "cuda_available": False,
        "gpu_name": None,
        "gpu_memory": None,
    }
    if device == "cuda":
        available, gpu_name, memory = _torch_cuda_info()
        info["cuda_available"] = available
        info["gpu_name"] = gpu_name
        info["gpu_memory"] = memory
    return info
