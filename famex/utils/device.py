"""Device utilities for FAMEX.

This module provides centralized device detection and management for CUDA vs CPU,
ensuring consistent device handling across the entire FAMEX codebase.
"""

from famex.utils.logging import get_famex_logger

logger = get_famex_logger(__name__)

# Backends that cannot use GPU (semi-empirical / CPU-only stacks).
CPU_ONLY_BACKENDS: frozenset[str] = frozenset({"tblite"})


def get_optimal_device(device: str | None = None) -> str:
    """Get the optimal device for computation.

    Parameters
    ----------
    device : str, optional
        Explicitly requested device. If None, auto-detects.

    Returns
    -------
    str
        Device to use ('cpu' or 'cuda')

    """
    if device is not None:
        return device.lower()

    # Auto-detect CUDA availability
    try:
        from famex.backends.dependencies import deps

        if deps.has("torch"):
            torch = deps.get("torch")
            if torch.cuda.is_available():
                return "cuda"
    except ImportError:
        pass

    # Fallback to CPU
    return "cpu"


def _requests_cuda(device: str | None) -> bool:
    """Return whether ``device`` is an explicit CUDA/GPU request."""
    if device is None:
        return False
    normalized = device.strip().lower()
    return normalized == "gpu" or normalized.startswith("cuda")


def resolve_backend_device(backend: str | None, device: str | None = None) -> str:
    """Resolve device for a backend.

    MLIP backends default to CUDA when available. CPU-only backends (e.g. TBLite)
    run on CPU. An explicit CUDA request for a CPU-only backend is an error.
    An explicit ``device`` is honored for MLIP backends.

    Parameters
    ----------
    backend : str, optional
        Backend name (e.g. ``uma``, ``mace``, ``tblite``).
    device : str, optional
        Explicit device request. If None, auto-select for MLIP backends.

    Returns
    -------
    str
        ``"cpu"`` or ``"cuda"``.

    Raises
    ------
    ValueError
        If a CPU-only backend is asked to run on CUDA.
    """
    backend_key = (backend or "").strip().lower()
    if backend_key in CPU_ONLY_BACKENDS:
        if _requests_cuda(device):
            msg = (
                f"Backend '{backend_key}' does not support CUDA. "
                "Use device='cpu' or omit device to run on CPU."
            )
            raise ValueError(msg)
        return "cpu"
    return get_optimal_device(device)


def print_device_info(device: str) -> None:
    """Print device information for user feedback.

    Parameters
    ----------
    device : str
        Device being used

    """
    if device == "cuda":
        try:
            from famex.backends.dependencies import deps

            if deps.has("torch"):
                torch = deps.get("torch")
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    logger.info(f"🚀 Using CUDA device: {gpu_name}")
                else:
                    logger.warning("⚠️  CUDA requested but not available, falling back to CPU")
            else:
                logger.warning("⚠️  PyTorch not available, using CPU")
        except ImportError:
            logger.warning("⚠️  PyTorch not available, using CPU")
    else:
        logger.info("💻 Using CPU device")


def validate_device(device: str | None) -> str:
    """Validate and normalize device parameter.

    This function consolidates device validation logic from across the codebase.

    Parameters
    ----------
    device : str, optional
        Device specification to validate

    Returns
    -------
    str
        Validated and normalized device string

    Raises
    ------
    ValueError
        If device parameter is invalid

    """
    if device is None:
        return get_optimal_device()

    device = device.lower()
    valid_devices = ["cpu", "cuda", "gpu"]

    if device not in valid_devices:
        msg = (
            f"Invalid device '{device}'. "
            f"Supported devices: {', '.join(valid_devices)} or None for auto-detection. "
            f"Example: device='cpu' or device='cuda'"
        )
        raise ValueError(
            msg,
        )

    # Normalize 'gpu' to 'cuda'
    if device == "gpu":
        device = "cuda"

    # Validate CUDA availability if requested
    if device == "cuda":
        try:
            from famex.backends.dependencies import deps

            if not deps.has("torch"):
                msg = (
                    "PyTorch not available to check CUDA availability. "
                    "Please install PyTorch or use device='cpu'. "
                    "Try: pip install torch"
                )
                raise ValueError(
                    msg,
                )

            torch = deps.get("torch")
            if not torch.cuda.is_available():
                msg = (
                    "CUDA device requested but CUDA is not available on this system. "
                    "Please use device='cpu' or install CUDA-enabled PyTorch. "
                    "Try: pip install torch --index-url https://download.pytorch.org/whl/cu118"
                )
                raise ValueError(
                    msg,
                )
        except ImportError:
            msg = (
                "PyTorch not available to check CUDA availability. "
                "Please install PyTorch or use device='cpu'. "
                "Try: pip install torch"
            )
            raise ValueError(
                msg,
            )

    return device


def get_device_info(device: str) -> dict:
    """Get detailed device information.

    Parameters
    ----------
    device : str
        Device to get information for

    Returns
    -------
    dict
        Dictionary containing device information

    """
    info = {
        "device": device,
        "cuda_available": False,
        "gpu_name": None,
        "gpu_memory": None,
    }

    if device == "cuda":
        try:
            from famex.backends.dependencies import deps

            if deps.has("torch"):
                torch = deps.get("torch")
                if torch.cuda.is_available():
                    info["cuda_available"] = True
                    info["gpu_name"] = torch.cuda.get_device_name(0)
                    info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory
        except ImportError:
            pass

    return info
