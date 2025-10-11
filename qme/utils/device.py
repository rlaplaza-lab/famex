"""
Device utilities for QME.

This module provides centralized device detection and management for CUDA vs CPU,
ensuring consistent device handling across the entire QME codebase.
"""

from typing import Optional

from qme.logging_utils import get_qme_logger

logger = get_qme_logger(__name__)


def get_optimal_device(device: Optional[str] = None) -> str:
    """
    Get the optimal device for computation.

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
        from qme.dependencies import deps

        if deps.has("torch"):
            torch = deps.get("torch")
            if torch.cuda.is_available():
                return "cuda"
    except ImportError:
        pass

    # Fallback to CPU
    return "cpu"


def print_device_info(device: str) -> None:
    """
    Print device information for user feedback.

    Parameters
    ----------
    device : str
        Device being used
    """
    if device == "cuda":
        try:
            from qme.dependencies import deps

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


def validate_device(device: Optional[str]) -> str:
    """
    Validate and normalize device parameter.

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
        raise ValueError(
            f"Invalid device '{device}'. "
            f"Supported devices: {', '.join(valid_devices)} or None for auto-detection. "
            f"Example: device='cpu' or device='cuda'"
        )

    # Normalize 'gpu' to 'cuda'
    if device == "gpu":
        device = "cuda"

    # Validate CUDA availability if requested
    if device == "cuda":
        try:
            from qme.dependencies import deps

            if not deps.has("torch"):
                raise ValueError(
                    "PyTorch not available to check CUDA availability. "
                    "Please install PyTorch or use device='cpu'. "
                    "Try: pip install torch"
                )

            torch = deps.get("torch")
            if not torch.cuda.is_available():
                raise ValueError(
                    "CUDA device requested but CUDA is not available on this system. "
                    "Please use device='cpu' or install CUDA-enabled PyTorch. "
                    "Try: pip install torch --index-url https://download.pytorch.org/whl/cu118"
                )
        except ImportError:
            raise ValueError(
                "PyTorch not available to check CUDA availability. "
                "Please install PyTorch or use device='cpu'. "
                "Try: pip install torch"
            )

    return device


def get_device_info(device: str) -> dict:
    """
    Get detailed device information.

    Parameters
    ----------
    device : str
        Device to get information for

    Returns
    -------
    dict
        Dictionary containing device information
    """
    info = {"device": device, "cuda_available": False, "gpu_name": None, "gpu_memory": None}

    if device == "cuda":
        try:
            from qme.dependencies import deps

            if deps.has("torch"):
                torch = deps.get("torch")
                if torch.cuda.is_available():
                    info["cuda_available"] = True
                    info["gpu_name"] = torch.cuda.get_device_name(0)
                    info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory
        except ImportError:
            pass

    return info
