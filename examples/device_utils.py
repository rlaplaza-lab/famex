"""
Device utilities for QME examples.

This module provides automatic device detection for CUDA vs CPU,
ensuring examples work optimally on both GPU and CPU systems.
"""

from typing import Optional


def get_optimal_device(device: Optional[str] = None) -> str:
    """
    Get the optimal device for computation.

    Parameters:
    -----------
    device : str, optional
        Explicitly requested device. If None, auto-detects.

    Returns:
    --------
    str
        Device to use ('cpu' or 'cuda')
    """
    if device is not None:
        return device.lower()

    # Auto-detect CUDA availability
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    # Fallback to CPU
    return "cpu"


def print_device_info(device: str) -> None:
    """
    Print device information for user feedback.

    Parameters:
    -----------
    device : str
        Device being used
    """
    if device == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print("🚀 Using CUDA device: {}".format(gpu_name))
            else:
                print("⚠️  CUDA requested but not available, falling back to CPU")
        except ImportError:
            print("⚠️  PyTorch not available, using CPU")
    else:
        print("💻 Using CPU device")
