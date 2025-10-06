#!/usr/bin/env python3
"""
Common interface utilities for QME examples.

This module provides standardized interfaces, output formatting, and common
functionality across all QME examples to ensure consistency.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import QME components
try:
    from qme.backend_availability import is_backend_available
    from qme.calculator_registry import calculator_registry
except ImportError as e:
    print(f"❌ Error importing QME: {e}")
    print("   Please ensure QME is installed and accessible")
    sys.exit(1)

# Import device utilities
from device_utils import get_optimal_device, print_device_info


class QMEExampleInterface:
    """Base class for standardized QME examples."""

    def __init__(self, name: str, description: str, epilog: str = ""):
        self.name = name
        self.description = description
        self.epilog = epilog
        self.start_time = time.time()

    def create_parser(self) -> argparse.ArgumentParser:
        """Create standardized argument parser."""
        parser = argparse.ArgumentParser(
            description=self.description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self.epilog,
        )

        # Standard arguments for all examples
        parser.add_argument(
            "--backends",
            type=str,
            help="Comma-separated list of backends to test (default: all available ML backends)",
        )
        parser.add_argument(
            "--device",
            type=str,
            default=None,
            choices=["cpu", "cuda"],
            help="Device to use for calculations (default: auto-detect CUDA if available)",
        )
        parser.add_argument(
            "--verbose", action="store_true", help="Print detailed progress information"
        )
        parser.add_argument(
            "--output",
            type=str,
            help="Output file for results (default: auto-generated based on example name)",
        )

        return parser

    def get_available_ml_backends(self) -> List[str]:
        """Get list of available ML backends (excluding mock)."""
        available = []
        ml_backends = [
            "aimnet2",
            "uma",
            "so3lr",
            "mace",
            "torchsim_mace",
            "torchsim_uma",
        ]

        for backend in ml_backends:
            if is_backend_available(backend):
                available.append(backend)

        return available

    def filter_available_backends(
        self, requested_backends: List[str], verbose: bool = False
    ) -> List[str]:
        """Filter requested backends to only available ones."""
        available = []
        for backend in requested_backends:
            if is_backend_available(backend):
                available.append(backend)
            elif verbose:
                print(f"Warning: Backend '{backend}' not available, skipping")

        return available

    def print_header(self, subtitle: str = ""):
        """Print standardized header."""
        print("=" * 80)
        print(f"QME {self.name}")
        if subtitle:
            print(f"{subtitle}")
        print("=" * 80)

    def print_backend_summary(
        self, backends: List[str], title: str = "Available Backends"
    ):
        """Print standardized backend summary."""
        print(f"\n📋 {title}")
        print("-" * 50)
        for i, backend in enumerate(backends, 1):
            print(f"  {i}. {backend}")
        print(f"Total: {len(backends)} backends")

    def print_configuration(self, config: Dict[str, Any]):
        """Print standardized configuration summary."""
        print("\nConfiguration:")
        for key, value in config.items():
            print(f"  {key}: {value}")

    def print_success(self, message: str = "Completed successfully!"):
        """Print standardized success message."""
        elapsed = time.time() - self.start_time
        print(f"\n✅ {message}")
        print(f"⏱️  Total time: {elapsed:.1f} seconds")

    def print_error(self, message: str):
        """Print standardized error message."""
        print(f"\n❌ {message}")

    def print_warning(self, message: str):
        """Print standardized warning message."""
        print(f"\n⚠️  {message}")

    def save_results(self, results: Dict[str, Any], output_file: Optional[str] = None):
        """Save results to JSON file."""
        if output_file is None:
            output_file = f"{self.name.lower().replace(' ', '_')}_results.json"

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path.absolute()}")

    def get_default_output_file(self) -> str:
        """Get default output file name for this example."""
        return f"{self.name.lower().replace(' ', '_')}_results.json"

    def get_device_info(self, device: Optional[str] = None) -> str:
        """Get optimal device and print info."""
        device = get_optimal_device(device)
        print_device_info(device)
        return device


def create_standard_epilog(example_type: str) -> str:
    """Create standardized epilog for different example types."""

    if example_type == "demo":
        return """
Examples:
  # Run with all available backends
  conda run -n py312 python cli_demo.py
  
  # Run with specific backends
  conda run -n py312 python cli_demo.py --backends uma,aimnet2
  
  # Run with verbose output
  conda run -n py312 python cli_demo.py --verbose
        """

    elif example_type == "timing":
        return """
Examples:
  # Run with all available backends
  conda run -n py312 python timing_benchmark.py
  
  # Run with specific backends
  conda run -n py312 python timing_benchmark.py --backends uma,aimnet2
  
  # Run on GPU
  conda run -n py312 python timing_benchmark.py --device cuda --verbose
        """

    elif example_type == "benchmark":
        return """
Examples:
  # Run with all available backends
  conda run -n py312 python benchmark.py
  
  # Run with specific backends
  conda run -n py312 python benchmark.py --backends uma,aimnet2
  
  # Quick test
  conda run -n py312 python benchmark.py --quick --verbose
        """

    else:
        return """
Examples:
  # Run with default settings
  conda run -n py312 python example.py
  
  # Run with specific backends
  conda run -n py312 python example.py --backends uma,aimnet2
  
  # Run with verbose output
  conda run -n py312 python example.py --verbose
        """


def print_standard_help():
    """Print standardized help information."""
    print("\n🔧 QME Examples Help")
    print("=" * 50)
    print("All QME examples follow a consistent interface:")
    print("\nCommon Options:")
    print("  --backends    Comma-separated list of backends to test")
    print("  --device      Device to use (cpu/cuda, default: auto-detect)")
    print("  --verbose     Print detailed progress information")
    print("  --output      Output file for results")
    print("  --help        Show this help message")
    print("\nAvailable Backends:")
    print("  uma           UMA (Meta AI, default)")
    print("  aimnet2       AIMNet2 (native PyTorch)")
    print("  mace          MACE (foundation models)")
    print("  so3lr         SO3LR (SO(3) neural networks)")
    print("  torchsim_*    TorchSim acceleration variants")
    print("\nFor more information, see the main README.md")
