#!/usr/bin/env python3
"""
QME CLI Demo - Comprehensive Backend Comparison

This example demonstrates QME's command-line interface capabilities by running
various optimization tasks across all available ML backends and comparing their
performance and reliability.

Usage:
    conda run -n py312 python cli_demo.py [--backends BACKEND1,BACKEND2,...]

Features:
    - Structure optimization using 'opt' command
    - Transition state optimization using 'tsopt' command
    - Two-ended optimization workflows
    - NEB path optimization
    - Comprehensive backend performance comparison
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Disable ASE GUI to prevent popup windows
os.environ["DISPLAY"] = ""
os.environ["MPLBACKEND"] = "Agg"

# Import QME for backend detection
try:
    from qme.backend_availability import is_backend_available
    # Import device utilities
    from device_utils import get_optimal_device, print_device_info
except ImportError as e:
    print(f"❌ Error importing QME: {e}")
    print("   Please ensure QME is installed and accessible")
    sys.exit(1)


def get_available_ml_backends() -> List[str]:
    """Get list of available ML backends (excluding mock)."""
    available = []
    ml_backends = ["aimnet2", "uma", "so3lr", "mace", "torchsim_mace", "torchsim_uma"]

    for backend in ml_backends:
        if is_backend_available(backend):
            available.append(backend)

    return available


def print_backend_summary(backends: List[str], title: str = "Available Backends"):
    """Print a formatted summary of backends."""
    print(f"\n📋 {title}")
    print("-" * 50)
    for i, backend in enumerate(backends, 1):
        print(f"  {i}. {backend}")
    print(f"Total: {len(backends)} backends")


def run_command(cmd, desc, backend, timeout=600) -> Tuple[bool, float, str, str]:
    """Run a CLI command and report results."""
    print(f"\n{'='*60}")
    print(f"Backend: {backend.upper()}")
    print(f"Task: {desc}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)

    try:
        start_time = time.time()
        # Create a modified environment with GUI disabled
        env = os.environ.copy()
        env["DISPLAY"] = ""
        env["MPLBACKEND"] = "Agg"

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent.parent,  # Run from qme root
            env=env,  # Pass the modified environment
        )
        end_time = time.time()
        runtime = end_time - start_time

        print(f"Runtime: {runtime:.2f} seconds")
        print(f"Exit code: {result.returncode}")

        if result.returncode == 0:
            print("Status: ✅ SUCCESS")
            return True, runtime, result.stdout, result.stderr
        else:
            print("Status: ❌ FAILED")
            if result.stderr:
                print("Error output:")
                print(
                    result.stderr[:500] + "..."
                    if len(result.stderr) > 500
                    else result.stderr
                )
            return False, runtime, result.stdout, result.stderr

    except subprocess.TimeoutExpired as e:
        print(f"Status: ❌ TIMEOUT after {timeout} seconds")
        # Clean up the process if it's still running
        if hasattr(e, "subprocess") and e.subprocess:
            try:
                e.subprocess.kill()
                e.subprocess.wait(timeout=5)
            except Exception:
                pass
        return False, timeout, "", f"Command timed out after {timeout} seconds"
    except Exception as e:
        print(f"Status: ❌ ERROR - {e}")
        return False, 0.0, "", str(e)


def create_example_commands(
    example_files: Path, backend: str, steps: int = 500
) -> List[Dict]:
    """Create example commands for a specific backend using default settings."""
    return [
        {
            "desc": "Structure optimization using 'opt' command",
            "cmd": [
                "qme",
                "opt",
                str(example_files / "reaction_001_reactant.xyz"),
                "--backend",
                backend,
                "--steps",
                str(steps),
                "--output",
                f"test_opt_{backend}.xyz",
                "--no-quiet",
            ],
        },
        {
            "desc": "Transition state optimization using 'tsopt' command",
            "cmd": [
                "qme",
                "tsopt",
                str(example_files / "reaction_001_ts.xyz"),
                "--backend",
                backend,
                "--steps",
                str(steps),
                "--output",
                f"test_tsopt_{backend}.xyz",
                "--no-quiet",
            ],
        },
        {
            "desc": "Two-ended minima optimization using 'opt' command",
            "cmd": [
                "qme",
                "opt",
                str(example_files / "reaction_001_reactant.xyz"),
                "--product",
                str(example_files / "reaction_001_product.xyz"),
                "--backend",
                backend,
                "--steps",
                str(steps),
                "--npoints",
                "5",
                "--interp",
                "geodesic",
                "--output",
                f"test_twoended_{backend}.xyz",
                "--no-quiet",
            ],
        },
        {
            "desc": "Two-ended TS optimization using 'tsopt' command",
            "cmd": [
                "qme",
                "tsopt",
                str(example_files / "reaction_001_reactant.xyz"),
                "--product",
                str(example_files / "reaction_001_product.xyz"),
                "--backend",
                backend,
                "--steps",
                str(steps),
                "--npoints",
                "5",
                "--interp",
                "geodesic",
                "--output",
                f"test_ts_twoended_{backend}.xyz",
                "--no-quiet",
            ],
        },
        {
            "desc": "NEB path optimization using 'tsopt' with --mode neb",
            "cmd": [
                "qme",
                "tsopt",
                str(example_files / "reaction_001_reactant.xyz"),
                "--product",
                str(example_files / "reaction_001_product.xyz"),
                "--mode",
                "neb",
                "--backend",
                backend,
                "--steps",
                str(steps),
                "--npoints",
                "7",
                "--interp",
                "geodesic",
                "--spring-constant",
                "2.0",  # Moderate spring constant for demo
                "--output",
                f"test_neb_{backend}.xyz",
                "--no-quiet",
            ],
        },
    ]


def demo_cli(backends: List[str] = None):
    """Demonstrate QME CLI with commands using default settings."""

    print("=" * 80)
    print("QME CLI Demo - Comprehensive Backend Comparison")
    print("=" * 80)
    print("Testing: opt, tsopt, two-ended, and NEB commands")

    # Ensure no config file interferes with defaults
    config_file = Path("qme.json")
    if config_file.exists():
        print(
            "\nFound qme.json config file - temporarily moving it for pure defaults test"
        )
        config_file.rename("qme.json.temp")
        print("Using built-in defaults only")

    # Get available ML backends
    try:
        if backends:
            # Filter requested backends to only available ones
            available_backends = []
            for backend in backends:
                if is_backend_available(backend):
                    available_backends.append(backend)
                else:
                    print(f"Warning: Backend '{backend}' not available, skipping")
        else:
            available_backends = get_available_ml_backends()

        if not available_backends:
            print("\n❌ No ML backends available for comparison.")
            print("Please install at least one ML backend:")
            print("  - UMA: pip install fairchem-core")
            print("  - MACE: pip install mace-torch")
            print("  - AIMNet2: pip install aimnet2")
            print("  - SO3LR: pip install so3lr")
            print("  - TorchSim: pip install torch-sim-atomistic (Python 3.11+)")
            return False

        print_backend_summary(available_backends, "Testing Backends")
    except Exception as e:
        print(f"❌ Error detecting backends: {e}")
        return False

    # Ensure we're in the right directory structure
    examples_dir = Path("examples")
    if not examples_dir.exists():
        print(
            "❌ Examples directory not found. Make sure to run from qme root directory."
        )
        return False

    example_files = examples_dir / "example_files"
    if not example_files.exists():
        print("❌ Example files directory not found.")
        return False

    # Check required files exist
    required_files = [
        "reaction_001_reactant.xyz",
        "reaction_001_product.xyz",
        "reaction_001_ts.xyz",
    ]

    for filename in required_files:
        if not (example_files / filename).exists():
            print(f"❌ Required file not found: {filename}")
            return False

    # Performance tracking
    backend_results = {}
    total_start_time = time.time()
    total_examples_per_backend = (
        5  # opt, tsopt, twoended, tsopt_twoended, and neb commands
    )
    steps = 500  # Reduced steps for faster testing

    # Run examples for each backend
    for backend in available_backends:
        print(f"\n{'='*80}")
        print(f"🧪 TESTING BACKEND: {backend.upper()}")
        print(f"{'='*80}")

        backend_results[backend] = {
            "examples": [],
            "total_time": 0.0,
            "successful": 0,
            "failed": 0,
        }

        examples = create_example_commands(example_files, backend, steps)

        backend_start_time = time.time()

        for i, example in enumerate(examples, 1):
            print(f"\n📋 Example {i}/{len(examples)} for {backend}")
            success, runtime, stdout, stderr = run_command(
                example["cmd"], example["desc"], backend
            )

            backend_results[backend]["examples"].append(
                {
                    "desc": example["desc"],
                    "success": success,
                    "runtime": runtime,
                    "stdout": stdout[:200] + "..." if len(stdout) > 200 else stdout,
                    "stderr": stderr[:200] + "..." if len(stderr) > 200 else stderr,
                }
            )

            if success:
                backend_results[backend]["successful"] += 1
            else:
                backend_results[backend]["failed"] += 1

            # Brief pause between examples
            time.sleep(0.5)

        backend_results[backend]["total_time"] = time.time() - backend_start_time

        # Backend summary
        print(f"\n📊 {backend.upper()} SUMMARY:")
        print(f"   ✅ Successful: {backend_results[backend]['successful']}")
        print(f"   ❌ Failed: {backend_results[backend]['failed']}")
        print(f"   ⏱️ Total time: {backend_results[backend]['total_time']:.2f}s")

    total_time = time.time() - total_start_time

    # Overall comparison summary
    print(f"\n{'='*80}")
    print("COMPREHENSIVE BACKEND COMPARISON")
    print(f"{'='*80}")

    print("\nBackend Performance Summary:")
    print(
        f"{'Backend':<12} {'Success':<8} {'Failed':<8} {'Total Time':<12} {'Avg Time/Task':<15}"
    )
    print("-" * 70)

    for backend in available_backends:
        results = backend_results[backend]
        avg_time = (
            results["total_time"] / len(results["examples"])
            if results["examples"]
            else 0
        )
        print(
            f"{backend:<12} {results['successful']:<8} {results['failed']:<8} "
            f"{results['total_time']:<12.2f} {avg_time:<15.2f}"
        )

    # Find best performing backend
    successful_backends = [
        (backend, results)
        for backend, results in backend_results.items()
        if results["successful"] > 0
    ]

    if successful_backends:
        # Sort by success rate, then by speed
        best_backend = max(
            successful_backends, key=lambda x: (x[1]["successful"], -x[1]["total_time"])
        )
        print(f"\nBest performing backend: {best_backend[0].upper()}")
        print(
            f"  Success rate: "
            f"{best_backend[1]['successful']}/{total_examples_per_backend} examples"
        )
        print(
            f"  Average time per task: "
            f"{best_backend[1]['total_time']/total_examples_per_backend:.2f}s"
        )

    print(f"\nTotal benchmark time: {total_time:.2f} seconds")

    # Check if all backends had some success
    total_successful = sum(
        results["successful"] for results in backend_results.values()
    )
    total_tests = len(available_backends) * total_examples_per_backend

    success_rate = total_successful / total_tests if total_tests > 0 else 0

    # Restore config file if it was moved
    temp_config = Path("qme.json.temp")
    if temp_config.exists():
        temp_config.rename("qme.json")
        print("\nRestored qme.json config file")

    if success_rate > 0.7:  # 70% success rate threshold
        print("\n✅ Overall demo successful! CLI commands working properly.")
        return True
    else:
        print(f"\n⚠️  Demo completed with {success_rate:.1%} success rate.")
        print("   Some backends may need attention or dependencies.")
        return success_rate > 0.3  # At least 30% working


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="QME CLI Demo - Comprehensive Backend Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  conda run -n py312 python cli_demo.py
  conda run -n py312 python cli_demo.py --backends aimnet2,uma
  conda run -n py312 python cli_demo.py --backends mace
        """,
    )

    parser.add_argument(
        "--backends",
        type=str,
        help="Comma-separated list of backends to test (default: all available)",
    )

    args = parser.parse_args()

    # Parse backends if provided
    backends = None
    if args.backends:
        backends = [b.strip() for b in args.backends.split(",")]

    try:
        success = demo_cli(backends)
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
