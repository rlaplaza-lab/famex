#!/usr/bin/env python3
"""
Demo script showing QME CLI functionality with comprehensive backend comparison.
Tests the new CLI commands: opt (minimize), tsopt (transition state optimization),
and neb (nudged elastic band from reactant-product endpoints) across all available backends.
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Import QME dependencies check similar to bh28_benchmark
try:
    from qme.dependencies import deps
except ImportError:
    print("❌ QME not properly installed. Please install with 'pip install -e .'")
    sys.exit(1)


def get_available_backends() -> List[str]:
    """Get list of available ML QME backends for demonstration."""
    available = []

    if deps.has("fairchem"):
        available.append("uma")
    if deps.has("so3lr"):
        available.append("so3lr")
    if deps.has("aimnet2"):
        available.append("aimnet2")
    if deps.has("mace"):
        available.append("mace")

    if len(available) == 0:
        print("❌ No ML backends available! Please install at least one.")
        return []

    return available


def run_command(cmd, desc, backend, timeout=300) -> Tuple[bool, float, str, str]:
    """Run a CLI command and report results."""
    print(f"\n{'='*80}")
    print(f"🔧 Backend: {backend}")
    print(f"📋 Task: {desc}")
    print(f"💻 Command: {' '.join(cmd)}")
    print(f"{'='*80}")

    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent.parent,  # Run from qme root
        )
        end_time = time.time()
        runtime = end_time - start_time

        print(f"⏱️  Runtime: {runtime:.2f} seconds")
        print(f"🚦 Exit code: {result.returncode}")

        if result.returncode == 0:
            print("✅ SUCCESS")
            return True, runtime, result.stdout, result.stderr
        else:
            print("❌ FAILED")
            if result.stderr:
                print("STDERR:")
                print(
                    result.stderr[:500] + "..."
                    if len(result.stderr) > 500
                    else result.stderr
                )
            return False, runtime, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT after {timeout} seconds")
        return False, timeout, "", f"Command timed out after {timeout} seconds"
    except Exception as e:
        print(f"❌ ERROR: {e}")
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
                str(example_files / "diels_alder_reactant.xyz"),
                "--backend",
                backend,
                "--steps",
                str(steps),
                "--output",
                f"test_opt_{backend}.xyz",
                "--verbose",
            ],
        },
        {
            "desc": "Transition state optimization using 'tsopt' command",
            "cmd": [
                "qme",
                "tsopt",
                str(example_files / "biaryl_ts.xyz"),
                "--backend",
                backend,
                "--steps",
                str(steps),
                "--output",
                f"test_tsopt_{backend}.xyz",
                "--verbose",
            ],
        },
        {
            "desc": "NEB method using 'neb' command",
            "cmd": [
                "qme",
                "neb",
                str(example_files / "diels_alder_reactant.xyz"),
                str(example_files / "diels_alder_product.xyz"),
                "--backend",
                backend,
                "--steps",
                str(steps),
                "--npoints",
                "5",
                "--interp-method",
                "linear",
                "--output",
                f"test_neb_{backend}.xyz",
                "--verbose",
            ],
        },
        {
            "desc": "Show default configuration (no config file)",
            "cmd": [
                "qme",
                "config",
                "--show",
            ],
        },
    ]


def demo_cli():
    """Demonstrate QME CLI with new commands using default settings."""

    print("🚀 QME CLI Demo: Testing opt, tsopt, and neb commands")
    print("=" * 80)

    # Ensure no config file interferes with defaults
    config_file = Path("qme.json")
    if config_file.exists():
        print(
            "⚠️ Found qme.json config file - temporarily moving it for pure defaults test"
        )
        config_file.rename("qme.json.temp")
        print("✅ Using built-in defaults only")

    # Get available backends
    try:
        available_backends = get_available_backends()
        if not available_backends:
            print("❌ No real ML backends available for comparison.")
            return False
        print(f"🔍 Available backends: {', '.join(available_backends)}")
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
        "diels_alder_reactant.xyz",
        "diels_alder_product.xyz",
        "biaryl_ts.xyz",
    ]

    for filename in required_files:
        if not (example_files / filename).exists():
            print(f"❌ Required file not found: {filename}")
            return False

    # Performance tracking
    backend_results = {}
    total_start_time = time.time()
    total_examples_per_backend = 4  # opt, tsopt, neb, and config commands
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
    print("🏆 COMPREHENSIVE BACKEND COMPARISON")
    print(f"{'='*80}")

    print("📊 Backend Performance Comparison:")
    print(
        f"{'Backend':<10} {'Success':<8} {'Failed':<8} {'Total Time':<12} {'Avg Time/Task':<15}"
    )
    print("-" * 60)

    for backend in available_backends:
        results = backend_results[backend]
        avg_time = (
            results["total_time"] / len(results["examples"])
            if results["examples"]
            else 0
        )
        print(
            f"{backend:<10} {results['successful']:<8} {results['failed']:<8} "
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
        print(f"\n🥇 Best performing backend: {best_backend[0].upper()}")
        print(
            f'   - Success rate: {best_backend[1]["successful"]
                                  }/{total_examples_per_backend} examples'
        )
        print(
            f'   - Average time per task: {best_backend[1]
                                           ["total_time"]/total_examples_per_backend:.2f}s'
        )

    print(f"\n⏱️  Total benchmark time: {total_time:.2f} seconds")

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
        print("✅ Restored qme.json config file")

    if success_rate > 0.7:  # 70% success rate threshold
        print("🎉 Overall demo successful! New CLI commands working properly.")
        return True
    else:
        print(f"⚠️  Demo completed with {success_rate:.1%} success rate.")
        print("   Some backends may need attention or dependencies.")
        return success_rate > 0.3  # At least 30% working


if __name__ == "__main__":
    try:
        success = demo_cli()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        sys.exit(1)
