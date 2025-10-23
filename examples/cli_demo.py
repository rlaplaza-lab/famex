#!/usr/bin/env python3
"""QME CLI Demo - Comprehensive Backend Comparison.

This example demonstrates QME's command-line interface capabilities by running
various optimization tasks across all available ML backends and comparing their
performance and reliability.

Features:
    - Structure optimization using 'minima' command
    - Transition state optimization using 'ts' command with multiple strategies:
      * Local TS optimization
      * Two-ended TS optimization via interpolation
      * Growing String Method (GSM) for TS optimization
    - Two-ended optimization workflows
    - Reaction path optimization using 'path' command with different strategies:
      * Raw interpolation path generation (interpolate strategy)
      * NEB path optimization (neb strategy)
      * CI-NEB (Climbing Image NEB) path optimization (cineb strategy)
      * IRC path from transition state (irc strategy)
    - Trajectory saving for multi-image results
    - Comprehensive backend performance comparison
"""

import os
import subprocess
import sys
import time
from pathlib import Path

# Disable ASE GUI to prevent popup windows
os.environ["DISPLAY"] = ""
os.environ["MPLBACKEND"] = "Agg"

# Import common interface
try:
    from qme.examples import QMEExampleInterface, create_standard_epilog
except ImportError:
    sys.exit(1)


def print_backend_summary(backends: list[str], title: str = "Available Backends") -> None:
    """Print a formatted summary of backends."""
    for _i, _backend in enumerate(backends, 1):
        pass


def run_command(cmd, desc, backend, timeout=600) -> tuple[bool, float, str, str]:
    """Run a CLI command and report results."""
    try:
        start_time = time.time()
        # Create a modified environment with GUI disabled
        env = os.environ.copy()
        env["DISPLAY"] = ""
        env["MPLBACKEND"] = "Agg"

        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent.parent,  # Run from qme root
            env=env,  # Pass the modified environment
        )
        end_time = time.time()
        runtime = end_time - start_time

        if result.returncode == 0:
            return True, runtime, result.stdout, result.stderr
        if result.stderr:
            pass
        return False, runtime, result.stdout, result.stderr

    except subprocess.TimeoutExpired as e:
        # Clean up the process if it's still running
        if hasattr(e, "subprocess") and e.subprocess:
            try:
                e.subprocess.kill()
                e.subprocess.wait(timeout=5)
            except Exception:
                pass
        return False, timeout, "", f"Command timed out after {timeout} seconds"
    except Exception as e:
        return False, 0.0, "", str(e)


def create_example_commands(example_files: Path, backend: str, steps: int = 500) -> list[dict]:
    """Create example commands for a specific backend using default settings.

    Notes:
    -----
    - TS optimizations are not supported on the 'mock' backend. We skip
      'tsopt' examples when backend == 'mock' to avoid hard errors. Timeouts
      are acceptable for heavy runs, but CLI argument mistakes are not.

    """
    # Prefer custom TS file if present, otherwise fall back to bundled example
    ts_input = str(example_files / "A_C_A_B_A_C_ts.xyz")

    commands = [
        {
            "desc": "Structure optimization using 'minima' command",
            "cmd": [
                "qme",
                "minima",
                "--strategy",
                "local",
                str(example_files / "A_C_A_B_A_C_reactant.xyz"),
                "--backend",
                backend,
                "--steps",
                str(steps),
                "--output",
                f"test_opt_{backend}.xyz",
            ],
        },
        {
            "desc": "Transition state optimization using 'ts' command with local strategy",
            "cmd": [
                "qme",
                "ts",
                "--strategy",
                "local",
                str(example_files / "A_C_A_B_A_C_ts.xyz"),
                "--backend",
                backend,
                "--steps",
                str(steps),
                "--validate-ts",
                "--output",
                f"test_tsopt_{backend}.xyz",
            ],
        },
        {
            "desc": "Two-ended minima optimization using 'minima' command with interpolate strategy",
            "cmd": [
                "qme",
                "minima",
                "--strategy",
                "interpolate",
                str(example_files / "A_C_A_B_A_C_reactant.xyz"),
                "--product",
                str(example_files / "A_C_A_B_A_C_product.xyz"),
                "--backend",
                backend,
                "--steps",
                str(steps),
                "--npoints",
                "3",
                "--interp",
                "geodesic",
                "--output",
                f"test_twoended_{backend}.xyz",
            ],
        },
        {
            "desc": "Two-ended TS optimization using 'ts' command with interpolate strategy",
            "cmd": [
                "qme",
                "ts",
                "--strategy",
                "interpolate",
                str(example_files / "A_C_A_B_A_C_reactant.xyz"),
                "--product",
                str(example_files / "A_C_A_B_A_C_product.xyz"),
                "--backend",
                backend,
                "--steps",
                str(steps),
                "--npoints",
                "7",
                "--interp",
                "geodesic",
                "--validate-ts",
                "--output",
                f"test_ts_twoended_{backend}.xyz",
            ],
        },
        {
            "desc": "Two-ended TS optimization using 'ts' command with growing_string strategy",
            "cmd": [
                "qme",
                "ts",
                "--strategy",
                "growing_string",
                str(example_files / "A_C_A_B_A_C_reactant.xyz"),
                "--product",
                str(example_files / "A_C_A_B_A_C_product.xyz"),
                "--backend",
                backend,
                "--steps",
                str(steps),
                "--npoints",
                "7",
                "--step-size",
                "0.1",
                "--max-images",
                "50",
                "--distance-threshold",
                "0.1",
                "--validate-ts",
                "--output",
                f"test_ts_gsm_{backend}.xyz",
            ],
        },
        {
            "desc": "Raw interpolation path generation using 'path' command with interpolate strategy",
            "cmd": [
                "qme",
                "path",
                "--strategy",
                "interpolate",
                str(example_files / "A_C_A_B_A_C_reactant.xyz"),
                "--product",
                str(example_files / "A_C_A_B_A_C_product.xyz"),
                "--backend",
                backend,
                "--npoints",
                "5",
                "--interp",
                "geodesic",
                "--output",
                f"test_interpolate_{backend}.xyz",
            ],
        },
        {
            "desc": "NEB path optimization (saves complete reaction pathway) using 'path' command with neb strategy",
            "cmd": [
                "qme",
                "path",
                "--strategy",
                "neb",
                str(example_files / "A_C_A_B_A_C_reactant.xyz"),
                "--product",
                str(example_files / "A_C_A_B_A_C_product.xyz"),
                "--backend",
                backend,
                "--steps",
                "50",  # Reduced steps for faster demo
                "--npoints",
                "3",
                "--spring-constant",
                "0.5",  # Lower spring constant for better convergence
                "--output",
                f"test_neb_{backend}.xyz",
            ],
        },
        {
            "desc": "CI-NEB path optimization (saves complete reaction pathway) using 'path' command with cineb strategy",
            "cmd": [
                "qme",
                "path",
                "--strategy",
                "cineb",
                str(example_files / "A_C_A_B_A_C_reactant.xyz"),
                "--product",
                str(example_files / "A_C_A_B_A_C_product.xyz"),
                "--backend",
                backend,
                "--steps",
                "50",  # Reduced steps for faster demo
                "--npoints",
                "3",
                "--spring-constant",
                "0.5",  # Lower spring constant for better convergence
                "--output",
                f"test_cineb_{backend}.xyz",
            ],
        },
        {
            "desc": "IRC path from transition state using 'path' command with irc strategy",
            "cmd": [
                "qme",
                "path",
                "--strategy",
                "irc",
                ts_input,
                "--backend",
                backend,
                "--steps",
                "50",
                "--step-size",
                "0.1",
                "--direction",
                "both",
                "--validate-ts",
                "--output",
                f"test_irc_{backend}.xyz",
            ],
        },
    ]

    if backend.lower() == "mock":
        # Filter out TS optimizations which are unsupported on mock backend
        commands = [
            c
            for c in commands
            if not c["desc"].startswith("Transition state optimization")
            and not c["desc"].startswith("Two-ended TS optimization")
            and not c["desc"].startswith("IRC path from transition state")
        ]

    return commands


def demo_cli(backends: list[str] | None = None, interface: QMEExampleInterface = None):
    """Demonstrate QME CLI with commands using default settings."""
    if interface is None:
        interface = QMEExampleInterface("CLI Demo", "Comprehensive Backend Comparison")

    interface.print_header("Testing: opt, tsopt, two-ended, GSM, NEB, CI-NEB, and IRC commands")

    # Get available ML backends
    if backends:
        available_backends = interface.filter_available_backends(backends, verbose=1)
    else:
        available_backends = interface.get_available_ml_backends()

    if not available_backends:
        interface.print_error("No ML backends available for comparison.")
        return False

    interface.print_backend_summary(available_backends, "Testing Backends")

    # Ensure we're in the right directory structure
    examples_dir = Path("examples")
    if not examples_dir.exists():
        return False

    example_files = examples_dir / "example_files"
    if not example_files.exists():
        return False

    # Check required files exist
    required_files = [
        "A_C_A_B_A_C_reactant.xyz",
        "A_C_A_B_A_C_product.xyz",
        "A_C_A_B_A_C_ts.xyz",
    ]

    for filename in required_files:
        if not (example_files / filename).exists():
            return False

    # Performance tracking
    backend_results = {}
    total_start_time = time.time()
    total_examples_per_backend = 9  # Complete matrix: minima+local, minima+interpolate, ts+local, ts+interpolate, ts+growing_string, path+interpolate, path+neb, path+cineb, path+irc
    steps = 500  # Reduced steps for faster testing

    # Run examples for each backend
    for backend in available_backends:
        backend_results[backend] = {
            "examples": [],
            "total_time": 0.0,
            "successful": 0,
            "failed": 0,
        }

        examples = create_example_commands(example_files, backend, steps)

        backend_start_time = time.time()

        for _i, example in enumerate(examples, 1):
            success, runtime, stdout, stderr = run_command(example["cmd"], example["desc"], backend)

            backend_results[backend]["examples"].append(
                {
                    "desc": example["desc"],
                    "success": success,
                    "runtime": runtime,
                    "stdout": stdout[:200] + "..." if len(stdout) > 200 else stdout,
                    "stderr": stderr[:200] + "..." if len(stderr) > 200 else stderr,
                },
            )

            if success:
                backend_results[backend]["successful"] += 1
            else:
                backend_results[backend]["failed"] += 1

            # Brief pause between examples
            time.sleep(0.5)

        backend_results[backend]["total_time"] = time.time() - backend_start_time

        # Backend summary

    time.time() - total_start_time

    # Overall comparison summary

    for backend in available_backends:
        results = backend_results[backend]
        results["total_time"] / len(results["examples"]) if results["examples"] else 0

    # Find best performing backend
    successful_backends = [
        (backend, results)
        for backend, results in backend_results.items()
        if results["successful"] > 0
    ]

    if successful_backends:
        # Sort by success rate, then by speed
        best_backend = max(
            successful_backends,
            key=lambda x: (x[1]["successful"], -x[1]["total_time"]),
        )
        best_backend[1]["successful"] + best_backend[1]["failed"]

    # Check if all backends had some success
    total_successful = sum(results["successful"] for results in backend_results.values())
    total_tests = len(available_backends) * total_examples_per_backend

    success_rate = total_successful / total_tests if total_tests > 0 else 0

    if success_rate >= 0.8:
        return True
    return success_rate > 0.3  # At least 30% working


def main() -> int | None:
    """Main entry point with argument parsing."""
    # Create standardized interface
    interface = QMEExampleInterface(
        name="CLI Demo",
        description="Comprehensive Backend Comparison",
        epilog=create_standard_epilog("demo"),
    )

    parser = interface.create_parser()
    args = parser.parse_args()

    # Set up logging based on verbosity level
    interface.setup_logging(args.verbose)

    # Parse backends if provided
    backends = None
    if args.backends:
        backends = [b.strip() for b in args.backends.split(",")]

    try:
        success = demo_cli(backends, interface)
        return 0 if success else 1
    except KeyboardInterrupt:
        return 1
    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(main())
