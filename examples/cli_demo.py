#!/usr/bin/env python3
"""QME CLI Demo - Comprehensive Backend Comparison."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Import common interface
from qme.example_utils import QMEExampleInterface, create_standard_epilog, setup_example_environment


def verify_frequency_output(
    cmd: list[str], stdout: str, stderr: str, cwd: Path
) -> tuple[bool, list[str]]:
    """Verify that frequency analysis was performed correctly.

    This function performs comprehensive validation of frequency analysis results:
    - Checks stdout for frequency summary
    - Validates JSON file structure
    - Verifies frequency values are reasonable
    - Checks TS validation results for TS optimizations
    - Validates thermodynamic properties

    Parameters
    ----------
    cmd : list[str]
        Command that was executed
    stdout : str
        Standard output from command
    stderr : str
        Standard error from command
    cwd : Path
        Working directory where output files should be created

    Returns
    -------
    tuple[bool, list[str]]
        (all_checks_passed, list_of_warnings)
    """
    warnings = []
    has_freq_flag = "--freq" in cmd or "--frequencies" in cmd

    if not has_freq_flag:
        return True, []  # No frequency analysis expected

    # Check for frequency summary in stdout
    freq_keywords = [
        "Frequency analysis completed",
        "Zero-point energy",
        "Free energy correction",
        "imaginary frequency",
        "imaginary frequencies",
    ]
    has_freq_summary = any(keyword.lower() in stdout.lower() for keyword in freq_keywords)

    if not has_freq_summary:
        warnings.append("Frequency summary not found in stdout")

    # Determine if this is a TS optimization
    is_ts_opt = "ts" in cmd or any("ts" in arg.lower() for arg in cmd)

    # Find output file from command
    output_file = None
    for i, arg in enumerate(cmd):
        if arg == "--output" and i + 1 < len(cmd):
            output_file = cmd[i + 1]
            break

    # Check for JSON file if we have output filename
    if output_file:
        # Handle both absolute and relative paths
        if os.path.isabs(output_file):
            json_file = Path(output_file).with_suffix(".json")
        else:
            json_file = (cwd / output_file).with_suffix(".json")
        if json_file.exists():
            # Verify JSON contains frequency_analysis and validate structure
            try:
                with open(json_file) as f:
                    json_data = json.load(f)
                    if "frequency_analysis" not in json_data:
                        warnings.append(
                            f"JSON file {json_file.name} exists but missing frequency_analysis"
                        )
                    else:
                        freq_analysis = json_data["frequency_analysis"]
                        if not isinstance(freq_analysis, dict):
                            warnings.append(
                                f"frequency_analysis in JSON is not a dictionary (got {type(freq_analysis)})"
                            )
                        else:
                            # Validate required keys
                            required_keys = [
                                "frequencies",
                                "zero_point_energy",
                                "thermodynamic_properties",
                            ]
                            for key in required_keys:
                                if key not in freq_analysis:
                                    warnings.append(
                                        f"Missing required key in frequency_analysis: {key}"
                                    )

                            # Validate frequency values
                            frequencies = freq_analysis.get("frequencies", [])
                            if isinstance(frequencies, list) and len(frequencies) > 0:
                                try:
                                    import numpy as np

                                    freq_array = np.array(frequencies)
                                    if np.any(np.isnan(freq_array)):
                                        warnings.append("Frequencies contain NaN values")
                                    if np.all(np.abs(freq_array) < 1e-6):
                                        warnings.append(
                                            "All frequencies are near zero (may indicate error)"
                                        )
                                    # Check for reasonable frequency range (should be between -5000 and 5000 cm^-1 typically)
                                    if np.any(np.abs(freq_array) > 10000):
                                        warnings.append(
                                            "Some frequencies are unusually large (>10000 cm^-1)"
                                        )
                                except (ValueError, TypeError, ImportError):
                                    warnings.append("Could not validate frequency values")

                            # Validate zero-point energy
                            zpe = freq_analysis.get("zero_point_energy")
                            if zpe is not None:
                                try:
                                    zpe_val = float(zpe)
                                    if zpe_val < 0 or zpe_val > 100:  # Reasonable range in eV
                                        warnings.append(
                                            f"Zero-point energy seems unreasonable: {zpe_val} eV"
                                        )
                                except (ValueError, TypeError):
                                    warnings.append("Zero-point energy is not a valid number")

                            # Validate thermodynamic properties
                            thermo = freq_analysis.get("thermodynamic_properties", {})
                            if not isinstance(thermo, dict):
                                warnings.append("thermodynamic_properties is not a dictionary")
                            else:
                                if "temperature" not in thermo:
                                    warnings.append(
                                        "Temperature missing from thermodynamic properties"
                                    )
                                if "entropy" not in thermo:
                                    warnings.append("Entropy missing from thermodynamic properties")

                            # For TS optimizations, check TS validation results
                            if is_ts_opt:
                                if "is_ts" not in freq_analysis:
                                    warnings.append(
                                        "is_ts key missing from frequency_analysis (TS optimization)"
                                    )
                                if "ts_analysis" not in freq_analysis:
                                    warnings.append(
                                        "ts_analysis key missing from frequency_analysis (TS optimization)"
                                    )
                                else:
                                    ts_analysis = freq_analysis.get("ts_analysis", {})
                                    if isinstance(ts_analysis, dict):
                                        if "n_imaginary_frequencies" not in ts_analysis:
                                            warnings.append(
                                                "n_imaginary_frequencies missing from ts_analysis"
                                            )
            except (json.JSONDecodeError, OSError) as e:
                warnings.append(f"Failed to read/parse JSON file {json_file.name}: {e}")
        else:
            warnings.append(f"Expected JSON file {json_file.name} not found")

    return len(warnings) == 0, warnings


def run_command(
    cmd: list[str], desc: str, backend: str, timeout: int = 600, verbose: bool = False
) -> tuple[bool, float, str, str]:
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
        return False, runtime, result.stdout, result.stderr

    except subprocess.TimeoutExpired as e:
        # Clean up the process if it's still running
        if hasattr(e, "subprocess") and e.subprocess:
            try:
                e.subprocess.kill()
                e.subprocess.wait(timeout=5)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                # Process already terminated or couldn't be killed
                # This is expected in some edge cases, so we silently continue
                pass
        return False, timeout, "", f"Command timed out after {timeout} seconds"
    except (subprocess.SubprocessError, OSError) as e:
        # Subprocess errors (failed to start, communication errors, etc.)
        # OSError covers file not found, permission errors, etc.
        return False, 0.0, "", f"Subprocess error: {e}"
    except Exception as e:
        # Catch-all for unexpected errors with context
        return False, 0.0, "", f"Unexpected error running command: {e}"


def create_example_commands(example_files: Path, backend: str, steps: int = 500) -> list[dict]:
    """Create example commands for a specific backend."""
    # Convert to absolute path for commands to work from any directory
    example_files_abs = example_files.resolve()
    # Prefer custom TS file if present, otherwise fall back to bundled example
    ts_input = str(example_files_abs / "A_C_A_B_A_C_ts.xyz")

    commands = [
        {
            "desc": "Structure optimization using 'minima' command with frequency analysis",
            "cmd": [
                "qme",
                "minima",
                "--strategy",
                "local",
                str(example_files_abs / "A_C_A_B_A_C_reactant.xyz"),
                "--backend",
                backend,
                "--steps",
                str(steps),
                "--freq",
                "--output",
                f"test_opt_{backend}.xyz",
            ],
        },
        {
            "desc": "Transition state optimization using 'ts' command with local strategy and frequency analysis",
            "cmd": [
                "qme",
                "ts",
                "--strategy",
                "local",
                str(example_files_abs / "A_C_A_B_A_C_ts.xyz"),
                "--backend",
                backend,
                "--steps",
                str(steps),
                "--freq",
                "--output",
                f"test_tsopt_{backend}.xyz",
            ],
        },
        {
            "desc": "Two-ended minima optimization using 'minima' command with interpolate strategy and frequency analysis",
            "cmd": [
                "qme",
                "minima",
                "--strategy",
                "interpolate",
                str(example_files_abs / "A_C_A_B_A_C_reactant.xyz"),
                "--product",
                str(example_files_abs / "A_C_A_B_A_C_product.xyz"),
                "--backend",
                backend,
                "--steps",
                str(steps),
                "--npoints",
                "3",
                "--interp",
                "geodesic",
                "--freq",
                "--output",
                f"test_twoended_{backend}.xyz",
            ],
        },
        {
            "desc": "Two-ended TS optimization using 'ts' command with interpolate strategy and frequency analysis",
            "cmd": [
                "qme",
                "ts",
                "--strategy",
                "interpolate",
                str(example_files_abs / "A_C_A_B_A_C_reactant.xyz"),
                "--product",
                str(example_files_abs / "A_C_A_B_A_C_product.xyz"),
                "--backend",
                backend,
                "--steps",
                str(steps),
                "--npoints",
                "7",
                "--interp",
                "geodesic",
                "--freq",
                "--output",
                f"test_ts_twoended_{backend}.xyz",
            ],
        },
        {
            "desc": "Two-ended TS optimization using 'ts' command with growing_string strategy and frequency analysis",
            "timeout": 1800,
            "cmd": [
                "qme",
                "ts",
                "--strategy",
                "growing_string",
                str(example_files_abs / "A_C_A_B_A_C_reactant.xyz"),
                "--product",
                str(example_files_abs / "A_C_A_B_A_C_product.xyz"),
                "--backend",
                backend,
                "--steps",
                "200",
                "--npoints",
                "15",
                "--step-size",
                "0.1",
                "--distance-threshold",
                "0.1",
                "--freq",
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
                str(example_files_abs / "A_C_A_B_A_C_reactant.xyz"),
                str(example_files_abs / "A_C_A_B_A_C_product.xyz"),
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
            "desc": "NEB path optimization (saves complete reaction pathway) using 'path' command with neb strategy and frequency analysis",
            "cmd": [
                "qme",
                "path",
                "--strategy",
                "neb",
                str(example_files_abs / "A_C_A_B_A_C_reactant.xyz"),
                str(example_files_abs / "A_C_A_B_A_C_product.xyz"),
                "--backend",
                backend,
                "--steps",
                "50",  # Reduced steps for faster demo
                "--npoints",
                "3",
                "--spring-constant",
                "0.5",  # Lower spring constant for better convergence
                "--freq",
                "--output",
                f"test_neb_{backend}.xyz",
            ],
        },
        {
            "desc": "CI-NEB path optimization (saves complete reaction pathway) using 'path' command with cineb strategy and frequency analysis",
            "cmd": [
                "qme",
                "path",
                "--strategy",
                "cineb",
                str(example_files_abs / "A_C_A_B_A_C_reactant.xyz"),
                str(example_files_abs / "A_C_A_B_A_C_product.xyz"),
                "--backend",
                backend,
                "--steps",
                "50",  # Reduced steps for faster demo
                "--npoints",
                "3",
                "--spring-constant",
                "0.5",  # Lower spring constant for better convergence
                "--freq",
                "--output",
                f"test_cineb_{backend}.xyz",
            ],
        },
        {
            "desc": "IRC path from transition state using 'path' command with irc strategy and frequency analysis",
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
                "--freq",
                "--output",
                f"test_irc_{backend}.xyz",
            ],
        },
        {
            "desc": "TS optimization with RFO optimizer and frequency analysis",
            "cmd": [
                "qme",
                "ts",
                "--strategy",
                "local",
                ts_input,
                "--backend",
                backend,
                "--local-optimizer",
                "rfo",
                "--steps",
                str(steps),
                "--freq",
                "--output",
                f"test_tsopt_rfo_{backend}.xyz",
            ],
        },
        {
            "desc": "Minima optimization with force-finite-diff-hessian flag and frequency analysis",
            "cmd": [
                "qme",
                "minima",
                "--strategy",
                "local",
                str(example_files_abs / "A_C_A_B_A_C_reactant.xyz"),
                "--backend",
                backend,
                "--steps",
                str(steps),
                "--force-finite-diff-hessian",
                "--freq",
                "--output",
                f"test_opt_fdhess_{backend}.xyz",
            ],
        },
        {
            "desc": "TS optimization with force-finite-diff-hessian flag and frequency analysis",
            "cmd": [
                "qme",
                "ts",
                "--strategy",
                "local",
                ts_input,
                "--backend",
                backend,
                "--steps",
                str(steps),
                "--force-finite-diff-hessian",
                "--freq",
                "--output",
                f"test_tsopt_fdhess_{backend}.xyz",
            ],
        },
        {
            "desc": "Minima optimization with custom temperature (500 K) and frequency analysis",
            "cmd": [
                "qme",
                "minima",
                "--strategy",
                "local",
                str(example_files_abs / "A_C_A_B_A_C_reactant.xyz"),
                "--backend",
                backend,
                "--steps",
                str(steps),
                "--temperature",
                "500.0",
                "--freq",
                "--output",
                f"test_opt_temp500_{backend}.xyz",
            ],
        },
        {
            "desc": "TS optimization with constraints and frequency analysis",
            "cmd": [
                "qme",
                "ts",
                "--strategy",
                "local",
                ts_input,
                "--backend",
                backend,
                "--steps",
                str(steps),
                "--constraints",
                "fix 0",  # Fix first atom
                "--freq",
                "--output",
                f"test_tsopt_constrained_{backend}.xyz",
            ],
        },
        {
            "desc": "Minima optimization with constraints and frequency analysis",
            "cmd": [
                "qme",
                "minima",
                "--strategy",
                "local",
                str(example_files_abs / "A_C_A_B_A_C_reactant.xyz"),
                "--backend",
                backend,
                "--steps",
                str(steps),
                "--constraints",
                "fix 0",  # Fix first atom
                "--freq",
                "--output",
                f"test_opt_constrained_{backend}.xyz",
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
            and not c["desc"].startswith("TS optimization")
        ]

    return commands


def demo_cli(
    backends: list[str] | None = None,
    interface: QMEExampleInterface | None = None,
    timeout: int = 600,
    skip_slow_backends: bool = False,
    command_indices: list[int] | None = None,
    command_pattern: str | None = None,
) -> bool:
    """Demonstrate QME CLI with commands using default settings."""
    if interface is None:
        interface = QMEExampleInterface("CLI Demo", "Comprehensive Backend Comparison")

    interface.print_header("Testing: opt, tsopt, two-ended, GSM, NEB, CI-NEB, and IRC commands")

    # Use provided backends or detect available ones
    if backends:
        available_backends = backends
    else:
        # Use interface to get available backends
        _, available_backends = interface.select_backend(
            requested_backends=None,
            verbose=interface.verbose if hasattr(interface, "verbose") else 1,
        )
        if not available_backends:
            from qme.backends.availability import get_available_backends

            available_backends = get_available_backends()

    if not available_backends:
        interface.print_error("No ML backends available for comparison.")
        return False

    # Skip slow backends if requested
    slow_backends = {"mace"}  # Known slow backends
    if skip_slow_backends:
        available_backends = [b for b in available_backends if b not in slow_backends]
        if not available_backends:
            interface.print_error("No backends remaining after skipping slow backends.")
            return False
        interface.print_warning(f"Skipping slow backends: {', '.join(slow_backends)}")

    interface.print_backend_summary(available_backends, "Testing Backends")

    print("\nChecking directory structure...", flush=True)
    # Find examples directory relative to script location
    script_dir = Path(__file__).parent
    examples_dir = script_dir
    if examples_dir.name != "examples":
        # If script is in examples/, we're already there
        # Otherwise, look for examples/ directory
        potential_examples = script_dir / "examples"
        if potential_examples.exists():
            examples_dir = potential_examples

    if not examples_dir.exists():
        print(f"ERROR: examples directory not found at {examples_dir.absolute()}", flush=True)
        return False
    print(f"✓ Found examples directory at {examples_dir.absolute()}", flush=True)

    example_files = examples_dir / "example_files"
    if not example_files.exists():
        print(f"ERROR: example_files directory not found at {example_files.absolute()}", flush=True)
        return False
    print(f"✓ Found example_files directory at {example_files.absolute()}", flush=True)

    # Check required files exist
    required_files = [
        "A_C_A_B_A_C_reactant.xyz",
        "A_C_A_B_A_C_product.xyz",
        "A_C_A_B_A_C_ts.xyz",
    ]

    for filename in required_files:
        if not (example_files / filename).exists():
            print(
                f"ERROR: Required file not found: {(example_files / filename).absolute()}",
                flush=True,
            )
            return False
    print("✓ All required files found", flush=True)

    # Performance tracking
    backend_results = {}
    total_start_time = time.time()
    # Complete matrix includes: minima+local, minima+interpolate, ts+local, ts+interpolate, ts+growing_string,
    # path+interpolate, path+neb, path+cineb, path+irc, ts+rfo, force-finite-diff-hessian (minima+ts),
    # custom temperature, constraints (minima+ts)
    total_examples_per_backend = 15
    steps = 500  # Reduced steps for faster testing

    print(f"\nStarting benchmark with {len(available_backends)} backend(s)...", flush=True)

    # Run examples for each backend
    for backend in available_backends:
        print(f"\n{'=' * 80}", flush=True)
        print(f"Testing backend: {backend}", flush=True)
        print(f"{'=' * 80}", flush=True)
        backend_results[backend] = {
            "examples": [],
            "total_time": 0.0,
            "successful": 0,
            "failed": 0,
        }

        examples = create_example_commands(example_files, backend, steps)

        # Filter commands if requested
        if command_indices is not None:
            # Filter by 1-based indices
            filtered_examples = []
            for idx in command_indices:
                if 1 <= idx <= len(examples):
                    filtered_examples.append((idx, examples[idx - 1]))
                else:
                    interface.print_warning(f"Command index {idx} out of range (1-{len(examples)})")
            examples = [ex for _, ex in sorted(filtered_examples)]
            print(
                f"Filtered to {len(examples)} command(s) by indices: {command_indices}", flush=True
            )
        elif command_pattern is not None:
            # Filter by pattern matching description
            filtered_examples = []
            pattern_lower = command_pattern.lower()
            for i, example in enumerate(examples, 1):
                if pattern_lower in example["desc"].lower():
                    filtered_examples.append((i, example))
            if filtered_examples:
                examples = [ex for _, ex in filtered_examples]
                print(
                    f"Filtered to {len(examples)} command(s) matching pattern '{command_pattern}'",
                    flush=True,
                )
            else:
                interface.print_warning(f"No commands found matching pattern '{command_pattern}'")
                print("Available commands:", flush=True)
                for i, ex in enumerate(examples, 1):
                    print(f"  {i}: {ex['desc']}", flush=True)
                return False

        print(f"Running {len(examples)} example command(s)", flush=True)

        backend_start_time = time.time()

        for i, example in enumerate(examples, 1):
            print(
                f"\n[{backend}] Running example {i}/{len(examples)}: {example['desc']}", flush=True
            )
            success, runtime, stdout, stderr = run_command(
                example["cmd"],
                example["desc"],
                backend,
                timeout=example.get("timeout", timeout),
            )

            # Verify frequency analysis if --freq flag is present
            freq_verified = True
            freq_warnings = []
            if "--freq" in example["cmd"] or "--frequencies" in example["cmd"]:
                cwd = Path(__file__).parent.parent
                freq_verified, freq_warnings = verify_frequency_output(
                    example["cmd"], stdout, stderr, cwd
                )
                if not freq_verified:
                    print("  ⚠ Frequency verification warnings:", flush=True)
                    for warning in freq_warnings:
                        print(f"    - {warning}", flush=True)

            backend_results[backend]["examples"].append(
                {
                    "desc": example["desc"],
                    "success": success,
                    "runtime": runtime,
                    "stdout": stdout[:200] + "..." if len(stdout) > 200 else stdout,
                    "stderr": stderr[:200] + "..." if len(stderr) > 200 else stderr,
                    "freq_verified": freq_verified,
                    "freq_warnings": freq_warnings,
                },
            )

            if success:
                backend_results[backend]["successful"] += 1
                status_msg = f"  ✓ Success (runtime: {runtime:.2f}s)"
                if not freq_verified:
                    status_msg += " [frequency verification warnings]"
                print(status_msg, flush=True)
            else:
                backend_results[backend]["failed"] += 1
                print(f"  ✗ Failed (runtime: {runtime:.2f}s)", flush=True)
                if stderr:
                    print(f"    Error: {stderr[:200]}", flush=True)

            # Brief pause between examples
            time.sleep(0.5)

        backend_results[backend]["total_time"] = time.time() - backend_start_time

        # Backend summary
        print(f"\n{backend} Summary:", flush=True)
        print(f"  Successful: {backend_results[backend]['successful']}/{len(examples)}", flush=True)
        print(f"  Failed: {backend_results[backend]['failed']}/{len(examples)}", flush=True)
        print(f"  Total time: {backend_results[backend]['total_time']:.2f}s", flush=True)

    total_time = time.time() - total_start_time
    print(f"\nOverall completion time: {total_time:.2f}s", flush=True)

    # Check if all backends had some success
    total_successful = sum(results["successful"] for results in backend_results.values())
    total_tests = len(available_backends) * total_examples_per_backend

    return total_successful == total_tests


@setup_example_environment
def main() -> int | None:
    """Parse arguments and run CLI demo."""
    # Create standardized interface
    interface = QMEExampleInterface(
        name="CLI Demo",
        description="Comprehensive Backend Comparison",
        epilog=create_standard_epilog("demo"),
    )

    parser = interface.create_parser()
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds for each command (default: 600)",
    )
    parser.add_argument(
        "--skip-slow-backends",
        action="store_true",
        help="Skip known slow backends (e.g., mace) to speed up testing",
    )
    parser.add_argument(
        "--command",
        type=int,
        action="append",
        dest="command_indices",
        help="Run specific command(s) by index (1-based). Can be specified multiple times. Example: --command 4 --command 5",
    )
    parser.add_argument(
        "--command-pattern",
        type=str,
        help="Run commands matching a pattern in their description. Example: --command-pattern 'TS optimization'",
    )
    parser.add_argument(
        "--list-commands",
        action="store_true",
        help="List all available commands and exit",
    )
    args = parser.parse_args()

    # Set up logging based on verbosity level
    interface.setup_logging(args.verbose)

    # List commands if requested
    if args.list_commands:
        script_dir = Path(__file__).parent
        example_files = script_dir / "example_files"
        if example_files.exists():
            # Use a dummy backend to generate command list
            commands = create_example_commands(example_files, "dummy_backend", steps=100)
            print("\nAvailable CLI Demo Commands:")
            print("=" * 80)
            for i, cmd in enumerate(commands, 1):
                print(f"{i:2d}. {cmd['desc']}")
                print(f"    Command: {' '.join(cmd['cmd'][:8])}...")
            print("=" * 80)
            print("\nUsage examples:")
            print("  # Run command 4 only:")
            print("  python cli_demo.py --backends mace --command 4")
            print("\n  # Run commands 4 and 5:")
            print("  python cli_demo.py --backends mace --command 4 --command 5")
            print("\n  # Run all TS optimization commands:")
            print("  python cli_demo.py --backends mace --command-pattern 'TS optimization'")
            print("\n  # Run with longer timeout for slow commands:")
            print("  python cli_demo.py --backends mace --command 4 --timeout 1200")
            return 0
        else:
            interface.print_error("example_files directory not found")
            return 1

    # Backend handling
    requested = [b.strip() for b in args.backends.split(",")] if args.backends else None
    _, backends = interface.select_backend(
        requested_backends=requested,
        verbose=args.verbose,
    )
    if not backends:
        backends = None  # Will auto-detect in demo_cli

    try:
        success = demo_cli(
            backends,
            interface,
            timeout=args.timeout,
            skip_slow_backends=args.skip_slow_backends,
            command_indices=args.command_indices,
            command_pattern=args.command_pattern,
        )
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
