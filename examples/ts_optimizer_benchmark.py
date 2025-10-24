#!/usr/bin/env python3
"""QME TS Optimizer Benchmark - Transition State Optimizer Comparison.

This benchmark compares the performance of different transition state optimizers
(sella, trust-krylov-ts) for transition state finding using various QME ML backends.
It focuses specifically on TS optimization to evaluate which optimizers work best
for finding transition states across different ML backends.

Usage:
    python ts_optimizer_benchmark.py [--backends BACKEND1,BACKEND2,...]
    python ts_optimizer_benchmark.py [--optimizers OPT1,OPT2,...]
    python ts_optimizer_benchmark.py [--device DEVICE]

Features:
    - Transition state optimizer comparison (sella, trust-krylov-ts)
    - All available ML backends tested
    - Detailed timing and convergence analysis
    - TS-specific optimization evaluation
    - Focus on TS finding capabilities
"""

import json
import sys
import warnings
from pathlib import Path
from typing import Any

from ase import Atoms

# Import QME components
try:
    pass  # QME components imported via benchmark_optimization function
except ImportError:
    sys.exit(1)

# Backend availability helpers

# Common interface and device utils
from qme.example_utils import QMEExampleInterface, benchmark_optimization, create_standard_epilog

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def create_ts_structure() -> Atoms:
    """Create a transition state structure for TS optimization using example files."""
    from pathlib import Path

    from ase.io import read

    # Use the actual TS structure from example files
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    return read(script_dir / "example_files" / "A_C_A_B_A_C_ts.xyz")


def benchmark_ts_optimizer(
    backend: str,
    optimizer: str,
    device: str | None = None,
    model_name: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Benchmark a single backend with a specific optimizer for transition state optimization.

    Suitable optimizers: Sella, Trust-Krylov-TS
    """
    return benchmark_optimization(
        backend=backend,
        optimizer=optimizer,
        device=device,
        model_name=model_name,
        verbose=verbose,
        test_ts=True,
        create_structure_func=create_ts_structure,
        suitable_optimizers=["sella", "trust-krylov-ts"],
    )


def print_frequency_analysis_summary(results_list: list[dict[str, Any]]) -> None:
    """Print a detailed frequency analysis summary for TS optimization."""
    # Print legend

    # Header

    # Results
    for results in results_list:
        if results["available"] and "frequency_results" in results:
            freq_results = results["frequency_results"]
            n_imag = freq_results.get("n_imaginary_frequencies", 0)
            is_valid = freq_results.get("is_valid_result", False)
            freq_results.get("zero_point_energy", 0)
            frequencies = freq_results.get("frequencies", [])

            # Format first 3 frequencies
            if len(frequencies) >= 3:
                f"[{frequencies[0]:.1f}, {frequencies[1]:.1f}, {frequencies[2]:.1f}]"
            else:
                pass

            # Status indicator
            if is_valid or n_imag != 1:
                pass
            else:
                pass

        else:
            pass

    # Summary statistics
    available_results = [r for r in results_list if r["available"] and "frequency_results" in r]
    if available_results:
        # Overall validation rate
        valid_count = sum(
            1 for r in available_results if r["frequency_results"].get("is_valid_result", False)
        )
        total_count = len(available_results)
        (valid_count / total_count * 100) if total_count > 0 else 0

        # TS-specific issues
        ts_results = available_results  # All results are TS results in this benchmark
        if ts_results:
            ts_valid = sum(
                1 for r in ts_results if r["frequency_results"].get("is_valid_result", False)
            )
            (ts_valid / len(ts_results) * 100) if ts_results else 0

            # TS-specific issues
            ts_with_wrong_freq = sum(
                1
                for r in ts_results
                if r["frequency_results"].get("n_imaginary_frequencies", 0) != 1
            )
            if ts_with_wrong_freq > 0:
                pass


def print_optimizer_summary(results_list: list[dict[str, Any]]) -> None:
    """Print a summary table focused on TS optimizer comparison."""
    # Print legend first, before the table
    print(f"\n{'=' * 120}")
    print("TS OPTIMIZER COMPARISON")
    print(f"{'=' * 120}")
    print("Legend: ✅ = Success, ❌ = Failed, ⚠️ = Warning, ⏱️ = Time")

    # Header
    print(
        f"\n{'Backend':<12} {'Optimizer':<15} {'Status':<8} {'Total Time':<12} {'Opt Steps':<10} "
        f"{'Time/Step':<12} {'Final Energy':<15} {'Max Force':<12} {'Valid':<8}"
    )
    print("=" * 120)

    # Results
    for results in results_list:
        if results["available"]:
            timings = results["timings"]
            opt_results = results.get("optimization_results", {})
            freq_results = results.get("frequency_results", {})
            steps_taken = opt_results.get("steps_taken", 0)
            time_per_step = timings.get("avg_time_per_step", 0)
            backend = results.get("backend", "unknown")
            optimizer = results.get("optimizer", "unknown")
            converged = opt_results.get("converged", False)
            final_energy = opt_results.get("final_energy", None)
            max_force = opt_results.get("max_force", None)
            is_valid = freq_results.get("is_valid_result", False)

            # Handle None values for formatting
            steps_str = str(steps_taken) if steps_taken is not None else "N/A"
            energy_str = f"{final_energy:.6f}" if final_energy is not None else "N/A"
            force_str = f"{max_force:.6f}" if max_force is not None else "N/A"
            valid_str = "✅" if is_valid else "❌"
            status = "✅" if converged else "❌"

            print(
                f"{backend:<12} {optimizer:<15} {status:<8} {timings.get('total', 0):<12.3f} {steps_str:<10} "
                f"{time_per_step:<12.6f} {energy_str:<15} {force_str:<12} {valid_str:<8}"
            )

        else:
            backend = results.get("backend", "unknown")
            optimizer = results.get("optimizer", "unknown")
            status = "❌"
            print(
                f"{backend:<12} {optimizer:<15} {status:<8} {'N/A':<12} {'N/A':<10} "
                f"{'N/A':<12} {'N/A':<15} {'N/A':<12} {'N/A':<8}"
            )

    # Optimizer performance analysis
    available_results = [r for r in results_list if r["available"]]
    if available_results:
        # Group by optimizer
        optimizer_groups = {}
        for result in available_results:
            opt_name = result.get("optimizer", "unknown")
            if opt_name not in optimizer_groups:
                optimizer_groups[opt_name] = []
            optimizer_groups[opt_name].append(result)

        for _opt_name, opt_results in optimizer_groups.items():
            # Calculate statistics - filter out None values
            steps_list = [
                r["optimization_results"].get("steps_taken", 0)
                for r in opt_results
                if r["optimization_results"].get("steps_taken") is not None
            ]
            time_per_step_list = [
                r["timings"].get("avg_time_per_step", 0)
                for r in opt_results
                if r["timings"].get("avg_time_per_step") is not None
            ]
            total_time_list = [
                r["timings"].get("optimization", 0)
                for r in opt_results
                if r["timings"].get("optimization") is not None
            ]
            converged_list = [
                r["optimization_results"].get("converged", False) for r in opt_results
            ]

            if steps_list:
                pass

            if time_per_step_list:
                pass

            if total_time_list:
                pass

            if converged_list:
                sum(converged_list) / len(converged_list) * 100

            # Quality analysis for TS optimization
            valid_result_list = [
                r["frequency_results"].get("is_valid_result", False)
                for r in opt_results
                if "frequency_results" in r
            ]
            if valid_result_list:
                sum(valid_result_list) / len(valid_result_list) * 100


def print_performance_summary(results_list: list[dict[str, Any]]) -> None:
    """Print comprehensive performance profiler data."""
    # Check for performance data
    has_performance_data = any(
        results.get("available") and "performance" in results for results in results_list
    )

    if not has_performance_data:
        return

    # Section 1: Calculator Calls & Memory

    # Header

    # Results
    for results in results_list:
        if results.get("available") and "performance" in results:
            perf = results["performance"]
            perf.get("calculator_calls", {})
            perf.get("memory", {})

    # Section 2: Detailed Timing Breakdown

    # Collect all unique timing sections from all results
    all_sections = set()
    for results in results_list:
        if results.get("available") and "performance" in results:
            perf = results["performance"]
            timings = perf.get("timings", {})
            all_sections.update(timings.keys())

    if not all_sections:
        return

    # Header for detailed timing

    # Show timing statistics for each section
    for section in sorted(all_sections):
        section_stats = []
        for results in results_list:
            if results.get("available") and "performance" in results:
                perf = results["performance"]
                timings = perf.get("timings", {})
                if section in timings:
                    section_stats.append(timings[section])

        if section_stats:
            # Calculate aggregate statistics across all results
            sum(stat.get("total_time", 0.0) for stat in section_stats)
            total_count = sum(stat.get("count", 0) for stat in section_stats)
            (
                sum(stat.get("avg_time", 0.0) * stat.get("count", 0) for stat in section_stats)
                / total_count
                if total_count > 0
                else 0.0
            )
            min(stat.get("min_time", 0.0) for stat in section_stats)
            max(stat.get("max_time", 0.0) for stat in section_stats)


def save_results(results_list: list[dict[str, Any]], output_file: str) -> None:
    """Save benchmark results to JSON file."""
    # If output_file is just a filename, save it in the examples directory
    if not Path(output_file).is_absolute() and "/" not in output_file:
        output_path = Path(__file__).parent / output_file
    else:
        output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results_list, f, indent=2, default=str)


def main() -> int:
    """Main function to run the TS optimizer comparison benchmark."""
    # Create standardized interface
    interface = QMEExampleInterface(
        name="TS Optimizer Benchmark",
        description="Transition State Optimizer Comparison",
        epilog=create_standard_epilog("benchmark"),
    )

    parser = interface.create_parser()

    # Add optimizer-specific arguments
    parser.add_argument(
        "--optimizers",
        type=str,
        help="Comma-separated list of optimizers to benchmark (default: sella, options: sella,trust-krylov-ts)",
    )

    args = parser.parse_args()

    interface.print_header()

    # Set up logging based on verbosity level
    interface.setup_logging(args.verbose)

    # Parse backends if provided
    if args.backends:
        available_backends = [b.strip() for b in args.backends.split(",")]
    else:
        from qme.backends.availability import get_available_backends
        available_backends = get_available_backends()

    # Determine which optimizers to test
    if args.optimizers:
        requested_optimizers = [o.strip().lower() for o in args.optimizers.split(",")]
        # Filter to only TS optimizers
        valid_optimizers = ["sella", "trust-krylov-ts"]
        ts_optimizers = [opt for opt in requested_optimizers if opt in valid_optimizers]
        if len(ts_optimizers) != len(requested_optimizers):
            [opt for opt in requested_optimizers if opt not in valid_optimizers]
    else:
        ts_optimizers = ["sella"]

    if not ts_optimizers:
        interface.print_error("No valid TS optimizers specified!")
        return 1

    interface.print_backend_summary(available_backends, "Benchmarking Backends")

    # Get device info
    device = interface.get_device_info(args.device)

    config = {
        "Device": device,
        "Output": args.output or interface.get_default_output_file(),
        "Verbose": args.verbose,
        "Test Types": "Transition State Optimization",
    }
    interface.print_configuration(config)

    len(available_backends) * len(ts_optimizers)

    # Run benchmarks
    results_list = []

    for backend in available_backends:
        for optimizer in ts_optimizers:
            try:
                results = benchmark_ts_optimizer(
                    backend=backend,
                    optimizer=optimizer,
                    device=device,
                    verbose=args.verbose,
                )
                results_list.append(results)
            except KeyboardInterrupt:
                break
            except Exception as e:
                results_list.append(
                    {
                        "backend": backend,
                        "optimizer": optimizer,
                        "device": device,
                        "test_ts": True,
                        "available": False,
                        "error": str(e),
                        "timings": {},
                        "optimization_results": {},
                        "frequency_results": {},
                    },
                )

    # Print summaries
    print_frequency_analysis_summary(results_list)
    print_optimizer_summary(results_list)
    print_performance_summary(results_list)

    # Save results
    save_results(results_list, args.output or interface.get_default_output_file())

    interface.print_success()
    return 0


if __name__ == "__main__":
    sys.exit(main())
