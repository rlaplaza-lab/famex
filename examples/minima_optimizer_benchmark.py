#!/usr/bin/env python3
"""QME Minima Optimizer Benchmark - Minima Optimization Comparison.

This benchmark compares the performance of different minima optimizers
(lbfgs, bfgs, fire) across various QME ML backends. It focuses specifically
on minima optimization to evaluate which optimizers work best for finding
energy minima across different ML backends.

Usage:
    python minima_optimizer_benchmark.py [--backends BACKEND1,BACKEND2,...]
    python minima_optimizer_benchmark.py [--optimizers lbfgs,bfgs,fire]
    python minima_optimizer_benchmark.py [--device DEVICE]

Features:
    - Minima optimizer comparison (lbfgs, bfgs, fire)
    - All available ML backends tested
    - Detailed timing and convergence analysis
    - Minima-specific optimization evaluation
    - Focus on minima finding capabilities
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
from qme.examples import QMEExampleInterface, benchmark_optimization, create_standard_epilog

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def create_minima_structure() -> Atoms:
    """Create a structure for minima optimization using example files."""
    import os

    from ase.io import read

    # Use the smaller A_C_A_B_A_C reactant structure as starting point for minima optimization
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return read(os.path.join(script_dir, "example_files", "A_C_A_B_A_C_reactant.xyz"))


def benchmark_minima_optimizer(
    backend: str,
    optimizer: str,
    device: str | None = None,
    model_name: str | None = None,
    verbose: int = 1,
) -> dict[str, Any]:
    """Benchmark a single backend with a specific optimizer for minima optimization.

    Only suitable optimizers: LBFGS, BFGS, FIRE, Trust-Krylov
    """
    return benchmark_optimization(
        backend=backend,
        optimizer=optimizer,
        device=device,
        model_name=model_name,
        verbose=verbose,
        test_ts=False,
        create_structure_func=create_minima_structure,
        suitable_optimizers=["lbfgs", "bfgs", "fire", "trust-krylov"],
    )


def print_frequency_analysis_summary(results_list: list[dict[str, Any]]) -> None:
    """Print a detailed frequency analysis summary for minima optimization."""
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
            if is_valid or n_imag > 0:
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

        # Minima-specific issues
        minima_with_imag = sum(
            1
            for r in available_results
            if r["frequency_results"].get("n_imaginary_frequencies", 0) > 0
        )
        if minima_with_imag > 0:
            pass


def print_optimizer_summary(results_list: list[dict[str, Any]]) -> None:
    """Print a summary table focused on minima optimizer comparison."""
    # Print legend first, before the table

    # Header

    # Results
    for results in results_list:
        if results["available"]:
            timings = results["timings"]
            opt_results = results.get("optimization_results", {})
            freq_results = results.get("frequency_results", {})
            steps_taken = opt_results.get("steps_taken", 0)
            timings.get("avg_time_per_step", 0)
            results.get("optimizer", "unknown")
            opt_results.get("converged", False)
            opt_results.get("final_energy", None)
            opt_results.get("max_force", None)
            freq_results.get("is_valid_result", False)

            # Handle None values for formatting
            str(steps_taken) if steps_taken is not None else "N/A"

        else:
            results.get("optimizer", "unknown")

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

            # Quality analysis for minima optimization
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
    """Main function to run the minima optimizer comparison benchmark."""
    # Create standardized interface
    interface = QMEExampleInterface(
        name="Minima Optimizer Benchmark",
        description="Minima Optimization Comparison",
        epilog=create_standard_epilog("benchmark"),
    )

    parser = interface.create_parser()

    # Add optimizer-specific arguments
    parser.add_argument(
        "--optimizers",
        type=str,
        help="Comma-separated list of optimizers to benchmark (default: lbfgs,bfgs,fire,trust-krylov)",
    )

    args = parser.parse_args()

    interface.print_header()

    # Set up logging based on verbosity level
    interface.setup_logging(args.verbose)

    # Determine which backends to test
    if args.backends:
        requested_backends = [b.strip() for b in args.backends.split(",")]
        available_backends = interface.filter_available_backends(
            requested_backends,
            verbose=args.verbose,
        )

        if not available_backends:
            interface.print_error("No requested backends are available!")
            return 1
    else:
        available_backends = interface.get_available_ml_backends()

        if not available_backends:
            interface.print_error("No ML backends available!")
            return 1

    # Determine which optimizers to test
    if args.optimizers:
        requested_optimizers = [o.strip().lower() for o in args.optimizers.split(",")]
        # Filter to only minima optimizers
        valid_optimizers = ["lbfgs", "bfgs", "fire", "trust-krylov"]
        minima_optimizers = [opt for opt in requested_optimizers if opt in valid_optimizers]
        if len(minima_optimizers) != len(requested_optimizers):
            [opt for opt in requested_optimizers if opt not in valid_optimizers]
    else:
        minima_optimizers = ["lbfgs", "bfgs", "fire", "trust-krylov"]

    if not minima_optimizers:
        interface.print_error("No valid minima optimizers specified!")
        return 1

    interface.print_backend_summary(available_backends, "Benchmarking Backends")

    # Get device info
    device = interface.get_device_info(args.device)

    config = {
        "Device": device,
        "Output": args.output or interface.get_default_output_file(),
        "Verbose": args.verbose,
        "Test Types": "Minima Optimization",
    }
    interface.print_configuration(config)

    len(available_backends) * len(minima_optimizers)

    # Run benchmarks
    results_list = []

    for backend in available_backends:
        for optimizer in minima_optimizers:
            try:
                results = benchmark_minima_optimizer(
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
                        "test_ts": False,
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
