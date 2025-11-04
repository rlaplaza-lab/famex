#!/usr/bin/env python3
"""QME TS Optimizer Benchmark - Transition State Optimizer Comparison.

This benchmark compares the performance of different transition state optimizers
(sella, trust-krylov-ts, rfo) for transition state finding using various QME ML backends.
It focuses specifically on TS optimization to evaluate which optimizers work best
for finding transition states across different ML backends.

By default, all three optimizers (sella, trust-krylov-ts, rfo) are tested on equal footing.

Usage:
    python ts_optimizer_benchmark.py [--backends BACKEND1,BACKEND2,...]
    python ts_optimizer_benchmark.py [--optimizers OPT1,OPT2,...]
    python ts_optimizer_benchmark.py [--device DEVICE]

Features:
    - Transition state optimizer comparison (sella, trust-krylov-ts, rfo)
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
    """Create a transition state structure for TS optimization using example files.

    Note: The structure should be reasonably close to the actual TS. If optimizers
    consistently fail to find a TS (0 or >1 imaginary frequencies), the starting
    structure may be too far from the saddle point.
    """
    from pathlib import Path

    from ase.io import read

    # Use the actual TS structure from example files
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    return read(script_dir / "example_files" / "reaction_001_ts.xyz")


def benchmark_ts_optimizer(
    backend: str,
    optimizer: str,
    device: str | None = None,
    model_name: str | None = None,
    verbose: bool = True,
    calculate_frequencies: bool = True,
    hessian_update_freq: int | None = None,
    force_finite_diff_hessian: bool = False,
) -> dict[str, Any]:
    """Benchmark a single backend with a specific optimizer for transition state optimization.

    Suitable optimizers: Sella, Trust-Krylov-TS, RFO

    Parameters
    ----------
    calculate_frequencies : bool
        Whether to perform frequency analysis to validate TS (default: True)
    hessian_update_freq : int | None
        Hessian update frequency for Hessian-based optimizers (None = use default)
    force_finite_diff_hessian : bool
        Force use of finite difference Hessians instead of analytical
    """
    # Prepare ts_kwargs if hessian_update_freq is specified for Hessian-based optimizers
    ts_kwargs = None
    if hessian_update_freq is not None and optimizer in ["trust-krylov-ts", "rfo"]:
        ts_kwargs = {"hessian_update_freq": hessian_update_freq}
    if force_finite_diff_hessian:
        if ts_kwargs is None:
            ts_kwargs = {}
        ts_kwargs["hessian_method"] = "finite_differences"

    return benchmark_optimization(
        backend=backend,
        optimizer=optimizer,
        device=device,
        model_name=model_name,
        verbose=verbose,
        test_ts=True,
        create_structure_func=create_ts_structure,
        suitable_optimizers=["sella", "trust-krylov-ts", "rfo"],
        calculate_frequencies=calculate_frequencies,
        ts_kwargs=ts_kwargs,
        force_finite_diff_hessian=force_finite_diff_hessian,
    )


def print_frequency_analysis_summary(results_list: list[dict[str, Any]]) -> None:
    """Print a detailed frequency analysis summary for TS optimization."""
    print(f"\n{'=' * 120}")
    print("TRANSITION STATE VALIDATION SUMMARY")
    print(f"{'=' * 120}")
    print(
        "A valid TS must have exactly 1 imaginary frequency (saddle point). "
        "Optimizers that fail to find a TS are marked as failed."
    )

    # Results
    print(
        f"\n{'Backend':<12} {'Optimizer':<15} {'Imag. Freq':<12} {'Status':<15} {'Lowest 3 Freq (cm⁻¹)':<25}"
    )
    print("=" * 120)

    failed_optimizers = []
    for results in results_list:
        if results["available"] and "frequency_results" in results:
            freq_results = results["frequency_results"]
            n_imag = freq_results.get("n_imaginary_frequencies", 0)
            is_valid = freq_results.get("is_valid_result", False)
            frequencies = freq_results.get("frequencies", [])
            method_used = freq_results.get("method_used", "unknown")

            backend = results.get("backend", "unknown")
            optimizer = results.get("optimizer", "unknown")

            # Format first 3 frequencies
            # Try to get frequencies from ts_analysis if main frequencies list is empty
            if not frequencies and freq_results.get("ts_analysis", {}):
                ts_analysis = freq_results.get("ts_analysis", {})
                all_freqs = ts_analysis.get("all_frequencies", [])
                if all_freqs and len(all_freqs) >= 3:
                    frequencies = all_freqs[:3]
                elif all_freqs:
                    frequencies = all_freqs

            if len(frequencies) >= 3:
                freq_str = f"[{frequencies[0]:.1f}, {frequencies[1]:.1f}, {frequencies[2]:.1f}]"
            elif len(frequencies) > 0:
                freq_str = f"[{', '.join(f'{f:.1f}' for f in frequencies)}]"
            elif method_used == "not_calculated":
                freq_str = "Failed"
            else:
                freq_str = "N/A"

            # Status indicator
            if is_valid:
                status = "✅ Valid TS"
            elif n_imag == 0:
                status = "❌ Minimum (0 imag)"
                failed_optimizers.append((backend, optimizer, "found minimum, not TS"))
            elif n_imag > 1:
                status = f"❌ Not TS ({n_imag} imag)"
                failed_optimizers.append(
                    (backend, optimizer, f"found {n_imag} imaginary frequencies, not TS")
                )
            else:
                status = "❌ Invalid"

            print(f"{backend:<12} {optimizer:<15} {n_imag:<12} {status:<15} {freq_str:<25}")

    # Summary statistics
    available_results = [r for r in results_list if r["available"] and "frequency_results" in r]
    if available_results:
        # Overall validation rate
        valid_count = sum(
            1 for r in available_results if r["frequency_results"].get("is_valid_result", False)
        )
        total_count = len(available_results)
        success_rate = (valid_count / total_count * 100) if total_count > 0 else 0

        print(f"\n{'=' * 120}")
        print(
            f"SUMMARY: {valid_count}/{total_count} optimizations found valid TS ({success_rate:.1f}% success rate)"
        )

        # TS-specific issues
        ts_results = available_results  # All results are TS results in this benchmark
        if ts_results:
            ts_with_wrong_freq = sum(
                1
                for r in ts_results
                if r["frequency_results"].get("n_imaginary_frequencies", 0) != 1
            )
            if ts_with_wrong_freq > 0:
                print(
                    f"\n⚠️  WARNING: {ts_with_wrong_freq} optimizer(s) failed to find transition states:"
                )
                for backend, optimizer, reason in failed_optimizers:
                    print(f"   - {backend}/{optimizer}: {reason}")


def print_optimizer_summary(results_list: list[dict[str, Any]]) -> None:
    """Print a summary table focused on TS optimizer comparison."""
    # Print legend first, before the table
    print(f"\n{'=' * 120}")
    print("TS OPTIMIZER COMPARISON")
    print(f"{'=' * 120}")
    print("Legend: ✅ = Success, ❌ = Failed, ⚠️ = Warning, ⏱️ = Time")

    # Header
    print(
        f"\n{'Backend':<12} {'Optimizer':<15} {'Status':<10} {'Total Time':<12} {'Opt Steps':<10} "
        f"{'Time/Step':<12} {'Final Energy':<15} {'Max Force':<12} {'TS Valid':<10}"
    )
    print("=" * 120)
    print(
        "Note: Status shows convergence + TS validation. '✅ TS' = converged & valid TS, "
        "'⚠️ No TS' = converged but not a TS (failed), '❌' = didn't converge"
    )

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

            # For TS optimization, status should reflect both convergence AND valid TS
            # A TS optimizer fails if it doesn't find a saddle point (1 imaginary frequency)
            n_imag = freq_results.get("n_imaginary_frequencies", 0)
            if converged and is_valid:
                status = "✅ TS"  # Converged and valid TS
            elif converged and not is_valid:
                status = "⚠️ No TS"  # Converged but not a TS (optimizer failed)
            else:
                status = "❌"  # Didn't converge

            valid_str = "✅" if is_valid else "❌"
            if not is_valid and n_imag != 1:
                # Add note about imaginary frequency count
                valid_str = f"❌ ({n_imag} imag)"

            print(
                f"{backend:<12} {optimizer:<15} {status:<10} {timings.get('total', 0):<12.3f} {steps_str:<10} "
                f"{time_per_step:<12.6f} {energy_str:<15} {force_str:<12} {valid_str:<10}"
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
        help="Comma-separated list of optimizers to benchmark (default: sella,trust-krylov-ts,rfo - all tested on equal footing)",
    )
    parser.add_argument(
        "--freq",
        action="store_true",
        default=True,
        help="Perform frequency analysis to validate TS (default: True). Use --no-freq to disable.",
    )
    parser.add_argument(
        "--no-freq",
        dest="freq",
        action="store_false",
        help="Skip frequency analysis (faster but no TS validation)",
    )
    parser.add_argument(
        "--hessian-update-freq",
        type=int,
        default=5,
        help="Hessian update frequency for Hessian-based optimizers (default: 5)",
    )
    parser.add_argument(
        "--force-finite-diff-hessian",
        action="store_true",
        help="Force use of finite difference Hessians instead of analytical",
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
    valid_optimizers = ["sella", "trust-krylov-ts", "rfo"]
    if args.optimizers:
        requested_optimizers = [o.strip().lower() for o in args.optimizers.split(",")]
        # Filter to only TS optimizers
        ts_optimizers = [opt for opt in requested_optimizers if opt in valid_optimizers]
        if len(ts_optimizers) != len(requested_optimizers):
            invalid_opts = [opt for opt in requested_optimizers if opt not in valid_optimizers]
            if invalid_opts:
                interface.print_warning(
                    f"Invalid optimizers ignored: {', '.join(invalid_opts)}. "
                    f"Valid options: {', '.join(valid_optimizers)}"
                )
    else:
        # Default: test all TS optimizers on equal footing
        ts_optimizers = ["sella", "trust-krylov-ts", "rfo"]

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
                    calculate_frequencies=args.freq,
                    hessian_update_freq=args.hessian_update_freq,
                    force_finite_diff_hessian=args.force_finite_diff_hessian,
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
