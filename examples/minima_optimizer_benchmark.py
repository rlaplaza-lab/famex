#!/usr/bin/env python3
"""QME Minima Optimizer Benchmark - Minima Optimization Comparison."""

import sys
import warnings
from pathlib import Path
from typing import Any

from ase import Atoms

# Import QME components (via benchmark_optimization function)
# Backend availability helpers
# Common interface and device utils
from qme.example_utils import QMEExampleInterface, benchmark_optimization, create_standard_epilog

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def create_minima_structure() -> Atoms:
    """Create a structure for minima optimization using example files."""
    from ase.io import read

    # Use the smaller A_C_A_B_A_C reactant structure as starting point for minima optimization
    script_dir = Path(__file__).parent
    return read(script_dir / "example_files" / "A_C_A_B_A_C_reactant.xyz")


def benchmark_minima_optimizer(
    backend: str,
    optimizer: str,
    device: str | None = None,
    model_name: str | None = None,
    verbose: int = 1,
    calculate_frequencies: bool = True,
    save_optimized_structure: bool = False,
    structure_label: str | None = None,
) -> dict[str, Any]:
    """Benchmark minima optimizer (LBFGS, BFGS, FIRE, Trust-Krylov)."""
    return benchmark_optimization(
        backend=backend,
        optimizer=optimizer,
        device=device,
        model_name=model_name,
        verbose=verbose,
        test_ts=False,
        create_structure_func=create_minima_structure,
        suitable_optimizers=["lbfgs", "bfgs", "fire", "trust-krylov"],
        calculate_frequencies=calculate_frequencies,
        save_optimized_structure=save_optimized_structure,
        structure_label=structure_label,
    )


def print_frequency_analysis_summary(results_list: list[dict[str, Any]]) -> None:
    """Print a detailed frequency analysis summary for minima optimization."""
    print(f"\n{'=' * 120}")
    print("MINIMA VALIDATION SUMMARY")
    print(f"{'=' * 120}")
    print(
        "A valid minimum should have 0 imaginary frequencies. Optimizers that produce imaginary modes "
        "are marked as failed."
    )

    print(
        f"\n{'Backend':<12} {'Optimizer':<15} {'Imag. Freq':<12} {'Status':<15} {'Lowest 3 Freq (cm⁻¹)':<25}"
    )
    print("=" * 120)

    available_results = [r for r in results_list if r.get("available") and "frequency_results" in r]
    failed_optimizers = []

    for results in available_results:
        freq_results = results["frequency_results"]
        n_imag = freq_results.get("n_imaginary_frequencies", 0)
        is_valid = freq_results.get("is_valid_result", False)
        method_used = freq_results.get("method_used", "unknown")

        backend = results.get("backend", "unknown")
        optimizer = results.get("optimizer", "unknown")

        all_freqs = freq_results.get("all_frequencies", [])
        if not all_freqs:
            all_freqs = freq_results.get("frequencies", [])
        if not all_freqs and freq_results.get("ts_analysis", {}):
            ts_analysis = freq_results.get("ts_analysis", {})
            all_freqs = ts_analysis.get("all_frequencies", [])

        if all_freqs:
            # Convert complex frequencies to real (take absolute value for sorting)
            filtered_freqs = [f for f in all_freqs if abs(f) > 10.0]
            if not filtered_freqs and all_freqs:
                filtered_freqs = all_freqs
            # Handle complex frequencies by converting to real for sorting
            frequencies = sorted(
                filtered_freqs, key=lambda x: abs(x) if isinstance(x, complex) else x
            )
        else:
            frequencies = []

        if len(frequencies) >= 3:
            freq_str = f"[{frequencies[0]:.1f}, {frequencies[1]:.1f}, {frequencies[2]:.1f}]"
        elif len(frequencies) > 0:
            freq_str = f"[{', '.join(f'{f:.1f}' for f in frequencies)}]"
        elif method_used == "not_calculated":
            freq_str = "Skipped"
        else:
            freq_str = "N/A"

        if is_valid and n_imag == 0:
            status = "✅ Valid Min"
        elif n_imag > 0:
            status = f"❌ {n_imag} imag"
            failed_optimizers.append((backend, optimizer, f"{n_imag} imaginary frequencies"))
        else:
            status = "❌ Invalid"

        print(f"{backend:<12} {optimizer:<15} {n_imag:<12} {status:<15} {freq_str:<25}")

    if available_results:
        valid_count = sum(
            1 for r in available_results if r["frequency_results"].get("is_valid_result", False)
        )
        total_count = len(available_results)
        success_rate = (valid_count / total_count * 100) if total_count > 0 else 0

        print(f"\n{'=' * 120}")
        print(
            f"SUMMARY: {valid_count}/{total_count} optimizations found valid minima ({success_rate:.1f}% success rate)"
        )

        if failed_optimizers:
            print(f"\n⚠️  WARNING: {len(failed_optimizers)} optimizer(s) showed imaginary modes:")
            for backend, optimizer, reason in failed_optimizers:
                print(f"   - {backend}/{optimizer}: {reason}")


def print_optimizer_summary(results_list: list[dict[str, Any]]) -> None:
    """Print a summary table focused on minima optimizer comparison."""
    # Print legend first, before the table
    print(f"\n{'=' * 120}")
    print("MINIMA OPTIMIZER COMPARISON")
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

        print(f"\n{'=' * 120}")
        print("OPTIMIZER PERFORMANCE ANALYSIS")
        print(f"{'=' * 120}")

        for opt_name, opt_results in optimizer_groups.items():
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

            print(f"\n{opt_name.upper()} OPTIMIZER:")
            print(f"  Total Tests: {len(opt_results)}")
            print(
                f"  Success Rate: {sum(converged_list)}/{len(converged_list)} ({sum(converged_list) / len(converged_list) * 100:.1f}%)"
            )

            if steps_list:
                avg_steps = sum(steps_list) / len(steps_list)
                min_steps = min(steps_list)
                max_steps = max(steps_list)
                print(f"  Average Steps: {avg_steps:.1f} (min: {min_steps}, max: {max_steps})")

            if time_per_step_list:
                avg_time_per_step = sum(time_per_step_list) / len(time_per_step_list)
                min_time_per_step = min(time_per_step_list)
                max_time_per_step = max(time_per_step_list)
                print(
                    f"  Average Time/Step: {avg_time_per_step:.6f}s (min: {min_time_per_step:.6f}s, max: {max_time_per_step:.6f}s)"
                )

            if total_time_list:
                avg_total_time = sum(total_time_list) / len(total_time_list)
                min_total_time = min(total_time_list)
                max_total_time = max(total_time_list)
                print(
                    f"  Average Total Time: {avg_total_time:.3f}s (min: {min_total_time:.3f}s, max: {max_total_time:.3f}s)"
                )

            # Quality analysis for minima optimization (computed but not displayed)


def print_performance_summary(results_list: list[dict[str, Any]]) -> None:
    """Print comprehensive performance profiler data."""
    # Check for performance data
    has_performance_data = any(
        results.get("available") and "performance" in results for results in results_list
    )

    if not has_performance_data:
        return

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

    # Show timing statistics for each section
    for section in sorted(all_sections):
        section_stats = []
        for results in results_list:
            if results.get("available") and "performance" in results:
                perf = results["performance"]
                timings = perf.get("timings", {})
                if section in timings:
                    section_stats.append(timings[section])


def main() -> int:
    """Run the minima optimizer comparison benchmark."""
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
    parser.add_argument(
        "--skip-slow-optimizers",
        action="store_true",
        help="Skip known slow optimizers (e.g., bfgs) to speed up testing",
    )
    parser.add_argument(
        "--freq",
        action="store_true",
        default=True,
        help="Perform frequency analysis to validate minima (default: True). Use --no-freq to disable.",
    )
    parser.add_argument(
        "--no-freq",
        dest="freq",
        action="store_false",
        help="Skip frequency analysis (faster but no minima validation)",
    )
    parser.add_argument(
        "--save-xyz",
        action="store_true",
        help="Save optimized structures as XYZ files in the current directory",
    )

    args = parser.parse_args()

    interface.print_header()

    # Set up logging based on verbosity level
    interface.setup_logging(args.verbose)

    # Backend handling
    requested = [b.strip() for b in args.backends.split(",")] if args.backends else None
    _, available_backends = interface.select_backend(
        requested_backends=requested,
        verbose=args.verbose,
    )
    if not available_backends:
        interface.print_error("No available backends found")
        return 1

    # Determine which optimizers to test
    if args.optimizers:
        requested_optimizers = [o.strip().lower() for o in args.optimizers.split(",")]
        # Filter to only minima optimizers
        valid_optimizers = ["lbfgs", "bfgs", "fire", "trust-krylov"]
        minima_optimizers = [opt for opt in requested_optimizers if opt in valid_optimizers]
    else:
        minima_optimizers = ["lbfgs", "bfgs", "fire", "trust-krylov"]

    # Skip slow optimizers if requested
    slow_optimizers = {"bfgs"}  # Known slow optimizers
    if args.skip_slow_optimizers:
        minima_optimizers = [opt for opt in minima_optimizers if opt not in slow_optimizers]
        if minima_optimizers:
            interface.print_warning(f"Skipping slow optimizers: {', '.join(slow_optimizers)}")

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

    # Run benchmarks
    results_list = []
    total_tests = len(available_backends) * len(minima_optimizers)
    current_test = 0

    for backend in available_backends:
        for optimizer in minima_optimizers:
            current_test += 1
            print(
                f"\n[{current_test}/{total_tests}] Testing {backend}/{optimizer}...",
                flush=True,
            )
            try:
                results = benchmark_minima_optimizer(
                    backend=backend,
                    optimizer=optimizer,
                    device=device,
                    verbose=args.verbose,
                    calculate_frequencies=args.freq,
                    save_optimized_structure=args.save_xyz,
                    structure_label=optimizer,
                )
                results_list.append(results)
                if results.get("available", False):
                    print(f"  ✓ Completed {backend}/{optimizer}", flush=True)
                else:
                    print(f"  ✗ Failed {backend}/{optimizer}", flush=True)
            except KeyboardInterrupt:
                print("\nInterrupted by user", flush=True)
                break
            except Exception as e:
                print(f"  ✗ Error {backend}/{optimizer}: {e}", flush=True)
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
    interface.save_results(results_list, args.output or interface.get_default_output_file())

    interface.print_success()
    return 0


if __name__ == "__main__":
    sys.exit(main())
