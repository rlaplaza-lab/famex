#!/usr/bin/env python3
"""QME TS Optimizer Benchmark - Transition State Optimizer Comparison."""

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


def create_ts_structure() -> Atoms:
    """Create a transition state structure for TS optimization."""
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
    calculate_frequencies: bool = True,
    hessian_update_freq: int | None = None,
    force_finite_diff_hessian: bool = False,
) -> dict[str, Any]:
    """Benchmark TS optimizer (Sella, Trust-Krylov-TS, RFO)."""
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


def _should_show_backend(results: dict[str, Any]) -> bool:
    """Determine if a backend result should be displayed.

    Filters out:
    - Backends that are not available (no useful data)
    - Backends not suitable for TS optimization (mock, torchsim_*)
    """
    backend = results.get("backend", "").lower()

    # Filter out backends not suitable for TS optimization
    if backend == "mock":
        return False
    if backend.startswith("torchsim_"):
        return False

    # Only show if available and has results
    return results.get("available", False)


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
        if _should_show_backend(results) and "frequency_results" in results:
            freq_results = results["frequency_results"]
            n_imag = freq_results.get("n_imaginary_frequencies", 0)
            is_valid = freq_results.get("is_valid_result", False)
            method_used = freq_results.get("method_used", "unknown")

            backend = results.get("backend", "unknown")
            optimizer = results.get("optimizer", "unknown")

            # Get frequencies for display - use all_frequencies if available (includes all frequencies)
            # Otherwise use vibrational frequencies. This ensures we see imaginary frequencies even if
            # they were filtered out of vibrational frequencies for some reason
            all_freqs = freq_results.get("all_frequencies", [])
            if not all_freqs:
                # Fallback to vibrational frequencies
                all_freqs = freq_results.get("frequencies", [])
            if not all_freqs and freq_results.get("ts_analysis", {}):
                # Final fallback: try to get from ts_analysis (vibrational frequencies)
                ts_analysis = freq_results.get("ts_analysis", {})
                all_freqs = ts_analysis.get("all_frequencies", [])

            # Filter out near-zero frequencies (trans/rot modes with numerical noise)
            # Keep only frequencies with |freq| > 10 cm^-1 to avoid showing trans/rot noise
            if all_freqs:
                filtered_freqs = [f for f in all_freqs if abs(f) > 10.0]
                # If filtering removed all frequencies, use original (edge case)
                if not filtered_freqs and all_freqs:
                    filtered_freqs = all_freqs
                # Sort frequencies so negative (imaginary) ones appear first
                frequencies = sorted(
                    filtered_freqs, key=lambda x: (x >= 0, x)
                )  # Negatives first, then positives
            else:
                frequencies = []

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

    # Summary statistics (filter to only showable backends)
    available_results = [
        r for r in results_list if _should_show_backend(r) and "frequency_results" in r
    ]
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
        if _should_show_backend(results):
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
            # Skip unavailable/unsuitable backends - don't show them at all
            continue

    # Optimizer performance analysis (filter to only showable backends)
    available_results = [r for r in results_list if _should_show_backend(r)]
    if available_results:
        # Group by optimizer
        optimizer_groups = {}
        for result in available_results:
            opt_name = result.get("optimizer", "unknown")
            if opt_name not in optimizer_groups:
                optimizer_groups[opt_name] = []
            optimizer_groups[opt_name].append(result)

        # Statistics computed but not displayed in current implementation


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

    # Run benchmarks
    results_list = []

    for backend in available_backends:
        for optimizer in ts_optimizers:
            # For Hessian-based optimizers, test multiple update frequencies
            if optimizer in ["trust-krylov-ts", "rfo"]:
                # Test with: single Hessian (None), every 5 steps, every 10 steps
                hessian_freqs = [None, 5, 10]
            else:
                # For other optimizers (e.g., sella), use default or specified value
                hessian_freqs = [args.hessian_update_freq]

            for hessian_freq in hessian_freqs:
                # Create a unique identifier for this configuration
                optimizer_name = optimizer
                if optimizer in ["trust-krylov-ts", "rfo"]:
                    if hessian_freq is None:
                        optimizer_name = f"{optimizer}_single_hessian"
                    else:
                        optimizer_name = f"{optimizer}_hessian_freq_{hessian_freq}"

                try:
                    results = benchmark_ts_optimizer(
                        backend=backend,
                        optimizer=optimizer,
                        device=device,
                        verbose=args.verbose,
                        calculate_frequencies=args.freq,
                        hessian_update_freq=hessian_freq,
                        force_finite_diff_hessian=args.force_finite_diff_hessian,
                    )
                    # Update optimizer name in results to reflect the configuration
                    results["optimizer"] = optimizer_name
                    results["hessian_update_freq"] = hessian_freq
                    results_list.append(results)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    results_list.append(
                        {
                            "backend": backend,
                            "optimizer": optimizer_name,
                            "hessian_update_freq": hessian_freq,
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
    interface.save_results(results_list, args.output or interface.get_default_output_file())

    interface.print_success()
    return 0


if __name__ == "__main__":
    sys.exit(main())
