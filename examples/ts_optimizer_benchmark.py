#!/usr/bin/env python3
"""
QME TS Optimizer Benchmark - Transition State Optimizer Comparison

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

import numpy as np
from ase import Atoms

# Import QME components
try:
    pass  # QME components imported via benchmark_optimization function
except ImportError as e:
    print(f"❌ Error importing QME: {e}")
    print("   Please ensure QME is installed and accessible")
    sys.exit(1)

# Backend availability helpers

# Common interface and device utils
from qme.examples import QMEExampleInterface, benchmark_optimization, create_standard_epilog

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def create_ts_structure() -> Atoms:
    """Create a transition state structure for TS optimization using example files."""
    import os

    from ase.io import read

    # Use the actual TS structure from example files
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    structure = read(os.path.join(script_dir, "example_files", "reaction_001_ts.xyz"))
    return structure


def benchmark_ts_optimizer(
    backend: str,
    optimizer: str,
    device: str | None = None,
    model_name: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Benchmark a single backend with a specific optimizer for transition state optimization.

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


def print_frequency_analysis_summary(results_list: list[dict[str, Any]]):
    """Print a detailed frequency analysis summary for TS optimization."""
    print(f"\n{'=' * 120}")
    print("FREQUENCY ANALYSIS SUMMARY - TRANSITION STATE OPTIMIZATION")
    print(f"{'=' * 120}")

    # Print legend
    print("📊 FREQUENCY VALIDATION:")
    print("   • Transition States should have exactly 1 imaginary frequency")
    print("   • Invalid results suggest optimization found a minimum or saddle point")
    print(f"{'-' * 120}")

    # Header
    print(
        f"{'Backend':<12} {'Optimizer':<12} {'Type':<4} {'Imag Freq':<10} "
        f"{'Valid':<8} {'ZPE (eV)':<12} {'First 3 Freq (cm⁻¹)':<25} {'Status':<15}"
    )
    print("=" * 120)

    # Results
    for results in results_list:
        if results["available"] and "frequency_results" in results:
            freq_results = results["frequency_results"]
            n_imag = freq_results.get("n_imaginary_frequencies", 0)
            is_valid = freq_results.get("is_valid_result", False)
            zpe = freq_results.get("zero_point_energy", 0)
            frequencies = freq_results.get("frequencies", [])

            # Format first 3 frequencies
            if len(frequencies) >= 3:
                freq_str = f"[{frequencies[0]:.1f}, {frequencies[1]:.1f}, {frequencies[2]:.1f}]"
            else:
                freq_str = "N/A"

            # Status indicator
            if is_valid:
                status = "✅ Valid"
            elif n_imag != 1:
                if n_imag == 0:
                    status = "❌ Found Minima"
                else:
                    status = f"❌ {n_imag} Imag Freq"
            else:
                status = "⚠️  Unknown"

            print(
                f"{results['backend']:<12} {results.get('optimizer', 'unknown'):<12} "
                f"{'TS':<4} "
                f"{n_imag if n_imag is not None else 'N/A':<10} "
                f"{'Yes' if is_valid else 'No' if is_valid is not None else 'N/A':<8} "
                f"{zpe:<12.4f} {freq_str:<25} {status:<15}"
            )
        else:
            print(
                f"{results['backend']:<12} {results.get('optimizer', 'unknown'):<12} "
                f"{'TS':<4} "
                f"{'N/A':<10} {'N/A':<8} {'N/A':<12} {'N/A':<25} {'N/A':<15}"
            )

    print("=" * 120)

    # Summary statistics
    available_results = [r for r in results_list if r["available"] and "frequency_results" in r]
    if available_results:
        print("\n🔍 FREQUENCY VALIDATION STATISTICS")
        print(f"{'=' * 60}")

        # Overall validation rate
        valid_count = sum(
            1 for r in available_results if r["frequency_results"].get("is_valid_result", False)
        )
        total_count = len(available_results)
        overall_success_rate = (valid_count / total_count * 100) if total_count > 0 else 0
        print(
            f"Overall Validation Success Rate: {overall_success_rate:.1f}% "
            f"({valid_count}/{total_count})"
        )

        # TS-specific issues
        ts_results = available_results  # All results are TS results in this benchmark
        if ts_results:
            ts_valid = sum(
                1 for r in ts_results if r["frequency_results"].get("is_valid_result", False)
            )
            ts_success_rate = (ts_valid / len(ts_results) * 100) if ts_results else 0
            print(
                f"Transition State Validation: {ts_success_rate:.1f}% "
                f"({ts_valid}/{len(ts_results)})"
            )

            # TS-specific issues
            ts_with_wrong_freq = sum(
                1
                for r in ts_results
                if r["frequency_results"].get("n_imaginary_frequencies", 0) != 1
            )
            if ts_with_wrong_freq > 0:
                print(
                    f"  ⚠️  {ts_with_wrong_freq} TS optimizations found "
                    f"incorrect number of imaginary frequencies"
                )


def print_optimizer_summary(results_list: list[dict[str, Any]]):
    """Print a summary table focused on TS optimizer comparison."""
    print(f"\n{'=' * 140}")
    print("TRANSITION STATE OPTIMIZER COMPARISON SUMMARY")
    print(f"{'=' * 140}")

    # Print legend first, before the table
    print("📊 COLUMN DEFINITIONS:")
    print("   Backend  = ML backend used")
    print("   Optimizer = Optimization algorithm")
    print("   Type     = Transition State (TS)")
    print("   Converged = Whether optimization converged")
    print("   Steps    = Number of optimization steps")
    print("   Time/Step = Average time per optimization step")
    print("   Total    = Total optimization time")
    print("   Final E  = Final energy (eV)")
    print("   Max F    = Maximum force (eV/Å)")
    print("   Valid Result = Whether result matches expected type (1 imag freq for TS)")
    print(f"{'-' * 150}")

    # Header
    print(
        f"{'Backend':<12} {'Optimizer':<12} {'Type':<4} {'Converged':<10} {'Steps':<8} "
        f"{'Time/Step (s)':<14} {'Total (s)':<10} {'Final E (eV)':<12} "
        f"{'Max F (eV/Å)':<12} {'Valid':<10}"
    )
    print("=" * 150)

    # Results
    for results in results_list:
        if results["available"]:
            timings = results["timings"]
            opt_results = results.get("optimization_results", {})
            freq_results = results.get("frequency_results", {})
            steps_taken = opt_results.get("steps_taken", 0)
            avg_time_per_step = timings.get("avg_time_per_step", 0)
            optimizer = results.get("optimizer", "unknown")
            converged = opt_results.get("converged", False)
            final_energy = opt_results.get("final_energy", None)
            max_force = opt_results.get("max_force", None)
            is_valid_result = freq_results.get("is_valid_result", False)

            # Handle None values for formatting
            steps_str = str(steps_taken) if steps_taken is not None else "N/A"
            avg_time_str = f"{avg_time_per_step:.4f}" if avg_time_per_step is not None else "N/A"
            final_energy_str = f"{final_energy:.3f}" if final_energy is not None else "N/A"
            max_force_str = f"{max_force:.6f}" if max_force is not None else "N/A"
            valid_result_str = "Yes" if is_valid_result else "No"

            print(
                f"{results['backend']:<12} {optimizer:<12} {'TS':<4} "
                f"{'Yes' if converged else 'No':<10} {steps_str:<8} "
                f"{avg_time_str:<14} "
                f"{timings.get('optimization', 0):<10.3f} "
                f"{final_energy_str:<12} "
                f"{max_force_str:<12} "
                f"{valid_result_str:<10}"
            )
        else:
            optimizer = results.get("optimizer", "unknown")
            print(
                f"{results['backend']:<12} {optimizer:<12} {'TS':<4} "
                f"{'N/A':<10} {'N/A':<8} {'N/A':<14} {'N/A':<10} "
                f"{'N/A':<12} {'N/A':<12} {'N/A':<10}"
            )

    print("=" * 150)

    # Optimizer performance analysis
    available_results = [r for r in results_list if r["available"]]
    if available_results:
        print("\n🔍 TRANSITION STATE OPTIMIZER PERFORMANCE ANALYSIS")
        print(f"{'=' * 80}")

        # Group by optimizer
        optimizer_groups = {}
        for result in available_results:
            opt_name = result.get("optimizer", "unknown")
            if opt_name not in optimizer_groups:
                optimizer_groups[opt_name] = []
            optimizer_groups[opt_name].append(result)

        for opt_name, opt_results in optimizer_groups.items():
            print(f"\n📈 {opt_name.upper()} OPTIMIZER PERFORMANCE")
            print(f"{'-' * 50}")

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
                print(
                    "  {:<30}: {:.1f} ± {:.1f}".format(
                        "Average Steps", np.mean(steps_list), np.std(steps_list)
                    )
                )
                print(f"  {'Min Steps':<30}: {min(steps_list):>8}")
                print(f"  {'Max Steps':<30}: {max(steps_list):>8}")

            if time_per_step_list:
                print(
                    "  {:<30}: {:.4f}s ± {:.4f}s".format(
                        "Avg Time/Step",
                        np.mean(time_per_step_list),
                        np.std(time_per_step_list),
                    )
                )
                print(f"  {'Min Time/Step':<30}: {min(time_per_step_list):>8.4f}s")
                print(f"  {'Max Time/Step':<30}: {max(time_per_step_list):>8.4f}s")

            if total_time_list:
                print(
                    "  {:<30}: {:.3f}s ± {:.3f}s".format(
                        "Avg Total Time",
                        np.mean(total_time_list),
                        np.std(total_time_list),
                    )
                )
                print(f"  {'Min Total Time':<30}: {min(total_time_list):>8.3f}s")
                print(f"  {'Max Total Time':<30}: {max(total_time_list):>8.3f}s")

            if converged_list:
                convergence_rate = sum(converged_list) / len(converged_list) * 100
                print(f"  {'Convergence Rate':<30}: {convergence_rate:>8.1f}%")

            # Quality analysis for TS optimization
            valid_result_list = [
                r["frequency_results"].get("is_valid_result", False)
                for r in opt_results
                if "frequency_results" in r
            ]
            if valid_result_list:
                ts_success_rate = sum(valid_result_list) / len(valid_result_list) * 100
                print(f"  {'TS Success Rate':<30}: {ts_success_rate:>8.1f}%")

            print(f"  {'Total Tests':<30}: {len(opt_results):>8}")


def save_results(results_list: list[dict[str, Any]], output_file: str):
    """Save benchmark results to JSON file."""
    # If output_file is just a filename, save it in the examples directory
    if not Path(output_file).is_absolute() and "/" not in output_file:
        output_path = Path(__file__).parent / output_file
    else:
        output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results_list, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


def main():
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

    # Determine which backends to test
    if args.backends:
        requested_backends = [b.strip() for b in args.backends.split(",")]
        available_backends = interface.filter_available_backends(requested_backends, verbose=True)

        if not available_backends:
            interface.print_error("No requested backends are available!")
            print("   Please install required dependencies.")
            return 1
    else:
        available_backends = interface.get_available_ml_backends()

        if not available_backends:
            interface.print_error("No ML backends available!")
            print("   Please install at least one ML backend:")
            print("   - UMA: pip install fairchem-core")
            print("   - MACE: pip install mace-torch")
            print("   - AIMNet2: pip install aimnet2")
            print("   - SO3LR: pip install so3lr")
            print("   - TorchSim: pip install torch-sim-atomistic")
            return 1

    # Determine which optimizers to test
    if args.optimizers:
        requested_optimizers = [o.strip().lower() for o in args.optimizers.split(",")]
        # Filter to only TS optimizers
        valid_optimizers = ["sella", "trust-krylov-ts"]
        ts_optimizers = [opt for opt in requested_optimizers if opt in valid_optimizers]
        if len(ts_optimizers) != len(requested_optimizers):
            invalid_opts = [opt for opt in requested_optimizers if opt not in valid_optimizers]
            print(f"Warning: Invalid optimizers ignored: {', '.join(invalid_opts)}")
            print(f"Valid TS optimizers: {', '.join(valid_optimizers)}")
    else:
        ts_optimizers = ["sella"]

    if not ts_optimizers:
        interface.print_error("No valid TS optimizers specified!")
        print("Valid options: sella, trust-krylov-ts")
        return 1

    interface.print_backend_summary(available_backends, "Benchmarking Backends")
    print(f"\nTS Optimizers: {', '.join(ts_optimizers)}")
    print("Test Type: Transition State Optimization")

    # Get device info
    device = interface.get_device_info(args.device)

    config = {
        "Device": device,
        "Output": args.output or interface.get_default_output_file(),
        "Verbose": args.verbose,
        "Test Types": "Transition State Optimization",
    }
    interface.print_configuration(config)

    total_tests = len(available_backends) * len(ts_optimizers)
    print(
        f"\nRunning benchmarks for {len(available_backends)} backend(s) × "
        f"{len(ts_optimizers)} TS optimizer(s) = {total_tests} tests..."
    )

    # Run benchmarks
    results_list = []

    print(f"\n{'=' * 80}")
    print("TRANSITION STATE OPTIMIZATION BENCHMARKS")
    print(f"{'=' * 80}")

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
                print("\nBenchmark interrupted by user.")
                break
            except Exception as e:
                print(f"\nUnexpected error with {backend}+{optimizer}: {e}")
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
                    }
                )

    # Print summaries
    print_frequency_analysis_summary(results_list)
    print_optimizer_summary(results_list)

    # Save results
    save_results(results_list, args.output or interface.get_default_output_file())

    interface.print_success()
    return 0


if __name__ == "__main__":
    sys.exit(main())
