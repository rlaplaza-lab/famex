#!/usr/bin/env python3
"""QME Timing Benchmark - ML Backend Performance Analysis."""

import sys
import time
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from ase import Atoms

# Use consolidated backend availability from qme.backend_availability
# Import QME components
from qme.backends.registry import calculator_registry
from qme.core.explorer import Explorer

# Import common interface
from qme.example_utils import QMEExampleInterface, create_standard_epilog
from qme.utils.device import get_optimal_device, print_device_info

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# QMEExampleInterface already imported above


def create_benchmark_molecule() -> Atoms:
    """Create a molecule for benchmarking using example files."""
    import os

    from ase.io import read

    # Use the ACABAC reactant structure from example files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return read(os.path.join(script_dir, "example_files", "A_C_A_B_A_C_reactant.xyz"))


def time_function(func: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Time a function call and return result and timing."""
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    return result, end_time - start_time


def benchmark_backend(
    backend: str,
    device: str | None = None,
    model_name: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Benchmark a single backend for optimization and frequency analysis."""
    # Auto-detect optimal device
    device = get_optimal_device(device)

    if verbose:
        print_device_info(device)

    results = {
        "backend": backend,
        "device": device,
        "model_name": model_name,
        "available": False,
        "error": None,
        "timings": {},
        "optimization_results": {},
        "frequency_results": {},
    }

    try:
        # Check if backend is available
        if not calculator_registry.is_backend_available(backend):
            results["error"] = f"Backend {backend} not available (dependencies missing)"
            return results

        results["available"] = True

        # Create benchmark molecule
        molecule = create_benchmark_molecule()

        # Initialize QME optimizer
        init_start = time.perf_counter()

        explorer = Explorer(
            atoms=molecule,
            backend=backend,
            model_name=model_name,
            device=device,
            default_charge=0,
            default_spin=1,
            local_optimizer="lbfgs",
            target="minima",
            strategy="local",
            profile=True,  # Enable profiling for timing benchmark
        )

        init_time = time.perf_counter() - init_start
        results["timings"]["initialization"] = init_time

        # Attach calculator to atoms object
        load_start = time.perf_counter()

        # Attach calculator using Explorer's method
        explorer._create_and_attach_calculator(explorer.atoms_list[0])

        load_time = time.perf_counter() - load_start
        results["timings"]["structure_loading"] = load_time

        # Test single energy calculation (first call - includes calculator initialization)
        energy_first_start = time.perf_counter()

        explorer.atoms_list[0].get_potential_energy()

        energy_first_time = time.perf_counter() - energy_first_start
        results["timings"]["single_energy_first"] = energy_first_time

        # Test single energy calculation (second call - pure evaluation)
        energy_second_start = time.perf_counter()

        explorer.atoms_list[0].get_potential_energy()

        energy_second_time = time.perf_counter() - energy_second_start
        results["timings"]["single_energy_second"] = energy_second_time

        # Store the pure evaluation time as the main single_energy metric
        results["timings"]["single_energy"] = energy_second_time

        # Test single force calculation
        force_start = time.perf_counter()

        forces = explorer.atoms_list[0].get_forces()
        max_force = float(np.max(np.abs(forces)))

        force_time = time.perf_counter() - force_start
        results["timings"]["single_forces"] = force_time

        # Geometry optimization using Explorer strategies
        opt_start = time.perf_counter()

        # Use Explorer's run method with proper strategy
        run_results = explorer.run(
            fmax=0.01,
            steps=1000,
        )

        opt_time = time.perf_counter() - opt_start

        # Handle results from Explorer's run method
        # The run() method returns a dictionary with standardized results
        if isinstance(run_results, dict):
            steps_taken = run_results.get("steps_taken", 0)
            converged = run_results.get("converged", False)
            optimized_atoms = run_results.get("optimized_atoms", explorer.atoms_list[0])
        else:
            # Fallback for unexpected format
            steps_taken = 0
            converged = True
            optimized_atoms = run_results

        # Calculate average time per optimization step
        if isinstance(steps_taken, (int, float)) and steps_taken > 0:
            avg_time_per_step = opt_time / steps_taken
        else:
            avg_time_per_step = None

        # Get optimization results
        forces = optimized_atoms.get_forces()
        max_force = float(np.max(np.abs(forces)))

        opt_results = {
            "converged": converged,
            "final_energy": float(optimized_atoms.get_potential_energy()),
            "steps_taken": steps_taken,
            "optimized_atoms": optimized_atoms,
            "max_force": max_force,
        }

        results["timings"]["optimization"] = opt_time
        results["timings"]["avg_time_per_step"] = avg_time_per_step
        results["optimization_results"] = {
            "converged": opt_results["converged"],
            "final_energy": opt_results["final_energy"],
            "max_force": opt_results["max_force"],
            "steps_taken": steps_taken,
        }

        # Extract performance data from profiler if available
        if isinstance(run_results, dict) and "performance" in run_results:
            results["performance"] = run_results["performance"]

        # Frequency analysis
        freq_start = time.perf_counter()

        freq_results = explorer.calculate_frequencies(
            atoms=optimized_atoms,  # Use the optimized atoms with the correct calculator
            delta=0.01,
            method="auto",
            temperature=298.15,
            save_hessian=False,  # Don't save large Hessian matrix
        )

        freq_time = time.perf_counter() - freq_start
        results["timings"]["frequency_analysis"] = freq_time
        results["frequency_results"] = {
            "n_frequencies": len(freq_results["frequencies"]),
            "frequencies": freq_results["frequencies"][:10],  # First 10 frequencies
            "zero_point_energy": freq_results["zero_point_energy"],
            "is_transition_state": freq_results["is_ts"],
            "is_minimum": freq_results.get("is_minimum", None),
            "method_used": freq_results["method_used"],
        }

        # Calculate total time (excluding None values)
        total_time = sum(v for v in results["timings"].values() if v is not None)
        results["timings"]["total"] = total_time

    except Exception as e:
        results["error"] = str(e)

    return results


def print_summary(results_list: list[dict[str, Any]]) -> None:
    """Print a summary table of all benchmark results."""
    # Print legend first, before the table
    print(f"\n{'=' * 120}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 120}")
    print("Legend: ✅ = Success, ❌ = Failed, ⚠️ = Warning, ⏱️ = Time")

    # Header
    print(
        f"\n{'Backend':<12} {'Status':<8} {'Total Time':<12} {'Opt Steps':<10} "
        f"{'Time/Step':<12} {'Final Energy':<15} {'Max Force':<12}"
    )
    print("=" * 120)

    # Results
    for results in results_list:
        if results["available"]:
            timings = results["timings"]
            opt_results = results.get("optimization_results", {})
            steps_taken = opt_results.get("steps_taken", 0)
            time_per_step = timings.get("avg_time_per_step", 0)

            backend = results.get("backend", "unknown")
            total_time = timings.get("total", 0.0)
            final_energy = opt_results.get("final_energy", 0.0)
            max_force = opt_results.get("max_force", 0.0)

            status = "✅"
            print(
                f"{backend:<12} {status:<8} {total_time:<12.3f} {steps_taken:<10} "
                f"{time_per_step:<12.6f} {final_energy:<15.6f} {max_force:<12.6f}"
            )

        else:
            backend = results.get("backend", "unknown")
            status = "❌"
            print(
                f"{backend:<12} {status:<8} {'N/A':<12} {'N/A':<10} "
                f"{'N/A':<12} {'N/A':<15} {'N/A':<12}"
            )

    # Note about detailed breakdown
    available_results = [r for r in results_list if r["available"]]
    if available_results:
        print(
            "\nNote: Detailed timing breakdown is now available in the Performance Profiler section below."
        )


def print_performance_summary(results_list: list[dict[str, Any]]) -> None:
    """Print comprehensive performance profiler data."""
    # Check for performance data
    has_performance_data = any(
        results.get("available") and "performance" in results for results in results_list
    )

    if not has_performance_data:
        print(f"\n{'=' * 120}")
        print("PERFORMANCE PROFILER SUMMARY")
        print(f"{'=' * 120}")
        print("No performance profiler data available.")
        return

    # Section 1: Calculator Calls & Memory
    print(f"\n{'=' * 120}")
    print("CALCULATOR CALLS & MEMORY USAGE")
    print(f"{'=' * 120}")

    # Header (no optimizer column for timing benchmark)
    print(
        f"{'Backend':<12} {'Energy Calls':<12} {'Force Calls':<12} "
        f"{'Hessian Calls':<13} {'Peak Mem (MB)':<14} {'GPU Peak (MB)':<14}"
    )
    print("=" * 120)

    # Results
    for results in results_list:
        if results.get("available") and "performance" in results:
            perf = results["performance"]
            calls = perf.get("calculator_calls", {})
            memory = perf.get("memory", {})

            backend = results.get("backend", "unknown")
            energy_calls = calls.get("energy", 0)
            force_calls = calls.get("forces", 0)
            hessian_calls = calls.get("hessian", 0)
            peak_mem = memory.get("peak_memory_mb", 0.0)
            gpu_peak = memory.get("gpu_peak_memory_mb", 0.0)

            print(
                f"{backend:<12} {energy_calls:<12} {force_calls:<12} "
                f"{hessian_calls:<13} {peak_mem:<14.1f} {gpu_peak:<14.1f}"
            )

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
    print(f"\n{'=' * 120}")
    print("DETAILED TIMING BREAKDOWN")
    print(f"{'=' * 120}")

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
            total_time = sum(stat.get("total_time", 0.0) for stat in section_stats)
            total_count = sum(stat.get("count", 0) for stat in section_stats)
            avg_time = (
                sum(stat.get("avg_time", 0.0) * stat.get("count", 0) for stat in section_stats)
                / total_count
                if total_count > 0
                else 0.0
            )
            min_time = min(stat.get("min_time", 0.0) for stat in section_stats)
            max_time = max(stat.get("max_time", 0.0) for stat in section_stats)

            print(f"\n{section.upper().replace('_', ' ')}:")
            print(f"  Total Time: {total_time:.3f}s")
            print(f"  Average Time: {avg_time:.3f}s")
            print(f"  Min Time: {min_time:.3f}s")
            print(f"  Max Time: {max_time:.3f}s")
            print(f"  Total Calls: {total_count}")


def main() -> int:
    """Main function to run the timing benchmark."""
    # Create standardized interface
    interface = QMEExampleInterface(
        name="Timing Benchmark",
        description="Performance Analysis",
        epilog=create_standard_epilog("timing"),
    )

    parser = interface.create_parser()
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

    interface.print_backend_summary(available_backends, "Benchmarking Backends")

    # Get device info
    device = interface.get_device_info(args.device)

    config = {
        "Device": device,
        "Output": args.output or interface.get_default_output_file(),
        "Verbose": args.verbose,
    }
    interface.print_configuration(config)

    # Run benchmarks
    results_list = []
    for backend in available_backends:
        try:
            results = benchmark_backend(backend=backend, device=device, verbose=args.verbose)
            results_list.append(results)
        except KeyboardInterrupt:
            break
        except Exception as e:
            interface.print_error(f"Unexpected error with {backend}: {e}")
            results_list.append(
                {
                    "backend": backend,
                    "available": False,
                    "error": str(e),
                    "timings": {},
                    "optimization_results": {},
                    "frequency_results": {},
                },
            )

    # Print summary
    print_summary(results_list)
    print_performance_summary(results_list)

    # Save results
    interface.save_results(results_list, args.output or interface.get_default_output_file())

    interface.print_success()
    return 0


if __name__ == "__main__":
    sys.exit(main())
