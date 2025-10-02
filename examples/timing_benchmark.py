#!/usr/bin/env python3
"""
Timing benchmark for QME backends.

This script benchmarks the performance of different QME backends (potentials)
for geometry optimization and frequency analysis using benzene as a test case.
It measures timing for each part of the process to identify bottlenecks,
particularly the cost of integrating potentials through ASE interfaces.

Usage:
    python timing_benchmark.py [--backends BACKEND1,BACKEND2,...] [--device DEVICE]
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from ase import Atoms
from ase.build import molecule

# Import standardized backend utilities
from backend_utils import (
    AVAILABLE_BACKENDS,
    AVAILABLE_ML_BACKENDS,
    filter_available_backends,
    get_available_backends,
    print_backend_summary,
)

# Import QME components (already imported in backend_utils)
from qme import Explorer, calculator_registry, deps
from qme.analysis.frequency import FrequencyAnalysis

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def create_benzene_molecule() -> Atoms:
    """Create a benzene molecule for benchmarking."""
    # Create benzene with slightly distorted geometry to ensure optimization is needed
    benzene = molecule("C6H6")

    # Add some distortion to make optimization non-trivial
    positions = benzene.get_positions()
    # Slightly distort the ring
    for i in range(6):  # Carbon atoms
        angle = i * np.pi / 3
        positions[i, 0] += 0.1 * np.cos(angle)  # x displacement
        positions[i, 1] += 0.1 * np.sin(angle)  # y displacement

    benzene.set_positions(positions)
    return benzene


def time_function(func, *args, **kwargs):
    """Time a function call and return result and timing."""
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    return result, end_time - start_time


def benchmark_backend(
    backend: str,
    device: str = "cpu",
    model_name: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Benchmark a single backend for optimization and frequency analysis.

    Parameters:
    -----------
    backend : str
        Backend name (e.g., 'mock', 'aimnet2', 'uma', 'so3lr', 'mace')
    device : str
        Device to use ('cpu' or 'cuda')
    model_name : str, optional
        Specific model name to use
    verbose : bool
        Whether to print progress information

    Returns:
    --------
    Dict[str, Any]
        Benchmark results including timings for each step
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Benchmarking backend: {backend}")
        print(f"Device: {device}")
        print(f"Model: {model_name or 'default'}")
        print(f"{'='*60}")

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
            if verbose:
                print(f"❌ Backend {backend} not available")
            return results

        results["available"] = True
        if verbose:
            print(f"✅ Backend {backend} is available")

        # Create benzene molecule
        if verbose:
            print("Creating benzene molecule...")
        benzene = create_benzene_molecule()

        # Initialize QME optimizer
        if verbose:
            print("Initializing QME optimizer...")
        init_start = time.perf_counter()

        explorer = Explorer(
            atoms=benzene,
            backend=backend,
            model_name=model_name,
            device=device,
            default_charge=0,
            default_spin=1,
        )

        init_time = time.perf_counter() - init_start
        results["timings"]["initialization"] = init_time

        if verbose:
            print(f"  Initialization time: {init_time:.3f} seconds")

        # Load structure
        if verbose:
            print("Loading structure...")
        load_start = time.perf_counter()

        # Structure already loaded in Explorer constructor

        load_time = time.perf_counter() - load_start
        results["timings"]["structure_loading"] = load_time

        if verbose:
            print(f"  Structure loading time: {load_time:.3f} seconds")

        # Test single energy calculation (first call - includes calculator initialization)
        if verbose:
            print(
                "Testing single energy calculation (first call - includes model loading)..."
            )
        energy_first_start = time.perf_counter()

        energy = explorer.atoms_list[0].get_potential_energy()

        energy_first_time = time.perf_counter() - energy_first_start
        results["timings"]["single_energy_first"] = energy_first_time

        if verbose:
            print(f"  First energy calculation time: {energy_first_time:.3f} seconds")
            print(f"  Energy: {energy:.6f} eV")

        # Test single energy calculation (second call - pure evaluation)
        if verbose:
            print(
                "Testing single energy calculation (second call - pure evaluation)..."
            )
        energy_second_start = time.perf_counter()

        energy2 = explorer.atoms_list[0].get_potential_energy()

        energy_second_time = time.perf_counter() - energy_second_start
        results["timings"]["single_energy_second"] = energy_second_time

        if verbose:
            print(f"  Second energy calculation time: {energy_second_time:.3f} seconds")
            print(f"  Energy: {energy2:.6f} eV")

        # Store the pure evaluation time as the main single_energy metric
        results["timings"]["single_energy"] = energy_second_time

        # Test single force calculation
        if verbose:
            print("Testing single force calculation...")
        force_start = time.perf_counter()

        forces = explorer.atoms_list[0].get_forces()
        max_force = float(np.max(np.abs(forces)))

        force_time = time.perf_counter() - force_start
        results["timings"]["single_forces"] = force_time

        if verbose:
            print(f"  Single force calculation time: {force_time:.3f} seconds")
            print(f"  Max force: {max_force:.6f} eV/Å")

        # Geometry optimization
        if verbose:
            print("Running geometry optimization...")
        opt_start = time.perf_counter()

        # We need to capture the step count directly from the ASE optimizer
        # since QME's convenience layer doesn't return it
        steps_taken = 0

        # Create the optimizer manually to capture step count
        from ase.optimize import BFGS

        opt = BFGS(explorer.atoms_list[0])
        opt.run(fmax=0.01, steps=1000)
        steps_taken = opt.get_number_of_steps()

        opt_time = time.perf_counter() - opt_start

        # Calculate average time per optimization step
        if isinstance(steps_taken, (int, float)) and steps_taken > 0:
            avg_time_per_step = opt_time / steps_taken
        else:
            avg_time_per_step = None

        # Get optimization results for compatibility
        atoms = explorer.atoms_list[0]
        forces = atoms.get_forces()
        max_force = float(np.max(np.abs(forces)))

        # Flatten forces for converged() method
        forces_flat = forces.flatten()

        opt_results = {
            "converged": opt.converged(forces_flat),
            "final_energy": float(atoms.get_potential_energy()),
            "steps_taken": steps_taken,
            "optimized_atoms": atoms,
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

        if verbose:
            print(f"  Optimization time: {opt_time:.3f} seconds")
            print(f"  Steps taken: {steps_taken}")
            if avg_time_per_step is not None:
                print(f"  Average time per step: {avg_time_per_step:.4f} seconds")
            print(f"  Converged: {opt_results['converged']}")
            print(f"  Final energy: {opt_results['final_energy']:.6f} eV")
            print(f"  Max force: {opt_results['max_force']:.6f} eV/Å")

        # Frequency analysis
        if verbose:
            print("Running frequency analysis...")
        freq_start = time.perf_counter()

        freq_results = explorer.calculate_frequencies(
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
            "method_used": freq_results["method_used"],
        }

        if verbose:
            print(f"  Frequency analysis time: {freq_time:.3f} seconds")
            print(f"  Number of frequencies: {len(freq_results['frequencies'])}")
            print(f"  First 5 frequencies: {freq_results['frequencies'][:5]}")
            print(f"  Zero-point energy: {freq_results['zero_point_energy']:.6f} eV")
            print(f"  Is transition state: {freq_results['is_ts']}")

        # Calculate total time (excluding None values)
        total_time = sum(v for v in results["timings"].values() if v is not None)
        results["timings"]["total"] = total_time

        if verbose:
            print(f"\nTotal time: {total_time:.3f} seconds")
            print(f"✅ Backend {backend} completed successfully")

    except Exception as e:
        results["error"] = str(e)
        if verbose:
            print(f"❌ Error with backend {backend}: {e}")

    return results


def print_summary(results_list: List[Dict[str, Any]]):
    """Print a summary table of all benchmark results."""
    print(f"\n{'='*120}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*120}")

    # Header
    print(
        f"{'Backend':<12} {'Available':<10} {'Total (s)':<10} {'Init (s)':<10} "
        f"{'Opt (s)':<10} {'Freq (s)':<10} {'E1st (s)':<10} {'E2nd (s)':<10} "
        f"{'Forces (s)':<10} {'Steps':<8} {'Avg/Step (s)':<12}"
    )
    print("-" * 120)
    print("📊 INDIVIDUAL MEASUREMENTS:")
    print("   E1st = Energy (1st call, includes model loading)")
    print("   E2nd = Energy (2nd call, pure evaluation)")
    print("   Init = Initialization time")
    print("   Opt  = Total optimization time")
    print("   Freq = Frequency analysis time")
    print("   Forces = Single force calculation time")
    print()
    print("🧮 CALCULATED METRICS:")
    print("   Steps = Number of optimization steps taken")
    print("   Avg/Step = Average time per optimization step (total opt time / steps)")
    print("-" * 120)

    # Results
    for results in results_list:
        if results["available"]:
            timings = results["timings"]
            opt_results = results.get("optimization_results", {})
            steps_taken = opt_results.get("steps_taken", 0)
            avg_time_per_step = timings.get("avg_time_per_step", 0)

            print(
                f"{results['backend']:<12} {'Yes':<10} "
                f"{timings.get('total', 0):<10.3f} "
                f"{timings.get('initialization', 0):<10.3f} "
                f"{timings.get('optimization', 0):<10.3f} "
                f"{timings.get('frequency_analysis', 0):<10.3f} "
                f"{timings.get('single_energy_first', 0):<10.3f} "
                f"{timings.get('single_energy_second', 0):<10.3f} "
                f"{timings.get('single_forces', 0):<10.3f} "
                f"{steps_taken:<8} "
                f"{avg_time_per_step:<12.4f}"
                if avg_time_per_step is not None
                else f"{'N/A':<12}"
            )
        else:
            print(
                f"{results['backend']:<12} {'No':<10} "
                f"{'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} "
                f"{'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<8} {'N/A':<12}"
            )

    print("-" * 100)

    # Detailed breakdown for available backends
    available_results = [r for r in results_list if r["available"]]
    if available_results:
        print("\nDETAILED BREAKDOWN:")
        print(f"{'='*60}")

        for results in available_results:
            print(f"\n{results['backend'].upper()}:")
            timings = results["timings"]
            total = timings.get("total", 0)
            opt_results = results.get("optimization_results", {})

            # Group timing information by category
            print("  📊 INDIVIDUAL MEASUREMENTS:")
            individual_steps = [
                "initialization",
                "structure_loading",
                "single_energy_first",
                "single_energy_second",
                "single_forces",
                "optimization",
                "frequency_analysis",
            ]

            for step in individual_steps:
                time_val = timings.get(step, 0)
                if time_val is not None and time_val > 0:
                    percentage = (time_val / total) * 100 if total > 0 else 0
                    step_name = step.replace("_", " ").title()

                    # Add special formatting for new metrics
                    if step == "single_energy_first":
                        step_name = "Energy (1st call - includes model loading)"
                    elif step == "single_energy_second":
                        step_name = "Energy (2nd call - pure evaluation)"

                    print(f"    {step_name:<25}: {time_val:.3f}s ({percentage:.1f}%)")

            # Group calculated/averaged information
            print("  🧮 CALCULATED METRICS:")
            steps_taken = opt_results.get("steps_taken", 0)
            avg_time_per_step = timings.get("avg_time_per_step", 0)

            if steps_taken > 0:
                print(f"    {'Optimization Steps':<25}: {steps_taken}")
            else:
                print(f"    {'Optimization Steps':<25}: N/A (not available)")

            if avg_time_per_step is not None and avg_time_per_step > 0:
                print(f"    {'Avg time per opt step':<25}: {avg_time_per_step:.4f}s")
                print(f"    {'(calculated from total opt time / steps)':<25}")
            else:
                print(f"    {'Avg time per opt step':<25}: N/A (steps not available)")

            # Show total time
            print(f"  📈 TOTAL TIME: {total:.3f}s")


def save_results(results_list: List[Dict[str, Any]], output_file: str):
    """Save benchmark results to JSON file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results_list, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


def main():
    """Main function to run the timing benchmark."""
    parser = argparse.ArgumentParser(description="QME Backend Timing Benchmark")
    parser.add_argument(
        "--backends",
        type=str,
        default=None,
        help="Comma-separated list of backends to benchmark (default: all available)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for calculations",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="timing_benchmark_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed progress information"
    )

    # Handle case where script is run directly without arguments
    # sys already imported at top

    if len(sys.argv) == 1:
        # No arguments provided, use defaults
        args = argparse.Namespace(
            backends="mock,aimnet2,uma,so3lr,mace,torchsim_mace,torchsim_uma",
            device="cpu",
            output="timing_benchmark_results.json",
            verbose=False,
        )
    else:
        args = parser.parse_args()

    print("🚀 QME Backend Timing Benchmark")
    print("=" * 50)

    # Determine which backends to test
    if args.backends:
        requested_backends = [b.strip() for b in args.backends.split(",")]
        available_backends = filter_available_backends(requested_backends, verbose=True)

        if not available_backends:
            print(
                "\nNo requested backends are available! Please install required dependencies."
            )
            return
    else:
        available_backends = get_available_backends(verbose=True)

        if not available_backends:
            print("\nNo backends available! Please install required dependencies.")
            return

    print_backend_summary(available_backends, "Benchmarking Backends")
    print(f"Device: {args.device}")
    print(f"Output: {args.output}")
    print()

    print(f"Running benchmarks for {len(available_backends)} backend(s)...")

    # Run benchmarks
    results_list = []
    for backend in available_backends:
        try:
            results = benchmark_backend(
                backend=backend, device=args.device, verbose=args.verbose
            )
            results_list.append(results)
        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user.")
            break
        except Exception as e:
            print(f"\nUnexpected error with {backend}: {e}")
            results_list.append(
                {
                    "backend": backend,
                    "available": False,
                    "error": str(e),
                    "timings": {},
                    "optimization_results": {},
                    "frequency_results": {},
                }
            )

    # Print summary
    print_summary(results_list)

    # Save results
    save_results(results_list, args.output)

    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()
