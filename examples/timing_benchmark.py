#!/usr/bin/env python3
"""
QME Timing Benchmark - ML Backend Performance Analysis

This benchmark evaluates the performance of different QME ML backends for simple
geometry optimization and frequency analysis using benzene as a test case. All
backends use the same default optimizer (BFGS) to ensure fair comparison of
backend performance rather than optimizer differences.

Usage:
    python timing_benchmark.py [--backends BACKEND1,BACKEND2,...]
    python timing_benchmark.py [--device DEVICE]

Features:
    - Simple geometry optimization + frequency analysis
    - All backends use same default optimizer (BFGS)
    - Individual energy and force calculation benchmarks
    - Detailed timing breakdown and performance comparison
    - ML backend performance comparison (not optimizer comparison)
"""

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from ase import Atoms
from ase.build import molecule

# Use consolidated backend availability from qme.backend_availability
from qme.backend_availability import get_available_ml_backends

# Import QME components
try:
    from qme.analysis.frequency import FrequencyAnalysis
    from qme.calculator_registry import calculator_registry
    from qme.core.explorer import Explorer
except ImportError as e:
    print(f"❌ Error importing QME: {e}")
    print("   Please ensure QME is installed and accessible")
    sys.exit(1)

# Import common interface
from common_interface import QMEExampleInterface, create_standard_epilog

from qme.utils.device import get_optimal_device, print_device_info

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# QMEExampleInterface already imported above; remove duplicate import


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
    device: Optional[str] = None,
    model_name: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Benchmark a single backend for optimization and frequency analysis.

    Parameters:
    -----------
    backend : str
        Backend name (e.g., 'mock', 'aimnet2', 'uma', 'so3lr', 'mace')
    device : str, optional
        Device to use ('cpu' or 'cuda'). Auto-detected if None.
    model_name : str, optional
        Specific model name to use
    verbose : bool
        Whether to print progress information

    Returns:
    --------
    Dict[str, Any]
        Benchmark results including timings for each step
    """
    # Auto-detect optimal device
    device = get_optimal_device(device)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Backend: {backend.upper()}")
        print_device_info(device)
        print(f"Model: {model_name or 'default'}")
        print("-" * 60)

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
                print("Status: ❌ Backend not available")
            return results

        results["available"] = True
        if verbose:
            print("Status: ✅ Backend available")

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
            print(f"Initialization time: {init_time:.3f} seconds")

        # Attach calculator to atoms object
        if verbose:
            print("Attaching calculator to atoms...")
        load_start = time.perf_counter()

        # Attach calculator using Explorer's method
        explorer._create_and_attach_calculator(explorer.atoms_list[0])

        load_time = time.perf_counter() - load_start
        results["timings"]["structure_loading"] = load_time

        if verbose:
            print(f"Calculator attachment time: {load_time:.3f} seconds")

        # Test single energy calculation (first call - includes calculator initialization)
        if verbose:
            print("Testing single energy calculation (first call - includes model loading)...")
        energy_first_start = time.perf_counter()

        energy = explorer.atoms_list[0].get_potential_energy()

        energy_first_time = time.perf_counter() - energy_first_start
        results["timings"]["single_energy_first"] = energy_first_time

        if verbose:
            print(f"First energy calculation time: {energy_first_time:.3f} seconds")
            print(f"Energy: {energy:.6f} eV")

        # Test single energy calculation (second call - pure evaluation)
        if verbose:
            print("Testing single energy calculation (second call - pure evaluation)...")
        energy_second_start = time.perf_counter()

        energy2 = explorer.atoms_list[0].get_potential_energy()

        energy_second_time = time.perf_counter() - energy_second_start
        results["timings"]["single_energy_second"] = energy_second_time

        if verbose:
            print(f"Second energy calculation time: {energy_second_time:.3f} seconds")
            print(f"Energy: {energy2:.6f} eV")

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
            print(f"Single force calculation time: {force_time:.3f} seconds")
            print(f"Max force: {max_force:.6f} eV/Å")

        # Geometry optimization using Explorer strategies
        if verbose:
            print("Running geometry optimization...")
        opt_start = time.perf_counter()

        # Use Explorer's run method with proper strategy
        run_results = explorer.run(
            mode="minima",
            fmax=0.01,
            steps=1000,
            local_optimizer_name="BFGS",
        )

        opt_time = time.perf_counter() - opt_start

        # Handle results from Explorer's run method
        # For local strategies, run() returns a list of results
        if isinstance(run_results, list) and len(run_results) == 1:
            strategy_result = run_results[0]
        else:
            strategy_result = run_results

        # Extract step tracking information from strategy result
        if isinstance(strategy_result, dict):
            steps_taken = strategy_result.get("steps_taken", 0)
            converged = strategy_result.get("converged", False)
            optimized_atoms = strategy_result.get("optimized_atoms", explorer.atoms_list[0])
        else:
            # Fallback for old format
            steps_taken = 0
            converged = True
            optimized_atoms = strategy_result

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

        if verbose:
            print(f"Optimization time: {opt_time:.3f} seconds")
            print(f"Steps taken: {steps_taken}")
            if avg_time_per_step is not None:
                print(f"Average time per step: {avg_time_per_step:.4f} seconds")
            print(f"Converged: {opt_results['converged']}")
            print(f"Final energy: {opt_results['final_energy']:.6f} eV")
            print(f"Max force: {opt_results['max_force']:.6f} eV/Å")

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
            print(f"Frequency analysis time: {freq_time:.3f} seconds")
            print(f"Number of frequencies: {len(freq_results['frequencies'])}")
            print(f"First 5 frequencies: {freq_results['frequencies'][:5]}")
            print(f"Zero-point energy: {freq_results['zero_point_energy']:.6f} eV")
            print(f"Is transition state: {freq_results['is_ts']}")

        # Calculate total time (excluding None values)
        total_time = sum(v for v in results["timings"].values() if v is not None)
        results["timings"]["total"] = total_time

        if verbose:
            print(f"\nTotal time: {total_time:.3f} seconds")
            print("Status: ✅ Completed successfully")

    except Exception as e:
        results["error"] = str(e)
        if verbose:
            print(f"Status: ❌ Error - {e}")

    return results


def print_summary(results_list: List[Dict[str, Any]]):
    """Print a summary table of all benchmark results."""
    print(f"\n{'=' * 120}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 120}")

    # Print legend first, before the table
    print("📊 COLUMN DEFINITIONS:")
    print("   E1st    = Energy (1st call, includes model loading)")
    print("   E2nd    = Energy (2nd call, pure evaluation)")
    print("   Init    = Initialization time")
    print("   Opt     = Total optimization time")
    print("   Freq    = Frequency analysis time")
    print("   Forces  = Single force calculation time")
    print("   Steps   = Number of optimization steps taken")
    print("   Avg/Step = Average time per optimization step")
    print(f"{'-' * 120}")

    # Header
    print(
        f"{'Backend':<12} {'Available':<10} {'Total (s)':<10} {'Init (s)':<10} "
        f"{'Opt (s)':<10} {'Freq (s)':<10} {'E1st (s)':<10} {'E2nd (s)':<10} "
        f"{'Forces (s)':<10} {'Steps':<8} {'Avg/Step (s)':<12}"
    )
    print("=" * 120)

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

    print("=" * 120)

    # Detailed breakdown for available backends
    available_results = [r for r in results_list if r["available"]]
    if available_results:
        print("\n🔍 DETAILED BREAKDOWN")
        print(f"{'=' * 80}")

        for results in available_results:
            backend_name = results["backend"].upper()
            print(f"\n📈 {backend_name} PERFORMANCE BREAKDOWN")
            print(f"{'-' * 50}")

            timings = results["timings"]
            total = timings.get("total", 0)
            opt_results = results.get("optimization_results", {})

            # Timing breakdown with percentages
            individual_steps = [
                ("initialization", "Initialization"),
                ("structure_loading", "Calculator Attachment"),
                ("single_energy_first", "Energy (1st call + model loading)"),
                ("single_energy_second", "Energy (2nd call, pure eval)"),
                ("single_forces", "Force Calculation"),
                ("optimization", "Geometry Optimization"),
                ("frequency_analysis", "Frequency Analysis"),
            ]

            for step_key, step_display in individual_steps:
                time_val = timings.get(step_key, 0)
                if time_val is not None and time_val > 0:
                    percentage = (time_val / total) * 100 if total > 0 else 0
                    print(f"  {step_display:<30}: {time_val:>8.3f}s ({percentage:>5.1f}%)")

            # Optimization metrics
            steps_taken = opt_results.get("steps_taken", 0)
            avg_time_per_step = timings.get("avg_time_per_step", 0)
            final_energy = opt_results.get("final_energy", None)
            converged = opt_results.get("converged", None)

            print(f"  {'-' * 48}")
            if steps_taken > 0:
                print(f"  {'Optimization Steps':<30}: {steps_taken:>8}")
                if avg_time_per_step is not None and avg_time_per_step > 0:
                    print(f"  {'Time per Step':<30}: {avg_time_per_step:>8.4f}s")

            if final_energy is not None:
                print(f"  {'Final Energy':<30}: {final_energy:>8.3f} eV")

            if converged is not None:
                status = "✅ Yes" if str(converged) == "True" else "❌ No"
                print(f"  {'Converged':<30}: {status:>8}")

            print(f"  {'=' * 48}")
            print(f"  {'TOTAL TIME':<30}: {total:>8.3f}s")


def save_results(results_list: List[Dict[str, Any]], output_file: str):
    """Save benchmark results to JSON file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results_list, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


def main():
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

    interface.print_backend_summary(available_backends, "Benchmarking Backends")

    # Get device info
    device = interface.get_device_info(args.device)

    config = {
        "Device": device,
        "Output": args.output or interface.get_default_output_file(),
        "Verbose": args.verbose,
    }
    interface.print_configuration(config)

    print(f"\nRunning benchmarks for {len(available_backends)} backend(s)...")

    # Run benchmarks
    results_list = []
    for backend in available_backends:
        try:
            results = benchmark_backend(backend=backend, device=device, verbose=args.verbose)
            results_list.append(results)
        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user.")
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
                }
            )

    # Print summary
    print_summary(results_list)

    # Save results
    save_results(results_list, args.output or interface.get_default_output_file())

    interface.print_success()
    return 0


if __name__ == "__main__":
    sys.exit(main())
