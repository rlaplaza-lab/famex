#!/usr/bin/env python3
"""
QME TS Optimizer Benchmark - Transition State Optimizer Comparison

This benchmark compares the performance of different transition state optimizers
(sella and geometric) for transition state finding using various QME ML backends.
It focuses specifically on TS optimization to evaluate which optimizers work best
for finding transition states across different ML backends.

Usage:
    conda run -n py312 python optimizer_comparison_benchmark.py [--backends BACKEND1,BACKEND2,...]
    conda run -n py312 python optimizer_comparison_benchmark.py [--optimizers OPT1,OPT2,...]
    conda run -n py312 python optimizer_comparison_benchmark.py [--device DEVICE]

Features:
    - Transition state optimizer comparison (sella vs geometric)
    - All available ML backends tested
    - Detailed timing and convergence analysis
    - TS-specific optimization evaluation
    - Focus on TS finding capabilities
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
from device_utils import get_optimal_device, print_device_info

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def get_available_ml_backends() -> List[str]:
    """Get list of available ML backends (excluding mock)."""
    available = []
    ml_backends = ["aimnet2", "uma", "so3lr", "mace", "torchsim_mace", "torchsim_uma"]

    for backend in ml_backends:
        if calculator_registry.is_backend_available(backend):
            available.append(backend)

    return available


def filter_available_backends(
    requested_backends: List[str], verbose: bool = False
) -> List[str]:
    """Filter requested backends to only available ones."""
    available = []
    for backend in requested_backends:
        if calculator_registry.is_backend_available(backend):
            available.append(backend)
        elif verbose:
            print(f"Warning: Backend '{backend}' not available, skipping")
    return available


def print_backend_summary(backends: List[str], title: str = "Available Backends"):
    """Print a formatted summary of backends."""
    print(f"\n📋 {title}")
    print("-" * 50)
    for i, backend in enumerate(backends, 1):
        print(f"  {i}. {backend}")
    print(f"Total: {len(backends)} backends")


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


def benchmark_optimizer(
    backend: str,
    optimizer: str,
    device: Optional[str] = None,
    model_name: Optional[str] = None,
    verbose: bool = True,
    test_ts: bool = False,
) -> Dict[str, Any]:
    """
    Benchmark a single backend with a specific optimizer for optimization and frequency analysis.

    Parameters:
    -----------
    backend : str
        Backend name (e.g., 'mock', 'aimnet2', 'uma', 'so3lr', 'mace')
    optimizer : str
        Optimizer name (e.g., 'sella', 'geometric', 'lbfgs', 'bfgs')
    device : str, optional
        Device to use ('cpu' or 'cuda'). Auto-detected if None.
    model_name : str, optional
        Specific model name to use
    verbose : bool
        Whether to print progress information
    test_ts : bool
        Whether to test transition state optimization

    Returns:
    --------
    Dict[str, Any]
        Benchmark results including timings for each step
    """
    # Auto-detect optimal device
    device = get_optimal_device(device)

    if verbose:
        print(f"\n{'='*60}")
        print("Backend: {}".format(backend.upper()))
        print("Optimizer: {}".format(optimizer.upper()))
        print_device_info(device)
        print("Model: {}".format(model_name or 'default'))
        print("Test Type: {}".format('Transition State' if test_ts else 'Minima'))
        print("-" * 60)

    results = {
        "backend": backend,
        "optimizer": optimizer,
        "device": device,
        "model_name": model_name,
        "test_ts": test_ts,
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
            local_optimizer=optimizer,
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
            print(
                "Testing single energy calculation (first call - includes model loading)..."
            )
        energy_first_start = time.perf_counter()

        energy = explorer.atoms_list[0].get_potential_energy()

        energy_first_time = time.perf_counter() - energy_first_start
        results["timings"]["single_energy_first"] = energy_first_time

        if verbose:
            print(f"First energy calculation time: {energy_first_time:.3f} seconds")
            print(f"Energy: {energy:.6f} eV")

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

        # Geometry optimization (minima or transition state)
        if verbose:
            opt_type = (
                "transition state optimization" if test_ts else "geometry optimization"
            )
            print(f"Running {opt_type}...")
        opt_start = time.perf_counter()

        # Use QME's optimization methods
        if test_ts:
            # For TS optimization, create a TS guess by slightly distorting the molecule
            ts_guess = benzene.copy()
            positions = ts_guess.get_positions()
            # Add some distortion to create a TS-like geometry
            positions[0, 0] += 0.2  # Move first carbon
            positions[1, 0] -= 0.2  # Move second carbon
            ts_guess.set_positions(positions)

            # Update explorer with TS guess
            explorer.atoms_list = [ts_guess]
            explorer._create_and_attach_calculator(ts_guess)

            opt_result = explorer.find_transition_state(fmax=0.01, steps=1000)
            steps_taken = opt_result.get("steps_taken", 0)
            converged = opt_result.get("converged", False)
            final_energy = opt_result.get("final_energy", None)
            max_force = opt_result.get("max_force", None)
            optimized_atoms = opt_result.get("optimized_atoms", ts_guess)
        else:
            # Minima optimization
            opt_result = explorer.optimize_minimum(fmax=0.01, steps=1000)
            steps_taken = opt_result.get("steps_taken", 0)
            converged = opt_result.get("converged", False)
            final_energy = opt_result.get("final_energy", None)
            max_force = opt_result.get("max_force", None)
            optimized_atoms = opt_result.get("optimized_atoms", benzene)

        opt_time = time.perf_counter() - opt_start

        # Calculate average time per optimization step
        if isinstance(steps_taken, (int, float)) and steps_taken > 0:
            avg_time_per_step = opt_time / steps_taken
        else:
            avg_time_per_step = None

        # Get final forces for verification
        if optimized_atoms is not None:
            final_forces = optimized_atoms.get_forces()
            final_max_force = float(np.max(np.abs(final_forces)))
        else:
            final_max_force = max_force

        opt_results = {
            "converged": converged,
            "final_energy": final_energy,
            "steps_taken": steps_taken,
            "optimized_atoms": optimized_atoms,
            "max_force": final_max_force,
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

        # Use optimized atoms for frequency analysis
        if optimized_atoms is not None:
            # Create a new explorer with optimized atoms
            freq_explorer = Explorer(
                atoms=optimized_atoms,
                backend=backend,
                model_name=model_name,
                device=device,
                default_charge=0,
                default_spin=1,
                local_optimizer=optimizer,
            )
            freq_explorer._create_and_attach_calculator(optimized_atoms)
        else:
            freq_explorer = explorer

        freq_results = freq_explorer.calculate_frequencies(
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


def print_optimizer_summary(results_list: List[Dict[str, Any]]):
    """Print a summary table focused on optimizer comparison."""
    print(f"\n{'='*140}")
    print("OPTIMIZER COMPARISON SUMMARY")
    print(f"{'='*140}")

    # Print legend first, before the table
    print("📊 COLUMN DEFINITIONS:")
    print("   Backend  = ML backend used")
    print("   Optimizer = Optimization algorithm")
    print("   Type     = Minima (M) or Transition State (TS)")
    print("   Converged = Whether optimization converged")
    print("   Steps    = Number of optimization steps")
    print("   Time/Step = Average time per optimization step")
    print("   Total    = Total optimization time")
    print("   Final E  = Final energy (eV)")
    print("   Max F    = Maximum force (eV/Å)")
    print(f"{'-'*140}")

    # Header
    print(
        f"{'Backend':<12} {'Optimizer':<12} {'Type':<4} {'Converged':<10} {'Steps':<8} "
        f"{'Time/Step (s)':<14} {'Total (s)':<10} {'Final E (eV)':<12} {'Max F (eV/Å)':<12}"
    )
    print("=" * 140)

    # Results
    for results in results_list:
        if results["available"]:
            timings = results["timings"]
            opt_results = results.get("optimization_results", {})
            steps_taken = opt_results.get("steps_taken", 0)
            avg_time_per_step = timings.get("avg_time_per_step", 0)
            optimizer = results.get("optimizer", "unknown")
            test_type = "TS" if results.get("test_ts", False) else "M"
            converged = opt_results.get("converged", False)
            final_energy = opt_results.get("final_energy", None)
            max_force = opt_results.get("max_force", None)

            # Handle None values for formatting
            steps_str = str(steps_taken) if steps_taken is not None else "N/A"
            avg_time_str = (
                f"{avg_time_per_step:.4f}" if avg_time_per_step is not None else "N/A"
            )
            final_energy_str = (
                f"{final_energy:.3f}" if final_energy is not None else "N/A"
            )
            max_force_str = f"{max_force:.6f}" if max_force is not None else "N/A"

            print(
                f"{results['backend']:<12} {optimizer:<12} {test_type:<4} "
                f"{'Yes' if converged else 'No':<10} {steps_str:<8} "
                f"{avg_time_str:<14} "
                f"{timings.get('optimization', 0):<10.3f} "
                f"{final_energy_str:<12} "
                f"{max_force_str:<12}"
            )
        else:
            optimizer = results.get("optimizer", "unknown")
            test_type = "TS" if results.get("test_ts", False) else "M"
            print(
                f"{results['backend']:<12} {optimizer:<12} {test_type:<4} "
                f"{'N/A':<10} {'N/A':<8} {'N/A':<14} {'N/A':<10} {'N/A':<12} {'N/A':<12}"
            )

    print("=" * 140)

    # Optimizer performance analysis
    available_results = [r for r in results_list if r["available"]]
    if available_results:
        print("\n🔍 OPTIMIZER PERFORMANCE ANALYSIS")
        print(f"{'='*80}")

        # Group by optimizer
        optimizer_groups = {}
        for result in available_results:
            opt_name = result.get("optimizer", "unknown")
            if opt_name not in optimizer_groups:
                optimizer_groups[opt_name] = []
            optimizer_groups[opt_name].append(result)

        for opt_name, opt_results in optimizer_groups.items():
            print(f"\n📈 {opt_name.upper()} OPTIMIZER PERFORMANCE")
            print(f"{'-'*50}")

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
                        "Avg Time/Step", np.mean(time_per_step_list), np.std(time_per_step_list)
                    )
                )
                print(f"  {'Min Time/Step':<30}: {min(time_per_step_list):>8.4f}s")
                print(f"  {'Max Time/Step':<30}: {max(time_per_step_list):>8.4f}s")

            if total_time_list:
                print(
                    "  {:<30}: {:.3f}s ± {:.3f}s".format(
                        "Avg Total Time", np.mean(total_time_list), np.std(total_time_list)
                    )
                )
                print(f"  {'Min Total Time':<30}: {min(total_time_list):>8.3f}s")
                print(f"  {'Max Total Time':<30}: {max(total_time_list):>8.3f}s")

            if converged_list:
                convergence_rate = sum(converged_list) / len(converged_list) * 100
                print(f"  {'Convergence Rate':<30}: {convergence_rate:>8.1f}%")

            print(f"  {'Total Tests':<30}: {len(opt_results):>8}")


def save_results(results_list: List[Dict[str, Any]], output_file: str):
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
    """Main function to run the optimizer comparison benchmark."""
    # Create standardized interface
    interface = QMEExampleInterface(
        name="TS Optimizer Benchmark",
        description="Transition State Optimizer Comparison",
        epilog=create_standard_epilog("benchmark")
    )
    
    parser = interface.create_parser()
    
    # Add optimizer-specific arguments
    parser.add_argument(
        "--optimizers",
        type=str,
        help="Comma-separated list of TS optimizers to benchmark (default: sella,geometric)",
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

    # Determine which TS optimizers to test
    if args.optimizers:
        requested_optimizers = [o.strip() for o in args.optimizers.split(",")]
        # Filter to only available TS optimizers
        available_optimizers = []
        for opt in requested_optimizers:
            if opt.lower() in ["sella", "geometric"]:
                available_optimizers.append(opt.lower())
            else:
                print(f"Warning: Unknown TS optimizer '{opt}', skipping. Available: sella, geometric")
    else:
        # Default TS optimizers
        available_optimizers = ["sella", "geometric"]

    interface.print_backend_summary(available_backends, "Benchmarking Backends")
    print(f"\nTS Optimizers: {', '.join(available_optimizers)}")
    print(f"Test Type: Transition State")
    
    # Get device info
    device = interface.get_device_info(args.device)
    
    config = {
        "Device": device,
        "Output": args.output or interface.get_default_output_file(),
        "Verbose": args.verbose,
        "Test Type": "Transition State"
    }
    interface.print_configuration(config)

    print(
        "\nRunning benchmarks for {} backend(s) × {} optimizer(s)...".format(
            len(available_backends), len(available_optimizers)
        )
    )

    # Run benchmarks
    results_list = []
    for backend in available_backends:
        for optimizer in available_optimizers:
            try:
                results = benchmark_optimizer(
                    backend=backend,
                    optimizer=optimizer,
                    device=device,
                    verbose=args.verbose,
                    test_ts=True,  # Always TS optimization
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
                        "device": args.device,
                        "test_ts": args.test_ts,
                        "available": False,
                        "error": str(e),
                        "timings": {},
                        "optimization_results": {},
                        "frequency_results": {},
                    }
                )

    # Print summary
    print_optimizer_summary(results_list)

    # Save results
    save_results(results_list, args.output or interface.get_default_output_file())

    interface.print_success()
    return 0


if __name__ == "__main__":
    sys.exit(main())
