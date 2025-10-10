#!/usr/bin/env python3
"""
QME Minima Optimizer Benchmark - Minima Optimization Comparison

This benchmark compares the performance of different minima optimizers
(lbfgs, bfgs, fire) across various QME ML backends. It focuses specifically
on minima optimization to evaluate which optimizers work best for finding
energy minima across different ML backends.

Usage:
    python minima_optimizer_benchmark.py [--backends BACKEND1,BACKEND2,...]
    python minima_optimizer_benchmark.py [--optimizers lbfgs,bfgs,fire,geometric]
    python minima_optimizer_benchmark.py [--device DEVICE]

Features:
    - Minima optimizer comparison (lbfgs, bfgs, fire, geometric)
    - All available ML backends tested
    - Detailed timing and convergence analysis
    - Minima-specific optimization evaluation
    - Focus on minima finding capabilities
"""

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from ase import Atoms

# Import QME components
try:
    from qme.analysis.frequency import FrequencyAnalysis
    from qme.calculator_registry import calculator_registry
    from qme.core.explorer import Explorer
except ImportError as e:
    print(f"❌ Error importing QME: {e}")
    print("   Please ensure QME is installed and accessible")
    sys.exit(1)

# Common interface and device utils
from common_interface import QMEExampleInterface, create_standard_epilog

# Backend availability helpers
from qme.backend_availability import get_available_ml_backends
from qme.utils.device import get_optimal_device, print_device_info

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def create_minima_structure() -> Atoms:
    """Create a structure for minima optimization using example files."""
    import os

    from ase.io import read

    # Use the reactant structure as starting point for minima optimization
    script_dir = os.path.dirname(os.path.abspath(__file__))
    structure = read(os.path.join(script_dir, "example_files", "reaction_001_reactant.xyz"))
    return structure


def benchmark_minima_optimizer(
    backend: str,
    optimizer: str,
    device: Optional[str] = None,
    model_name: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Benchmark a single backend with a specific optimizer for minima optimization.

    Only suitable optimizers: LBFGS, BFGS, FIRE
    """
    return _benchmark_optimization(
        backend=backend,
        optimizer=optimizer,
        device=device,
        model_name=model_name,
        verbose=verbose,
        test_ts=False,
        create_structure_func=create_minima_structure,
        suitable_optimizers=["lbfgs", "bfgs", "fire", "geometric"],
    )


def _benchmark_optimization(
    backend: str,
    optimizer: str,
    device: Optional[str] = None,
    model_name: Optional[str] = None,
    verbose: bool = True,
    test_ts: bool = False,
    create_structure_func=None,
    suitable_optimizers: List[str] = None,
) -> Dict[str, Any]:
    """
    Internal benchmark function for optimization and frequency analysis.

    Parameters:
    -----------
    backend : str
        Backend name (e.g., 'mock', 'aimnet2', 'uma', 'so3lr', 'mace')
    optimizer : str
        Optimizer name (e.g., 'lbfgs', 'bfgs', 'fire')
    device : str, optional
        Device to use ('cpu' or 'cuda'). Auto-detected if None.
    model_name : str, optional
        Specific model name to use
    verbose : bool
        Whether to print progress information
    test_ts : bool
        Whether to test transition state optimization
    create_structure_func : callable
        Function to create the initial structure
    suitable_optimizers : List[str]
        List of optimizers suitable for this task

    Returns:
    --------
    Dict[str, Any]
        Benchmark results including timings for each step
    """
    # Auto-detect optimal device
    device = get_optimal_device(device)

    if verbose:
        print(f"\n{'=' * 60}")
        print("Backend: {}".format(backend.upper()))
        print("Optimizer: {}".format(optimizer.upper()))
        print_device_info(device)
        print("Model: {}".format(model_name or "default"))
        print("Test Type: Minima Optimization")
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

        # Check if optimizer is suitable for this task
        if suitable_optimizers and optimizer.lower() not in suitable_optimizers:
            task = "TS" if test_ts else "minima"
            results["error"] = (
                f"Optimizer {optimizer} not suitable for {task} optimization. "
                f"Suitable: {', '.join(suitable_optimizers)}"
            )
            if verbose:
                print(f"Status: ❌ {results['error']}")
            return results

        results["available"] = True
        if verbose:
            print("Status: ✅ Backend available")

        # Create appropriate structure
        if verbose:
            print("Loading reactant structure for minima optimization...")
        structure = create_structure_func()

        # Initialize QME optimizer
        if verbose:
            print("Initializing QME optimizer...")
        init_start = time.perf_counter()

        explorer = Explorer(
            atoms=structure,
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

        # Minima optimization using Explorer strategies
        if verbose:
            print("Running minima optimization...")
        opt_start = time.perf_counter()

        # Use Explorer's run method with minima strategy
        run_results = explorer.run(
            mode="minima",
            fmax=0.01,  # Standard criteria for minima
            steps=1000,
            local_optimizer_name=optimizer,
        )

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

        # Get optimization results
        if optimized_atoms is not None and hasattr(optimized_atoms, "get_potential_energy"):
            final_energy = float(optimized_atoms.get_potential_energy())
            max_force = float(np.max(np.abs(optimized_atoms.get_forces())))
        else:
            final_energy = 0.0
            max_force = float("inf")

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

        # Frequency analysis (mandatory)
        if verbose:
            print("Running frequency analysis...")
        freq_start = time.perf_counter()

        # Use the explorer's calculate_frequencies method directly
        # This method handles the calculator attachment automatically
        freq_results = explorer.calculate_frequencies(
            atoms=optimized_atoms,  # Use optimized atoms if available
            delta=0.01,
            method="auto",
            temperature=298.15,
            save_hessian=False,  # Don't save large Hessian matrix
        )

        freq_time = time.perf_counter() - freq_start
        results["timings"]["frequency_analysis"] = freq_time

        # Enhanced validation based on task type
        frequencies = freq_results["frequencies"]
        n_imaginary = sum(1 for f in frequencies if f < 0)
        is_ts = freq_results["is_ts"]

        # Quality check for minima optimization: expect 0 imaginary frequencies
        is_valid_result = (n_imaginary == 0) and not is_ts
        result_type = "minima"
        if not is_valid_result:
            if verbose:
                print(
                    f"⚠️  WARNING: Expected minima but found {n_imaginary} " "imaginary frequencies"
                )
                if n_imaginary > 0:
                    print(
                        "   This suggests the optimizer found a TS or "
                        "saddle point instead of a minimum"
                    )

        results["frequency_results"] = {
            "n_frequencies": len(frequencies),
            "frequencies": frequencies[:10],  # First 10 frequencies
            "zero_point_energy": freq_results["zero_point_energy"],
            "is_transition_state": is_ts,
            "is_valid_result": is_valid_result,
            "n_imaginary_frequencies": n_imaginary,
            "method_used": freq_results["method_used"],
            "result_type": result_type,
        }

        if verbose:
            print(f"Frequency analysis time: {freq_time:.3f} seconds")
            print(f"Number of frequencies: {len(frequencies)}")
            print(f"Imaginary frequencies: {n_imaginary}")
            print(f"First 5 frequencies: {frequencies[:5]}")
            print(f"Zero-point energy: {freq_results['zero_point_energy']:.6f} eV")
            print(f"Is transition state: {is_ts}")
            print(f"Valid minima: {is_valid_result}")

        # Calculate total time (excluding None values)
        total_time = sum(v for v in results["timings"].values() if v is not None)
        results["timings"]["total"] = total_time

        if verbose:
            print(f"\nTotal time: {total_time:.3f} seconds")
            print("Status: ✅ Completed successfully")

    except Exception as e:
        import traceback

        results["error"] = str(e)
        if verbose:
            print(f"Status: ❌ Error - {e}")
            print(f"Traceback: {traceback.format_exc()}")

    return results


def print_frequency_analysis_summary(results_list: List[Dict[str, Any]]):
    """Print a detailed frequency analysis summary for minima optimization."""
    print(f"\n{'=' * 120}")
    print("FREQUENCY ANALYSIS SUMMARY - MINIMA OPTIMIZATION")
    print(f"{'=' * 120}")

    # Print legend
    print("📊 FREQUENCY VALIDATION:")
    print("   • Minima should have 0 imaginary frequencies (all positive)")
    print("   • Invalid results suggest optimization found a saddle point or TS")
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
            result_type = freq_results.get("result_type", "unknown")

            # Format first 3 frequencies
            if len(frequencies) >= 3:
                freq_str = f"[{frequencies[0]:.1f}, {frequencies[1]:.1f}, {frequencies[2]:.1f}]"
            else:
                freq_str = "N/A"

            # Status indicator
            if is_valid:
                status = "✅ Valid"
            else:
                if n_imag > 0:
                    status = f"❌ {n_imag} Imag Freq"
                else:
                    status = "⚠️  Unknown"

            print(
                f"{results['backend']:<12} {results.get('optimizer', 'unknown'):<12} "
                f"{'M':<4} "
                f"{n_imag if n_imag is not None else 'N/A':<10} "
                f"{'Yes' if is_valid else 'No' if is_valid is not None else 'N/A':<8} "
                f"{zpe:<12.4f} {freq_str:<25} {status:<15}"
            )
        else:
            print(
                f"{results['backend']:<12} {results.get('optimizer', 'unknown'):<12} "
                f"{'M':<4} "
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

        # Minima-specific issues
        minima_with_imag = sum(
            1
            for r in available_results
            if r["frequency_results"].get("n_imaginary_frequencies", 0) > 0
        )
        if minima_with_imag > 0:
            print(f"  ⚠️  {minima_with_imag} minima optimizations found imaginary frequencies")


def print_optimizer_summary(results_list: List[Dict[str, Any]]):
    """Print a summary table focused on minima optimizer comparison."""
    print(f"\n{'=' * 140}")
    print("MINIMA OPTIMIZER COMPARISON SUMMARY")
    print(f"{'=' * 140}")

    # Print legend first, before the table
    print("📊 COLUMN DEFINITIONS:")
    print("   Backend  = ML backend used")
    print("   Optimizer = Optimization algorithm")
    print("   Type     = Minima (M)")
    print("   Converged = Whether optimization converged")
    print("   Steps    = Number of optimization steps")
    print("   Time/Step = Average time per optimization step")
    print("   Total    = Total optimization time")
    print("   Final E  = Final energy (eV)")
    print("   Max F    = Maximum force (eV/Å)")
    print("   Valid Result = Whether result matches expected type (0 imag freq for minima)")
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
                (
                    f"{results['backend']:<12} {optimizer:<12} {'M':<4} "
                    f"{'Yes' if converged else 'No':<10} {steps_str:<8} "
                    f"{avg_time_str:<14} "
                    f"{timings.get('optimization', 0):<10.3f} "
                    f"{final_energy_str:<12} "
                    f"{max_force_str:<12} "
                    f"{valid_result_str:<10}"
                )
            )
        else:
            optimizer = results.get("optimizer", "unknown")
            print(
                f"{results['backend']:<12} {optimizer:<12} {'M':<4} "
                f"{'N/A':<10} {'N/A':<8} {'N/A':<14} {'N/A':<10} "
                f"{'N/A':<12} {'N/A':<12} {'N/A':<10}"
            )

    print("=" * 150)

    # Optimizer performance analysis
    available_results = [r for r in results_list if r["available"]]
    if available_results:
        print("\n🔍 MINIMA OPTIMIZER PERFORMANCE ANALYSIS")
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

            # Quality analysis for minima optimization
            valid_result_list = [
                r["frequency_results"].get("is_valid_result", False)
                for r in opt_results
                if "frequency_results" in r
            ]
            if valid_result_list:
                minima_success_rate = sum(valid_result_list) / len(valid_result_list) * 100
                print(f"  {'Minima Success Rate':<30}: {minima_success_rate:>8.1f}%")

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
        help="Comma-separated list of optimizers to benchmark (default: lbfgs,bfgs,fire)",
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
        # Filter to only minima optimizers
        valid_optimizers = ["lbfgs", "bfgs", "fire", "geometric"]
        minima_optimizers = [opt for opt in requested_optimizers if opt in valid_optimizers]
        if len(minima_optimizers) != len(requested_optimizers):
            invalid_opts = [opt for opt in requested_optimizers if opt not in valid_optimizers]
            print(f"Warning: Invalid optimizers ignored: {', '.join(invalid_opts)}")
            print(f"Valid minima optimizers: {', '.join(valid_optimizers)}")
    else:
        minima_optimizers = ["lbfgs", "bfgs", "fire", "geometric"]

    if not minima_optimizers:
        interface.print_error("No valid minima optimizers specified!")
        print("Valid options: lbfgs, bfgs, fire, geometric")
        return 1

    interface.print_backend_summary(available_backends, "Benchmarking Backends")
    print(f"\nMinima Optimizers: {', '.join(minima_optimizers)}")
    print("Test Type: Minima Optimization")

    # Get device info
    device = interface.get_device_info(args.device)

    config = {
        "Device": device,
        "Output": args.output or interface.get_default_output_file(),
        "Verbose": args.verbose,
        "Test Types": "Minima Optimization",
    }
    interface.print_configuration(config)

    total_tests = len(available_backends) * len(minima_optimizers)
    print(
        f"\nRunning benchmarks for {len(available_backends)} backend(s) × "
        f"{len(minima_optimizers)} minima optimizer(s) = {total_tests} tests..."
    )

    # Run benchmarks
    results_list = []

    print(f"\n{'=' * 80}")
    print("MINIMA OPTIMIZATION BENCHMARKS")
    print(f"{'=' * 80}")

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
                print("\nBenchmark interrupted by user.")
                break
            except Exception as e:
                print(f"\nUnexpected error with {backend}+{optimizer}: {e}")
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
