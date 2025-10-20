#!/usr/bin/env python3
"""
Common interface utilities for QME examples.

This module provides standardized interfaces, output formatting, and common
functionality across all QME examples to ensure consistency.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any  # Dict, List, Optional  # Unused for now

# Import QME components
try:
    from qme.backend_availability import is_backend_available
    from qme.calculator_registry import calculator_registry
except ImportError as e:
    print(f"❌ Error importing QME: {e}")
    print("   Please ensure QME is installed and accessible")
    sys.exit(1)

# Import device utilities
from qme.utils.device import get_optimal_device, print_device_info


class QMEExampleInterface:
    """Base class for standardized QME examples."""

    def __init__(self, name: str, description: str, epilog: str = ""):
        self.name = name
        self.description = description
        self.epilog = epilog
        self.start_time = time.time()

    def create_parser(self) -> argparse.ArgumentParser:
        """Create standardized argument parser."""
        parser = argparse.ArgumentParser(
            description=self.description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self.epilog,
        )

        # Standard arguments for all examples
        parser.add_argument(
            "--backends",
            type=str,
            help="Comma-separated list of backends to test (default: all available ML backends)",
        )
        parser.add_argument(
            "--device",
            type=str,
            default=None,
            choices=["cpu", "cuda"],
            help="Device to use for calculations (default: auto-detect CUDA if available)",
        )
        parser.add_argument(
            "--verbose",
            type=int,
            choices=[0, 1, 2],
            default=1,
            help="Verbosity level: 0=quiet, 1=normal (default), 2=verbose"
        )
        parser.add_argument(
            "--output",
            type=str,
            help="Output file for results (default: auto-generated based on example name)",
        )

        return parser

    def get_available_ml_backends(self) -> list[str]:
        """Get list of available ML backends (excluding mock)."""
        from qme.backend_availability import get_available_ml_backends

        return get_available_ml_backends()

    def filter_available_backends(
        self, requested_backends: list[str], verbose: int = 0
    ) -> list[str]:
        """Filter requested backends to only available ones."""
        available = []
        for backend in requested_backends:
            if is_backend_available(backend):
                available.append(backend)
            elif verbose >= 1:
                print(f"Warning: Backend '{backend}' not available, skipping")

        return available

    def print_header(self, subtitle: str = ""):
        """Print standardized header."""
        print("=" * 80)
        print(f"QME {self.name}")
        if subtitle:
            print(f"{subtitle}")
        print("=" * 80)

    def print_backend_summary(self, backends: list[str], title: str = "Available Backends"):
        """Print standardized backend summary."""
        print(f"\n📋 {title}")
        print("-" * 50)
        for i, backend in enumerate(backends, 1):
            print(f"  {i}. {backend}")
        print(f"Total: {len(backends)} backends")

    def print_configuration(self, config: dict[str, Any]):
        """Print standardized configuration summary."""
        print("\nConfiguration:")
        for key, value in config.items():
            print(f"  {key}: {value}")

    def print_success(self, message: str = "Completed successfully!"):
        """Print standardized success message."""
        elapsed = time.time() - self.start_time
        print(f"\n✅ {message}")
        print(f"⏱️  Total time: {elapsed:.1f} seconds")

    def print_error(self, message: str):
        """Print standardized error message."""
        print(f"\n❌ {message}")

    def print_warning(self, message: str):
        """Print standardized warning message."""
        print(f"\n⚠️  {message}")

    def save_results(self, results: dict[str, Any], output_file: str | None = None):
        """Save results to JSON file."""
        if output_file is None:
            output_file = f"{self.name.lower().replace(' ', '_')}_results.json"

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path.absolute()}")

    def get_default_output_file(self) -> str:
        """Get default output file name for this example."""
        return f"{self.name.lower().replace(' ', '_')}_results.json"

    def get_device_info(self, device: str | None = None) -> str:
        """Get optimal device and print info."""
        device = get_optimal_device(device)
        print_device_info(device)
        return device

    def setup_logging(self, verbose: int = 1) -> None:
        """Set up QME logging based on verbosity level."""
        from qme.logging_utils import setup_qme_logging
        setup_qme_logging(verbosity=verbose)


def create_standard_epilog(example_type: str) -> str:
    """Create standardized epilog for different example types."""

    if example_type == "demo":
        return """
Examples:
  # Run with all available backends
  python cli_demo.py

  # Run with specific backends
  python cli_demo.py --backends uma,aimnet2

  # Run with verbose output
  python cli_demo.py --verbose 2
        """

    elif example_type == "timing":
        return """
Examples:
  # Run with all available backends
  python timing_benchmark.py

  # Run with specific backends
  python timing_benchmark.py --backends uma,aimnet2

  # Run on GPU
  python timing_benchmark.py --device cuda --verbose 2
        """

    elif example_type == "benchmark":
        return """
Examples:
  # Run with all available backends
  python benchmark.py

  # Run with specific backends
  python benchmark.py --backends uma,aimnet2

  # Run with verbose output
  python benchmark.py --verbose 2
        """

    elif example_type == "benchmark_quick":
        return """
Examples:
  # Run with all available backends
  python benchmark.py

  # Run with specific backends
  python benchmark.py --backends uma,aimnet2

  # Quick test with subset of data
  python benchmark.py --quick --verbose 2

  # Very quick test with minimal data
  python benchmark.py --quicker --verbose 2
        """

    else:
        return """
Examples:
  # Run with default settings
  python example.py

  # Run with specific backends
  python example.py --backends uma,aimnet2

  # Run with verbose output
  python example.py --verbose 2
        """


def benchmark_optimization(
    backend: str,
    optimizer: str,
    device: str | None = None,
    model_name: str | None = None,
    verbose: int = 1,
    test_ts: bool = False,
    create_structure_func=None,
    suitable_optimizers: list[str] = None,
) -> dict[str, Any]:
    """
    Common benchmark function for optimization and frequency analysis.

    This function eliminates code duplication between minima and TS optimizer benchmarks.

    Parameters:
    -----------
    backend : str
        Backend name (e.g., 'mock', 'aimnet2', 'uma', 'so3lr', 'mace', 'orb')
    optimizer : str
        Optimizer name (e.g., 'lbfgs', 'bfgs', 'fire', 'sella')
    device : str, optional
        Device to use ('cpu' or 'cuda'). Auto-detected if None.
    model_name : str, optional
        Specific model name to use
    verbose : int
        Verbosity level: 0=quiet, 1=normal, 2=verbose
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
    import time

    import numpy as np

    # Import QME components
    try:
        # from qme.analysis.frequency import FrequencyAnalysis  # Unused for now
        from qme.core.explorer import Explorer
    except ImportError as e:
        return {
            "backend": backend,
            "optimizer": optimizer,
            "available": False,
            "error": f"Failed to import QME: {e}",
            "timings": {},
            "optimization_results": {},
            "frequency_results": {},
        }

    # Auto-detect optimal device
    device = get_optimal_device(device)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Backend: {backend.upper()}")
        print(f"Optimizer: {optimizer.upper()}")
        print_device_info(device)
        print("Model: {}".format(model_name or "default"))
        test_type = "Transition State Optimization" if test_ts else "Minima Optimization"
        print(f"Test Type: {test_type}")
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
            structure_type = "TS" if test_ts else "reactant"
            print(f"Loading {structure_type} structure for optimization...")
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

        # Optimization using Explorer strategies
        if verbose:
            opt_type = "transition state optimization" if test_ts else "minima optimization"
            print(f"Running {opt_type}...")
        opt_start = time.perf_counter()

        # Use Explorer's run method with appropriate strategy
        try:
            if test_ts:
                run_results = explorer.run(
                    mode="ts",
                    fmax=0.005,  # Stricter criteria for TS
                    steps=1000,
                    local_optimizer_name=optimizer,
                )
            else:
                run_results = explorer.run(
                    mode="minima",
                    fmax=0.01,  # Standard criteria for minima
                    steps=1000,
                    local_optimizer_name=optimizer,
                )
        except ValueError as e:
            if "not suitable for transition state optimization" in str(e):
                results["error"] = str(e)
                if verbose:
                    print(f"Status: ❌ {results['error']}")
                return results
            raise

        # Handle results from Explorer's run method
        # For local strategies, run() returns a list of results
        if isinstance(run_results, list) and len(run_results) == 1:
            strategy_result = run_results[0]
        else:
            strategy_result = run_results

        # Extract step tracking information from strategy result
        if isinstance(strategy_result, dict):
            steps_taken = strategy_result.get("steps_taken", 0)
            converged_raw = strategy_result.get("converged", False)
            # Ensure converged is always a boolean
            if isinstance(converged_raw, str):
                converged = converged_raw.lower() in ("true", "1", "yes")
            else:
                converged = bool(converged_raw)
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

        # Enhanced validation based on task type using proper frequency analysis
        frequencies = freq_results["frequencies"]

        if test_ts:
            # TS optimization validation
            is_ts = freq_results["is_ts"]
            ts_analysis = freq_results.get("ts_analysis", {})
            n_imaginary = ts_analysis.get("n_imaginary_frequencies", 0)
            is_valid_result = is_ts and (n_imaginary == 1)
            result_type = "TS"

            if not is_valid_result:
                if verbose:
                    print(
                        f"⚠️  WARNING: Expected TS but found {n_imaginary} " "imaginary frequencies"
                    )
                    if n_imaginary == 0:
                        print("   This suggests the optimizer found a minimum instead of a TS")
                    elif n_imaginary > 1:
                        print(
                            "   This suggests the structure is not a proper TS "
                            "(too many imaginary frequencies)"
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
                "ts_analysis": ts_analysis,
            }

            if verbose:
                print(f"Frequency analysis time: {freq_time:.3f} seconds")
                print(f"Number of frequencies: {len(frequencies)}")
                print(f"Imaginary frequencies: {n_imaginary}")
                print(f"First 5 frequencies: {frequencies[:5]}")
                print(f"Zero-point energy: {freq_results['zero_point_energy']:.6f} eV")
                print(f"Is transition state: {is_ts}")
                print(f"Valid TS: {is_valid_result}")
        else:
            # Minima optimization validation
            is_minimum = freq_results["is_minimum"]
            minima_analysis = freq_results.get("minima_analysis", {})
            n_significant_imaginary = minima_analysis.get("n_significant_imaginary_frequencies", 0)
            n_small_negative = minima_analysis.get("n_small_negative_frequencies", 0)
            is_valid_result = is_minimum
            result_type = "minima"

            if not is_valid_result:
                if verbose:
                    print(
                        f"⚠️  WARNING: Expected minima but found {n_significant_imaginary} "
                        f"significant imaginary frequencies"
                    )
                    if n_significant_imaginary > 0:
                        print(
                            "   This suggests the optimizer found a TS or "
                            "saddle point instead of a minimum"
                        )
                    if n_small_negative > 0:
                        print(
                            f"   Note: {n_small_negative} small negative frequencies detected "
                            f"(likely numerical noise)"
                        )

            results["frequency_results"] = {
                "n_frequencies": len(frequencies),
                "frequencies": frequencies[:10],  # First 10 frequencies
                "zero_point_energy": freq_results["zero_point_energy"],
                "is_minimum": is_minimum,
                "is_valid_result": is_valid_result,
                "n_significant_imaginary_frequencies": n_significant_imaginary,
                "n_small_negative_frequencies": n_small_negative,
                "method_used": freq_results["method_used"],
                "result_type": result_type,
                "minima_analysis": minima_analysis,
            }

            if verbose:
                print(f"Frequency analysis time: {freq_time:.3f} seconds")
                print(f"Number of frequencies: {len(frequencies)}")
                print(f"Significant imaginary frequencies: {n_significant_imaginary}")
                if n_small_negative > 0:
                    print(f"Small negative frequencies (likely noise): {n_small_negative}")
                print(f"First 5 frequencies: {frequencies[:5]}")
                print(f"Zero-point energy: {freq_results['zero_point_energy']:.6f} eV")
                print(f"Is minimum: {is_minimum}")
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


def print_standard_help():
    """Print standardized help information."""
    print("\n🔧 QME Examples Help")
    print("=" * 50)
    print("All QME examples follow a consistent interface:")
    print("\nCommon Options:")
    print("  --backends    Comma-separated list of backends to test")
    print("  --device      Device to use (cpu/cuda, default: auto-detect)")
    print("  --verbose     Verbosity level: 0=quiet, 1=normal, 2=verbose")
    print("  --output      Output file for results")
    print("  --help        Show this help message")
    print("\nAvailable Backends:")
    print("  uma           UMA (Meta AI, default)")
    print("  aimnet2       AIMNet2 (native PyTorch)")
    print("  mace          MACE (foundation models)")
    print("  so3lr         SO3LR (SO(3) neural networks)")
    print("  torchsim_*    TorchSim acceleration variants")
    print("\nFor more information, see the main README.md")
