#!/usr/bin/env python3
"""Common interface utilities for QME examples.

This module provides standardized interfaces, output formatting, and common
functionality across all QME examples to ensure consistency.
"""

import argparse
import json
import time
from collections.abc import Callable, Sequence, Sized
from pathlib import Path
from typing import Any

# Import QME components
from qme.backends.availability import is_backend_available
from qme.backends.registry import calculator_registry

# Import device utilities
from qme.utils.device import get_optimal_device, print_device_info
from qme.utils.logging import get_qme_logger

logger = get_qme_logger(__name__)


class QMEExampleInterface:
    """Base class for standardized QME examples."""

    def __init__(self, name: str, description: str, epilog: str = "") -> None:
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
            help="Verbosity level: 0=quiet, 1=normal (default), 2=verbose",
        )
        parser.add_argument(
            "--output",
            type=str,
            help="Output file for results (default: auto-generated based on example name)",
        )

        return parser

    def get_available_ml_backends(self) -> list[str]:
        """Get list of available ML backends (excluding mock)."""
        from qme.backends.availability import get_available_ml_backends

        return get_available_ml_backends()

    def filter_available_backends(
        self,
        requested_backends: list[str],
        verbose: int = 0,
    ) -> list[str]:
        """Filter requested backends to only available ones."""
        available = []
        for backend in requested_backends:
            if is_backend_available(backend):
                available.append(backend)
            elif verbose >= 1:
                logger.warning("Backend '%s' not available, skipping", backend)

        return available

    def print_header(self, subtitle: str = "") -> None:
        """Print standardized header."""
        logger.info("=" * 80)
        logger.info("QME %s", self.name)
        if subtitle:
            logger.info("%s", subtitle)
        logger.info("=" * 80)

    def print_backend_summary(self, backends: list[str], title: str = "Available Backends") -> None:
        """Print standardized backend summary."""
        logger.info("\n📋 %s", title)
        logger.info("-" * 50)
        for i, backend in enumerate(backends, 1):
            logger.info("  %d. %s", i, backend)
        logger.info("Total: %d backends", len(backends))

    def print_configuration(self, config: dict[str, Any]) -> None:
        """Print standardized configuration summary."""
        logger.info("\nConfiguration:")
        for key, value in config.items():
            logger.info("  %s: %s", key, value)

    def print_success(self, message: str = "Completed successfully!") -> None:
        """Print standardized success message."""
        elapsed = time.time() - self.start_time
        logger.info("\n✅ %s", message)
        logger.info("⏱️  Total time: %.1f seconds", elapsed)

    def print_error(self, message: str) -> None:
        """Print standardized error message."""
        logger.error("\n❌ %s", message)

    def print_warning(self, message: str) -> None:
        """Print standardized warning message."""
        logger.warning("\n⚠️  %s", message)

    def save_results(self, results: dict[str, Any], output_file: str | None = None) -> None:
        """Save results to JSON file."""
        if output_file is None:
            output_file = f"{self.name.lower().replace(' ', '_')}_results.json"

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

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
        from qme.utils.logging import setup_qme_logging

        setup_qme_logging(verbosity=verbose)


def create_standard_epilog(example_type: str) -> str:
    """Create standardized epilog for different example types."""
    epilogs = {
        "demo": """
Examples:
  # Run with all available backends
  python cli_demo.py

  # Run with specific backends
  python cli_demo.py --backends uma,aimnet2

  # Run with verbose output
  python cli_demo.py --verbose 2
        """,
        "timing": """
Examples:
  # Run with all available backends
  python timing_benchmark.py

  # Run with specific backends
  python timing_benchmark.py --backends uma,aimnet2

  # Run on GPU
  python timing_benchmark.py --device cuda --verbose 2
        """,
        "benchmark": """
Examples:
  # Run with all available backends
  python benchmark.py

  # Run with specific backends
  python benchmark.py --backends uma,aimnet2

  # Run with verbose output
  python benchmark.py --verbose 2
        """,
        "benchmark_quick": """
Examples:
  # Run with all available backends
  python benchmark.py

  # Run with specific backends
  python benchmark.py --backends uma,aimnet2

  # Quick test with subset of data
  python benchmark.py --quick --verbose 2

  # Very quick test with minimal data
  python benchmark.py --quicker --verbose 2
        """,
    }

    return epilogs.get(
        example_type,
        """
Examples:
  # Run with default settings
  python example.py

  # Run with specific backends
  python example.py --backends uma,aimnet2

  # Run with verbose output
  python example.py --verbose 2
        """,
    )


def benchmark_optimization(
    backend: str,
    optimizer: str,
    device: str | None = None,
    model_name: str | None = None,
    verbose: int = 1,
    test_ts: bool = False,
    create_structure_func: Callable[[], Any] | None = None,
    suitable_optimizers: list[str] | None = None,
) -> dict[str, Any]:
    """Common benchmark function for optimization and frequency analysis.

    This function eliminates code duplication between minima and TS optimizer benchmarks.

    Parameters
    ----------
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
    -------
    Dict[str, Any]
        Benchmark results including timings for each step

    """
    import time

    import numpy as np

    # Import QME components
    try:
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
        print_device_info(device)

    results: dict[str, Any] = {
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
            return results

        # Check if optimizer is suitable for this task
        if suitable_optimizers and optimizer.lower() not in suitable_optimizers:
            task = "TS" if test_ts else "minima"
            results["error"] = (
                f"Optimizer {optimizer} not suitable for {task} optimization. "
                f"Suitable: {', '.join(suitable_optimizers)}"
            )
            return results

        results["available"] = True

        # Create appropriate structure
        if create_structure_func is None:
            results["error"] = "create_structure_func is required"
            return results
        structure = create_structure_func()

        # Initialize QME optimizer
        init_start = time.perf_counter()

        explorer = Explorer(
            atoms=structure,
            backend=backend,
            model_name=model_name,
            device=device,
            default_charge=0,
            default_spin=1,
            local_optimizer=optimizer,
            target="ts" if test_ts else "minima",
            strategy="local",
            profile=True,  # Enable profiling for benchmark mode
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

        # Optimization using Explorer strategies
        opt_start = time.perf_counter()

        # Use Explorer's run method with appropriate strategy
        try:
            if test_ts:
                run_results = explorer.run(
                    fmax=0.005,  # Stricter criteria for TS
                    steps=1000,
                )
            else:
                run_results = explorer.run(
                    fmax=0.01,  # Standard criteria for minima
                    steps=1000,
                )
        except ValueError as e:
            if "not suitable for transition state optimization" in str(e):
                results["error"] = str(e)
                return results
            raise

        # Handle results from Explorer's run method
        # The run() method returns a dictionary with standardized results
        # Note: run_results should always be a dict from Explorer.run()
        steps_taken = run_results.get("steps_taken", 0)
        converged_raw = run_results.get("converged", False)
        # Ensure converged is always a boolean
        if isinstance(converged_raw, str):
            converged = converged_raw.lower() in ("true", "1", "yes")
        else:
            converged = bool(converged_raw)
        optimized_atoms = run_results.get("optimized_atoms", explorer.atoms_list[0])

        # Get optimization results
        if optimized_atoms is not None and hasattr(optimized_atoms, "get_potential_energy"):
            final_energy = float(optimized_atoms.get_potential_energy())
            if hasattr(optimized_atoms, "get_forces"):
                max_force = float(np.max(np.abs(optimized_atoms.get_forces())))
            else:
                max_force = float("inf")
        else:
            final_energy = 0.0
            max_force = float("inf")

        opt_time = time.perf_counter() - opt_start

        # Calculate average time per optimization step
        if isinstance(steps_taken, int | float) and steps_taken > 0:
            avg_time_per_step = opt_time / steps_taken
        else:
            avg_time_per_step = None

        # Get final forces for verification
        if optimized_atoms is not None and hasattr(optimized_atoms, "get_forces"):
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

        # Extract performance data from profiler if available
        if isinstance(run_results, dict) and "performance" in run_results:
            results["performance"] = run_results["performance"]

        if verbose and avg_time_per_step is not None:
            pass

        # Frequency analysis (mandatory)
        freq_start = time.perf_counter()

        # Use the explorer's calculate_frequencies method directly
        # This method handles the calculator attachment automatically
        # Ensure optimized_atoms is the right type for calculate_frequencies
        atoms_for_freq = (
            optimized_atoms
            if isinstance(optimized_atoms, type(None) | type(explorer.atoms_list[0]))
            else None
        )
        freq_results = explorer.calculate_frequencies(
            atoms=atoms_for_freq,  # Use optimized atoms if available
            delta=0.01,
            method="auto",
            temperature=298.15,
            save_hessian=False,  # Don't save large Hessian matrix
        )

        freq_time = time.perf_counter() - freq_start
        results["timings"]["frequency_analysis"] = freq_time

        # Enhanced validation based on task type using proper frequency analysis
        frequencies = freq_results.get("frequencies", [])
        if not isinstance(frequencies, list | np.ndarray):
            frequencies = []

        if test_ts:
            # TS optimization validation
            is_ts = freq_results.get("is_ts", False)
            ts_analysis = freq_results.get("ts_analysis", {})
            n_imaginary = (
                ts_analysis.get("n_imaginary_frequencies", 0)
                if isinstance(ts_analysis, dict)
                else 0
            )
            is_valid_result = is_ts and (n_imaginary == 1)
            result_type = "TS"

            if not is_valid_result and verbose:
                if n_imaginary == 0 or n_imaginary > 1:
                    pass

            results["frequency_results"] = {
                "n_frequencies": len(frequencies) if isinstance(frequencies, Sized) else 0,
                "frequencies": frequencies[:10] if isinstance(frequencies, Sequence) else [],
                "zero_point_energy": freq_results.get("zero_point_energy", 0.0),
                "is_transition_state": is_ts,
                "is_valid_result": is_valid_result,
                "n_imaginary_frequencies": n_imaginary,
                "method_used": freq_results.get("method_used", "unknown"),
                "result_type": result_type,
                "ts_analysis": ts_analysis,
            }
        else:
            # Minima optimization validation
            is_minimum = freq_results.get("is_minimum", False)
            minima_analysis = freq_results.get("minima_analysis", {})
            n_significant_imaginary = (
                minima_analysis.get("n_significant_imaginary_frequencies", 0)
                if isinstance(minima_analysis, dict)
                else 0
            )
            n_small_negative = (
                minima_analysis.get("n_small_negative_frequencies", 0)
                if isinstance(minima_analysis, dict)
                else 0
            )
            is_valid_result = is_minimum
            result_type = "minima"

            if not is_valid_result and verbose:
                if n_significant_imaginary > 0:
                    pass
                if n_small_negative > 0:
                    pass

            results["frequency_results"] = {
                "n_frequencies": len(frequencies) if isinstance(frequencies, Sized) else 0,
                "frequencies": frequencies[:10] if isinstance(frequencies, Sequence) else [],
                "zero_point_energy": freq_results.get("zero_point_energy", 0.0),
                "is_minimum": is_minimum,
                "is_valid_result": is_valid_result,
                "n_significant_imaginary_frequencies": n_significant_imaginary,
                "n_small_negative_frequencies": n_small_negative,
                "method_used": freq_results.get("method_used", "unknown"),
                "result_type": result_type,
                "minima_analysis": minima_analysis,
            }

            if verbose and n_small_negative > 0:
                pass

        # Calculate total time (excluding None values)
        timings = results.get("timings", {})
        if isinstance(timings, dict):
            total_time = sum(v for v in timings.values() if v is not None)
            results["timings"]["total"] = total_time

    except Exception as e:
        results["error"] = str(e)

    return results
