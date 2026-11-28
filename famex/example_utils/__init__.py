#!/usr/bin/env python3
"""Common interface utilities for FAMEX examples.

This module provides standardized interfaces, output formatting, and common
functionality across all FAMEX examples to ensure consistency.
"""

import argparse
import contextlib
import functools
import gc
import json
import os
import time
from collections.abc import Callable, Generator, Sequence, Sized
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import write as ase_write

from famex.backends.availability import (
    filter_available_backends as filter_available_backends_global,
)
from famex.backends.availability import (
    is_backend_available,
)
from famex.backends.cache import get_unified_cache
from famex.backends.constants import DEFAULT_UMA_MODEL
from famex.backends.registry import calculator_registry, create_calculator
from famex.core.explorer import Explorer
from famex.utils.device import get_optimal_device, print_device_info
from famex.utils.logging import get_famex_logger
from famex.utils.path_security import PathSecurityError, sanitize_filename

logger = get_famex_logger(__name__)


def release_cuda_memory() -> None:
    """Drop cached calculators and return free blocks to the CUDA allocator."""
    get_unified_cache().calculator_cache.clear()
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()
    torch.cuda.empty_cache()


class FAMEXExampleInterface:
    """Base class for standardized FAMEX examples."""

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
        from famex.backends.availability import get_available_ml_backends

        return get_available_ml_backends()

    def filter_available_backends(
        self,
        requested_backends: list[str],
        verbose: int = 0,
    ) -> list[str]:
        """Filter requested backends to only available ones."""
        return filter_available_backends_global(
            requested_backends,
            verbose=bool(verbose >= 1),
        )

    def print_header(self, subtitle: str = "") -> None:
        """Print standardized header."""
        logger.info("=" * 80)
        logger.info("FAMEX %s", self.name)
        if subtitle:
            logger.info("%s", subtitle)
        logger.info("=" * 80)

    def print_backend_summary(self, backends: list[str], title: str = "Available Backends") -> None:
        """Print standardized backend summary."""
        logger.info("\n%s", title)
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
        logger.info("\n%s", message)
        logger.info("Total time: %.1f seconds", elapsed)

    def print_error(self, message: str) -> None:
        """Print standardized error message."""
        logger.error("\n%s", message)

    def print_warning(self, message: str) -> None:
        """Print standardized warning message."""
        logger.warning("\n%s", message)

    def save_results(
        self,
        results: dict[str, Any] | list[dict[str, Any]],
        output_file: str | None = None,
    ) -> None:
        """Save results to JSON file.

        Parameters
        ----------
        results : dict[str, Any] | list[dict[str, Any]]
            Results to save (can be a dict or list of dicts)
        output_file : str | None
            Output file path (default: auto-generated based on example name)
        """
        if output_file is None:
            output_file = self.get_default_output_file()

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    def get_default_output_file(self) -> str:
        """Get default output file name for this example."""
        return f"{self.name.lower().replace(' ', '_')}_results.json"

    def get_device_info(self, device: str | None = None) -> str:
        """Get optimal device and print info."""
        device = get_optimal_device(device)
        print_device_info(device)
        return device

    def setup_logging(self, verbose: int = 1) -> None:
        """Set up FAMEX logging based on verbosity level."""
        from famex.utils.logging import setup_famex_logging

        setup_famex_logging(verbosity=verbose)

    def select_backend(
        self,
        requested_backends: list[str] | None,
        preferred_backends: list[str] | None = None,
        required_backends: list[str] | None = None,
        verbose: int = 1,
    ) -> tuple[str | None, list[str]]:
        """Select a backend with fallback logic.

        Convenience method that wraps select_backend_with_fallback.

        Parameters
        ----------
        requested_backends : list[str] | None
            Backends requested by user (from --backends argument)
        preferred_backends : list[str] | None
            Preferred backends in order of preference (default: None, auto-select)
        required_backends : list[str] | None
            Backends that must be available (default: None)
        verbose : int
            Verbosity level

        Returns
        -------
        tuple[str | None, list[str]]
            Selected backend (or None if none available) and list of all available backends
        """
        return select_backend_with_fallback(
            self, requested_backends, preferred_backends, required_backends, verbose
        )


def _benchmark_initial_results(
    backend: str,
    optimizer: str,
    device: str,
    model_name: str | None,
    test_ts: bool,
) -> dict[str, Any]:
    return {
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


def _benchmark_prepare_ts_initial_hessian(
    *,
    structure: Atoms,
    backend: str,
    model_name: str | None,
    device: str,
    force_finite_diff_hessian: bool,
    verbose: int,
) -> Any | None:
    try:
        from famex.analysis.frequency import FrequencyAnalysis

        charge = getattr(structure, "charge", None)
        if charge is None and hasattr(structure, "info") and "charge" in structure.info:
            charge = structure.info["charge"]
        charge = charge if charge is not None else 0

        multiplicity = getattr(structure, "spin", None)
        if multiplicity is None and hasattr(structure, "info") and "spin" in structure.info:
            multiplicity = structure.info["spin"]
        multiplicity = multiplicity if multiplicity is not None else 1

        temp_calc = create_calculator(
            backend=backend,
            model_name=model_name,
            model_path=None,
            device=device,
            default_charge=charge,
            default_spin=multiplicity,
            charge=charge,
            mult=multiplicity,
        )
        structure.calc = temp_calc

        freq_analysis = FrequencyAnalysis(atoms=structure, calculator=temp_calc, verbose=0)
        hessian_method = "finite_differences" if force_finite_diff_hessian else "auto"
        initial_hessian = freq_analysis.calculate_hessian(method=hessian_method)
        structure.calc = None
        return initial_hessian
    except Exception as e:
        if verbose >= 1:
            logger.warning("Failed to compute initial Hessian: %s. Continuing without it.", e)
        return None


def _benchmark_record_single_point_timings(explorer: Explorer, results: dict[str, Any]) -> None:
    atoms = explorer.atoms_list[0]

    energy_first_start = time.perf_counter()
    atoms.get_potential_energy()
    energy_first_time = time.perf_counter() - energy_first_start
    results["timings"]["single_energy_first"] = energy_first_time

    energy_second_start = time.perf_counter()
    atoms.get_potential_energy()
    energy_second_time = time.perf_counter() - energy_second_start
    results["timings"]["single_energy_second"] = energy_second_time
    results["timings"]["single_energy"] = energy_second_time

    force_start = time.perf_counter()
    forces = atoms.get_forces()
    results["timings"]["single_forces"] = time.perf_counter() - force_start
    results["single_point_max_force"] = float(np.max(np.abs(forces)))


def _benchmark_collect_optimization_metrics(
    run_results: dict[str, Any],
    explorer: Explorer,
    opt_time: float,
) -> tuple[Any, bool, Any, float, float, np.ndarray | None, float | None]:
    steps_taken = run_results.get("steps_taken", 0)
    converged_raw = run_results.get("converged", False)
    if isinstance(converged_raw, str):
        converged = converged_raw.lower() in ("true", "1", "yes")
    else:
        converged = bool(converged_raw)
    optimized_atoms = run_results.get("optimized_atoms", explorer.atoms_list[0])

    final_forces: np.ndarray | None = None
    if optimized_atoms is not None and hasattr(optimized_atoms, "get_potential_energy"):
        final_energy = float(optimized_atoms.get_potential_energy())
        if hasattr(optimized_atoms, "get_forces"):
            final_forces = np.asarray(optimized_atoms.get_forces())
            max_force = float(np.max(np.abs(final_forces)))
        else:
            max_force = float("inf")
    else:
        final_energy = 0.0
        max_force = float("inf")

    avg_time_per_step = (
        (opt_time / steps_taken)
        if isinstance(steps_taken, int | float) and steps_taken > 0
        else None
    )
    return (
        steps_taken,
        converged,
        optimized_atoms,
        final_energy,
        max_force,
        final_forces,
        avg_time_per_step,
    )


def _benchmark_maybe_write_structure(
    *,
    save_optimized_structure: bool,
    optimized_atoms: Any,
    structure_output_dir: str | Path | None,
    structure_label: str | None,
    optimizer: str,
    backend: str,
    test_ts: bool,
    verbose: int,
) -> str | None:
    if not save_optimized_structure or optimized_atoms is None:
        return None

    output_dir = Path(structure_output_dir) if structure_output_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_label = structure_label if structure_label else optimizer
    if not raw_label:
        raw_label = "optimizer"
    try:
        sanitized = sanitize_filename(raw_label)
    except PathSecurityError:
        sanitized = "optimizer"
    try:
        backend_key = sanitize_filename(backend)
    except PathSecurityError:
        backend_key = "backend"
    target_tag = "ts" if test_ts else "min"
    filename = f"{backend_key}_{sanitized}_{target_tag}.xyz"
    xyz_path = output_dir / filename

    try:
        if isinstance(optimized_atoms, Atoms):
            ase_write(xyz_path, optimized_atoms)
            return str(xyz_path)
        if (
            isinstance(optimized_atoms, list)
            and optimized_atoms
            and isinstance(optimized_atoms[0], Atoms)
            and all(isinstance(a, Atoms) for a in optimized_atoms)
        ):
            ase_write(xyz_path, optimized_atoms)
            return str(xyz_path)
        if verbose >= 1:
            logger.warning(
                "Cannot write optimized_atoms: invalid type %s",
                type(optimized_atoms),
            )
    except Exception as exc:
        if verbose >= 1:
            logger.warning("Failed to write optimized structure to %s: %s", xyz_path, exc)
    return None


def _benchmark_build_frequency_results(
    *,
    test_ts: bool,
    calculate_frequencies: bool,
    freq_results: dict[str, Any],
) -> dict[str, Any]:
    frequencies = freq_results.get("frequencies", []) if freq_results else []
    if not isinstance(frequencies, list | np.ndarray):
        frequencies = []

    if test_ts:
        if calculate_frequencies and freq_results:
            is_ts = freq_results.get("is_ts", False)
            ts_analysis = freq_results.get("ts_analysis", {})
            n_imaginary = (
                ts_analysis.get("n_imaginary_frequencies", 0)
                if isinstance(ts_analysis, dict)
                else 0
            )
            is_valid_result = is_ts and (n_imaginary == 1)
        else:
            is_ts = False
            ts_analysis = {}
            n_imaginary = 0
            is_valid_result = False

        all_frequencies_from_freq = freq_results.get("all_frequencies", []) if freq_results else []
        return {
            "n_frequencies": (len(frequencies) if isinstance(frequencies, Sized) else 0),
            "frequencies": (frequencies[:10] if isinstance(frequencies, Sequence) else []),
            "all_frequencies": (
                all_frequencies_from_freq[:20]
                if isinstance(all_frequencies_from_freq, Sequence)
                else []
            ),
            "zero_point_energy": (
                freq_results.get("zero_point_energy", 0.0) if freq_results else 0.0
            ),
            "is_transition_state": is_ts,
            "is_valid_result": is_valid_result,
            "n_imaginary_frequencies": n_imaginary,
            "method_used": (
                freq_results.get("method_used", "unknown") if freq_results else "not_calculated"
            ),
            "result_type": "TS",
            "ts_analysis": ts_analysis,
        }

    if calculate_frequencies and freq_results:
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
    else:
        is_minimum = False
        minima_analysis = {}
        n_significant_imaginary = 0
        n_small_negative = 0
        is_valid_result = False

    return {
        "n_frequencies": (len(frequencies) if isinstance(frequencies, Sized) else 0),
        "frequencies": (frequencies[:10] if isinstance(frequencies, Sequence) else []),
        "zero_point_energy": (freq_results.get("zero_point_energy", 0.0) if freq_results else 0.0),
        "is_minimum": is_minimum,
        "is_valid_result": is_valid_result,
        "n_significant_imaginary_frequencies": n_significant_imaginary,
        "n_small_negative_frequencies": n_small_negative,
        "method_used": (
            freq_results.get("method_used", "unknown") if freq_results else "not_calculated"
        ),
        "result_type": "minima",
        "minima_analysis": minima_analysis,
    }


def create_standard_epilog(script: str, *, quick: bool = False) -> str:
    """Examples block for a benchmark script's ``--help``."""
    lines = [
        "Examples:",
        f"  python {script}",
        f"  python {script} --backends uma,aimnet2",
    ]
    if quick:
        lines.append(f"  python {script} --quick")
    return "\n".join(lines) + "\n"


def benchmark_optimization(
    backend: str,
    optimizer: str,
    device: str | None = None,
    model_name: str | None = None,
    verbose: int = 1,
    test_ts: bool = False,
    create_structure_func: Callable[[], Any] | None = None,
    suitable_optimizers: list[str] | None = None,
    calculate_frequencies: bool = True,
    ts_kwargs: dict[str, Any] | None = None,
    force_finite_diff_hessian: bool = False,
    save_optimized_structure: bool = False,
    structure_label: str | None = None,
    structure_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run common benchmark for optimization and frequency analysis.

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
    calculate_frequencies : bool
        Whether to perform frequency analysis (default: True).
        Frequency analysis is recommended for TS validation.
    ts_kwargs : dict[str, Any] | None
        Optional keyword arguments to pass to TS optimizers via Explorer's ts_kwargs.
        For example: {"hessian_update_freq": 5} to set Hessian update frequency.
    force_finite_diff_hessian : bool
        If True, forces use of finite difference Hessians for TS optimizers and frequency
        calculations instead of analytical Hessians (default: False).
    save_optimized_structure : bool
        If True, writes the optimized structure(s) to an XYZ file (default: False).
    structure_label : str | None
        Optional label used to build the XYZ filename. Falls back to optimizer name.
    structure_output_dir : str | Path | None
        Directory where the XYZ file should be written (default: current working directory).

    Returns
    -------
    Dict[str, Any]
        Benchmark results including timings for each step

    """
    device = get_optimal_device(device)

    if verbose:
        print_device_info(device)

    results = _benchmark_initial_results(backend, optimizer, device, model_name, test_ts)

    try:
        if not calculator_registry.is_backend_available(backend):
            results["error"] = f"Backend '{backend}' is not available (dependencies missing)"
            return results

        if suitable_optimizers and optimizer.lower() not in suitable_optimizers:
            task = "TS" if test_ts else "minima"
            results["error"] = (
                f"Optimizer {optimizer} not suitable for {task} optimization. "
                f"Suitable: {', '.join(suitable_optimizers)}"
            )
            return results

        results["available"] = True

        if create_structure_func is None:
            results["error"] = "create_structure_func is required"
            return results
        structure = create_structure_func()

        initial_hessian = None
        if test_ts:
            initial_hessian = _benchmark_prepare_ts_initial_hessian(
                structure=structure,
                backend=backend,
                model_name=model_name,
                device=device,
                force_finite_diff_hessian=force_finite_diff_hessian,
                verbose=verbose,
            )

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
            verbose=verbose,
            ts_kwargs=ts_kwargs if test_ts else None,
            force_finite_diff_hessian=force_finite_diff_hessian if test_ts else False,
            initial_hessian=initial_hessian,
        )

        init_time = time.perf_counter() - init_start
        results["timings"]["initialization"] = init_time

        load_start = time.perf_counter()

        explorer._create_and_attach_calculator(explorer.atoms_list[0])

        load_time = time.perf_counter() - load_start
        results["timings"]["structure_loading"] = load_time

        _benchmark_record_single_point_timings(explorer, results)

        opt_start = time.perf_counter()

        try:
            if test_ts:
                run_results = explorer.run(
                    fmax=0.005,  # Stricter criteria for TS
                    steps=100,  # Increased limit for TS benchmarking to allow full convergence
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
            raise e from e

        opt_time = time.perf_counter() - opt_start
        (
            steps_taken,
            converged,
            optimized_atoms,
            final_energy,
            _max_force,
            final_forces,
            avg_time_per_step,
        ) = _benchmark_collect_optimization_metrics(run_results, explorer, opt_time)
        final_max_force = (
            float(np.max(np.abs(final_forces))) if final_forces is not None else _max_force
        )

        results["timings"]["optimization"] = opt_time
        results["timings"]["avg_time_per_step"] = avg_time_per_step
        results["optimization_results"] = {
            "converged": converged,
            "final_energy": final_energy,
            "max_force": final_max_force,
            "steps_taken": steps_taken,
        }

        saved_xyz_path = _benchmark_maybe_write_structure(
            save_optimized_structure=save_optimized_structure,
            optimized_atoms=optimized_atoms,
            structure_output_dir=structure_output_dir,
            structure_label=structure_label,
            optimizer=optimizer,
            backend=backend,
            test_ts=test_ts,
            verbose=verbose,
        )
        if saved_xyz_path is not None:
            results["optimization_results"]["structure_file"] = saved_xyz_path

        if isinstance(run_results, dict) and "performance" in run_results:
            results["performance"] = run_results["performance"]

        freq_start = time.perf_counter()
        freq_results = {}

        if calculate_frequencies:
            atoms_for_freq = (
                optimized_atoms
                if isinstance(optimized_atoms, type(None) | type(explorer.atoms_list[0]))
                else None
            )
            try:
                freq_results = explorer.calculate_frequencies(
                    atoms=atoms_for_freq,  # Use optimized atoms if available
                    delta=0.01,
                    method="auto",
                    temperature=298.15,
                    save_hessian=False,  # Don't save large Hessian matrix
                )
            except Exception as e:
                if verbose >= 1:
                    logger.warning("Frequency analysis failed: %s", e)
                freq_results = {}

        freq_time = time.perf_counter() - freq_start
        results["timings"]["frequency_analysis"] = freq_time if calculate_frequencies else 0.0

        results["frequency_results"] = _benchmark_build_frequency_results(
            test_ts=test_ts,
            calculate_frequencies=calculate_frequencies,
            freq_results=freq_results,
        )

        timings = results.get("timings", {})
        if isinstance(timings, dict):
            total_time = sum(v for v in timings.values() if v is not None)
            results["timings"]["total"] = total_time

    except Exception as e:
        logger.exception(
            "benchmark_optimization failed for backend=%s optimizer=%s",
            backend,
            optimizer,
        )
        results["error"] = str(e)

    return results


def create_water_molecule() -> Atoms:
    """Create a standard water molecule for testing."""
    return Atoms(
        symbols="OHH",
        positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0],
        ],
    )


def create_methane_molecule() -> Atoms:
    """Create a standard methane molecule for testing."""
    return Atoms(
        symbols="CHHHH",
        positions=[
            [0.0, 0.0, 0.0],
            [1.09, 0.0, 0.0],
            [-0.36, 1.03, 0.0],
            [-0.36, -0.51, 0.89],
            [-0.36, -0.51, -0.89],
        ],
    )


def setup_example_environment(func: Callable | None = None) -> Callable:
    """Decorator/context manager to set up environment for examples.

    Disables GUI popups and sets up headless operation for matplotlib/ASE.
    Can be used as a decorator or context manager.

    Parameters
    ----------
    func : callable, optional
        Function to decorate. If None, returns a context manager.

    Examples
    --------
    As decorator:
        @setup_example_environment
        def main():
            pass

    As context manager:
        with setup_example_environment():
            pass
    """
    original_display = os.environ.get("DISPLAY")
    original_mplbackend = os.environ.get("MPLBACKEND")
    original_qt = os.environ.get("QT_QPA_PLATFORM")

    def _setup() -> None:
        """Set environment variables for headless operation."""
        os.environ["DISPLAY"] = ""
        os.environ["MPLBACKEND"] = "Agg"
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    def _restore() -> None:
        """Restore original environment variables."""
        if original_display is not None:
            os.environ["DISPLAY"] = original_display
        elif "DISPLAY" in os.environ:
            del os.environ["DISPLAY"]

        if original_mplbackend is not None:
            os.environ["MPLBACKEND"] = original_mplbackend
        elif "MPLBACKEND" in os.environ:
            del os.environ["MPLBACKEND"]

        if original_qt is not None:
            os.environ["QT_QPA_PLATFORM"] = original_qt
        elif "QT_QPA_PLATFORM" in os.environ:
            del os.environ["QT_QPA_PLATFORM"]

    @contextlib.contextmanager
    def _context_manager() -> Generator[None, None, None]:
        """Context manager for environment setup."""
        _setup()
        try:
            yield
        finally:
            _restore()

    if func is None:
        return _context_manager()

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        _setup()
        try:
            return func(*args, **kwargs)
        finally:
            _restore()

    return wrapper


def select_backend_with_fallback(
    interface: FAMEXExampleInterface,
    requested_backends: list[str] | None,
    preferred_backends: list[str] | None = None,
    required_backends: list[str] | None = None,
    verbose: int = 1,
) -> tuple[str | None, list[str]]:
    """Select a backend with fallback logic.

    Parameters
    ----------
    interface : FAMEXExampleInterface
        Example interface instance
    requested_backends : list[str] | None
        Backends requested by user (from --backends argument)
    preferred_backends : list[str] | None
        Preferred backends in order of preference (default: None, auto-select)
    required_backends : list[str] | None
        Backends that must be available (default: None)
    verbose : int
        Verbosity level

    Returns
    -------
    tuple[str | None, list[str]]
        Selected backend (or None if none available) and list of all available backends
    """
    available_backends = interface.get_available_ml_backends()

    if requested_backends:
        filtered = interface.filter_available_backends(requested_backends, verbose=verbose)
        if required_backends:
            filtered_required = [b for b in filtered if b in required_backends]
            if not filtered_required:
                return None, available_backends
        if filtered:
            return filtered[0], filtered

    if preferred_backends:
        for backend in preferred_backends:
            if backend in available_backends:
                return backend, available_backends

    if available_backends:
        return available_backends[0], available_backends

    return None, available_backends


def get_calculator_for_backend(
    backend: str,
    device: str | None = None,
    model_name: str | None = None,
) -> Any:
    """Create a calculator for the specified backend.

    Parameters
    ----------
    backend : str
        Backend name (e.g., 'mace', 'uma', 'aimnet2')
    device : str | None
        Device to use ('cpu' or 'cuda'). Auto-detected if None.
    model_name : str | None
        Specific model name to use (backend-specific)

    Returns
    -------
    Calculator
        Configured calculator instance

    Raises
    ------
    RuntimeError
        If backend is not available or calculator creation fails
    """
    import famex

    if not is_backend_available(backend):
        raise RuntimeError(f"Backend '{backend}' is not available")

    device = get_optimal_device(device)

    if backend == "mace":
        calc = famex.get_mace_calculator(device=device)
    elif backend == "uma":
        calc = famex.get_uma_calculator(model_name=model_name or DEFAULT_UMA_MODEL, device=device)
    elif backend == "aimnet2":
        calc = famex.get_aimnet2_calculator(device=device)
    elif backend == "so3lr":
        calc = famex.get_so3lr_calculator(device=device)
    else:
        calc = create_calculator(
            backend=backend,
            model_name=model_name,
            model_path=None,
            device=device,
            default_charge=0,
            default_spin=1,
        )

    return calc
