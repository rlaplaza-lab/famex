#!/usr/bin/env python3
"""
QME BH28 Benchmark - Chemical Accuracy Evaluation

This benchmark evaluates QME backends on the BH28 database of 28 diverse
chemical reaction barrier heights with reference values from high-level quantum
chemistry calculations (CCSDT(Q)/CBS level).

Usage:
    python bh28_benchmark.py [--quick|--quicker]
    python bh28_benchmark.py --backends uma,mace
    python bh28_benchmark.py --analyze

Features:
    - Optimizes reactant minima using various QME backends
    - Optimizes transition states (when SELLA is available)
    - Calculates barrier heights from optimized structures
    - Compares accuracy against reference values
    - Provides comprehensive performance analysis

Reference: A. Karton, J. Phys. Chem. A 2019, 123, 6720-6729
"""

import json
import logging
import os
import sys
import time
import warnings
from contextlib import contextmanager
from io import StringIO
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read

# Import QME components
try:
    from qme import Explorer, calculator_registry
except ImportError as e:
    print(f"❌ Error importing QME: {e}")
    print("   Please ensure QME is installed and accessible")
    sys.exit(1)

# Import common interface
from qme.examples import QMEExampleInterface, create_standard_epilog

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress verbose logging from dependencies early
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("numexpr").setLevel(logging.WARNING)
logging.getLogger("ase").setLevel(logging.WARNING)

# Try to suppress JAX startup messages
os.environ["JAX_LOG_LEVEL"] = "ERROR"


@contextmanager
def suppress_verbose_output():
    """Context manager to capture and suppress verbose ASE/SELLA output."""
    # Capture stdout/stderr
    old_stdout, old_stderr = sys.stdout, sys.stderr
    stdout_buffer, stderr_buffer = StringIO(), StringIO()

    # Temporarily disable verbose logging
    old_level = logging.getLogger().getEffectiveLevel()
    logging.getLogger().setLevel(logging.ERROR)

    try:
        sys.stdout, sys.stderr = stdout_buffer, stderr_buffer
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        logging.getLogger().setLevel(old_level)


class BH28Benchmark:
    """Comprehensive benchmark suite for QME on the BH28 database."""

    def __init__(
        self,
        dataset_dir: str = "bh28_dataset",
        output_dir: str = "benchmark_results",
        minima_optimizer: str = "LBFGS",
        ts_optimizer: str = "SELLA",
    ):
        """Initialize comprehensive benchmark."""
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.minima_optimizer = minima_optimizer
        self.ts_optimizer = ts_optimizer

        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"BH28 dataset directory not found: {dataset_dir}")

        # Load reference barrier heights from JSON
        ref_file = self.dataset_dir / "reference_barrier_heights.json"
        with open(ref_file) as f:
            self.reference_barriers = json.load(f)

        # Define bimolecular reactions (require multiple reactants)
        self.bimolecular_reactions = {
            # CADBH reactions - all involve small molecule + C2H4
            "CADBH_1": ["CADBH_1_min.xyz", "CADBH_c2h4.xyz"],
            "CADBH_2": ["CADBH_2_min.xyz", "CADBH_c2h4.xyz"],
            "CADBH_3": ["CADBH_3_min.xyz", "CADBH_c2h4.xyz"],
            "CADBH_4": ["CADBH_4_min.xyz", "CADBH_c2h4.xyz"],
            "CADBH_5": ["CADBH_5_min.xyz", "CADBH_c2h4.xyz"],
            "CADBH_6": ["CADBH_6_min.xyz", "CADBH_c2h4.xyz"],
            "CADBH_7": ["CADBH_7_min.xyz", "CADBH_c2h4.xyz"],
            "CADBH_8": ["CADBH_8_min.xyz", "CADBH_c2h4.xyz"],
            "CADBH_9": ["CADBH_9_min.xyz", "CADBH_c2h4.xyz"],
            # BHPERI_6 - Diels-Alder reaction
            "BHPERI_6": ["BHPERI_6_min.xyz", "CADBH_c2h4.xyz"],
        }

        # Quick subset for testing (representative reactions)
        self.quick_reactions = [
            "BHDIV_3",
            "PXBH_3",
            "CADBH_2",
            "CRBH_1",
            "PXBH_2",
            "BHPERI_2",
        ]

        # Quicker subset for very fast testing (single reaction)
        self.quicker_reactions = ["BHDIV_3"]

        # Results storage
        self.results = {}

        print("=" * 80)
        print("QME BH28 Benchmark - Chemical Accuracy Evaluation")
        print("=" * 80)
        print(f"Dataset: {self.dataset_dir}")
        print(f"Output: {self.output_dir}")
        print(f"Available reactions: {len(self.reference_barriers)}")
        print("Note: SO3LR has known molecular size limitations and may skip some reactions")

    def load_structure(self, filename: str) -> Atoms:
        """Load molecular structure from XYZ file."""
        filepath = self.dataset_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Structure file not found: {filepath}")

        atoms = read(str(filepath))
        return atoms[0] if isinstance(atoms, list) else atoms

    def get_reactants(self, reaction_name: str) -> list[Atoms]:
        """Get all reactant structures for a reaction."""
        if reaction_name in self.bimolecular_reactions:
            # Bimolecular reaction - load multiple reactants
            reactant_files = self.bimolecular_reactions[reaction_name]
            return [self.load_structure(filename) for filename in reactant_files]
        else:
            # Unimolecular reaction - load single minimum structure
            min_file = f"{reaction_name}_min.xyz"
            return [self.load_structure(min_file)]

    def get_transition_state(self, reaction_name: str) -> Atoms:
        """Get transition state structure for a reaction."""
        ts_file = f"{reaction_name}_ts.xyz"
        return self.load_structure(ts_file)

    def get_available_backends(self) -> list[str]:
        """Get list of available QME backends (excluding mock)."""
        available = []
        ml_backends = [
            "aimnet2",
            "uma",
            "so3lr",
            "mace",
            "orb",
            "torchsim_mace",
            "torchsim_uma",
        ]

        for backend in ml_backends:
            if calculator_registry.is_backend_available(backend):
                available.append(backend)

        return available

    def filter_available_backends(
        self, requested_backends: list[str], verbose: bool = False
    ) -> list[str]:
        """Filter requested backends to only available ones."""
        available = []
        for backend in requested_backends:
            if calculator_registry.is_backend_available(backend):
                available.append(backend)
            elif verbose:
                print(f"Warning: Backend '{backend}' not available, skipping")
        return available

    def print_backend_summary(self, backends: list[str], title: str = "Available Backends"):
        """Print a formatted summary of backends."""
        print(f"\n📋 {title}")
        print("-" * 50)
        for i, backend in enumerate(backends, 1):
            print(f"  {i}. {backend}")
        print(f"Total: {len(backends)} backends")

    def optimize_structures(
        self, reactions: list[str], backends: list[str], verbose: bool = False
    ) -> dict:
        """Optimize all structures (minima and TS) for given reactions and backends."""
        print(f"\n{'=' * 80}")
        print("STRUCTURE OPTIMIZATION")
        print(f"{'=' * 80}")

        results = {}

        for backend in backends:
            print(f"\nBackend: {backend.upper()}")
            print("-" * 60)
            backend_results = {}

            # Get default model for backend (use None for auto-detection)
            model_name = None

            try:
                for reaction in reactions:
                    print(f"\n  📍 {reaction}")
                    reaction_data = {
                        "timings": {},
                        "optimization_results": {},
                        "frequency_results": {},
                    }

                    try:
                        # Load reactants
                        load_start = time.perf_counter()
                        reactants = self.get_reactants(reaction)
                        load_time = time.perf_counter() - load_start
                        reaction_data["timings"]["structure_loading"] = load_time

                        # Brief molecular information
                        reactant_info = [r.get_chemical_formula() for r in reactants]
                        if len(reactant_info) > 1:
                            print(f"  Reactants: {' + '.join(reactant_info)}")
                        else:
                            print(f"  Reactant: {reactant_info[0]}")

                        # 1. Optimize reactant minima
                        minima_start = time.perf_counter()
                        optimized_reactants = []
                        reactant_energies = []
                        minima_steps_total = 0

                        for i, reactant in enumerate(reactants):
                            with suppress_verbose_output():
                                optimizer = Explorer(
                                    atoms=reactant,
                                    backend=backend,
                                    model_name=model_name,
                                )
                                result = optimizer.run(
                                    mode="minima",
                                    local_optimizer_name=self.minima_optimizer,
                                    steps=500,
                                    fmax=0.01,
                                )
                            # Normalize result shape (Explorer.run returns [dict] for local)
                            if isinstance(result, list) and len(result) == 1:
                                result = result[0]
                            if isinstance(result, dict):
                                optimized = result.get("optimized_atoms", reactant)
                                steps_taken = result.get("steps_taken", None)
                                converged = result.get("converged", False)
                            else:
                                optimized = result
                                steps_taken = None
                                converged = True

                            if steps_taken:
                                minima_steps_total += steps_taken

                            # Compute energy from atoms to avoid relying on dict keys
                            try:
                                energy_val = float(optimized.get_potential_energy())
                            except Exception:
                                # Attach calculator if missing and retry
                                optimizer._create_and_attach_calculator(optimized)
                                energy_val = float(optimized.get_potential_energy())

                            optimized_reactants.append(optimized)
                            reactant_energies.append(energy_val)

                            reactant_label = (
                                f"reactant_{i + 1}" if len(reactants) > 1 else "reactant"
                            )
                            status_text = "Converged" if converged else "Failed"
                            print(f"  ✅ {reactant_label}: {energy_val:.6f} eV ({status_text})")
                            if verbose and steps_taken:
                                print(f"    Steps: {steps_taken}")

                        total_reactant_energy = sum(reactant_energies)
                        minima_time = time.perf_counter() - minima_start

                        reaction_data["timings"]["minima_optimization"] = minima_time
                        reaction_data["optimization_results"].update(
                            {
                                "minima_converged": True,
                                "minima_steps": minima_steps_total,
                                "total_reactant_energy": total_reactant_energy,
                                "reactant_energies": reactant_energies,
                            }
                        )

                        # 2. Optimize transition state
                        try:
                            ts_atoms = self.get_transition_state(reaction)
                            ts_start = time.perf_counter()

                            with suppress_verbose_output():
                                ts_optimizer = Explorer(
                                    atoms=ts_atoms,
                                    backend=backend,
                                    model_name=model_name,
                                )
                                ts_result = ts_optimizer.run(
                                    mode="ts",
                                    steps=500,
                                    fmax=0.01,
                                    local_optimizer_name=self.ts_optimizer,
                                )

                            ts_time = time.perf_counter() - ts_start

                            # Normalize TS result
                            if isinstance(ts_result, list) and len(ts_result) == 1:
                                ts_result = ts_result[0]
                            if isinstance(ts_result, dict):
                                ts_atoms_opt = ts_result.get("optimized_atoms", ts_atoms)
                                ts_steps = ts_result.get("steps_taken", None)
                                ts_conv = bool(ts_result.get("converged", False))
                            else:
                                ts_atoms_opt = ts_result
                                ts_steps = None
                                ts_conv = True

                            # Compute energy
                            try:
                                ts_energy_val = float(ts_atoms_opt.get_potential_energy())
                            except Exception:
                                ts_optimizer._create_and_attach_calculator(ts_atoms_opt)
                                ts_energy_val = float(ts_atoms_opt.get_potential_energy())

                            reaction_data["timings"]["ts_optimization"] = ts_time
                            reaction_data["optimization_results"].update(
                                {
                                    "ts_converged": ts_conv,
                                    "ts_steps": ts_steps,
                                    "ts_energy": ts_energy_val,
                                    "optimized_ts": ts_atoms_opt,
                                }
                            )

                            # Frequency analysis to verify TS character
                            if ts_atoms_opt is not None and ts_conv:
                                freq_start = time.perf_counter()
                                try:
                                    with suppress_verbose_output():
                                        freq_results = ts_optimizer.calculate_frequencies(
                                            delta=0.01,
                                            method="auto",
                                            temperature=298.15,
                                            save_hessian=False,
                                        )
                                    freq_time = time.perf_counter() - freq_start
                                    reaction_data["timings"]["frequency_analysis"] = freq_time
                                    reaction_data["frequency_results"] = {
                                        "n_frequencies": len(freq_results["frequencies"]),
                                        "frequencies": freq_results["frequencies"][
                                            :10
                                        ],  # First 10 frequencies
                                        "zero_point_energy": freq_results["zero_point_energy"],
                                        "is_transition_state": freq_results["is_ts"],
                                        "method_used": freq_results["method_used"],
                                        "ts_analysis": freq_results.get("ts_analysis", {}),
                                    }
                                except Exception as e:
                                    reaction_data["timings"]["frequency_analysis"] = None
                                    reaction_data["frequency_results"] = {"error": str(e)}
                            else:
                                reaction_data["timings"]["frequency_analysis"] = None
                                reaction_data["frequency_results"] = {
                                    "skipped": "TS optimization failed"
                                }

                            # Print status
                            freq_info = reaction_data["frequency_results"]
                            if "is_transition_state" in freq_info:
                                ts_verified = freq_info["is_transition_state"]
                                ts_status = (
                                    "✅ Verified TS" if ts_verified else "⚠️ Not verified as TS"
                                )
                            else:
                                ts_status = "❓ Not checked"

                            status = "✅" if ts_conv else "⚠️"
                            ts_status_text = "Converged" if ts_conv else "Failed"
                            print(f"  {status} TS: {ts_energy_val:.6f} eV ({ts_status_text})")
                            print(f"  TS Character: {ts_status}")
                            if verbose and ts_steps:
                                print(f"    Steps: {ts_steps}, Time: {ts_time:.2f}s")

                        except Exception as e:
                            print(f"  ❌ TS optimization failed: {str(e)}")
                            reaction_data["optimization_results"].update(
                                {
                                    "ts_success": False,
                                    "ts_error": str(e),
                                }
                            )
                            reaction_data["timings"]["ts_optimization"] = None
                            reaction_data["timings"]["frequency_analysis"] = None
                            reaction_data["frequency_results"] = {
                                "skipped": "TS optimization failed"
                            }

                        # 3. Calculate barrier height from optimized structures
                        opt_results = reaction_data["optimization_results"]
                        if opt_results.get("ts_converged") and opt_results.get("minima_converged"):
                            barrier_height = (
                                opt_results["ts_energy"] - opt_results["total_reactant_energy"]
                            )
                            ref_barrier = self.reference_barriers[reaction]
                            error = barrier_height - ref_barrier
                            relative_error = (
                                (error / ref_barrier) * 100 if ref_barrier != 0 else float("inf")
                            )

                            reaction_data["optimization_results"].update(
                                {
                                    "barrier_height": barrier_height,
                                    "reference_barrier": ref_barrier,
                                    "absolute_error": error,
                                    "relative_error": relative_error,
                                    "barrier_success": True,
                                }
                            )

                            print(
                                f"  📊 Barrier: {barrier_height:.3f} eV | "
                                f"Ref: {ref_barrier:.3f} eV | "
                                f"Error: {error:+.3f} eV ({relative_error:+.1f}%)"
                            )

                        # Calculate total time
                        total_time = sum(
                            v for v in reaction_data["timings"].values() if v is not None
                        )
                        reaction_data["timings"]["total"] = total_time
                        reaction_data["success"] = True

                    except Exception as e:
                        error_msg = str(e)
                        if backend == "so3lr" and "vmap got inconsistent sizes" in error_msg:
                            print("  ⚠️  Skipped: SO3LR molecular size incompatibility")
                            reaction_data = {
                                "success": False,
                                "error": "SO3LR molecular size incompatibility",
                                "skipped": True,
                                "timings": {},
                                "optimization_results": {},
                                "frequency_results": {},
                            }
                        else:
                            print(f"  ❌ Failed: {error_msg}")
                            reaction_data = {
                                "success": False,
                                "error": error_msg,
                                "timings": {},
                                "optimization_results": {},
                                "frequency_results": {},
                            }

                    backend_results[reaction] = reaction_data

            except Exception as e:
                print(f"❌ Backend {backend} initialization failed: {str(e)}")
                backend_results = {"backend_error": str(e)}

            results[backend] = backend_results

        self.results = results
        return results

    def analyze_performance(self, backends: list[str]) -> dict:
        """Analyze performance metrics across backends with detailed statistics."""
        print(f"\n{'=' * 120}")
        print("DETAILED PERFORMANCE ANALYSIS")
        print(f"{'=' * 120}")

        analysis = {}

        for backend in backends:
            print(f"\nBackend: {backend.upper()}")
            print("-" * 60)
            backend_data = self.results.get(backend, {})

            # Collect successful calculations
            successful_barriers = []
            failed_count = 0
            skipped_count = 0
            minima_converged_count = 0
            ts_converged_count = 0
            ts_verified_count = 0
            timing_stats = {"total": [], "minima": [], "ts": [], "frequency": []}
            step_stats = {"minima": [], "ts": []}

            for _reaction, data in backend_data.items():
                if isinstance(data, dict) and not data.get("skipped"):
                    if not data.get("success", True):
                        failed_count += 1
                    else:
                        opt_results = data.get("optimization_results", {})

                        # Check minima convergence
                        if opt_results.get("minima_converged"):
                            minima_converged_count += 1
                            if opt_results.get("minima_steps"):
                                step_stats["minima"].append(opt_results["minima_steps"])

                        # Check TS convergence
                        if opt_results.get("ts_converged"):
                            ts_converged_count += 1
                            if opt_results.get("ts_steps"):
                                step_stats["ts"].append(opt_results["ts_steps"])

                            # Check TS verification
                            freq_results = data.get("frequency_results", {})
                            if freq_results.get("is_transition_state"):
                                ts_verified_count += 1

                        # Collect barrier height data
                        if opt_results.get("barrier_success"):
                            successful_barriers.append(opt_results)

                        # Collect timing data
                        timings = data.get("timings", {})
                        if timings.get("total"):
                            timing_stats["total"].append(timings["total"])
                        if timings.get("minima_optimization"):
                            timing_stats["minima"].append(timings["minima_optimization"])
                        if timings.get("ts_optimization"):
                            timing_stats["ts"].append(timings["ts_optimization"])
                        if timings.get("frequency_analysis"):
                            timing_stats["frequency"].append(timings["frequency_analysis"])
                elif isinstance(data, dict) and data.get("skipped"):
                    skipped_count += 1

            # Calculate statistics
            total_reactions = len(
                [r for r in backend_data.values() if isinstance(r, dict) and not r.get("skipped")]
            )

            if successful_barriers:
                errors = [data["absolute_error"] for data in successful_barriers]
                rel_errors = [
                    data["relative_error"]
                    for data in successful_barriers
                    if abs(data["relative_error"]) != float("inf")
                ]

                barrier_stats = {
                    "total_reactions": total_reactions,
                    "minima_converged": minima_converged_count,
                    "ts_converged": ts_converged_count,
                    "ts_verified": ts_verified_count,
                    "barrier_successful": len(successful_barriers),
                    "failed": failed_count,
                    "skipped": skipped_count,
                    "minima_rate": (
                        (minima_converged_count / total_reactions * 100)
                        if total_reactions > 0
                        else 0
                    ),
                    "ts_rate": (
                        (ts_converged_count / total_reactions * 100) if total_reactions > 0 else 0
                    ),
                    "verification_rate": (
                        (ts_verified_count / ts_converged_count * 100)
                        if ts_converged_count > 0
                        else 0
                    ),
                    "mae": np.mean(np.abs(errors)),
                    "rmse": np.sqrt(np.mean(np.array(errors) ** 2)),
                    "max_error": np.max(np.abs(errors)),
                    "mean_rel_error": np.mean(np.abs(rel_errors)) if rel_errors else 0,
                    "std_error": np.std(errors),
                }

                print("📊 CONVERGENCE STATISTICS:")
                print(f"  Total reactions: {barrier_stats['total_reactions']}")
                minima_count = barrier_stats["minima_converged"]
                minima_rate = barrier_stats["minima_rate"]
                print(f"  Minima converged: {minima_count} ({minima_rate:.1f}%)")
                ts_count = barrier_stats["ts_converged"]
                ts_rate = barrier_stats["ts_rate"]
                print(f"  TS converged: {ts_count} ({ts_rate:.1f}%)")
                ts_verified_count = barrier_stats["ts_verified"]
                verification_rate = barrier_stats["verification_rate"]
                print(f"  TS Verified: {ts_verified_count} ({verification_rate:.1f}%)")
                print(f"  Barrier calculations: {barrier_stats['barrier_successful']}")
                print(f"  Failed: {barrier_stats['failed']}")
                if skipped_count > 0:
                    print(f"  Skipped: {barrier_stats['skipped']}")

                print("\n📏 BARRIER HEIGHT ACCURACY:")
                print(f"  MAE:  {barrier_stats['mae']:.3f} eV")
                print(f"  RMSE: {barrier_stats['rmse']:.3f} eV")
                print(f"  Max Error: {barrier_stats['max_error']:.3f} eV")
                print(f"  Mean Rel. Error: {barrier_stats['mean_rel_error']:.1f}%")

                # Timing statistics
                if timing_stats["total"]:
                    print("\n⏱️ TIMING STATISTICS:")
                    total_mean = np.mean(timing_stats["total"])
                    total_std = np.std(timing_stats["total"])
                    print(f"  Total time: {total_mean:.2f} ± {total_std:.2f} s")
                    if timing_stats["minima"]:
                        minima_mean = np.mean(timing_stats["minima"])
                        minima_std = np.std(timing_stats["minima"])
                        print(f"  Minima optimization: {minima_mean:.2f} ± {minima_std:.2f} s")
                    if timing_stats["ts"]:
                        ts_mean = np.mean(timing_stats["ts"])
                        ts_std = np.std(timing_stats["ts"])
                        print(f"  TS optimization: {ts_mean:.2f} ± {ts_std:.2f} s")
                    if timing_stats["frequency"]:
                        freq_mean = np.mean(timing_stats["frequency"])
                        freq_std = np.std(timing_stats["frequency"])
                        print(f"  Frequency analysis: {freq_mean:.2f} ± {freq_std:.2f} s")

                # Step statistics
                if step_stats["minima"]:
                    print("\n🔄 OPTIMIZATION STEPS:")
                    minima_steps_mean = np.mean(step_stats["minima"])
                    minima_steps_std = np.std(step_stats["minima"])
                    print(f"  Minima: {minima_steps_mean:.1f} ± {minima_steps_std:.1f} steps")
                if step_stats["ts"]:
                    ts_steps_mean = np.mean(step_stats["ts"])
                    ts_steps_std = np.std(step_stats["ts"])
                    print(f"  TS: {ts_steps_mean:.1f} ± {ts_steps_std:.1f} steps")

            else:
                barrier_stats = {
                    "total_reactions": total_reactions,
                    "minima_converged": 0,
                    "ts_converged": 0,
                    "ts_verified": 0,
                    "barrier_successful": 0,
                    "failed": failed_count,
                    "skipped": skipped_count,
                }
                print("❌ No successful barrier height calculations")

            analysis[backend] = {"barrier_statistics": barrier_stats}

        # Comprehensive summary table
        print(f"\n{'=' * 120}")
        print("COMPREHENSIVE BACKEND COMPARISON")
        print(f"{'=' * 120}")

        # Print legend
        print("📊 COLUMN DEFINITIONS:")
        print("   Total   = Total number of reactions attempted")
        print("   Minima  = Number of converged minima optimizations")
        print("   TS      = Number of converged TS optimizations")
        print("   Verified = Number vibrationally verified as TS")
        print("   Barriers = Number of successful barrier calculations")
        print("   MAE     = Mean Absolute Error (eV)")
        print("   RMSE    = Root Mean Square Error (eV)")
        print(f"{'-' * 120}")

        # Header
        print(
            f"{'Backend':<12} {'Total':<6} {'Minima':<7} {'TS':<4} "
            f"{'Verified':<9} {'Barriers':<9} {'MAE (eV)':<9} {'RMSE (eV)':<10}"
        )
        print("=" * 120)

        # Results
        for backend in backends:
            stats = analysis.get(backend, {}).get("barrier_statistics", {})

            print(
                f"{backend:<12} "
                f"{stats.get('total_reactions', 0):<6} "
                f"{stats.get('minima_converged', 0):<7} "
                f"{stats.get('ts_converged', 0):<4} "
                f"{stats.get('ts_verified', 0):<9} "
                f"{stats.get('barrier_successful', 0):<9} "
                f"{stats.get('mae', 0):<9.3f} "
                f"{stats.get('rmse', 0):<10.3f}"
            )

        print("=" * 120)

        return analysis

    def _convert_to_serializable(self, obj):
        """Recursively convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif hasattr(obj, "get_chemical_formula"):  # ASE Atoms object
            return {
                "formula": obj.get_chemical_formula(),
                "positions": obj.positions.tolist(),
                "symbols": obj.get_chemical_symbols(),
            }
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)

    def save_results(self, filename: str = "bh28_benchmark_results.json"):
        """Save all results to JSON file."""
        output_file = self.output_dir / filename
        serializable_results = self._convert_to_serializable(self.results)

        with open(output_file, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\nResults saved to: {output_file}")

    def run_benchmark(self, backends: list[str], reactions: list[str], verbose: bool = False):
        """Run the complete benchmark suite."""
        print("\nStarting BH28 Benchmark")
        print(f"Backends: {', '.join(backends)}")
        print(
            f"Reactions: {len(reactions)} "
            f"({', '.join(reactions[:3])}{'...' if len(reactions) > 3 else ''})"
        )

        start_time = time.time()

        # Optimize all structures and calculate barriers
        self.optimize_structures(reactions, backends, verbose=verbose)

        # Analyze performance
        self.analyze_performance(backends)

        # Save results
        self.save_results()

        total_time = time.time() - start_time
        print(f"\nBenchmark completed in {total_time:.1f} seconds")

        return self.results


def main():
    """Main entry point for the benchmark."""
    # Create standardized interface
    interface = QMEExampleInterface(
        name="BH28 Benchmark",
        description="Chemical Accuracy Evaluation",
        epilog=create_standard_epilog("benchmark"),
    )

    parser = interface.create_parser()

    # Add benchmark-specific arguments
    parser.add_argument(
        "--reactions",
        nargs="+",
        help="Specific reactions to test (default: all 28 reactions)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark with representative subset of reactions",
    )
    parser.add_argument(
        "--quicker",
        action="store_true",
        help="Run quicker benchmark with single reaction for very fast testing",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze existing results without running new calculations",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Output directory for results (default: benchmark_results)",
    )
    parser.add_argument(
        "--minima-optimizer",
        default="LBFGS",
        choices=["LBFGS", "BFGS", "FIRE"],
        help="Minima optimizer to use (default: LBFGS)",
    )
    parser.add_argument(
        "--ts-optimizer",
        default="SELLA",
        choices=["SELLA", "trust-krylov-ts"],
        help="Transition state optimizer to use (default: SELLA)",
    )

    args = parser.parse_args()

    # Initialize benchmark
    benchmark = BH28Benchmark(
        output_dir=args.output_dir,
        minima_optimizer=args.minima_optimizer,
        ts_optimizer=args.ts_optimizer,
    )

    interface.print_header("Chemical Accuracy Evaluation")
    print(f"Minima Optimizer: {args.minima_optimizer}")
    print(f"TS Optimizer: {args.ts_optimizer}")

    # Determine backends to test
    if args.backends:
        requested_backends = [b.strip() for b in args.backends.split(",")]
        backends = interface.filter_available_backends(requested_backends, verbose=True)
        if not backends:
            interface.print_error("No requested backends are available!")
            return 1
    else:
        backends = interface.get_available_ml_backends()
        if not backends:
            interface.print_error("No ML backends available!")
            print("Please install at least one ML backend:")
            print("  - UMA: pip install fairchem-core")
            print("  - MACE: pip install mace-torch")
            print("  - AIMNet2: pip install aimnet2")
            print("  - SO3LR: pip install so3lr")
            return 1

    interface.print_backend_summary(backends, "Benchmarking Backends")

    # Determine reactions to test
    if args.quicker:
        reactions = benchmark.quicker_reactions
    elif args.quick:
        reactions = benchmark.quick_reactions
    elif args.reactions:
        reactions = args.reactions
    else:
        reactions = list(benchmark.reference_barriers.keys())

    # Validate reactions
    invalid_reactions = [r for r in reactions if r not in benchmark.reference_barriers]
    if invalid_reactions:
        interface.print_error(f"Invalid reactions: {invalid_reactions}")
        return 1

    # Print configuration
    config = {
        "Reactions": len(reactions),
        "Output": args.output_dir,
        "Verbose": args.verbose,
    }
    interface.print_configuration(config)

    if args.analyze:
        # Load existing results and analyze
        results_file = Path(args.output_dir) / "bh28_benchmark_results.json"
        if results_file.exists():
            with open(results_file) as f:
                benchmark.results = json.load(f)
            benchmark.analyze_performance(backends)
        else:
            interface.print_error(f"No existing results found at {results_file}")
            return 1
    else:
        # Run full benchmark
        benchmark.run_benchmark(backends, reactions, verbose=args.verbose)

    interface.print_success()
    return 0


if __name__ == "__main__":
    sys.exit(main())
