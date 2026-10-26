#!/usr/bin/env python3
"""FAMEX TS Optimizer Benchmark - unified local and two-ended TS comparison.

Trust-radius tuning (RFO and SciPy TS optimizers):
  trust_radius       Initial step bound in Å (RFO default 0.02; trust-krylov 0.05; others 0.1)
  max_trust_radius   Upper cap in Å (RFO default 0.06; trust-krylov 0.15; others 0.3)
  min_trust_radius   Lower cap in Å (default 0.001)
  alpha              RFO metric scaling (default 1.0)
Pass via Explorer ts_kwargs or --ts-kw trust_radius=0.05 max_trust_radius=0.15.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read

from famex.example_utils import (
    FAMEXExampleInterface,
    benchmark_optimization,
    create_standard_epilog,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

BH28_DATASET_DIR = Path(__file__).parent / "bh28_benchmark" / "bh28_dataset"
Z93_DATASET_DIR = Path(__file__).parent / "zimmermann93_benchmark" / "zimmermann93_dataset"

BH28_TS_SUBSET = [
    "BHDIV_3",
    "PXBH_3",
    "CADBH_2",
    "CRBH_1",
    "PXBH_2",
    "BHPERI_2",
]

LOCAL_OPTIMIZERS_DEFAULT = [
    "sella",
    "sella-analytical",
    "rfo",
    "trust-krylov",
    "trust-ncg",
    "trust-exact",
]

TWO_ENDED_STRATEGIES = ["interpolate", "cineb", "growing_string"]

TRUST_RADIUS_PRESETS: list[tuple[float, float]] = [
    (0.02, 0.06),
    (0.05, 0.15),
    (0.1, 0.3),
]

TRUST_SWEEP_OPTIMIZERS = ["rfo", "trust-exact", "trust-krylov"]

PARAM_SWEEP_HESSIAN_FREQS: list[int | None] = [None, 1, 5]
PARAM_SWEEP_DENSE_OPTIMIZERS = frozenset({"rfo", "trust-exact"})
PARAM_SWEEP_OPTIMIZERS = ["rfo", "trust-exact", "trust-krylov", "trust-ncg"]
PARAM_SWEEP_BASELINE_OPTIMIZERS = ["sella", "sella-analytical"]

TRUST_SWEEP_STRUCTURES = [
    ("A_C_A_B_A_C_ts", Path(__file__).parent / "example_files" / "A_C_A_B_A_C_ts.xyz"),
    ("BHDIV_3", BH28_DATASET_DIR / "BHDIV_3_ts.xyz"),
    ("PXBH_3", BH28_DATASET_DIR / "PXBH_3_ts.xyz"),
]

HESSIAN_BASED_OPTIMIZERS = frozenset(
    {
        "rfo",
        "trust-krylov",
        "trustkrylov",
        "trust_krylov",
        "trust-ncg",
        "trustncg",
        "trust_ncg",
        "trust-exact",
        "trustexact",
        "trust_exact",
        "newton-cg",
        "newtoncg",
        "newton_cg",
    }
)

FMAX_TWO_ENDED = 0.05
MAX_STEPS_TWO_ENDED = 300
NPOINTS_TWO_ENDED = 11


def _bh28_reactions(full: bool, subset: bool) -> list[str]:
    if full:
        ref_file = BH28_DATASET_DIR / "reference_barrier_heights.json"
        with open(ref_file) as f:
            return sorted(json.load(f).keys())
    if subset:
        return list(BH28_TS_SUBSET)
    return []


def create_ts_structure(reaction: str | None = None) -> Atoms:
    """Create a transition state structure for TS optimization."""
    script_dir = Path(__file__).parent
    if reaction is not None:
        return read(script_dir / "bh28_benchmark" / "bh28_dataset" / f"{reaction}_ts.xyz")
    return read(script_dir / "example_files" / "A_C_A_B_A_C_ts.xyz")


def _build_ts_kwargs(
    optimizer: str,
    *,
    hessian_update_freq: int | None = None,
    trust_radius: float | None = None,
    max_trust_radius: float | None = None,
    force_finite_diff_hessian: bool = False,
    explicit_hessian_update_freq: bool = False,
) -> dict[str, Any] | None:
    """Build ts_kwargs for Hessian-based TS optimizers."""
    normalized = optimizer.lower()
    ts_kwargs: dict[str, Any] = {}

    if normalized in HESSIAN_BASED_OPTIMIZERS and (
        explicit_hessian_update_freq or hessian_update_freq is not None
    ):
        ts_kwargs["hessian_update_freq"] = hessian_update_freq

    if trust_radius is not None:
        ts_kwargs["trust_radius"] = trust_radius
    if max_trust_radius is not None:
        ts_kwargs["max_trust_radius"] = max_trust_radius

    if normalized in {
        "trust-krylov",
        "trustkrylov",
        "trust_krylov",
        "trust-ncg",
        "trustncg",
        "trust_ncg",
        "trust-exact",
        "trustexact",
        "trust_exact",
    }:
        ts_kwargs.setdefault("ts_search", True)
        ts_kwargs.setdefault("use_bfgs_update", False)

    if force_finite_diff_hessian:
        ts_kwargs["hessian_method"] = "finite_differences"

    return ts_kwargs or None


def benchmark_ts_optimizer(
    backend: str,
    optimizer: str,
    device: str | None = None,
    model_name: str | None = None,
    verbose: bool = True,
    calculate_frequencies: bool = True,
    hessian_update_freq: int | None = None,
    trust_radius: float | None = None,
    max_trust_radius: float | None = None,
    force_finite_diff_hessian: bool = False,
    save_optimized_structure: bool = False,
    structure_label: str | None = None,
    create_structure_func: Callable[[], Atoms] | None = None,
    explicit_hessian_update_freq: bool = False,
) -> dict[str, Any]:
    """Benchmark TS optimizer (Sella, RFO, SciPy trust-region)."""
    ts_kwargs = _build_ts_kwargs(
        optimizer,
        hessian_update_freq=hessian_update_freq,
        trust_radius=trust_radius,
        max_trust_radius=max_trust_radius,
        force_finite_diff_hessian=force_finite_diff_hessian,
        explicit_hessian_update_freq=explicit_hessian_update_freq,
    )

    return benchmark_optimization(
        backend=backend,
        optimizer=optimizer,
        device=device,
        model_name=model_name,
        verbose=verbose,
        test_ts=True,
        create_structure_func=create_structure_func or create_ts_structure,
        suitable_optimizers=[
            "sella",
            "sella-analytical",
            "rfo",
            "trust-krylov",
            "trust-ncg",
            "trust-exact",
            "newton-cg",
        ],
        calculate_frequencies=calculate_frequencies,
        ts_kwargs=ts_kwargs,
        force_finite_diff_hessian=force_finite_diff_hessian,
        save_optimized_structure=save_optimized_structure,
        structure_label=structure_label,
    )


def _optimizer_kwargs_for_two_ended(
    optimizer: str,
    trust_radius: float | None = None,
    max_trust_radius: float | None = None,
) -> dict[str, Any]:
    if optimizer == "sella":
        return {"internal": True, "order": 1}
    if optimizer == "rfo":
        return {
            "hessian_update_freq": 1,
            "trust_radius": trust_radius if trust_radius is not None else 0.02,
            "max_trust_radius": max_trust_radius if max_trust_radius is not None else 0.06,
        }
    if optimizer in {"trust-krylov", "trustkrylov", "trust_krylov"}:
        return {
            "ts_search": True,
            "hessian_update_freq": 1,
            "trust_radius": trust_radius if trust_radius is not None else 0.05,
            "max_trust_radius": max_trust_radius if max_trust_radius is not None else 0.15,
            "use_bfgs_update": False,
        }
    return {
        "ts_search": True,
        "hessian_update_freq": 1,
        "trust_radius": trust_radius if trust_radius is not None else 0.1,
        "max_trust_radius": max_trust_radius if max_trust_radius is not None else 0.3,
        "use_bfgs_update": False,
    }


def _run_two_ended(
    reactant: Atoms,
    product: Atoms,
    strategy: str,
    optimizer: str,
    backend: str,
    device: str,
    model_name: str | None,
    fmax: float,
    steps: int,
    npoints: int,
) -> dict[str, Any]:
    from famex import Explorer

    explorer = Explorer(
        atoms=[reactant.copy(), product.copy()],
        backend=backend,
        model_name=model_name,
        device=device,
        target="ts",
        strategy=strategy,
        local_optimizer=optimizer,
        verbose=0,
        ts_kwargs=_optimizer_kwargs_for_two_ended(optimizer) if optimizer != "sella" else None,
    )

    start = time.perf_counter()
    if strategy == "interpolate":
        result = explorer.run(
            fmax=fmax,
            steps=steps,
            npoints=npoints,
            calculate_frequencies=True,
        )
    elif strategy == "cineb":
        result = explorer.run(
            fmax=fmax,
            steps=steps,
            npoints=npoints,
            spring_constant=5.0,
            calculate_frequencies=True,
        )
    else:
        result = explorer.run(
            fmax=fmax,
            steps=steps,
            npoints=npoints,
            step_size=0.1,
            distance_threshold=0.5,
            optimize_endpoints=True,
            refine_ts=True,
            calculate_frequencies=True,
        )
    elapsed = time.perf_counter() - start

    opt_atoms = result.get("optimized_atoms", result.get("ts_structure"))
    if isinstance(opt_atoms, list):
        opt_atoms = opt_atoms[0]
    if not isinstance(opt_atoms, Atoms):
        raise TypeError(f"Unexpected optimized_atoms type: {type(opt_atoms)}")

    freq = result.get("frequency_analysis", {})
    ts_analysis = freq.get("ts_analysis", {}) if isinstance(freq, dict) else {}
    return {
        "strategy": strategy,
        "optimizer": optimizer,
        "backend": backend,
        "converged": bool(result.get("converged", False)),
        "steps": int(result.get("steps_taken", 0)),
        "final_force": float(np.max(np.abs(opt_atoms.get_forces()))),
        "time_s": elapsed,
        "strings_met": bool(result.get("strings_met", True)),
        "is_valid_ts": bool(freq.get("is_ts", result.get("is_ts", False))),
        "n_imaginary": int(ts_analysis.get("n_imaginary_frequencies", -1)),
    }


def _should_show_backend(results: dict[str, Any]) -> bool:
    backend = results.get("backend", "").lower()
    if backend == "mock":
        return False
    return results.get("available", False)


def print_bh28_summary(results_list: list[dict[str, Any]]) -> None:
    """Print per-optimizer success rates on the BH28 dataset."""
    by_optimizer: dict[str, list[dict[str, Any]]] = {}
    for result in results_list:
        if not _should_show_backend(result):
            continue
        opt = result.get("optimizer", "unknown")
        by_optimizer.setdefault(opt, []).append(result)

    print(f"\n{'=' * 120}")
    print("BH28 DATASET — per-optimizer summary (converged + valid TS)")
    print(f"{'=' * 120}")
    print(
        f"{'Optimizer':<28} {'Valid TS':<12} {'Converged':<12} "
        f"{'Avg Steps':<12} {'Avg Time (s)':<14}"
    )
    print("=" * 120)

    for optimizer, rows in sorted(by_optimizer.items()):
        valid = sum(
            1
            for r in rows
            if r.get("optimization_results", {}).get("converged")
            and r.get("frequency_results", {}).get("is_valid_result")
        )
        converged = sum(1 for r in rows if r.get("optimization_results", {}).get("converged"))
        total = len(rows)
        steps = [
            r.get("optimization_results", {}).get("steps_taken", 0)
            for r in rows
            if r.get("optimization_results", {}).get("converged")
        ]
        times = [r.get("timings", {}).get("total", 0.0) for r in rows]
        avg_steps = sum(steps) / len(steps) if steps else 0.0
        avg_time = sum(times) / len(times) if times else 0.0
        print(
            f"{optimizer:<28} {valid}/{total:<10} {converged}/{total:<10} "
            f"{avg_steps:<12.1f} {avg_time:<14.1f}"
        )


def print_trust_sweep_summary(results: list[dict[str, Any]]) -> None:
    """Print trust-radius sweep summary."""
    if not results:
        return

    print(f"\n{'=' * 110}")
    print("TRUST-RADIUS SWEEP — local TS (converged + valid TS)")
    print(f"{'=' * 110}")
    print(
        f"{'Structure':<18} {'Optimizer':<15} {'TrustR':<8} {'MaxTR':<8} "
        f"{'Conv':<6} {'ValidTS':<10} {'Time(s)':<8}"
    )
    print("-" * 110)
    for row in results:
        if "error" in row:
            continue
        print(
            f"{row.get('structure', '?'):<18} "
            f"{row.get('optimizer', '?'):<15} "
            f"{row.get('trust_radius', 0):<8.2f} "
            f"{row.get('max_trust_radius', 0):<8.2f} "
            f"{'Y' if row.get('converged') else 'N':<6} "
            f"{'Y' if row.get('is_valid_ts') else 'N':<10} "
            f"{row.get('time_s', 0):<8.1f}"
        )


def print_two_ended_summary(results: list[dict[str, Any]]) -> None:
    """Print two-ended TS strategy comparison summary."""
    if not results:
        return

    print(f"\n{'=' * 110}")
    print("TWO-ENDED TS (interpolate vs cineb vs growing_string)")
    print(f"{'=' * 110}")
    print(
        f"{'Reaction':<14} {'Strategy':<16} {'Conv':<6} {'Steps':<7} {'Force':<10} "
        f"{'ValidTS':<10} {'Imag':<6} {'Time(s)':<8}"
    )
    print("-" * 110)
    for r in results:
        if "error" in r:
            continue
        print(
            f"{r.get('reaction', '?'):<14} "
            f"{r.get('strategy', '?'):<16} "
            f"{'Y' if r.get('converged') else 'N':<6} "
            f"{r.get('steps', 0):<7} "
            f"{r.get('final_force', 0):<10.4f} "
            f"{'Y' if r.get('is_valid_ts') else 'N':<10} "
            f"{r.get('n_imaginary', -1):<6} "
            f"{r.get('time_s', 0):<8.1f}"
        )

    print("\nStrategy success rates (converged + valid TS):")
    for strategy in TWO_ENDED_STRATEGIES:
        subset = [r for r in results if r.get("strategy") == strategy and "error" not in r]
        ok = sum(1 for r in subset if r.get("converged") and r.get("is_valid_ts"))
        total = len(subset)
        print(f"  {strategy:<16}: {ok}/{total} ({100 * ok / total if total else 0:.0f}%)")


def print_frequency_analysis_summary(results_list: list[dict[str, Any]]) -> None:
    """Print a detailed frequency analysis summary for TS optimization."""
    print(f"\n{'=' * 120}")
    print("TRANSITION STATE VALIDATION SUMMARY")
    print(f"{'=' * 120}")
    print(
        "A valid TS must have exactly 1 imaginary frequency (saddle point). "
        "Optimizers that fail to find a TS are marked as failed."
    )

    print(
        f"\n{'Backend':<12} {'Optimizer':<15} {'Imag. Freq':<12} {'Status':<15} {'Lowest 3 Freq (cm⁻¹)':<25}"
    )
    print("=" * 120)

    failed_optimizers = []
    for results in results_list:
        if _should_show_backend(results) and "frequency_results" in results:
            freq_results = results["frequency_results"]
            n_imag = freq_results.get("n_imaginary_frequencies", 0)
            is_valid = freq_results.get("is_valid_result", False)
            method_used = freq_results.get("method_used", "unknown")

            backend = results.get("backend", "unknown")
            optimizer = results.get("optimizer", "unknown")

            all_freqs = freq_results.get("all_frequencies", [])
            if not all_freqs:
                all_freqs = freq_results.get("frequencies", [])
            if not all_freqs and freq_results.get("ts_analysis", {}):
                ts_analysis = freq_results.get("ts_analysis", {})
                all_freqs = ts_analysis.get("all_frequencies", [])

            if all_freqs:
                filtered_freqs = [f for f in all_freqs if abs(f) > 10.0]
                if not filtered_freqs and all_freqs:
                    filtered_freqs = all_freqs
                frequencies = sorted(
                    filtered_freqs,
                    key=lambda x: (
                        abs(x) >= 0 if isinstance(x, complex) else x >= 0,
                        abs(x) if isinstance(x, complex) else x,
                    ),
                )
            else:
                frequencies = []

            if len(frequencies) >= 3:
                freq_str = f"[{frequencies[0]:.1f}, {frequencies[1]:.1f}, {frequencies[2]:.1f}]"
            elif len(frequencies) > 0:
                freq_str = f"[{', '.join(f'{f:.1f}' for f in frequencies)}]"
            elif method_used == "not_calculated":
                freq_str = "Failed"
            else:
                freq_str = "N/A"

            if is_valid:
                status = "✅ Valid TS"
            elif n_imag == 0:
                status = "❌ Minimum (0 imag)"
                failed_optimizers.append((backend, optimizer, "found minimum, not TS"))
            elif n_imag > 1:
                status = f"❌ Not TS ({n_imag} imag)"
                failed_optimizers.append(
                    (backend, optimizer, f"found {n_imag} imaginary frequencies, not TS")
                )
            else:
                status = "❌ Invalid"

            print(f"{backend:<12} {optimizer:<15} {n_imag:<12} {status:<15} {freq_str:<25}")

    available_results = [
        r for r in results_list if _should_show_backend(r) and "frequency_results" in r
    ]
    if available_results:
        valid_count = sum(
            1 for r in available_results if r["frequency_results"].get("is_valid_result", False)
        )
        total_count = len(available_results)
        success_rate = (valid_count / total_count * 100) if total_count > 0 else 0

        print(f"\n{'=' * 120}")
        print(
            f"SUMMARY: {valid_count}/{total_count} optimizations found valid TS ({success_rate:.1f}% success rate)"
        )

        ts_with_wrong_freq = sum(
            1
            for r in available_results
            if r["frequency_results"].get("n_imaginary_frequencies", 0) != 1
        )
        if ts_with_wrong_freq > 0:
            print(
                f"\n⚠️  WARNING: {ts_with_wrong_freq} optimizer(s) failed to find transition states:"
            )
            for backend, optimizer, reason in failed_optimizers:
                print(f"   - {backend}/{optimizer}: {reason}")


def print_optimizer_summary(results_list: list[dict[str, Any]]) -> None:
    """Print a summary table focused on TS optimizer comparison."""
    print(f"\n{'=' * 120}")
    print("TS OPTIMIZER COMPARISON")
    print(f"{'=' * 120}")
    print("Legend: ✅ = Success, ❌ = Failed, ⚠️ = Warning, ⏱️ = Time")

    print(
        f"\n{'Backend':<12} {'Optimizer':<15} {'Status':<10} {'Total Time':<12} {'Opt Steps':<10} "
        f"{'Time/Step':<12} {'Final Energy':<15} {'Max Force':<12} {'TS Valid':<10}"
    )
    print("=" * 120)
    print(
        "Note: Status shows convergence + TS validation. '✅ TS' = converged & valid TS, "
        "'⚠️ No TS' = converged but not a TS (failed), '❌' = didn't converge"
    )

    for results in results_list:
        if not _should_show_backend(results):
            continue

        timings = results["timings"]
        opt_results = results.get("optimization_results", {})
        freq_results = results.get("frequency_results", {})
        steps_taken = opt_results.get("steps_taken", 0)
        time_per_step = timings.get("avg_time_per_step", 0)
        backend = results.get("backend", "unknown")
        optimizer = results.get("optimizer", "unknown")
        converged = opt_results.get("converged", False)
        final_energy = opt_results.get("final_energy", None)
        max_force = opt_results.get("max_force", None)
        is_valid = freq_results.get("is_valid_result", False)

        steps_str = str(steps_taken) if steps_taken is not None else "N/A"
        energy_str = f"{final_energy:.6f}" if final_energy is not None else "N/A"
        force_str = f"{max_force:.6f}" if max_force is not None else "N/A"

        n_imag = freq_results.get("n_imaginary_frequencies", 0)
        if converged and is_valid:
            status = "✅ TS"
        elif converged and not is_valid:
            status = "⚠️ No TS"
        else:
            status = "❌"

        valid_str = "✅" if is_valid else "❌"
        if not is_valid and n_imag != 1:
            valid_str = f"❌ ({n_imag} imag)"

        print(
            f"{backend:<12} {optimizer:<15} {status:<10} {timings.get('total', 0):<12.3f} {steps_str:<10} "
            f"{time_per_step:<12.6f} {energy_str:<15} {force_str:<12} {valid_str:<10}"
        )


def print_performance_summary(results_list: list[dict[str, Any]]) -> None:
    """Print comprehensive performance profiler data."""
    has_performance_data = any(
        results.get("available") and "performance" in results for results in results_list
    )
    if not has_performance_data:
        return

    all_sections = set()
    for results in results_list:
        if results.get("available") and "performance" in results:
            perf = results["performance"]
            timings = perf.get("timings", {})
            all_sections.update(timings.keys())

    if not all_sections:
        return

    for section in sorted(all_sections):
        for results in results_list:
            if results.get("available") and "performance" in results:
                perf = results["performance"]
                timings = perf.get("timings", {})
                if section in timings:
                    pass


def _run_trust_radius_sweep(
    backend: str,
    device: str,
    model_name: str | None,
    verbose: int,
    calculate_frequencies: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    print("\nRunning trust-radius sweep ...")
    for name, path in TRUST_SWEEP_STRUCTURES:
        if not path.exists():
            print(f"  Skipping {name}: {path} not found")
            continue
        for trust_radius, max_trust_radius in TRUST_RADIUS_PRESETS:
            for opt in TRUST_SWEEP_OPTIMIZERS:
                try:
                    row = benchmark_ts_optimizer(
                        backend=backend,
                        optimizer=opt,
                        device=device,
                        model_name=model_name,
                        verbose=verbose,
                        calculate_frequencies=calculate_frequencies,
                        trust_radius=trust_radius,
                        max_trust_radius=max_trust_radius,
                        hessian_update_freq=1,
                        structure_label=f"{name}_{opt}_r{trust_radius:.2f}",
                        create_structure_func=lambda p=path: read(str(p)),
                    )
                    opt_results = row.get("optimization_results", {})
                    freq_results = row.get("frequency_results", {})
                    results.append(
                        {
                            "structure": name,
                            "optimizer": opt,
                            "trust_radius": trust_radius,
                            "max_trust_radius": max_trust_radius,
                            "converged": bool(opt_results.get("converged", False)),
                            "is_valid_ts": bool(freq_results.get("is_valid_result", False)),
                            "steps": int(opt_results.get("steps_taken", 0)),
                            "time_s": row.get("timings", {}).get("total", 0.0),
                        }
                    )
                    print(
                        f"  {name}/{opt} r={trust_radius:.2f}: "
                        f"conv={opt_results.get('converged')}, "
                        f"valid_ts={freq_results.get('is_valid_result')}, "
                        f"time={row.get('timings', {}).get('total', 0):.1f}s"
                    )
                except Exception as exc:
                    results.append(
                        {
                            "structure": name,
                            "optimizer": opt,
                            "trust_radius": trust_radius,
                            "max_trust_radius": max_trust_radius,
                            "error": str(exc),
                        },
                    )
                    print(f"  {name}/{opt} r={trust_radius:.2f}: ERROR {exc}")
    return results


def _run_two_ended_benchmarks(
    backend: str,
    device: str,
    model_name: str | None,
    quick: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    reactions = sorted(
        p.stem.replace("_reactant", "") for p in Z93_DATASET_DIR.glob("reaction_*_reactant.*")
    )
    if quick:
        reactions = reactions[:3]

    print(f"\nRunning two-ended TS on {len(reactions)} Zimmermann-93 reactions ...")
    two_ended_optimizer = "rfo"
    for reaction in reactions:
        reactant = read(str(Z93_DATASET_DIR / f"{reaction}_reactant.xyz"))
        product = read(str(Z93_DATASET_DIR / f"{reaction}_product.xyz"))
        if isinstance(reactant, list):
            reactant = reactant[0]
        if isinstance(product, list):
            product = product[0]

        for strategy in TWO_ENDED_STRATEGIES:
            try:
                row = _run_two_ended(
                    reactant,
                    product,
                    strategy,
                    two_ended_optimizer,
                    backend,
                    device,
                    model_name,
                    FMAX_TWO_ENDED,
                    MAX_STEPS_TWO_ENDED,
                    NPOINTS_TWO_ENDED,
                )
                row["reaction"] = reaction
                results.append(row)
                print(
                    f"  {reaction}/{strategy}: conv={row['converged']}, "
                    f"valid_ts={row['is_valid_ts']}, time={row['time_s']:.1f}s"
                )
            except Exception as exc:
                results.append(
                    {"reaction": reaction, "strategy": strategy, "error": str(exc)},
                )
                print(f"  {reaction}/{strategy}: ERROR {exc}")
    return results


def _collect_structures(
    full_bh28: bool,
    bh28_subset: bool,
    skip_example: bool,
) -> list[tuple[str | None, str]]:
    structures: list[tuple[str | None, str]] = []
    if full_bh28:
        structures.extend((reaction, reaction) for reaction in _bh28_reactions(True, False))
    elif bh28_subset:
        structures.extend((reaction, reaction) for reaction in _bh28_reactions(False, True))
    if not full_bh28 and not bh28_subset or not skip_example and (full_bh28 or bh28_subset):
        structures.append((None, "A_C_A_B_A_C_ts"))
    return structures


def _hessian_freqs_for_optimizer(
    optimizer: str,
    hessian_sweep: bool,
    default_freq: int,
) -> list[int | None]:
    if optimizer in HESSIAN_BASED_OPTIMIZERS and hessian_sweep:
        return [None, 1, 5]
    if optimizer in HESSIAN_BASED_OPTIMIZERS:
        return [default_freq]
    return [None]


def _hessian_freqs_for_param_sweep(optimizer: str) -> list[int | None]:
    if optimizer in PARAM_SWEEP_DENSE_OPTIMIZERS:
        return list(PARAM_SWEEP_HESSIAN_FREQS)
    if optimizer in HESSIAN_BASED_OPTIMIZERS:
        return [1]
    return [None]


def _param_sweep_config_key(
    optimizer: str,
    hessian_update_freq: int | None,
    trust_radius: float | None,
    max_trust_radius: float | None,
) -> tuple[str, int | None, float | None, float | None]:
    return (optimizer, hessian_update_freq, trust_radius, max_trust_radius)


def _aggregate_param_sweep_rows(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, int | None, float | None, float | None], dict[str, Any]]:
    aggregated: dict[tuple[str, int | None, float | None, float | None], dict[str, Any]] = {}
    for row in rows:
        if row.get("error") or not _should_show_backend(row):
            continue
        key = _param_sweep_config_key(
            row.get("optimizer", "unknown"),
            row.get("hessian_update_freq"),
            row.get("trust_radius"),
            row.get("max_trust_radius"),
        )
        bucket = aggregated.setdefault(
            key,
            {
                "optimizer": key[0],
                "hessian_update_freq": key[1],
                "trust_radius": key[2],
                "max_trust_radius": key[3],
                "total": 0,
                "valid": 0,
                "converged": 0,
                "steps": [],
                "times": [],
            },
        )
        bucket["total"] += 1
        opt_results = row.get("optimization_results", {})
        freq_results = row.get("frequency_results", {})
        converged = bool(opt_results.get("converged", False))
        valid = converged and bool(freq_results.get("is_valid_result", False))
        if valid:
            bucket["valid"] += 1
        if converged:
            bucket["converged"] += 1
            bucket["steps"].append(int(opt_results.get("steps_taken", 0)))
        bucket["times"].append(float(row.get("timings", {}).get("total", 0.0)))
    return aggregated


def print_param_sweep_summary(rows: list[dict[str, Any]]) -> None:
    """Print combined hessian/trust-radius sweep summary."""
    if not rows:
        return

    aggregated = _aggregate_param_sweep_rows(rows)
    if not aggregated:
        print("\nNo param-sweep results to summarize.")
        return

    print(f"\n{'=' * 130}")
    print("PARAM SWEEP — per-config summary (converged + valid TS)")
    print(f"{'=' * 130}")
    print(
        f"{'Optimizer':<16} {'HessFreq':<10} {'TrustR':<8} {'MaxTR':<8} "
        f"{'Valid':<10} {'Conv':<10} {'AvgSteps':<10} {'AvgTime(s)':<12}"
    )
    print("-" * 130)

    def sort_key(item: tuple[tuple[str, int | None, float | None, float | None], dict[str, Any]]):
        key, stats = item
        avg_steps = sum(stats["steps"]) / len(stats["steps"]) if stats["steps"] else float("inf")
        avg_time = sum(stats["times"]) / len(stats["times"]) if stats["times"] else float("inf")
        return (-stats["valid"], -stats["converged"], avg_steps, avg_time, key[0], key[1] or -1)

    for _key, stats in sorted(aggregated.items(), key=sort_key):
        hess_label = (
            "single" if stats["hessian_update_freq"] is None else str(stats["hessian_update_freq"])
        )
        trust_r = stats["trust_radius"]
        max_tr = stats["max_trust_radius"]
        avg_steps = sum(stats["steps"]) / len(stats["steps"]) if stats["steps"] else 0.0
        avg_time = sum(stats["times"]) / len(stats["times"]) if stats["times"] else 0.0
        print(
            f"{stats['optimizer']:<16} {hess_label:<10} "
            f"{(trust_r if trust_r is not None else 0.0):<8.2f} "
            f"{(max_tr if max_tr is not None else 0.0):<8.2f} "
            f"{stats['valid']}/{stats['total']:<8} "
            f"{stats['converged']}/{stats['total']:<8} "
            f"{avg_steps:<10.1f} {avg_time:<12.1f}"
        )


def recommend_param_defaults(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pick best config per optimizer: valid TS rate, then avg steps, then avg time."""
    aggregated = _aggregate_param_sweep_rows(rows)
    if not aggregated:
        return []

    by_optimizer: dict[
        str, list[tuple[tuple[str, int | None, float | None, float | None], dict[str, Any]]]
    ] = {}
    for key, stats in aggregated.items():
        by_optimizer.setdefault(stats["optimizer"], []).append((key, stats))

    recommendations: list[dict[str, Any]] = []
    print(f"\n{'=' * 130}")
    print("PARAM SWEEP — recommended defaults per optimizer")
    print(f"{'=' * 130}")
    print(
        f"{'Optimizer':<16} {'HessFreq':<10} {'TrustR':<8} {'MaxTR':<8} "
        f"{'Valid':<10} {'AvgSteps':<10} {'AvgTime(s)':<12}"
    )
    print("-" * 130)

    for optimizer in sorted(by_optimizer):
        candidates = by_optimizer[optimizer]

        def rank(item: tuple[tuple[str, int | None, float | None, float | None], dict[str, Any]]):
            _, stats = item
            valid_rate = stats["valid"] / stats["total"] if stats["total"] else 0.0
            avg_steps = (
                sum(stats["steps"]) / len(stats["steps"]) if stats["steps"] else float("inf")
            )
            avg_time = sum(stats["times"]) / len(stats["times"]) if stats["times"] else float("inf")
            return (-valid_rate, -stats["valid"], avg_steps, avg_time)

        _, best = min(candidates, key=rank)
        hess_label = (
            "single" if best["hessian_update_freq"] is None else str(best["hessian_update_freq"])
        )
        avg_steps = sum(best["steps"]) / len(best["steps"]) if best["steps"] else 0.0
        avg_time = sum(best["times"]) / len(best["times"]) if best["times"] else 0.0
        print(
            f"{optimizer:<16} {hess_label:<10} "
            f"{(best['trust_radius'] if best['trust_radius'] is not None else 0.0):<8.2f} "
            f"{(best['max_trust_radius'] if best['max_trust_radius'] is not None else 0.0):<8.2f} "
            f"{best['valid']}/{best['total']:<8} "
            f"{avg_steps:<10.1f} {avg_time:<12.1f}"
        )
        recommendations.append(
            {
                "optimizer": optimizer,
                "hessian_update_freq": best["hessian_update_freq"],
                "trust_radius": best["trust_radius"],
                "max_trust_radius": best["max_trust_radius"],
                "valid": best["valid"],
                "total": best["total"],
                "avg_steps": avg_steps,
                "avg_time": avg_time,
            }
        )

    return recommendations


def _run_param_sweep(
    backend: str,
    device: str,
    model_name: str | None,
    verbose: int,
    calculate_frequencies: bool,
    structures: list[tuple[str | None, str]],
    force_finite_diff_hessian: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    sweep_optimizers = list(PARAM_SWEEP_OPTIMIZERS) + list(PARAM_SWEEP_BASELINE_OPTIMIZERS)

    print(
        f"\nRunning param sweep on {len(structures)} structure(s), "
        f"{len(PARAM_SWEEP_OPTIMIZERS)} Hessian-based optimizers "
        f"+ {len(PARAM_SWEEP_BASELINE_OPTIMIZERS)} baselines ..."
    )

    for reaction, structure_label_name in structures:
        for optimizer in sweep_optimizers:
            hessian_freqs = _hessian_freqs_for_param_sweep(optimizer)
            trust_presets: list[tuple[float | None, float | None]] = (
                list(TRUST_RADIUS_PRESETS)
                if optimizer in PARAM_SWEEP_OPTIMIZERS
                else [(None, None)]
            )

            for hessian_freq in hessian_freqs:
                for trust_radius, max_trust_radius in trust_presets:
                    hess_label = "single" if hessian_freq is None else str(hessian_freq)
                    trust_label = (
                        f"r{trust_radius:.2f}"
                        if trust_radius is not None and max_trust_radius is not None
                        else "default"
                    )
                    label = f"{structure_label_name}_{optimizer}_{hess_label}_{trust_label}"

                    try:
                        row = benchmark_ts_optimizer(
                            backend=backend,
                            optimizer=optimizer,
                            device=device,
                            model_name=model_name,
                            verbose=verbose,
                            calculate_frequencies=calculate_frequencies,
                            hessian_update_freq=hessian_freq,
                            trust_radius=trust_radius,
                            max_trust_radius=max_trust_radius,
                            force_finite_diff_hessian=force_finite_diff_hessian,
                            structure_label=label,
                            create_structure_func=lambda r=reaction: create_ts_structure(r),
                            explicit_hessian_update_freq=optimizer in HESSIAN_BASED_OPTIMIZERS,
                        )
                        row["optimizer"] = optimizer
                        row["hessian_update_freq"] = hessian_freq
                        row["trust_radius"] = trust_radius
                        row["max_trust_radius"] = max_trust_radius
                        if reaction is not None:
                            row["reaction"] = reaction
                        results.append(row)

                        opt_results = row.get("optimization_results", {})
                        freq_results = row.get("frequency_results", {})
                        print(
                            f"  {structure_label_name}/{optimizer} h={hess_label} {trust_label}: "
                            f"conv={opt_results.get('converged')}, "
                            f"valid_ts={freq_results.get('is_valid_result')}, "
                            f"time={row.get('timings', {}).get('total', 0):.1f}s"
                        )
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:
                        results.append(
                            {
                                "backend": backend,
                                "optimizer": optimizer,
                                "hessian_update_freq": hessian_freq,
                                "trust_radius": trust_radius,
                                "max_trust_radius": max_trust_radius,
                                "reaction": reaction,
                                "device": device,
                                "test_ts": True,
                                "available": False,
                                "error": str(exc),
                                "timings": {},
                                "optimization_results": {},
                                "frequency_results": {},
                            },
                        )
                        print(
                            f"  {structure_label_name}/{optimizer} h={hess_label} {trust_label}: "
                            f"ERROR {exc}"
                        )

    return results


def main() -> int:
    """Run the unified TS optimizer comparison benchmark."""
    interface = FAMEXExampleInterface(
        name="TS Optimizer Benchmark",
        description="Unified transition state optimizer comparison (local + two-ended)",
        epilog=create_standard_epilog("benchmark"),
    )

    parser = interface.create_parser()

    parser.add_argument(
        "--optimizers",
        type=str,
        help="Comma-separated list of optimizers to benchmark",
    )
    parser.add_argument(
        "--freq",
        action="store_true",
        default=True,
        help="Perform frequency analysis to validate TS (default: True). Use --no-freq to disable.",
    )
    parser.add_argument(
        "--no-freq",
        dest="freq",
        action="store_false",
        help="Skip frequency analysis (faster but no TS validation)",
    )
    parser.add_argument(
        "--hessian-update-freq",
        type=int,
        default=5,
        help="Hessian update frequency for Hessian-based optimizers (default: 5)",
    )
    parser.add_argument(
        "--hessian-sweep",
        action="store_true",
        help="Sweep hessian update frequencies [single, 1, 5] on local TS runs",
    )
    parser.add_argument(
        "--trust-sweep",
        action="store_true",
        help="Run trust-radius preset sweep on selected structures and optimizers",
    )
    parser.add_argument(
        "--param-sweep",
        action="store_true",
        help=(
            "Run combined hessian_update_freq x trust-radius grid on BH28 structures "
            "(use with --full-bh28 or --bh28-subset)"
        ),
    )
    parser.add_argument(
        "--force-finite-diff-hessian",
        action="store_true",
        help="Force use of finite difference Hessians instead of analytical",
    )
    parser.add_argument(
        "--save-xyz",
        action="store_true",
        help="Save optimized structures as XYZ files in the current directory",
    )
    parser.add_argument(
        "--full-bh28",
        action="store_true",
        help="Benchmark all 28 BH28 TS structures",
    )
    parser.add_argument(
        "--bh28-subset",
        action="store_true",
        help="Benchmark the 6-reaction BH28 TS subset",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Backend model name (default: uma-s-1p2 for UMA)",
    )
    parser.add_argument(
        "--skip-example",
        action="store_true",
        help="With --full-bh28 or --bh28-subset, skip the single-structure example benchmark",
    )
    parser.add_argument(
        "--two-ended",
        action="store_true",
        help="Also run two-ended Zimmermann-93 TS benchmarks after local section",
    )
    parser.add_argument(
        "--two-ended-only",
        action="store_true",
        help="Skip local TS benchmarks and run two-ended section only",
    )
    parser.add_argument(
        "--quick-two-ended",
        action="store_true",
        help="Use only 3 Zimmermann-93 reactions for two-ended benchmarks",
    )

    args = parser.parse_args()

    interface.print_header()
    interface.setup_logging(args.verbose)

    requested = [b.strip() for b in args.backends.split(",")] if args.backends else None
    _, available_backends = interface.select_backend(
        requested_backends=requested,
        verbose=args.verbose,
    )
    if not available_backends:
        interface.print_error(
            "No available backends found. Please install at least one ML backend:\n"
            "  - MACE: pip install mace-torch\n"
            "  - UMA: pip install fairchem-core\n"
            "  - AIMNet2: pip install aimnet2\n"
            "  - SO3LR: pip install so3lr"
        )
        return 1

    valid_optimizers = list(LOCAL_OPTIMIZERS_DEFAULT) + ["newton-cg"]
    if args.optimizers:
        requested_optimizers = [o.strip().lower() for o in args.optimizers.split(",")]
        ts_optimizers = [opt for opt in requested_optimizers if opt in valid_optimizers]
        invalid_opts = [opt for opt in requested_optimizers if opt not in valid_optimizers]
        if invalid_opts:
            interface.print_warning(
                f"Invalid optimizers ignored: {', '.join(invalid_opts)}. "
                f"Valid options: {', '.join(valid_optimizers)}"
            )
    else:
        ts_optimizers = list(LOCAL_OPTIMIZERS_DEFAULT)

    if not ts_optimizers:
        interface.print_error("No valid TS optimizers specified!")
        return 1

    interface.print_backend_summary(available_backends, "Benchmarking Backends")
    device = interface.get_device_info(args.device)

    structure_label = "single example"
    if args.full_bh28:
        structure_label = "full BH28 (28 reactions)"
    elif args.bh28_subset:
        structure_label = "BH28 subset (6 reactions)"

    config = {
        "Device": device,
        "Model": args.model_name or "backend default",
        "Output": args.output or interface.get_default_output_file(),
        "Verbose": args.verbose,
        "Test Types": "Transition State Optimization",
        "Structures": structure_label,
        "Hessian sweep": args.hessian_sweep,
        "Trust sweep": args.trust_sweep,
        "Param sweep": args.param_sweep,
        "Two-ended": args.two_ended or args.two_ended_only,
    }
    interface.print_configuration(config)

    all_results: dict[str, Any] = {
        "backend": available_backends[0] if len(available_backends) == 1 else available_backends,
        "model_name": args.model_name,
        "device": device,
        "local": [],
        "trust_sweep": [],
        "param_sweep": [],
        "two_ended": [],
    }

    results_list: list[dict[str, Any]] = []
    run_local = not args.two_ended_only and not args.trust_sweep and not args.param_sweep

    if args.param_sweep and not args.full_bh28 and not args.bh28_subset:
        interface.print_error("--param-sweep requires --full-bh28 or --bh28-subset")
        return 1

    if args.trust_sweep and not args.two_ended_only:
        for backend in available_backends:
            all_results["trust_sweep"].extend(
                _run_trust_radius_sweep(
                    backend,
                    device,
                    args.model_name,
                    args.verbose,
                    args.freq,
                )
            )
        print_trust_sweep_summary(all_results["trust_sweep"])

    if args.param_sweep and not args.two_ended_only:
        structures = _collect_structures(args.full_bh28, args.bh28_subset, skip_example=True)
        param_sweep_results: list[dict[str, Any]] = []
        for backend in available_backends:
            param_sweep_results.extend(
                _run_param_sweep(
                    backend,
                    device,
                    args.model_name,
                    args.verbose,
                    args.freq,
                    structures,
                    args.force_finite_diff_hessian,
                )
            )
        all_results["param_sweep"] = param_sweep_results
        print_param_sweep_summary(param_sweep_results)
        all_results["recommended_defaults"] = recommend_param_defaults(param_sweep_results)

    if run_local:
        structures = _collect_structures(args.full_bh28, args.bh28_subset, args.skip_example)

        for backend in available_backends:
            for optimizer in ts_optimizers:
                hessian_freqs = _hessian_freqs_for_optimizer(
                    optimizer,
                    args.hessian_sweep,
                    args.hessian_update_freq,
                )

                for hessian_freq in hessian_freqs:
                    for reaction, structure_label_name in structures:
                        optimizer_name = optimizer
                        if optimizer in HESSIAN_BASED_OPTIMIZERS and args.hessian_sweep:
                            if hessian_freq is None:
                                optimizer_name = f"{optimizer}_single_hessian"
                            else:
                                optimizer_name = f"{optimizer}_hessian_freq_{hessian_freq}"

                        label = structure_label_name
                        if reaction is not None:
                            label = f"{reaction}_{optimizer_name}"

                        try:
                            results = benchmark_ts_optimizer(
                                backend=backend,
                                optimizer=optimizer,
                                device=device,
                                model_name=args.model_name,
                                verbose=args.verbose,
                                calculate_frequencies=args.freq,
                                hessian_update_freq=hessian_freq,
                                force_finite_diff_hessian=args.force_finite_diff_hessian,
                                save_optimized_structure=args.save_xyz,
                                structure_label=label,
                                create_structure_func=lambda r=reaction: create_ts_structure(r),
                            )
                            results["optimizer"] = optimizer_name
                            results["hessian_update_freq"] = hessian_freq
                            if reaction is not None:
                                results["reaction"] = reaction
                            results_list.append(results)
                        except KeyboardInterrupt:
                            break
                        except Exception as e:
                            results_list.append(
                                {
                                    "backend": backend,
                                    "optimizer": optimizer_name,
                                    "hessian_update_freq": hessian_freq,
                                    "reaction": reaction,
                                    "device": device,
                                    "test_ts": True,
                                    "available": False,
                                    "error": str(e),
                                    "timings": {},
                                    "optimization_results": {},
                                    "frequency_results": {},
                                },
                            )

        all_results["local"] = results_list

        if args.full_bh28 or args.bh28_subset:
            print_bh28_summary(results_list)
        print_frequency_analysis_summary(results_list)
        print_optimizer_summary(results_list)
        print_performance_summary(results_list)

    if args.two_ended or args.two_ended_only:
        for backend in available_backends:
            all_results["two_ended"].extend(
                _run_two_ended_benchmarks(
                    backend,
                    device,
                    args.model_name,
                    args.quick_two_ended,
                )
            )
        print_two_ended_summary(all_results["two_ended"])

    interface.save_results(all_results, args.output or interface.get_default_output_file())
    interface.print_success()
    return 0


if __name__ == "__main__":
    sys.exit(main())
