#!/usr/bin/env python3
"""Run local and two-ended TS benchmarks on GPU and print a comparison summary.

Default trust radii for RFO/SciPy TS optimizers are 0.1 / 0.3 Å (larger than
geomeTRIC's 0.01 / 0.03 for ML potentials). Use --trust-sweep to compare presets.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

BH28_TS_SUBSET = [
    "BHDIV_3",
    "PXBH_3",
    "CADBH_2",
    "CRBH_1",
    "PXBH_2",
    "BHPERI_2",
]

LOCAL_OPTIMIZERS = [
    "sella",
    "rfo",
    "trust-krylov",
    "trust-ncg",
    "trust-exact",
]

TWO_ENDED_STRATEGIES = ["interpolate", "cineb", "growing_string"]

FMAX_LOCAL = 0.05
FMAX_TWO_ENDED = 0.05
MAX_STEPS_LOCAL = 500
MAX_STEPS_TWO_ENDED = 300
NPOINTS = 11

TRUST_RADIUS_PRESETS: list[tuple[float, float]] = [
    (0.02, 0.06),
    (0.05, 0.15),
    (0.1, 0.3),
]

TRUST_SWEEP_OPTIMIZERS = ["rfo", "trust-exact", "trust-krylov"]

TRUST_SWEEP_STRUCTURES = [
    ("A_C_A_B_A_C_ts", Path(__file__).parent / "example_files" / "A_C_A_B_A_C_ts.xyz"),
    ("BHDIV_3", Path(__file__).parent / "bh28_benchmark" / "bh28_dataset" / "BHDIV_3_ts.xyz"),
    ("PXBH_3", Path(__file__).parent / "bh28_benchmark" / "bh28_dataset" / "PXBH_3_ts.xyz"),
]


def _optimizer_kwargs(
    optimizer: str,
    trust_radius: float = 0.1,
    max_trust_radius: float = 0.3,
) -> dict[str, Any]:
    if optimizer == "sella":
        return {"internal": True, "order": 1}
    if optimizer == "rfo":
        return {
            "hessian_update_freq": 1,
            "trust_radius": trust_radius,
            "max_trust_radius": max_trust_radius,
        }
    return {
        "ts_search": True,
        "hessian_update_freq": 1,
        "trust_radius": trust_radius,
        "max_trust_radius": max_trust_radius,
        "use_bfgs_update": False,
    }


def _validate_ts(atoms: Atoms) -> dict[str, Any]:
    from famex.analysis.frequency import FrequencyAnalysis

    freq = FrequencyAnalysis(atoms=atoms, calculator=atoms.calc, verbose=0)
    freq.calculate_hessian(method="auto")
    freq.diagonalize_hessian()
    ts_info = freq.is_transition_state()
    return {
        "n_imaginary": int(ts_info.get("n_imaginary_frequencies", 0)),
        "is_valid_ts": bool(ts_info.get("is_transition_state", False)),
        "imaginary_frequencies": ts_info.get("imaginary_frequencies", []),
    }


def _run_local_ts(
    atoms: Atoms,
    optimizer: str,
    backend: str,
    device: str,
    fmax: float,
    steps: int,
    trust_radius: float = 0.1,
    max_trust_radius: float = 0.3,
) -> dict[str, Any]:
    from famex import Explorer

    atoms = atoms.copy()
    opt_kwargs = _optimizer_kwargs(optimizer, trust_radius, max_trust_radius)
    explorer = Explorer(
        atoms=atoms,
        backend=backend,
        device=device,
        target="ts",
        strategy="local",
        local_optimizer=optimizer,
        verbose=0,
        ts_kwargs=opt_kwargs if optimizer != "sella" else None,
    )
    explorer._create_and_attach_calculator(explorer.atoms_list[0])

    start = time.perf_counter()
    result = explorer.run(fmax=fmax, steps=steps, calculate_frequencies=True)
    elapsed = time.perf_counter() - start

    opt_atoms = result["optimized_atoms"]
    if not isinstance(opt_atoms, Atoms):
        raise TypeError("Expected single Atoms from local TS run")

    freq = result.get("frequency_analysis", {})
    return {
        "optimizer": optimizer,
        "converged": bool(result.get("converged", False)),
        "steps": int(result.get("steps_taken", 0)),
        "final_force": float(np.max(np.abs(opt_atoms.get_forces()))),
        "time_s": elapsed,
        "is_valid_ts": bool(freq.get("is_ts", result.get("is_ts", False))),
        "n_imaginary": int(
            freq.get("ts_analysis", {}).get("n_imaginary_frequencies", -1)
            if isinstance(freq.get("ts_analysis"), dict)
            else -1
        ),
        "trust_radius": trust_radius,
        "max_trust_radius": max_trust_radius,
    }


def _run_two_ended(
    reactant: Atoms,
    product: Atoms,
    strategy: str,
    optimizer: str,
    backend: str,
    device: str,
    fmax: float,
    steps: int,
    npoints: int,
) -> dict[str, Any]:
    from famex import Explorer

    explorer = Explorer(
        atoms=[reactant.copy(), product.copy()],
        backend=backend,
        device=device,
        target="ts",
        strategy=strategy,
        local_optimizer=optimizer,
        verbose=0,
        ts_kwargs=_optimizer_kwargs(optimizer) if optimizer != "sella" else None,
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
        "converged": bool(result.get("converged", False)),
        "steps": int(result.get("steps_taken", 0)),
        "final_force": float(np.max(np.abs(opt_atoms.get_forces()))),
        "time_s": elapsed,
        "strings_met": bool(result.get("strings_met", True)),
        "is_valid_ts": bool(freq.get("is_ts", result.get("is_ts", False))),
        "n_imaginary": int(ts_analysis.get("n_imaginary_frequencies", -1)),
    }


def _print_local_summary(results: list[dict[str, Any]], title: str) -> None:
    print(f"\n{'=' * 100}")
    print(title)
    print(f"{'=' * 100}")
    print(
        f"{'Optimizer':<15} {'Conv':<6} {'Steps':<7} {'Force':<10} "
        f"{'ValidTS':<10} {'Imag':<6} {'Time(s)':<8}"
    )
    print("-" * 100)
    for r in results:
        print(
            f"{r['optimizer']:<15} "
            f"{'Y' if r['converged'] else 'N':<6} "
            f"{r['steps']:<7} "
            f"{r['final_force']:<10.4f} "
            f"{'Y' if r['is_valid_ts'] else 'N':<10} "
            f"{r['n_imaginary']:<6} "
            f"{r['time_s']:<8.1f}"
        )


def _print_two_ended_summary(results: list[dict[str, Any]]) -> None:
    print(f"\n{'=' * 110}")
    print("TWO-ENDED TS (interpolate vs cineb vs growing_string, local optimizer = sella)")
    print(f"{'=' * 110}")
    print(
        f"{'Reaction':<14} {'Strategy':<16} {'Conv':<6} {'Steps':<7} {'Force':<10} "
        f"{'ValidTS':<10} {'Imag':<6} {'Time(s)':<8}"
    )
    print("-" * 110)
    for r in results:
        print(
            f"{r['reaction']:<14} "
            f"{r['strategy']:<16} "
            f"{'Y' if r['converged'] else 'N':<6} "
            f"{r['steps']:<7} "
            f"{r['final_force']:<10.4f} "
            f"{'Y' if r['is_valid_ts'] else 'N':<10} "
            f"{r['n_imaginary']:<6} "
            f"{r['time_s']:<8.1f}"
        )

    print("\nStrategy success rates (converged + valid TS):")
    for strategy in TWO_ENDED_STRATEGIES:
        subset = [r for r in results if r["strategy"] == strategy]
        ok = sum(1 for r in subset if r["converged"] and r["is_valid_ts"])
        total = len(subset)
        print(f"  {strategy:<16}: {ok}/{total} ({100 * ok / total if total else 0:.0f}%)")


def _run_trust_radius_sweep(
    backend: str,
    device: str,
    fmax: float,
    steps: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    print("\nRunning trust-radius sweep ...")
    for name, path in TRUST_SWEEP_STRUCTURES:
        if not path.exists():
            print(f"  Skipping {name}: {path} not found")
            continue
        atoms = read(str(path))
        for trust_radius, max_trust_radius in TRUST_RADIUS_PRESETS:
            for opt in TRUST_SWEEP_OPTIMIZERS:
                try:
                    row = _run_local_ts(
                        atoms,
                        opt,
                        backend,
                        device,
                        fmax,
                        steps,
                        trust_radius=trust_radius,
                        max_trust_radius=max_trust_radius,
                    )
                    row["structure"] = name
                    results.append(row)
                    print(
                        f"  {name}/{opt} r={trust_radius:.2f}: "
                        f"conv={row['converged']}, valid_ts={row['is_valid_ts']}, "
                        f"time={row['time_s']:.1f}s"
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


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU TS benchmark suite")
    parser.add_argument("--backend", default="uma")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument(
        "--output",
        default="examples/benchmark_results/ts_benchmark_suite_results.json",
    )
    parser.add_argument("--quick-two-ended", action="store_true", help="Use 3 Zimmermann reactions")
    parser.add_argument(
        "--two-ended-only",
        action="store_true",
        help="Skip local TS benchmarks and run two-ended section only",
    )
    parser.add_argument(
        "--trust-sweep",
        action="store_true",
        help="Run trust-radius preset sweep on selected structures and optimizers",
    )
    args = parser.parse_args()

    from famex.backends.registry import calculator_registry
    from famex.utils.device import get_optimal_device, print_device_info

    if not calculator_registry.is_backend_available(args.backend):
        print(f"Backend {args.backend} not available")
        return 1

    device = get_optimal_device(args.device)
    print_device_info(device)

    all_results: dict[str, Any] = {
        "backend": args.backend,
        "device": device,
        "local_example": [],
        "local_bh28": [],
        "two_ended": [],
        "trust_sweep": [],
    }

    if args.trust_sweep and not args.two_ended_only:
        all_results["trust_sweep"] = _run_trust_radius_sweep(
            args.backend,
            device,
            FMAX_LOCAL,
            MAX_STEPS_LOCAL,
        )

    # --- Local TS: example structure ---
    if not args.two_ended_only and not args.trust_sweep:
        example_ts = read(Path(__file__).parent / "example_files" / "A_C_A_B_A_C_ts.xyz")
        print("\nRunning local TS on example_files/A_C_A_B_A_C_ts.xyz ...")
        for opt in LOCAL_OPTIMIZERS:
            try:
                row = _run_local_ts(
                    example_ts, opt, args.backend, device, FMAX_LOCAL, MAX_STEPS_LOCAL
                )
                row["structure"] = "A_C_A_B_A_C_ts"
                all_results["local_example"].append(row)
                print(
                    f"  {opt}: conv={row['converged']}, valid_ts={row['is_valid_ts']}, time={row['time_s']:.1f}s"
                )
            except Exception as exc:
                all_results["local_example"].append({"optimizer": opt, "error": str(exc)})
                print(f"  {opt}: ERROR {exc}")

        _print_local_summary(
            [r for r in all_results["local_example"] if "error" not in r],
            "LOCAL TS — example structure",
        )

        # --- Local TS: BH28 subset ---
        bh28_dir = Path(__file__).parent / "bh28_benchmark" / "bh28_dataset"
        print("\nRunning local TS on BH28 subset ...")
        for reaction in BH28_TS_SUBSET:
            ts_path = bh28_dir / f"{reaction}_ts.xyz"
            if not ts_path.exists():
                continue
            atoms = read(str(ts_path))
            for opt in LOCAL_OPTIMIZERS:
                try:
                    row = _run_local_ts(
                        atoms, opt, args.backend, device, FMAX_LOCAL, MAX_STEPS_LOCAL
                    )
                    row["reaction"] = reaction
                    all_results["local_bh28"].append(row)
                    print(
                        f"  {reaction}/{opt}: conv={row['converged']}, "
                        f"valid_ts={row['is_valid_ts']}, time={row['time_s']:.1f}s"
                    )
                except Exception as exc:
                    all_results["local_bh28"].append(
                        {"reaction": reaction, "optimizer": opt, "error": str(exc)},
                    )
                    print(f"  {reaction}/{opt}: ERROR {exc}")

        print(f"\n{'=' * 100}")
        print("LOCAL TS — BH28 subset summary (converged + valid TS per optimizer)")
        print(f"{'=' * 100}")
        for opt in LOCAL_OPTIMIZERS:
            subset = [
                r
                for r in all_results["local_bh28"]
                if r.get("optimizer") == opt and "error" not in r
            ]
            ok = sum(1 for r in subset if r["converged"] and r["is_valid_ts"])
            avg_time = np.mean([r["time_s"] for r in subset]) if subset else 0.0
            avg_steps = np.mean([r["steps"] for r in subset]) if subset else 0.0
            print(
                f"  {opt:<15}: {ok}/{len(subset)} valid TS, "
                f"avg steps={avg_steps:.1f}, avg time={avg_time:.1f}s"
            )

    # --- Two-ended TS ---
    z93_dir = Path(__file__).parent / "zimmermann93_benchmark" / "zimmermann93_dataset"
    reactions = sorted(
        p.stem.replace("_reactant", "") for p in z93_dir.glob("reaction_*_reactant.*")
    )
    if args.quick_two_ended:
        reactions = reactions[:3]

    print(f"\nRunning two-ended TS on {len(reactions)} Zimmermann-93 reactions ...")
    two_ended_optimizer = "sella"
    for reaction in reactions:
        reactant = read(str(z93_dir / f"{reaction}_reactant.xyz"))
        product = read(str(z93_dir / f"{reaction}_product.xyz"))
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
                    args.backend,
                    device,
                    FMAX_TWO_ENDED,
                    MAX_STEPS_TWO_ENDED,
                    NPOINTS,
                )
                row["reaction"] = reaction
                all_results["two_ended"].append(row)
                print(
                    f"  {reaction}/{strategy}: conv={row['converged']}, "
                    f"valid_ts={row['is_valid_ts']}, time={row['time_s']:.1f}s"
                )
            except Exception as exc:
                all_results["two_ended"].append(
                    {"reaction": reaction, "strategy": strategy, "error": str(exc)},
                )
                print(f"  {reaction}/{strategy}: ERROR {exc}")

    _print_two_ended_summary([r for r in all_results["two_ended"] if "error" not in r])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
