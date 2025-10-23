#!/usr/bin/env python3
"""Run Trust-Krylov TS vs Sella comparisons on the full BH28 dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read

from qme.backends.dependencies import deps
from qme.optimizers.scipy_optimizers import TrustKrylovTS
from qme.potentials.uma_potential import get_uma_calculator

DATASET_DIR = Path(__file__).resolve().parents[1] / "examples" / "bh28_benchmark" / "bh28_dataset"
REFERENCE_FILE = DATASET_DIR / "reference_barrier_heights.json"
REPORT_DIR = (
    Path(__file__).resolve().parents[1] / "examples" / "bh28_benchmark" / "benchmark_results"
)
REPORT_PATH = REPORT_DIR / "trust_krylov_ts_vs_sella_bh28.json"
FORCE_THRESHOLD = 0.25
FMAX_TARGET = 0.18
MAX_STEPS = 24


def _max_force(atoms) -> float:
    return float(np.max(np.abs(atoms.get_forces())))


def _load_ts_atoms(reaction: str):
    atoms = read(str(DATASET_DIR / f"{reaction}_ts.xyz"))
    if isinstance(atoms, list):
        atoms = atoms[0]
    return atoms


def _attach_calculator(atoms):
    atoms.calc = get_uma_calculator()
    return atoms


def _run_sella(base_atoms) -> tuple[dict[str, Any], np.ndarray | None]:
    atoms = base_atoms.copy()
    _attach_calculator(atoms)
    from sella import Sella  # Imported lazily to honour dependency checks

    optimizer = Sella(atoms, internal=True, order=1)
    record: dict[str, Any] = {}
    try:
        optimizer.run(fmax=FMAX_TARGET, steps=MAX_STEPS)
        record["converged"] = bool(optimizer.converged())
    except Exception as exc:  # noqa: BLE001
        record["converged"] = False
        record["error"] = str(exc)
    record["max_force"] = _max_force(atoms)
    record["energy"] = float(atoms.get_potential_energy())
    record["steps"] = getattr(optimizer, "nsteps", None)
    return record, atoms.get_positions() if atoms is not None else None


def _run_trust_krylov(base_atoms) -> tuple[dict[str, Any], np.ndarray | None]:
    atoms = base_atoms.copy()
    _attach_calculator(atoms)

    optimizer = TrustKrylovTS(
        atoms,
        hessian_update_freq=1,
        adaptive_hessian=False,
        negative_mode_boost=8e-3,
        min_positive_eigenvalue=4e-3,
        index_tolerance=5e-4,
        mode_recompute_interval=1,
    )
    record: dict[str, Any] = {}
    try:
        record["converged"] = bool(optimizer.run(fmax=FMAX_TARGET, steps=MAX_STEPS))
    except Exception as exc:  # noqa: BLE001
        record["converged"] = False
        record["error"] = str(exc)
    record["max_force"] = _max_force(atoms)
    record["energy"] = float(atoms.get_potential_energy())
    record["steps"] = optimizer.get_number_of_steps()
    return record, atoms.get_positions() if atoms is not None else None


def _prepare_report_dir() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    if not DATASET_DIR.exists():
        msg = f"BH28 dataset not found at {DATASET_DIR}"
        raise SystemExit(msg)
    if not deps.has("sella"):
        msg = "Sella dependency is required for this comparison"
        raise SystemExit(msg)

    with REFERENCE_FILE.open() as handle:
        reference_data = json.load(handle)

    reactions = sorted(reference_data.keys())
    summary: dict[str, Any] = {
        "fmax_target": FMAX_TARGET,
        "max_steps": MAX_STEPS,
        "force_success_threshold": FORCE_THRESHOLD,
        "reactions": {},
        "success_counts": {
            "sella": 0,
            "trust_krylov_ts": 0,
        },
    }

    energy_diffs = []
    rmsd_stats = []

    for reaction in reactions:
        base_atoms = _load_ts_atoms(reaction)

        baseline_atoms = base_atoms.copy()
        _attach_calculator(baseline_atoms)
        initial_force = _max_force(baseline_atoms)
        initial_energy = float(baseline_atoms.get_potential_energy())

        sella_result, sella_positions = _run_sella(base_atoms)
        trust_result, trust_positions = _run_trust_krylov(base_atoms)

        sella_result["initial_force"] = initial_force
        sella_result["initial_energy"] = initial_energy
        trust_result["initial_force"] = initial_force
        trust_result["initial_energy"] = initial_energy

        sella_result["force_reduction"] = initial_force - sella_result["max_force"]
        trust_result["force_reduction"] = initial_force - trust_result["max_force"]

        sella_result["success"] = (
            sella_result["max_force"] < FORCE_THRESHOLD
            and sella_result["max_force"] <= 0.95 * initial_force
        )
        trust_result["success"] = (
            trust_result["max_force"] < FORCE_THRESHOLD
            and trust_result["max_force"] <= 0.95 * initial_force
        )

        if sella_result["success"]:
            summary["success_counts"]["sella"] += 1
        if trust_result["success"]:
            summary["success_counts"]["trust_krylov_ts"] += 1

        position_rmsd = None
        if sella_positions is not None and trust_positions is not None:
            diff = sella_positions - trust_positions
            position_rmsd = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
            rmsd_stats.append(position_rmsd)

        if sella_result["success"] and trust_result["success"]:
            energy_diffs.append(abs(trust_result["energy"] - sella_result["energy"]))

        summary["reactions"][reaction] = {
            "initial_force": round(initial_force, 6),
            "initial_energy": round(initial_energy, 6),
            "sella": {
                key: (round(val, 6) if isinstance(val, float) else val)
                for key, val in sella_result.items()
            },
            "trust_krylov_ts": {
                key: (round(val, 6) if isinstance(val, float) else val)
                for key, val in trust_result.items()
            },
            "position_rmsd": None if position_rmsd is None else round(position_rmsd, 6),
        }

    if energy_diffs:
        summary["energy_diff_stats"] = {
            "mean_abs_difference": float(np.mean(energy_diffs)),
            "median_abs_difference": float(np.median(energy_diffs)),
            "max_abs_difference": float(np.max(energy_diffs)),
        }
    else:
        summary["energy_diff_stats"] = None

    if rmsd_stats:
        summary["position_rmsd_stats"] = {
            "mean": float(np.mean(rmsd_stats)),
            "median": float(np.median(rmsd_stats)),
            "max": float(np.max(rmsd_stats)),
        }
    else:
        summary["position_rmsd_stats"] = None

    _prepare_report_dir()
    with REPORT_PATH.open("w") as handle:
        json.dump(summary, handle, indent=2)

    if summary["energy_diff_stats"]:
        summary["energy_diff_stats"]
    if summary["position_rmsd_stats"]:
        summary["position_rmsd_stats"]

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
