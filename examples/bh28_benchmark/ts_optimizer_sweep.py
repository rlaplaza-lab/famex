#!/usr/bin/env python3
"""Compare TS optimizers on the BH28 benchmark subset (UMA backend)."""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
from ase.io import read

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Use unbuffered stdout for long runs
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

BH28_TS_SUBSET = [
    "BHDIV_3",
    "PXBH_3",
    "CADBH_2",
    "CRBH_1",
    "PXBH_2",
    "BHPERI_2",
]

OPTIMIZERS = [
    "sella",
    "rfo",
    "trust-krylov",
    "trust-ncg",
    "trust-exact",
]

FMAX = 0.05
MAX_STEPS = 500


def _max_force(atoms) -> float:
    return float(np.max(np.abs(atoms.get_forces())))


def _initial_ts_info(atoms) -> tuple[int, float]:
    from famex.analysis.frequency import FrequencyAnalysis

    freq = FrequencyAnalysis(atoms=atoms, calculator=atoms.calc, verbose=0)
    freq.calculate_hessian(method="auto")
    freq.diagonalize_hessian()
    result = freq.is_transition_state()
    n_imag = int(result.get("n_imaginary_frequencies", 0))
    return n_imag, _max_force(atoms)


def _validate_ts(atoms) -> tuple[bool, int]:
    from famex.analysis.frequency import FrequencyAnalysis

    freq = FrequencyAnalysis(atoms=atoms, calculator=atoms.calc, verbose=0)
    freq.calculate_hessian(method="auto")
    freq.diagonalize_hessian()
    result = freq.is_transition_state()
    n_imag = int(result.get("n_imaginary_frequencies", 0))
    return n_imag == 1, n_imag


def _load_reaction(dataset_dir: Path, reaction: str):
    atoms = read(str(dataset_dir / f"{reaction}_ts.xyz"))
    if isinstance(atoms, list):
        atoms = atoms[0]
    from famex.potentials import get_uma_calculator

    atoms.calc = get_uma_calculator()
    return atoms


def _run_optimizer(atoms, optimizer_name: str):
    from famex.strategies.helpers import _get_local_optimizer_class

    calculator = atoms.calc
    atoms = atoms.copy()
    atoms.calc = calculator
    opt_class = _get_local_optimizer_class(optimizer_name)

    kwargs: dict = {"logfile": None, "verbose": 0}
    if optimizer_name == "sella":
        kwargs.update({"internal": True, "order": 1})
    elif optimizer_name == "rfo":
        kwargs.update(
            {
                "hessian_update_freq": 1,
                "trust_radius": 0.1,
                "max_trust_radius": 0.3,
            },
        )
    else:
        kwargs.update(
            {
                "ts_search": True,
                "hessian_update_freq": 1,
                "trust_radius": 0.1,
                "max_trust_radius": 0.3,
                "use_bfgs_update": False,
            },
        )

    start = time.perf_counter()
    opt = opt_class(atoms, **kwargs)
    converged = opt.run(fmax=FMAX, steps=MAX_STEPS)
    elapsed = time.perf_counter() - start
    steps = getattr(opt, "nsteps", getattr(opt, "get_number_of_steps", lambda: 0)())
    hessian_calls = getattr(opt, "hessian_calls", None)
    return {
        "converged": bool(converged),
        "steps": int(steps),
        "hessian_calls": hessian_calls,
        "final_force": _max_force(atoms),
        "time": elapsed,
        "atoms": atoms,
    }


def main() -> int:
    dataset_dir = Path(__file__).parent / "bh28_dataset"
    if not dataset_dir.exists():
        print(f"Dataset not found: {dataset_dir}")
        return 1

    print(
        f"{'Reaction':<12} {'Optimizer':<15} {'Conv':<6} {'Steps':<7} {'Hess':<6} "
        f"{'Force':<10} {'InitImag':<9} {'ValidTS':<8} {'Time(s)':<8}"
    )
    print("=" * 90)

    summary: dict[str, dict[str, int]] = {opt: {"ok": 0, "total": 0} for opt in OPTIMIZERS}

    for reaction in BH28_TS_SUBSET:
        ts_file = dataset_dir / f"{reaction}_ts.xyz"
        if not ts_file.exists():
            print(f"Skipping missing {reaction}")
            continue

        for optimizer in OPTIMIZERS:
            try:
                atoms = _load_reaction(dataset_dir, reaction)
                init_n_imag, _ = _initial_ts_info(atoms)
                result = _run_optimizer(atoms, optimizer)
                valid, n_imag = _validate_ts(result["atoms"])
                summary[optimizer]["total"] += 1
                if result["converged"] and valid:
                    summary[optimizer]["ok"] += 1

                hess_str = (
                    str(result["hessian_calls"]) if result["hessian_calls"] is not None else "-"
                )
                print(
                    f"{reaction:<12} {optimizer:<15} "
                    f"{'Y' if result['converged'] else 'N':<6} "
                    f"{result['steps']:<7} {hess_str:<6} "
                    f"{result['final_force']:<10.4f} "
                    f"{init_n_imag:<9} "
                    f"{'Y' if valid else f'N({n_imag})':<8} "
                    f"{result['time']:<8.1f}",
                )
            except Exception as exc:
                print(
                    f"{reaction:<12} {optimizer:<15} ERROR  -      -      -        -        {exc}"
                )

    print("\nSUMMARY (converged + valid TS):")
    for optimizer, stats in summary.items():
        total = stats["total"]
        ok = stats["ok"]
        rate = 100.0 * ok / total if total else 0.0
        print(f"  {optimizer:<15}: {ok}/{total} ({rate:.0f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
