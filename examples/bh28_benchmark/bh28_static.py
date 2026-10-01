#!/usr/bin/env python3
"""Single-point BH28 barriers on the published reference geometries.

The reoptimization benchmark starts from these geometries and relaxes them.
This script evaluates the barrier on the geometries as published, with no
optimization, using the same reactant partitioning and the same reference
barriers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ase import Atoms
from bh28_benchmark import BH28Benchmark

from famex.backends.registry import calculator_registry

REPO = Path(__file__).resolve().parents[2]


def _energy(atoms: Atoms, calculator) -> float:
    image = atoms.copy()
    image.calc = calculator
    return float(image.get_potential_energy())


def static_barriers(backend: str, model_name: str | None, device: str) -> dict:
    benchmark = BH28Benchmark(dataset_dir=str(REPO / "examples/bh28_benchmark/bh28_dataset"))
    calculator = calculator_registry.create_calculator(
        backend, model_name=model_name, device=device
    )
    reactions = {}
    for name, reference in benchmark.reference_barriers.items():
        reactants = benchmark.get_reactants(name)
        ts = benchmark.get_transition_state(name)
        reactant_energy = sum(_energy(reactant, calculator) for reactant in reactants)
        ts_energy = _energy(ts, calculator)
        barrier = ts_energy - reactant_energy
        reactions[name] = {
            "reference_barrier": reference,
            "reactant_energy": reactant_energy,
            "ts_energy": ts_energy,
            "barrier_height": barrier,
            "absolute_error": barrier - reference,
            "n_reactants": len(reactants),
        }
    return {
        "backend": backend,
        "model_name": model_name,
        "device": device,
        "reactions": reactions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = static_barriers(args.backend, args.model, args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload))
    errors = [row["absolute_error"] for row in payload["reactions"].values()]
    print(
        f"{args.backend} {args.model or 'default'}: "
        f"{len(errors)} static barriers written to {output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
