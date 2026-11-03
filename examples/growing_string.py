#!/usr/bin/env python3
"""Growing-string (DE-GSM) transition-state search.

Uses a small Zimmermann H8B2N2 reaction that reliably yields a first-order
saddle with UMA. ``require_ts=True`` makes a missed saddle a hard failure.

Equivalent CLI::

    famex ts --strategy growing_string example_files/zimmermann_002_reactant.xyz
        --product example_files/zimmermann_002_product.xyz --freq --require-ts
"""

import sys
from pathlib import Path

from ase.io import read

import famex

HERE = Path(__file__).resolve().parent
REACTANT = HERE / "example_files" / "zimmermann_002_reactant.xyz"
PRODUCT = HERE / "example_files" / "zimmermann_002_product.xyz"
OUTPUT = HERE / "growing_string_ts.xyz"
PATH_OUTPUT = HERE / "growing_string_path.xyz"


def main() -> None:
    reactant = read(REACTANT)
    product = read(PRODUCT)
    explorer = famex.Explorer(
        atoms=[reactant, product],
        target="ts",
        strategy="growing_string",
        verbose=0,  # match CLI default quietness
    )
    result = explorer.run(
        npoints=11,
        fmax=0.05,
        steps=200,
        step_size=0.1,
        distance_threshold=0.5,
        calculate_frequencies=True,
        require_ts=True,
    )
    ts = result["optimized_atoms"]
    path = result.get("trajectory", [])
    explorer.save_structure(ts, str(OUTPUT))
    if path:
        explorer.save_trajectory(path, str(PATH_OUTPUT))
        print(f"Path images: {len(path)}")

    freq = result.get("frequency_analysis") or {}
    n_imag = (freq.get("ts_analysis") or {}).get("n_imaginary_frequencies")
    print(f"Imaginary modes: {n_imag}")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
