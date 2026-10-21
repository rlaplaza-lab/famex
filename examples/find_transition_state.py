#!/usr/bin/env python3
"""Refine a transition-state guess with local TS optimization.

Uses ``require_ts=True`` so the script fails (non-zero exit) unless a
first-order saddle is confirmed by frequency analysis.

Equivalent CLI::

    famex ts --strategy local example_files/A_C_A_B_A_C_ts.xyz --freq --require-ts
"""

from pathlib import Path

from ase.io import read

import famex

HERE = Path(__file__).resolve().parent
TS_GUESS = HERE / "example_files" / "A_C_A_B_A_C_ts.xyz"
OUTPUT = HERE / "found_ts.xyz"


def main() -> None:
    ts_guess = read(TS_GUESS)
    explorer = famex.Explorer(
        atoms=ts_guess,
        target="ts",
        strategy="local",
        verbose=0,  # match CLI default quietness
    )
    result = explorer.run(
        fmax=0.05,
        steps=200,
        calculate_frequencies=True,
        require_ts=True,
    )
    ts = result["optimized_atoms"]
    explorer.save_structure(ts, str(OUTPUT))

    freq = result.get("frequency_analysis") or {}
    n_imag = (freq.get("ts_analysis") or {}).get("n_imaginary_frequencies")
    print(f"Imaginary modes: {n_imag}")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
