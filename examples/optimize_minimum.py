#!/usr/bin/env python3
"""Local energy minimization of a reactant geometry.

Equivalent CLI::

    famex minima --strategy local example_files/A_C_A_B_A_C_reactant.xyz --freq
"""

import math
from pathlib import Path
from typing import Any

from ase.io import read

import famex

HERE = Path(__file__).resolve().parent
REACTANT = HERE / "example_files" / "A_C_A_B_A_C_reactant.xyz"
OUTPUT = HERE / "optimized_minimum.xyz"


def _require_frequencies(frequency_analysis: Any) -> None:
    frequencies = (
        frequency_analysis.get("frequencies") if isinstance(frequency_analysis, dict) else None
    )
    usable = (
        isinstance(frequencies, list)
        and len(frequencies) > 0
        and all(isinstance(value, int | float) and math.isfinite(value) for value in frequencies)
    )
    if not usable:
        raise SystemExit("Frequency analysis did not return finite frequencies")


def main() -> None:
    atoms = read(REACTANT)
    explorer = famex.Explorer(atoms=atoms, target="minima", strategy="local")
    result = explorer.run(fmax=0.05, steps=500, calculate_frequencies=True)
    optimized = result["optimized_atoms"]
    explorer.save_structure(optimized, str(OUTPUT))
    energy = float(optimized.get_potential_energy())
    converged = bool(result.get("converged"))
    print(f"Final energy: {energy:.6f} eV")
    print(f"Converged: {converged}")
    print(f"Wrote {OUTPUT}")
    _require_frequencies(result.get("frequency_analysis"))
    if not converged or not math.isfinite(energy):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
