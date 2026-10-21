#!/usr/bin/env python3
"""Local energy minimization of a reactant geometry.

Equivalent CLI::

    famex minima --strategy local example_files/A_C_A_B_A_C_reactant.xyz --freq
"""

from pathlib import Path

from ase.io import read

import famex

HERE = Path(__file__).resolve().parent
REACTANT = HERE / "example_files" / "A_C_A_B_A_C_reactant.xyz"
OUTPUT = HERE / "optimized_minimum.xyz"


def main() -> None:
    atoms = read(REACTANT)
    explorer = famex.Explorer(atoms=atoms, target="minima", strategy="local")
    result = explorer.run(fmax=0.05, steps=500, calculate_frequencies=True)
    optimized = result["optimized_atoms"]
    explorer.save_structure(optimized, str(OUTPUT))
    energy = optimized.get_potential_energy()
    print(f"Final energy: {energy:.6f} eV")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
