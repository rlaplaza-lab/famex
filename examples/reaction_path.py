#!/usr/bin/env python3
"""Nudged elastic band path between reactant and product.

Equivalent CLI::

    famex path --strategy neb example_files/A_C_A_B_A_C_reactant.xyz
        example_files/A_C_A_B_A_C_product.xyz --npoints 5 --steps 300
"""

from pathlib import Path

from ase.io import read

import famex

HERE = Path(__file__).resolve().parent
REACTANT = HERE / "example_files" / "A_C_A_B_A_C_reactant.xyz"
PRODUCT = HERE / "example_files" / "A_C_A_B_A_C_product.xyz"
OUTPUT = HERE / "neb_path.xyz"


def main() -> None:
    reactant = read(REACTANT)
    product = read(PRODUCT)
    explorer = famex.Explorer(
        atoms=[reactant, product],
        target="path",
        strategy="neb",
        verbose=0,  # match CLI default quietness
    )
    result = explorer.run(npoints=5, fmax=0.05, steps=300, spring_constant=0.5)
    path = result.get("trajectory", result["optimized_atoms"])
    explorer.save_trajectory(path, str(OUTPUT))
    converged = bool(result.get("converged"))
    print(f"Path images: {len(path)}")
    print(f"Converged: {converged}")
    print(f"Wrote {OUTPUT}")
    if not converged:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
