"""
Mock potential moved into qme.potentials
"""

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.data import atomic_numbers, covalent_radii


class MockCalculator(Calculator):
    implemented_properties = ["energy", "forces", "hessian"]

    BACKEND_CONFIGS = {
        "uma": {"force_constant": 5.0, "name": "MockUMA"},
        "aimnet2": {"force_constant": 5.0, "name": "MockAIMNet2"},
        "so3lr": {"force_constant": 5.0, "name": "MockSO3LR"},
        "mace": {"force_constant": 5.0, "name": "MockMACE"},
        "generic": {"force_constant": 5.0, "name": "MockGeneric"},
    }

    def __init__(self, backend="generic", charge=0, mult=1, **kwargs):
        Calculator.__init__(self, **kwargs)

        if backend not in self.BACKEND_CONFIGS:
            backend = "generic"

        config = self.BACKEND_CONFIGS[backend].copy()
        config.update({"charge": charge, "mult": mult})
        config.update(kwargs)

        self.backend = backend
        self.force_constant = config["force_constant"]
        self.charge = config["charge"]
        self.mult = config["mult"]

    def calculate(
        self, atoms=None, properties=["energy", "forces"], system_changes=all_changes
    ):
        if atoms is None:
            raise ValueError("Atoms object must be provided")
        # Simple zero energy for placeholder
        if "energy" in properties:
            self.results["energy"] = 0.0
        if "forces" in properties:
            self.results["forces"] = np.zeros_like(atoms.positions)
