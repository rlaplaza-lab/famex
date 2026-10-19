"""PySCF DFT backend (default: B3LYP-D3BJ / def2-SVP with density fitting).

Energy and forces only; Hessian via finite differences (like TBLite).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree

from famex.backends.constants import BACKEND_PYSCF, DEFAULT_PYSCF_MODEL
from famex.backends.dependencies import deps
from famex.potentials.base_potential import BasePotential

DEFAULT_XC, DEFAULT_BASIS = DEFAULT_PYSCF_MODEL.split("/", 1)


def parse_pyscf_model_name(model_name: str | None = None) -> tuple[str, str]:
    """Parse ``xc/basis`` (e.g. ``b3lyp-d3bj/def2-svp``) into ``(xc, basis)``."""
    name = (model_name or DEFAULT_PYSCF_MODEL).strip()
    if "/" not in name:
        return name or DEFAULT_XC, DEFAULT_BASIS
    xc, basis = name.split("/", 1)
    return (xc.strip() or DEFAULT_XC), (basis.strip() or DEFAULT_BASIS)


def xc_needs_dispersion(xc: str) -> bool:
    """Return True if the XC string requests a D3/D4 correction."""
    lower = xc.lower()
    return "-d3" in lower or "-d4" in lower


class PySCFPotential(BasePotential):
    """PySCF DFT calculator (energy and forces). Override with ``model_name="xc/basis"``."""

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        model_name: str | None = None,
        density_fit: bool = True,
        charge: int | None = None,
        mult: int | None = None,
        default_charge: int = 0,
        default_spin: int = 1,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        self.xc, self.basis = parse_pyscf_model_name(model_name)
        self.density_fit = density_fit
        self.charge = charge
        self.mult = mult
        self.default_charge = int(default_charge)
        self.default_spin = int(default_spin)
        self.verbose = int(verbose)
        self._calc: Any | None = None
        self._pyscf_gto: Any | None = None
        self._pyscf_dft: Any | None = None

        super().__init__(
            backend=BACKEND_PYSCF,
            model_name=f"{self.xc}/{self.basis}",
            **kwargs,
        )

    def _load_calculator(self) -> None:
        if self._calc is not None:
            return
        if not deps.has("pyscf"):
            msg = "PySCF is required for the pyscf backend. Install with: pip install famex[pyscf]"
            raise ImportError(msg)
        if xc_needs_dispersion(self.xc) and not deps.has("pyscf_dispersion"):
            msg = (
                f"XC '{self.xc}' requires dispersion corrections. "
                "Install with: pip install famex[pyscf]"
            )
            raise ImportError(msg)
        from pyscf import dft, gto

        self._pyscf_gto = gto
        self._pyscf_dft = dft
        self._calc = dft

    def _resolve_charge_mult(self, atoms: Atoms) -> tuple[int, int]:
        charge = (
            int(self.charge)
            if self.charge is not None
            else int(atoms.info.get("charge", self.default_charge))
        )
        mult = (
            int(self.mult)
            if self.mult is not None
            else int(atoms.info.get("spin", self.default_spin))
        )
        if mult < 1:
            msg = f"Spin multiplicity must be >= 1, got {mult}"
            raise ValueError(msg)
        return charge, mult

    def _run_pyscf(self, atoms: Atoms, charge: int, mult: int) -> tuple[float, np.ndarray]:
        gto = self._pyscf_gto
        dft = self._pyscf_dft
        if gto is None or dft is None:
            msg = "PySCF modules not loaded"
            raise RuntimeError(msg)

        spin = mult - 1
        verbose = 4 if self.verbose >= 3 else 0

        mol = gto.Mole()
        mol.atom = [
            (sym, tuple(float(x) for x in pos))
            for sym, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions(), strict=True)
        ]
        mol.basis = self.basis
        mol.charge = charge
        mol.spin = spin
        mol.unit = "Angstrom"
        mol.verbose = verbose
        mol.build()

        mf = dft.RKS(mol) if spin == 0 else dft.UKS(mol)
        mf.xc = self.xc
        mf.verbose = verbose
        if self.density_fit:
            mf = mf.density_fit()

        energy_ha = float(mf.kernel())
        if not mf.converged:
            msg = f"PySCF SCF did not converge (xc={self.xc}, basis={self.basis})"
            raise RuntimeError(msg)

        grad = np.asarray(mf.nuc_grad_method().kernel(), dtype=float)
        return energy_ha * Hartree, -grad * (Hartree / Bohr)

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: Sequence[str] | None = None,
        system_changes: Any = None,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        if self.atoms is None:
            msg = "No atoms provided for PySCF calculation"
            raise ValueError(msg)

        self._require_calc()
        charge, mult = self._resolve_charge_mult(self.atoms)
        energy, forces = self._run_pyscf(self.atoms, charge, mult)
        self.results["energy"] = energy
        self.results["forces"] = forces
