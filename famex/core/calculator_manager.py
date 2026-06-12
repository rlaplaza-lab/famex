"""Calculator management for FAMEX Explorer."""

from __future__ import annotations

from typing import Any

from ase import Atoms

from famex.backends.registry import create_calculator
from famex.core.charge_spin import check_missing_charge_spin, extract_charge_spin_from_atoms


class CalculatorManager:
    """Manages calculator creation and attachment for Explorer."""

    def __init__(
        self,
        backend: str,
        model_name: str | None = None,
        model_path: str | None = None,
        device: str | None = None,
        default_charge: int = 0,
        default_spin: int = 1,
        verbose: int = 1,
    ) -> None:
        self.backend = backend
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.default_charge = default_charge
        self.default_spin = default_spin
        self.verbose = verbose
        self._calculator_created = False
        self._warned_about_defaults = False

    def get_effective_model_name(self) -> str:
        if self.model_name is not None:
            return self.model_name

        backend_lower = self.backend.lower()
        if backend_lower == "uma":
            return "uma-s-1p2"
        if backend_lower == "aimnet2":
            return "aimnet2"
        if backend_lower == "mace":
            return "mace-omol-0"
        if backend_lower == "so3lr":
            return self.model_path or "so3lr-model"
        if backend_lower == "mock":
            return "mock-model"
        return "default-model"

    def create_and_attach_calculator(self, atoms: Atoms) -> Any:
        charge, spin = extract_charge_spin_from_atoms(atoms, self.default_charge, self.default_spin)
        charge_missing, spin_missing = check_missing_charge_spin(atoms)

        if (charge_missing or spin_missing) and not self._warned_about_defaults:
            from famex.utils.logging import get_famex_logger

            logger = get_famex_logger(__name__)

            missing_parts = []
            if charge_missing:
                missing_parts.append(f"charge={charge}")
            if spin_missing:
                missing_parts.append(f"spin={spin}")

            if self.verbose >= 2:
                logger.debug(
                    f"Charge and/or spin not specified in atoms. Using defaults: {', '.join(missing_parts)}",
                )
            elif charge != 0 or spin != 1:
                logger.warning(
                    f"Charge and/or spin not specified in atoms. Using defaults: {', '.join(missing_parts)}",
                )
            self._warned_about_defaults = True

        if getattr(atoms, "info", None) is not None:
            try:
                atoms.info["charge"] = int(atoms.info.get("charge", charge))
            except (ValueError, TypeError):
                atoms.info["charge"] = int(charge)
            try:
                atoms.info["spin"] = int(atoms.info.get("spin", spin))
            except (ValueError, TypeError):
                atoms.info["spin"] = int(spin)

        if not self._calculator_created:
            from famex.utils.logging import print_model_info

            effective_model_name = self.get_effective_model_name()
            print_model_info(
                self.backend,
                effective_model_name,
                self.model_path,
                self.device,
                verbose=self.verbose,
            )
            self._calculator_created = True

        calc = create_calculator(
            backend=self.backend,
            model_name=self.model_name,
            model_path=self.model_path,
            device=self.device,
            default_charge=self.default_charge,
            default_spin=self.default_spin,
            charge=charge,
            mult=spin,
            verbose=self.verbose,
        )
        atoms.calc = calc
        return calc
