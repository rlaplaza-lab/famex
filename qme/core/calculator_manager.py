"""Calculator management for QME Explorer.

This module provides the CalculatorManager class for handling calculator creation,
caching, and attachment to atoms objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ase import Atoms

from qme.backends.registry import create_calculator
from qme.core.charge_spin import check_missing_charge_spin, extract_charge_spin_from_atoms

if TYPE_CHECKING:
    pass


class CalculatorManager:
    """Manages calculator creation and attachment for Explorer.

    Handles calculator creation, caching, charge/spin extraction, and attachment
    to atoms objects. Provides centralized calculator management logic.
    """

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
        """Initialize CalculatorManager.

        Parameters
        ----------
        backend : str
            Calculator backend name
        model_name : str, optional
            Specific model name to use
        model_path : str, optional
            Path to local model file
        device : str, optional
            Device for computations ("cpu" or "cuda")
        default_charge : int, default 0
            Default charge when not specified in atoms
        default_spin : int, default 1
            Default spin multiplicity when not specified in atoms
        verbose : int, default 1
            Verbosity level for logging
        """
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
        """Get the effective model name that will actually be used by the backend.

        Returns the model name that will be used after applying backend-specific defaults.

        Returns
        -------
        str
            Effective model name
        """
        if self.model_name is not None:
            return self.model_name

        # Apply backend-specific defaults
        backend_lower = self.backend.lower()
        if backend_lower == "uma":
            return "uma-s-1p1"
        if backend_lower == "aimnet2":
            return "aimnet2"
        if backend_lower == "mace":
            return "mace-omol-0"
        if backend_lower == "so3lr":
            # SO3LR requires a model_path, not model_name
            return self.model_path or "so3lr-model"
        if backend_lower == "mock":
            return "mock-model"
        return "default-model"

    def create_and_attach_calculator(self, atoms: Atoms) -> Any:
        """Create and attach an ASE calculator to atoms.

        Prefers explicit charge/spin found on Geometry-like objects or
        in atoms.info. Falls back to defaults otherwise.

        Parameters
        ----------
        atoms : Atoms
            Atoms object to attach calculator to

        Returns
        -------
        Any
            The created calculator object
        """
        # Extract charge and spin using helper function
        charge, spin = extract_charge_spin_from_atoms(atoms, self.default_charge, self.default_spin)

        # Check if we're using defaults and warn once
        charge_missing, spin_missing = check_missing_charge_spin(atoms)

        if (charge_missing or spin_missing) and not self._warned_about_defaults:
            from qme.utils.logging import get_qme_logger

            logger = get_qme_logger(__name__)

            missing_parts = []
            if charge_missing:
                missing_parts.append(f"charge={charge}")
            if spin_missing:
                missing_parts.append(f"spin={spin}")

            # Use debug level instead of warning since defaults are reasonable
            # and charge/spin are often not critical for neutral closed-shell systems
            if self.verbose >= 2:
                logger.debug(
                    f"Charge and/or spin not specified in atoms. Using defaults: {', '.join(missing_parts)}",
                )
            else:
                # Only warn at verbose level 1 if it's a charged or open-shell system
                # (charge != 0 or spin != 1), otherwise it's just informational
                if charge != 0 or spin != 1:
                    logger.warning(
                        f"Charge and/or spin not specified in atoms. Using defaults: {', '.join(missing_parts)}",
                    )
                # For neutral closed-shell (charge=0, spin=1), don't warn at all
            self._warned_about_defaults = True

        # Ensure atoms.info contains values so calculators that read
        # atoms.info (UMA, SO3LR, etc.) see the intended settings.
        if getattr(atoms, "info", None) is not None:
            # Coerce to built-in int types to satisfy backends that enforce strict typing
            try:
                atoms.info["charge"] = int(atoms.info.get("charge", charge))
            except (ValueError, TypeError):
                atoms.info["charge"] = int(charge)
            try:
                atoms.info["spin"] = int(atoms.info.get("spin", spin))
            except (ValueError, TypeError):
                atoms.info["spin"] = int(spin)

        # Show model initialization info when creating the first calculator
        if not self._calculator_created:
            from qme.utils.logging import print_model_info

            # Get the effective model name that will actually be used
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
