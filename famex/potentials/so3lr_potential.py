"""SO3LR Neural Network Potential integration for ASE.

SO3LR is an open source neural network potential with SO(3) invariant architecture.
This module provides ASE Calculator interface for SO3LR models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from ase.calculators.calculator import all_changes

from famex.backends.dependencies import deps
from famex.potentials.base_potential import BasePotential
from famex.utils.logging import get_famex_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase import Atoms

logger = get_famex_logger(__name__)


class SO3LRPotential(BasePotential):
    """ASE Calculator interface for SO3LR neural network potential.

    This is a wrapper around the native SO3LR ASE calculator to provide
    compatibility with the FAMEX interface.
    """

    implemented_properties = ["energy", "forces", "hessian"]

    def __init__(
        self,
        model_path: str | None = None,
        model_name: str = "so3lr-small",
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize SO3LR potential calculator.

        Parameters
        ----------
        model_path : str, optional
            Path to trained SO3LR model file (currently not used by SO3LR)
        model_name : str
            Name of pre-trained SO3LR model (default: "so3lr-small")
        device : str, optional
            Device to run computations on ('cpu', 'cuda'). Auto-detected if None.

        """
        if not deps.has("so3lr"):
            msg = (
                "SO3LR is required for SO3LR potentials. "
                "Install SO3LR with: git clone "
                "https://github.com/general-molecular-simulations/so3lr.git && "
                "cd so3lr && pip install ."
            )
            raise ImportError(
                msg,
            )

        # Store additional SO3LR-specific parameters
        self.model_path = model_path

        # SO3LR-specific attributes
        self._calc: Any | None = None
        self._hessian_calc: Any | None = None

        super().__init__(
            backend="so3lr",
            model_name=model_name,
            device=device,
            **kwargs,
        )

    @staticmethod
    def _ensure_default_charge(atoms: Atoms) -> None:
        if "charge" not in atoms.info:
            atoms.info["charge"] = 0.0

    def _create_so3lr_calculator(self, *, calculate_hessian: bool, dtype: Any) -> Any:
        so3lr = deps.get("so3lr")
        if so3lr is None:
            logger.error("SO3LR module not available")
            msg = "SO3LR module not available"
            raise RuntimeError(msg)

        # lr_cutoff=1000 Å is the SO3LR recommendation for gas-phase systems.
        return so3lr.So3lrCalculator(
            calculate_stress=False,
            calculate_hessian=calculate_hessian,
            lr_cutoff=1000.0,
            dtype=dtype,
        )

    def _load_calculator(self) -> None:
        """Load the SO3LR ASE calculator."""
        # Skip if already loaded
        if self._calc is not None:
            return
        # After this point, we know _calc is None, so we need to load it

        from famex.utils.ml_warnings import quiet_backend_loading

        # Don't show model info - let the outer context handle it
        with quiet_backend_loading(
            "so3lr",
            self.model_name,
            self.model_path,
            self.device,
            show_model_info=False,
        ):
            self._calc = self._create_so3lr_calculator(
                calculate_hessian=False,
                dtype=np.float32,
            )

    def _ensure_hessian_calculator(self) -> Any:
        if self._hessian_calc is None:
            self._hessian_calc = self._create_so3lr_calculator(
                calculate_hessian=True,
                dtype=np.float64,
            )
        return self._hessian_calc

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: Sequence[str] | None = None,
        system_changes: Any = all_changes,
    ) -> None:
        """Calculate properties using SO3LR potential."""
        if atoms is None:
            return

        super().calculate(atoms, properties, system_changes)

        self._ensure_default_charge(atoms)

        # Ensure calculator is loaded
        if self._calc is None:
            self._load_calculator()
        # After _load_calculator() returns without exception, _calc is guaranteed to be set
        assert self._calc is not None
        # External library call can raise exceptions even with valid object:
        # RuntimeError may occur due to calculation failures (convergence, numerical issues, etc.)
        self._calc.calculate(atoms, properties, system_changes)

        # Extract results from the underlying calculator
        if properties is not None and "energy" in properties:
            results = getattr(self._calc, "results", None)
            if isinstance(results, dict):
                self.results["energy"] = results.get("energy", self.results.get("energy"))
            else:
                self.results["energy"] = self.results.get("energy")

        if properties is not None and "forces" in properties:
            results = getattr(self._calc, "results", None)
            if isinstance(results, dict):
                self.results["forces"] = results.get("forces", self.results.get("forces"))
            else:
                self.results["forces"] = self.results.get("forces")

    def get_hessian(self, atoms: Atoms | None = None) -> np.ndarray:
        """Analytical Hessian (3N x 3N, eV/Å²) from SO3LR's JAX second derivatives."""
        if atoms is not None:
            self.atoms = atoms
        if self.atoms is None:
            msg = "No atoms provided for Hessian calculation"
            raise ValueError(msg)

        self._ensure_default_charge(self.atoms)

        hessian_calc = self._ensure_hessian_calculator()
        hessian_calc.calculate(self.atoms, ["energy", "forces", "hessian"], all_changes)
        n = 3 * len(self.atoms)
        hessian = np.asarray(hessian_calc.results["hessian"], dtype=np.float64).reshape(n, n)
        hessian = cast(np.ndarray, 0.5 * (hessian + hessian.T))

        self.results["hessian"] = hessian
        return hessian

    def get_property(
        self, prop: str, atoms: Atoms | None = None, allow_calculation: bool = True
    ) -> Any:
        """Get energy, forces, or the analytical Hessian."""
        del allow_calculation
        if atoms is not None:
            self.atoms = atoms
        if prop == "energy":
            return self.get_potential_energy(atoms)
        if prop == "forces":
            return self.get_forces(atoms)
        if prop == "hessian":
            return self.get_hessian(atoms)
        msg = f"Property '{prop}' not supported by SO3LRPotential"
        raise KeyError(msg)
