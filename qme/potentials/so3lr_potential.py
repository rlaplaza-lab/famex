"""SO3LR Neural Network Potential integration for ASE.

SO3LR is an open source neural network potential with SO(3) invariant architecture.
This module provides ASE Calculator interface for SO3LR models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from ase.calculators.calculator import all_changes

from qme.backends.dependencies import deps
from qme.potentials.base_potential import BasePotential
from qme.utils.logging import get_qme_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase import Atoms

logger = get_qme_logger(__name__)


class SO3LRPotential(BasePotential):
    """ASE Calculator interface for SO3LR neural network potential.

    This is a wrapper around the native SO3LR ASE calculator to provide
    compatibility with the QME interface.
    """

    implemented_properties = ["energy", "forces"]

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
        # Standard backend attribute used by BasePotential helpers
        self._calc: Any | None = None

        # Initialize base class (this will call _load_calculator)
        super().__init__(model_name=model_name, device=device, **kwargs)

    def _load_calculator(self) -> None:
        """Load the SO3LR ASE calculator."""
        # Skip if already loaded
        if self._calc is not None:
            return
        # After this point, we know _calc is None, so we need to load it

        from qme.utils.ml_warnings import quiet_backend_loading

        # Don't show model info - let the outer context handle it
        with quiet_backend_loading(
            "so3lr",
            self.model_name,
            self.model_path,
            self.device,
            show_model_info=False,
        ):
            # Get SO3LR module
            so3lr = deps.get("so3lr")
            if so3lr is None:
                logger.error("SO3LR module not available")
                msg = "SO3LR module not available"
                raise RuntimeError(msg)

        # Create SO3LR calculator with appropriate parameters
        # Use high cutoff for gas-phase systems as recommended
        lr_cutoff = 1000.0

        self._calc = so3lr.So3lrCalculator(
            calculate_stress=False,
            lr_cutoff=lr_cutoff,
            dtype=np.float32,
        )

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

        # Ensure atoms has charge information as required by SO3LR
        if "charge" not in atoms.info:
            atoms.info["charge"] = 0.0

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

    def get_potential_energy(
        self,
        atoms: Atoms | None = None,
        force_consistent: bool = False,
    ) -> float:
        """Get potential energy (ASE-compatible)."""
        if atoms is not None:
            self.atoms = atoms

        return super().get_potential_energy(atoms, force_consistent)

    def get_forces(self, atoms: Atoms | None = None) -> np.ndarray | None:
        """Get forces (ASE-compatible)."""
        if atoms is not None:
            self.atoms = atoms

        return super().get_forces(atoms)

    def _get_backend_name(self) -> str:
        """Get the backend name for SO3LR."""
        return "so3lr"


def get_so3lr_calculator(
    model_path: str | None = None,
    model_name: str = "so3lr-small",
    device: str | None = None,
    **kwargs: Any,
) -> SO3LRPotential:
    """Convenience function to get SO3LR calculator.

    Parameters
    ----------
    model_path : str, optional
        Path to trained SO3LR model file (currently not used by SO3LR)
    model_name : str
        Name of pre-trained SO3LR model (currently not used by SO3LR)
    device : str, optional
        Device preference ('cpu', 'cuda')
    **kwargs :
        Additional arguments passed to SO3LRPotential

    Returns:
    -------
    SO3LRPotential
        Configured SO3LR calculator

    """
    return SO3LRPotential(model_path=model_path, model_name=model_name, device=device, **kwargs)
