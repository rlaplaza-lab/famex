"""MACE Machine Learning Potential integration for ASE.

This module implements a MACE calculator integration using the MACE-OMOL-0
foundation model for molecular systems, transition metals, and cations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from famex.backends.constants import DEFAULT_MACE_MODEL
from famex.backends.dependencies import deps
from famex.backends.mace_compat import format_mace_e3nn_conflict_message, is_mace_e3nn_error
from famex.potentials._load_utils import raise_backend_load_error
from famex.potentials.base_potential import BasePotential
from famex.utils.hessian_ops import symmetrize_hessian
from famex.utils.logging import get_famex_logger
from famex.utils.ml_warnings import quiet_backend_loading

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase import Atoms

logger = get_famex_logger(__name__)


class MACEPotential(BasePotential):
    """MACE potential calculator using foundation models.

    This calculator provides access to MACE foundation models, particularly
    the MACE-OMOL-0 model which is excellent for molecules, transition metals,
    and cations with charge/spin embedding capabilities.

    Supports analytical Hessian calculations for efficient frequency analysis.
    """

    implemented_properties = ["energy", "forces", "hessian"]

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize MACE potential calculator.

        Parameters
        ----------
        model_name : str, optional
            MACE model to use. Defaults to ``DEFAULT_MACE_MODEL``.
            Available options:
            - "mace-omol-0": Large model for molecules/transition metals/cations
            - "mace-mp-0": Materials Project models (small, medium, large)
            - "mace-off23": Organic chemistry models (small, medium, large)
        device : str, optional
            Device to run computations on ('cpu', 'cuda'). Auto-detected if None.
        **kwargs : dict
            Additional arguments passed to Calculator

        """
        if model_name is None:
            model_name = DEFAULT_MACE_MODEL
        self._calc: Any | None = None

        super().__init__(
            backend="mace",
            model_name=model_name,
            device=device,
            **kwargs,
        )

    def _raise_e3nn_conflict(
        self, error: Exception, *, hessian: bool = False, context: str
    ) -> None:
        msg = format_mace_e3nn_conflict_message(error, hessian=hessian)
        logger.error("MACE %s failed due to e3nn compatibility issues", context)
        logger.debug("%s error details: %s", context, error)
        raise ImportError(msg) from error

    def _load_calculator(self) -> None:
        """Load the MACE calculator implementation."""
        if self._calc is not None:
            return

        try:
            deps.require("torch", purpose="MACE backend")

            with quiet_backend_loading(
                "mace",
                self.model_name,
                None,
                self.device,
                show_model_info=False,
            ):
                if self.model_name == DEFAULT_MACE_MODEL:
                    from mace.calculators import mace_omol

                    self._calc = mace_omol(device=self.device)
                elif self.model_name and self.model_name.startswith("mace-mp"):
                    from mace.calculators import mace_mp

                    model_size = self.model_name.replace("mace-mp-", "") or "medium"
                    self._calc = mace_mp(model=model_size, device=self.device)
                elif self.model_name and self.model_name.startswith("mace-off"):
                    from mace.calculators import mace_off

                    model_size = self.model_name.replace("mace-off-", "") or "medium"
                    self._calc = mace_off(model=model_size, device=self.device)
                else:
                    from mace.calculators import mace_omol

                    self._calc = mace_omol(device=self.device)
        except (ImportError, ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
            if is_mace_e3nn_error(exc):
                self._raise_e3nn_conflict(exc, context="compatibility issue")
            raise_backend_load_error("mace", self.model_name, exc)

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: Sequence[str] | None = None,
        system_changes: Any = None,
    ) -> None:
        """Calculate properties using the MACE calculator."""
        super().calculate(atoms, properties, system_changes)

        calc = self._require_calc()

        try:
            calc.calculate(self.atoms, properties, system_changes)
        except (AttributeError, RuntimeError) as e:
            if is_mace_e3nn_error(e):
                self._raise_e3nn_conflict(e, context="calculation")
            logger.exception("Unexpected error during MACE calculation")
            raise e from e

        try:
            self.results = calc.results.copy()
        except (AttributeError, KeyError, TypeError):
            if properties is None:
                properties = self.implemented_properties
            if "energy" in properties and hasattr(calc, "results"):
                results = calc.results
                if isinstance(results, dict):
                    self.results["energy"] = results.get("energy")
                else:
                    self.results["energy"] = getattr(results, "energy", None)

    def get_potential_energy(
        self, atoms: Atoms | None = None, force_consistent: bool = False
    ) -> float:
        """Get potential energy."""
        try:
            return super().get_potential_energy(atoms, force_consistent)
        except (AttributeError, RuntimeError) as e:
            if is_mace_e3nn_error(e):
                self._raise_e3nn_conflict(e, context="energy calculation")
            logger.exception("Unexpected error during MACE energy calculation")
            raise e from e

    def get_hessian(self, atoms: Atoms | None = None) -> Any:
        """Get analytical Hessian matrix.

        Returns the Hessian matrix (3N x 3N) from MACE's analytical implementation.
        This is much faster and more accurate than finite differences.

        Parameters
        ----------
        atoms : Atoms, optional
            Atoms object to calculate Hessian for

        Returns
        -------
        np.ndarray
            Hessian matrix of shape (3N, 3N) where N is the number of atoms
        """
        if atoms is not None:
            self.atoms = atoms

        calc = self._require_calc()

        if not hasattr(calc, "get_hessian"):
            msg = (
                "MACE calculator does not support analytical Hessians. "
                "This might be due to an older version of mace-torch. "
                "Please update to the latest version."
            )
            logger.warning("MACE calculator does not support analytical Hessians")
            raise NotImplementedError(msg)

        try:
            hessian = calc.get_hessian(atoms=self.atoms)
            if hasattr(hessian, "shape") and len(hessian.shape) == 3:
                if self.atoms is not None:
                    n_atoms = len(self.atoms)
                    if hessian.shape == (3 * n_atoms, n_atoms, 3):
                        hessian = hessian.reshape(3 * n_atoms, 3 * n_atoms)
            hessian = np.asarray(hessian, dtype=np.float64)
            return symmetrize_hessian(hessian)
        except (AttributeError, RuntimeError) as e:
            if is_mace_e3nn_error(e):
                self._raise_e3nn_conflict(e, hessian=True, context="Hessian calculation")
            raise e from e
