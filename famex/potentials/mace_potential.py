"""MACE Machine Learning Potential integration for ASE.

This module implements a MACE calculator integration using the MACE-OMOL-0
foundation model for molecular systems, transition metals, and cations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from famex.backends.constants import DEFAULT_MACE_MODEL
from famex.backends.dependencies import deps
from famex.backends.mace_compat import format_mace_e3nn_conflict_message, is_mace_e3nn_error
from famex.potentials.base_potential import BasePotential
from famex.utils.logging import get_famex_logger

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

    def _load_calculator(self) -> None:
        """Load the MACE calculator implementation."""
        if self._calc is not None:
            return

        from famex.utils.ml_warnings import quiet_backend_loading

        if not deps.has("torch"):
            msg = "PyTorch is required for MACE backend. Install with: pip install torch"
            raise ImportError(msg)

        with quiet_backend_loading(
            "mace",
            self.model_name,
            None,
            self.device,
            show_model_info=False,
        ):
            try:
                if self.model_name == DEFAULT_MACE_MODEL:
                    from mace.calculators import mace_omol

                    self._calc = mace_omol(device=self.device or "cpu")
                elif self.model_name and self.model_name.startswith("mace-mp"):
                    from mace.calculators import mace_mp

                    model_size = self.model_name.replace("mace-mp-", "") or "medium"
                    self._calc = mace_mp(model=model_size, device=self.device or "cpu")
                elif self.model_name and self.model_name.startswith("mace-off"):
                    from mace.calculators import mace_off

                    model_size = self.model_name.replace("mace-off-", "") or "medium"
                    self._calc = mace_off(model=model_size, device=self.device or "cpu")
                else:
                    from mace.calculators import mace_omol

                    self._calc = mace_omol(device=self.device or "cpu")

            except ImportError as e:
                logger.error("MACE not available: %s. Install with: pip install mace-torch", e)
                msg = f"MACE not available ({e}). Install with: pip install mace-torch"
                raise ImportError(msg) from e
            except (ValueError, AttributeError, RuntimeError) as e:
                if is_mace_e3nn_error(e):
                    msg = format_mace_e3nn_conflict_message(e)
                    logger.error("MACE compatibility issue: e3nn version mismatch")
                    logger.debug("Compatibility error details: %s", e)
                    raise ImportError(msg) from e
                logger.exception("Unexpected error loading MACE calculator")
                raise

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
                msg = format_mace_e3nn_conflict_message(e)
                logger.error("MACE calculation failed due to e3nn compatibility issues")
                logger.debug("Calculation error details: %s", e)
                raise ImportError(msg) from e
            logger.exception("Unexpected error during MACE calculation")
            raise

        try:
            self.results = calc.results.copy()
        except (AttributeError, KeyError, TypeError):
            if properties is None:
                properties = self.implemented_properties
            if "energy" in properties and hasattr(calc, "results"):
                self.results["energy"] = getattr(calc.results, "energy", None)

    def get_potential_energy(
        self, atoms: Atoms | None = None, force_consistent: bool = False
    ) -> float:
        """Get potential energy."""
        try:
            return super().get_potential_energy(atoms, force_consistent)
        except (AttributeError, RuntimeError) as e:
            if is_mace_e3nn_error(e):
                msg = format_mace_e3nn_conflict_message(e)
                logger.error("MACE energy calculation failed due to e3nn compatibility issues")
                logger.debug("Energy calculation error details: %s", e)
                raise ImportError(msg) from e
            logger.exception("Unexpected error during MACE energy calculation")
            raise

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
                        return hessian.reshape(3 * n_atoms, 3 * n_atoms)
            return hessian
        except (AttributeError, RuntimeError) as e:
            if is_mace_e3nn_error(e):
                msg = format_mace_e3nn_conflict_message(e, hessian=True)
                logger.error("MACE Hessian calculation failed due to e3nn compatibility issues")
                logger.debug("Hessian calculation error details: %s", e)
                raise ImportError(msg) from e
            raise

    def get_property(
        self, prop: str, atoms: Atoms | None = None, allow_calculation: bool = True
    ) -> Any:
        """Get a specific property from the calculator.

        This method is used by ASE's property system and frequency analysis.

        Parameters
        ----------
        prop : str
            Property name ('energy', 'forces', 'hessian', etc.)
        atoms : Atoms, optional
            Atoms object to calculate property for
        allow_calculation : bool, default=True
            Whether calculation is allowed (ASE standard parameter, ignored here)

        Returns
        -------
        Any
            The requested property
        """
        if atoms is not None:
            self.atoms = atoms

        if prop == "energy":
            return self.get_potential_energy(atoms)
        if prop == "forces":
            return self.get_forces(atoms)
        if prop == "hessian":
            return self.get_hessian(atoms)
        msg = f"Property '{prop}' not supported by MACEPotential"
        raise KeyError(msg)


def get_mace_calculator(
    model_name: str | None = None,
    device: str | None = None,
    **kwargs: Any,
) -> MACEPotential:
    """Create MACE calculator.

    Parameters
    ----------
    model_name : str, optional
        MACE model to use. Defaults to ``DEFAULT_MACE_MODEL``.
    device : str, optional
        Device for computations ('cpu', 'cuda'). Auto-detected if None.
    **kwargs : dict
        Additional arguments passed to MACEPotential

    Returns
    -------
    MACEPotential
        Configured MACE calculator instance

    Examples
    --------
    >>> calc = get_mace_calculator()  # Uses MACE-OMOL-0
    >>> calc = get_mace_calculator(model_name="mace-mp-medium")
    >>> calc = get_mace_calculator(model_name="mace-off-large", device="cuda")

    """
    return MACEPotential(model_name=model_name, device=device, **kwargs)
