"""MACE Machine Learning Potential integration for ASE.

This module implements a MACE calculator integration using the MACE-OMOL-0
foundation model for molecular systems, transition metals, and cations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qme.backends.dependencies import deps
from qme.potentials.base_potential import BasePotential
from qme.utils.logging import get_qme_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase import Atoms

logger = get_qme_logger(__name__)


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
            MACE model to use. Defaults to "mace-omol-0"
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
            model_name = "mace-omol-0"
        # Placeholder for the underlying calculator implementation (standardized)
        self._calc: Any | None = None

        super().__init__(model_name=model_name, device=device, **kwargs)

    def _load_calculator(self) -> None:
        """Load the MACE calculator implementation."""
        # Skip if already loaded
        if self._calc is not None:
            return

        from qme.utils.ml_warnings import quiet_backend_loading

        if not deps.has("torch"):
            msg = "PyTorch is required for MACE backend. Install with: pip install torch"
            raise ImportError(
                msg,
            )

        # Don't show model info - let the outer context handle it
        with quiet_backend_loading(
            "mace",
            self.model_name,
            None,
            self.device,
            show_model_info=False,
        ):
            try:
                # Try to import MACE calculators
                if self.model_name == "mace-omol-0":
                    from mace.calculators import mace_omol

                    self._calc = mace_omol(device=self.device or "cpu")
                elif self.model_name and self.model_name.startswith("mace-mp"):
                    from mace.calculators import mace_mp

                    # Extract model size from model name (e.g., mace-mp-medium -> medium)
                    model_size = self.model_name.replace("mace-mp-", "") or "medium"
                    self._calc = mace_mp(model=model_size, device=self.device or "cpu")
                elif self.model_name and self.model_name.startswith("mace-off"):
                    from mace.calculators import mace_off

                    # Extract model size from model name (e.g., mace-off-medium -> medium)
                    model_size = self.model_name.replace("mace-off-", "") or "medium"
                    self._calc = mace_off(model=model_size, device=self.device or "cpu")
                else:
                    # Default to MACE-OMOL for unknown model names
                    from mace.calculators import mace_omol

                    self._calc = mace_omol(device=self.device or "cpu")

            except ImportError as e:
                logger.error("MACE not available: %s. Install with: pip install mace-torch", e)
                msg = f"MACE not available ({e}). Install with: pip install mace-torch"
                raise ImportError(msg)
            except (ValueError, AttributeError, RuntimeError) as e:
                if any(
                    phrase in str(e)
                    for phrase in [
                        "too many values to unpack",
                        "_compiled_main",
                        "tensor size",
                    ]
                ):
                    msg = (
                        f"MACE compatibility issue with e3nn versions. "
                        f"MACE 0.3.14 was built with e3nn==0.4.4, but e3nn "
                        f"{self._get_e3nn_version()} is installed. "
                        f"This causes serialization format incompatibilities. "
                        f"\n\nWorkaround options:"
                        f"\n1. Use a separate environment with e3nn==0.4.4 for MACE"
                        f"\n2. Use UMA backend instead (compatible with current e3nn)"
                        f"\n3. Wait for MACE update compatible with e3nn 0.5+"
                        f"\n\nOriginal error: {e}"
                    )
                    logger.error("MACE compatibility issue: e3nn version mismatch")
                    logger.debug("Compatibility error details: %s", e)
                    raise ImportError(
                        msg,
                    )
                logger.exception("Unexpected error loading MACE calculator")
                raise

    def _get_e3nn_version(self) -> str:
        """Get the installed e3nn version."""
        from typing import cast

        try:
            import e3nn

            version = e3nn.__version__
            return cast(str, version)
        except (ImportError, AttributeError):
            return "unknown"

    def _get_backend_name(self) -> str:
        """Get the backend name for this calculator."""
        return "mace"

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: Sequence[str] | None = None,
        system_changes: Any = None,
    ) -> None:
        """Calculate properties using the MACE calculator."""
        # Common setup
        super().calculate(atoms, properties, system_changes)

        # Ensure calculator is loaded
        if self._calc is None:
            self._load_calculator()

        if self._calc is None:
            logger.error("Failed to load MACE calculator")
            msg = "Failed to load MACE calculator"
            raise RuntimeError(msg)

        # Delegate to the underlying MACE calculator
        try:
            self._calc.calculate(self.atoms, properties, system_changes)
        except (AttributeError, RuntimeError) as e:
            if any(phrase in str(e) for phrase in ["_compiled_main", "tensor size"]):
                msg = (
                    f"MACE calculation failed due to e3nn compatibility issues. "
                    f"MACE 0.3.14 was built with e3nn==0.4.4, but e3nn "
                    f"{self._get_e3nn_version()} is installed. "
                    f"\n\nWorkaround options:"
                    f"\n1. Use a separate environment with e3nn==0.4.4 for MACE"
                    f"\n2. Use UMA backend instead (compatible with current e3nn)"
                    f"\n3. Wait for MACE update compatible with e3nn 0.5+"
                    f"\n\nOriginal error: {e}"
                )
                logger.error("MACE calculation failed due to e3nn compatibility issues")
                logger.debug("Calculation error details: %s", e)
                raise ImportError(
                    msg,
                )
            logger.exception("Unexpected error during MACE calculation")
            raise

        # Copy results from underlying calculator
        try:
            self.results = self._calc.results.copy()
        except (AttributeError, KeyError, TypeError):
            # If underlying calculator does not use .results dict or has different structure,
            # attempt to extract common properties as fallback
            # AttributeError: .results doesn't exist
            # KeyError: .results exists but copy() fails
            # TypeError: .results exists but isn't dict-like
            if properties is None:
                properties = self.implemented_properties
            if "energy" in properties and hasattr(self._calc, "results"):
                self.results["energy"] = getattr(self._calc.results, "energy", None)

    def get_potential_energy(
        self, atoms: Atoms | None = None, force_consistent: bool = False
    ) -> float:
        """Get potential energy."""
        if atoms is not None:
            self.atoms = atoms
        # Ensure calculator is loaded
        try:
            return super().get_potential_energy(atoms, force_consistent)
        except (AttributeError, RuntimeError) as e:
            if any(phrase in str(e) for phrase in ["_compiled_main", "tensor size"]):
                msg = (
                    f"MACE calculation failed due to e3nn compatibility issues. "
                    f"MACE 0.3.14 was built with e3nn==0.4.4, but e3nn "
                    f"{self._get_e3nn_version()} is installed. "
                    f"\n\nWorkaround options:"
                    f"\n1. Use a separate environment with e3nn==0.4.4 for MACE"
                    f"\n2. Use UMA backend instead (compatible with current e3nn)"
                    f"\n3. Wait for MACE update compatible with e3nn 0.5+"
                    f"\n\nOriginal error: {e}"
                )
                logger.error("MACE energy calculation failed due to e3nn compatibility issues")
                logger.debug("Energy calculation error details: %s", e)
                raise ImportError(
                    msg,
                )
            logger.exception("Unexpected error during MACE energy calculation")
            raise

    def get_forces(self, atoms: Atoms | None = None) -> Any | None:
        """Get forces on atoms."""
        if atoms is not None:
            self.atoms = atoms
        # Ensure calculator is loaded
        return super().get_forces(atoms)

    def get_stress(self, atoms: Atoms | None = None) -> Any:
        """Get stress tensor (if supported)."""
        if atoms is not None:
            self.atoms = atoms
        # Ensure calculator is loaded
        if self._calc is None:
            self._load_calculator()

        if self._calc is not None and hasattr(self._calc, "get_stress"):
            return self._calc.get_stress(atoms)
        msg = "Stress calculation not supported by this MACE model"
        raise NotImplementedError(msg)

    def get_hessian(self, atoms: Atoms | None = None) -> Any:
        """Get analytical Hessian matrix.

        Returns the Hessian matrix (3N x 3N) from MACE's analytical implementation.
        This is much faster and more accurate than finite differences.

        Parameters
        ----------
        atoms : Atoms, optional
            Atoms object to calculate Hessian for

        Returns:
        -------
        np.ndarray
            Hessian matrix of shape (3N, 3N) where N is the number of atoms
        """
        if atoms is not None:
            self.atoms = atoms

        # Ensure calculator is loaded
        if self._calc is None:
            self._load_calculator()

        if self._calc is None:
            logger.error("Failed to load MACE calculator for Hessian calculation")
            msg = "Failed to load MACE calculator"
            raise RuntimeError(msg)

        if not hasattr(self._calc, "get_hessian"):
            msg = (
                "MACE calculator does not support analytical Hessians. "
                "This might be due to an older version of mace-torch. "
                "Please update to the latest version."
            )
            logger.warning("MACE calculator does not support analytical Hessians")
            raise NotImplementedError(msg)

        try:
            hessian = self._calc.get_hessian(atoms=self.atoms)
            # MACE returns Hessian in shape (3N, N, 3), reshape to (3N, 3N)
            if hasattr(hessian, "shape") and len(hessian.shape) == 3:
                if self.atoms is not None:
                    n_atoms = len(self.atoms)
                    if hessian.shape == (3 * n_atoms, n_atoms, 3):
                        return hessian.reshape(3 * n_atoms, 3 * n_atoms)
            return hessian
        except (AttributeError, RuntimeError) as e:
            # Handle e3nn compatibility issues similar to other methods
            if any(phrase in str(e) for phrase in ["_compiled_main", "tensor size"]):
                msg = (
                    f"MACE Hessian calculation failed due to e3nn compatibility issues. "
                    f"MACE 0.3.14 was built with e3nn==0.4.4, but e3nn "
                    f"{self._get_e3nn_version()} is installed. "
                    f"\n\nWorkaround options:"
                    f"\n1. Use a separate environment with e3nn==0.4.4 for MACE"
                    f"\n2. Use finite difference Hessians (automatic fallback)"
                    f"\n3. Wait for MACE update compatible with e3nn 0.5+"
                    f"\n\nOriginal error: {e}"
                )
                logger.error("MACE Hessian calculation failed due to e3nn compatibility issues")
                logger.debug("Hessian calculation error details: %s", e)
                raise ImportError(msg)
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

        Returns:
        -------
        Any
            The requested property
        """
        if atoms is not None:
            self.atoms = atoms

        if prop == "energy":
            return self.get_potential_energy(atoms)
        elif prop == "forces":
            return self.get_forces(atoms)
        elif prop == "hessian":
            return self.get_hessian(atoms)
        else:
            msg = f"Property '{prop}' not supported by MACEPotential"
            raise KeyError(msg)


def get_mace_calculator(
    model_name: str | None = None,
    device: str | None = None,
    **kwargs: Any,
) -> MACEPotential:
    """Factory function to create MACE calculator.

    Parameters
    ----------
    model_name : str, optional
        MACE model to use. Defaults to "mace-omol-0"
    device : str, optional
        Device for computations ('cpu', 'cuda'). Auto-detected if None.
    **kwargs : dict
        Additional arguments passed to MACEPotential

    Returns:
    -------
    MACEPotential
        Configured MACE calculator instance

    Examples:
    --------
    >>> calc = get_mace_calculator()  # Uses MACE-OMOL-0
    >>> calc = get_mace_calculator(model_name="mace-mp-medium")
    >>> calc = get_mace_calculator(model_name="mace-off-large", device="cuda")

    """
    return MACEPotential(model_name=model_name, device=device, **kwargs)
