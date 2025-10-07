"""
MACE Machine Learning Potential integration for ASE.

This module implements a MACE calculator integration using the MACE-OMOL-0
foundation model for molecular systems, transition metals, and cations.
"""

from typing import Optional

from qme.dependencies import deps
from qme.potentials.base_potential import BasePotential


class MACEPotential(BasePotential):
    """
    MACE potential calculator using foundation models.

    This calculator provides access to MACE foundation models, particularly
    the MACE-OMOL-0 model which is excellent for molecules, transition metals,
    and cations with charge/spin embedding capabilities.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None, **kwargs):
        """
        Initialize MACE potential calculator.

        Parameters:
        -----------
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
        self._calc = None

        super().__init__(model_name=model_name, device=device, **kwargs)

    def _load_calculator(self):
        """Load the MACE calculator implementation."""
        # Skip if already loaded
        if hasattr(self, "_calc") and self._calc is not None:
            return

        from qme.logging_utils import quiet_backend_loading

        if not deps.has("torch"):
            raise ImportError(
                "PyTorch is required for MACE backend. " "Install with: pip install torch"
            )

        with quiet_backend_loading("mace", self.model_name, None, self.device):
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
                raise ImportError(f"MACE not available ({e}). Install with: pip install mace-torch")
            except (ValueError, AttributeError, RuntimeError) as e:
                if any(
                    phrase in str(e)
                    for phrase in [
                        "too many values to unpack",
                        "_compiled_main",
                        "tensor size",
                    ]
                ):
                    raise ImportError(
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
                else:
                    raise

    def _get_e3nn_version(self) -> str:
        """Get the installed e3nn version."""
        try:
            import e3nn

            return e3nn.__version__
        except (ImportError, AttributeError):
            return "unknown"

    def _get_backend_name(self) -> str:
        """Get the backend name for this calculator."""
        return "mace"

    def calculate(self, atoms=None, properties=None, system_changes=None):
        """Calculate properties using the MACE calculator."""
        # Common setup
        super().calculate(atoms, properties, system_changes)

        # Ensure calculator is loaded
        # Ensure calculator is loaded
        if self._calc is None:
            self._load_calculator()

        if self._calc is None:
            raise RuntimeError("Failed to load MACE calculator")

        # Delegate to the underlying MACE calculator
        try:
            self._calc.calculate(self.atoms, properties, system_changes)
        except (AttributeError, RuntimeError) as e:
            if any(phrase in str(e) for phrase in ["_compiled_main", "tensor size"]):
                raise ImportError(
                    f"MACE calculation failed due to e3nn compatibility issues. "
                    f"MACE 0.3.14 was built with e3nn==0.4.4, but e3nn "
                    f"{self._get_e3nn_version()} is installed. "
                    f"\n\nWorkaround options:"
                    f"\n1. Use a separate environment with e3nn==0.4.4 for MACE"
                    f"\n2. Use UMA backend instead (compatible with current e3nn)"
                    f"\n3. Wait for MACE update compatible with e3nn 0.5+"
                    f"\n\nOriginal error: {e}"
                )
            else:
                raise

        # Copy results from underlying calculator
        try:
            self.results = self._calc.results.copy()
        except Exception:
            # If underlying calculator does not use .results dict, attempt to
            # extract common properties
            if properties is None:
                properties = self.implemented_properties
            if "energy" in properties and hasattr(self._calc, "results"):
                self.results["energy"] = getattr(self._calc.results, "energy", None)

    def get_potential_energy(self, atoms=None, force_consistent=False):
        """Get potential energy."""
        if atoms is not None:
            self.atoms = atoms
        # Ensure calculator is loaded
        try:
            return super().get_potential_energy(atoms, force_consistent)
        except (AttributeError, RuntimeError) as e:
            if any(phrase in str(e) for phrase in ["_compiled_main", "tensor size"]):
                raise ImportError(
                    f"MACE calculation failed due to e3nn compatibility issues. "
                    f"MACE 0.3.14 was built with e3nn==0.4.4, but e3nn "
                    f"{self._get_e3nn_version()} is installed. "
                    f"\n\nWorkaround options:"
                    f"\n1. Use a separate environment with e3nn==0.4.4 for MACE"
                    f"\n2. Use UMA backend instead (compatible with current e3nn)"
                    f"\n3. Wait for MACE update compatible with e3nn 0.5+"
                    f"\n\nOriginal error: {e}"
                )
            else:
                raise

    def get_forces(self, atoms=None):
        """Get forces on atoms."""
        if atoms is not None:
            self.atoms = atoms
        # Ensure calculator is loaded
        return super().get_forces(atoms)

    def get_stress(self, atoms=None):
        """Get stress tensor (if supported)."""
        if atoms is not None:
            self.atoms = atoms
        # Ensure calculator is loaded
        if self._calc is None:
            self._load_calculator()

        if hasattr(self._calc, "get_stress"):
            return self._calc.get_stress(atoms)
        else:
            msg = "Stress calculation not supported by this MACE model"
            raise NotImplementedError(msg)


def get_mace_calculator(
    model_name: Optional[str] = None, device: Optional[str] = None, **kwargs
) -> MACEPotential:
    """
    Factory function to create MACE calculator.

    Parameters:
    -----------
    model_name : str, optional
        MACE model to use. Defaults to "mace-omol-0"
    device : str, optional
        Device for computations ('cpu', 'cuda'). Auto-detected if None.
    **kwargs : dict
        Additional arguments passed to MACEPotential

    Returns:
    --------
    MACEPotential
        Configured MACE calculator instance

    Examples:
    ---------
    >>> calc = get_mace_calculator()  # Uses MACE-OMOL-0
    >>> calc = get_mace_calculator(model_name="mace-mp-medium")
    >>> calc = get_mace_calculator(model_name="mace-off-large", device="cuda")
    """
    return MACEPotential(model_name=model_name, device=device, **kwargs)
