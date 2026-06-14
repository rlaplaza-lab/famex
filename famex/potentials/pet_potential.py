"""PET (UPET) Machine Learning Potential integration for ASE.

This module implements integration with lab-cosmo's UPET universal interatomic
potentials based on the Point Edge Transformer architecture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from famex.backends.constants import DEFAULT_PET_MODEL
from famex.backends.dependencies import deps
from famex.potentials.base_potential import BasePotential
from famex.utils.logging import get_famex_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase import Atoms

logger = get_famex_logger(__name__)


def parse_pet_model_name(model_name: str | None) -> tuple[str, str]:
    """Parse FAMEX model_name into UPET model and version.

    Supports optional version suffix: ``pet-mad-s@1.5.0`` → ``("pet-mad-s", "1.5.0")``.
    """
    if model_name is None:
        return DEFAULT_PET_MODEL, "latest"
    if "@" in model_name:
        model, version = model_name.rsplit("@", 1)
        return model, version
    return model_name, "latest"


class PETPotential(BasePotential):
    """ASE Calculator interface for UPET neural network potentials.

    Wraps UPET's ``UPETCalculator`` for universal materials and molecular
    modeling with PET-MAD, PET-OAM, PET-OMat, and related model families.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        model_path: str | None = None,
        version: str = "latest",
        **kwargs: Any,
    ) -> None:
        """Initialize PET potential calculator.

        Parameters
        ----------
        model_name : str, optional
            UPET model identifier (e.g. ``pet-mad-s``, ``pet-oam-xl``).
            An optional version suffix is supported: ``pet-mad-s@1.5.0``.
        device : str, optional
            Device for computations (``cpu``, ``cuda``). Auto-detected if None.
        model_path : str, optional
            Path to a local UPET checkpoint file. When set, ``model_name`` and
            ``version`` are ignored by UPET (parsed from filename).
        version : str, default "latest"
            UPET model version. Overridden by ``@`` suffix in ``model_name``.
        **kwargs
            Additional arguments passed to BasePotential.
        """
        self._calc: Any | None = None
        self.model_path = model_path

        parsed_model, parsed_version = parse_pet_model_name(model_name)
        if model_name is not None and "@" in model_name:
            version = parsed_version
        self.pet_model = parsed_model
        self.pet_version = version

        super().__init__(model_name=model_name or DEFAULT_PET_MODEL, device=device, **kwargs)

    def _load_calculator(self) -> None:
        """Load the UPET calculator implementation."""
        if self._calc is not None:
            return

        from famex.utils.ml_warnings import quiet_backend_loading

        if not deps.has("torch"):
            msg = "PyTorch is required for PET backend. Install with: pip install torch"
            raise ImportError(msg)

        if not deps.has("upet"):
            msg = "upet is required for PET backend. Install with: pip install upet"
            raise ImportError(msg)

        with quiet_backend_loading(
            "pet",
            self.pet_model,
            self.model_path,
            self.device,
            show_model_info=False,
        ):
            try:
                from upet.calculator import UPETCalculator

                device = self.device or "cpu"
                if self.model_path is not None:
                    self._calc = UPETCalculator(checkpoint_path=self.model_path, device=device)
                else:
                    self._calc = UPETCalculator(
                        model=self.pet_model,
                        version=self.pet_version,
                        device=device,
                    )
            except ImportError as exc:
                logger.error("UPET not available: %s. Install with: pip install upet", exc)
                msg = f"UPET not available ({exc}). Install with: pip install upet"
                raise ImportError(msg) from exc

    def _get_backend_name(self) -> str:
        """Get the backend name for this calculator."""
        return "pet"

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: Sequence[str] | None = None,
        system_changes: Any = None,
    ) -> None:
        """Calculate properties using the UPET calculator."""
        super().calculate(atoms, properties, system_changes)

        if self._calc is None:
            self._load_calculator()

        if self._calc is None:
            logger.error("Failed to load PET calculator")
            msg = "Failed to load PET calculator"
            raise RuntimeError(msg)

        self._calc.calculate(self.atoms, properties, system_changes)
        self.results = self._calc.results.copy()

    def get_potential_energy(
        self, atoms: Atoms | None = None, force_consistent: bool = False
    ) -> float:
        """Get potential energy."""
        if atoms is not None:
            self.atoms = atoms
        return super().get_potential_energy(atoms, force_consistent)

    def get_forces(self, atoms: Atoms | None = None) -> Any | None:
        """Get forces on atoms."""
        if atoms is not None:
            self.atoms = atoms
        return super().get_forces(atoms)

    def get_stress(self, atoms: Atoms | None = None) -> Any:
        """Get stress tensor if supported by the underlying UPET model."""
        if atoms is not None:
            self.atoms = atoms
        if self._calc is None:
            self._load_calculator()

        if self._calc is not None and hasattr(self._calc, "get_stress"):
            return self._calc.get_stress(atoms)
        msg = "Stress calculation not supported by this PET model"
        raise NotImplementedError(msg)


def get_pet_calculator(
    model_name: str | None = None,
    device: str | None = None,
    model_path: str | None = None,
    version: str = "latest",
    **kwargs: Any,
) -> PETPotential:
    """Create PET (UPET) calculator.

    Parameters
    ----------
    model_name : str, optional
        UPET model to use. Defaults to ``pet-mad-s``.
        Version can be specified as ``pet-mad-s@1.5.0``.
    device : str, optional
        Device for computations (``cpu``, ``cuda``). Auto-detected if None.
    model_path : str, optional
        Path to a local UPET checkpoint file.
    version : str, default "latest"
        UPET model version when not embedded in ``model_name``.
    **kwargs
        Additional arguments passed to PETPotential.

    Returns
    -------
    PETPotential
        Configured PET calculator instance.

    Examples
    --------
    >>> calc = get_pet_calculator()  # Uses pet-mad-s @ latest
    >>> calc = get_pet_calculator(model_name="pet-mad-s@1.5.0")
    >>> calc = get_pet_calculator(model_name="pet-oam-xl", device="cuda")

    """
    return PETPotential(
        model_name=model_name,
        device=device,
        model_path=model_path,
        version=version,
        **kwargs,
    )
