"""Base potential classes for QME.

This base class centralizes attributes expected by concrete potential
implementations (model_name, device, atoms, results, implemented_properties)
and provides a compatible ``calculate`` signature so subclasses can call
``super().calculate(atoms, properties, system_changes)`` to perform common
setup work.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ase import Atoms


class BasePotential:
    """Abstract base class for ML potential calculators.

    Concrete backends should implement a compatible interface or provide
    an ASE Calculator wrapper. The base class stores commonly-used
    attributes and provides a no-op ``calculate`` implementation that
    performs standard setup.

    Parameters
    ----------
    backend : str, default "generic"
        Backend identifier (e.g., 'uma', 'mace', 'aimnet2', 'mock')
    model_name : str, optional
        Name of the model to use
    device : str, optional
        Device for computations ('cpu', 'cuda')
    implemented_properties : list of str, default []
        List of properties this calculator can compute
    supports_batch_evaluation : bool, default False
        Whether this calculator supports batch evaluation
    **kwargs
        Additional arguments for specific backends
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the base potential calculator.

        Parameters
        ----------
        **kwargs
            Keyword arguments for configuration
        """
        # Generic backend label (e.g., 'uma', 'mace', 'aimnet2', 'mock')
        self.backend: str = kwargs.get("backend", "generic")

        # Common configuration passed by derived classes
        self.model_name: str | None = kwargs.get("model_name")
        self.device: str | None = kwargs.get("device")

        # ASE-style state
        self.atoms: Atoms | None = None
        self.results: dict[str, Any] = {}

        # Default implemented properties; subclasses may override
        # Preserve class-level implemented_properties if it exists
        if hasattr(self, 'implemented_properties'):
            # Class already defines implemented_properties, use it unless explicitly overridden
            self.implemented_properties = kwargs.get("implemented_properties", self.implemented_properties)
        else:
            # No class-level definition, use from kwargs or default to empty list
            self.implemented_properties: list[str] = kwargs.get("implemented_properties", [])

        # Batch evaluation support
        self._supports_batch_evaluation: bool = kwargs.get("supports_batch_evaluation", False)

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: Sequence[str] | None = None,
        system_changes: Any | None = None,
    ) -> None:
        """Base calculate method that performs minimal setup.

        Subclasses should call this via super().calculate(...) to ensure
        ``self.atoms`` is set and default properties are applied.
        """
        if atoms is not None:
            self.atoms = atoms

        if properties is None:
            properties = self.implemented_properties

        # No explicit computation at base level; subclasses will populate
        # ``self.results`` when appropriate.
        return

    def _prepare_calculation(self, atoms: Atoms | None = None) -> Any | None:
        """Prepare for a calculation by setting atoms and ensuring backend is loaded.

        Parameters
        ----------
        atoms : Atoms, optional
            An optional ASE Atoms object. If provided, it will be set
            as `self.atoms`.

        Returns
        -------
        Any or None
            The loaded backend calculator object, or None if loading failed.
        """
        if atoms is not None:
            self.atoms = atoms
        return self.ensure_loaded()

    def _backend_obj(self) -> Any | None:
        """Return the standardized backend calculator stored in ``self._calc``.

        If the attribute is not present or is None, return None.
        """
        return getattr(self, "_calc", None)

    def ensure_loaded(self) -> Any | None:
        """Ensure the underlying backend calculator is loaded.

        Calls subclass ``_load_calculator`` if no backend object is present.
        After calling the loader this will return the backend object or
        None if loading failed.
        """
        if self._backend_obj() is not None:
            return self._backend_obj()

        # Attempt to call subclass loader if available
        if hasattr(self, "_load_calculator"):
            self._load_calculator()

        return self._backend_obj()

    def get_potential_energy(
        self, atoms: Atoms | None = None, force_consistent: bool = False
    ) -> float:
        """Generic get_potential_energy that delegates to underlying backend.

        If the backend provides ``get_potential_energy`` it is delegated to.
        Otherwise, a ``calculate`` call is performed and the stored
        ``self.results['energy']`` is returned.
        """
        backend = self._prepare_calculation(atoms)
        if backend is None:
            # Fallback: run a calculation to populate results
            self.calculate(self.atoms, properties=["energy"], system_changes=None)
            return float(self.results.get("energy", 0.0))

        if hasattr(backend, "get_potential_energy"):
            # Delegate to backend implementation
            return float(backend.get_potential_energy(self.atoms, force_consistent))

        # Backend does not implement the ASE helper -> run calculate
        self.calculate(self.atoms, properties=["energy"], system_changes=None)
        return float(self.results.get("energy", 0.0))

    def get_forces(self, atoms: Atoms | None = None) -> Any | None:
        """Generic get_forces that delegates to underlying backend.

        If the backend provides ``get_forces`` it is delegated to. Otherwise a
        ``calculate`` call is performed and the stored ``self.results['forces']``
        is returned.
        """
        backend = self._prepare_calculation(atoms)
        if backend is None:
            self.calculate(self.atoms, properties=["forces"], system_changes=None)
            return self.results.get("forces")

        if hasattr(backend, "get_forces"):
            return backend.get_forces(self.atoms)

        self.calculate(self.atoms, properties=["forces"], system_changes=None)
        return self.results.get("forces")

    @property
    def supports_batch_evaluation(self) -> bool:
        """Whether this calculator supports batch evaluation."""
        return self._supports_batch_evaluation

    def calculate_batch(
        self, atoms_list: list[Atoms], properties: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Calculate properties for a batch of structures.

        Parameters
        ----------
        atoms_list : List[Atoms]
            List of ASE Atoms objects to calculate properties for
        properties : List[str], optional
            Properties to calculate (default: ["energy", "forces"])

        Returns
        -------
        List[dict]
            List of result dictionaries, one for each structure
        """
        if not self.supports_batch_evaluation:
            raise NotImplementedError(
                f"Batch evaluation not supported by {self.__class__.__name__}. "
                "Use individual calculations instead."
            )

        # Default implementation: calculate individually
        results = []
        for atoms in atoms_list:
            self.calculate(atoms, properties)
            results.append(self.results.copy())

        return results


__all__ = ["BasePotential"]
