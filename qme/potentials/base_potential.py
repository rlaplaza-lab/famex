"""Base potential classes for QME.

This base class centralizes attributes expected by concrete potential
implementations (model_name, device, atoms, results, implemented_properties)
and provides a compatible ``calculate`` signature so subclasses can call
``super().calculate(atoms, properties, system_changes)`` to perform common
setup work.
"""

from typing import Any, List, Optional


class BasePotential:
    """Abstract base class for ML potential calculators.

    Concrete backends should implement a compatible interface or provide
    an ASE Calculator wrapper. The base class stores commonly-used
    attributes and provides a no-op ``calculate`` implementation that
    performs standard setup.
    """

    def __init__(self, **kwargs: Any):
        # Generic backend label (e.g., 'uma', 'mace', 'aimnet2', 'mock')
        self.backend: str = kwargs.get("backend", "generic")

        # Common configuration passed by derived classes
        self.model_name: Optional[str] = kwargs.get("model_name")
        self.device: Optional[str] = kwargs.get("device")

        # ASE-style state
        self.atoms = None
        self.results: dict = {}

        # Default implemented properties; subclasses may override
        self.implemented_properties: List[str] = kwargs.get(
            "implemented_properties", []
        )

    def calculate(self, atoms=None, properties=None, system_changes=None):
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

    # Helper methods to reduce boilerplate in concrete potential wrappers
    def _backend_obj(self):
        """Return the underlying backend calculator object if present.

        This searches a small set of common attribute names used by
        existing potential wrappers so subclasses can be migrated to a
        single attribute name (`self._calc`) incrementally.
        """
        candidates = ("_calc", "calculator", "fairchem_calc", "aimnet2_calc", "calc")
        for name in candidates:
            if hasattr(self, name):
                obj = getattr(self, name)
                if obj is not None:
                    return obj
        return None

    def ensure_loaded(self):
        """Ensure the underlying backend calculator is loaded.

        Calls subclass ``_load_calculator`` if no backend object is present.
        After calling the loader this will return the backend object or
        None if loading failed.
        """
        if self._backend_obj() is not None:
            return self._backend_obj()

        # Attempt to call subclass loader if available
        if hasattr(self, "_load_calculator"):
            try:
                self._load_calculator()
            except Exception:
                # Let callers decide how to handle loader failures
                pass

        return self._backend_obj()

    def get_potential_energy(self, atoms=None, force_consistent: bool = False):
        """Generic get_potential_energy that delegates to underlying backend.

        If the backend provides ``get_potential_energy`` it is delegated to.
        Otherwise, a ``calculate`` call is performed and the stored
        ``self.results['energy']`` is returned.
        """
        if atoms is not None:
            self.atoms = atoms

        backend = self.ensure_loaded()
        if backend is None:
            # Fallback: run a calculation to populate results
            self.calculate(self.atoms, properties=["energy"], system_changes=None)
            return float(self.results.get("energy", 0.0))

        if hasattr(backend, "get_potential_energy"):
            # Delegate to backend implementation
            return backend.get_potential_energy(self.atoms, force_consistent)

        # Backend does not implement the ASE helper -> run calculate
        self.calculate(self.atoms, properties=["energy"], system_changes=None)
        return float(self.results.get("energy", 0.0))

    def get_forces(self, atoms=None):
        """Generic get_forces that delegates to underlying backend.

        If the backend provides ``get_forces`` it is delegated to. Otherwise a
        ``calculate`` call is performed and the stored ``self.results['forces']``
        is returned.
        """
        if atoms is not None:
            self.atoms = atoms

        backend = self.ensure_loaded()
        if backend is None:
            self.calculate(self.atoms, properties=["forces"], system_changes=None)
            return self.results.get("forces")

        if hasattr(backend, "get_forces"):
            return backend.get_forces(self.atoms)

        self.calculate(self.atoms, properties=["forces"], system_changes=None)
        return self.results.get("forces")


__all__ = ["BasePotential"]
