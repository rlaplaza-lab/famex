"""Base strategy classes for FAMEX Explorer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ase import Atoms

if TYPE_CHECKING:
    from collections.abc import Sequence

    from famex.core.protocols import ExplorerProtocol, PerformanceProfilerProtocol


@dataclass
class StrategyMetadata:
    """Metadata for strategy registration."""

    name: str
    """Registry name of the strategy."""
    target: str
    """Target the strategy serves (``minima``, ``ts`` or ``path``)."""
    strategy: str
    """Strategy keyword, such as ``local`` or ``neb``."""
    description: str
    """One-line description shown by :meth:`~famex.core.explorer.Explorer.list_strategies`."""
    aliases: list[str]
    """Other names accepted for the strategy."""
    requires_multiple_structures: bool = False
    """Whether the strategy needs at least two input structures."""


class BaseStrategy(ABC):
    """Base class for all optimization strategies.

    Parameters
    ----------
    explorer : Explorer
        Explorer that runs the strategy and provides calculators and settings.
    profiler : PerformanceProfiler, optional
        Profiler whose summary is added to the result.

    """

    metadata: StrategyMetadata
    """Registration metadata of the strategy."""

    def __init__(
        self,
        explorer: ExplorerProtocol,
        profiler: PerformanceProfilerProtocol | None = None,
    ) -> None:
        self.explorer = explorer
        self.profiler = profiler

    @abstractmethod
    def run(
        self,
        atoms_list: list[Atoms],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute strategy. Returns standardized result dict."""

    def validate_inputs(self, atoms_list: list[Atoms]) -> None:
        """Check that the input structures suit the strategy.

        Parameters
        ----------
        atoms_list : list[Atoms]
            Input structures.

        Raises
        ------
        ValueError
            If no structures are given, too few are given for a multi-structure
            strategy, or the structures differ in atom count or composition.

        """
        if not atoms_list:
            msg = "No atoms provided"
            raise ValueError(msg)

        if self.metadata.requires_multiple_structures and len(atoms_list) < 2:
            msg = (
                f"Strategy '{self.metadata.name}' requires multiple structures "
                f"(at least 2), but only {len(atoms_list)} provided"
            )
            raise ValueError(msg)

        if len(atoms_list) > 1:
            self._validate_structure_compatibility(atoms_list)

    def _validate_structure_compatibility(self, atoms_list: list[Atoms]) -> None:
        if not atoms_list:
            return

        n_atoms = len(atoms_list[0])
        for i, atoms in enumerate(atoms_list[1:], start=1):
            if len(atoms) != n_atoms:
                msg = (
                    f"Incompatible structures: structure 0 has {n_atoms} atoms, "
                    f"but structure {i} has {len(atoms)} atoms"
                )
                raise ValueError(msg)

        symbols_0 = atoms_list[0].get_chemical_symbols()
        for i, atoms in enumerate(atoms_list[1:], start=1):
            symbols_i = atoms.get_chemical_symbols()
            if symbols_i != symbols_0:
                msg = (
                    f"Incompatible structures: structure 0 has composition {symbols_0}, "
                    f"but structure {i} has composition {symbols_i}"
                )
                raise ValueError(msg)

    def _merge_profiler_results(self, result: dict[str, Any]) -> dict[str, Any]:
        if self.profiler is not None:
            result["performance"] = self.profiler.get_summary()
        return result

    def prepare_result(
        self,
        optimized_atoms: Atoms | Sequence[Atoms],
        **metadata: Any,
    ) -> dict[str, Any]:
        """Build the standard result dictionary.

        Parameters
        ----------
        optimized_atoms : Atoms or Sequence[Atoms]
            Final structure or structures.
        **metadata
            Extra result entries. Values that are not ``Atoms``, ``list``, ``bool``,
            ``int``, ``float`` or ``str`` are converted to strings with a warning.

        Returns
        -------
        dict[str, Any]
            Result with ``optimized_atoms``, ``strategy`` and the extra entries.

        """
        result: dict[str, Any] = {
            "optimized_atoms": optimized_atoms,
            "strategy": self.metadata.name,
        }
        for key, value in metadata.items():
            if isinstance(value, (Atoms, list, bool, int, float, str)):  # noqa: UP038
                result[key] = value
            else:
                import warnings

                warnings.warn(
                    f"Metadata key '{key}' has non-standard type {type(value).__name__}. "
                    f"Converting to string. Consider using standard types (Atoms, list, bool, int, float, str) "
                    f"to preserve information.",
                    UserWarning,
                    stacklevel=2,
                )
                result[key] = str(value)
        return result
