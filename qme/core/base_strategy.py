"""Base strategy classes for QME Explorer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ase import Atoms

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qme.core.protocols import ExplorerProtocol, PerformanceProfilerProtocol


@dataclass
class StrategyMetadata:
    """Metadata for strategy registration."""

    name: str
    target: str
    strategy: str
    description: str
    aliases: list[str]
    requires_multiple_structures: bool = False


class BaseStrategy(ABC):
    """Base class for all optimization strategies."""

    metadata: StrategyMetadata

    def __init__(
        self,
        explorer: ExplorerProtocol,
        profiler: PerformanceProfilerProtocol | None = None,
    ) -> None:
        """Initialize strategy with explorer instance."""
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
        """Validate inputs before running."""
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
        """Validate structures are compatible."""
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
        """Merge profiler results into strategy result."""
        if self.profiler is not None:
            result["performance"] = self.profiler.get_summary()
        return result

    def prepare_result(
        self,
        optimized_atoms: Atoms | Sequence[Atoms],
        **metadata: Any,
    ) -> dict[str, Any]:
        """Standardize result format."""
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
