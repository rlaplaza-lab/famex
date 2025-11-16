"""Protocol interfaces for type checking in QME core module.

This module provides Protocol classes that define the interface contracts
for components that interact with strategies and the Explorer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypedDict

if TYPE_CHECKING:
    from ase import Atoms


class ExplorerProtocol(Protocol):
    """Protocol defining the interface that strategies expect from Explorer.

    This allows strategies to work with Explorer instances while maintaining
    type safety and enabling testing with mock objects.
    """

    backend: str
    model_name: str | None
    model_path: str | None
    device: str | None
    default_charge: int
    default_spin: int
    verbose: int
    atoms_list: list[Atoms]

    def _create_and_attach_calculator(self, atoms: Atoms) -> object:
        """Create and attach calculator to atoms object."""
        ...

    def _apply_constraints(self, atoms: Atoms) -> list[object]:
        """Apply constraints to atoms object."""
        ...

    def _get_effective_optimizer(self) -> str:
        """Get the effective optimizer name."""
        ...

    def calculate_frequencies(
        self,
        atoms: Atoms | None = None,
        delta: float = 0.01,
        method: str = "auto",
        temperature: float = 298.15,
        save_hessian: bool = True,
        indices: list[int] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Calculate vibrational frequencies and thermodynamic properties."""
        ...


class PerformanceProfilerProtocol(Protocol):
    """Protocol for performance profiler interface."""

    def get_summary(self) -> dict[str, object]:
        """Get profiling summary dictionary."""
        ...

    def snapshot_memory(self) -> object:
        """Take a memory snapshot."""
        ...

    def profile_section(self, name: str, parent: str | None = None) -> Any:
        """Context manager for timing a code section."""
        ...


# TypedDict classes for strategy result dictionaries


class BaseStrategyResult(TypedDict, total=False):
    """Base TypedDict for strategy result dictionaries.

    All strategy results must include at least:
    - optimized_atoms: The optimized structure(s)
    - strategy: The strategy name
    - converged: Whether optimization converged

    Additional fields are optional and strategy-specific.
    """

    optimized_atoms: Atoms | list[Atoms]
    strategy: str
    converged: bool
    steps_taken: int
    performance: dict[str, object]  # Performance profiling data


class MinimaStrategyResult(TypedDict, total=False):
    """TypedDict for minima optimization results."""

    optimized_atoms: Atoms | list[Atoms]
    strategy: str
    converged: bool
    steps_taken: int
    performance: dict[str, object]
    frequency_analysis: dict[str, object]
    is_minimum: bool
    free_energy_correction: float


class TSStrategyResult(TypedDict, total=False):
    """TypedDict for transition state optimization results."""

    optimized_atoms: Atoms | list[Atoms]
    strategy: str
    converged: bool
    steps_taken: int
    performance: dict[str, object]
    frequency_analysis: dict[str, object]
    ts_validation: dict[str, object]
    is_ts: bool
    free_energy_correction: float


class PathStrategyResult(TypedDict, total=False):
    """TypedDict for path optimization results (NEB, CI-NEB, IRC, etc.)."""

    optimized_atoms: Atoms | list[Atoms]
    strategy: str
    converged: bool
    steps_taken: int
    performance: dict[str, object]
    npoints: int
    method: str
    climb: bool  # For NEB/CI-NEB
    path_energies: list[float]  # Energy along the path
    barrier_height: float  # Energy barrier (for TS paths)
