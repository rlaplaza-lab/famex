"""Protocol interfaces for type checking in QME core module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypedDict

if TYPE_CHECKING:
    from ase import Atoms


class ExplorerProtocol(Protocol):
    """Protocol defining the interface that strategies expect from Explorer."""

    backend: str
    model_name: str | None
    model_path: str | None
    device: str | None
    default_charge: int
    default_spin: int
    verbose: int
    atoms_list: list[Atoms]

    def _create_and_attach_calculator(self, atoms: Atoms) -> object:
        """Create and attach calculator to atoms."""
        ...

    def _apply_constraints(self, atoms: Atoms) -> list[object]:
        """Apply constraints to atoms."""
        ...

    def _get_effective_optimizer(self) -> str:
        """Get effective optimizer name."""
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
    """Protocol for performance profiler."""

    def get_summary(self) -> dict[str, object]:
        """Get profiling summary."""
        ...

    def snapshot_memory(self) -> object:
        """Take memory snapshot."""
        ...

    def profile_section(self, name: str, parent: str | None = None) -> Any:
        """Context manager for timing code sections."""
        ...


class BaseStrategyResult(TypedDict, total=False):
    """Base TypedDict for strategy result dictionaries."""

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
