"""Protocol interfaces for type checking in FAMEX."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypedDict

if TYPE_CHECKING:
    from ase import Atoms


class CalculatorProtocol(Protocol):
    """Minimal calculator interface for Hessian and frequency analysis."""

    def get_forces(self, atoms: Atoms | None = None) -> Any: ...

    def get_potential_energy(
        self,
        atoms: Atoms | None = None,
        force_consistent: bool = False,
    ) -> float: ...


class ExplorerProtocol(Protocol):
    backend: str
    model_name: str | None
    model_path: str | None
    device: str | None
    default_charge: int
    default_spin: int
    verbose: int
    atoms_list: list[Atoms]

    def _create_and_attach_calculator(self, atoms: Atoms) -> object: ...

    def _apply_constraints(self, atoms: Atoms) -> list[object]: ...

    def _get_effective_optimizer(self) -> str: ...

    def calculate_frequencies(
        self,
        atoms: Atoms | None = None,
        delta: float = 0.01,
        method: str = "auto",
        temperature: float = 298.15,
        save_hessian: bool = True,
        indices: list[int] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]: ...


class PerformanceProfilerProtocol(Protocol):
    def get_summary(self) -> dict[str, object]: ...

    def snapshot_memory(self) -> object: ...

    def profile_section(self, name: str, parent: str | None = None) -> Any: ...


class BaseStrategyResult(TypedDict, total=False):
    optimized_atoms: Atoms | list[Atoms]
    strategy: str
    converged: bool
    steps_taken: int
    performance: dict[str, object]


class MinimaStrategyResult(TypedDict, total=False):
    optimized_atoms: Atoms | list[Atoms]
    strategy: str
    converged: bool
    steps_taken: int
    performance: dict[str, object]
    frequency_analysis: dict[str, object]
    is_minimum: bool
    free_energy_correction: float


class TSStrategyResult(TypedDict, total=False):
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
    optimized_atoms: Atoms | list[Atoms]
    strategy: str
    converged: bool
    steps_taken: int
    performance: dict[str, object]
    npoints: int
    method: str
    climb: bool
    path_energies: list[float]
    barrier_height: float
