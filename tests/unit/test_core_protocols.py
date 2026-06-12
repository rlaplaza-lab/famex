"""Tests for FAMEX core protocols.

Protocols are type-checking interfaces, but we can test that they exist
and that classes implementing them work correctly.
"""

from __future__ import annotations

# Import protocols to ensure they're defined
from famex.core.protocols import (
    BaseStrategyResult,
    ExplorerProtocol,
    MinimaStrategyResult,
    PathStrategyResult,
    PerformanceProfilerProtocol,
    TSStrategyResult,
)


class TestProtocolsExist:
    """Test that protocols are properly defined."""

    def test_explorer_protocol_exists(self):
        """Test that ExplorerProtocol is defined."""
        assert ExplorerProtocol is not None
        # Check that it has expected attributes
        assert hasattr(ExplorerProtocol, "__protocol_attrs__") or hasattr(
            ExplorerProtocol, "__annotations__"
        )

    def test_performance_profiler_protocol_exists(self):
        """Test that PerformanceProfilerProtocol is defined."""
        assert PerformanceProfilerProtocol is not None

    def test_typed_dicts_exist(self):
        """Test that TypedDict classes exist."""
        assert BaseStrategyResult is not None
        assert MinimaStrategyResult is not None
        assert TSStrategyResult is not None
        assert PathStrategyResult is not None


class TestTypedDictUsage:
    """Test that TypedDict classes can be used correctly."""

    def test_base_strategy_result_structure(self):
        """Test BaseStrategyResult structure."""
        from ase import Atoms

        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])

        result: BaseStrategyResult = {
            "optimized_atoms": atoms,
            "strategy": "minima:local",
            "converged": True,
            "steps_taken": 10,
            "performance": {"time": 1.0},
        }

        assert result["optimized_atoms"] == atoms
        assert result["strategy"] == "minima:local"
        assert result["converged"] is True
        assert result["steps_taken"] == 10

    def test_minima_strategy_result_structure(self):
        """Test MinimaStrategyResult structure."""
        from ase import Atoms

        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])

        result: MinimaStrategyResult = {
            "optimized_atoms": atoms,
            "strategy": "minima:local",
            "converged": True,
            "steps_taken": 10,
            "performance": {},
            "frequency_analysis": {},
            "is_minimum": True,
            "free_energy_correction": 0.1,
        }

        assert result["is_minimum"] is True
        assert result["free_energy_correction"] == 0.1

    def test_ts_strategy_result_structure(self):
        """Test TSStrategyResult structure."""
        from ase import Atoms

        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])

        result: TSStrategyResult = {
            "optimized_atoms": atoms,
            "strategy": "ts:local",
            "converged": True,
            "steps_taken": 10,
            "performance": {},
            "frequency_analysis": {},
            "ts_validation": {},
            "is_ts": True,
            "free_energy_correction": 0.1,
        }

        assert result["is_ts"] is True
        assert result["ts_validation"] == {}

    def test_path_strategy_result_structure(self):
        """Test PathStrategyResult structure."""
        from ase import Atoms

        atoms1 = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        atoms2 = Atoms("H2", positions=[[0, 0, 0], [0.8, 0, 0]])

        result: PathStrategyResult = {
            "optimized_atoms": [atoms1, atoms2],
            "strategy": "path:neb",
            "converged": True,
            "steps_taken": 10,
            "performance": {},
            "npoints": 5,
            "method": "linear",
            "climb": False,
            "path_energies": [1.0, 1.1, 1.2],
            "barrier_height": 0.2,
        }

        assert result["npoints"] == 5
        assert result["method"] == "linear"
        assert result["climb"] is False
        assert len(result["path_energies"]) == 3
        assert result["barrier_height"] == 0.2


class TestProtocolImplementation:
    """Test that actual implementations match protocols."""

    def test_explorer_implements_protocol(self):
        """Test that Explorer implements ExplorerProtocol."""
        from ase import Atoms

        from famex.core.explorer import Explorer

        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        explorer = Explorer(atoms, backend="mock")

        # Check that Explorer has required attributes
        assert hasattr(explorer, "backend")
        assert hasattr(explorer, "atoms_list")
        assert hasattr(explorer, "verbose")
        assert hasattr(explorer, "_create_and_attach_calculator")
        assert hasattr(explorer, "calculate_frequencies")

    def test_profiler_implements_protocol(self):
        """Test that PerformanceProfiler implements PerformanceProfilerProtocol."""
        from famex.utils.profiler import PerformanceProfiler

        profiler = PerformanceProfiler()

        # Check that PerformanceProfiler has required methods
        assert hasattr(profiler, "get_summary")
        assert hasattr(profiler, "snapshot_memory")
        assert hasattr(profiler, "profile_section")
