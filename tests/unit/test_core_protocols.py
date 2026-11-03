"""Tests for FAMEX core protocols.

Protocols are type-checking interfaces, but we can test that they exist
and that classes implementing them work correctly.
"""

from __future__ import annotations

from famex.core.protocols import (
    ExplorerProtocol,
    PerformanceProfilerProtocol,
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
