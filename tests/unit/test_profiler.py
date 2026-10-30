from __future__ import annotations

import sys
import time
from unittest.mock import MagicMock, patch

import pytest

from famex.utils.profiler import (
    MemoryInfo,
    PerformanceProfiler,
    TimingEntry,
)


@pytest.fixture(autouse=True)
def _synthetic_clock(monkeypatch):
    """Advance a synthetic clock on sleep; avoid real wall-clock waits."""
    state = {"t": 1000.0}

    def perf_counter() -> float:
        return state["t"]

    def sleep(seconds: float) -> None:
        state["t"] += float(seconds)

    monkeypatch.setattr(time, "perf_counter", perf_counter)
    monkeypatch.setattr(time, "sleep", sleep)
    monkeypatch.setattr("famex.utils.profiler.time.perf_counter", perf_counter)


class TestTimingEntry:
    def test_timing_entry_creation(self):
        entry = TimingEntry(name="test", start_time=0.0)
        assert entry.name == "test"
        assert entry.start_time == 0.0
        assert entry.end_time is None
        assert entry.duration is None
        assert entry.parent is None

    def test_timing_entry_finish(self):
        entry = TimingEntry(name="test", start_time=time.perf_counter())
        time.sleep(0.01)
        duration = entry.finish()

        assert entry.end_time is not None
        assert entry.duration is not None
        assert duration == entry.duration
        assert duration == pytest.approx(0.01)


class TestMemoryInfo:
    def test_memory_info_creation(self):
        info = MemoryInfo(ram_mb=100.0)
        assert info.ram_mb == 100.0
        assert info.gpu_mb is None
        assert info.ram_percent == 0.0
        assert info.gpu_percent is None

    def test_memory_info_with_gpu(self):
        info = MemoryInfo(ram_mb=100.0, gpu_mb=50.0, gpu_percent=25.0)
        assert info.ram_mb == 100.0
        assert info.gpu_mb == 50.0
        assert info.gpu_percent == 25.0


class TestPerformanceProfiler:
    def test_profiler_initialization(self):
        profiler = PerformanceProfiler()

        assert len(profiler._timings) == 0
        assert len(profiler._memory_snapshots) > 0
        assert len(profiler._active_timings) == 0
        assert profiler._start_time > 0
        assert len(profiler._memory_snapshots) == 1

    def test_snapshot_memory(self):
        profiler = PerformanceProfiler()
        initial_count = len(profiler._memory_snapshots)

        memory_info = profiler.snapshot_memory()

        assert isinstance(memory_info, MemoryInfo)
        assert memory_info.ram_mb >= 0
        assert len(profiler._memory_snapshots) == initial_count + 1

    def test_increment_call(self):
        profiler = PerformanceProfiler()

        profiler.increment_call("energy")
        assert profiler._calculator_calls["energy"] == 1

        profiler.increment_call("energy", count=2)
        assert profiler._calculator_calls["energy"] == 3

        profiler.increment_call("custom_call", count=5)
        assert profiler._calculator_calls["custom_call"] == 5

    def test_profile_section_context_manager(self):
        profiler = PerformanceProfiler()

        with profiler.profile_section("test_section"):
            time.sleep(0.01)

        assert "test_section" in profiler._timings
        assert len(profiler._timings["test_section"]) == 1

        timing = profiler._timings["test_section"][0]
        assert timing.name == "test_section"
        assert timing.duration is not None
        assert timing.duration == pytest.approx(0.01)

    def test_profile_section_nested(self):
        profiler = PerformanceProfiler()

        with (
            profiler.profile_section("parent"),
            profiler.profile_section("child", parent="parent"),
        ):
            time.sleep(0.01)

        assert "parent" in profiler._timings
        assert "child" in profiler._timings

        child_timing = profiler._timings["child"][0]
        assert child_timing.parent == "parent"

    def test_get_summary(self):
        profiler = PerformanceProfiler()

        profiler.increment_call("energy", count=5)
        profiler.increment_call("forces", count=3)

        with profiler.profile_section("test"):
            time.sleep(0.01)

        profiler.snapshot_memory()

        summary = profiler.get_summary()

        assert isinstance(summary, dict)
        assert "total_time" in summary
        assert "timings" in summary
        assert "memory" in summary
        assert "calculator_calls" in summary
        assert "resources" in summary
        assert "gpu_available" in summary

        assert summary["calculator_calls"]["energy"] == 5
        assert summary["calculator_calls"]["forces"] == 3
        assert "test" in summary["timings"]

    def test_get_summary_timing_stats(self):
        profiler = PerformanceProfiler()

        for _ in range(3):
            with profiler.profile_section("repeated"):
                time.sleep(0.01)

        summary = profiler.get_summary()

        assert "repeated" in summary["timings"]
        timing_stats = summary["timings"]["repeated"]
        assert timing_stats["count"] == 3
        assert "total_time" in timing_stats
        assert "avg_time" in timing_stats
        assert "min_time" in timing_stats
        assert "max_time" in timing_stats

    def test_calculate_memory_stats(self):
        profiler = PerformanceProfiler()

        for _ in range(3):
            profiler.snapshot_memory()

        summary = profiler.get_summary()

        memory_stats = summary["memory"]
        assert "initial_mb" in memory_stats
        assert "peak_mb" in memory_stats
        assert "final_mb" in memory_stats
        assert "delta_mb" in memory_stats
        assert "avg_mb" in memory_stats
        assert "snapshots" in memory_stats
        assert memory_stats["snapshots"] >= 4

    def test_calculate_resource_stats(self):
        profiler = PerformanceProfiler()

        summary = profiler.get_summary()

        resource_stats = summary["resources"]
        assert isinstance(resource_stats, dict)

        if "error" not in resource_stats:
            assert "cpu_percent" in resource_stats
            assert "process_memory_mb" in resource_stats
            assert "system_memory_total_mb" in resource_stats

    def test_gpu_detection(self):
        profiler = PerformanceProfiler()

        assert isinstance(profiler._gpu_available, bool)

        summary = profiler.get_summary()
        assert isinstance(summary["gpu_available"], bool)

    def test_gpu_memory_tracking(self, monkeypatch):
        profiler = PerformanceProfiler()

        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=64 * 1024 * 1024)
        mock_process.memory_percent.return_value = 1.0
        mock_psutil = MagicMock()
        mock_psutil.Process.return_value = mock_process
        monkeypatch.setattr("famex.utils.profiler._get_psutil", lambda: mock_psutil)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 100 * 1024 * 1024
        mock_torch.cuda.max_memory_allocated.return_value = 200 * 1024 * 1024
        monkeypatch.setitem(sys.modules, "torch", mock_torch)

        with patch.object(profiler, "_gpu_available", True):
            memory_info = profiler._get_memory_info()

        assert memory_info.ram_mb == 64.0
        assert memory_info.gpu_mb == 100.0
        assert memory_info.gpu_percent == 50.0

    def test_memory_snapshots_ordering(self):
        profiler = PerformanceProfiler()

        profiler.snapshot_memory()
        time.sleep(0.01)
        profiler.snapshot_memory()
        time.sleep(0.01)
        profiler.snapshot_memory()

        for i in range(len(profiler._memory_snapshots) - 1):
            assert profiler._memory_snapshots[i][0] <= profiler._memory_snapshots[i + 1][0]

    def test_profile_section_exception_handling(self):
        profiler = PerformanceProfiler()

        with pytest.raises(ValueError):  # noqa: SIM117
            with profiler.profile_section("exception_test"):
                raise ValueError("Test exception")

        assert "exception_test" in profiler._timings
        assert len(profiler._timings["exception_test"]) == 1
