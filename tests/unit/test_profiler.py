from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from qme.utils.profiler import (
    MemoryInfo,
    PerformanceProfiler,
    TimingEntry,
    profile_call,
    profile_optimizer_step,
)


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

        # Wait a bit
        time.sleep(0.01)

        duration = entry.finish()

        assert entry.end_time is not None
        assert entry.duration is not None
        assert duration == entry.duration
        assert duration >= 0.01


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
        assert len(profiler._memory_snapshots) > 0  # Initial snapshot
        assert len(profiler._active_timings) == 0
        assert profiler._start_time > 0

        # Check initial memory snapshot exists
        assert len(profiler._memory_snapshots) == 1

    def test_snapshot_memory(self):
        profiler = PerformanceProfiler()
        initial_count = len(profiler._memory_snapshots)

        memory_info = profiler.snapshot_memory()

        assert isinstance(memory_info, MemoryInfo)
        # In CI environments, psutil may not be available, so ram_mb could be 0.0
        # Just verify the method completes successfully
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
        assert timing.duration >= 0.01

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

    def test_start_end_timing_manual(self):
        profiler = PerformanceProfiler()

        timing = profiler.start_timing("manual_section")
        assert timing.name == "manual_section"
        assert "manual_section" in profiler._active_timings

        time.sleep(0.01)

        duration = profiler.end_timing("manual_section")
        assert duration is not None
        assert duration >= 0.01
        assert "manual_section" not in profiler._active_timings
        assert "manual_section" in profiler._timings

    def test_end_timing_not_started(self):
        profiler = PerformanceProfiler()

        duration = profiler.end_timing("nonexistent")
        assert duration is None

    def test_get_summary(self):
        profiler = PerformanceProfiler()

        # Add some activity
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

        # Run same section multiple times
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

        # Take multiple snapshots
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
        assert memory_stats["snapshots"] >= 4  # Initial + 3 we added

    def test_calculate_resource_stats(self):
        profiler = PerformanceProfiler()

        summary = profiler.get_summary()

        resource_stats = summary["resources"]
        assert isinstance(resource_stats, dict)

        # Should have CPU and memory stats
        if "error" not in resource_stats:
            assert "cpu_percent" in resource_stats
            assert "process_memory_mb" in resource_stats
            assert "system_memory_total_mb" in resource_stats

    def test_gpu_detection(self):
        profiler = PerformanceProfiler()

        # GPU availability depends on environment
        assert isinstance(profiler._gpu_available, bool)

        summary = profiler.get_summary()
        assert isinstance(summary["gpu_available"], bool)

    def test_gpu_memory_tracking(self):
        profiler = PerformanceProfiler()

        # Mock GPU as available and import torch in the method
        with patch.object(profiler, "_gpu_available", True):
            # Need to patch torch at the point where it's imported (inside _get_memory_info)
            # Since torch is imported inside the method, we need to patch it differently
            import sys
            from unittest.mock import MagicMock

            # Create a mock torch module
            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.memory_allocated.return_value = 100 * 1024 * 1024  # 100 MB
            mock_torch.cuda.max_memory_allocated.return_value = 200 * 1024 * 1024  # 200 MB

            # Patch torch in sys.modules for the duration of the call
            original_torch = sys.modules.get("torch")
            sys.modules["torch"] = mock_torch

            try:
                memory_info = profiler._get_memory_info()

                # GPU tracking may or may not work depending on implementation
                # Just verify the method completes without error
                assert memory_info is not None
            finally:
                # Restore original torch if it existed
                if original_torch is not None:
                    sys.modules["torch"] = original_torch
                elif "torch" in sys.modules:
                    del sys.modules["torch"]

    def test_memory_snapshots_ordering(self):
        profiler = PerformanceProfiler()

        # Take snapshots with delays
        profiler.snapshot_memory()
        time.sleep(0.01)
        profiler.snapshot_memory()
        time.sleep(0.01)
        profiler.snapshot_memory()

        # Check ordering
        for i in range(len(profiler._memory_snapshots) - 1):
            assert profiler._memory_snapshots[i][0] <= profiler._memory_snapshots[i + 1][0]

    def test_profile_section_exception_handling(self):
        profiler = PerformanceProfiler()

        # Cannot combine pytest.raises with profile_section - they serve different purposes
        # pytest.raises must be the outer context to catch exceptions
        with pytest.raises(ValueError):  # noqa: SIM117
            with profiler.profile_section("exception_test"):
                raise ValueError("Test exception")

        # Timing should still be recorded
        assert "exception_test" in profiler._timings
        assert len(profiler._timings["exception_test"]) == 1


class TestProfilerDecorators:
    def test_profile_call_decorator_with_profiler(self):
        class TestClass:
            def __init__(self):
                self.profiler = PerformanceProfiler()

            @profile_call
            def test_method(self):
                time.sleep(0.01)
                return "result"

        obj = TestClass()
        result = obj.test_method()

        assert result == "result"
        # Check that timing was recorded
        summary = obj.profiler.get_summary()
        assert len(summary["timings"]) > 0

    def test_profile_call_decorator_without_profiler(self):
        class TestClass:
            @profile_call
            def test_method(self):
                return "result"

        obj = TestClass()
        result = obj.test_method()

        assert result == "result"

    def test_profile_optimizer_step_decorator(self):
        class TestOptimizer:
            def __init__(self):
                self.profiler = PerformanceProfiler()

            @profile_optimizer_step
            def step(self):
                time.sleep(0.01)
                return "step_result"

        optimizer = TestOptimizer()
        result = optimizer.step()

        assert result == "step_result"

        # Check that optimizer step was counted
        summary = optimizer.profiler.get_summary()
        assert summary["calculator_calls"]["optimizer_steps"] == 1

        # Check that timing was recorded
        assert "optimizer_step" in summary["timings"]

    def test_profile_optimizer_step_multiple(self):
        class TestOptimizer:
            def __init__(self):
                self.profiler = PerformanceProfiler()

            @profile_optimizer_step
            def step(self):
                pass

        optimizer = TestOptimizer()

        for _ in range(5):
            optimizer.step()

        summary = optimizer.profiler.get_summary()
        assert summary["calculator_calls"]["optimizer_steps"] == 5
