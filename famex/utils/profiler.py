"""Performance profiling system for FAMEX.

This module provides comprehensive performance tracking capabilities including
timing, memory usage, calculator call counts, and GPU/CPU resource utilization.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from types import ModuleType
from typing import Any, cast

# psutil is optional - import lazily when needed
_psutil: ModuleType | bool | None = None


def _get_psutil() -> ModuleType | None:
    """Get psutil module, importing it lazily."""
    global _psutil
    if _psutil is None:
        try:
            import psutil

            _psutil = psutil
        except ImportError:
            _psutil = False  # Mark as unavailable
    if _psutil is False or _psutil is None:
        return None
    # At this point, _psutil must be ModuleType
    return cast(ModuleType, _psutil)


@dataclass
class MemoryInfo:
    """Memory usage information."""

    ram_mb: float
    gpu_mb: float | None = None
    ram_percent: float = 0.0
    gpu_percent: float | None = None


@dataclass
class TimingEntry:
    """Timing information for a specific operation."""

    name: str
    start_time: float
    end_time: float | None = None
    duration: float | None = None
    parent: str | None = None

    def finish(self) -> float:
        """Mark timing as finished and calculate duration."""
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        return self.duration


class PerformanceProfiler:
    """Comprehensive performance profiler for FAMEX operations.

    Tracks timing, memory usage, calculator calls, and resource utilization
    with hierarchical timing contexts and fine-grained metrics.

    Example:
        profiler = PerformanceProfiler()
        with profiler.profile_section("optimization"):
            # Do optimization work
            profiler.increment_call("energy")
            profiler.snapshot_memory()

        summary = profiler.get_summary()

    """

    def __init__(self) -> None:
        """Initialize the profiler."""
        self._timings: dict[str, list[TimingEntry]] = {}
        self._memory_snapshots: list[tuple[float, MemoryInfo]] = []
        self._calculator_calls: dict[str, int] = {
            "energy": 0,
            "forces": 0,
            "hessian": 0,
            "optimizer_steps": 0,
        }
        self._active_timings: dict[str, TimingEntry] = {}
        self._start_time = time.perf_counter()

        # GPU detection (must be done before memory info)
        self._gpu_available = self._detect_gpu()

        self._initial_memory = self._get_memory_info()
        self._memory_snapshots.append((self._start_time, self._initial_memory))

    def _detect_gpu(self) -> bool:
        """Detect if GPU is available."""
        try:
            import torch

            return bool(torch.cuda.is_available())
        except ImportError:
            return False

    def _get_memory_info(self) -> MemoryInfo:
        """Get current memory usage information."""
        psutil = _get_psutil()
        if psutil is None:
            # psutil not available, return minimal info
            return MemoryInfo(ram_mb=0.0, gpu_mb=None, ram_percent=0.0, gpu_percent=None)

        process = psutil.Process(os.getpid())
        ram_info = process.memory_info()
        ram_mb = ram_info.rss / 1024 / 1024  # Convert to MB
        ram_percent = process.memory_percent()

        gpu_mb = None
        gpu_percent = None

        if self._gpu_available:
            try:
                import torch

                if torch.cuda.is_available():
                    gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
                    max_mem = torch.cuda.max_memory_allocated()
                    if max_mem > 0:
                        gpu_percent = (torch.cuda.memory_allocated() / max_mem) * 100
                        if gpu_percent > 100:  # Handle case where current > max
                            gpu_percent = None
            except Exception:
                pass

        return MemoryInfo(
            ram_mb=ram_mb,
            gpu_mb=gpu_mb,
            ram_percent=ram_percent,
            gpu_percent=gpu_percent,
        )

    def snapshot_memory(self) -> MemoryInfo:
        """Take a snapshot of current memory usage."""
        memory_info = self._get_memory_info()
        current_time = time.perf_counter()
        self._memory_snapshots.append((current_time, memory_info))
        return memory_info

    def increment_call(self, call_type: str, count: int = 1) -> None:
        """Increment call counter for a specific type."""
        if call_type in self._calculator_calls:
            self._calculator_calls[call_type] += count
        else:
            self._calculator_calls[call_type] = count

    @contextmanager
    def profile_section(self, name: str, parent: str | None = None) -> Any:
        """Context manager for timing a code section."""
        timing = TimingEntry(name=name, start_time=time.perf_counter(), parent=parent)

        self._active_timings[name] = timing

        try:
            yield timing
        finally:
            timing.finish()
            del self._active_timings[name]

            if name not in self._timings:
                self._timings[name] = []
            self._timings[name].append(timing)

    def start_timing(self, name: str, parent: str | None = None) -> TimingEntry:
        """Start timing a section (manual control)."""
        timing = TimingEntry(name=name, start_time=time.perf_counter(), parent=parent)
        self._active_timings[name] = timing
        return timing

    def end_timing(self, name: str) -> float | None:
        """End timing a section (manual control)."""
        if name in self._active_timings:
            timing = self._active_timings[name]
            duration = timing.finish()
            del self._active_timings[name]

            if name not in self._timings:
                self._timings[name] = []
            self._timings[name].append(timing)

            return duration
        return None

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary."""
        total_time = time.perf_counter() - self._start_time

        # Calculate timing statistics
        timing_stats = {}
        for name, timings in self._timings.items():
            if timings:
                durations = [t.duration for t in timings if t.duration is not None]
                if durations:
                    timing_stats[name] = {
                        "total_time": sum(durations),
                        "count": len(durations),
                        "avg_time": sum(durations) / len(durations),
                        "min_time": min(durations),
                        "max_time": max(durations),
                    }

        # Calculate memory statistics
        memory_stats = self._calculate_memory_stats()

        # Calculate call statistics
        call_stats = dict(self._calculator_calls)

        # Calculate resource utilization
        resource_stats = self._calculate_resource_stats()

        return {
            "total_time": total_time,
            "timings": timing_stats,
            "memory": memory_stats,
            "calculator_calls": call_stats,
            "resources": resource_stats,
            "gpu_available": self._gpu_available,
        }

    def _calculate_memory_stats(self) -> dict[str, Any]:
        """Calculate memory usage statistics."""
        if not self._memory_snapshots:
            return {}

        ram_values = [snapshot[1].ram_mb for snapshot in self._memory_snapshots]
        gpu_values = [
            snapshot[1].gpu_mb
            for snapshot in self._memory_snapshots
            if snapshot[1].gpu_mb is not None
        ]

        initial_ram = self._initial_memory.ram_mb
        initial_gpu = self._initial_memory.gpu_mb

        stats = {
            "initial_mb": initial_ram,
            "peak_mb": max(ram_values) if ram_values else 0,
            "final_mb": ram_values[-1] if ram_values else 0,
            "delta_mb": (ram_values[-1] - initial_ram) if ram_values else 0,
            "avg_mb": sum(ram_values) / len(ram_values) if ram_values else 0,
            "snapshots": len(self._memory_snapshots),
        }

        if gpu_values:
            stats.update(
                {
                    "gpu_initial_mb": initial_gpu or 0,
                    "gpu_peak_mb": max(gpu_values),
                    "gpu_final_mb": gpu_values[-1],
                    "gpu_delta_mb": gpu_values[-1] - (initial_gpu or 0),
                    "gpu_avg_mb": sum(gpu_values) / len(gpu_values),
                },
            )

        return stats

    def _calculate_resource_stats(self) -> dict[str, Any]:
        """Calculate resource utilization statistics."""
        psutil = _get_psutil()
        if psutil is None:
            # psutil not available, return minimal stats
            return {"error": "psutil not available"}

        try:
            process = psutil.Process(os.getpid())
            cpu_percent = process.cpu_percent()

            stats = {
                "cpu_percent": cpu_percent,
                "process_memory_mb": process.memory_info().rss / 1024 / 1024,
                "process_memory_percent": process.memory_percent(),
            }

            # Add system-wide stats
            system_memory = psutil.virtual_memory()
            stats.update(
                {
                    "system_memory_total_mb": system_memory.total / 1024 / 1024,
                    "system_memory_available_mb": system_memory.available / 1024 / 1024,
                    "system_memory_percent": system_memory.percent,
                },
            )

            # Add GPU stats if available
            if self._gpu_available:
                try:
                    import torch

                    if torch.cuda.is_available():
                        stats.update(
                            {
                                "gpu_memory_allocated_mb": torch.cuda.memory_allocated()
                                / 1024
                                / 1024,
                                "gpu_memory_cached_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                                "gpu_device_count": torch.cuda.device_count(),
                            },
                        )
                except Exception:
                    pass

            return stats

        except Exception:
            return {"error": "Could not calculate resource stats"}


def profile_call(func: Any) -> Any:
    """Profile individual function calls."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Check if profiler is available in the instance
        profiler = None
        if args and hasattr(args[0], "profiler"):
            profiler = args[0].profiler

        if profiler is None:
            return func(*args, **kwargs)

        func_name = f"{func.__module__}.{func.__name__}"
        with profiler.profile_section(func_name):
            return func(*args, **kwargs)

    return wrapper


def profile_optimizer_step(func: Any) -> Any:
    """Profile optimizer step functions."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Check if profiler is available
        profiler = None
        if args and hasattr(args[0], "profiler"):
            profiler = args[0].profiler

        if profiler is None:
            return func(*args, **kwargs)

        # Increment optimizer step counter
        profiler.increment_call("optimizer_steps")

        with profiler.profile_section("optimizer_step"):
            return func(*args, **kwargs)

    return wrapper
