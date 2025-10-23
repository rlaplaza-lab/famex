#!/usr/bin/env python3
"""QME Performance Profiling Example.

This example demonstrates how to use QME's performance profiling system
to track detailed performance metrics during optimization.

Usage:
    python profiling_example.py
"""

import sys

from ase.build import molecule


def main() -> int | None:
    """Demonstrate QME performance profiling."""
    try:
        from qme import Explorer

        # Create a simple molecule for testing
        benzene = molecule("C6H6")

        explorer_no_prof = Explorer(
            benzene,
            backend="mace",
            target="minima",
            strategy="local",
            verbose=1,
        )
        explorer_no_prof.run(steps=10)

        try:
            explorer_prof = Explorer(
                benzene,
                backend="mace",
                target="minima",
                strategy="local",
                profile=True,
                verbose=1,
            )
            result_prof = explorer_prof.run(steps=10)
        except Exception:
            # Create a dummy result for demonstration
            result_prof = {"converged": False, "steps_taken": 0}

        # Display performance metrics
        if "performance" in result_prof:
            perf = result_prof["performance"]

            perf["calculator_calls"]

            timings = perf["timings"]
            for _section, _data in timings.items():
                pass

            if perf["gpu_available"]:
                if "gpu_peak_mb" in perf["memory"]:
                    pass
            else:
                pass

        else:
            pass

        from qme import PerformanceProfiler

        profiler = PerformanceProfiler()

        # Manual profiling
        with profiler.profile_section("custom_operation"):
            # Simulate some work
            import time

            time.sleep(0.1)
            profiler.increment_call("energy", 3)
            profiler.snapshot_memory()

        profiler.get_summary()

        return 0

    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(main())
