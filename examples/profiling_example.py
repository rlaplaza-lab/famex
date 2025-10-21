#!/usr/bin/env python3
"""
QME Performance Profiling Example

This example demonstrates how to use QME's performance profiling system
to track detailed performance metrics during optimization.

Usage:
    python profiling_example.py
"""

import sys

from ase.build import molecule


def main():
    """Demonstrate QME performance profiling."""
    print("🚀 QME Performance Profiling Example")
    print("=" * 50)

    try:
        from qme import Explorer

        # Create a simple molecule for testing
        print("Creating benzene molecule...")
        benzene = molecule("C6H6")

        print("\n1. Running optimization WITHOUT profiling:")
        print("-" * 40)
        explorer_no_prof = Explorer(benzene, backend="mock", target="minima", strategy="local", verbose=1)
        result_no_prof = explorer_no_prof.run(steps=10)
        print(f"   Converged: {result_no_prof['converged']}")
        print(f"   Steps taken: {result_no_prof.get('steps_taken', 'N/A')}")
        print(f"   Performance data: {'No' if 'performance' not in result_no_prof else 'Yes'}")

        print("\n2. Running optimization WITH profiling:")
        print("-" * 40)
        try:
            explorer_prof = Explorer(benzene, backend="mock", target="minima", strategy="local", profile=True, verbose=1)
            result_prof = explorer_prof.run(steps=10)
            print(f"   Converged: {result_prof['converged']}")
            print(f"   Steps taken: {result_prof.get('steps_taken', 'N/A')}")
        except Exception as e:
            print(f"   ⚠️  Mock calculator limitation: {e}")
            print("   This is expected - mock calculator doesn't support forces")
            # Create a dummy result for demonstration
            result_prof = {"converged": False, "steps_taken": 0}

        # Display performance metrics
        if "performance" in result_prof:
            perf = result_prof["performance"]
            print("\n📊 Performance Metrics:")
            print(f"   Total time: {perf['total_time']:.3f} seconds")
            print(f"   Memory snapshots: {perf['memory']['snapshots']}")
            print(f"   Peak memory: {perf['memory']['peak_mb']:.1f} MB")
            print(f"   Memory delta: {perf['memory']['delta_mb']:.1f} MB")

            print("\n🔧 Calculator Calls:")
            calls = perf['calculator_calls']
            print(f"   Energy evaluations: {calls['energy']}")
            print(f"   Force evaluations: {calls['forces']}")
            print(f"   Hessian evaluations: {calls['hessian']}")
            print(f"   Optimizer steps: {calls['optimizer_steps']}")

            print("\n⏱️ Timing Breakdown:")
            timings = perf['timings']
            for section, data in timings.items():
                print(f"   {section}: {data['total_time']:.3f}s (avg: {data['avg_time']:.3f}s, count: {data['count']})")

            if perf['gpu_available']:
                print("\n🎮 GPU Information:")
                print("   GPU available: Yes")
                if 'gpu_peak_mb' in perf['memory']:
                    print(f"   GPU peak memory: {perf['memory']['gpu_peak_mb']:.1f} MB")
            else:
                print("\n🎮 GPU Information:")
                print("   GPU available: No")

        else:
            print("   ❌ No performance data found!")

        print("\n3. Advanced Usage - Direct Profiler Access:")
        print("-" * 40)
        from qme import PerformanceProfiler

        profiler = PerformanceProfiler()

        # Manual profiling
        with profiler.profile_section("custom_operation"):
            # Simulate some work
            import time
            time.sleep(0.1)
            profiler.increment_call("energy", 3)
            profiler.snapshot_memory()

        summary = profiler.get_summary()
        print(f"   Custom operation time: {summary['timings']['custom_operation']['total_time']:.3f}s")
        print(f"   Energy calls: {summary['calculator_calls']['energy']}")
        print(f"   Memory snapshots: {summary['memory']['snapshots']}")

        print("\n✅ Performance profiling example completed successfully!")
        print("\nKey Benefits:")
        print("  • Track optimization performance in detail")
        print("  • Monitor memory usage and GPU utilization")
        print("  • Count calculator calls for cost analysis")
        print("  • Profile custom operations with context managers")
        print("  • Zero overhead when profiling is disabled")
        print("\nNote: For full functionality, use real ML backends (uma, aimnet2, mace, so3lr)")
        print("instead of the mock calculator. The mock calculator has limitations for")
        print("demonstration purposes.")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
