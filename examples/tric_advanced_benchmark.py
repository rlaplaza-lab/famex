#!/usr/bin/env python3
"""
QME TRIC Advanced Features Benchmark

This benchmark showcases the advanced TRIC optimizer features including:
- TR (Translation-Rotation) projection
- Smart connectivity-based bond detection  
- P-RFO transition state optimization
- Comparison with standard optimizers

Usage:
    python tric_advanced_benchmark.py [--backends BACKEND1,BACKEND2,...]
    python tric_advanced_benchmark.py [--test-minima] [--test-ts]
    python tric_advanced_benchmark.py [--device DEVICE]

Features:
    - TRIC vs standard optimizers comparison
    - Advanced features demonstration
    - Performance analysis and timing
    - Convergence behavior evaluation
"""

import json
import sys
import warnings
from pathlib import Path
from typing import Any, List

import numpy as np
from ase import Atoms

# Import QME components
try:
    pass  # QME components imported via benchmark_optimization function
except ImportError as e:
    print(f"❌ Error importing QME: {e}")
    print("   Please ensure QME is installed and accessible")
    sys.exit(1)

# Common interface and device utils
from qme.examples import QMEExampleInterface, benchmark_optimization, create_standard_epilog

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def create_test_molecule() -> Atoms:
    """Create a test molecule for TRIC benchmarking."""
    # Create a small organic molecule that will benefit from TRIC
    # CH3OH (methanol) - has internal rotations and good for testing
    positions = [
        [0.0, 0.0, 0.0],      # C
        [1.0, 0.0, 0.0],      # O  
        [1.5, 0.9, 0.0],      # H (OH)
        [-0.5, 0.9, 0.0],     # H (CH)
        [-0.5, -0.9, 0.0],    # H (CH)
        [0.0, 0.0, 1.0],      # H (CH)
    ]
    symbols = ['C', 'O', 'H', 'H', 'H', 'H']
    
    return Atoms(symbols=symbols, positions=positions)


def create_ts_molecule() -> Atoms:
    """Create a test molecule for TS optimization."""
    # Create a simple TS-like structure (H2 + H -> H3)
    positions = [
        [0.0, 0.0, 0.0],      # H1
        [1.5, 0.0, 0.0],      # H2  
        [0.75, 1.3, 0.0],     # H3 (forming TS)
    ]
    symbols = ['H', 'H', 'H']
    
    return Atoms(symbols=symbols, positions=positions)


def run_tric_benchmark(
    backend: str,
    model_name: str,
    device: str,
    verbose: bool = True,
    test_minima: bool = True,
    test_ts: bool = True,
) -> dict[str, Any]:
    """Run comprehensive TRIC benchmark for a specific backend."""
    results = {
        "backend": backend,
        "model_name": model_name,
        "device": device,
        "test_minima": test_minima,
        "test_ts": test_ts,
        "optimizer_results": {},
        "error": None,
        "timings": {}
    }
    
    try:
        # Test minima optimization
        if test_minima:
            print(f"\n🧪 Testing TRIC minima optimization with {backend}...")
            
            # Test TRIC
            tric_result = benchmark_optimization(
                backend=backend,
                optimizer="tric",
                device=device,
                model_name=model_name,
                verbose=verbose,
                test_ts=False,
                create_structure_func=create_test_molecule,
                suitable_optimizers=["tric"]
            )
            
            # Test standard optimizers for comparison
            standard_optimizers = ["lbfgs", "bfgs", "fire"]
            standard_results = {}
            
            for opt in standard_optimizers:
                print(f"   Testing {opt.upper()}...")
                try:
                    result = benchmark_optimization(
                        backend=backend,
                        optimizer=opt,
                        device=device,
                        model_name=model_name,
                        verbose=False,
                        test_ts=False,
                        create_structure_func=create_test_molecule,
                        suitable_optimizers=[opt]
                    )
                    standard_results[opt] = result
                except Exception as e:
                    print(f"   ❌ {opt.upper()} failed: {e}")
                    standard_results[opt] = {"error": str(e)}
            
            results["optimizer_results"]["minima"] = {
                "tric": tric_result,
                **standard_results
            }
        
        # Test TS optimization
        if test_ts:
            print(f"\n🎯 Testing TRIC TS optimization with {backend}...")
            
            # Test TRIC TS
            tric_ts_result = benchmark_optimization(
                backend=backend,
                optimizer="tric",
                device=device,
                model_name=model_name,
                verbose=verbose,
                test_ts=True,
                create_structure_func=create_ts_molecule,
                suitable_optimizers=["tric"]
            )
            
            # Test SELLA for comparison (if available)
            try:
                print("   Testing SELLA...")
                sella_result = benchmark_optimization(
                    backend=backend,
                    optimizer="sella",
                    device=device,
                    model_name=model_name,
                    verbose=False,
                    test_ts=True,
                    create_structure_func=create_ts_molecule,
                    suitable_optimizers=["sella"]
                )
                results["optimizer_results"]["ts"] = {
                    "tric": tric_ts_result,
                    "sella": sella_result
                }
            except Exception as e:
                print(f"   ⚠️  SELLA not available: {e}")
                results["optimizer_results"]["ts"] = {
                    "tric": tric_ts_result,
                    "sella": {"error": str(e)}
                }
    
    except Exception as e:
        results["error"] = str(e)
        print(f"❌ Benchmark failed for {backend}: {e}")
    
    return results


def analyze_tric_results(results_list: List[dict[str, Any]]):
    """Analyze and display TRIC benchmark results."""
    print(f"\n{'=' * 100}")
    print("TRIC ADVANCED FEATURES BENCHMARK ANALYSIS")
    print(f"{'=' * 100}")
    
    for result in results_list:
        if result["error"]:
            print(f"\n❌ {result['backend']}: {result['error']}")
            continue
            
        print(f"\n🔬 {result['backend'].upper()} Results:")
        print("-" * 50)
        
        # Minima optimization analysis
        if "minima" in result["optimizer_results"]:
            minima_results = result["optimizer_results"]["minima"]
            print("Minima Optimization:")
            
            for optimizer, opt_result in minima_results.items():
                if "error" in opt_result:
                    print(f"  {optimizer.upper()}: ❌ {opt_result['error']}")
                    continue
                    
                conv = opt_result.get("optimization_results", {}).get("converged", False)
                steps = opt_result.get("optimization_results", {}).get("nsteps", "N/A")
                time = opt_result.get("timings", {}).get("optimization", "N/A")
                
                status = "✅" if conv else "❌"
                print(f"  {optimizer.upper()}: {status} Steps: {steps}, Time: {time:.2f}s" if isinstance(time, (int, float)) else f"  {optimizer.upper()}: {status} Steps: {steps}, Time: {time}")
        
        # TS optimization analysis
        if "ts" in result["optimizer_results"]:
            ts_results = result["optimizer_results"]["ts"]
            print("\nTS Optimization:")
            
            for optimizer, opt_result in ts_results.items():
                if "error" in opt_result:
                    print(f"  {optimizer.upper()}: ❌ {opt_result['error']}")
                    continue
                    
                conv = opt_result.get("optimization_results", {}).get("converged", False)
                steps = opt_result.get("optimization_results", {}).get("nsteps", "N/A")
                time = opt_result.get("timings", {}).get("optimization", "N/A")
                
                status = "✅" if conv else "❌"
                print(f"  {optimizer.upper()}: {status} Steps: {steps}, Time: {time:.2f}s" if isinstance(time, (int, float)) else f"  {optimizer.upper()}: {status} Steps: {steps}, Time: {time}")


def save_results(results_list: List[dict[str, Any]], output_path: str):
    """Save benchmark results to JSON file."""
    with open(output_path, "w") as f:
        json.dump(results_list, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


def main():
    """Main function to run the TRIC advanced features benchmark."""
    # Create standardized interface
    interface = QMEExampleInterface(
        name="TRIC Advanced Features Benchmark",
        description="Advanced TRIC Optimizer Features Demonstration",
        epilog=create_standard_epilog("benchmark"),
    )

    parser = interface.create_parser()
    
    # Add TRIC-specific arguments
    parser.add_argument(
        "--test-minima",
        action="store_true",
        help="Test minima optimization (default: True)",
    )
    parser.add_argument(
        "--test-ts", 
        action="store_true",
        help="Test transition state optimization (default: True)",
    )
    # Note: --output argument is already defined in the base interface

    args = parser.parse_args()
    
    # Set defaults for test flags
    if not args.test_minima and not args.test_ts:
        args.test_minima = True
        args.test_ts = True

    interface.print_header("TRIC Advanced Features Benchmark")
    print("Features being tested:")
    print("  • TR (Translation-Rotation) projection")
    print("  • Smart connectivity-based bond detection")
    print("  • P-RFO transition state optimization")
    print("  • Comparison with standard optimizers")
    print()

    # Determine which backends to test
    if args.backends:
        requested_backends = [b.strip() for b in args.backends.split(",")]
        available_backends = interface.filter_available_backends(requested_backends, verbose=True)
        if not available_backends:
            interface.print_error("No requested backends are available!")
            return 1
    else:
        available_backends = interface.get_available_backends(verbose=True)
        if not available_backends:
            interface.print_error("No backends are available!")
            print("Available backends: mock, aimnet2, uma, so3lr, mace, orb")
            print("Install additional backends:")
            print("   - UMA: pip install uma")
            print("   - MACE: pip install mace")
            print("   - TorchSim: pip install torch-sim-atomistic")
            return 1

    # Configuration
    config = {
        "device": args.device,
        "model_name": "default",
        "test_minima": args.test_minima,
        "test_ts": args.test_ts,
    }

    interface.print_configuration(config)

    total_tests = len(available_backends) * (int(args.test_minima) + int(args.test_ts))
    print(
        f"\nRunning TRIC benchmarks for {len(available_backends)} backend(s) × "
        f"{int(args.test_minima) + int(args.test_ts)} test type(s) = {total_tests} tests..."
    )

    # Run benchmarks
    results_list = []

    print(f"\n{'=' * 80}")
    print("TRIC ADVANCED FEATURES BENCHMARKS")
    print(f"{'=' * 80}")

    for backend in available_backends:
        print(f"\n🚀 Testing {backend.upper()} backend...")
        
        result = run_tric_benchmark(
            backend=backend,
            model_name="default",
            device=args.device,
            verbose=args.verbose,
            test_minima=args.test_minima,
            test_ts=args.test_ts,
        )
        
        results_list.append(result)

    # Analyze results
    analyze_tric_results(results_list)

    # Save results
    output_file = args.output or "tric_advanced_benchmark_results.json"
    save_results(results_list, output_file)

    print(f"\n🎉 TRIC Advanced Features Benchmark Complete!")
    print(f"   • Tested {len(available_backends)} backends")
    print(f"   • Minima optimization: {'✅' if args.test_minima else '❌'}")
    print(f"   • TS optimization: {'✅' if args.test_ts else '❌'}")
    print(f"   • Results saved to: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
