# QME Backend Timing Benchmark

This directory contains a comprehensive timing benchmark for QME backends (potentials) to analyze performance characteristics and identify bottlenecks in the integration with ASE interfaces.

## Overview

The `timing_benchmark.py` script benchmarks the performance of different QME backends for:
- Geometry optimization of benzene
- Frequency analysis (Hessian calculation)
- Individual energy and force calculations

## Features

- **Comprehensive Timing**: Measures time for each step of the optimization and frequency analysis process
- **Multiple Backends**: Supports all QME backends (mock, aimnet2, uma, so3lr, mace, torchsim, torchsim_mace, torchsim_fairchem)
- **Detailed Breakdown**: Shows timing for initialization, optimization, frequency analysis, and individual calculations
- **JSON Output**: Saves results to JSON file for further analysis
- **Error Handling**: Gracefully handles unavailable backends and provides fallback information

## Usage

### Basic Usage
```bash
# Run benchmark with all available backends
python timing_benchmark.py

# Run with specific backends
python timing_benchmark.py --backends mock,aimnet2,uma,torchsim_mace

# Run with verbose output
python timing_benchmark.py --backends mock --verbose

# Run on GPU (if available)
python timing_benchmark.py --device cuda

# Specify output file
python timing_benchmark.py --output my_benchmark_results.json
```

### Command Line Options

- `--backends`: Comma-separated list of backends to benchmark (default: "mock,aimnet2,uma,so3lr,mace,torchsim,torchsim_mace,torchsim_fairchem")
- `--device`: Device to use for calculations ("cpu" or "cuda", default: "cpu")
- `--output`: Output file for results (default: "timing_benchmark_results.json")
- `--verbose`: Print detailed progress information

## Output

The benchmark provides:

1. **Real-time Progress**: Shows optimization steps and timing for each backend
2. **Summary Table**: Overview of all backends with total times and breakdown
3. **Detailed Breakdown**: Percentage of time spent in each step
4. **JSON Results**: Complete results saved to file for analysis

### Example Output

```
====================================================================================================
BENCHMARK SUMMARY
====================================================================================================
Backend      Available  Total (s)  Init (s)   Opt (s)    Freq (s)   Energy (s) Forces (s)
----------------------------------------------------------------------------------------------------
mock         Yes        20.449     0.000      18.629     1.766      0.002      0.001     
aimnet2      Yes        15.234     2.145      12.456     0.633      0.001      0.001     
uma          Yes        25.678     5.234      18.234     2.210      0.003      0.002     
----------------------------------------------------------------------------------------------------

DETAILED BREAKDOWN:
============================================================

MOCK:
  Initialization      : 0.000s (0.0%)
  Structure Loading   : 0.051s (0.3%)
  Single Energy       : 0.002s (0.0%)
  Single Forces       : 0.001s (0.0%)
  Optimization        : 18.629s (91.1%)
  Frequency Analysis  : 1.766s (8.6%)
```

## Understanding the Results

### Timing Components

1. **Initialization**: Time to create and configure the QME optimizer
2. **Structure Loading**: Time to load and prepare the molecular structure
3. **Single Energy**: Time for a single energy calculation
4. **Single Forces**: Time for a single force calculation
5. **Optimization**: Total time for geometry optimization (usually the largest component)
6. **Frequency Analysis**: Time for Hessian calculation and frequency analysis

### Key Insights

- **Optimization Time**: Usually dominates the total time, especially for complex backends
- **ASE Integration Cost**: The overhead of integrating ML potentials through ASE interfaces
- **Backend Comparison**: Relative performance of different ML potentials
- **Frequency Analysis**: Cost of numerical Hessian calculation vs. potential analytical methods

## Backend Availability

The benchmark automatically detects available backends based on installed dependencies:

- **mock**: Always available (for testing)
- **aimnet2**: Requires PyTorch
- **uma**: Requires fairchem-core and PyTorch
- **so3lr**: Requires so3lr package
- **mace**: Requires mace-torch and PyTorch

## Example Analysis

The benchmark helps identify:

1. **Bottlenecks**: Which steps take the most time
2. **Backend Performance**: Relative speed of different ML potentials
3. **ASE Overhead**: Cost of the ASE interface integration
4. **Optimization Efficiency**: How well different backends converge

## Requirements

- QME package installed
- ASE (Atomic Simulation Environment)
- NumPy
- Optional: PyTorch, fairchem-core, so3lr, mace-torch (for specific backends)

## Notes

- The benchmark uses benzene as a test case with a slightly distorted initial geometry
- Optimization uses BFGS with tight convergence criteria (fmax=0.01)
- Frequency analysis uses finite differences (central differences)
- Results may vary depending on system specifications and installed dependencies
