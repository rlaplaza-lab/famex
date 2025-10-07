# Timing Benchmark - ML Backend Performance Analysis

Comprehensive performance benchmark for QME ML backends using simple geometry optimization and frequency analysis. All backends use the same default optimizer (BFGS) to ensure fair comparison of backend performance rather than optimizer differences.

## Features

- Simple geometry optimization + frequency analysis
- All backends use same default optimizer (BFGS)
- Individual energy and force calculations
- Detailed timing breakdown and performance comparison
- ML backend performance comparison (not optimizer comparison)

## Usage

### Basic Usage
```bash
# Run with all available backends
conda run -n py312 python timing_benchmark.py

# Run with specific backends
conda run -n py312 python timing_benchmark.py --backends uma,aimnet2

# Run with verbose output
conda run -n py312 python timing_benchmark.py --verbose
```

### Advanced Usage
```bash
# Run on GPU
conda run -n py312 python timing_benchmark.py --device cuda

# Specify output file
conda run -n py312 python timing_benchmark.py --output my_results.json
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--backends` | Comma-separated list of backends to test | All available ML backends |
| `--device` | Device to use (cpu/cuda) | Auto-detect CUDA if available |
| `--verbose` | Print detailed progress information | False |
| `--output` | Output file for results | Auto-generated based on example name |
| `--help` | Show help message | - |

## Supported Backends

- **UMA**: Default backend (Meta AI, general purpose)
- **AIMNet2**: Native PyTorch implementation
- **MACE**: Foundation models for high accuracy
- **SO3LR**: SO(3) neural networks for research
- **TorchSim variants**: Maximum performance acceleration

## Output

The timing benchmark provides detailed performance analysis for each backend.

### Results Format
- Performance breakdown by optimization step
- Timing analysis for each component
- Summary tables for easy comparison
- JSON output for further analysis

### Example Output
```
================================================================================
QME Timing Benchmark
================================================================================

📋 Benchmarking Backends
--------------------------------------------------
  1. uma
Total: 1 backends

[Detailed execution output...]

========================================================================================================================
BENCHMARK SUMMARY
========================================================================================================================
Backend      Available  Total (s)  Init (s)   Opt (s)    Freq (s)   E1st (s)   E2nd (s)   Forces (s) Steps    Avg/Step (s)
========================================================================================================================
uma          Yes        9.282      0.000      0.797      3.951      4.476      0.000      0.000      14       0.0570
========================================================================================================================

✅ Completed successfully!
⏱️  Total time: 10.1 seconds
```

## Requirements

- QME package installed
- ASE (Atomic Simulation Environment)
- NumPy
- Optional: PyTorch, fairchem-core, so3lr, mace-torch, torch-sim-atomistic (for specific backends)

## Troubleshooting

### Common Issues

**Backend not available:**
```
Error: Backend 'uma' not available. Install with: pip install fairchem-core
```
**Solution**: Install the required backend dependencies.

**CUDA not available:**
```
Warning: CUDA not available, falling back to CPU
```
**Solution**: Install PyTorch with CUDA support or use `--device cpu` explicitly.

## See Also

- [Main QME Documentation](../docs/)
- [Other Examples](./)
- [Troubleshooting Guide](../docs/reference/troubleshooting.md)
