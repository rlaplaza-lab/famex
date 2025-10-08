# TS Optimizer Benchmark - Transition State Optimizer Comparison

Compares the performance of different transition state optimizers (sella and geometric) for transition state finding using various QME ML backends. Focuses specifically on TS optimization to evaluate which optimizers work best for finding transition states.

## Features

- Transition state optimizer comparison (sella vs geometric)
- All available ML backends tested
- Detailed timing and convergence analysis
- TS-specific optimization evaluation
- Focus on TS finding capabilities

## Usage

### Basic Usage
```bash
# Run with all available backends
python optimizer_comparison_benchmark.py

# Run with specific backends
python optimizer_comparison_benchmark.py --backends uma,aimnet2

# Test specific TS optimizers
python optimizer_comparison_benchmark.py --optimizers sella,geometric
```

### Advanced Usage
```bash
# Run on GPU
python optimizer_comparison_benchmark.py --device cuda

# Test specific TS optimizers
python optimizer_comparison_benchmark.py --optimizers sella,geometric

# Verbose output
python optimizer_comparison_benchmark.py --verbose
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--backends` | Comma-separated list of backends to test | All available ML backends |
| `--device` | Device to use (cpu/cuda) | Auto-detect CUDA if available |
| `--verbose` | Print detailed progress information | False |
| `--output` | Output file for results | Auto-generated based on example name |
| `--optimizers` | Comma-separated list of optimizers to test | sella,geometric |
| `--test-ts` | Test transition state optimization | False |
| `--help` | Show help message | - |

## Supported Backends

- **UMA**: Default backend (Meta AI, general purpose)
- **AIMNet2**: Native PyTorch implementation
- **MACE**: Foundation models for high accuracy
- **SO3LR**: SO(3) neural networks for research
- **TorchSim variants**: Maximum performance acceleration

## Supported TS Optimizers

- **Sella**: Specialized for transition state optimization
- **Geometric**: General purpose geometry optimization (TS capable)

## Output

The optimizer comparison provides detailed performance analysis for each optimizer-backend combination.

### Results Format
- Performance comparison tables
- Convergence analysis
- Timing breakdown per optimizer
- Success/failure rates

### Example Output
```
================================================================================
QME Optimizer Comparison Benchmark
================================================================================

📋 Benchmarking Backends
--------------------------------------------------
  1. uma
Total: 1 backends

Optimizers: sella, geometric, lbfgs, bfgs
Test Type: Transition State

[Detailed execution output...]

============================================================================================================================================
OPTIMIZER COMPARISON SUMMARY
============================================================================================================================================
Backend      Optimizer    Type Converged  Steps    Time/Step (s)  Total (s)  Final E (eV) Max F (eV/Å)
============================================================================================================================================
uma          sella        TS   Yes        N/A      N/A            36.971     -6314.532    0.005599
uma          lbfgs        TS   Yes        N/A      N/A            1.088      -6319.221    0.006913
uma          bfgs         TS   Yes        N/A      N/A            1.240      -6319.221    0.006846
============================================================================================================================================

✅ Completed successfully!
⏱️  Total time: 45.2 seconds
```

## Requirements

- QME package installed
- ASE (Atomic Simulation Environment)
- NumPy
- Optional: PyTorch, fairchem-core, so3lr, mace-torch, torch-sim-atomistic (for specific backends)
- Optional: SELLA optimizer (for transition state optimizations)

## Troubleshooting

### Common Issues

**Backend not available:**
```
Error: Backend 'uma' not available. Install with: pip install fairchem-core
```
**Solution**: Install the required backend dependencies.

**Optimizer not available:**
```
Error: Unknown optimizer: geometric
```
**Solution**: Use supported optimizers: sella, geometric, lbfgs, bfgs, fire.

**CUDA not available:**
```
Warning: CUDA not available, falling back to CPU
```
**Solution**: Install PyTorch with CUDA support or use `--device cpu` explicitly.

## See Also

- [Main QME Documentation](../docs/)
- [Other Examples](./)
- [Troubleshooting Guide](../docs/reference/troubleshooting.md)
