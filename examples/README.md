# QME Examples

This directory contains standardized examples and benchmarks demonstrating the capabilities of the QME (Quick Mechanistic Exploration) package.

## Available Examples

### 1. CLI Demo (`cli_demo.py`)
**Comprehensive Backend Comparison**

Demonstrates QME's command-line interface capabilities by running various optimization tasks across all available ML backends and comparing their performance and reliability.

**Features:**
- Structure optimization using 'opt' command
- Transition state optimization using 'tsopt' command
- Two-ended optimization workflows
- NEB path optimization
- Comprehensive backend performance comparison

### 2. Timing Benchmark (`timing_benchmark.py`)
**ML Backend Performance Analysis**

Comprehensive performance benchmark for QME ML backends using simple geometry optimization and frequency analysis. All backends use the same default optimizer (BFGS) to ensure fair comparison of backend performance.

**Features:**
- Simple geometry optimization + frequency analysis
- All backends use same default optimizer (BFGS)
- Individual energy and force calculations
- Detailed timing breakdown and performance comparison
- ML backend performance comparison (not optimizer comparison)

### 3. TS Optimizer Benchmark (`optimizer_comparison_benchmark.py`)
**Transition State Optimizer Comparison**

Compares the performance of different transition state optimizers (sella and geometric) for transition state finding using various QME ML backends.

**Features:**
- Transition state optimizer comparison (sella vs geometric)
- All available ML backends tested
- Detailed timing and convergence analysis
- TS-specific optimization evaluation
- Focus on TS finding capabilities

### 4. BH28 Benchmark (`bh28_benchmark/`)
**Chemical Accuracy Evaluation**

Comprehensive evaluation of QME backends on the BH28 database of chemical reaction barrier heights.

**Features:**
- Optimizes reactant minima and transition states
- Calculates barrier heights and compares against reference values
- Provides accuracy analysis (MAE, RMSE) for different backends
- Tests 28 diverse chemical reactions

### 5. Zimmermann-93 Benchmark (`zimmermann93_benchmark/`)
**Two-Ended Transition State Search**

Two-ended transition state search benchmark using the Zimmermann-93 dataset.

**Features:**
- Tests reactant→product transition state finding
- Compares located TS geometries with reference structures
- Evaluates NEB-like path optimization capabilities

## Standardized Interface

All examples follow a consistent command-line interface:

### Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--backends` | Comma-separated list of backends to test | All available ML backends |
| `--device` | Device to use (cpu/cuda) | Auto-detect CUDA if available |
| `--verbose` | Print detailed progress information | False |
| `--output` | Output file for results | Auto-generated based on example name |
| `--help` | Show help message | - |

### Basic Usage

```bash
# Run with all available backends
conda run -n py312 python [example_name].py

# Run with specific backends
conda run -n py312 python [example_name].py --backends uma,aimnet2

# Run with verbose output
conda run -n py312 python [example_name].py --verbose

# Run on GPU
conda run -n py312 python [example_name].py --device cuda
```

### Quick Start Examples

```bash
# CLI Demo - Test all backends
conda run -n py312 python cli_demo.py

# Timing Benchmark - Performance analysis
conda run -n py312 python timing_benchmark.py --device cuda --verbose

# Optimizer Comparison - Test different optimizers
conda run -n py312 python optimizer_comparison_benchmark.py --test-ts

# BH28 Benchmark - Chemical accuracy
conda run -n py312 python bh28_benchmark/bh28_benchmark.py --quick

# Zimmermann-93 Benchmark - Two-ended TS search
conda run -n py312 python zimmermann93_benchmark/zimmermann93_benchmark.py --quicker
```

## Supported Backends

- **UMA**: Default backend (Meta AI, general purpose, requires fairchem-core and PyTorch)
- **AIMNet2**: Native PyTorch implementation
- **MACE**: Foundation models for high accuracy
- **SO3LR**: SO(3) neural networks for research
- **TorchSim variants**: Maximum performance acceleration

## Requirements

- QME package installed
- ASE (Atomic Simulation Environment)
- NumPy
- Optional: PyTorch, fairchem-core, so3lr, mace-torch, torch-sim-atomistic (for specific backends)
- Optional: SELLA optimizer (for transition state optimizations)

## Supporting Files

- **`common_interface.py`**: Standardized interface utilities
- **`device_utils.py`**: Automatic device detection utilities
- **`example_files/`**: Sample molecular structure files (XYZ format)
- **`README_TEMPLATE.md`**: Template for creating new example READMEs

## Understanding Results

### Standardized Output Format

All examples provide:
- **Consistent headers**: Clear identification of what's running
- **Progress indicators**: Real-time feedback during execution
- **Summary tables**: Formatted results for easy comparison
- **Error handling**: Clear error messages and suggestions
- **JSON output**: Machine-readable results for further analysis

### Example Output Structure

```
================================================================================
QME [Example Name] - [Description]
================================================================================

📋 Available Backends
--------------------------------------------------
  1. uma
  2. aimnet2
Total: 2 backends

Configuration:
  Device: cuda
  Output: example_results.json
  Verbose: True

[Detailed execution output...]

✅ Completed successfully!
⏱️  Total time: 45.2 seconds
```

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
- [Troubleshooting Guide](../docs/reference/troubleshooting.md)
- [Developer Guide](../docs/developer_guide/)
