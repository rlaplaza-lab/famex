# QME Examples

This directory contains examples and benchmarks demonstrating the capabilities of the QME (Quick Mechanistic Exploration) package.

## Available Examples

### 1. CLI Demo (`cli_demo.py`)
A simple demonstration of QME's command-line interface capabilities.

### 2. Timing Benchmark (`timing_benchmark.py`)
Comprehensive performance benchmark for QME backends analyzing:
- Geometry optimization of benzene
- Frequency analysis (Hessian calculation) 
- Individual energy and force calculations
- Detailed timing breakdown and performance comparison

**Usage:**
```bash
# Run benchmark with all available backends
python timing_benchmark.py

# Run with specific backends
python timing_benchmark.py --backends mock,aimnet2,uma,torchsim_mace

# Run on GPU (if available)
python timing_benchmark.py --device cuda
```

### 3. BH28 Benchmark (`bh28_benchmark/`)
Comprehensive evaluation of QME backends on the BH28 database of chemical reaction barrier heights.

**Features:**
- Optimizes reactant minima and transition states
- Calculates barrier heights and compares against reference values
- Provides accuracy analysis (MAE, RMSE) for different backends
- Tests 28 diverse chemical reactions

**Usage:**
```bash
cd bh28_benchmark
python bh28_benchmark.py --quick  # Quick test
python bh28_benchmark.py --backends uma so3lr  # Specific backends
```

### 4. Zimmermann-93 Benchmark (`zimmermann93_benchmark/`)
Two-ended transition state search benchmark using the Zimmermann-93 dataset.

**Features:**
- Tests reactant→product transition state finding
- Compares located TS geometries with reference structures
- Evaluates NEB-like path optimization capabilities

**Usage:**
```bash
cd zimmermann93_benchmark
python zimmermann93_benchmark.py --quick
```

## Supporting Files

- **`backend_utils.py`**: Shared utilities for backend availability detection and testing
- **`example_files/`**: Sample molecular structure files (XYZ format)
- **`timing_benchmark_results.json`**: Results from timing benchmark runs

## Backend Coverage

All examples support the full range of QME backends:
- **Mock**: Always available (for testing)
- **AIMNet2**: Requires PyTorch
- **UMA**: Requires fairchem-core and PyTorch  
- **SO3LR**: Requires so3lr package
- **MACE**: Requires mace-torch and PyTorch
- **TorchSim variants**: torchsim_mace, torchsim_fairchem (requires torch-sim-atomistic)

## Requirements

- QME package installed
- ASE (Atomic Simulation Environment)
- NumPy
- Optional: PyTorch, fairchem-core, so3lr, mace-torch, torch-sim-atomistic (for specific backends)
- Optional: SELLA optimizer (for transition state optimizations)

## Running Examples

All examples should be run using the conda py312 environment and follow a standardized interface:

```bash
conda run -n py312 python <example_script.py> [options]
```

### Common Interface

All examples now provide:
- **Consistent help**: Use `--help` to see available options
- **Backend selection**: Use `--backends backend1,backend2` to test specific backends
- **Standardized output**: Clean, professional formatting with consistent messaging
- **Self-contained**: No external utility dependencies - each example is independent

### Quick Start Examples

```bash
# Run with all available backends
conda run -n py312 python cli_demo.py
conda run -n py312 python timing_benchmark.py

# Run with specific backends
conda run -n py312 python cli_demo.py --backends aimnet2,uma
conda run -n py312 python timing_benchmark.py --backends mace --verbose

# Run quick benchmarks
conda run -n py312 python bh28_benchmark/bh28_benchmark.py --quick
conda run -n py312 python zimmermann93_benchmark/zimmermann93_benchmark.py --quicker
```

## Understanding Results

### Timing Benchmark
- Shows performance breakdown by optimization step
- Identifies bottlenecks in ASE integration
- Compares relative backend performance

### Chemical Benchmarks (BH28, Zimmermann-93)
- Evaluate accuracy against high-level quantum chemistry references
- Test optimization convergence and reliability
- Provide backend recommendations for different use cases

Results are saved as JSON files for further analysis and comparison across different systems and configurations.
