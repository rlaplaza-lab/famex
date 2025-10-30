# QME Examples

This directory contains standardized examples and benchmarks demonstrating the capabilities of the QME (Quick Mechanistic Exploration) package.

## Available Examples

### 1. CLI Demo (`cli_demo.py`)
**Comprehensive Backend Comparison**

Demonstrates QME's command-line interface capabilities by running various optimization tasks across all available ML backends and comparing their performance and reliability.

**Features:**
- Structure optimization using 'minima' command (outputs single structure, defaults to BFGS)
- Transition state optimization using 'ts' command (outputs single TS, defaults to Sella)
- Reaction path optimization using dedicated 'path' command:
  * Raw interpolation path generation
  * NEB path optimization (saves complete reaction pathways)
  * CI-NEB (Climbing Image NEB) path optimization (saves complete reaction pathways)
  * IRC (Intrinsic Reaction Coordinate) path from transition state
- Comprehensive backend performance comparison
- Trajectory saving for multi-image results

### 2. IRC Demo (`irc_demo.py`)
**IRC Path Calculation from Transition State**

Demonstrates IRC (Intrinsic Reaction Coordinate) calculations that follow the gradient
downhill from a transition state to generate the minimum energy path connecting
reactants and products. Now uses standardized interface for consistent behavior.

**Features:**
- Calculates IRC path from a single transition state structure
- Follows gradient in both forward and backward directions
- Generates complete reaction pathway through the TS
- Energy profile analysis along the IRC path
- Saves trajectory for visualization
- Standardized interface with consistent output formatting

**Usage:**
```bash
python irc_demo.py example_files/A_C_A_B_A_C_ts.xyz --backends uma --steps 50
python irc_demo.py ts.xyz --backends aimnet2 --direction both --step-size 0.1 --device cuda
```

### 3. Growing String Demo (`growing_string_demo.py`)
**Growing String Method for Transition State Search**

Demonstrates the Growing String Method (DE-GSM) for finding transition states between
reactant and product configurations. This method dynamically grows a string of images
between the endpoints to locate the transition state. Now uses standardized interface
for consistent behavior.

**Features:**
- Growing string method (DE-GSM) for TS search
- Dynamic image addition between reactant and product
- Optional endpoint optimization before growing
- Optional TS refinement after finding
- Configurable step size and convergence criteria
- Saves complete reaction pathway
- Standardized interface with consistent output formatting

**Usage:**
```bash
python growing_string_demo.py --reactant r.xyz --product p.xyz --backends uma
python growing_string_demo.py --backends mock --npoints 20 --steps 100 --device cuda
```

### 4. Timing Benchmark (`timing_benchmark.py`)
**ML Backend Performance Analysis**

Comprehensive performance benchmark for QME ML backends using simple geometry optimization and frequency analysis. All backends use the same default optimizer (BFGS) to ensure fair comparison of backend performance.

**Features:**
- Simple geometry optimization + frequency analysis
- All backends use same default optimizer (BFGS)
- Individual energy and force calculations
- Detailed timing breakdown and performance comparison
- ML backend performance comparison (not optimizer comparison)

### 5. Minima Optimizer Benchmark (`minima_optimizer_benchmark.py`)
**Minima Optimization Comparison**

Compares the performance of different minima optimizers (lbfgs, bfgs, fire, trust-krylov) for minima finding using various QME ML backends.

**Features:**
- Minima optimizer comparison (lbfgs, bfgs, fire, trust-krylov)
- All available ML backends tested
- Detailed timing and convergence analysis
- Minima-specific optimization evaluation
- Focus on minima finding capabilities

### 6. TS Optimizer Benchmark (`ts_optimizer_benchmark.py`)
**Transition State Optimizer Comparison**

Compares the performance of transition state optimizers (sella, trust-krylov-ts) for transition state finding using various QME ML backends.

**Features:**
- Transition state optimizer comparison (sella, trust-krylov-ts)
- All available ML backends tested
- Detailed timing and convergence analysis
- TS-specific optimization evaluation
- Focus on TS finding capabilities

### 7. BH28 Benchmark (`bh28_benchmark/`)
**Chemical Accuracy Evaluation**

Comprehensive evaluation of QME backends on the BH28 database of chemical reaction barrier heights.

**Features:**
- Optimizes reactant minima and transition states
- Calculates barrier heights and compares against reference values
- Provides accuracy analysis (MAE, RMSE) for different backends
- Tests 28 diverse chemical reactions

### 8. Zimmermann-93 Benchmark (`zimmermann93_benchmark/`)
**Two-Ended Transition State Search**

Two-ended transition state search benchmark using the Zimmermann-93 dataset.

**Features:**
- Tests reactant→product transition state finding
- Compares located TS geometries with reference structures
- Evaluates NEB and CI-NEB path optimization capabilities
- Saves complete reaction pathways as trajectory files

### 9. Thermochemistry Demo (`thermochemistry_demo.py`)
**Enhanced Thermochemistry Capabilities**

Standalone feature demonstration showing QME's enhanced thermochemistry capabilities
including quasi-harmonic corrections, complete statistical thermodynamics, solvation
corrections, and symmetry handling. This is a feature-specific demo rather than a
benchmark or comprehensive comparison.

**Features:**
- Quasi-harmonic corrections (Grimme and Truhlar methods)
- Complete statistical thermodynamics (translational, rotational, electronic)
- Solvation corrections
- Symmetry handling
- GoodVibes-inspired implementation

**Usage:**
```bash
python thermochemistry_demo.py
```

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
python [example_name].py

# Run with specific backends
python [example_name].py --backends uma,aimnet2

# Run with verbose output
python [example_name].py --verbose

# Run on GPU
python [example_name].py --device cuda
```

### Quick Start Examples

```bash
# CLI Demo - Test all backends
python cli_demo.py

# IRC Demo - IRC path calculation
python irc_demo.py example_files/A_C_A_B_A_C_ts.xyz --backends uma

# Growing String Demo - TS search with growing string method
python growing_string_demo.py --backends uma --npoints 15

# Timing Benchmark - Performance analysis
python timing_benchmark.py --device cuda --verbose

# Minima Optimizer Comparison - Test minima optimizers
python minima_optimizer_benchmark.py --backends uma,aimnet2

# TS Optimizer Comparison - Test TS optimizers
python ts_optimizer_benchmark.py --backends uma,aimnet2

# BH28 Benchmark - Chemical accuracy
python bh28_benchmark/bh28_benchmark.py --quick

# Zimmermann-93 Benchmark - Two-ended TS search
python zimmermann93_benchmark/zimmermann93_benchmark.py --quicker
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

- **`example_utils/`**: Standardized interface utilities (`QMEExampleInterface`)
- **`minima_optimizer_benchmark.py`**: Focused minima optimizer comparison
- **`ts_optimizer_benchmark.py`**: Focused TS optimizer comparison
- **`uma_hessian_method_comparison.py`**: UMA Hessian computation method comparison
- **`thermochemistry_demo.py`**: Feature-specific thermochemistry demonstration
- **`example_files/`**: Sample molecular structure files (XYZ format)

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
- [FAQ and Troubleshooting](../docs/FAQ.md)
- [User Guide](../docs/USER_GUIDE.md)
