# CLI Demo - Comprehensive Backend Comparison

Demonstrates QME's command-line interface capabilities by running various optimization tasks across all available ML backends and comparing their performance and reliability.

## Features

- Structure optimization using 'opt' command
- Transition state optimization using 'tsopt' command
- Two-ended optimization workflows
- NEB path optimization
- CI-NEB (Climbing Image NEB) path optimization
- Comprehensive backend performance comparison

## Usage

### Basic Usage
```bash
# Run with all available backends
python cli_demo.py

# Run with specific backends
python cli_demo.py --backends uma,aimnet2

# Run with verbose output
python cli_demo.py --verbose
```

### Advanced Usage
```bash
# Run on GPU
python cli_demo.py --device cuda

# Specify output file
python cli_demo.py --output my_results.json
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

The CLI demo tests various QME commands including NEB and CI-NEB path optimizations and reports success/failure rates for each backend.

### Results Format
- Success/failure counts per backend
- Total execution time
- Individual command execution times
- Overall performance summary

### Example Output
```
================================================================================
QME CLI Demo
Testing: opt, tsopt, two-ended, NEB, and CI-NEB commands
================================================================================

📋 Testing Backends
--------------------------------------------------
  1. uma
Total: 1 backends

[Detailed command execution...]

📊 UMA SUMMARY:
   ✅ Successful: 4
   ❌ Failed: 1
   ⏱️ Total time: 782.70s

✅ Overall demo successful! CLI commands working properly.
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
