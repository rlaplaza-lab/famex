# [Example Name] - [Brief Description]

[One-line description of what this example does and why it's useful]

## Features

- [Feature 1]: [Description]
- [Feature 2]: [Description]
- [Feature 3]: [Description]

## Usage

### Basic Usage
```bash
# Run with all available backends
conda run -n py312 python [example_name].py

# Run with specific backends
conda run -n py312 python [example_name].py --backends uma,aimnet2

# Run with verbose output
conda run -n py312 python [example_name].py --verbose
```

### Advanced Usage
```bash
# Run on GPU
conda run -n py312 python [example_name].py --device cuda

# Specify output file
conda run -n py312 python [example_name].py --output my_results.json

# Quick test mode (if applicable)
conda run -n py312 python [example_name].py --quick
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

[Description of what the example produces]

### Results Format
[Description of output format, files created, etc.]

### Example Output
```
[Sample output showing what users can expect]
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

