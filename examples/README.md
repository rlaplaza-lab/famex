# QME Examples

Examples and benchmarks demonstrating QME capabilities.

## Available Examples

| Example | Description |
|---------|-------------|
| `cli_demo.py` | CLI interface demo with backend comparison |
| `irc_demo.py` | IRC path calculation from transition state |
| `growing_string_demo.py` | Growing string method (DE-GSM) for TS search |
| `timing_benchmark.py` | ML backend performance comparison |
| `minima_optimizer_benchmark.py` | Minima optimizer comparison (lbfgs, bfgs, fire, trust-krylov) |
| `ts_optimizer_benchmark.py` | TS optimizer comparison (sella, trust-krylov-ts, rfo) |
| `thermochemistry_demo.py` | Thermochemistry capabilities (quasi-harmonic, solvation, etc.) |
| `uma_hessian_method_comparison.py` | Diagnostic script for UMA Hessian validation |
| `bh28_benchmark/` | Chemical accuracy evaluation (28 reactions) |
| `zimmermann93_benchmark/` | Two-ended TS search benchmark |

## Usage

Common options for all examples:
- `--backends`: Comma-separated backend list (default: all available)
- `--device`: cpu or cuda (default: auto-detect)
- `--verbose`: Detailed progress output
- `--help`: Show help message

**Quick start:**
```bash
python cli_demo.py
python irc_demo.py example_files/A_C_A_B_A_C_ts.xyz --backends uma
python timing_benchmark.py --device cuda
python bh28_benchmark/bh28_benchmark.py --quick
```

## Requirements

- QME package installed
- At least one ML backend (see [README](../README.md))
- SELLA optimizer is included with QME (core dependency)

## Supporting Files

- `example_utils/`: Standardized interface utilities
- `example_files/`: Sample molecular structures (XYZ format)

## Troubleshooting

**Backend not available:** Install backend dependencies (see [README](../README.md))

**CUDA not available:** Install PyTorch with CUDA or use `--device cpu`

See [Main Documentation](../docs/) for more information.
