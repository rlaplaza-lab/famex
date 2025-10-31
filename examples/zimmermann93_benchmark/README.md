# Zimmermann-93 NEB / Two-ended TS Benchmark

Comprehensive evaluation of QME backends for two-ended transition state searches using the Zimmermann-93 dataset.

## Overview

Tests QME backends on two-ended (reactant→product) transition state searches, evaluating how well ML potentials locate TS when given both reactant and product structures.

**Methodology:**
1. Interpolates reaction paths (geodesic)
2. Identifies TS candidates (highest-energy images)
3. Optimizes transition states
4. Compares geometries (RMSD vs reference)
5. Analyzes success rates and accuracy

## Usage

```bash
# Run quick benchmark with representative subset
python zimmermann93_benchmark.py --quick

# Test specific backends
python zimmermann93_benchmark.py --backends uma mace

# Run comprehensive benchmark on all available backends
python zimmermann93_benchmark.py

# Analysis only (load existing results)
python zimmermann93_benchmark.py --analyze

python zimmermann93_benchmark.py --quick
```

### Command Line Options

- `--backends`: Comma-separated list of backends to test
- `--quick`: Run on a representative subset of reactions for faster testing
- `--analyze`: Load and analyze existing results without running new calculations
- `--device`: Device to use for calculations ("cpu" or "cuda")
- `--output`: Output file for results (default: benchmark_results/zimmermann93_benchmark_results.json)

## Zimmermann-93 Dataset

The benchmark uses a subset of the Zimmermann-93 dataset containing diverse organic reactions with:

- **Reactant structures** (`*_reactant.xyz`)
- **Product structures** (`*_product.xyz`)
- **Reference transition states** (`*_ts.xyz`)

Each reaction tests the ability to find transition states connecting known reactant and product structures.

## Output

Results are saved to `benchmark_results/zimmermann93_benchmark_results.json` and include:

- **Located TS structures** for all successful optimizations
- **RMSD values** comparing located vs. reference geometries
- **Success rates** for TS optimization convergence
- **Timing statistics** for performance analysis
- **Detailed logs** of optimization progress

### Example Output

```
🏆 BACKEND COMPARISON
Backend      Success  Avg RMSD (Å)  Median RMSD (Å)  Avg Time
-----------------------------------------------------------------
uma          8/10     0.45          0.32              12.3s
so3lr        7/10     0.52          0.41              8.7s
aimnet2      9/10     0.38          0.29              6.2s
```

## Directory Structure

```
zimmermann93_benchmark/
├── zimmermann93_benchmark.py     # Main benchmark script
├── README.md                     # This file
├── zimmermann93_dataset/         # Dataset files
│   ├── *_reactant.xyz           # Reactant structures
│   ├── *_product.xyz            # Product structures
│   └── *_ts.xyz                 # Reference TS structures
└── benchmark_results/            # Output directory
    └── zimmermann93_benchmark_results.json  # Results file
```

## Requirements

- QME package installed
- At least one ML backend (see [README](../../README.md))
- SELLA optimizer recommended (`pip install sella`)
- Python 3.10+

## Understanding Results

### Key Metrics

- **Success Rate**: Percentage of reactions where TS optimization converged
- **RMSD (Root Mean Square Deviation)**: Geometric similarity to reference TS (lower is better)
- **Timing**: Average time per TS optimization

### Interpretation

- **Low RMSD (< 0.5 Å)**: Excellent geometric agreement with reference
- **Medium RMSD (0.5-1.0 Å)**: Good agreement, likely chemically meaningful
- **High RMSD (> 1.0 Å)**: Poor agreement, may indicate optimization issues

### Backend Comparison

This benchmark complements the BH28 benchmark by focusing on:
- **Geometric accuracy** rather than energetic accuracy
- **Two-ended searches** rather than single-ended optimization
- **Path finding capabilities** of different ML potentials

## Notes

- Focuses on geometric accuracy (RMSD) vs BH28's energetic accuracy
- SELLA optimizer strongly recommended for reliable TS optimization
- Results may vary with interpolation quality and initial TS guesses
