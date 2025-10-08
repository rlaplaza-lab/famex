# Zimmermann-93 NEB / Two-ended TS Benchmark

Comprehensive evaluation of QME backends for two-ended transition state searches using the Zimmermann-93 dataset.

## Overview

The **Zimmermann-93 Benchmark** (`zimmermann93_benchmark.py`) tests QME backends on two-ended (reactant→product) transition state searches. This benchmark evaluates how well machine learning potentials can locate transition states when given both reactant and product structures.

## What the Benchmark Does

1. **Interpolates reaction paths** between reactant and product structures (geodesic interpolation)
2. **Optimizes paths** with simplified NEB-like routines (when available)
3. **Identifies TS candidates** by selecting highest-energy interpolated images
4. **Optimizes transition states** using QME's Explorer with transition state optimization
5. **Compares geometries** by computing RMSD between located and reference TS structures
6. **Analyzes success rates** and geometric accuracy across backends

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

Each reaction tests the ability to find transition states connecting known reactant and product structures, which is a common scenario in computational chemistry.

## Methodology

### Path Interpolation
1. **Geodesic interpolation** between reactant and product geometries
2. **Multiple images** generated along the reaction coordinate
3. **Energy evaluation** at each interpolated point

### TS Optimization
1. **Highest-energy image** selected as initial TS guess
2. **QME transition state optimization** using available methods:
   - **SELLA optimizer** (recommended, requires `pip install sella`)
   - **Fallback methods** for basic TS optimization
3. **Convergence criteria** applied for reliable TS location

### Analysis
1. **RMSD calculation** between located and reference TS geometries
2. **Success rate** tracking for optimization convergence
3. **Performance comparison** across different backends

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

- **QME package**: Base QME installation
- **ML backends**: At least one of: UMA (`fairchem-core`), SO3LR, AIMNet2, MACE
- **Transition states**: SELLA optimizer (`pip install sella`) - highly recommended for reliable TS optimization
- **Python environment**: Python 3.10+ recommended

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

- This benchmark mirrors the behavior of `bh28_benchmark.py` but focuses on geometry comparison
- The benchmark will fail early if no ML backends are available
- SELLA optimizer is strongly recommended for reliable transition state optimization
- Results may vary depending on interpolation quality and initial TS guesses
- For reproducibility, use the same conda environment and backend versions across runs

## Relationship to Other Benchmarks

- **BH28 Benchmark**: Tests energetic accuracy of barrier heights
- **Zimmermann-93 Benchmark**: Tests geometric accuracy of TS structures
- **Timing Benchmark**: Tests computational performance and optimization efficiency

Together, these benchmarks provide comprehensive evaluation of QME backend capabilities.
