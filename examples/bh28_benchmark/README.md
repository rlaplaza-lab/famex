# BH28 Benchmark

Comprehensive evaluation of QME backends on the BH28 database of chemical reaction barrier heights.

## Overview

The **BH28 Benchmark** (`bh28_benchmark.py`) provides a unified evaluation of all QME backends on the BH28 database of chemical reaction barrier heights. This benchmark tests the accuracy of machine learning potentials for predicting reaction barriers across diverse chemical reactions.

## What the Benchmark Does

1. **Optimizes reactant minima** using various QME ML backends
2. **Optimizes transition states** using SELLA optimizer
3. **Calculates barrier heights** from optimized structures
4. **Compares accuracy** against reference values from high-level quantum chemistry
5. **Provides performance analysis** and backend recommendations

## Usage

```bash
# Run comprehensive benchmark on all available backends
python bh28_benchmark.py

# Quick test with representative subset of reactions
python bh28_benchmark.py --quick

# Test specific backends
python bh28_benchmark.py --backends so3lr uma

# Test specific reactions
python bh28_benchmark.py --reactions BHDIV_3 PXBH_2 CADBH_1

# Analysis only (load existing results)
python bh28_benchmark.py --analyze

python bh28_benchmark.py --quick
```

### Command Line Options

- `--backends`: Comma-separated list of backends to test
- `--reactions`: Comma-separated list of specific reactions to test
- `--quick`: Run on a representative subset of reactions for faster testing
- `--analyze`: Load and analyze existing results without running new calculations
- `--device`: Device to use for calculations ("cpu" or "cuda")
- `--output`: Output file for results (default: benchmark_results/bh28_benchmark_results.json)

## BH28 Dataset

The benchmark uses the BH28 dataset containing 28 diverse chemical reactions with reference barrier heights from CCSDT(Q)/CBS calculations:

- **PXBH**: Proton exchange barriers (H-abstraction reactions)
- **BHPERI**: Pericyclic reactions (Diels-Alder, Cope rearrangement, etc.)
- **CADBH**: Cyclization and addition barriers (bimolecular: small molecule + C2H4)
- **CRBH**: Cyclic radical barriers
- **BHDIV**: Diverse barrier heights (various reaction types)

Each reaction includes:
- Reactant minimum structure (`*_min.xyz`)
- Transition state structure (`*_ts.xyz`)
- Reference barrier height from high-level quantum chemistry

## Output

Results are saved to `benchmark_results/bh28_benchmark_results.json` and include:

- **Optimized structures** for all reactants and transition states
- **Energy values** and calculated barrier heights
- **Performance metrics** (MAE, RMSE, timing statistics)
- **Detailed analysis** comparing backend accuracy
- **Success rates** for optimization convergence

### Example Output

```
🏆 BACKEND COMPARISON
Backend      Success  MAE (eV)   RMSE (eV)   Avg Time
-----------------------------------------------------------------
so3lr        4/6      0.324      0.456       4.4s
uma          5/6      0.245      0.312       2.1s
aimnet2      6/6      0.156      0.198       1.8s
```

## Directory Structure

```
bh28_benchmark/
├── bh28_benchmark.py          # Main benchmark script
├── README.md                  # This file
├── bh28_dataset/             # Dataset files
│   ├── *_min.xyz            # Reactant minimum structures
│   ├── *_ts.xyz             # Transition state structures
│   └── reference_barrier_heights.json  # Reference values
└── benchmark_results/        # Output directory
    └── bh28_benchmark_results.json     # Results file
```

## Requirements

- **QME package**: Base QME installation
- **ML backends**: At least one of: UMA (`fairchem-core`), SO3LR, AIMNet2, MACE
- **Transition states**: SELLA optimizer (`pip install sella`) - recommended for proper TS optimization
- **Python environment**: Python 3.10+ recommended

## Understanding Results

### Performance Metrics

- **Success Rate**: Percentage of reactions where optimization converged
- **MAE (Mean Absolute Error)**: Average absolute difference from reference barrier heights
- **RMSE (Root Mean Square Error)**: Root mean square difference from reference
- **Timing**: Average time per reaction optimization

### Backend Recommendations

- **High Accuracy**: AIMNet2 typically shows lowest errors
- **Speed**: UMA often provides good speed/accuracy balance
- **Robustness**: SO3LR may be more stable for difficult cases

## Citation

The BH28 dataset is from:
A. Karton, "Highly Accurate CCSDT(Q)/CBS Reaction Barrier Heights for a Diverse Set of Transition Structures", J. Phys. Chem. A 2019, 123, 6720-6729.

## Notes

- The benchmark automatically detects available backends
- Failed optimizations are reported but don't stop the benchmark
- Results can be analyzed separately using the `--analyze` flag
- For reproducibility, use the same conda environment across runs
