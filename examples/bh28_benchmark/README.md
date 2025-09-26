# QME Examples

This directory contains examples demonstrating the capabilities of the QME (Quick Mechanistic Exploration) package.

## Comprehensive BH28 Benchmark

The main example is the **Comprehensive BH28 Benchmark** (`bh28_benchmark.py`), which provides a unified evaluation of all QME backends on the BH28 database of chemical reaction barrier heights.

### What the Benchmark Does

1. **Optimizes reactant minima** using various QME ML backends
2. **Optimizes transition states** using SELLA optimizer
3. **Calculates barrier heights** from optimized structures
4. **Compares accuracy** against reference values from high-level quantum chemistry
5. **Provides performance analysis** and backend recommendations

### Usage

```bash
# Run comprehensive benchmark on all available backends
python bh28_benchmark.py

# Quick test with representative subset of reactions
python bh28_benchmark.py --quick

# Test specific backends
python bh28_benchmark.py --backends mock so3lr uma

# Test specific reactions
python bh28_benchmark.py --reactions BHDIV_3 PXBH_2 CADBH_1

# Analysis only (load existing results)
python bh28_benchmark.py --analyze
```

### BH28 Dataset

The benchmark uses the BH28 dataset containing 28 diverse chemical reactions with reference barrier heights from CCSDT(Q)/CBS calculations:

- **PXBH**: Proton exchange barriers (H-abstraction reactions)
- **BHPERI**: Pericyclic reactions (Diels-Alder, Cope rearrangement, etc.)
- **CADBH**: Cyclization and addition barriers (bimolecular: small molecule + C2H4)
- **CRBH**: Cyclic radical barriers
- **BHDIV**: Diverse barrier heights (various reaction types)

### Output

Results are saved to `benchmark_results/bh28_benchmark_results.json` and include:

- **Optimized structures** for all reactants and transition states
- **Energy values** and barrier heights
- **Performance metrics** (MAE, RMSE, timing statistics)
- **Detailed analysis** comparing backend accuracy

### Example Output

```
🏆 BACKEND COMPARISON
Backend      Success  MAE (eV)   RMSE (eV)   Avg Time  
-----------------------------------------------------------------
so3lr        4/6      0.324      0.456       4.4s      
uma          5/6      0.245      0.312       2.1s      
aimnet2      6/6      0.156      0.198       1.8s      
```

## Supporting Files

- **`bh28_dataset/`**: XYZ structure files and reference barrier heights
- **`benchmark_results/`**: Output directory for results

## Requirements

- **QME package**: Base QME installation
- **ML backends**: At least one of: UMA (`fairchem-core`), SO3LR, or AIMNET2
- **Transition states**: SELLA optimizer (`pip install sella`) - recommended for proper TS optimization

## Citation

The BH28 dataset is from:
A. Karton, "Highly Accurate CCSDT(Q)/CBS Reaction Barrier Heights for a Diverse Set of Transition Structures", J. Phys. Chem. A 2019, 123, 6720-6729.
