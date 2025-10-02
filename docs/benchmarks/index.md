# Benchmarks and Examples

QME includes comprehensive benchmarks and examples to evaluate backend performance and demonstrate usage patterns.

## Available Benchmarks

### [BH28 Benchmark](bh28.md)
Comprehensive evaluation of QME backends on the BH28 database of chemical reaction barrier heights.

**Features:**
- Tests accuracy of ML potentials for predicting reaction barriers
- 28 diverse chemical reactions with reference values from high-level quantum chemistry
- Compares backend performance across different reaction types

### [Zimmermann-93 Benchmark](zimmermann93.md)  
Two-ended transition state search benchmark using the Zimmermann-93 dataset.

**Features:**
- Tests reactant→product transition state finding
- Compares located TS geometries with reference structures
- Evaluates NEB-like path optimization capabilities

### [Performance Benchmarks](performance.md)
Comprehensive timing and performance analysis for all QME backends.

**Features:**
- Geometry optimization timing
- Frequency analysis performance
- Individual energy/force calculation benchmarks
- Memory usage and GPU utilization analysis

## Example Gallery

### [Basic Examples](examples.md)
Collection of common QME usage patterns and workflows.

**Includes:**
- Simple optimization examples
- Transition state search examples
- Batch processing workflows
- Integration with other tools

## Running Benchmarks

All benchmarks are located in the `examples/` directory and can be run independently:

```bash
# Quick performance benchmark
cd examples
python timing_benchmark.py --backends uma aimnet2 --device cuda

# BH28 accuracy benchmark
cd examples/bh28_benchmark
python bh28_benchmark.py --quick

# Zimmermann-93 TS benchmark
cd examples/zimmermann93_benchmark  
python zimmermann93_benchmark.py --quick
```

## Benchmark Results Interpretation

### Accuracy Metrics
- **MAE (Mean Absolute Error)**: Average absolute difference from reference values
- **RMSE (Root Mean Square Error)**: Root mean square difference from reference
- **Success Rate**: Percentage of calculations that converged successfully

### Performance Metrics
- **Optimization Time**: Time to converge optimization
- **Energy Evaluation Time**: Time for single energy/force calculation
- **Memory Usage**: Peak memory consumption during calculation
- **GPU Utilization**: Efficiency of GPU usage (when applicable)

### Backend Comparison Guidelines

#### For Accuracy (BH28 Results)
Lower MAE/RMSE values indicate better agreement with high-level quantum chemistry:
- **Excellent**: MAE < 0.1 eV
- **Good**: MAE 0.1-0.3 eV  
- **Acceptable**: MAE 0.3-0.5 eV
- **Poor**: MAE > 0.5 eV

#### For Geometry (Zimmermann-93 Results)
Lower RMSD values indicate better geometric agreement:
- **Excellent**: RMSD < 0.3 Å
- **Good**: RMSD 0.3-0.6 Å
- **Acceptable**: RMSD 0.6-1.0 Å
- **Poor**: RMSD > 1.0 Å

#### For Performance (Timing Results)
Faster backends enable larger-scale studies:
- **Very Fast**: < 1 second per optimization
- **Fast**: 1-10 seconds per optimization
- **Moderate**: 10-60 seconds per optimization
- **Slow**: > 60 seconds per optimization

## Reproducibility

### Environment Requirements
- Python 3.10+ (3.11+ for TorchSim)
- Specific backend dependencies
- Consistent hardware for performance comparisons

### Running Reproducible Benchmarks
```bash
# Use consistent environment
conda create -n qme-bench python=3.12
conda activate qme-bench
pip install qme-ml-ml[uma,aimnet2,mace]

# Run with fixed random seeds (where applicable)
python timing_benchmark.py --seed 42

# Save detailed results
python bh28_benchmark.py --output detailed_results.json
```

### Version Tracking
Benchmark results depend on:
- QME version
- Backend package versions
- PyTorch version
- Hardware specifications

Always record these details when comparing results.

## Contributing Benchmarks

We welcome contributions of new benchmarks! See the [Developer Guide](../developer_guide/contributing.md) for guidelines on:

- Adding new benchmark datasets
- Implementing performance tests
- Documentation standards
- Code review process

## Citation

If you use QME benchmarks in your research, please cite:

```bibtex
@software{qme2025,
  title={QME: Quick Mechanistic Exploration},
  author={QME Development Team},
  year={2025},
  url={https://github.com/rlaplaza-lab/qme}
}
```

Additionally, please cite the original benchmark datasets:
- **BH28**: Karton, A. J. Phys. Chem. A 2019, 123, 6720-6729
- **Zimmermann-93**: Original dataset references in individual benchmark documentation
