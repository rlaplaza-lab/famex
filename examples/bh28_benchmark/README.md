# BH28 Benchmark

Barrier heights on the BH28 set (28 reactions, CCSDT(Q)/CBS references).

## Usage

```bash
python bh28_benchmark.py
python bh28_benchmark.py --quick
python bh28_benchmark.py --backends so3lr,uma
python bh28_benchmark.py --reactions BHDIV_3 PXBH_2 CADBH_1
python bh28_benchmark.py --quicker
python bh28_benchmark.py --analyze
```

### Command Line Options

- `--backends`: Comma-separated list of backends to test
- `--reactions`: Specific reactions to test (default: all 28)
- `--quick`: Run on a representative subset of reactions for faster testing
- `--quicker`: Run a single reaction
- `--analyze`: Load and analyze existing results without running new calculations
- `--device`: Device to use for calculations ("cpu" or "cuda")
- `--output-dir`: Directory for results (default: `benchmark_results/`). The results file is `bh28_benchmark_results.json`
- `--minima-optimizer`: Minima optimizer (`LBFGS`, `BFGS`, or `FIRE`; default `LBFGS`)
- `--ts-optimizer`: Transition-state optimizer (`SELLA` or `rfo`; default `SELLA` for this script)

## Dataset

28 reactions with reference barrier heights from CCSDT(Q)/CBS:

- **PXBH**: Proton exchange barriers (H-abstraction reactions)
- **BHPERI**: Pericyclic reactions (Diels-Alder, Cope rearrangement, etc.)
- **CADBH**: Cyclization and addition barriers (bimolecular: small molecule + C2H4)
- **CRBH**: Cyclic radical barriers
- **BHDIV**: Diverse barrier heights (various reaction types)

Each reaction has a reactant minimum (`*_min.xyz`), a transition state (`*_ts.xyz`), and a reference barrier in `reference_barrier_heights.json`.

Results go to `benchmark_results/bh28_benchmark_results.json`.

For TS optimizer comparison on these structures, use `../ts_optimizer_benchmark.py` with `--bh28-subset` or `--full-bh28`.

## Citation

A. Karton, "Highly Accurate CCSDT(Q)/CBS Reaction Barrier Heights for a Diverse Set of Transition Structures", J. Phys. Chem. A 2019, 123, 6720-6729.
