# Zimmermann-93 Growing-String TS Benchmark

Two-ended growing-string transition-state search on a subset of the Zimmermann-93 dataset.

## Usage

From this directory:

```bash
python zimmermann93_benchmark.py
python zimmermann93_benchmark.py --quick
python zimmermann93_benchmark.py --quicker
python zimmermann93_benchmark.py --backends uma,mace
python zimmermann93_benchmark.py --merge-only
```

From the repo root, the parallel runner writes `benchmark_runs/{backend}/zimmermann93_benchmark_results.json`, shards under `benchmark_runs/zimmermann93_shards/`, and logs under `benchmark_runs/logs/zimmermann93_parallel/`:

```bash
python examples/zimmermann93_benchmark/run_zimmermann93_parallel.py
```

The single-process script still defaults to `benchmark_results/zimmermann93_benchmark_results.json` in the working directory.

### Command Line Options

- `--backends`: Comma-separated list of backends to test
- `--quick`: Run on a representative subset of reactions for faster testing
- `--quicker`: Run a single reaction
- `--device`: Device to use for calculations ("cpu" or "cuda")
- `--output-dir`: Directory for results (default: `benchmark_results/`). The canonical file is `zimmermann93_benchmark_results.json`
- `--merge-only`: Merge per-reaction shards into the canonical results file and exit
- `--npoints`: Number of nodes on the growing string (default: 11)

## Dataset

- Reactant structures (`*_reactant.xyz`)
- Product structures (`*_product.xyz`)
- Reference transition states (`*_ts.xyz`)
