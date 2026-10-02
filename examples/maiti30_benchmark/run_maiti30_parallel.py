#!/usr/bin/env python3
"""Parallel Maiti-30 runner across conda environments.

Same growing-string driver as Zimmermann-93, on the four OMol checkpoints that
cover transition metals. Skips AIMNet2 (no TM elements).

Skips reactions already present in benchmark_runs/{backend}/maiti30_benchmark_results.json.
Writes per-reaction shards under benchmark_runs/maiti30_shards/{backend}/ and merges
them back into the canonical JSON after each worker finishes.

Usage:
  python examples/maiti30_benchmark/run_maiti30_parallel.py
  python examples/maiti30_benchmark/run_maiti30_parallel.py --backends uma,mace
  python examples/maiti30_benchmark/run_maiti30_parallel.py --merge-only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "examples" / "maiti30_benchmark" / "maiti30_dataset"
BENCH_SCRIPT = REPO_ROOT / "examples" / "maiti30_benchmark" / "maiti30_benchmark.py"
BENCHMARK_ROOT = REPO_ROOT / "benchmark_runs"
SHARD_ROOT = BENCHMARK_ROOT / "maiti30_shards"
LOG_ROOT = BENCHMARK_ROOT / "logs" / "maiti30_parallel"

# Backend key -> (conda env, optional --model-name, result subdirectory)
BACKEND_SPEC: dict[str, tuple[str, str | None, str]] = {
    "uma": ("famex-uma", None, "uma"),
    "mace": ("famex-mace", None, "mace"),
    "pet": ("famex-benchmark-pet", "pet-omol-s", "pet_omol_s"),
    "orb": ("famex-orb", "orbmol-v2", "orbmol_v2"),
}

DEFAULT_WORKERS = {
    "uma": 1,
    "mace": 1,
    "pet": 1,
    "orb": 1,
}


def discover_reactions() -> list[str]:
    reactions = []
    for p in sorted(DATASET.glob("reaction_*_reactant.*")):
        reactions.append(p.stem.replace("_reactant", ""))
    return reactions


def result_key(backend: str) -> str:
    _env, model, _outdir = BACKEND_SPEC[backend]
    return f"{backend}:{model}" if model else backend


def load_done(backend: str) -> set[str]:
    done: set[str] = set()
    _env, model, outdir = BACKEND_SPEC[backend]
    canon = BENCHMARK_ROOT / outdir / "maiti30_benchmark_results.json"
    key = result_key(backend)
    if canon.exists():
        try:
            data = json.loads(canon.read_text())
            for candidate in (key, backend, f"{backend}:{model}" if model else None):
                if candidate and isinstance(data.get(candidate), dict):
                    done.update(k for k in data[candidate] if k.startswith("reaction_"))
        except (json.JSONDecodeError, OSError):
            pass
    shard_dir = SHARD_ROOT / backend
    for directory in (shard_dir, shard_dir / backend, shard_dir / key.replace(":", "_")):
        if directory.is_dir():
            for shard in directory.glob("reaction_*.json"):
                done.add(shard.stem)
    return done


def chunk_list(items: list[str], n_chunks: int) -> list[list[str]]:
    if n_chunks <= 0:
        return [items]
    n_chunks = min(n_chunks, max(1, len(items)))
    chunks: list[list[str]] = [[] for _ in range(n_chunks)]
    for i, item in enumerate(items):
        chunks[i % n_chunks].append(item)
    return [c for c in chunks if c]


def run_worker(
    backend: str,
    reactions: list[str],
    worker_id: int,
    device: str,
) -> tuple[str, int, int, str]:
    env_name, model, outdir = BACKEND_SPEC[backend]
    out_dir = BENCHMARK_ROOT / outdir
    shard_dir = SHARD_ROOT / backend
    canon = out_dir / "maiti30_benchmark_results.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    log_path = LOG_ROOT / f"{backend}_worker{worker_id}.log"
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        env_name,
        "python",
        str(BENCH_SCRIPT),
        "--backends",
        backend,
        "--device",
        device,
        "--output-dir",
        str(out_dir),
        "--shard-dir",
        str(shard_dir),
        "--canonical-results",
        str(canon),
        "--skip-existing",
        "--shards-only",
        "--reactions",
        *reactions,
    ]
    if model:
        cmd.extend(["--model-name", model])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    with open(log_path, "w") as log:
        log.write(f"CMD: {' '.join(cmd)}\n")
        log.write(f"Reactions ({len(reactions)}): {reactions}\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    return backend, worker_id, proc.returncode, str(log_path)


def merge_backend(backend: str) -> None:
    env_name, model, outdir = BACKEND_SPEC[backend]
    out_dir = BENCHMARK_ROOT / outdir
    shard_dir = SHARD_ROOT / backend
    canon = out_dir / "maiti30_benchmark_results.json"
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        env_name,
        "python",
        str(BENCH_SCRIPT),
        "--backends",
        backend,
        "--output-dir",
        str(out_dir),
        "--shard-dir",
        str(shard_dir),
        "--canonical-results",
        str(canon),
        "--merge-only",
    ]
    if model:
        cmd.extend(["--model-name", model])
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)


def run_backend(backend: str, workers: int, device: str) -> int:
    all_rxn = discover_reactions()
    done = load_done(backend)
    missing = [r for r in all_rxn if r not in done]
    print(f"\n=== {backend} ===")
    print(f"  dataset: {len(all_rxn)}  done: {len(done)}  missing: {len(missing)}")
    if not missing:
        print("  nothing to do; merging shards")
        merge_backend(backend)
        return 0

    chunks = chunk_list(missing, workers)
    print(f"  launching {len(chunks)} worker(s) on device={device}")
    failed = 0
    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        futures = {
            pool.submit(run_worker, backend, chunk, i, device): i for i, chunk in enumerate(chunks)
        }
        for fut in as_completed(futures):
            backend_name, wid, rc, log_path = fut.result()
            status = "OK" if rc == 0 else f"FAIL({rc})"
            print(f"  worker {wid} {status}  log={log_path}")
            if rc != 0:
                failed += 1
    merge_backend(backend)
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends",
        default="uma,mace,pet,orb",
        help="Comma-separated backends (default: uma,mace,pet,orb)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers-uma", type=int, default=DEFAULT_WORKERS["uma"])
    parser.add_argument("--workers-mace", type=int, default=DEFAULT_WORKERS["mace"])
    parser.add_argument("--workers-pet", type=int, default=DEFAULT_WORKERS["pet"])
    parser.add_argument("--workers-orb", type=int, default=DEFAULT_WORKERS["orb"])
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Only merge shards into canonical JSONs",
    )
    args = parser.parse_args()

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    for b in backends:
        if b not in BACKEND_SPEC:
            print(f"Unknown backend {b}; known: {list(BACKEND_SPEC)}", file=sys.stderr)
            return 1

    if not DATASET.exists():
        print(f"Dataset missing: {DATASET}", file=sys.stderr)
        return 1

    workers = {
        "uma": args.workers_uma,
        "mace": args.workers_mace,
        "pet": args.workers_pet,
        "orb": args.workers_orb,
    }

    print(f"Repo: {REPO_ROOT}")
    print(f"Reactions in dataset: {len(discover_reactions())}")
    print(f"Backends: {backends}")
    print(f"Device: {args.device}")

    if args.merge_only:
        for b in backends:
            merge_backend(b)
            done = load_done(b)
            print(f"  {b}: {len(done)} reactions after merge")
        return 0

    t0 = time.time()
    failed_total = 0

    # One backend at a time on the single GPU.
    for backend in backends:
        failed_total += run_backend(backend, workers[backend], args.device)

    elapsed = time.time() - t0
    print(f"\nAll backends finished in {elapsed / 3600:.2f} h  failures={failed_total}")
    for b in backends:
        done = load_done(b)
        print(f"  {b}: {len(done)}/{len(discover_reactions())} reactions")
    return 1 if failed_total else 0


if __name__ == "__main__":
    sys.exit(main())
