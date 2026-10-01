#!/usr/bin/env python3
"""Parallel Zimmermann-93 runner across conda environments.

Skips reactions already present in benchmark_runs/{backend}/zimmermann93_benchmark_results.json.
Writes per-reaction shards under benchmark_runs/zimmermann93_shards/{backend}/ and merges
them back into the canonical JSON after each worker finishes.

Default schedule (single 16 GB GPU):
  - aimnet2: up to 2 workers
  - uma: 1 worker (can overlap with aimnet2 if VRAM allows; default is sequential backends)
  - mace: 1 worker alone

Usage:
  python run_zimmermann93_parallel.py
  python run_zimmermann93_parallel.py --backends aimnet2,uma --workers-aimnet2 2
  python run_zimmermann93_parallel.py --merge-only
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

REPO_ROOT = Path(__file__).resolve().parent
DATASET = REPO_ROOT / "examples" / "zimmermann93_benchmark" / "zimmermann93_dataset"
BENCH_SCRIPT = (
    REPO_ROOT / "examples" / "zimmermann93_benchmark" / "zimmermann93_benchmark.py"
)
BENCHMARK_ROOT = REPO_ROOT / "benchmark_runs"
SHARD_ROOT = BENCHMARK_ROOT / "zimmermann93_shards"
LOG_ROOT = BENCHMARK_ROOT / "logs" / "zimmermann93_parallel"

ENV_MAP = {
    "aimnet2": "famex-aimnet2",
    "uma": "famex-uma",
    "mace": "famex-mace",
}

DEFAULT_WORKERS = {
    "aimnet2": 2,
    "uma": 1,
    "mace": 1,
}


def discover_reactions() -> list[str]:
    reactions = []
    for p in sorted(DATASET.glob("reaction_*_reactant.*")):
        reactions.append(p.stem.replace("_reactant", ""))
    return reactions


def load_done(backend: str) -> set[str]:
    done: set[str] = set()
    canon = BENCHMARK_ROOT / backend / "zimmermann93_benchmark_results.json"
    if canon.exists():
        try:
            data = json.loads(canon.read_text())
            blob = data.get(backend, {})
            if isinstance(blob, dict):
                done.update(k for k in blob if k.startswith("reaction_"))
        except (json.JSONDecodeError, OSError):
            pass
    shard_dir = SHARD_ROOT / backend
    # Flat layout (preferred) and nested layout from early runs
    for directory in (shard_dir, shard_dir / backend):
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
    """Run one worker process. Returns (backend, worker_id, returncode, log_path)."""
    env_name = ENV_MAP[backend]
    out_dir = BENCHMARK_ROOT / backend
    shard_dir = SHARD_ROOT / backend
    canon = out_dir / "zimmermann93_benchmark_results.json"
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
    env = os.environ.copy()
    # Give each worker a unique CUDA MPS-friendly identity; sharing one GPU.
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
    env_name = ENV_MAP[backend]
    out_dir = BENCHMARK_ROOT / backend
    shard_dir = SHARD_ROOT / backend
    canon = out_dir / "zimmermann93_benchmark_results.json"
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
            pool.submit(run_worker, backend, chunk, i, device): i
            for i, chunk in enumerate(chunks)
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
        default="aimnet2,uma,mace",
        help="Comma-separated backends (default: aimnet2,uma,mace)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers-aimnet2", type=int, default=DEFAULT_WORKERS["aimnet2"])
    parser.add_argument("--workers-uma", type=int, default=DEFAULT_WORKERS["uma"])
    parser.add_argument("--workers-mace", type=int, default=DEFAULT_WORKERS["mace"])
    parser.add_argument(
        "--overlap-aimnet2-uma",
        action="store_true",
        help="Run aimnet2 and uma backends concurrently (higher VRAM use)",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Only merge shards into canonical JSONs",
    )
    args = parser.parse_args()

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    for b in backends:
        if b not in ENV_MAP:
            print(f"Unknown backend {b}; known: {list(ENV_MAP)}", file=sys.stderr)
            return 1

    workers = {
        "aimnet2": args.workers_aimnet2,
        "uma": args.workers_uma,
        "mace": args.workers_mace,
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

    # Schedule: optionally overlap aimnet2+uma, then mace alone.
    remaining = list(backends)
    if args.overlap_aimnet2_uma and "aimnet2" in remaining and "uma" in remaining:
        print("\n--- Overlapping aimnet2 + uma ---")
        with ThreadPoolExecutor(max_workers=2) as pool:
            futs = [
                pool.submit(run_backend, "aimnet2", workers["aimnet2"], args.device),
                pool.submit(run_backend, "uma", workers["uma"], args.device),
            ]
            for fut in as_completed(futs):
                failed_total += fut.result()
        remaining = [b for b in remaining if b not in ("aimnet2", "uma")]

    for backend in remaining:
        failed_total += run_backend(backend, workers[backend], args.device)

    elapsed = time.time() - t0
    print(f"\nAll backends finished in {elapsed / 3600:.2f} h  failures={failed_total}")
    for b in backends:
        done = load_done(b)
        print(f"  {b}: {len(done)}/{len(discover_reactions())} reactions")
    return 1 if failed_total else 0


if __name__ == "__main__":
    sys.exit(main())
