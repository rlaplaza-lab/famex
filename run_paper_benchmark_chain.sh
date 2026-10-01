#!/usr/bin/env bash
# Wait for Zimmermann parallel orchestrator, then run extras + paper finalize.
set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${REPO_ROOT}/benchmark_runs/logs/zimmermann93_parallel/orchestrator.log"
ORCH_PID_FILE="${REPO_ROOT}/benchmark_runs/logs/zimmermann93_parallel/orchestrator.pid"
CHAIN_LOG="${REPO_ROOT}/benchmark_runs/logs/paper_chain.log"

mkdir -p "$(dirname "$CHAIN_LOG")"

echo "[$(date -Is)] paper chain started" | tee -a "$CHAIN_LOG"

# Wait until no run_zimmermann93_parallel / zimmermann workers remain
while pgrep -f "run_zimmermann93_parallel.py" >/dev/null 2>&1 \
   || pgrep -f "zimmermann93_benchmark.py" >/dev/null 2>&1; do
  python3 - <<'PY' | tee -a "$CHAIN_LOG"
from run_zimmermann93_parallel import discover_reactions, load_done
n=len(discover_reactions())
parts=[]
for b in ["aimnet2","uma","mace"]:
    parts.append(f"{b}={len(load_done(b))}/{n}")
print("progress:", " ".join(parts))
PY
  sleep 300
done

echo "[$(date -Is)] Zimmermann workers finished; merging" | tee -a "$CHAIN_LOG"
python3 "${REPO_ROOT}/run_zimmermann93_parallel.py" --merge-only | tee -a "$CHAIN_LOG"

echo "[$(date -Is)] Starting extra paper benchmarks" | tee -a "$CHAIN_LOG"
bash "${REPO_ROOT}/run_extra_paper_benchmarks.sh" | tee -a "$CHAIN_LOG" || true

echo "[$(date -Is)] Finalizing figures + manuscript" | tee -a "$CHAIN_LOG"
conda run --no-capture-output -n famex-aimnet2 \
  python "${REPO_ROOT}/finalize_paper_from_benchmarks.py" | tee -a "$CHAIN_LOG"

echo "[$(date -Is)] paper chain complete" | tee -a "$CHAIN_LOG"
