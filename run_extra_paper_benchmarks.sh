#!/usr/bin/env bash
# Extra paper benchmarks: MACE-OFF BH28 + optional so3lr/orb/pet BH28+timing.
# Intended to run when the GPU is free (after Zimmermann parallel completes).
set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${REPO_ROOT}/benchmark_runs"
LOG_ROOT="${OUTPUT_ROOT}/logs/extra_paper"
DEVICE="${DEVICE:-cuda}"
mkdir -p "$LOG_ROOT"

run_logged() {
  local label="$1"
  shift
  local log="${LOG_ROOT}/${label}.log"
  echo "[$(date -Is)] START ${label}: $*" | tee -a "${LOG_ROOT}/summary.log"
  if "$@" 2>&1 | tee "$log"; then
    echo "[$(date -Is)] OK ${label}" | tee -a "${LOG_ROOT}/summary.log"
    return 0
  fi
  echo "[$(date -Is)] FAIL ${label} (see ${log})" | tee -a "${LOG_ROOT}/summary.log"
  return 1
}

FAILED=0

# --- MACE-OFF medium on full BH28 (reuse famex-mace env) ---
MACE_OFF_OUT="${OUTPUT_ROOT}/mace_off_medium"
mkdir -p "$MACE_OFF_OUT"
if [[ -f "${MACE_OFF_OUT}/bh28_benchmark_results.json" ]]; then
  echo "Skipping mace-off BH28 (results already exist)"
else
  run_logged mace_off_bh28 \
    conda run --no-capture-output -n famex-mace \
      python "${REPO_ROOT}/examples/bh28_benchmark/bh28_benchmark.py" \
        --backends "mace:mace-off-medium" \
        --device "$DEVICE" \
        --output-dir "$MACE_OFF_OUT" || FAILED=1
fi

# --- Create and run optional backends (BH28 + timing only) ---
setup_and_run_extra() {
  local backend="$1"
  local env_name="famex-benchmark-${backend}"
  local pyver="3.10"
  local pip_pkgs=""
  local model_spec="$backend"

  case "$backend" in
    so3lr) pip_pkgs="so3lr" ;;
    orb)   pip_pkgs="orb-models>=0.7.0"; pyver="3.12"; model_spec="orb:orbmol-v2" ;;
    pet)   pip_pkgs="upet torch"; pyver="3.11" ;;
    *) echo "Unknown backend $backend"; return 1 ;;
  esac

  if ! conda env list | rg -q "^[^#]*[[:space:]]${env_name}[[:space:]]"; then
    echo "Creating env ${env_name} (python=${pyver})"
    conda create -n "$env_name" "python=${pyver}" -y || return 1
    conda run -n "$env_name" pip install -e "$REPO_ROOT" || return 1
    # shellcheck disable=SC2086
    conda run -n "$env_name" pip install $pip_pkgs || return 1
  else
    if [[ "$backend" == "orb" ]] && ! conda run -n "$env_name" python -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'; then
      echo "Env ${env_name} is older than Python 3.12. OrbMol-v2 needs orb-models>=0.7.0; recreate the env or use famex-orb."
      return 1
    fi
    echo "Reusing env ${env_name}"
  fi

  local out="${OUTPUT_ROOT}/${backend}"
  mkdir -p "$out"
  if [[ ! -f "${out}/bh28_benchmark_results.json" ]]; then
    run_logged "${backend}_bh28" \
      conda run --no-capture-output -n "$env_name" \
        python "${REPO_ROOT}/examples/bh28_benchmark/bh28_benchmark.py" \
        --backends "$model_spec" \
        --device "$DEVICE" \
        --output-dir "$out" || return 1
  else
    echo "Skipping ${backend} BH28 (exists)"
  fi

  if [[ ! -f "${out}/timing_benchmark_results.json" ]]; then
    run_logged "${backend}_timing" \
      conda run --no-capture-output -n "$env_name" \
        python "${REPO_ROOT}/examples/timing_benchmark.py" \
          --backends "$model_spec" \
          --device "$DEVICE" \
          --output "${out}/timing_benchmark_results.json" || return 1
  else
    echo "Skipping ${backend} timing (exists)"
  fi
  return 0
}

for backend in so3lr orb pet; do
  if setup_and_run_extra "$backend"; then
    echo "Extra backend ${backend}: OK"
  else
    echo "Extra backend ${backend}: FAILED (recorded; continuing)"
    FAILED=1
  fi
done

echo "Extra paper benchmarks finished. FAILED=${FAILED}"
exit "$FAILED"
