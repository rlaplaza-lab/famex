#!/usr/bin/env bash
set -u
set -o pipefail

# Full benchmark runner for BH28 + Zimmermann93 across multiple backends
# (excluding tblite by default), using isolated conda environments.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${REPO_ROOT}/benchmark_runs/full_multienv"
LOG_ROOT="${OUTPUT_ROOT}/logs"
DEVICE="cuda"
SKIP_ENV_SETUP=0
RECREATE_ENVS=0

DEFAULT_BACKENDS=("aimnet2" "mace" "uma" "so3lr" "orb" "pet")
BACKENDS=("${DEFAULT_BACKENDS[@]}")

usage() {
  cat <<'EOF'
Usage:
  bash run_full_multienv_benchmarks.sh [options]

Options:
  --device <cpu|cuda>           Device passed to BH28 benchmark (default: cuda)
  --output-root <path>          Output root directory
  --backends <csv>              Comma-separated backends (default: aimnet2,mace,uma,so3lr,orb,pet)
  --skip-env-setup              Reuse existing conda envs (no env creation/install)
  --recreate-envs               Remove and recreate envs before installation
  -h, --help                    Show this help

Examples:
  bash run_full_multienv_benchmarks.sh
  bash run_full_multienv_benchmarks.sh --device cpu --skip-env-setup
  bash run_full_multienv_benchmarks.sh --backends uma,mace,aimnet2
EOF
}

join_by() {
  local delimiter="$1"
  shift
  local first=1 out=""
  for item in "$@"; do
    if [[ $first -eq 1 ]]; then
      out="$item"
      first=0
    else
      out="${out}${delimiter}${item}"
    fi
  done
  printf '%s' "$out"
}

contains_backend() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      LOG_ROOT="${OUTPUT_ROOT}/logs"
      shift 2
      ;;
    --backends)
      IFS=',' read -r -a BACKENDS <<< "$2"
      shift 2
      ;;
    --skip-env-setup)
      SKIP_ENV_SETUP=1
      shift
      ;;
    --recreate-envs)
      RECREATE_ENVS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$DEVICE" != "cpu" && "$DEVICE" != "cuda" ]]; then
  echo "Invalid --device value: $DEVICE (must be cpu or cuda)" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH. Install conda/mamba first." >&2
  exit 1
fi

if [[ ${#BACKENDS[@]} -eq 0 ]]; then
  echo "No backends requested." >&2
  exit 1
fi

for b in "${BACKENDS[@]}"; do
  if [[ "$b" == "tblite" || "$b" == "mock" ]]; then
    echo "Refusing backend '$b'. This runner excludes tblite and mock." >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

declare -a SUMMARY_LINES=()
FAILED=0

setup_env() {
  local backend="$1"
  local env_name="famex-benchmark-${backend}"
  local pyver="3.10"

  if [[ "$backend" == "pet" ]]; then
    pyver="3.11"
  fi

  if [[ "$SKIP_ENV_SETUP" -eq 1 ]]; then
    echo "[env:$env_name] SKIP_ENV_SETUP=1, reusing existing environment"
    return 0
  fi

  local env_list
  env_list="$(conda env list)"
  if echo "$env_list" | rg -q "^[^#]*[[:space:]]${env_name}[[:space:]]"; then
    if [[ "$RECREATE_ENVS" -eq 1 ]]; then
      echo "[env:$env_name] Removing existing environment (recreate requested)"
      conda env remove -n "$env_name" -y >/dev/null
    else
      echo "[env:$env_name] Environment already exists; updating dependencies"
    fi
  fi

  env_list="$(conda env list)"
  if ! echo "$env_list" | rg -q "^[^#]*[[:space:]]${env_name}[[:space:]]"; then
    echo "[env:$env_name] Creating python=${pyver}"
    conda create -n "$env_name" "python=${pyver}" -y >/dev/null
  fi

  echo "[env:$env_name] Installing famex editable + backend deps"
  conda run -n "$env_name" pip install -e "$REPO_ROOT"

  case "$backend" in
    aimnet2)
      conda run -n "$env_name" pip install torch
      ;;
    mace)
      conda run -n "$env_name" pip install mace-torch
      ;;
    uma)
      conda run -n "$env_name" pip install "fairchem-core>=2.21.0" torch
      ;;
    so3lr)
      conda run -n "$env_name" pip install so3lr
      ;;
    orb)
      conda run -n "$env_name" pip install orb-models torch
      ;;
    pet)
      conda run -n "$env_name" pip install upet torch
      ;;
    *)
      echo "Unsupported backend: $backend" >&2
      return 1
      ;;
  esac
}

run_one_suite() {
  local env_name="$1"
  local suite="$2"
  local backend="$3"
  local log_file="$4"
  shift 4
  local cmd=("$@")

  echo "[run:$backend:$suite] ${cmd[*]}"
  if "${cmd[@]}" 2>&1 | tee "$log_file"; then
    SUMMARY_LINES+=("$backend,$suite,OK,$log_file")
    return 0
  fi
  SUMMARY_LINES+=("$backend,$suite,FAIL,$log_file")
  FAILED=1
  return 1
}

echo "Repo root:     $REPO_ROOT"
echo "Output root:   $OUTPUT_ROOT"
echo "Log root:      $LOG_ROOT"
echo "Device(BH28):  $DEVICE"
echo "Backends:      $(join_by "," "${BACKENDS[@]}")"
echo "Skip env setup:$SKIP_ENV_SETUP"
echo "Recreate envs: $RECREATE_ENVS"
echo

for backend in "${BACKENDS[@]}"; do
  env_name="famex-benchmark-${backend}"
  backend_root="${OUTPUT_ROOT}/${backend}"
  bh28_out="${backend_root}/bh28"
  z93_out="${backend_root}/zimmermann93"
  mkdir -p "$bh28_out" "$z93_out"

  echo "=================================================================="
  echo "Backend: ${backend} (env: ${env_name})"
  echo "=================================================================="

  if ! setup_env "$backend"; then
    echo "[env:$env_name] Setup failed; skipping both suites for $backend" >&2
    SUMMARY_LINES+=("$backend,bh28,FAIL,env_setup")
    SUMMARY_LINES+=("$backend,zimmermann93,FAIL,env_setup")
    FAILED=1
    continue
  fi

  bh28_log="${LOG_ROOT}/${backend}_bh28.log"
  z93_log="${LOG_ROOT}/${backend}_zimmermann93.log"

  run_one_suite \
    "$env_name" \
    "bh28" \
    "$backend" \
    "$bh28_log" \
    conda run -n "$env_name" python "$REPO_ROOT/examples/bh28_benchmark/bh28_benchmark.py" \
      --backends "$backend" \
      --output-dir "$bh28_out" \
      --device "$DEVICE"

  run_one_suite \
    "$env_name" \
    "zimmermann93" \
    "$backend" \
    "$z93_log" \
    conda run -n "$env_name" python "$REPO_ROOT/examples/zimmermann93_benchmark/zimmermann93_benchmark.py" \
      --backends "$backend" \
      --output-dir "$z93_out"
done

echo
echo "============================== SUMMARY =============================="
printf "%-12s %-14s %-6s %s\n" "Backend" "Suite" "State" "Log/Reason"
printf "%-12s %-14s %-6s %s\n" "------------" "--------------" "------" "----------"
for line in "${SUMMARY_LINES[@]}"; do
  IFS=',' read -r b s st lr <<< "$line"
  printf "%-12s %-14s %-6s %s\n" "$b" "$s" "$st" "$lr"
done
echo "===================================================================="
echo "Results root: $OUTPUT_ROOT"
echo

if [[ "$FAILED" -ne 0 ]]; then
  echo "One or more benchmark runs failed." >&2
  exit 1
fi

echo "All requested benchmark runs completed successfully."
exit 0
