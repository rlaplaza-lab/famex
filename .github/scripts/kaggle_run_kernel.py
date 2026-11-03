#!/usr/bin/env python3
"""Push a Kaggle GPU kernel, wait for completion, and report parsed logs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from kaggle_log_common import parse_kaggle_log, redact_text


def run_cmd(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.stdout:
        print(
            redact_text(completed.stdout),
            end="" if completed.stdout.endswith("\n") else "\n",
        )
    if completed.stderr:
        print(
            redact_text(completed.stderr),
            end="" if completed.stderr.endswith("\n") else "\n",
            file=sys.stderr,
        )
    if check and completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode, cmd, output=completed.stdout, stderr=completed.stderr
        )
    return completed


def _is_gpu_session_limit(text: str) -> bool:
    lowered = text.lower()
    return "maximum batch gpu session" in lowered or "gpu session count" in lowered


def write_kernel_metadata(
    *,
    staging_dir: Path,
    slug: str,
    title: str,
    code_file: Path,
    dataset_sources: list[str],
    timeout_seconds: int,
    machine_shape: str,
    push_retries: int = 60,
    push_retry_seconds: int = 120,
) -> None:
    run_cmd(["kaggle", "kernels", "init", "-p", str(staging_dir)])
    metadata_path = staging_dir / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    owner = str(metadata.get("id", slug)).split("/")[0]
    metadata["id"] = f"{owner}/{slug}"
    metadata["title"] = title
    metadata["code_file"] = str(code_file.resolve())
    metadata["language"] = "python"
    metadata["kernel_type"] = "script"
    metadata["is_private"] = True
    metadata["enable_gpu"] = True
    metadata["enable_tpu"] = False
    metadata["enable_internet"] = True
    metadata["dataset_sources"] = dataset_sources
    if machine_shape:
        metadata["machine_shape"] = machine_shape
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    push_cmd = ["kaggle", "kernels", "push", "-p", str(staging_dir)]
    if timeout_seconds > 0:
        push_cmd.extend(["--timeout", str(timeout_seconds)])

    # Kaggle caps concurrent batch GPU sessions at 2. Retry when another
    # famex/scgo kernel (or a cancelled CI leftover) still holds a slot.
    # Note: newer kaggle CLIs may print the limit error and still exit 0.
    last: subprocess.CompletedProcess[str] | None = None
    for attempt in range(1, push_retries + 1):
        last = run_cmd(push_cmd, check=False)
        combined = f"{last.stdout or ''}\n{last.stderr or ''}"
        hit_limit = _is_gpu_session_limit(combined)
        if last.returncode == 0 and not hit_limit:
            return
        if hit_limit and attempt < push_retries:
            print(
                f"GPU session limit hit (attempt {attempt}/{push_retries}); "
                f"retrying in {push_retry_seconds}s",
                flush=True,
            )
            time.sleep(push_retry_seconds)
            continue
        if hit_limit:
            raise SystemExit(f"Kaggle GPU session limit still hit after {push_retries} attempts")
        raise subprocess.CalledProcessError(
            last.returncode or 1, push_cmd, output=last.stdout, stderr=last.stderr
        )
    assert last is not None
    raise subprocess.CalledProcessError(
        last.returncode or 1, push_cmd, output=last.stdout, stderr=last.stderr
    )


def download_kernel_log(slug: str, output_dir: Path) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(["kaggle", "kernels", "output", slug, "-p", str(output_dir)], check=False)
    log_files = sorted(output_dir.glob("*.log"))
    if not log_files:
        log_files = sorted(output_dir.rglob("*.log"))
    return log_files[0] if log_files else None


def wait_for_kernel(slug: str, *, fetch_seconds: int) -> str:
    while True:
        completed = run_cmd(["kaggle", "kernels", "status", slug], check=False)
        status = (completed.stdout or completed.stderr or "").strip()
        print(status)
        lowered = status.lower()
        if "error" in lowered:
            return "error"
        if "cancel" in lowered:
            return "cancel"
        if "complete" in lowered:
            return "complete"
        time.sleep(fetch_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", default="famexgpuci")
    parser.add_argument(
        "--slug-suffix",
        default="",
        help="Appended to --slug and --title (e.g. 'mace' → famexgpuci-mace)",
    )
    parser.add_argument("--title", default="FamexGpuCI")
    parser.add_argument("--code-file", default=".github/scripts/kaggle_gpu_runner.py")
    parser.add_argument("--dataset", action="append", default=["rlaplaza/famexcisrc"])
    parser.add_argument("--fetch-seconds", type=int, default=15)
    parser.add_argument("--timeout-seconds", type=int, default=10800)
    parser.add_argument("--log-dir", default="/tmp/kaggle-kernel-log")
    parser.add_argument(
        "--machine-shape",
        default="NvidiaTeslaT4",
        help="Kaggle GPU type (T4 required; P100 sm_60 is incompatible with cu124 torch)",
    )
    args = parser.parse_args()

    suffix = str(args.slug_suffix or "").strip().strip("-")
    slug = f"{args.slug}-{suffix}" if suffix else args.slug
    title = f"{args.title}-{suffix}" if suffix else args.title

    code_file = Path(args.code_file)
    if not code_file.is_file():
        raise SystemExit(f"Kernel code file not found: {code_file}")

    staging = Path("/tmp/kaggle-kernel-push")
    staging.mkdir(parents=True, exist_ok=True)
    staged_code = staging / code_file.name
    staged_code.write_text(code_file.read_text(encoding="utf-8"), encoding="utf-8")

    write_kernel_metadata(
        staging_dir=staging,
        slug=slug,
        title=title,
        code_file=staged_code,
        dataset_sources=args.dataset,
        timeout_seconds=args.timeout_seconds,
        machine_shape=args.machine_shape,
    )

    final_status = wait_for_kernel(slug, fetch_seconds=args.fetch_seconds)
    log_path = download_kernel_log(slug, Path(args.log_dir))
    if log_path is not None:
        print(f"::group::Kaggle kernel log ({log_path.name})")
        print(redact_text(parse_kaggle_log(log_path.read_text(encoding="utf-8", errors="replace"))))
        print("::endgroup::")
    else:
        print("No Kaggle kernel .log file found", file=sys.stderr)

    if final_status == "complete":
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
