#!/usr/bin/env python3
"""Download and print a Kaggle kernel log (NDJSON or plain text)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from kaggle_log_common import parse_kaggle_log, redact_text


def main() -> int:
    if len(sys.argv) != 3:
        print(
            f"usage: {sys.argv[0]} <kernel-slug> <output-dir>",
            file=sys.stderr,
        )
        return 2

    kernel = sys.argv[1]
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    completed = subprocess.run(
        ["kaggle", "kernels", "output", kernel, "-p", str(output_dir)],
        check=False,
        capture_output=True,
        text=True,
    )
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

    log_files = sorted(output_dir.glob("*.log"))
    if not log_files:
        log_files = sorted(output_dir.rglob("*.log"))
    if not log_files:
        print(f"No .log files under {output_dir}", file=sys.stderr)
        return 1

    for log_path in log_files:
        print(f"::group::Kaggle kernel log ({log_path.name})")
        print(redact_text(parse_kaggle_log(log_path.read_text(encoding="utf-8", errors="replace"))))
        print("::endgroup::")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
