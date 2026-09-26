#!/usr/bin/env python3
"""Parse, mask, persist, and optionally export Kaggle API credentials."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def normalize_token(value: str) -> str:
    return value.replace("\n", "").replace("\r", "").strip()


def mask_value(value: str) -> None:
    normalized = normalize_token(value)
    if normalized:
        print(f"::add-mask::{normalized}", flush=True)


def parse_secret(raw: str) -> tuple[str, str, str]:
    """Return ``(api_token, username, legacy_key)``."""
    cleaned = raw.strip()
    if cleaned.startswith("{"):
        data = json.loads(cleaned)
        token = normalize_token(
            str(data.get("token") or data.get("api_token") or data.get("key") or "")
        )
        username = normalize_token(str(data.get("username", "")))
        if token.startswith("KGAT_"):
            return token, "", ""
        if token:
            return "", username, token
        raise SystemExit("KAGGLE_API_TOKEN JSON missing token/key field")

    token = normalize_token(cleaned)
    if token.startswith("KGAT_"):
        return token, "", ""
    raise SystemExit(
        "KAGGLE_API_TOKEN must be a KGAT_* access token or JSON "
        '{"username": "...", "key": "..."} for legacy credentials'
    )


def credentials_exist(kaggle_dir: Path) -> bool:
    return (kaggle_dir / "access_token").is_file() or (kaggle_dir / "kaggle.json").is_file()


def write_credential_files(api_token: str, username: str, key: str) -> None:
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)

    access_token = kaggle_dir / "access_token"
    legacy_file = kaggle_dir / "kaggle.json"
    access_token.unlink(missing_ok=True)
    legacy_file.unlink(missing_ok=True)

    if api_token:
        access_token.write_text(api_token, encoding="utf-8")
        access_token.chmod(0o600)
        return

    if username and key:
        legacy_file.write_text(
            json.dumps({"username": username, "key": key}),
            encoding="utf-8",
        )
        legacy_file.chmod(0o600)
        return

    raise SystemExit("No Kaggle credentials: set KAGGLE_API_TOKEN or KAGGLE_USERNAME+KAGGLE_KEY")


def emit_github_outputs(api_token: str, username: str, key: str) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with Path(output_path).open("a", encoding="utf-8") as fh:
        fh.write(f"api_token={api_token}\n")
        fh.write(f"username={username}\n")
        fh.write(f"key={key}\n")


def configure_from_env(*, emit_outputs: bool) -> None:
    api_token = normalize_token(os.environ.get("KAGGLE_API_TOKEN", ""))
    username = normalize_token(os.environ.get("KAGGLE_USERNAME", ""))
    key = normalize_token(os.environ.get("KAGGLE_KEY", ""))

    if api_token and not username and not key:
        if api_token.startswith("{"):
            api_token, username, key = parse_secret(api_token)
        elif not api_token.startswith("KGAT_"):
            raise SystemExit(
                "KAGGLE_API_TOKEN must be a KGAT_* access token or JSON "
                '{"username": "...", "key": "..."} for legacy credentials'
            )
    elif not api_token and not (username and key):
        raise SystemExit(
            "No Kaggle credentials: set KAGGLE_API_TOKEN or KAGGLE_USERNAME+KAGGLE_KEY"
        )

    mask_value(api_token)
    mask_value(username)
    mask_value(key)
    write_credential_files(api_token, username, key)
    if emit_outputs:
        emit_github_outputs(api_token, username, key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit-outputs",
        action="store_true",
        help="Write masked api_token/username/key to GITHUB_OUTPUT for kaggle-action",
    )
    parser.add_argument(
        "--require-existing",
        action="store_true",
        help="Skip configuration when ~/.kaggle credentials already exist",
    )
    args = parser.parse_args()

    kaggle_dir = Path.home() / ".kaggle"
    if args.require_existing and credentials_exist(kaggle_dir):
        return

    configure_from_env(emit_outputs=args.emit_outputs)


if __name__ == "__main__":
    main()
