#!/usr/bin/env python3
"""Shared helpers for redacting and parsing Kaggle kernel logs.

Both kaggle_run_kernel.py and dump_kaggle_kernel_log.py consume kernel logs
that may contain secrets (Kaggle API tokens, KGAT_* auth strings, HF tokens);
keep the redaction/parsing logic here so the two call sites stay in sync.
"""

from __future__ import annotations

import json
import re

_REDACT_PATTERNS = (
    re.compile(r"KGAT_[A-Za-z0-9_-]+"),
    re.compile(r"hf_[A-Za-z0-9]+"),
    re.compile(
        r"(?i)(api[_-]?token|access[_-]?token|kaggle[_-]?key|hf[_-]?token|"
        r"hugging[_-]?face[_-]?hub[_-]?token)\s*[:=]\s*\S+"
    ),
)


def redact_text(text: str) -> str:
    redacted = text
    for pattern in _REDACT_PATTERNS:
        redacted = pattern.sub("***", redacted)
    return redacted


def parse_kaggle_log(text: str) -> str:
    text = text.strip()
    if not text:
        return text

    entries: list[dict] = []
    try:
        payload = json.loads(text)
        if isinstance(payload, list):
            entries = [item for item in payload if isinstance(item, dict)]
        elif isinstance(payload, dict):
            entries = [payload]
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(text):
            while idx < len(text) and text[idx] in ", \n\r\t":
                idx += 1
            if idx >= len(text):
                break
            try:
                obj, end = decoder.raw_decode(text, idx)
            except json.JSONDecodeError:
                return text
            if isinstance(obj, dict):
                entries.append(obj)
            idx = end

    if not entries:
        return text

    chunks: list[str] = []
    for entry in entries:
        data = entry.get("data")
        if isinstance(data, str):
            chunks.append(data)
    return "".join(chunks) if chunks else text
