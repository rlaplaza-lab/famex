"""Shared HTTP download helpers for FAMEX."""

from __future__ import annotations

import time
from collections.abc import Collection, Sequence

import requests

USER_AGENT = "famex (https://github.com/rlaplaza-lab/famex)"
DEFAULT_TIMEOUT_S = 60.0
RETRY_DELAYS_S: tuple[float, ...] = (1.0, 2.0, 4.0)
TRANSIENT_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


def create_session(*, user_agent: str = USER_AGENT) -> requests.Session:
    """Return a ``requests.Session`` with the FAMEX User-Agent."""
    session = requests.Session()
    session.headers["User-Agent"] = user_agent
    return session


def get_with_retries(
    url: str,
    *,
    session: requests.Session | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    retry_status_codes: Collection[int] = TRANSIENT_STATUS_CODES,
    retry_delays: Sequence[float] = RETRY_DELAYS_S,
) -> requests.Response:
    """GET ``url`` with retries on transient status codes and transport errors."""
    owns_session = session is None
    active = session if session is not None else create_session()
    retry_codes = frozenset(retry_status_codes)
    delays: tuple[float | None, ...] = (*retry_delays, None)

    try:
        for delay in delays:
            try:
                response = active.get(url, timeout=timeout)
            except requests.RequestException:
                if delay is None:
                    raise
                time.sleep(delay)
                continue
            if response.status_code not in retry_codes or delay is None:
                return response
            time.sleep(delay)
    finally:
        if owns_session:
            active.close()

    raise RuntimeError(f"Failed to GET {url}")
